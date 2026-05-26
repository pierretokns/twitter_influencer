#!/usr/bin/env python3
"""
Small retrieval/embedding component benchmark for the Brandon news agent.

This is intentionally synthetic and deterministic. It checks whether embedding
models retrieve the right source for finance, local-model, citation, and refusal
questions before we spend time wiring them into the full webchat/RAG pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-m3",
    "google/embeddinggemma-300m",
    "nvidia/llama-nemotron-embed-1b-v2",
    "nvidia/omni-embed-nemotron-3b",
]


DOCUMENTS = [
    {
        "id": "finance_jpm",
        "text": "J.P. Morgan expanded internal AI workflows for research summarization, controls, developer productivity, and model-risk review across regulated banking teams.",
    },
    {
        "id": "finance_hedge_funds",
        "text": "Acadian Asset Management, Balyasny, Arrowstreet Capital, and Citadel are evaluating agent workflows for research triage, signal discovery, and quantitative investment review.",
    },
    {
        "id": "payments_risk",
        "text": "Mastercard and Visa use AI in fraud detection, payments risk, identity, authorization, and compliance workflows.",
    },
    {
        "id": "local_models",
        "text": "Unsloth GGUF dynamic quants, llama.cpp, Qwen, Gemma, Phi, and LiquidAI models are candidates for local CPU inference on a Hetzner VM.",
    },
    {
        "id": "data_flywheel",
        "text": "NVIDIA NeMo Curator and Data Flywheel patterns collect traces, remove sensitive data, filter low-quality records, and produce training and evaluation datasets.",
    },
    {
        "id": "phoenix_eval",
        "text": "Arize Phoenix traces agent runs and evaluates retrieval relevance, hallucination risk, QA correctness, and prompt regressions.",
    },
    {
        "id": "unsupported_refusal",
        "text": "A source-grounded assistant should refuse unsupported questions when provided sources do not contain the requested fact, exact date, price, or contract clause.",
    },
    {
        "id": "citation_verification",
        "text": "Citation verification checks that each sentence cites a source with matching entities and semantic support, then removes or replaces weak citations.",
    },
]


QUERIES = [
    {
        "id": "jpm_controls",
        "query": "Which source discusses J.P. Morgan AI controls and model-risk review?",
        "gold": ["finance_jpm"],
        "slice": "finance",
    },
    {
        "id": "hedge_fund_signal",
        "query": "Find the hedge fund and asset manager source mentioning Acadian, Balyasny, Arrowstreet, and Citadel.",
        "gold": ["finance_hedge_funds"],
        "slice": "finance",
    },
    {
        "id": "payments_fraud",
        "query": "Which document covers Mastercard, Visa, fraud detection, and payments risk?",
        "gold": ["payments_risk"],
        "slice": "finance",
    },
    {
        "id": "local_cpu_models",
        "query": "What source is about llama.cpp, GGUF quants, and local CPU model candidates?",
        "gold": ["local_models"],
        "slice": "local_models",
    },
    {
        "id": "curation",
        "query": "Which source is about curating traces into training and evaluation data?",
        "gold": ["data_flywheel"],
        "slice": "curation",
    },
    {
        "id": "rag_eval",
        "query": "Which source mentions Phoenix tracing and hallucination or retrieval evaluation?",
        "gold": ["phoenix_eval"],
        "slice": "retrieval_eval",
    },
    {
        "id": "unsupported_refusal",
        "query": "Find the source that tells the assistant to refuse when the sources lack the requested fact.",
        "gold": ["unsupported_refusal"],
        "slice": "refusal",
    },
    {
        "id": "citation_check",
        "query": "Which document explains checking citations against source entities and semantic support?",
        "gold": ["citation_verification"],
        "slice": "citation",
    },
]


def hf_env() -> None:
    token_path = Path("~/.cache/huggingface/token").expanduser()
    if token_path.exists() and "HF_TOKEN" not in os.environ:
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            os.environ["HF_TOKEN"] = token


def load_model(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    return SentenceTransformer(model_name, **kwargs)


def encode(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype=np.float32)


def evaluate_model(model_name: str, batch_size: int) -> dict[str, Any]:
    hf_env()
    started = time.time()
    try:
        model = load_model(model_name)
        load_sec = time.time() - started
        doc_texts = [doc["text"] for doc in DOCUMENTS]
        query_texts = [query["query"] for query in QUERIES]
        doc_emb = encode(model, doc_texts, batch_size=batch_size)
        query_emb = encode(model, query_texts, batch_size=batch_size)
        scores = query_emb @ doc_emb.T

        rows = []
        reciprocal_ranks = []
        hit_at_1 = 0
        hit_at_3 = 0
        for i, query in enumerate(QUERIES):
            ranked_idx = list(np.argsort(-scores[i]))
            ranked_ids = [DOCUMENTS[j]["id"] for j in ranked_idx]
            gold = set(query["gold"])
            rank = next((idx + 1 for idx, doc_id in enumerate(ranked_ids) if doc_id in gold), None)
            reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
            if rank == 1:
                hit_at_1 += 1
            if rank is not None and rank <= 3:
                hit_at_3 += 1
            rows.append(
                {
                    "query_id": query["id"],
                    "slice": query["slice"],
                    "gold": query["gold"],
                    "top3": ranked_ids[:3],
                    "rank": rank,
                    "top_score": round(float(scores[i, ranked_idx[0]]), 4),
                }
            )

        total_sec = time.time() - started
        return {
            "model": model_name,
            "ok": True,
            "load_sec": round(load_sec, 3),
            "total_sec": round(total_sec, 3),
            "hit_at_1": round(hit_at_1 / len(QUERIES), 3),
            "hit_at_3": round(hit_at_3 / len(QUERIES), 3),
            "mrr": round(float(sum(reciprocal_ranks) / len(reciprocal_ranks)), 3),
            "rows": rows,
        }
    except Exception as exc:
        return {
            "model": model_name,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "total_sec": round(time.time() - started, 3),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="output_data/model_bench/retrieval_components")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "retrieval_component_results.jsonl"
    summary_path = out_dir / "retrieval_component_summary.json"

    results = []
    with results_path.open("w", encoding="utf-8") as out:
        for model_name in args.models:
            print(f"MODEL {model_name}", flush=True)
            result = evaluate_model(model_name, args.batch_size)
            results.append(result)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            print(json.dumps({k: result.get(k) for k in ["model", "ok", "hit_at_1", "hit_at_3", "mrr", "total_sec", "error"]}, ensure_ascii=False), flush=True)

    ranked = sorted(
        results,
        key=lambda row: (row.get("ok", False), row.get("mrr", 0), row.get("hit_at_1", 0), -row.get("total_sec", 10**9)),
        reverse=True,
    )
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results_path": str(results_path),
        "ranked": [
            {
                "rank": i + 1,
                "model": row["model"],
                "ok": row["ok"],
                "hit_at_1": row.get("hit_at_1"),
                "hit_at_3": row.get("hit_at_3"),
                "mrr": row.get("mrr"),
                "total_sec": row.get("total_sec"),
                "error": row.get("error"),
            }
            for i, row in enumerate(ranked)
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
