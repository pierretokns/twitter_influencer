#!/usr/bin/env python3
"""
Reranker component benchmark for source selection and citation support.

This does not replace Elo. It tests whether OSS cross-encoder rerankers can
improve the deterministic layers around retrieval, citation support, and
tournament prefiltering.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_MODELS = [
    "BAAI/bge-reranker-v2-m3",
    "mixedbread-ai/mxbai-rerank-base-v2",
    "Qwen/Qwen3-Reranker-0.6B",
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


RETRIEVAL_QUERIES = [
    ("jpm_controls", "Which source discusses J.P. Morgan AI controls and model-risk review?", "finance_jpm"),
    ("hedge_fund_signal", "Find the hedge fund source mentioning Acadian, Balyasny, Arrowstreet, and Citadel.", "finance_hedge_funds"),
    ("payments_fraud", "Which document covers Mastercard, Visa, fraud detection, and payments risk?", "payments_risk"),
    ("local_cpu_models", "What source is about llama.cpp, GGUF quants, and local CPU model candidates?", "local_models"),
    ("curation", "Which source is about curating traces into training and evaluation data?", "data_flywheel"),
    ("rag_eval", "Which source mentions Phoenix tracing and hallucination or retrieval evaluation?", "phoenix_eval"),
    ("unsupported_refusal", "Find the source that says to refuse when sources lack the requested fact.", "unsupported_refusal"),
    ("citation_check", "Which document explains checking citations against source entities and semantic support?", "citation_verification"),
]


CITATION_PAIRS = [
    {
        "id": "jpm_positive",
        "claim": "J.P. Morgan expanded AI workflows for research and model-risk review.",
        "source_id": "finance_jpm",
        "label": 1,
    },
    {
        "id": "jpm_gemma_negative",
        "claim": "J.P. Morgan is using Gemma GGUF for analyst assistants.",
        "source_id": "local_models",
        "label": 0,
    },
    {
        "id": "hedge_positive",
        "claim": "Balyasny, Arrowstreet, Acadian, and Citadel are evaluating agent workflows for research triage.",
        "source_id": "finance_hedge_funds",
        "label": 1,
    },
    {
        "id": "payments_positive",
        "claim": "Mastercard and Visa use AI for fraud detection and payments risk.",
        "source_id": "payments_risk",
        "label": 1,
    },
    {
        "id": "unsupported_negative",
        "claim": "Anthropic changed a subscription price tier on May 25, 2026.",
        "source_id": "unsupported_refusal",
        "label": 0,
    },
    {
        "id": "citation_positive",
        "claim": "Citation verification should check matching entities and semantic support.",
        "source_id": "citation_verification",
        "label": 1,
    },
    {
        "id": "phoenix_negative",
        "claim": "Phoenix is a GGUF quantization tool for running local CPU models.",
        "source_id": "phoenix_eval",
        "label": 0,
    },
]


def hf_env() -> None:
    token_path = Path("~/.cache/huggingface/token").expanduser()
    if token_path.exists() and "HF_TOKEN" not in os.environ:
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            os.environ["HF_TOKEN"] = token


def load_cross_encoder(model_name: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, trust_remote_code=True)


def predict_scores(model: Any, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return [float(score) for score in scores]


def retrieval_eval(model: Any, batch_size: int) -> dict[str, Any]:
    rows = []
    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal = []
    doc_by_id = {doc["id"]: doc for doc in DOCUMENTS}
    for query_id, query, gold in RETRIEVAL_QUERIES:
        pairs = [(query, doc["text"]) for doc in DOCUMENTS]
        scores = predict_scores(model, pairs, batch_size)
        ranked = sorted(zip(DOCUMENTS, scores), key=lambda item: item[1], reverse=True)
        ranked_ids = [doc["id"] for doc, _ in ranked]
        rank = ranked_ids.index(gold) + 1 if gold in ranked_ids else None
        if rank == 1:
            hit_at_1 += 1
        if rank is not None and rank <= 3:
            hit_at_3 += 1
        reciprocal.append(0.0 if rank is None else 1.0 / rank)
        rows.append(
            {
                "query_id": query_id,
                "gold": gold,
                "rank": rank,
                "top3": ranked_ids[:3],
                "gold_score": round(float(scores[[doc["id"] for doc in DOCUMENTS].index(gold)]), 4),
                "top_score": round(float(ranked[0][1]), 4),
                "gold_text": doc_by_id[gold]["text"],
            }
        )
    total = len(RETRIEVAL_QUERIES)
    return {
        "hit_at_1": round(hit_at_1 / total, 3),
        "hit_at_3": round(hit_at_3 / total, 3),
        "mrr": round(sum(reciprocal) / total, 3),
        "rows": rows,
    }


def citation_eval(model: Any, batch_size: int) -> dict[str, Any]:
    doc_by_id = {doc["id"]: doc["text"] for doc in DOCUMENTS}
    pairs = [(row["claim"], doc_by_id[row["source_id"]]) for row in CITATION_PAIRS]
    scores = predict_scores(model, pairs, batch_size)
    positives = [score for score, row in zip(scores, CITATION_PAIRS) if row["label"] == 1]
    negatives = [score for score, row in zip(scores, CITATION_PAIRS) if row["label"] == 0]
    pairwise_total = 0
    pairwise_correct = 0
    for pos in positives:
        for neg in negatives:
            pairwise_total += 1
            if pos > neg:
                pairwise_correct += 1
    rows = []
    for row, score in zip(CITATION_PAIRS, scores):
        rows.append({**row, "score": round(float(score), 4)})
    return {
        "positive_avg": round(sum(positives) / max(1, len(positives)), 4),
        "negative_avg": round(sum(negatives) / max(1, len(negatives)), 4),
        "pairwise_accuracy": round(pairwise_correct / max(1, pairwise_total), 3),
        "rows": rows,
    }


def evaluate_model(model_name: str, batch_size: int) -> dict[str, Any]:
    hf_env()
    started = time.time()
    try:
        model = load_cross_encoder(model_name)
        load_sec = time.time() - started
        retrieval = retrieval_eval(model, batch_size)
        citation = citation_eval(model, batch_size)
        total_sec = time.time() - started
        return {
            "model": model_name,
            "ok": True,
            "load_sec": round(load_sec, 3),
            "total_sec": round(total_sec, 3),
            "retrieval": retrieval,
            "citation": citation,
            "score": round((retrieval["mrr"] + citation["pairwise_accuracy"]) / 2, 3),
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
    parser.add_argument("--out-dir", default="output_data/model_bench/reranker_components")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "reranker_component_results.jsonl"
    summary_path = out_dir / "reranker_component_summary.json"

    rows = []
    with results_path.open("w", encoding="utf-8") as out:
        for model_name in args.models:
            print(f"MODEL {model_name}", flush=True)
            row = evaluate_model(model_name, args.batch_size)
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(
                json.dumps(
                    {
                        "model": row["model"],
                        "ok": row["ok"],
                        "score": row.get("score"),
                        "retrieval_mrr": row.get("retrieval", {}).get("mrr"),
                        "citation_pairwise": row.get("citation", {}).get("pairwise_accuracy"),
                        "total_sec": row.get("total_sec"),
                        "error": row.get("error"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    ranked = sorted(rows, key=lambda row: (row.get("ok", False), row.get("score", 0), -row.get("total_sec", 1e9)), reverse=True)
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results_path": str(results_path),
        "ranked": [
            {
                "rank": i + 1,
                "model": row["model"],
                "ok": row["ok"],
                "score": row.get("score"),
                "retrieval_mrr": row.get("retrieval", {}).get("mrr"),
                "retrieval_hit_at_1": row.get("retrieval", {}).get("hit_at_1"),
                "citation_pairwise_accuracy": row.get("citation", {}).get("pairwise_accuracy"),
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
