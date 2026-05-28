#!/usr/bin/env python3
"""Small late-interaction ColBERT/ColBERT-like retrieval smoke.

This intentionally stays separate from retrieval_pipeline_matrix_bench.py because
ColBERT-style models return token/multi-vector embeddings and must be scored with
MaxSim, not single-vector cosine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.retrieval_pipeline_matrix_bench import CASES, doc_blob, load_corpus, score_ranked


DEFAULT_COLBERT_MODELS = [
    "answerdotai/answerai-colbert-small-v1",
    "mixedbread-ai/mxbai-edge-colbert-v0-32m",
    "mixedbread-ai/mxbai-edge-colbert-v0-17m",
]


def hf_env() -> None:
    token_path = Path("~/.cache/huggingface/token").expanduser()
    if token_path.exists() and "HF_TOKEN" not in os.environ:
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            os.environ["HF_TOKEN"] = token


def normalize_token_matrix(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected token embedding matrix with 2 dims, got shape={arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def encode_token_embeddings(model: Any, texts: list[str], batch_size: int) -> list[np.ndarray]:
    encoded = model.encode(
        texts,
        batch_size=batch_size,
        output_value="token_embeddings",
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return [normalize_token_matrix(item) for item in encoded]


def maxsim_score(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    sims = query_tokens @ doc_tokens.T
    return float(np.max(sims, axis=1).sum())


def evaluate_colbert_model(
    db_path: Path,
    model_name: str,
    limit: int,
    batch_size: int,
    max_sources: int,
    context_k: int,
    text_chars: int,
) -> dict[str, Any]:
    started = time.time()
    try:
        hf_env()
        from sentence_transformers import SentenceTransformer

        docs = load_corpus(db_path, limit, text_chars)
        doc_texts = [doc_blob(doc) for doc in docs]
        query_texts = [case.query for case in CASES]
        load_started = time.time()
        model = SentenceTransformer(model_name, trust_remote_code=True)
        load_sec = time.time() - load_started

        doc_started = time.time()
        doc_tokens = encode_token_embeddings(model, doc_texts, batch_size)
        doc_encode_sec = time.time() - doc_started
        query_started = time.time()
        query_tokens = encode_token_embeddings(model, query_texts, batch_size)
        query_encode_sec = time.time() - query_started

        score_started = time.time()
        rows = []
        for case, q_tokens in zip(CASES, query_tokens):
            scores = [maxsim_score(q_tokens, d_tokens) for d_tokens in doc_tokens]
            top_idx = list(np.argsort(-np.asarray(scores)))[:max_sources]
            ranked = [{**docs[index], "score": round(float(scores[index]), 5)} for index in top_idx]
            rows.append(score_ranked(case, ranked, context_k))
        score_sec = time.time() - score_started

        avg_context = sum(row["context_recall"] for row in rows) / len(rows)
        avg_top10 = sum(row["top10_recall"] for row in rows) / len(rows)
        finance_cases = [row for row in rows if row["slice"] in {"finance", "finance_ai"}]
        finance_context = sum(row["context_recall"] for row in finance_cases) / len(finance_cases)
        finance_top10 = sum(row["top10_recall"] for row in finance_cases) / len(finance_cases)
        return {
            "model": model_name,
            "ok": True,
            "passed": avg_context >= 0.28 and avg_top10 >= 0.45,
            "passed_finance_ai_gate": finance_context >= 0.30 and finance_top10 >= 0.50,
            "avg_context_recall": round(avg_context, 3),
            "avg_top10_recall": round(avg_top10, 3),
            "finance_ai_context_recall": round(finance_context, 3),
            "finance_ai_top10_recall": round(finance_top10, 3),
            "elapsed_sec": round(time.time() - started, 3),
            "timings": {
                "offline_model_load_sec": round(load_sec, 3),
                "offline_doc_encode_sec": round(doc_encode_sec, 3),
                "online_query_encode_sec": round(query_encode_sec, 3),
                "online_maxsim_score_sec": round(score_sec, 3),
                "online_total_sec": round(query_encode_sec + score_sec, 3),
                "online_avg_per_query_sec": round((query_encode_sec + score_sec) / len(CASES), 3),
            },
            "cases": rows,
        }
    except (BrokenPipeError, KeyboardInterrupt):
        raise
    except Exception as exc:
        return {
            "model": model_name,
            "ok": False,
            "passed": False,
            "passed_finance_ai_gate": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_sec": round(time.time() - started, 3),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="output_data/ai_news.db")
    parser.add_argument("--out-dir", default="output_data/model_bench/colbert_late_interaction")
    parser.add_argument("--models", nargs="*", default=DEFAULT_COLBERT_MODELS)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-sources", type=int, default=10)
    parser.add_argument("--context-k", type=int, default=3)
    parser.add_argument("--text-chars", type=int, default=800)
    args = parser.parse_args()

    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = (ROOT / args.db).resolve()
    results_path = out_dir / "colbert_late_interaction_results.jsonl"
    summary_path = out_dir / "colbert_late_interaction_summary.json"
    rows = []
    with results_path.open("w", encoding="utf-8") as out:
        for model_name in args.models:
            print(f"COLBERT model={model_name}", flush=True)
            row = evaluate_colbert_model(
                db_path,
                model_name,
                args.limit,
                args.batch_size,
                args.max_sources,
                args.context_k,
                args.text_chars,
            )
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(json.dumps({key: row.get(key) for key in ("model", "ok", "passed", "passed_finance_ai_gate", "avg_context_recall", "avg_top10_recall", "finance_ai_context_recall", "finance_ai_top10_recall", "elapsed_sec", "timings", "error")}, ensure_ascii=False), flush=True)

    ranked = sorted(
        rows,
        key=lambda row: (
            row.get("ok", False),
            row.get("passed_finance_ai_gate", False),
            row.get("finance_ai_context_recall", 0.0),
            row.get("avg_context_recall", 0.0),
            row.get("avg_top10_recall", 0.0),
            -row.get("elapsed_sec", 1e9),
        ),
        reverse=True,
    )
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "db": str(db_path),
        "doc_count": args.limit,
        "case_count": len(CASES),
        "results_path": str(results_path),
        "ranked": [{"rank": index + 1, **row} for index, row in enumerate(ranked)],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
