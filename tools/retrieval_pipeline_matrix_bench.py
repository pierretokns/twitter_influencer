#!/usr/bin/env python3
"""Compare embedding + reranker retrieval pipelines on Brandon news slices."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_EMBEDDERS = [
    "production_bge_m3_hybrid",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-m3",
    "google/embeddinggemma-300m",
    "nvidia/llama-nemotron-embed-1b-v2",
]

DEFAULT_RERANKERS = [
    "none",
    "BAAI/bge-reranker-v2-m3",
    "mixedbread-ai/mxbai-rerank-base-v2",
    "Qwen/Qwen3-Reranker-0.6B",
]


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    slice: str
    query: str
    gold_terms: tuple[str, ...]


CASES = [
    RetrievalCase(
        "finance_regulated_ai",
        "finance",
        "AI news for Brandon involving hedge funds, asset managers, banks, payments, J.P. Morgan, Citadel, Mastercard, Visa, Balyasny, Arrowstreet, or Acadian.",
        ("j.p. morgan", "jpmorgan", "citadel", "mastercard", "visa", "balyasny", "arrowstreet", "acadian", "hedge fund", "payments"),
    ),
    RetrievalCase(
        "local_cpu_models",
        "local_models",
        "Local CPU model candidates, GGUF, llama.cpp, Qwen, Gemma, Phi, LiquidAI, NVIDIA, and quantized open models.",
        ("gguf", "llama.cpp", "qwen", "gemma", "phi", "liquid", "lfm", "nvidia", "quant"),
    ),
    RetrievalCase(
        "youtube_model_drops",
        "youtube",
        "YouTube or video drops about Qwen, NVIDIA, Gemini, model releases, coding agents, evals, and AI engineering demos.",
        ("youtube", "video", "qwen", "nvidia", "gemini", "model", "release", "agent", "eval"),
    ),
    RetrievalCase(
        "ai_engineering_blogs",
        "ai_engineering",
        "AI engineering blog posts from Augment Code, Steve Yegge, Corey Quinn, software agents, model routing, codebase context, and developer workflows.",
        ("augment", "steve yegge", "last week in aws", "agent", "codebase", "developer", "model routing", "workflow"),
    ),
    RetrievalCase(
        "governance_enterprise",
        "governance",
        "Enterprise AI governance, Singapore AI Verify, agentic AI governance, NIST AI RMF, controls, audit logs, testing, and risk.",
        ("singapore", "ai verify", "agentic ai", "nist", "risk", "governance", "testing", "controls"),
    ),
    RetrievalCase(
        "conferences_events",
        "events",
        "AI conference deadlines, calls for papers, Boston AI events, NYC AI events, MIT CSAIL, NYU, workshops, and seminars.",
        ("conference", "deadline", "call for papers", "cfp", "boston", "nyc", "mit", "csail", "nyu", "seminar"),
    ),
    RetrievalCase(
        "evals_rag_curation",
        "evals_rag",
        "RAG evals, retrieval benchmarks, Phoenix, Arize, NVIDIA NeMo Curator, data flywheel, traces, and fine-tuning datasets.",
        ("rag", "eval", "benchmark", "phoenix", "arize", "nemo", "curator", "data flywheel", "trace", "fine-tuning"),
    ),
]


def hf_env() -> None:
    token_path = Path("~/.cache/huggingface/token").expanduser()
    if token_path.exists() and "HF_TOKEN" not in os.environ:
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            os.environ["HF_TOKEN"] = token


def normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def has_term(text: str, term: str) -> bool:
    return normalize(term) in normalize(text)


def doc_blob(doc: dict[str, Any]) -> str:
    return " ".join(str(doc.get(key) or "") for key in ("id", "type", "source", "title", "text", "url"))


def load_corpus(db_path: Path, limit: int, text_chars: int) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    docs: list[dict[str, Any]] = []
    queries = [
        (
            "web_article",
            """
            SELECT article_id AS id, source_name AS source, title, url,
                   COALESCE(description, '') || ' ' || COALESCE(content, '') AS text,
                   scraped_at
            FROM web_articles
            WHERE is_ai_relevant = 1
            ORDER BY scraped_at DESC
            LIMIT ?
            """,
        ),
        (
            "youtube_video",
            """
            SELECT video_id AS id, channel_name AS source, title, url,
                   COALESCE(description, '') || ' ' || COALESCE(transcript, '') AS text,
                   scraped_at
            FROM youtube_videos
            WHERE is_ai_relevant = 1
            ORDER BY scraped_at DESC
            LIMIT ?
            """,
        ),
        (
            "tweet",
            """
            SELECT tweet_id AS id, username AS source, text AS title, url, text, scraped_at
            FROM tweets
            WHERE is_ai_relevant = 1
            ORDER BY scraped_at DESC
            LIMIT ?
            """,
        ),
    ]
    per_type = max(50, limit // len(queries))
    for source_type, sql in queries:
        for row in conn.execute(sql, (per_type,)):
            text = " ".join([str(row["title"] or ""), str(row["text"] or "")]).strip()
            if len(text) < 20:
                continue
            docs.append(
                {
                    "id": f"{source_type}:{row['id']}",
                    "type": source_type,
                    "source": row["source"],
                    "title": row["title"],
                    "text": text[:text_chars],
                    "url": row["url"],
                }
            )
    conn.close()
    return docs[:limit]


def encode_sentence_transformer(
    model_name: str,
    texts: list[str],
    batch_size: int,
    model_max_length: int | None,
    truncate_dim: int | None,
    task: str | None = None,
    prompt_name: str | None = None,
) -> np.ndarray:
    hf_env()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, trust_remote_code=True)
    if model_max_length:
        model.max_seq_length = model_max_length
    encode_kwargs: dict[str, Any] = {}
    if truncate_dim:
        encode_kwargs["truncate_dim"] = truncate_dim
    if task:
        encode_kwargs["task"] = task
    if prompt_name:
        encode_kwargs["prompt_name"] = prompt_name
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        **encode_kwargs,
    )
    return np.asarray(emb, dtype=np.float32)


def load_reranker(model_name: str) -> Any:
    hf_env()
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, trust_remote_code=True)


def rerank_docs(model: Any, query: str, docs: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    pairs = [(query, doc_blob(doc)[:2400]) for doc in docs]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    ranked = sorted(zip(docs, scores), key=lambda item: float(item[1]), reverse=True)
    return [{**doc, "rerank_score": round(float(score), 5)} for doc, score in ranked]


def score_ranked(case: RetrievalCase, ranked: list[dict[str, Any]], context_k: int) -> dict[str, Any]:
    top_context = ranked[:context_k]
    top_blob = "\n".join(doc_blob(doc) for doc in top_context)
    top10_blob = "\n".join(doc_blob(doc) for doc in ranked[:10])
    context_hits = [term for term in case.gold_terms if has_term(top_blob, term)]
    top10_hits = [term for term in case.gold_terms if has_term(top10_blob, term)]
    return {
        "case_id": case.case_id,
        "slice": case.slice,
        "gold_terms": list(case.gold_terms),
        "context_hits": context_hits,
        "top10_hits": top10_hits,
        "context_recall": round(len(context_hits) / len(case.gold_terms), 3),
        "top10_recall": round(len(top10_hits) / len(case.gold_terms), 3),
        "top_context": [
            {key: doc.get(key) for key in ("id", "type", "source", "title", "url", "score", "rerank_score")}
            for doc in top_context
        ],
    }


def production_retrieve(db_path: Path, case: RetrievalCase, max_sources: int, context_k: int, reranker: str) -> list[dict[str, Any]]:
    from agents.chat_agent import ChatAgent

    agent = ChatAgent(db_path=str(db_path))
    options = {
        "max_sources": max_sources,
        "context_max_sources": context_k,
        "use_chunks": True,
        "enable_reranker": reranker != "none",
        "reranker_model": reranker,
    }
    sources, _warning = agent._retrieve_sources(case.query, options)
    ranked_sources, reranker_info = agent._rerank_sources(case.query, sources, options)
    context_sources = agent._select_context_sources(
        case.query,
        ranked_sources,
        {**options, "_source_order_is_reranked": reranker_info.get("applied", False)},
    )
    ranked = context_sources + [src for src in ranked_sources if src not in context_sources]
    return [
        {
            "id": source.id,
            "type": source.type,
            "source": source.author,
            "title": source.title,
            "text": source.text,
            "url": source.url,
        }
        for source in ranked
    ]


def evaluate_pipeline(
    db_path: Path,
    docs: list[dict[str, Any]],
    embedder_name: str,
    reranker_name: str,
    batch_size: int,
    max_sources: int,
    context_k: int,
    model_max_length: int | None,
    truncate_dim: int | None,
) -> dict[str, Any]:
    started = time.time()
    rows: list[dict[str, Any]] = []
    try:
        if embedder_name == "production_bge_m3_hybrid":
            for case in CASES:
                ranked = production_retrieve(db_path, case, max_sources, context_k, reranker_name)
                rows.append(score_ranked(case, ranked, context_k))
        else:
            doc_texts = [doc_blob(doc) for doc in docs]
            query_texts = [case.query for case in CASES]
            is_jina = embedder_name.startswith("jinaai/jina-embeddings-v4")
            task = "retrieval" if is_jina else None
            doc_emb = encode_sentence_transformer(
                embedder_name,
                doc_texts,
                batch_size,
                model_max_length,
                truncate_dim,
                task=task,
                prompt_name="passage" if is_jina else None,
            )
            query_emb = encode_sentence_transformer(
                embedder_name,
                query_texts,
                batch_size,
                model_max_length,
                truncate_dim,
                task=task,
                prompt_name="query" if is_jina else None,
            )
            scores = query_emb @ doc_emb.T
            reranker = None if reranker_name == "none" else load_reranker(reranker_name)
            for i, case in enumerate(CASES):
                top_idx = list(np.argsort(-scores[i]))[:max_sources]
                ranked = [{**docs[j], "score": round(float(scores[i, j]), 5)} for j in top_idx]
                if reranker is not None:
                    ranked = rerank_docs(reranker, case.query, ranked, batch_size)
                rows.append(score_ranked(case, ranked, context_k))
        avg_context = sum(row["context_recall"] for row in rows) / len(rows)
        avg_top10 = sum(row["top10_recall"] for row in rows) / len(rows)
        passed = avg_context >= 0.28 and avg_top10 >= 0.45
        return {
            "embedder": embedder_name,
            "reranker": reranker_name,
            "ok": True,
            "passed": passed,
            "avg_context_recall": round(avg_context, 3),
            "avg_top10_recall": round(avg_top10, 3),
            "elapsed_sec": round(time.time() - started, 3),
            "cases": rows,
        }
    except Exception as exc:
        return {
            "embedder": embedder_name,
            "reranker": reranker_name,
            "ok": False,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_sec": round(time.time() - started, 3),
        }


def load_existing_results(results_path: Path) -> list[dict[str, Any]]:
    if not results_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"[warn] Skipping malformed result line in {results_path}", flush=True)
    return rows


def write_summary(
    summary_path: Path,
    db_path: Path,
    docs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    results_path: Path,
) -> None:
    ranked = sorted(
        rows,
        key=lambda row: (
            row.get("ok", False),
            row.get("avg_context_recall", 0.0),
            row.get("avg_top10_recall", 0.0),
            -row.get("elapsed_sec", 1e9),
        ),
        reverse=True,
    )
    slice_names = sorted({case.slice for case in CASES})
    slice_summary: dict[str, list[dict[str, Any]]] = {}
    for slice_name in slice_names:
        slice_rows = []
        for row in rows:
            cases = [case for case in row.get("cases", []) if case.get("slice") == slice_name]
            if not cases:
                continue
            avg_context = sum(case.get("context_recall", 0.0) for case in cases) / len(cases)
            avg_top10 = sum(case.get("top10_recall", 0.0) for case in cases) / len(cases)
            slice_rows.append(
                {
                    "embedder": row["embedder"],
                    "reranker": row["reranker"],
                    "ok": row["ok"],
                    "context_recall": round(avg_context, 3),
                    "top10_recall": round(avg_top10, 3),
                    "elapsed_sec": row.get("elapsed_sec"),
                }
            )
        slice_rows.sort(
            key=lambda row: (
                row.get("ok", False),
                row.get("context_recall", 0.0),
                row.get("top10_recall", 0.0),
                -row.get("elapsed_sec", 1e9),
            ),
            reverse=True,
        )
        slice_summary[slice_name] = [
            {"rank": index + 1, **row} for index, row in enumerate(slice_rows)
        ]
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "db": str(db_path),
        "doc_count": len(docs),
        "case_count": len(CASES),
        "results_path": str(results_path),
        "complete_rows": len(rows),
        "slice_count": len(slice_names),
        "slice_summary": slice_summary,
        "ranked": [
            {
                "rank": i + 1,
                "embedder": row["embedder"],
                "reranker": row["reranker"],
                "ok": row["ok"],
                "passed": row.get("passed"),
                "avg_context_recall": row.get("avg_context_recall"),
                "avg_top10_recall": row.get("avg_top10_recall"),
                "elapsed_sec": row.get("elapsed_sec"),
                "error": row.get("error"),
            }
            for i, row in enumerate(ranked)
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="output_data/ai_news.db")
    parser.add_argument("--out-dir", default="output_data/model_bench/retrieval_pipeline_matrix")
    parser.add_argument("--embedders", nargs="*", default=DEFAULT_EMBEDDERS)
    parser.add_argument("--rerankers", nargs="*", default=DEFAULT_RERANKERS)
    parser.add_argument("--limit", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-sources", type=int, default=30)
    parser.add_argument("--context-k", type=int, default=3)
    parser.add_argument("--text-chars", type=int, default=2400, help="Characters of title/content kept per document before embedding")
    parser.add_argument("--model-max-length", type=int, default=None, help="Override SentenceTransformer max_seq_length")
    parser.add_argument("--truncate-dim", type=int, default=None, help="Use Matryoshka truncate_dim for models that support it")
    parser.add_argument("--resume", action="store_true", help="Append to existing results and skip completed embedder/reranker pairs")
    args = parser.parse_args()

    db_path = (ROOT / args.db).resolve()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = load_corpus(db_path, args.limit, args.text_chars)

    results_path = out_dir / "retrieval_pipeline_matrix_results.jsonl"
    summary_path = out_dir / "retrieval_pipeline_matrix_summary.json"
    rows = load_existing_results(results_path) if args.resume else []
    completed = {(row.get("embedder"), row.get("reranker")) for row in rows}
    if rows:
        write_summary(summary_path, db_path, docs, rows, results_path)
        print(f"[resume] Loaded {len(rows)} existing rows from {results_path}", flush=True)

    mode = "a" if args.resume else "w"
    with results_path.open(mode, encoding="utf-8") as out:
        for embedder in args.embedders:
            for reranker in args.rerankers:
                if embedder == "production_bge_m3_hybrid" and reranker not in {"none", "BAAI/bge-reranker-v2-m3", "mixedbread-ai/mxbai-rerank-base-v2", "Qwen/Qwen3-Reranker-0.6B"}:
                    continue
                if args.resume and (embedder, reranker) in completed:
                    print(f"SKIP existing embedder={embedder} reranker={reranker}", flush=True)
                    continue
                print(f"PIPELINE embedder={embedder} reranker={reranker}", flush=True)
                row = evaluate_pipeline(
                    db_path,
                    docs,
                    embedder,
                    reranker,
                    args.batch_size,
                    args.max_sources,
                    args.context_k,
                    args.model_max_length,
                    args.truncate_dim,
                )
                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                write_summary(summary_path, db_path, docs, rows, results_path)
                print(json.dumps({k: row.get(k) for k in ("embedder", "reranker", "ok", "passed", "avg_context_recall", "avg_top10_recall", "elapsed_sec", "error")}, ensure_ascii=False), flush=True)

    write_summary(summary_path, db_path, docs, rows, results_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
