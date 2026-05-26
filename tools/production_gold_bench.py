#!/usr/bin/env python3
"""
Score production-seed gold cases.

This runner focuses on cheap gates first:
- actual ChatAgent retrieval vs gold source IDs
- citation-pair support heuristics

It can also run provided-source RAG generation through llama.cpp for the top
models, but that is intentionally optional because CPU runs are slow.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


TOP_MODELS = [
    "LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf",
    "unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf",
    "unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf",
]

SYSTEM_PROMPT = """You are a source-grounded AI news assistant for Brandon.
Use only the numbered sources. Cite factual claims inline with numeric markers such as [1] or [2].
If the sources do not support the requested claim, say that directly.
Keep the answer concise and practical.
Do not include hidden reasoning, analysis, scratchpad notes, or thinking text."""

CITATION_STRICT_PROMPT = """Every factual sentence must end with one or more numeric citations like [1] or [2].
Never write a factual sentence without a citation.
Do not write letters, email greetings, signoffs, subjects, or placeholders.
Do not use [N]; use only source numbers that exist in the SOURCES list.

Example style:
Mastercard describes AI as a way to improve financial fraud detection and data-driven safeguards [1].
Visa reports AI-enabled social-engineering threats in payments security [2].
Together, these signals matter for regulated financial workflows because fraud, risk, and payment-network security are operational priorities [1][2]."""


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def source_to_uid(source: Any) -> str:
    if source.type == "twitter":
        return f"tweet:{source.id}"
    if source.type == "web":
        return f"web:{source.id}"
    if source.type == "youtube":
        return f"youtube:{source.id}"
    return f"{source.type}:{source.id}"


def parent_gold_uids(conn: Any, gold_sources: list[dict[str, Any]]) -> set[str]:
    uids = set()
    paragraph_ids = [
        source["db_id"]
        for source in gold_sources
        if source.get("db_table") == "article_paragraphs" and str(source.get("db_id", "")).isdigit()
    ]
    for source in gold_sources:
        if source.get("db_table") != "article_paragraphs":
            uids.add(source["source_uid"])
    if paragraph_ids:
        placeholders = ",".join("?" for _ in paragraph_ids)
        rows = conn.execute(
            f"SELECT id, article_id FROM article_paragraphs WHERE id IN ({placeholders})",
            paragraph_ids,
        ).fetchall()
        for row in rows:
            uids.add(f"web:{row['article_id']}")
    return uids


def exact_gold_uids(gold_sources: list[dict[str, Any]]) -> set[str]:
    return {source["source_uid"] for source in gold_sources}


def retrieval_agent(db_path: str, alpha: float) -> Any:
    from agents.chat_agent import ChatAgent

    agent = object.__new__(ChatAgent)
    agent.db_path = db_path
    agent.alpha = alpha
    return agent


def score_retrieval_cases(
    cases: list[dict[str, Any]],
    db_path: str,
    max_sources: int,
    context_max_sources: int,
    alpha: float,
    recall_threshold: float,
    context_min_hits: int,
    enable_reranker: bool,
    reranker_model: str,
) -> list[dict[str, Any]]:
    import sqlite3

    agent = retrieval_agent(db_path, alpha)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = []
    for case in cases:
        if case.get("case_type") != "retrieval":
            continue
        started = time.time()
        retrieval_options = {
            "max_sources": max_sources,
            "context_max_sources": context_max_sources,
            "use_chunks": True,
            "enable_reranker": enable_reranker,
            "reranker_model": reranker_model,
        }
        sources, warning = agent._retrieve_sources(case["query"], retrieval_options)
        ranked_sources, reranker_info = agent._rerank_sources(case["query"], sources, retrieval_options)
        context_sources = agent._select_context_sources(
            case["query"],
            ranked_sources,
            {
                **retrieval_options,
                "_source_order_is_reranked": reranker_info["applied"],
            },
        )
        elapsed = time.time() - started
        retrieved_uids = [source_to_uid(source) for source in sources]
        ranked_uids = [source_to_uid(source) for source in ranked_sources]
        context_uids = [source_to_uid(source) for source in context_sources]
        exact_uids = exact_gold_uids(case.get("gold_sources", []))
        parent_uids = parent_gold_uids(conn, case.get("gold_sources", []))
        exact_hits = [uid for uid in retrieved_uids if uid in exact_uids]
        parent_hits = [uid for uid in retrieved_uids if uid in parent_uids]
        context_parent_hits = [uid for uid in context_uids if uid in parent_uids]
        exact_recall_at_k = len(set(exact_hits)) / max(1, len(exact_uids))
        parent_recall_at_k = len(set(parent_hits)) / max(1, len(parent_uids))
        context_parent_recall_at_k = len(set(context_parent_hits)) / max(1, len(parent_uids))
        first_hit_rank = next((idx + 1 for idx, uid in enumerate(retrieved_uids) if uid in parent_uids), None)
        context_first_hit_rank = next((idx + 1 for idx, uid in enumerate(context_uids) if uid in parent_uids), None)
        mrr = 1.0 / first_hit_rank if first_hit_rank else 0.0
        context_mrr = 1.0 / context_first_hit_rank if context_first_hit_rank else 0.0
        retrieved_blob = " ".join(
            " ".join([source.id, source.type, source.author or "", source.title or "", source.text or ""])
            for source in context_sources
        ).lower()
        term_hits = [term for term in case.get("required_terms", []) if term.lower() in retrieved_blob]
        term_coverage = len(term_hits) / max(1, len(case.get("required_terms", [])))
        candidate_retrieval_passed = parent_recall_at_k >= recall_threshold
        context_pack_passed = len(set(context_parent_hits)) >= context_min_hits and term_coverage >= 0.25
        passed = candidate_retrieval_passed and context_pack_passed
        rows.append(
            {
                "case_id": case["case_id"],
                "slice": case["slice"],
                "case_type": "retrieval",
                "elapsed_sec": round(elapsed, 3),
                "warning": warning,
                "reranker": reranker_info,
                "retrieved_uids": retrieved_uids,
                "ranked_uids": ranked_uids,
                "context_uids": context_uids,
                "gold_uids": sorted(parent_uids),
                "exact_gold_uids": sorted(exact_uids),
                "parent_gold_uids": sorted(parent_uids),
                "hits": parent_hits,
                "exact_hits": exact_hits,
                "parent_hits": parent_hits,
                "context_parent_hits": context_parent_hits,
                "recall_at_k": round(parent_recall_at_k, 3),
                "exact_recall_at_k": round(exact_recall_at_k, 3),
                "parent_recall_at_k": round(parent_recall_at_k, 3),
                "context_parent_recall_at_k": round(context_parent_recall_at_k, 3),
                "mrr": round(mrr, 3),
                "context_mrr": round(context_mrr, 3),
                "term_coverage": round(term_coverage, 3),
                "term_hits": term_hits,
                "recall_threshold": recall_threshold,
                "context_min_hits": context_min_hits,
                "candidate_retrieval_passed": candidate_retrieval_passed,
                "context_pack_passed": context_pack_passed,
                "passed": passed,
            }
        )
    return rows


def claim_terms(text: str) -> set[str]:
    stop = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "using",
        "can",
        "help",
        "claim",
        "source",
        "assistant",
        "assistants",
        "publish",
        "published",
        "material",
        "relevant",
        "workflow",
        "workflows",
    }
    terms = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9.]*", text.lower().replace("-", " ")):
        token = token.strip(".")
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        if token not in stop:
            terms.add(token)
    return terms


def score_citation_pairs(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        if case.get("case_type") != "citation_pair":
            continue
        claim = case["claim"]
        source = case["cited_source"]
        claim_set = claim_terms(claim)
        source_set = claim_terms(" ".join([source.get("title", ""), source.get("author", ""), source.get("text", "")]))
        overlap = sorted(claim_set & source_set)
        overlap_rate = len(overlap) / max(1, len(claim_set))
        predicted = "supported" if overlap_rate >= 0.35 else "unsupported"
        passed = predicted == case["label"]
        rows.append(
            {
                "case_id": case["case_id"],
                "slice": case["slice"],
                "case_type": "citation_pair",
                "label": case["label"],
                "predicted": predicted,
                "overlap": overlap,
                "overlap_rate": round(overlap_rate, 3),
                "passed": passed,
            }
        )
    return rows


def hf_env() -> dict[str, str]:
    env = os.environ.copy()
    token_path = Path("~/.cache/huggingface/token").expanduser()
    if token_path.exists() and "HF_TOKEN" not in env:
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            env["HF_TOKEN"] = token
    return env


def model_args(model: str) -> list[str]:
    if ":" in model and model.endswith(".gguf"):
        repo, hf_file = model.split(":", 1)
        return ["--hf-repo", repo, "--hf-file", hf_file]
    return ["-hf", model]


def clean_output(text: str) -> str:
    text = re.sub(r"\x1b\[[0-9;]*m", "", text.strip())
    text = re.sub(r".\x08", "", text)
    text = re.sub(r"[|/\\-]\s*", "", text)
    if " ... (truncated)\n\n" in text:
        text = text.split(" ... (truncated)\n\n", 1)[-1].strip()
    if "ANSWER:" in text:
        text = text.rsplit("ANSWER:", 1)[-1].strip()
    text = re.sub(r"\n?Exiting\.\.\.\s*$", "", text)
    return text.strip()


def clip_text(text: str, limit: int | None) -> str:
    clean = normalize_text(text)
    if not limit or len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def build_prompt(
    case: dict[str, Any],
    source_limit: int | None = None,
    source_char_limit: int | None = None,
    citation_strict: bool = False,
) -> str:
    source_blocks = []
    sources = case["provided_sources"][:source_limit] if source_limit else case["provided_sources"]
    for idx, source in enumerate(sources, start=1):
        source_text = clip_text(source["text"], source_char_limit)
        source_blocks.append(f"[{idx}] {source['title']}\n{source_text}\n{source.get('url', '')}")
    prompt_parts = [SYSTEM_PROMPT]
    if citation_strict and not case.get("unsupported"):
        prompt_parts.append(CITATION_STRICT_PROMPT)
    return "\n\n".join(prompt_parts) + "\n\nSOURCES:\n\n" + "\n\n".join(source_blocks) + f"\n\nQUESTION: {case['query']}\n\nANSWER:"


def supported_expected_terms(
    case: dict[str, Any],
    source_limit: int | None = None,
    source_char_limit: int | None = None,
) -> list[str]:
    sources = case["provided_sources"][:source_limit] if source_limit else case["provided_sources"]
    blob = " ".join(clip_text(source.get("text", ""), source_char_limit) for source in sources).lower()
    supported = [term for term in case.get("expected_terms", []) if term.lower() in blob]
    return supported or list(case.get("expected_terms", []))


def score_answer(
    case: dict[str, Any],
    answer: str,
    source_limit: int | None = None,
    source_char_limit: int | None = None,
) -> dict[str, Any]:
    lower = answer.lower()
    thinking_leak = any(
        marker in lower
        for marker in (
            "[start thinking]",
            "thinking process",
            "**analyze the request",
            "scratchpad",
            "hidden reasoning",
        )
    )
    citations = sorted({int(match) for match in re.findall(r"\[(\d+)\]", answer)})
    active_source_count = min(len(case["provided_sources"]), source_limit) if source_limit else len(case["provided_sources"])
    valid_citations = [num for num in citations if 1 <= num <= active_source_count]
    invalid_citations = [num for num in citations if num not in valid_citations]
    active_expected_terms = supported_expected_terms(case, source_limit, source_char_limit)
    expected_hits = [term for term in active_expected_terms if term.lower() in lower]
    expected_coverage = len(expected_hits) / max(1, len(active_expected_terms))
    if case.get("unsupported"):
        refusal_markers = [
            "not support",
            "do not contain",
            "cannot determine",
            "insufficient",
            "don't have",
            "no information",
            "not specified",
            "not available",
        ]
        behavior_passed = any(marker in lower for marker in refusal_markers)
    else:
        behavior_passed = bool(valid_citations) and expected_coverage >= 0.35
    return {
        "expected_hits": expected_hits,
        "active_expected_terms": active_expected_terms,
        "expected_coverage": round(expected_coverage, 3),
        "citations": citations,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "thinking_leak": thinking_leak,
        "passed": behavior_passed and not invalid_citations and not thinking_leak,
    }


def run_generation_cases(
    cases: list[dict[str, Any]],
    llama_cli: Path,
    models: list[str],
    ctx: int,
    threads: int,
    temp: float,
    timeout: int,
    max_tokens: int,
    case_ids: set[str] | None,
    source_limit: int | None,
    source_char_limit: int | None,
    citation_strict: bool,
    citation_retry_on_missing: bool,
) -> list[dict[str, Any]]:
    rows = []
    rag_cases = [
        case
        for case in cases
        if case.get("case_type") == "rag_generation" and (not case_ids or case["case_id"] in case_ids)
    ]
    for model in models:
        for case in rag_cases:
            prompt = build_prompt(case, source_limit, source_char_limit, citation_strict)
            prompt_chars = len(prompt)
            prompt_word_estimate = len(prompt.split())
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tmp:
                tmp.write(prompt)
                prompt_path = Path(tmp.name)
            cmd = [
                str(llama_cli),
                *model_args(model),
                "-f",
                str(prompt_path),
                "-c",
                str(ctx),
                "-n",
                str(max_tokens),
                "-t",
                str(threads),
                "--temp",
                str(temp),
                "--no-display-prompt",
                "--single-turn",
            ]
            started = time.time()
            try:
                proc = subprocess.run(
                    cmd,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    capture_output=True,
                    timeout=timeout,
                    env=hf_env(),
                    check=False,
                )
                elapsed = time.time() - started
                answer = clean_output(proc.stdout + "\n" + proc.stderr)
                score = score_answer(case, answer, source_limit, source_char_limit)
                error = None if proc.returncode == 0 else f"exit {proc.returncode}"
            except subprocess.TimeoutExpired:
                elapsed = time.time() - started
                answer = ""
                score = {"passed": False, "expected_hits": [], "expected_coverage": 0, "citations": [], "valid_citations": [], "invalid_citations": []}
                error = "timeout"
            finally:
                prompt_path.unlink(missing_ok=True)
            retry_used = False
            retry_error = None
            if (
                citation_retry_on_missing
                and not citation_strict
                and error is None
                and not case.get("unsupported")
                and not score.get("valid_citations")
                and score.get("expected_coverage", 0) >= 0.35
            ):
                retry_prompt = build_prompt(case, source_limit, source_char_limit, citation_strict=True)
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tmp:
                    tmp.write(retry_prompt)
                    retry_prompt_path = Path(tmp.name)
                retry_cmd = [
                    str(llama_cli),
                    *model_args(model),
                    "-f",
                    str(retry_prompt_path),
                    "-c",
                    str(ctx),
                    "-n",
                    str(max_tokens),
                    "-t",
                    str(threads),
                    "--temp",
                    str(temp),
                    "--no-display-prompt",
                    "--single-turn",
                ]
                retry_started = time.time()
                try:
                    retry_proc = subprocess.run(
                        retry_cmd,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        capture_output=True,
                        timeout=timeout,
                        env=hf_env(),
                        check=False,
                    )
                    elapsed += time.time() - retry_started
                    retry_answer = clean_output(retry_proc.stdout + "\n" + retry_proc.stderr)
                    retry_score = score_answer(case, retry_answer, source_limit, source_char_limit)
                    retry_error = None if retry_proc.returncode == 0 else f"exit {retry_proc.returncode}"
                    retry_used = True
                    if retry_error is None and retry_score.get("passed"):
                        answer = retry_answer
                        score = retry_score
                        error = None
                    elif retry_error is not None and error is None:
                        retry_error = retry_error
                except subprocess.TimeoutExpired:
                    elapsed += time.time() - retry_started
                    retry_error = "timeout"
                    retry_used = True
                finally:
                    retry_prompt_path.unlink(missing_ok=True)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "slice": case["slice"],
                    "case_type": "rag_generation",
                    "model": model,
                    "elapsed_sec": round(elapsed, 3),
                    "error": error,
                    "prompt_chars": prompt_chars,
                    "prompt_word_estimate": prompt_word_estimate,
                    "source_limit": source_limit,
                    "source_char_limit": source_char_limit,
                    "citation_strict": citation_strict,
                    "citation_retry_on_missing": citation_retry_on_missing,
                    "citation_retry_used": retry_used,
                    "citation_retry_error": retry_error,
                    "max_tokens": max_tokens,
                    "ctx": ctx,
                    "answer": answer,
                    "score": score,
                    "passed": score["passed"] and error is None,
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}
    for row in rows:
        kind = row["case_type"]
        bucket = by_type.setdefault(kind, {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += int(bool(row.get("passed")))
        if kind == "retrieval":
            bucket["candidate_retrieval_passed"] = bucket.get("candidate_retrieval_passed", 0) + int(
                bool(row.get("candidate_retrieval_passed"))
            )
            bucket["context_pack_passed"] = bucket.get("context_pack_passed", 0) + int(
                bool(row.get("context_pack_passed"))
            )
            bucket["parent_recall_at_k_sum"] = bucket.get("parent_recall_at_k_sum", 0.0) + row.get("parent_recall_at_k", 0.0)
            bucket["context_parent_recall_at_k_sum"] = bucket.get("context_parent_recall_at_k_sum", 0.0) + row.get("context_parent_recall_at_k", 0.0)
            bucket["context_mrr_sum"] = bucket.get("context_mrr_sum", 0.0) + row.get("context_mrr", 0.0)
        if row.get("model"):
            model_bucket = by_model.setdefault(row["model"], {"total": 0, "passed": 0, "elapsed_sec": 0.0})
            model_bucket["total"] += 1
            model_bucket["passed"] += int(bool(row.get("passed")))
            model_bucket["elapsed_sec"] += row.get("elapsed_sec", 0.0)
    for bucket in by_type.values():
        bucket["pass_rate"] = round(bucket["passed"] / max(1, bucket["total"]), 3)
        if "candidate_retrieval_passed" in bucket:
            bucket["candidate_retrieval_pass_rate"] = round(
                bucket["candidate_retrieval_passed"] / max(1, bucket["total"]), 3
            )
            bucket["context_pack_pass_rate"] = round(
                bucket["context_pack_passed"] / max(1, bucket["total"]), 3
            )
            bucket["avg_parent_recall_at_k"] = round(
                bucket.pop("parent_recall_at_k_sum") / max(1, bucket["total"]), 3
            )
            bucket["avg_context_parent_recall_at_k"] = round(
                bucket.pop("context_parent_recall_at_k_sum") / max(1, bucket["total"]), 3
            )
            bucket["avg_context_mrr"] = round(
                bucket.pop("context_mrr_sum") / max(1, bucket["total"]), 3
            )
    for bucket in by_model.values():
        bucket["pass_rate"] = round(bucket["passed"] / max(1, bucket["total"]), 3)
        bucket["avg_elapsed_sec"] = round(bucket["elapsed_sec"] / max(1, bucket["total"]), 3)
        del bucket["elapsed_sec"]
    return {"by_type": by_type, "by_model": by_model}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="output_data/gold_eval/production_seed_v1.jsonl")
    parser.add_argument("--db", default="output_data/ai_news.vm.db")
    parser.add_argument("--out-dir", default="output_data/gold_eval/production_seed_v1_bench")
    parser.add_argument("--modes", nargs="*", choices=["retrieval", "citation", "generation"], default=["retrieval", "citation"])
    parser.add_argument("--llama-cli", default="~/opt/llama.cpp/llama-cli")
    parser.add_argument("--models", nargs="*", default=TOP_MODELS)
    parser.add_argument("--max-sources", type=int, default=10)
    parser.add_argument("--context-max-sources", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--retrieval-recall-threshold", type=float, default=0.25)
    parser.add_argument("--context-min-hits", type=int, default=1)
    parser.add_argument("--enable-reranker", action="store_true")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--case-ids", nargs="*", help="Optional case IDs to run for generation mode")
    parser.add_argument("--source-limit", type=int, help="Optional max provided sources per generation case")
    parser.add_argument("--source-char-limit", type=int, help="Optional max characters per provided source text")
    parser.add_argument("--citation-strict", action="store_true", help="Add citation examples and stricter citation instructions for generation mode")
    parser.add_argument("--citation-retry-on-missing", action="store_true", help="Retry supported answers with stricter citation instructions when content is present but citations are missing")
    args = parser.parse_args()

    cases = load_cases(Path(args.gold).expanduser().resolve())
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    if "retrieval" in args.modes:
        rows.extend(
            score_retrieval_cases(
                cases,
                args.db,
                args.max_sources,
                args.context_max_sources,
                args.alpha,
                args.retrieval_recall_threshold,
                args.context_min_hits,
                args.enable_reranker,
                args.reranker_model,
            )
        )
    if "citation" in args.modes:
        rows.extend(score_citation_pairs(cases))
    if "generation" in args.modes:
        llama_cli = Path(args.llama_cli).expanduser().resolve()
        rows.extend(
            run_generation_cases(
                cases,
                llama_cli,
                args.models,
                args.ctx,
                args.threads,
                args.temp,
                args.timeout,
                args.max_tokens,
                set(args.case_ids or []),
                args.source_limit,
                args.source_char_limit,
                args.citation_strict,
                args.citation_retry_on_missing,
            )
        )

    results_path = out_dir / "production_gold_results.jsonl"
    summary_path = out_dir / "production_gold_summary.json"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "gold_path": str(Path(args.gold).expanduser().resolve()),
        "db_path": str(Path(args.db).expanduser().resolve()),
        "modes": args.modes,
        "rows": len(rows),
        **summarize(rows),
        "outputs": {"results": str(results_path), "summary": str(summary_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
