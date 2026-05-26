#!/usr/bin/env python3
"""Smoke-test the ChatAgent local llama.cpp backend and save evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_QUERIES = [
    {
        "id": "finance_targeted",
        "query": (
            "What AI news mentions Balyasny, J.P. Morgan, Mastercard, Visa, "
            "or investment research? Give Brandon a concise cited update."
        ),
        "expect_citations": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="output_data/ai_news.db")
    parser.add_argument(
        "--out-dir",
        default="output_data/model_bench/local_chat_backend_smoke",
    )
    parser.add_argument("--query", action="append", help="Override default query; may repeat")
    return parser.parse_args()


def event_to_dict(event: Any) -> dict[str, Any]:
    return {
        "event": getattr(event, "event", ""),
        "data": getattr(event, "data", {}) or {},
    }


def run_query(agent: Any, query_spec: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    events = [
        event_to_dict(event)
        for event in agent.stream_response_sync(
            query_spec["query"],
            session_id=f"local-smoke-{uuid.uuid4()}",
            history=[],
            options={
                "max_sources": int(os.getenv("CHAT_RETRIEVAL_MAX_SOURCES", "15")),
                "use_chunks": True,
            },
        )
    ]
    elapsed = round(time.time() - started, 3)
    answer = "".join(
        event["data"].get("token", "")
        for event in events
        if event.get("event") == "token"
    )
    done = next((event["data"] for event in events if event.get("event") == "done"), {})
    sources_event = next((event["data"] for event in events if event.get("event") == "sources"), {})
    errors = [event["data"] for event in events if event.get("event") == "error"]
    warnings = [event["data"] for event in events if event.get("event") == "warning"]
    citations = [event["data"] for event in events if event.get("event") == "citation"]
    has_timing_leak = bool(re.search(r"\[\s*Prompt:|Generation:|llama_print_timings", answer))
    citation_markers = sorted({int(match) for match in re.findall(r"\[(\d+)\]", answer)})
    citations_count = int(done.get("citations_count", len(citations)) or 0)
    passed = (
        not errors
        and not has_timing_leak
        and (
            not query_spec.get("expect_citations")
            or citations_count > 0
            or bool(citation_markers)
        )
    )
    return {
        "id": query_spec["id"],
        "query": query_spec["query"],
        "passed": passed,
        "elapsed_sec": elapsed,
        "answer": answer,
        "answer_chars": len(answer),
        "citations_count": citations_count,
        "citation_markers": citation_markers,
        "sources": sources_event.get("sources", []),
        "retrieved_source_count": sources_event.get("retrieved_source_count"),
        "reranked": sources_event.get("reranked"),
        "reranker_model": sources_event.get("reranker_model"),
        "warnings": warnings,
        "errors": errors,
        "has_timing_leak": has_timing_leak,
    }


def main() -> int:
    args = parse_args()
    if args.query:
        queries = [
            {"id": f"query_{idx + 1}", "query": query, "expect_citations": True}
            for idx, query in enumerate(args.query)
        ]
    else:
        queries = DEFAULT_QUERIES

    os.environ.setdefault("CHAT_BACKEND", "llama_cpp")

    from agents.chat_agent import ChatAgent

    agent = ChatAgent(db_path=str((ROOT / args.db).resolve()))
    results = [run_query(agent, query_spec) for query_spec in queries]
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backend": agent.backend,
        "model": agent.model_id,
        "db": args.db,
        "env": {
            key: os.getenv(key)
            for key in (
                "CHAT_BACKEND",
                "CHAT_LLAMA_MODEL",
                "CHAT_LLAMA_CLI",
                "CHAT_MAX_TOKENS",
                "CHAT_LLAMA_THREADS",
                "CHAT_ENABLE_RERANKER",
                "CHAT_CONTEXT_MAX_SOURCES",
                "CHAT_CONTEXT_SOURCE_CHARS",
            )
        },
        "passed": all(result["passed"] for result in results),
        "queries": results,
    }

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "local_chat_backend_smoke.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(out_path), "passed": summary["passed"]}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
