#!/usr/bin/env python3
"""Build held-out production traces from real gold eval cases."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def source_block(source: dict[str, Any], limit: int = 900) -> str:
    text = " ".join((source.get("text") or "").split())
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return (
        f"Title: {source.get('title', '')}\n"
        f"Author: {source.get('author', '')}\n"
        f"URL: {source.get('url', '')}\n"
        f"Text: {text}"
    )


def verifier_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["citation", "sentence", "source", "similarity", "entity_overlap", "status", "reason"],
        "properties": {
            "citation": {"type": "integer", "minimum": 1, "maximum": 10},
            "sentence": {"type": "string"},
            "source": {"type": "string"},
            "similarity": {"type": "number", "minimum": 0, "maximum": 1},
            "entity_overlap": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
            "status": {"type": "string", "enum": ["verified", "weak", "invalid"]},
            "reason": {"type": "string"},
        },
    }


def route_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["route", "confidence", "reason", "required_sources"],
        "properties": {
            "route": {
                "type": "string",
                "enum": [
                    "answer_from_sources",
                    "refuse_insufficient_sources",
                    "run_retrieval",
                    "verify_citations",
                    "qa_review",
                ],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
            "required_sources": {"type": "array", "maxItems": 6, "items": {"type": "string"}},
        },
    }


def retrieval_gate_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["query_supported", "coverage_passed", "hit_rate", "missing_terms", "gate_reason"],
        "properties": {
            "query_supported": {"type": "boolean"},
            "coverage_passed": {"type": "boolean"},
            "hit_rate": {"type": "number", "minimum": 0, "maximum": 1},
            "missing_terms": {"type": "array", "maxItems": 10, "items": {"type": "string"}},
            "gate_reason": {"type": "string"},
        },
    }


def finance_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["finance_relevant", "priority_entities", "domain_tags", "reason"],
        "properties": {
            "finance_relevant": {"type": "boolean"},
            "priority_entities": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
            "domain_tags": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "string",
                    "enum": [
                        "hedge_fund",
                        "asset_manager",
                        "bank",
                        "payments",
                        "fintech",
                        "risk",
                        "investment_research",
                        "regulated_finance",
                    ],
                },
            },
            "reason": {"type": "string"},
        },
    }


def delivery_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["audience", "channels", "citation_count", "include_social_copy", "status", "notes"],
        "properties": {
            "audience": {"type": "string"},
            "channels": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {"type": "string", "enum": ["webchat", "daily_digest", "email", "discord", "linkedin"]},
            },
            "citation_count": {"type": "integer", "minimum": 0, "maximum": 20},
            "include_social_copy": {"type": "boolean"},
            "status": {"type": "string", "enum": ["ready", "hold_for_review", "blocked"]},
            "notes": {"type": "string"},
        },
    }


def make_trace(
    trace_id: str,
    trace_type: str,
    slice_name: str,
    source_case: dict[str, Any],
    prompt: str,
    schema: dict[str, Any],
    required_values: dict[str, list[str]],
    expected: dict[str, Any],
) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "case_id": trace_id,
        "source_case_id": source_case.get("case_id"),
        "trace_type": trace_type,
        "case_type": "structured_contract",
        "slice": slice_name,
        "prompt": prompt,
        "schema": schema,
        "required_values": required_values,
        "expected": expected,
        "provenance": {
            "source": "production_expanded_v2_gold",
            "source_case_id": source_case.get("case_id"),
            "db_source_uids": [
                source.get("source_uid")
                for source in source_case.get("provided_sources", source_case.get("gold_sources", []))
            ],
        },
        "training_status": "heldout_only",
        "human_review_required": True,
    }


def citation_status(case: dict[str, Any]) -> str:
    if case.get("label") == "supported":
        return "verified"
    support_type = case.get("support_type", "")
    if "mismatch" in support_type or support_type == "unsupported":
        return "invalid"
    return "weak"


def build_traces(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []

    for case in cases:
        if case.get("case_type") == "citation_pair":
            status = citation_status(case)
            source = case["cited_source"]
            prompt = (
                "Emit one citation verification result for this generated sentence and cited source.\n\n"
                f"Sentence: {case['claim']} [1]\n"
                f"Cited source 1:\n{source_block(source)}\n\n"
                "Use verified only when the source directly supports the full sentence, including "
                "named entities, model names, and actions. Use weak for partial support and invalid "
                "for entity/action mismatches."
            )
            traces.append(
                make_trace(
                    f"heldout_{case['case_id']}_verifier",
                    "citation_verifier",
                    case["slice"],
                    case,
                    prompt,
                    verifier_schema(),
                    {
                        "status": [status],
                        "all": list(case.get("expected_reason_terms", []))[:6],
                    },
                    {"status": status, "label": case.get("label")},
                )
            )

        if case.get("case_type") == "rag_generation" and case.get("unsupported"):
            sources = "\n\n".join(
                f"[{idx}] {source_block(source, limit=520)}"
                for idx, source in enumerate(case.get("provided_sources", [])[:4], start=1)
            )
            route_prompt = (
                f"User query: {case['query']}\n\n"
                f"Retrieved sources:\n{sources}\n\n"
                "Emit the route decision for the local news agent. If the provided sources do not "
                "support the exact requested claim, route to refuse_insufficient_sources."
            )
            traces.append(
                make_trace(
                    f"heldout_{case['case_id']}_route",
                    "unsupported_source_route",
                    case["slice"],
                    case,
                    route_prompt,
                    route_schema(),
                    {
                        "route": ["refuse_insufficient_sources"],
                        "all": ["source", "insufficient"],
                    },
                    {"route": "refuse_insufficient_sources"},
                )
            )
            gate_prompt = (
                f"Query: {case['query']}\n\n"
                f"Retrieved sources:\n{sources}\n\n"
                "Emit a retrieval gate result. Because this is a hard-negative trace, generation "
                "should not proceed unless the exact requested claim is supported."
            )
            traces.append(
                make_trace(
                    f"heldout_{case['case_id']}_gate",
                    "retrieval_gate",
                    case["slice"],
                    case,
                    gate_prompt,
                    retrieval_gate_schema(),
                    {
                        "query_supported": ["false"],
                        "coverage_passed": ["false"],
                        "gate_reason": ["missing", "insufficient", "unsupported"],
                    },
                    {"query_supported": False, "coverage_passed": False},
                )
            )

        if case.get("case_type") == "rag_generation" and case.get("slice") == "finance_domain_signal" and not case.get("unsupported"):
            source = (case.get("provided_sources") or [None])[0]
            if source:
                prompt = (
                    "Classify this source for Brandon's finance-domain news slice.\n\n"
                    f"Source:\n{source_block(source)}\n\n"
                    "Return the finance relevance result. A source is finance relevant when it "
                    "mentions banks, hedge funds, asset managers, fintech/payments, regulated "
                    "financial services, risk/compliance, or investment research."
                )
                terms = [term for term in case.get("expected_terms", []) if term]
                traces.append(
                    make_trace(
                        f"heldout_{case['case_id']}_finance_relevance",
                        "finance_relevance",
                        "finance_domain_signal",
                        case,
                        prompt,
                        finance_schema(),
                        {
                            "finance_relevant": ["true"],
                            "all": terms[:5] or ["finance"],
                        },
                        {"finance_relevant": True},
                    )
                )
                delivery_prompt = (
                    "Create the final delivery payload for Brandon's background news summary after QA passed.\n"
                    "The summary is finance-facing, uses verified citations from the provided sources, "
                    "and should be delivered to webchat and the daily digest. Do not include social "
                    "posting or influencer language."
                )
                traces.append(
                    make_trace(
                        f"heldout_{case['case_id']}_delivery",
                        "delivery_payload",
                        "delivery",
                        case,
                        delivery_prompt,
                        delivery_schema(),
                        {
                            "audience": ["brandon"],
                            "channels": ["webchat", "daily_digest"],
                            "include_social_copy": ["false"],
                            "status": ["ready"],
                        },
                        {"status": "ready", "include_social_copy": False},
                    )
                )

    seen: set[str] = set()
    unique = []
    for trace in traces:
        if trace["trace_id"] in seen:
            continue
        seen.add(trace["trace_id"])
        unique.append(trace)
    return unique


def write_openai_messages(traces: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            row = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a strict local news-agent control-plane model. Return only the requested structured object.",
                    },
                    {"role": "user", "content": trace["prompt"]},
                    {"role": "assistant", "content": json.dumps(trace["expected"], ensure_ascii=False)},
                ],
                "metadata": {
                    "trace_id": trace["trace_id"],
                    "trace_type": trace["trace_type"],
                    "slice": trace["slice"],
                    "training_status": "heldout_only",
                    "requires_human_review_before_training": True,
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="output_data/gold_eval/production_expanded_v2.jsonl")
    parser.add_argument("--out-dir", default="output_data/gold_eval/heldout_production_traces_v1")
    args = parser.parse_args()

    gold_path = (ROOT / args.gold).resolve()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = build_traces(load_jsonl(gold_path))
    traces_path = out_dir / "heldout_traces.jsonl"
    messages_path = out_dir / "heldout_openai_messages.jsonl"
    summary_path = out_dir / "heldout_summary.json"
    with traces_path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
    write_openai_messages(traces, messages_path)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gold": str(gold_path),
        "trace_count": len(traces),
        "by_trace_type": dict(Counter(trace["trace_type"] for trace in traces)),
        "by_slice": dict(Counter(trace["slice"] for trace in traces)),
        "training_status": "heldout_only; do not train on these rows unless a future split explicitly moves rows out of holdout",
        "outputs": {
            "traces_jsonl": str(traces_path),
            "openai_messages_jsonl": str(messages_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
