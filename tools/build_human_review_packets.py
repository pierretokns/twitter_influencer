#!/usr/bin/env python3
"""Build human-review packets from production gold cases and model outputs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def model_label(model: str) -> str:
    if "LFM2-2.6B" in model:
        return "LFM2-2.6B"
    if "Phi-4-mini" in model:
        return "Phi-4-mini"
    if "Nemotron" in model:
        return "NVIDIA-Nemotron-Nano"
    if "FunctionGemma" in model or "functiongemma" in model:
        return "FunctionGemma-270M"
    return model


def clean_answer(answer: str) -> str:
    text = re.sub(r"\x1b\[[0-9;]*m", "", answer or "")
    text = re.sub(r"\[\s*Prompt:.*$", "", text, flags=re.S)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(answer: str) -> list[str]:
    text = clean_answer(answer)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def citation_numbers(sentence: str) -> list[int]:
    return sorted({int(match) for match in re.findall(r"\[(\d+)\]", sentence)})


def active_sources(case: dict[str, Any], source_limit: int | None, source_char_limit: int | None) -> list[dict[str, Any]]:
    sources = list(case.get("provided_sources", []))
    if source_limit:
        sources = sources[:source_limit]
    clipped: list[dict[str, Any]] = []
    for idx, source in enumerate(sources, start=1):
        copied = dict(source)
        copied["review_source_number"] = idx
        if source_char_limit and len(copied.get("text", "")) > source_char_limit:
            copied["text"] = copied["text"][: max(0, source_char_limit - 3)].rstrip() + "..."
        clipped.append(copied)
    return clipped


def result_key(row: dict[str, Any]) -> tuple[str, str]:
    return row["case_id"], model_label(row["model"])


def best_rows(result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the strongest row per case/model when retry and baseline outputs both exist."""
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in result_rows:
        key = result_key(row)
        current = selected.get(key)
        row_score = row.get("score", {})
        current_score = current.get("score", {}) if current else {}
        row_key = (
            int(bool(row.get("passed"))),
            row_score.get("expected_coverage", 0),
            len(row_score.get("valid_citations", [])),
            -row.get("elapsed_sec", 0),
        )
        current_key = (
            int(bool(current.get("passed"))),
            current_score.get("expected_coverage", 0),
            len(current_score.get("valid_citations", [])),
            -current.get("elapsed_sec", 0),
        ) if current else None
        if current is None or row_key > current_key:
            selected[key] = row
    return sorted(selected.values(), key=lambda row: (row["case_id"], model_label(row["model"])))


def build_packets(
    gold_cases: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    source_limit: int | None,
    source_char_limit: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cases = {case["case_id"]: case for case in gold_cases}
    answer_rows: list[dict[str, Any]] = []
    citation_rows: list[dict[str, Any]] = []
    counts = Counter()
    by_model = Counter()
    by_slice = Counter()

    for row in best_rows(result_rows):
        case = cases.get(row["case_id"])
        if not case or case.get("case_type") != "rag_generation":
            continue
        label = model_label(row["model"])
        sources = active_sources(case, row.get("source_limit") or source_limit, row.get("source_char_limit") or source_char_limit)
        answer = clean_answer(row.get("answer", ""))
        answer_id = f"{case['case_id']}::{label}"
        review_status = "needs_review"
        if case.get("unsupported"):
            review_focus = ["unsupported_refusal", "no_fabricated_specifics"]
        else:
            review_focus = ["factual_consistency", "citation_support", "finance_relevance", "usefulness"]

        answer_rows.append(
            {
                "review_id": answer_id,
                "case_id": case["case_id"],
                "slice": case["slice"],
                "model": row["model"],
                "model_label": label,
                "query": case["query"],
                "unsupported_expected": bool(case.get("unsupported")),
                "expected_terms": case.get("expected_terms", []),
                "score": row.get("score", {}),
                "passed": row.get("passed"),
                "answer": answer,
                "sources": sources,
                "review_focus": review_focus,
                "human_labels": {
                    "acceptable": None,
                    "factual_consistency": None,
                    "usefulness": None,
                    "missing_important_context": None,
                    "notes": "",
                },
                "review_status": review_status,
            }
        )
        counts["answers"] += 1
        by_model[label] += 1
        by_slice[case["slice"]] += 1

        source_by_num = {source["review_source_number"]: source for source in sources}
        for sent_idx, sentence in enumerate(split_sentences(answer), start=1):
            nums = citation_numbers(sentence)
            if not nums:
                if not case.get("unsupported"):
                    counts["uncited_supported_sentences"] += 1
                    citation_rows.append(
                        {
                            "review_id": f"{answer_id}::sent-{sent_idx}",
                            "answer_review_id": answer_id,
                            "case_id": case["case_id"],
                            "slice": case["slice"],
                            "model": row["model"],
                            "model_label": label,
                            "sentence_index": sent_idx,
                            "sentence": sentence,
                            "citation_numbers": [],
                            "cited_sources": [],
                            "query": case["query"],
                            "unsupported_expected": False,
                            "expected_terms": case.get("expected_terms", []),
                            "human_labels": {
                                "support_status": None,
                                "supporting_source_numbers": [],
                                "supporting_spans": [],
                                "issue_tags": ["missing_citation_candidate"],
                                "notes": "",
                            },
                            "review_status": review_status,
                        }
                    )
                continue
            cited_sources = [source_by_num[num] for num in nums if num in source_by_num]
            citation_rows.append(
                {
                    "review_id": f"{answer_id}::sent-{sent_idx}",
                    "answer_review_id": answer_id,
                    "case_id": case["case_id"],
                    "slice": case["slice"],
                    "model": row["model"],
                    "model_label": label,
                    "sentence_index": sent_idx,
                    "sentence": sentence,
                    "citation_numbers": nums,
                    "cited_sources": cited_sources,
                    "query": case["query"],
                    "unsupported_expected": bool(case.get("unsupported")),
                    "expected_terms": case.get("expected_terms", []),
                    "human_labels": {
                        "support_status": None,
                        "supporting_source_numbers": [],
                        "supporting_spans": [],
                        "issue_tags": [],
                        "notes": "",
                    },
                    "review_status": review_status,
                }
            )
            counts["citation_sentences"] += 1
            if len(cited_sources) != len(nums):
                counts["out_of_range_citation_sentences"] += 1

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            **dict(counts),
            "citation_review_rows": len(citation_rows),
            "answer_review_rows": len(answer_rows),
        },
        "by_model": dict(by_model),
        "by_slice": dict(by_slice),
        "review_instructions": {
            "answer_rows": "Label acceptable/useful/factual consistency for the full answer.",
            "citation_rows": "For each cited sentence, label support_status as supported, partial, unsupported, or out_of_range, and copy minimal supporting spans.",
            "holdout_rule": "Do not train on reviewed rows until a separate train/holdout split is created.",
        },
    }
    return answer_rows, citation_rows, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(path: Path, answer_rows: list[dict[str, Any]], citation_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    lines = [
        "# Human Review Packet",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        f"Answers: `{len(answer_rows)}`",
        f"Cited sentences: `{len(citation_rows)}`",
        "",
        "## Review Instructions",
        "",
        "- Mark answer rows for acceptability, factual consistency, usefulness, and missing context.",
        "- Mark citation rows as `supported`, `partial`, `unsupported`, or `out_of_range`.",
        "- Copy the smallest source span that supports the sentence when support exists.",
        "- Keep reviewed rows as holdout eval data until an explicit train/holdout split is made.",
        "",
        "## Answer Rows",
        "",
    ]
    for row in answer_rows:
        lines.extend(
            [
                f"### {row['review_id']}",
                "",
                f"- Slice: `{row['slice']}`",
                f"- Passed current scorer: `{row['passed']}`",
                f"- Query: {row['query']}",
                "",
                row["answer"],
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="output_data/gold_eval/production_expanded_v2.jsonl")
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--out-dir", default="output_data/gold_eval/human_review_v1")
    parser.add_argument("--source-limit", type=int, default=3)
    parser.add_argument("--source-char-limit", type=int, default=700)
    args = parser.parse_args()

    gold_cases = load_jsonl(Path(args.gold).expanduser().resolve())
    result_rows: list[dict[str, Any]] = []
    for result_path in args.results:
        result_rows.extend(load_jsonl(Path(result_path).expanduser().resolve()))

    answer_rows, citation_rows, summary = build_packets(
        gold_cases,
        result_rows,
        args.source_limit,
        args.source_char_limit,
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    answer_path = out_dir / "answer_review.jsonl"
    citation_path = out_dir / "citation_span_review.jsonl"
    summary_path = out_dir / "review_summary.json"
    markdown_path = out_dir / "review_packet.md"
    write_jsonl(answer_path, answer_rows)
    write_jsonl(citation_path, citation_rows)
    summary["outputs"] = {
        "answer_review": str(answer_path),
        "citation_span_review": str(citation_path),
        "summary": str(summary_path),
        "markdown": str(markdown_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(markdown_path, answer_rows, citation_rows, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
