#!/usr/bin/env python3
"""Audit benchmark coverage by workflow slice and artifact maturity."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "output_data" / "model_bench" / "benchmark_coverage"

GOLD_PATH = "output_data/gold_eval/production_expanded_v2.jsonl"
HELDOUT_PATH = "output_data/gold_eval/heldout_production_traces_v1/heldout_traces.jsonl"
ANSWER_REVIEW_PATH = "output_data/gold_eval/human_review_v1/answer_review.jsonl"
CITATION_REVIEW_PATH = "output_data/gold_eval/human_review_v1/citation_span_review.jsonl"
REVIEW_SPLIT_PATH = "output_data/gold_eval/human_review_v1_splits/split_summary.json"

SLICE_GATES = {
    "rag_webchat_inline_citations": {
        "min_gold_cases": 6,
        "min_heldout_contracts": 6,
        "min_review_rows": 20,
        "required_case_types": ["rag_generation", "retrieval", "citation_pair"],
    },
    "finance_domain_signal": {
        "min_gold_cases": 5,
        "min_heldout_contracts": 2,
        "min_review_rows": 20,
        "required_case_types": ["rag_generation", "retrieval"],
    },
    "source_grounded_refusal": {
        "min_gold_cases": 3,
        "min_heldout_contracts": 6,
        "min_review_rows": 10,
        "required_case_types": ["rag_generation"],
    },
    "local_model_ops": {
        "min_gold_cases": 2,
        "min_heldout_contracts": 0,
        "min_review_rows": 10,
        "required_case_types": ["rag_generation", "retrieval"],
    },
    "pre_finetune_system_improvements": {
        "min_gold_cases": 2,
        "min_heldout_contracts": 0,
        "min_review_rows": 10,
        "required_case_types": ["rag_generation", "retrieval"],
    },
    "structured_control_plane": {
        "min_gold_cases": 1,
        "min_heldout_contracts": 16,
        "min_review_rows": 0,
        "required_case_types": ["retrieval"],
    },
    "data_curation_eval": {
        "min_gold_cases": 3,
        "min_heldout_contracts": 0,
        "min_review_rows": 0,
        "required_case_types": ["retrieval", "rag_generation"],
    },
    "delivery": {
        "min_gold_cases": 0,
        "min_heldout_contracts": 2,
        "min_review_rows": 0,
        "required_case_types": [],
    },
}


def load_jsonl(rel_path: str) -> list[dict[str, Any]]:
    path = ROOT / rel_path
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_json(rel_path: str) -> dict[str, Any] | None:
    path = ROOT / rel_path
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def count_by(rows: list[dict[str, Any]], key: str) -> Counter[str]:
    return Counter(str(row.get(key)) for row in rows if row.get(key) is not None)


def build_audit() -> dict[str, Any]:
    gold_rows = load_jsonl(GOLD_PATH)
    heldout_rows = load_jsonl(HELDOUT_PATH)
    answer_rows = load_jsonl(ANSWER_REVIEW_PATH)
    citation_rows = load_jsonl(CITATION_REVIEW_PATH)
    review_split = load_json(REVIEW_SPLIT_PATH) or {}

    gold_by_slice = count_by(gold_rows, "slice")
    heldout_by_slice = count_by(heldout_rows, "slice")
    answer_review_by_slice = count_by(answer_rows, "slice")
    citation_review_by_slice = count_by(citation_rows, "slice")

    gold_case_types_by_slice: dict[str, set[str]] = defaultdict(set)
    heldout_trace_types_by_slice: dict[str, set[str]] = defaultdict(set)
    for row in gold_rows:
        if row.get("slice") and row.get("case_type"):
            gold_case_types_by_slice[str(row["slice"])].add(str(row["case_type"]))
    for row in heldout_rows:
        if row.get("slice") and row.get("trace_type"):
            heldout_trace_types_by_slice[str(row["slice"])].add(str(row["trace_type"]))

    slice_results = []
    for slice_name, gate in SLICE_GATES.items():
        gold_count = gold_by_slice.get(slice_name, 0)
        heldout_count = heldout_by_slice.get(slice_name, 0)
        if slice_name == "structured_control_plane":
            heldout_count = len(heldout_rows)
        review_count = answer_review_by_slice.get(slice_name, 0) + citation_review_by_slice.get(slice_name, 0)
        case_types = sorted(gold_case_types_by_slice.get(slice_name, set()))
        missing_case_types = sorted(set(gate["required_case_types"]) - set(case_types))
        failures = []
        warnings = []
        if gold_count < gate["min_gold_cases"]:
            failures.append(f"gold_cases {gold_count} < {gate['min_gold_cases']}")
        if heldout_count < gate["min_heldout_contracts"]:
            failures.append(f"heldout_contracts {heldout_count} < {gate['min_heldout_contracts']}")
        if missing_case_types:
            warnings.append(f"missing_case_types={','.join(missing_case_types)}")
        if review_count < gate["min_review_rows"]:
            warnings.append(f"review_rows {review_count} < {gate['min_review_rows']}")
        slice_results.append(
            {
                "slice": slice_name,
                "coverage_passed": not failures,
                "maturity": "sufficient_for_current_gate" if not failures else "needs_more_cases",
                "gold_cases": gold_count,
                "heldout_contracts": heldout_count,
                "review_rows": review_count,
                "gold_case_types": case_types,
                "heldout_trace_types": sorted(heldout_trace_types_by_slice.get(slice_name, set())),
                "failures": failures,
                "warnings": warnings,
            }
        )

    review_counts = review_split.get("counts", {})
    hard_failures = [row for row in slice_results if row["failures"]]
    warnings = [row for row in slice_results if row["warnings"]]
    next_actions = []
    if any(row["slice"] == "data_curation_eval" for row in hard_failures):
        next_actions.append("Add more data_curation_eval gold cases; current coverage is below the slice gate.")
    if not any(row["slice"] == "data_curation_eval" for row in hard_failures):
        next_actions.append("Run the top local RAG models on the expanded data_curation_eval generation case.")
    next_actions.extend(
        [
            "Add held-out structured/control-plane traces as schemas change.",
            "Human-label review rows before using this data for fine-tuning.",
        ]
    )
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "coverage_gate_passed": not hard_failures,
        "human_label_ready": (
            review_counts.get("reviewed_rows", 0) > 0
            and review_counts.get("candidate_train_pending_approval_rows", 0) > 0
        ),
        "artifacts": {
            "gold": GOLD_PATH,
            "heldout": HELDOUT_PATH,
            "answer_review": ANSWER_REVIEW_PATH,
            "citation_review": CITATION_REVIEW_PATH,
            "review_split": REVIEW_SPLIT_PATH,
        },
        "totals": {
            "gold_rows": len(gold_rows),
            "heldout_contract_rows": len(heldout_rows),
            "answer_review_rows": len(answer_rows),
            "citation_review_rows": len(citation_rows),
            "reviewed_rows": review_counts.get("reviewed_rows", 0),
            "candidate_train_pending_approval_rows": review_counts.get("candidate_train_pending_approval_rows", 0),
        },
        "by_slice": slice_results,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "next_actions": next_actions,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Benchmark Coverage Audit",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        f"- Coverage gate passed: `{report['coverage_gate_passed']}`",
        f"- Human-label ready: `{report['human_label_ready']}`",
        "",
        "## Totals",
        "",
    ]
    for key, value in report["totals"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Slices", ""])
    for row in report["by_slice"]:
        lines.append(
            f"- `{row['slice']}`: `{row['maturity']}` "
            f"(gold `{row['gold_cases']}`, heldout `{row['heldout_contracts']}`, review `{row['review_rows']}`)"
        )
        if row["failures"]:
            lines.append(f"  - Failures: {', '.join(row['failures'])}")
        if row["warnings"]:
            lines.append(f"  - Warnings: {', '.join(row['warnings'])}")
    lines.extend(["", "## Next Actions", ""])
    for action in report["next_actions"]:
        lines.append(f"- {action}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR.relative_to(ROOT)))
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0 after writing artifacts.")
    args = parser.parse_args()

    report = build_audit()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark_coverage_audit.json"
    md_path = out_dir / "benchmark_coverage_audit.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "coverage_gate_passed": report["coverage_gate_passed"],
                "human_label_ready": report["human_label_ready"],
                "hard_failures": [row["slice"] for row in report["hard_failures"]],
                "warnings": [row["slice"] for row in report["warnings"]],
            },
            indent=2,
        )
    )
    if args.no_fail:
        return 0
    return 0 if report["coverage_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
