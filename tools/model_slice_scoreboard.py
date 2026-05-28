#!/usr/bin/env python3
"""Build a per-slice scoreboard for current local-model evaluations."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "output_data" / "model_bench" / "slice_scoreboard"

RESULT_ARTIFACTS = [
    "output_data/gold_eval/production_expanded_v2_generation_top3_compact_v2_chat_guardrails_rescored/production_gold_results.jsonl",
    "output_data/gold_eval/production_expanded_v2_generation_top3_compact/production_gold_results.jsonl",
    "output_data/gold_eval/production_expanded_v2_generation_phi_citation_retry/production_gold_results.jsonl",
    "output_data/gold_eval/production_expanded_v2_generation_curation_delta_top3/production_gold_results.jsonl",
    "output_data/model_bench/constrained_json_heldout_production_traces_v1_phi_functiongemma/constrained_results.jsonl",
    "output_data/model_bench/constrained_json_heldout_production_traces_v1_lfm/constrained_results.jsonl",
]


def model_short(model: str) -> str:
    if "LFM2-2.6B" in model:
        return "LFM2-2.6B"
    if "Phi-4-mini" in model:
        return "Phi-4-mini"
    if "NVIDIA-Nemotron" in model:
        return "NVIDIA Nano"
    if "functiongemma" in model.lower():
        return "FunctionGemma 270M"
    return model


def load_jsonl(rel_path: str) -> list[dict[str, Any]]:
    path = ROOT / rel_path
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def heldout_case_map() -> dict[str, dict[str, str]]:
    rows = load_jsonl("output_data/gold_eval/heldout_production_traces_v1/heldout_traces.jsonl")
    return {
        row["case_id"]: {
            "slice": row.get("slice", "unknown"),
            "case_type": row.get("trace_type", row.get("case_type", "structured_contract")),
        }
        for row in rows
    }


def row_score(row: dict[str, Any]) -> tuple[bool, float]:
    score = row.get("score") or {}
    if "passed" in row:
        passed = bool(row.get("passed"))
    else:
        passed = bool(score.get("passed"))
    if score.get("score_pct") is not None:
        pct = float(score["score_pct"])
    elif score.get("expected_coverage") is not None:
        citation_ok = not score.get("invalid_citations") and bool(score.get("valid_citations"))
        no_thinking = not bool(score.get("thinking_leak"))
        pct = 100.0 * float(score.get("expected_coverage", 0.0))
        if not citation_ok:
            pct -= 25.0
        if not no_thinking:
            pct -= 50.0
        pct = max(0.0, min(100.0, pct))
    else:
        pct = 100.0 if passed else 0.0
    return passed, round(pct, 1)


def role_for_case_type(case_type: str) -> str:
    if case_type == "rag_generation":
        return "rag_generation"
    if case_type in {"finance_relevance", "delivery_payload", "unsupported_source_route", "retrieval_gate", "citation_verifier"}:
        return f"structured_{case_type}"
    if case_type == "citation_pair":
        return "citation_pair_heuristic"
    if case_type == "retrieval":
        return "retrieval"
    return case_type or "unknown"


def comparative_score(pass_rate: float, avg_score_pct: float) -> float:
    """A quality-first score for comparing models within the same slice role."""
    return round((70.0 * pass_rate) + (30.0 * (avg_score_pct / 100.0)), 1)


def recommendation_for(row: dict[str, Any], role_rows: list[dict[str, Any]]) -> str:
    if row["rank"] == 1 and row["pass_rate"] >= 0.8:
        return "keep_primary"
    if row["pass_rate"] >= 0.8:
        return "keep_candidate"
    if row["pass_rate"] >= 0.5 and row["avg_score_pct"] >= 75.0:
        return "investigate_or_repair"
    if row["cases"] <= 1 and row["pass_rate"] == 0 and len(role_rows) < 3:
        return "needs_more_evidence"
    return "count_out_for_role"


def build_scoreboard() -> dict[str, Any]:
    case_map = heldout_case_map()
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    artifacts_used = []
    for artifact in RESULT_ARTIFACTS:
        rows = load_jsonl(artifact)
        if not rows:
            continue
        artifacts_used.append(artifact)
        for row in rows:
            model = row.get("model", "unknown")
            case_id = row.get("case_id", "")
            slice_name = row.get("slice") or case_map.get(case_id, {}).get("slice", "unknown")
            case_type = row.get("case_type") or case_map.get(case_id, {}).get("case_type", "unknown")
            role = role_for_case_type(case_type)
            passed, pct = row_score(row)
            key = (slice_name, role, model)
            bucket = buckets.setdefault(
                key,
                {
                    "slice": slice_name,
                    "role": role,
                    "model": model,
                    "label": model_short(model),
                    "cases": 0,
                    "passed": 0,
                    "score_sum": 0.0,
                    "elapsed_sum": 0.0,
                    "case_types": set(),
                    "artifacts": set(),
                    "failures": [],
                },
            )
            bucket["cases"] += 1
            bucket["passed"] += int(passed)
            bucket["score_sum"] += pct
            bucket["elapsed_sum"] += float(row.get("elapsed_sec") or 0.0)
            bucket["case_types"].add(case_type)
            bucket["artifacts"].add(artifact)
            if not passed:
                reasons = []
                score = row.get("score") or {}
                reasons.extend(score.get("reasons") or [])
                if score.get("thinking_leak"):
                    reasons.append("thinking_leak")
                if row.get("error"):
                    reasons.append(str(row["error"]))
                bucket["failures"].append(
                    {
                        "case_id": case_id,
                        "case_type": case_type,
                        "score_pct": pct,
                        "reasons": reasons[:5],
                    }
                )

    by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_slice_role: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for bucket in buckets.values():
        cases = max(1, bucket["cases"])
        pass_rate = round(bucket["passed"] / cases, 3)
        avg_score_pct = round(bucket["score_sum"] / cases, 1)
        row = {
            "slice": bucket["slice"],
            "role": bucket["role"],
            "model": bucket["model"],
            "label": bucket["label"],
            "cases": bucket["cases"],
            "passed": bucket["passed"],
            "pass_rate": pass_rate,
            "avg_score_pct": avg_score_pct,
            "comparative_score_pct": comparative_score(pass_rate, avg_score_pct),
            "avg_elapsed_sec": round(bucket["elapsed_sum"] / cases, 3),
            "case_types": sorted(bucket["case_types"]),
            "artifacts": sorted(bucket["artifacts"]),
            "failures": bucket["failures"][:6],
        }
        by_slice[row["slice"]].append(row)
        by_slice_role[row["slice"]][row["role"]].append(row)
    for rows in by_slice.values():
        rows.sort(
            key=lambda item: (item["comparative_score_pct"], item["pass_rate"], item["avg_score_pct"], -item["avg_elapsed_sec"]),
            reverse=True,
        )
    by_slice_role_out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for slice_name, roles in by_slice_role.items():
        by_slice_role_out[slice_name] = {}
        for role, rows in roles.items():
            rows.sort(
                key=lambda item: (item["comparative_score_pct"], item["pass_rate"], item["avg_score_pct"], -item["avg_elapsed_sec"]),
                reverse=True,
            )
            for index, row in enumerate(rows, start=1):
                row["rank"] = index
                row["recommendation"] = recommendation_for(row, rows)
            by_slice_role_out[slice_name][role] = rows
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifacts_used": artifacts_used,
        "by_slice": dict(sorted(by_slice.items())),
        "by_slice_role": {name: by_slice_role_out[name] for name in sorted(by_slice_role_out)},
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Model Slice Scoreboard",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
    ]
    for slice_name, roles in report.get("by_slice_role", {}).items():
        lines.extend([f"## {slice_name}", ""])
        for role, rows in roles.items():
            lines.append(f"### {role}")
            for row in rows:
                lines.append(
                    f"- #{row['rank']} `{row['label']}`: comparative `{row['comparative_score_pct']}`, "
                    f"pass `{row['passed']}/{row['cases']}` ({row['pass_rate'] * 100:.1f}%), "
                    f"avg score `{row['avg_score_pct']}`, "
                    f"status `{row['recommendation']}`, "
                    f"avg `{row['avg_elapsed_sec']}` sec"
                )
                if row["failures"]:
                    failed_ids = ", ".join(failure["case_id"] for failure in row["failures"][:3])
                    lines.append(f"  - Failures: {failed_ids}")
            lines.append("")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR.relative_to(ROOT)))
    args = parser.parse_args()

    report = build_scoreboard()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "model_slice_scoreboard.json"
    md_path = out_dir / "model_slice_scoreboard.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "slices": sorted(report["by_slice_role"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
