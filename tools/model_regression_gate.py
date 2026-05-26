#!/usr/bin/env python3
"""Evaluate current model artifacts against deployment regression gates.

This gate intentionally checks production-shaped artifacts, not generic benchmark
scores. It answers: can the current local-model constellation remain the default
while we collect more labels for possible fine-tuning?
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "output_data" / "model_bench" / "regression_gate"


def load_json(rel_path: str) -> dict[str, Any] | None:
    path = ROOT / rel_path
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def model_row(summary: dict[str, Any] | None, model_contains: str) -> dict[str, Any] | None:
    if not summary:
        return None
    for row in summary.get("models", []):
        if model_contains in row.get("model", ""):
            return row
    for row in summary.get("by_model", {}).values():
        model = row.get("model", "")
        if model_contains in model:
            return row
    for model, row in summary.get("by_model", {}).items():
        if model_contains in model:
            merged = dict(row)
            merged["model"] = model
            return merged
    return None


def pass_rate(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    if row.get("pass_rate") is not None:
        return float(row["pass_rate"])
    total = row.get("total") or row.get("cases")
    passed = row.get("passed") or row.get("passed_cases")
    if total:
        return float(passed or 0) / float(total)
    if row.get("score_pct") is not None:
        return float(row["score_pct"]) / 100.0
    return None


def check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    passed: bool,
    severity: str,
    evidence: str,
    detail: dict[str, Any] | None = None,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "severity": severity,
            "evidence": evidence,
            "detail": detail or {},
        }
    )


def build_gate() -> dict[str, Any]:
    rag_summary_path = "output_data/gold_eval/production_expanded_v2_generation_top3_compact/production_gold_summary.json"
    phi_retry_path = "output_data/gold_eval/production_expanded_v2_generation_phi_citation_retry/production_gold_summary.json"
    heldout_structured_path = (
        "output_data/model_bench/constrained_json_heldout_production_traces_v1_phi_functiongemma/"
        "constrained_summary.json"
    )
    heldout_lfm_structured_path = (
        "output_data/model_bench/constrained_json_heldout_production_traces_v1_lfm/"
        "constrained_summary.json"
    )
    local_chat_smoke_path = "output_data/model_bench/local_chat_backend_smoke/local_chat_backend_smoke.json"
    llm_client_smoke_path = "output_data/model_bench/llm_client_local_smoke/llm_client_local_smoke.json"
    hosted_audit_path = "output_data/model_bench/hosted_model_migration_audit/hosted_model_migration_audit.json"
    review_split_path = "output_data/gold_eval/human_review_v1_splits/split_summary.json"
    decision_path = "output_data/model_bench/production_decision/production_model_decision.json"

    rag_summary = load_json(rag_summary_path)
    phi_retry = load_json(phi_retry_path)
    structured = load_json(heldout_structured_path)
    lfm_structured = load_json(heldout_lfm_structured_path)
    local_chat_smoke = load_json(local_chat_smoke_path)
    llm_client_smoke = load_json(llm_client_smoke_path)
    hosted_audit = load_json(hosted_audit_path)
    review_split = load_json(review_split_path)
    decision = load_json(decision_path)

    lfm_rag = model_row(rag_summary, "LFM2-2.6B")
    phi_rag = model_row(rag_summary, "Phi-4-mini")
    nano_rag = model_row(rag_summary, "NVIDIA-Nemotron")
    phi_retry_row = model_row(phi_retry, "Phi-4-mini")
    phi_structured = model_row(structured, "Phi-4-mini")
    functiongemma_structured = model_row(structured, "functiongemma")
    lfm_structured_row = model_row(lfm_structured, "LFM2-2.6B")

    checks: list[dict[str, Any]] = []
    check(
        checks,
        name="rag_primary_lfm_expanded_compact",
        passed=(pass_rate(lfm_rag) == 1.0 and (lfm_rag or {}).get("total", 0) >= 8),
        severity="hard",
        evidence=rag_summary_path,
        detail=lfm_rag,
    )
    check(
        checks,
        name="rag_backup_phi_validation_retry",
        passed=(pass_rate(phi_retry_row) == 1.0 and (phi_retry_row or {}).get("total", 0) >= 8),
        severity="hard",
        evidence=phi_retry_path,
        detail=phi_retry_row,
    )
    check(
        checks,
        name="rag_count_out_nvidia_nano",
        passed=(pass_rate(nano_rag) is not None and pass_rate(nano_rag) <= 0.25),
        severity="hard",
        evidence=rag_summary_path,
        detail=nano_rag,
    )
    check(
        checks,
        name="structured_primary_phi_heldout",
        passed=(
            (phi_structured or {}).get("passed_cases") == (phi_structured or {}).get("cases")
            and (phi_structured or {}).get("cases", 0) >= 16
        ),
        severity="hard",
        evidence=heldout_structured_path,
        detail=phi_structured,
    )
    check(
        checks,
        name="structured_count_out_functiongemma_general_use",
        passed=(pass_rate(functiongemma_structured) is not None and pass_rate(functiongemma_structured) < 0.9),
        severity="hard",
        evidence=heldout_structured_path,
        detail=functiongemma_structured,
    )
    check(
        checks,
        name="structured_count_out_lfm_json_schema_path",
        passed=(pass_rate(lfm_structured_row) is not None and pass_rate(lfm_structured_row) < 0.25),
        severity="hard",
        evidence=heldout_lfm_structured_path,
        detail=lfm_structured_row,
    )
    first_query = (local_chat_smoke or {}).get("queries", [{}])[0] if local_chat_smoke else {}
    check(
        checks,
        name="local_chat_backend_smoke",
        passed=(
            bool((local_chat_smoke or {}).get("passed"))
            and (first_query.get("citations_count") or 0) >= 1
            and not bool(first_query.get("has_timing_leak"))
        ),
        severity="hard",
        evidence=local_chat_smoke_path,
        detail={
            "passed": (local_chat_smoke or {}).get("passed"),
            "backend": (local_chat_smoke or {}).get("backend"),
            "model": (local_chat_smoke or {}).get("model"),
            "query": first_query.get("query"),
            "citations_count": first_query.get("citations_count"),
            "has_timing_leak": first_query.get("has_timing_leak"),
            "reranked": first_query.get("reranked"),
        },
    )
    check(
        checks,
        name="shared_llm_client_local_smoke",
        passed=bool((llm_client_smoke or {}).get("passed")),
        severity="hard",
        evidence=llm_client_smoke_path,
        detail=llm_client_smoke,
    )
    hosted_remaining = []
    if decision and decision.get("hosted_migration"):
        hosted_remaining = decision["hosted_migration"].get("remaining", [])
    elif hosted_audit:
        resolved = {"legacy_runtime_guarded", "hosted_deployment_guarded"}
        hosted_remaining = [
            row
            for row in hosted_audit.get("files", [])
            if row.get("hits")
            and not str(row.get("status", "")).startswith("local_")
            and row.get("status") not in resolved
        ]
    check(
        checks,
        name="hosted_runtime_paths_resolved_or_guarded",
        passed=not hosted_remaining,
        severity="hard",
        evidence=decision_path,
        detail={"remaining": hosted_remaining},
    )
    review_counts = (review_split or {}).get("counts", {})
    check(
        checks,
        name="fine_tune_labels_available",
        passed=(
            review_counts.get("reviewed_rows", 0) > 0
            and review_counts.get("candidate_train_pending_approval_rows", 0) > 0
        ),
        severity="warn",
        evidence=review_split_path,
        detail=review_counts,
    )

    hard_failures = [row for row in checks if row["severity"] == "hard" and not row["passed"]]
    warnings = [row for row in checks if row["severity"] == "warn" and not row["passed"]]
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "deployment_gate_passed": not hard_failures,
        "fine_tune_ready": not warnings,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "checks": checks,
        "decision": {
            "default_generator": "LFM2-2.6B",
            "webchat_backup": "Phi-4-mini with citation retry",
            "structured_control_plane": "Phi-4-mini constrained JSON",
            "counted_out": {
                "NVIDIA Nano": "user-facing RAG generation",
                "FunctionGemma 270M": "general structured/control-plane use",
                "LFM2-2.6B": "llama.cpp JSON-schema control-plane path",
            },
            "fine_tuning_assessment": (
                "Do not fine-tune before first local deployment. Current hard gates pass, "
                "but reviewed rows and approved train candidates are still missing."
            ),
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Model Regression Gate",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        f"- Deployment gate passed: `{report['deployment_gate_passed']}`",
        f"- Fine-tune ready: `{report['fine_tune_ready']}`",
        "",
        "## Decisions",
        "",
    ]
    for key, value in report["decision"].items():
        if key == "counted_out":
            lines.append("- Counted out:")
            for model, scope in value.items():
                lines.append(f"  - `{model}`: {scope}")
        else:
            lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Checks", ""])
    for row in report["checks"]:
        status = "pass" if row["passed"] else "fail"
        lines.append(f"- `{row['name']}`: **{status}** ({row['severity']})")
        lines.append(f"  - Evidence: `{row['evidence']}`")
    if report["hard_failures"]:
        lines.extend(["", "## Hard Failures", ""])
        for row in report["hard_failures"]:
            lines.append(f"- `{row['name']}` from `{row['evidence']}`")
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for row in report["warnings"]:
            lines.append(f"- `{row['name']}` from `{row['evidence']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR.relative_to(ROOT)))
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0 after writing artifacts.")
    args = parser.parse_args()

    report = build_gate()
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "model_regression_gate.json"
    md_path = out_dir / "model_regression_gate.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "deployment_gate_passed": report["deployment_gate_passed"],
                "fine_tune_ready": report["fine_tune_ready"],
                "hard_failures": [row["name"] for row in report["hard_failures"]],
                "warnings": [row["name"] for row in report["warnings"]],
            },
            indent=2,
        )
    )
    if args.no_fail:
        return 0
    return 0 if report["deployment_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
