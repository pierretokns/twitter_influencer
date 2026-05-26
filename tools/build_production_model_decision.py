#!/usr/bin/env python3
"""Build a concise base-vs-finetune decision report from current eval outputs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output_data" / "model_bench" / "production_decision"


def load_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def load_json_if_exists(path: str) -> dict[str, Any] | None:
    full_path = ROOT / path
    if not full_path.exists():
        return None
    return json.loads(full_path.read_text(encoding="utf-8"))


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with (ROOT / path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_jsonl_if_exists(path: str) -> list[dict[str, Any]]:
    if not (ROOT / path).exists():
        return []
    return load_jsonl(path)


def model_short(model: str) -> str:
    if "LFM2-2.6B" in model:
        return "LFM2-2.6B"
    if "gemma-4-E2B" in model:
        return "Gemma 4 E2B"
    if "Phi-4-mini" in model:
        return "Phi-4-mini"
    if "Nemotron" in model:
        return "NVIDIA Nano"
    if "FunctionGemma" in model or "functiongemma" in model.lower():
        return "FunctionGemma 270M"
    return model


def strict_generation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, dict[str, Any]] = {}
    for row in rows:
        model = row["model"]
        score = row.get("score", {})
        answer_lower = row.get("answer", "").lower()
        thinking_leak = bool(score.get("thinking_leak")) or any(
            marker in answer_lower
            for marker in (
                "[start thinking]",
                "thinking process",
                "**analyze the request",
                "scratchpad",
                "hidden reasoning",
            )
        )
        strict_pass = bool(row.get("passed")) and not thinking_leak
        item = by_model.setdefault(
            model,
            {
                "model": model,
                "label": model_short(model),
                "cases": 0,
                "raw_pass": 0,
                "strict_pass": 0,
                "avg_elapsed_sec": 0.0,
                "failures": [],
            },
        )
        item["cases"] += 1
        item["raw_pass"] += int(bool(row.get("passed")))
        item["strict_pass"] += int(strict_pass)
        item["avg_elapsed_sec"] += row.get("elapsed_sec", 0.0)
        if not strict_pass:
            reason = []
            if thinking_leak:
                reason.append("thinking leak")
            if not row.get("passed"):
                reason.append("scorer fail")
            item["failures"].append({"case_id": row["case_id"], "reason": ", ".join(reason)})
    for item in by_model.values():
        item["raw_pass_rate"] = round(item["raw_pass"] / max(1, item["cases"]), 3)
        item["strict_pass_rate"] = round(item["strict_pass"] / max(1, item["cases"]), 3)
        item["avg_elapsed_sec"] = round(item["avg_elapsed_sec"] / max(1, item["cases"]), 3)
    return {
        "models": sorted(
            by_model.values(),
            key=lambda item: (item["strict_pass_rate"], item["raw_pass_rate"], -item["avg_elapsed_sec"]),
            reverse=True,
        )
    }


def retrieval_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    retrieval_rows = [row for row in rows if row.get("case_type") == "retrieval"]
    return {
        "passed": sum(1 for row in retrieval_rows if row.get("passed")),
        "total": len(retrieval_rows),
        "candidate_retrieval_passed": sum(1 for row in retrieval_rows if row.get("candidate_retrieval_passed")),
        "context_pack_passed": sum(1 for row in retrieval_rows if row.get("context_pack_passed")),
        "avg_context_mrr": round(
            sum(row.get("context_mrr", 0.0) for row in retrieval_rows) / max(1, len(retrieval_rows)),
            3,
        ),
        "cases": [
            {
                "case_id": row["case_id"],
                "passed": row["passed"],
                "parent_recall_at_k": row.get("parent_recall_at_k", row.get("recall_at_k")),
                "context_parent_recall_at_k": row.get("context_parent_recall_at_k"),
                "context_mrr": row.get("context_mrr"),
                "term_coverage": row.get("term_coverage"),
                "mrr": row.get("mrr"),
            }
            for row in retrieval_rows
        ],
    }


def top_structured_models() -> list[dict[str, Any]]:
    current_paths = [
        "output_data/model_bench/constrained_json_heldout_production_traces_v1_phi_functiongemma/constrained_summary.json",
        "output_data/model_bench/constrained_json_heldout_production_traces_v1_lfm/constrained_summary.json",
        "output_data/model_bench/constrained_json_production_contracts_v2_phi_retry/constrained_summary.json",
        "output_data/model_bench/constrained_json_production_contracts_v2_functiongemma_retry/constrained_summary.json",
        "output_data/model_bench/constrained_json_production_contracts_phi_fewshot_v1/constrained_summary.json",
    ]
    rows: list[dict[str, Any]] = []
    for current_path in current_paths:
        current = load_json_if_exists(current_path)
        if not current:
            continue
        for row in current.get("models", []):
            rows.append(
                {
                    "model": row["model"],
                    "label": model_short(row["model"]),
                    "score_pct": row.get("score_pct"),
                    "passed_cases": row.get("passed_cases"),
                    "cases": row.get("cases"),
                    "avg_elapsed_sec": row.get("avg_elapsed_sec"),
                    "fail_reasons": row.get("fail_reasons", {}),
                    "artifact": current_path,
                }
            )
    if rows:
        best_by_model: dict[str, dict[str, Any]] = {}
        for row in rows:
            existing = best_by_model.get(row["model"])
            row_key = (
                int("heldout_production_traces" in row.get("artifact", "")),
                row.get("cases") or 0,
                row.get("score_pct") or 0,
                row.get("passed_cases") or 0,
            )
            existing_key = (
                int("heldout_production_traces" in existing.get("artifact", "")),
                existing.get("cases") or 0,
                existing.get("score_pct") or 0,
                existing.get("passed_cases") or 0,
            ) if existing else None
            if existing is None or row_key > existing_key:
                best_by_model[row["model"]] = row
        deduped = list(best_by_model.values())
        deduped.sort(key=lambda item: (item.get("score_pct") or 0, item.get("passed_cases") or 0, -(item.get("avg_elapsed_sec") or 0)), reverse=True)
        return deduped[:3]

    path = ROOT / "output_data" / "model_bench" / "constrained_structured_v1" / "constrained_structured_summary.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "model": row["model"],
            "label": model_short(row["model"]),
            "score_pct": row.get("score_pct"),
            "passed": row.get("passed"),
            "avg_elapsed_sec": row.get("elapsed_sec"),
        }
        for row in data.get("leaders", [])[:3]
    ]


def service_rag_summary() -> dict[str, Any] | None:
    path = (
        "output_data/model_bench/rag_webchat_reranked_top3_numeric_citations_rescored/"
        "rag_webchat_summary.json"
    )
    data = load_json_if_exists(path)
    if not data:
        return None
    return {
        "artifact": path,
        "retrieval": data.get("retrieval", {}),
        "generation": data.get("generation", []),
        "slice_leaders": data.get("slice_leaders", {}),
    }


def phi_citation_retry_summary() -> dict[str, Any] | None:
    path = (
        "output_data/gold_eval/production_expanded_v2_generation_phi_citation_retry/"
        "production_gold_summary.json"
    )
    data = load_json_if_exists(path)
    if not data:
        return None
    model_rows = list(data.get("by_model", {}).values())
    if not model_rows:
        return None
    row = model_rows[0]
    return {
        "artifact": path,
        "model": "Phi-4-mini",
        "passed": row.get("passed"),
        "total": row.get("total"),
        "pass_rate": row.get("pass_rate"),
        "avg_elapsed_sec": row.get("avg_elapsed_sec"),
        "method": "numeric citation prompt plus validation-error retry when supported content lacks citations",
    }


def human_review_summary() -> dict[str, Any] | None:
    path = "output_data/gold_eval/human_review_v1/review_summary.json"
    data = load_json_if_exists(path)
    if not data:
        return None
    return {
        "artifact": path,
        "counts": data.get("counts", {}),
        "by_model": data.get("by_model", {}),
        "by_slice": data.get("by_slice", {}),
        "outputs": data.get("outputs", {}),
    }


def review_split_summary() -> dict[str, Any] | None:
    path = "output_data/gold_eval/human_review_v1_splits/split_summary.json"
    data = load_json_if_exists(path)
    if not data:
        return None
    return {
        "artifact": path,
        "counts": data.get("counts", {}),
        "unlabeled_by_model": data.get("unlabeled_by_model", {}),
        "unlabeled_by_slice": data.get("unlabeled_by_slice", {}),
        "training_rule": data.get("training_rule"),
        "outputs": data.get("outputs", {}),
    }


def hosted_migration_audit_summary() -> dict[str, Any] | None:
    path = "output_data/model_bench/hosted_model_migration_audit/hosted_model_migration_audit.json"
    data = load_json_if_exists(path)
    if not data:
        return None
    resolved_statuses = {
        "hosted_deployment_guarded",
        "legacy_runtime_guarded",
    }
    remaining = [
        {
            "path": row.get("path"),
            "workflow": row.get("workflow"),
            "status": row.get("status"),
            "decision": row.get("decision"),
            "replacement": row.get("replacement"),
        }
        for row in data.get("files", [])
        if row.get("hits")
        and not str(row.get("status", "")).startswith("local_")
        and row.get("status") not in resolved_statuses
    ]
    guarded = [
        {
            "path": row.get("path"),
            "workflow": row.get("workflow"),
            "status": row.get("status"),
            "decision": row.get("decision"),
            "replacement": row.get("replacement"),
        }
        for row in data.get("files", [])
        if row.get("hits") and row.get("status") in resolved_statuses
    ]
    return {
        "artifact": path,
        "counts": data.get("counts", {}),
        "remaining": remaining,
        "guarded": guarded,
        "migration_decision": data.get("migration_decision"),
    }


def llm_client_smoke_summary() -> dict[str, Any] | None:
    path = "output_data/model_bench/llm_client_local_smoke/llm_client_local_smoke.json"
    data = load_json_if_exists(path)
    if not data:
        return None
    return {
        "artifact": path,
        "passed": data.get("passed"),
        "backend": data.get("backend"),
        "prose_model": model_short(data.get("prose_model", "")),
        "json_model": model_short(data.get("json_model", "")),
        "tests": data.get("tests", []),
    }


def heldout_trace_summary() -> dict[str, Any] | None:
    path = "output_data/gold_eval/heldout_production_traces_v1/heldout_summary.json"
    data = load_json_if_exists(path)
    if not data:
        return None
    return {
        "artifact": path,
        "trace_count": data.get("trace_count"),
        "by_trace_type": data.get("by_trace_type", {}),
        "by_slice": data.get("by_slice", {}),
        "training_status": data.get("training_status"),
    }


def heldout_structured_summary() -> dict[str, Any] | None:
    paths = [
        "output_data/model_bench/constrained_json_heldout_production_traces_v1_phi_functiongemma/constrained_summary.json",
        "output_data/model_bench/constrained_json_heldout_production_traces_v1_lfm/constrained_summary.json",
    ]
    rows: list[dict[str, Any]] = []
    artifacts = []
    for path in paths:
        data = load_json_if_exists(path)
        if not data:
            continue
        artifacts.append(path)
        for row in data.get("models", []):
            rows.append(
                {
                    "model": row.get("model"),
                    "label": model_short(row.get("model", "")),
                    "cases": row.get("cases"),
                    "passed_cases": row.get("passed_cases"),
                    "score_pct": row.get("score_pct"),
                    "avg_elapsed_sec": row.get("avg_elapsed_sec"),
                    "fail_reasons": row.get("fail_reasons", {}),
                }
            )
    if not rows:
        return None
    rows.sort(key=lambda item: (item.get("score_pct") or 0, item.get("passed_cases") or 0, -(item.get("avg_elapsed_sec") or 0)), reverse=True)
    return {"artifacts": artifacts, "models": rows}


def local_chat_smoke_summary() -> dict[str, Any] | None:
    path = "output_data/model_bench/local_chat_backend_smoke/local_chat_backend_smoke.json"
    data = load_json_if_exists(path)
    if not data:
        return None
    query_rows = data.get("queries", [])
    first_query = query_rows[0] if query_rows else {}
    return {
        "artifact": path,
        "passed": data.get("passed"),
        "backend": data.get("backend"),
        "model": model_short(data.get("model", "")),
        "raw_model": data.get("model"),
        "citations_count": first_query.get("citations_count"),
        "has_timing_leak": first_query.get("has_timing_leak"),
        "reranked": first_query.get("reranked"),
        "reranker_model": first_query.get("reranker_model"),
        "query": first_query.get("query"),
    }


def regression_gate_summary() -> dict[str, Any] | None:
    path = "output_data/model_bench/regression_gate/model_regression_gate.json"
    data = load_json_if_exists(path)
    if not data:
        return None
    return {
        "artifact": path,
        "deployment_gate_passed": data.get("deployment_gate_passed"),
        "fine_tune_ready": data.get("fine_tune_ready"),
        "hard_failures": [row.get("name") for row in data.get("hard_failures", [])],
        "warnings": [row.get("name") for row in data.get("warnings", [])],
        "model_roster": data.get("model_roster", []),
        "decision": data.get("decision", {}),
        "checks": [
            {
                "name": row.get("name"),
                "passed": row.get("passed"),
                "severity": row.get("severity"),
                "evidence": row.get("evidence"),
            }
            for row in data.get("checks", [])
        ],
    }


def build_report() -> dict[str, Any]:
    expanded_generation_path = (
        "output_data/gold_eval/production_expanded_v2_generation_top3_compact/"
        "production_gold_results.jsonl"
    )
    seed_generation_path = (
        "output_data/gold_eval/production_seed_v1_generation_rag_top3_singleturn_v2/"
        "production_gold_results.jsonl"
    )
    generation_rows = load_jsonl_if_exists(expanded_generation_path if (ROOT / expanded_generation_path).exists() else seed_generation_path)
    retrieval_rows_k10 = load_jsonl_if_exists(
        "output_data/gold_eval/production_seed_v1_bench_parent_recall/production_gold_results.jsonl"
    )
    retrieval_rows_k15 = load_jsonl_if_exists(
        "output_data/gold_eval/production_seed_v1_retrieval_k15/production_gold_results.jsonl"
    )
    expanded_retrieval_path = (
        "output_data/gold_eval/production_expanded_v2_retrieval_citation_context_expanded/"
        "production_gold_results.jsonl"
    )
    expanded_retrieval_rows = load_jsonl_if_exists(expanded_retrieval_path)
    generation = strict_generation_summary(generation_rows)
    retrieval_k10 = retrieval_summary(retrieval_rows_k10)
    retrieval_k15 = retrieval_summary(retrieval_rows_k15)
    expanded_retrieval = retrieval_summary(expanded_retrieval_rows) if expanded_retrieval_rows else None
    service_rag = service_rag_summary()
    structured_top = top_structured_models()
    phi_retry = phi_citation_retry_summary()
    human_review = human_review_summary()
    review_splits = review_split_summary()
    hosted_audit = hosted_migration_audit_summary()
    llm_client_smoke = llm_client_smoke_summary()
    local_chat_smoke = local_chat_smoke_summary()
    heldout_traces = heldout_trace_summary()
    heldout_structured = heldout_structured_summary()
    regression_gate = regression_gate_summary()

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence": {
            "production_seed_gold": "output_data/gold_eval/production_seed_v1.jsonl",
            "production_expanded_gold": "output_data/gold_eval/production_expanded_v2.jsonl",
            "production_rag_generation": expanded_generation_path if (ROOT / expanded_generation_path).exists() else seed_generation_path,
            "production_expanded_retrieval_citation": expanded_retrieval_path if expanded_retrieval_rows else None,
            "phi_citation_retry": phi_retry["artifact"] if phi_retry else None,
            "human_review_packet": human_review["artifact"] if human_review else None,
            "human_review_splits": review_splits["artifact"] if review_splits else None,
            "hosted_model_migration_audit": hosted_audit["artifact"] if hosted_audit else None,
            "llm_client_local_smoke": llm_client_smoke["artifact"] if llm_client_smoke else None,
            "heldout_production_traces": heldout_traces["artifact"] if heldout_traces else None,
            "heldout_structured_contracts": heldout_structured["artifacts"] if heldout_structured else None,
            "local_chat_backend_smoke": local_chat_smoke["artifact"] if local_chat_smoke else None,
            "model_regression_gate": regression_gate["artifact"] if regression_gate else None,
            "service_shaped_rag_generation": (
                "output_data/model_bench/rag_webchat_reranked_top3_numeric_citations_rescored/"
                "rag_webchat_summary.json"
            ),
            "retrieval_k10_parent": "output_data/gold_eval/production_seed_v1_bench_parent_recall/production_gold_results.jsonl",
            "retrieval_k15_parent": "output_data/gold_eval/production_seed_v1_retrieval_k15/production_gold_results.jsonl",
            "structured_production_contracts": (
                "output_data/model_bench/constrained_json_production_contracts_v2_phi_retry/"
                "constrained_summary.json"
            ),
        },
        "regression_gate": regression_gate,
        "workflow_decisions": {
            "retrieval": {
                "decision": "base retrieval system mostly sufficient after candidate widening, BGE reranking, and top-3 context compression; structured-output retrieval still needs better evidence selection",
                "recommended_setting": "retrieve 15 sources, rerank with BAAI/bge-reranker-v2-m3, then compress to top 3 for local generation",
                "k10": retrieval_k10,
                "k15": retrieval_k15,
                "expanded_v1": expanded_retrieval,
                "service_shaped_rag_retrieval": service_rag.get("retrieval") if service_rag else None,
                "fine_tune_needed": False,
            },
            "rag_generation_and_webchat": {
                "decision": "base models are sufficient for expanded compact RAG with LFM2-2.6B; Phi-4-mini remains a backup but needs stricter citation prompting, while NVIDIA Nano is counted out for user-facing RAG due thinking leakage and 0/8 expanded pass",
                "top_models": generation["models"],
                "phi_backup_with_retry": phi_retry,
                "service_shaped_top_models": service_rag.get("generation", []) if service_rag else [],
                "slice_leaders": service_rag.get("slice_leaders", {}) if service_rag else {},
                "local_chat_backend_smoke": local_chat_smoke,
                "primary": "LFM2-2.6B",
                "backup": "Phi-4-mini",
                "demote": "NVIDIA Nano is not a RAG generator after 0/8 expanded compact pass with thinking leakage; Gemma 4 E2B remains demoted for thinking leakage",
                "fine_tune_needed": False,
            },
            "source_grounded_refusal": {
                "decision": "base sufficient on seed refusal cases with LFM2-2.6B or Phi; keep deterministic insufficient-source gates",
                "fine_tune_needed": False,
                "future_training_candidate": "hard negatives only if larger held-out refusal set exposes failures",
            },
            "structured_planning": {
                "decision": "base sufficient for expanded and held-out production contracts using Phi-4-mini with constrained JSON, verifier few-shot examples, and validation retry",
                "top_models": structured_top,
                "heldout_traces": heldout_traces,
                "heldout_structured": heldout_structured,
                "primary": "Phi-4-mini",
                "demote": "FunctionGemma is fast but only 7/16 on held-out production traces, so keep it only for trivial routes after validation; LFM2-2.6B fails this llama.cpp JSON-schema path with sampler initialization errors",
                "fine_tune_needed": False,
                "remaining_gate": "human-review failures and add more held-out traces as production schemas change before considering fine-tuning",
            },
            "brandon_finance_style": {
                "decision": "human-review packet and split tooling exist, but there are zero reviewed rows; not enough labels for final style or fine-tuning decision",
                "fine_tune_needed": "optional later",
                "human_review": human_review,
                "review_splits": review_splits,
                "next_gate": "label answer usefulness, finance relevance, and sentence/source citation support in the review packet",
            },
        },
        "current_deployment_recommendation": {
            "default_generator": "LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf",
            "webchat_backup": "unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf",
            "structured_control_plane": "unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf",
            "simple_gate_candidate": "lmstudio-community/functiongemma-270m-it-GGUF:functiongemma-270m-it-F16.gguf",
            "avoid_for_webchat_until_fixed": "unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf",
            "retriever": "BAAI/bge-m3",
            "reranker": "BAAI/bge-reranker-v2-m3",
            "chat_backend": "CHAT_BACKEND=llama_cpp",
            "chat_llama_cli": "CHAT_LLAMA_CLI=~/opt/llama.cpp/llama-cli",
            "chat_local_smoke": (
                "not yet captured as artifact"
                if not local_chat_smoke
                else (
                    f"artifact pass={local_chat_smoke['passed']}, "
                    f"model={local_chat_smoke['model']}, "
                    f"citations_count={local_chat_smoke['citations_count']}, "
                    f"timing_leak={local_chat_smoke['has_timing_leak']}"
                )
            ),
            "fine_tuning_needed_before_first_local_deployment": False,
        },
        "hosted_migration": hosted_audit,
        "llm_client_local_smoke": llm_client_smoke,
        "remaining_before_goal_completion": [
            "Human-label the generated review packet; current split summary has zero reviewed rows and zero approved training candidates.",
            "Keep the model regression gate passing and add more held-out traces as production behavior changes.",
            "Keep the disabled AgentCore/hosted paths guarded unless they are rewritten to call Hetzner-local services; current scanned hosted paths are either local-backed docs/code or disabled-by-default legacy runtimes.",
        ],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Production Model Decision",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Recommendation",
        "",
    ]
    dep = report["current_deployment_recommendation"]
    for key, value in dep.items():
        lines.append(f"- **{key}**: `{value}`")
    if report.get("regression_gate"):
        gate = report["regression_gate"]
        lines.extend(
            [
                "",
                "## Regression Gate",
                "",
                f"- Artifact: `{gate['artifact']}`",
                f"- Deployment gate passed: `{gate['deployment_gate_passed']}`",
                f"- Fine-tune ready: `{gate['fine_tune_ready']}`",
                f"- Hard failures: `{', '.join(gate['hard_failures']) if gate['hard_failures'] else 'none'}`",
                f"- Warnings: `{', '.join(gate['warnings']) if gate['warnings'] else 'none'}`",
            ]
        )
        counted_out = gate.get("decision", {}).get("counted_out", {})
        if counted_out:
            lines.append("- Counted out by gate:")
            for model, scope in counted_out.items():
                lines.append(f"  - `{model}`: {scope}")
        if gate.get("model_roster"):
            lines.append("- Model roster:")
            for row in gate["model_roster"]:
                lines.append(
                    f"  - `{row['label']}`: `{row['status']}` for {row['task']} "
                    f"(good_enough=`{row['good_enough']}`)"
                )
    lines.extend(["", "## Workflow Decisions", ""])
    for name, decision in report["workflow_decisions"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Decision: {decision['decision']}")
        lines.append(f"- Fine-tune needed: `{decision['fine_tune_needed']}`")
        if "recommended_setting" in decision:
            lines.append(f"- Setting: {decision['recommended_setting']}")
        if "primary" in decision:
            lines.append(f"- Primary: `{decision['primary']}`")
        if "backup" in decision:
            lines.append(f"- Backup: `{decision['backup']}`")
        if decision.get("phi_backup_with_retry"):
            retry = decision["phi_backup_with_retry"]
            lines.append(
                "- Backup validation/retry: "
                f"`{retry['model']}` pass `{retry['passed']}/{retry['total']}` "
                f"({retry['pass_rate']}) with {retry['method']}; avg `{retry['avg_elapsed_sec']}` sec"
            )
        if decision.get("local_chat_backend_smoke"):
            smoke = decision["local_chat_backend_smoke"]
            lines.append(
                "- Local backend smoke: "
                f"pass `{smoke['passed']}`, model `{smoke['model']}`, "
                f"citations `{smoke['citations_count']}`, "
                f"timing leak `{smoke['has_timing_leak']}`, "
                f"reranked `{smoke['reranked']}`"
            )
        if "top_models" in decision and decision["top_models"]:
            lines.append("- Top models:")
            for row in decision["top_models"][:3]:
                model = row.get("label") or model_short(row.get("model", ""))
                score = row.get("score_pct", row.get("strict_pass_rate", row.get("raw_pass_rate")))
                passed = row.get("passed_cases", row.get("strict_pass", row.get("passed")))
                cases = row.get("cases", "")
                lines.append(f"  - `{model}`: score `{score}`, pass `{passed}/{cases}`")
        if decision.get("heldout_structured"):
            lines.append("- Held-out structured traces:")
            for row in decision["heldout_structured"]["models"][:3]:
                lines.append(
                    f"  - `{row['label']}`: score `{row['score_pct']}`, "
                    f"pass `{row['passed_cases']}/{row['cases']}`, avg `{row['avg_elapsed_sec']}` sec"
                )
        if "demote" in decision:
            lines.append(f"- Demote: {decision['demote']}")
        if decision.get("human_review"):
            review = decision["human_review"]
            counts = review.get("counts", {})
            lines.append(
                "- Human review packet: "
                f"`{counts.get('answer_review_rows', 0)}` answers and "
                f"`{counts.get('citation_review_rows', 0)}` citation/support rows"
            )
        if decision.get("review_splits"):
            split_counts = decision["review_splits"].get("counts", {})
            lines.append(
                "- Review split status: "
                f"`{split_counts.get('reviewed_rows', 0)}` reviewed rows, "
                f"`{split_counts.get('candidate_train_pending_approval_rows', 0)}` train candidates, "
                f"`{split_counts.get('unlabeled_or_invalid_rows', 0)}` unlabeled/invalid rows"
            )
        lines.append("")
    lines.append("## Remaining Before Goal Completion")
    lines.append("")
    for item in report["remaining_before_goal_completion"]:
        lines.append(f"- {item}")
    if report.get("hosted_migration"):
        lines.extend(["", "## Hosted Migration Audit", ""])
        audit = report["hosted_migration"]
        counts = audit.get("counts", {})
        lines.append(f"- Artifact: `{audit['artifact']}`")
        lines.append(f"- Files with hosted/stale hits: `{counts.get('files_with_hosted_hits', 0)}`")
        lines.append(f"- Migration decision: {audit.get('migration_decision')}")
        if audit.get("remaining"):
            lines.append("- Remaining paths:")
            for row in audit.get("remaining", [])[:12]:
                lines.append(
                    f"  - `{row['path']}`: `{row['status']}`; replacement `{row.get('replacement', 'n/a')}`"
                )
        else:
            lines.append("- Remaining paths: `0` unresolved hosted runtime paths in scanned files")
        if audit.get("guarded"):
            lines.append(f"- Guarded legacy paths: `{len(audit['guarded'])}`")
    if report.get("llm_client_local_smoke"):
        smoke = report["llm_client_local_smoke"]
        lines.extend(["", "## Shared LLM Client Smoke", ""])
        lines.append(f"- Artifact: `{smoke['artifact']}`")
        lines.append(f"- Pass: `{smoke['passed']}`")
        lines.append(f"- Backend: `{smoke['backend']}`")
        lines.append(f"- Prose model: `{smoke['prose_model']}`")
        lines.append(f"- JSON model: `{smoke['json_model']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = OUT / "production_model_decision.json"
    md_path = OUT / "production_model_decision.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
