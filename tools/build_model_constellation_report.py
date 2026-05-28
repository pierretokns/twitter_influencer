#!/usr/bin/env python3
"""
Build the current local-model constellation report from benchmark summaries.

The report is intentionally opinionated: it preserves raw benchmark leaders and
adds deployment recommendations for the Hetzner CPU news-agent workflow.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


MODEL_LABELS = {
    "LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf": "LFM2-2.6B Q4_K_M",
    "LiquidAI/LFM2-1.2B-GGUF:LFM2-1.2B-Q4_K_M.gguf": "LFM2-1.2B Q4_K_M",
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf": "NVIDIA Nemotron 3 Nano 4B Q4_K_M",
    "unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf": "Gemma 4 E2B IT UD-IQ2_M",
    "unsloth/gemma-4-E4B-it-GGUF:gemma-4-E4B-it-UD-IQ2_M.gguf": "Gemma 4 E4B IT UD-IQ2_M",
    "unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf": "Phi-4-mini-instruct Q3_K_M",
    "unsloth/Qwen3.6-35B-A3B-GGUF:Qwen3.6-35B-A3B-UD-IQ1_M.gguf": "Qwen3.6-35B-A3B UD-IQ1_M",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def short_model(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def leader(model: str, score: float, source: str, *, passed: int | None = None, notes: list[str] | None = None) -> dict[str, Any]:
    row = {
        "model": model,
        "label": short_model(model),
        "score_pct": score,
        "source": source,
    }
    if passed is not None:
        row["passed_cases"] = passed
    if notes:
        row["notes"] = notes
    return row


def leaders_from_role(summary: dict[str, Any], role: str, source: str, limit: int = 3) -> list[dict[str, Any]]:
    rows = summary["role_summary"][role]["leaders"][:limit]
    return [
        leader(
            row["model"],
            row["score_pct"],
            source,
            passed=row.get("passed_tasks"),
            notes=[f"{task}: {', '.join(reasons)}" for task, reasons in row.get("fail_reasons", {}).items()],
        )
        for row in rows
    ]


def leaders_from_rag(summary: dict[str, Any], slice_name: str, source: str) -> list[dict[str, Any]]:
    rows = summary["slice_leaders"][slice_name]
    return [
        leader(
            row["model"],
            round(100 * row["points"] / max(1, row["max_points"]), 1),
            source,
            passed=int(row["passed"]),
            notes=row.get("reasons") or None,
        )
        for row in rows
    ]


def leaders_from_constrained(summary: dict[str, Any], source: str, limit: int = 3) -> list[dict[str, Any]]:
    rows = summary["leaders"][:limit]
    return [
        leader(
            row["model"],
            row["score_pct"],
            source,
            passed=1 if row.get("passed") else 0,
            notes=row.get("reasons") or None,
        )
        for row in rows
    ]


def append_unique(base: list[dict[str, Any]], extras: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    seen = {row["model"] for row in base}
    out = list(base)
    for row in extras:
        if row["model"] not in seen:
            out.append(row)
            seen.add(row["model"])
        if len(out) >= limit:
            break
    return out[:limit]


def retrieval_row_label(row: dict[str, Any]) -> str:
    embedder = row.get("embedder") or row.get("model") or "unknown"
    reranker = row.get("reranker")
    if reranker and reranker != "none":
        return f"{embedder} + {reranker}"
    return embedder


def retrieval_score(row: dict[str, Any]) -> float:
    context = float(row.get("finance_ai_context_recall", row.get("avg_context_recall") or 0.0))
    top10 = float(row.get("finance_ai_top10_recall", row.get("avg_top10_recall") or 0.0))
    return round(100.0 * ((0.6 * context) + (0.4 * top10)), 1)


def leaders_from_finance_retrieval(summary: dict[str, Any], source: str, limit: int = 3) -> list[dict[str, Any]]:
    rows = [
        row
        for row in summary.get("finance_ai_gate", [])
        if row.get("ok") and row.get("passed_finance_ai_gate")
    ]
    rows.sort(
        key=lambda row: (
            retrieval_score(row),
            row.get("finance_ai_context_recall", 0.0),
            row.get("finance_ai_top10_recall", 0.0),
            -float((row.get("timings") or {}).get("online_avg_per_query_sec", row.get("elapsed_sec") or 1e9)),
        ),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for row in rows[:limit]:
        timings = row.get("timings") or {}
        out.append(
            leader(
                retrieval_row_label(row),
                retrieval_score(row),
                source,
                passed=1,
                notes=[
                    f"finance_ai_gate={row.get('finance_ai_context_recall')}/{row.get('finance_ai_top10_recall')}",
                    f"online_avg_sec={timings.get('online_avg_per_query_sec')}",
                ],
            )
        )
    return out


def leaders_from_overall_retrieval(summary: dict[str, Any], source: str, limit: int = 3) -> list[dict[str, Any]]:
    rows = [row for row in summary.get("ranked", []) if row.get("ok") and row.get("passed")]
    rows.sort(
        key=lambda row: (
            retrieval_score(row),
            row.get("avg_context_recall", 0.0),
            row.get("avg_top10_recall", 0.0),
            -float((row.get("timings") or {}).get("online_avg_per_query_sec", row.get("elapsed_sec") or 1e9)),
        ),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for row in rows[:limit]:
        timings = row.get("timings") or {}
        score = round(100.0 * ((0.6 * row.get("avg_context_recall", 0.0)) + (0.4 * row.get("avg_top10_recall", 0.0))), 1)
        out.append(
            leader(
                retrieval_row_label(row),
                score,
                source,
                passed=1,
                notes=[
                    f"avg={row.get('avg_context_recall')}/{row.get('avg_top10_recall')}",
                    f"online_avg_sec={timings.get('online_avg_per_query_sec')}",
                ],
            )
        )
    return out


def leaders_from_fast_finance_retrieval(
    summary: dict[str, Any],
    source: str,
    limit: int = 3,
    max_online_avg_sec: float = 1.0,
) -> list[dict[str, Any]]:
    rows = []
    for row in summary.get("finance_ai_gate", []):
        timings = row.get("timings") or {}
        online_avg = timings.get("online_avg_per_query_sec")
        if not row.get("ok") or not row.get("passed_finance_ai_gate"):
            continue
        if online_avg is None or float(online_avg) > max_online_avg_sec:
            continue
        rows.append(row)
    rows.sort(
        key=lambda row: (
            retrieval_score(row),
            row.get("finance_ai_context_recall", 0.0),
            row.get("finance_ai_top10_recall", 0.0),
        ),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for row in rows[:limit]:
        timings = row.get("timings") or {}
        out.append(
            leader(
                retrieval_row_label(row),
                retrieval_score(row),
                source,
                passed=1,
                notes=[
                    f"finance_ai_gate={row.get('finance_ai_context_recall')}/{row.get('finance_ai_top10_recall')}",
                    f"online_avg_sec={timings.get('online_avg_per_query_sec')}",
                    f"fast_gate_sec<={max_online_avg_sec}",
                ],
            )
        )
    return out


def build_report(root: Path) -> dict[str, Any]:
    bench = root / "output_data" / "model_bench"
    expanded = load_json(bench / "expanded_v1" / "expanded_summary.json")
    expanded_nvidia = load_json(bench / "expanded_nvidia_nano" / "expanded_summary.json")
    rag = load_json(bench / "rag_webchat_focus_k10_rescored_citations" / "rag_webchat_summary.json")
    retrieval = load_json(bench / "retrieval_components_v1" / "retrieval_component_summary.json")
    alias_retrieval = load_json(bench / "rag_webchat_alias_retrieval_probe_v2" / "rag_webchat_summary.json")
    finance_retrieval = load_json_if_exists(bench / "retrieval_pipeline_matrix_finance_ai_gate_v1" / "retrieval_pipeline_matrix_summary.json")
    e2rank_retrieval = load_json_if_exists(bench / "e2rank_listwise_smoke_300" / "retrieval_pipeline_matrix_summary.json")
    pplx_retrieval = load_json_if_exists(bench / "retrieval_pipeline_matrix_pplx_0_6b_smoke_300_v1" / "retrieval_pipeline_matrix_summary.json")
    colbert_probe = load_json_if_exists(bench / "colbert_late_interaction_mixedbread_smoke" / "colbert_late_interaction_summary.json")
    constrained_structured_path = bench / "constrained_structured_v1" / "constrained_structured_summary.json"
    constrained_structured = load_json(constrained_structured_path)

    nvidia_structured = leaders_from_role(expanded_nvidia, "structured", "expanded_nvidia_nano")
    nvidia_refusal = leaders_from_role(expanded_nvidia, "refusal", "expanded_nvidia_nano")
    nvidia_brief = leaders_from_role(expanded_nvidia, "brief", "expanded_nvidia_nano")
    nvidia_citation = leaders_from_role(expanded_nvidia, "citation", "expanded_nvidia_nano")

    slices: dict[str, dict[str, Any]] = {
        "general_brandon_brief": {
            "top3": append_unique(
                leaders_from_role(expanded, "brief", "expanded_v1", 2),
                nvidia_brief + leaders_from_role(expanded, "brief", "expanded_v1", 4),
            ),
            "recommendation": "Use LFM2-2.6B as default brief writer; use Gemma 4 E2B when prose quality matters more than speed. Keep NVIDIA Nano as fallback only if structured/cautious wording is more important than warmth.",
            "fine_tune": "Optional SFT for Brandon-specific style and second-order insight consistency; not required for first deployment.",
        },
        "finance_domain_signal": {
            "top3": leaders_from_rag(rag, "finance", "rag_webchat_focus_k10_rescored_citations"),
            "supporting_expanded_top3": leaders_from_role(expanded, "finance", "expanded_v1"),
            "retrieval_coverage": alias_retrieval["retrieval"]["rows"][0],
            "recommendation": "Use LFM2-2.6B for sourced finance synthesis, Gemma 4 E2B as finance-writing backup, NVIDIA Nano third. Use the expanded finance result to keep LFM2-1.2B/Phi in reserve for non-RAG lightweight classification.",
            "fine_tune": "Not required for first deployment if retrieval coverage remains strong; useful later for finance-entity specificity and house style.",
        },
        "source_grounded_refusal": {
            "top3": leaders_from_rag(rag, "refusal", "rag_webchat_focus_k10_rescored_citations"),
            "supporting_expanded_top3": append_unique(nvidia_refusal, leaders_from_role(expanded, "refusal", "expanded_v1"), 3),
            "recommendation": "Use deterministic retrieval coverage and citation validators before generation. LFM2-2.6B is best in the focused sourced run, but all base models need guardrails around unsupported-source citation style.",
            "fine_tune": "High priority. Train SFT plus DPO/ORPO/KTO hard negatives for unsupported, weak, stale, conflicting, and over-refusal cases.",
        },
        "rag_webchat_inline_citations": {
            "top3": leaders_from_rag(rag, "citation", "rag_webchat_focus_k10_rescored_citations"),
            "supporting_prompt_citation_top3": append_unique(nvidia_citation, leaders_from_role(expanded, "citation", "expanded_v1"), 3),
            "recommendation": "Use LFM2-2.6B for webchat citation generation. Treat NVIDIA Nano's older citation strength as prompt-only/synthetic; in real webchat it produced invalid [N] placeholders.",
            "fine_tune": "Medium/high priority for citation hygiene. Keep post-generation citation verification regardless of fine-tuning.",
        },
        "agentic_planning_structured_output": {
            "top3": leaders_from_constrained(constrained_structured, "constrained_structured_v1"),
            "supporting_prompt_structured_top3": append_unique(nvidia_structured, leaders_from_role(expanded, "structured", "expanded_v1"), 3),
            "recommendation": "Use bounded JSON-schema constrained decoding through llama-completion raw mode for planning contracts. LFM2-2.6B is the best constrained planner in the current run; Phi-4-mini and Gemma 4 E2B are viable backups. NVIDIA Nano remains a good prompt-only structured model, but missed the required semantic labels in constrained raw mode.",
            "fine_tune": "Do not fine-tune first. Constrained decoding solved schema validity for the top candidates; add more planning cases before training. Fine-tune only if semantic role/tool choices fail after schema constraints and retries.",
        },
        "retrieval_embeddings": {
            "top3": (
                leaders_from_finance_retrieval(finance_retrieval, "retrieval_pipeline_matrix_finance_ai_gate_v1")
                if finance_retrieval
                else retrieval["ranked"][:3]
            ),
            "fast_top3": (
                leaders_from_fast_finance_retrieval(finance_retrieval, "retrieval_pipeline_matrix_finance_ai_gate_v1")
                if finance_retrieval
                else retrieval["ranked"][:3]
            ),
            "overall_top3": leaders_from_overall_retrieval(finance_retrieval, "retrieval_pipeline_matrix_finance_ai_gate_v1") if finance_retrieval else retrieval["ranked"][:3],
            "counted_out_or_deprioritized": [
                "PPLX 0.6B failed the 300-doc finance-AI gate on the current SentenceTransformers path.",
                "E2Rank-0.6B custom path works, but the 300-doc smoke missed the pass threshold and scored 0.0/0.0 on finance-AI.",
                "ColBERT probes are not fair quality evidence yet because tested HF repos fell back to generic SentenceTransformers mean-pooling.",
                "NVIDIA Nemotron embed 1B remains impractical on this 8GB CPU VM without an optimized runtime/swap plan.",
            ],
            "recommendation": "For fast Brandon finance-AI retrieval, prioritize Granite 97M no-reranker and EmbeddingGemma no-reranker. Keep production BGE-M3 hybrid for the currently wired service path until a reindex/backfill migration is implemented. Use ModernBERT reranking only for slower background synthesis or validation paths.",
            "fine_tune": "No LLM fine-tune needed for retrieval yet; source coverage and reranking are higher leverage.",
        },
    }

    deployment = {
        "default_generator": "LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf",
        "finance_backup": "unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf",
        "structured_planner_candidate": "LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf",
        "structured_planner_backup": "unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf",
        "retriever": "BAAI/bge-m3",
        "candidate_retriever_for_next_reindex": "ibm-granite/granite-embedding-97m-multilingual-r2",
        "fallback_retriever": "google/embeddinggemma-300m or sentence-transformers/all-MiniLM-L6-v2 depending on storage/runtime constraints",
        "base_model_sufficient_for_first_deployment": True,
        "fine_tuning_needed_before_first_deployment": False,
        "fine_tuning_priority": [
            "unsupported-source refusal with citation-grounded refusals",
            "citation hygiene and placeholder-citation suppression",
            "Brandon/finance style and second-order insight consistency",
            "semantic planning only if expanded constrained-schema evals fail",
        ],
        "prune_or_do_not_restore": [
            "LiquidAI/LFM2-8B-A1B unless retested with a new quant or task",
            "Qwen3.5 small variants from earlier smoke tests",
            "Gemma 4 26B on this 8GB CPU VM unless storage/swap constraints change",
            "NVIDIA llama-nemotron-embed-1b-v2 on current no-swap VM",
        ],
        "keep_cached_or_retest": [
            "LFM2-2.6B",
            "Gemma 4 E2B",
            "NVIDIA Nemotron 3 Nano 4B",
            "Phi-4-mini-instruct for structured fallback",
            "Qwen3.6-35B-A3B only if disk allows a slow quality reference",
        ],
    }

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_paths": {
            "expanded_v1": str(bench / "expanded_v1" / "expanded_summary.json"),
            "expanded_nvidia_nano": str(bench / "expanded_nvidia_nano" / "expanded_summary.json"),
            "constrained_structured": str(constrained_structured_path),
            "rag_webchat_rescored": str(bench / "rag_webchat_focus_k10_rescored_citations" / "rag_webchat_summary.json"),
            "rag_alias_retrieval": str(bench / "rag_webchat_alias_retrieval_probe_v2" / "rag_webchat_summary.json"),
            "retrieval_components": str(bench / "retrieval_components_v1" / "retrieval_component_summary.json"),
            "finance_ai_retrieval_matrix": str(bench / "retrieval_pipeline_matrix_finance_ai_gate_v1" / "retrieval_pipeline_matrix_summary.json") if finance_retrieval else None,
            "pplx_retrieval_smoke": str(bench / "retrieval_pipeline_matrix_pplx_0_6b_smoke_300_v1" / "retrieval_pipeline_matrix_summary.json") if pplx_retrieval else None,
            "e2rank_retrieval_smoke": str(bench / "e2rank_listwise_smoke_300" / "retrieval_pipeline_matrix_summary.json") if e2rank_retrieval else None,
            "colbert_runtime_probe": str(bench / "colbert_late_interaction_mixedbread_smoke" / "colbert_late_interaction_summary.json") if colbert_probe else None,
        },
        "deployment": deployment,
        "slices": slices,
        "remaining_gaps": [
            "Acadian and payments still missing from the finance top-10 retrieval probe.",
            "Citation-webchat retrieval passes only at the minimum gate and should get richer citation/retrieval eval sources.",
            "Production service is still wired to BGE-M3 hybrid embeddings; Granite/EmbeddingGemma retrieval leaders need a reindex/backfill path before they can replace production retrieval.",
            "ColBERT needs a model-specific loader or official example path before quality count-out.",
            "Structured output now passes one bounded constrained-schema planning contract for LFM2-2.6B, Phi-4-mini, and Gemma 4 E2B, but needs more planning cases and a production wrapper around llama-completion raw mode.",
            "Need a final production wiring change from hosted Claude/Bedrock to local llama.cpp service if deployment migration is in scope.",
        ],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Local Model Constellation Report",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Deployment Recommendation",
        "",
    ]
    dep = report["deployment"]
    for key in [
        "default_generator",
        "finance_backup",
        "structured_planner_candidate",
        "structured_planner_backup",
        "retriever",
        "candidate_retriever_for_next_reindex",
        "fallback_retriever",
    ]:
        if key in dep:
            lines.append(f"- **{key}**: `{dep[key]}`")
    lines.extend(
        [
            f"- **Base model sufficient for first deployment**: `{dep['base_model_sufficient_for_first_deployment']}`",
            f"- **Fine-tuning needed before first deployment**: `{dep['fine_tuning_needed_before_first_deployment']}`",
            "",
            "## Slices",
            "",
        ]
    )
    for slice_name, data in report["slices"].items():
        lines.append(f"### {slice_name}")
        lines.append("")
        lines.append(data["recommendation"])
        lines.append("")
        lines.append(f"Fine-tune: {data['fine_tune']}")
        lines.append("")
        lines.append("| Rank | Model | Score | Source | Notes |")
        lines.append("|---:|---|---:|---|---|")
        for i, row in enumerate(data["top3"], 1):
            notes = "; ".join(row.get("notes", []))
            lines.append(
                f"| {i} | `{row.get('model', row.get('model', ''))}` | {row.get('score_pct', row.get('mrr', ''))} | {row.get('source', '')} | {notes} |"
            )
        lines.append("")
        if data.get("fast_top3"):
            lines.append("Fast CPU shortlist:")
            lines.append("")
            lines.append("| Rank | Model | Score | Source | Notes |")
            lines.append("|---:|---|---:|---|---|")
            for i, row in enumerate(data["fast_top3"], 1):
                notes = "; ".join(row.get("notes", []))
                lines.append(
                    f"| {i} | `{row.get('model', '')}` | {row.get('score_pct', '')} | {row.get('source', '')} | {notes} |"
                )
            lines.append("")
    lines.append("## Remaining Gaps")
    lines.append("")
    for gap in report["remaining_gaps"]:
        lines.append(f"- {gap}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-dir", default="output_data/model_bench/constellation")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(root)
    json_path = out_dir / "model_constellation_report.json"
    md_path = out_dir / "model_constellation_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
