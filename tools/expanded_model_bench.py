#!/usr/bin/env python3
"""
Expanded per-role benchmark for local GGUF chat models.

This builds on model_quality_bench.py but runs multiple harder cases per role
against the models that were top candidates in the first role benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from model_quality_bench import Task, run_task, task_leaderboards


ROLE_MODELS = {
    "brief": [
        "LiquidAI/LFM2-1.2B-GGUF:LFM2-1.2B-Q4_K_M.gguf",
        "LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf",
        "LiquidAI/LFM2-8B-A1B-GGUF:LFM2-8B-A1B-Q4_0.gguf",
        "unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf",
    ],
    "finance": [
        "LiquidAI/LFM2-1.2B-GGUF:LFM2-1.2B-Q4_K_M.gguf",
        "LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf",
        "unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf",
        "unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf",
    ],
    "refusal": [
        "LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf",
        "unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf",
        "unsloth/gemma-4-E4B-it-GGUF:gemma-4-E4B-it-UD-IQ2_M.gguf",
        "unsloth/Qwen3.6-35B-A3B-GGUF:Qwen3.6-35B-A3B-UD-IQ1_M.gguf",
    ],
    "structured": [
        "unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf",
        "unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf",
        "unsloth/Qwen3.6-35B-A3B-GGUF:Qwen3.6-35B-A3B-UD-IQ1_M.gguf",
    ],
    "citation": [
        "unsloth/gemma-4-E4B-it-GGUF:gemma-4-E4B-it-UD-IQ2_M.gguf",
        "unsloth/Qwen3.6-35B-A3B-GGUF:Qwen3.6-35B-A3B-UD-IQ1_M.gguf",
        "unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf",
        "LiquidAI/LFM2-1.2B-GGUF:LFM2-1.2B-Q4_K_M.gguf",
    ],
}


EXPANDED_TASKS = [
    Task(
        task_id="brief_multi_source",
        kind="brief",
        max_tokens=420,
        prompt="""Sources:
[1] Arize Phoenix released an update focused on tracing multi-agent runs and comparing eval results across prompt versions.
[2] NVIDIA described a data-flywheel workflow where production traces are filtered, deduplicated, labeled, and promoted into train/eval datasets.
[3] LiquidAI released LFM2 MoE models that can run with low active parameter counts, making slow background CPU inference plausible.
[4] A team reported that local GGUF models were slower than hosted APIs but cheaper and easier to audit for overnight news workflows.

Write Brandon's AI news brief in 5 bullets. Use at least 3 citations. Include one second-order insight. Keep it practical and do not mention social posting.""",
    ),
    Task(
        task_id="brief_no_overclaim",
        kind="brief",
        max_tokens=360,
        prompt="""Sources:
[1] Gemma 4 E2B passed all six first-pass benchmark slices on the current VM.
[2] Qwen3.6-35B-A3B passed XML and citation slices but was much slower and failed unconstrained JSON.
[3] Phi-4-mini-instruct was strongest on unconstrained JSON among the tested candidates.
[4] LiquidAI LFM2-1.2B and 2.6B led the brief and finance slices but struggled with structured output.

Write a 4-bullet status brief for Brandon. Use citations. Include one second-order insight. Do not claim any model is production-ready without caveats.""",
    ),
    Task(
        task_id="finance_named_firms",
        kind="finance",
        max_tokens=360,
        prompt="""Sources:
[1] J.P. Morgan is expanding internal AI systems for research synthesis, controls, and developer productivity.
[2] Acadian, Balyasny, Arrowstreet, and Citadel are relevant buy-side comparables for AI research workflows, signal discovery, and model-risk review.
[3] Mastercard and Visa are applying AI to fraud, identity, payments risk, and compliance workflows.
[4] Local open-source models are relevant when regulated firms need auditability, private prototyping, and repeatable evals.

Write 4 bullets for Brandon. Mention at least 4 named companies. Cite every bullet. Focus on why this matters for a finance-facing role.""",
    ),
    Task(
        task_id="finance_domain_models",
        kind="finance",
        max_tokens=340,
        prompt="""Sources:
[1] Domain-specific financial models can help with filings, research notes, risk controls, and entity-heavy retrieval, but need strict evals.
[2] Open-source local models can reduce data exposure during prototyping, but weaker models may hallucinate finance claims.
[3] Phoenix-style traces and NeMo Curator-style filtering can turn production failures into eval and training examples.
[4] Brandon cares about hedge funds, asset managers, fintech, payments networks, and highly regulated financial institutions.

Write 4 bullets on model and workflow implications for Brandon. Include citations and at least one concrete workflow recommendation.""",
    ),
    Task(
        task_id="refusal_missing_finance_claim",
        kind="refusal",
        max_tokens=180,
        prompt="""Sources:
[1] Mastercard uses AI in fraud detection and payment risk workflows.
[2] Visa uses AI in identity and compliance workflows.

Question: Which open-source model did Citadel deploy internally last week, and what benchmark score did it report?

Answer in 2 sentences using only the sources.""",
    ),
    Task(
        task_id="refusal_conflicting_sources",
        kind="refusal",
        max_tokens=220,
        prompt="""Sources:
[1] A benchmark note says Gemma 4 E2B passed all six local benchmark slices.
[2] A separate note says Qwen3.6 was best for XML but failed unconstrained JSON.

Question: Did the sources prove that Qwen3.6 is the best overall model for Brandon's whole workflow?

Answer in 2 sentences using only the sources.""",
    ),
    Task(
        task_id="json_delivery_contract",
        kind="json",
        max_tokens=340,
        prompt="""Return strict JSON only with keys:
"brief_id", "audience", "sections", "required_checks".

Create a delivery contract for Brandon's daily brief. "sections" must be an array of objects with "name" and "purpose". Include sections for general AI news, finance relevance, citations, and operational caveats.""",
    ),
    Task(
        task_id="xml_agent_contract",
        kind="xml",
        max_tokens=360,
        prompt="""Return XML only with this shape:
<workflow>
  <step name=""></step>
  <roles>
    <role name=""><responsibility></responsibility></role>
  </roles>
  <model_requirements></model_requirements>
  <eval_checks></eval_checks>
</workflow>

Design the retrieval-agent contract for webchat inline citations. Include retrieval, reranking, answer generation, citation extraction, citation verification, and refusal handling.""",
    ),
    Task(
        task_id="citation_conflicting_evidence",
        kind="citation",
        max_tokens=320,
        prompt="""Sources:
[1] Gemma 4 E4B scored 10/10 on citation discipline in the first local benchmark.
[2] Qwen3.6 scored 10/10 on citation discipline but was much slower.
[3] LiquidAI LFM2-1.2B scored 8/10 on citation discipline and was much faster.
[4] Phi-4-mini-instruct requires a cloud GPU and cannot run on CPU.

Write 4 bullets recommending citation-verifier candidates for a CPU-only VM. Cite each bullet. Do not cite [4] for any claim that Phi is viable locally.""",
    ),
    Task(
        task_id="citation_source_limits",
        kind="citation",
        max_tokens=280,
        prompt="""Sources:
[1] Arize Phoenix supports tracing and eval workflows.
[2] NVIDIA NeMo Curator supports data filtering and synthetic data generation workflows.
[3] LLaMA-Factory supports SFT and preference-tuning workflows.
[4] The sources do not say that any of these tools automatically guarantees citation correctness.

Write 4 bullets about which tooling helps citation quality. Cite each bullet and make clear that tooling does not guarantee correctness.""",
    ),
]


ROLE_TASKS = {
    "brief": ["brief_multi_source", "brief_no_overclaim"],
    "finance": ["finance_named_firms", "finance_domain_models"],
    "refusal": ["refusal_missing_finance_claim", "refusal_conflicting_sources"],
    "structured": ["json_delivery_contract", "xml_agent_contract"],
    "citation": ["citation_conflicting_evidence", "citation_source_limits"],
}


def model_roles() -> dict[str, list[str]]:
    roles: dict[str, list[str]] = {}
    for role, models in ROLE_MODELS.items():
        for model in models:
            roles.setdefault(model, []).append(role)
    return roles


def summarize_by_role(rows: list[dict]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for role, task_ids in ROLE_TASKS.items():
        role_rows = [row for row in rows if row["task_id"] in task_ids]
        models = sorted({row["model"] for row in role_rows})
        model_summaries = []
        for model in models:
            ms = [row for row in role_rows if row["model"] == model]
            total = sum(row["score"]["points"] for row in ms)
            max_total = sum(row["score"]["max_points"] for row in ms)
            model_summaries.append(
                {
                    "model": model,
                    "tasks": len(ms),
                    "passed_tasks": sum(1 for row in ms if row["score"]["passed"]),
                    "score_pct": round(100 * total / max(1, max_total), 1),
                    "avg_elapsed_sec": round(sum(row["elapsed_sec"] for row in ms) / max(1, len(ms)), 3),
                    "fail_reasons": {
                        row["task_id"]: row["score"]["reasons"]
                        for row in ms
                        if row["score"]["reasons"]
                    },
                }
            )
        model_summaries.sort(key=lambda item: (item["score_pct"], item["passed_tasks"], -item["avg_elapsed_sec"]), reverse=True)
        summary[role] = {"task_ids": task_ids, "leaders": model_summaries}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-cli", default="~/opt/llama.cpp/llama-cli")
    parser.add_argument("--out-dir", default="output_data/model_bench/expanded_v1")
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--temp", type=float, default=0.15)
    parser.add_argument("--timeout", type=int, default=1500)
    parser.add_argument("--roles", nargs="*", choices=sorted(ROLE_MODELS), default=sorted(ROLE_MODELS))
    parser.add_argument("--models", nargs="*", help="Optional override: run these models for all selected roles")
    args = parser.parse_args()

    llama_cli = Path(args.llama_cli).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "expanded_results.jsonl"
    summary_path = out_dir / "expanded_summary.json"

    task_by_id = {task.task_id: task for task in EXPANDED_TASKS}
    roles_by_model = model_roles()
    rows = []
    with results_path.open("w", encoding="utf-8") as out:
        if args.models:
            model_plan = {model: list(args.roles) for model in args.models}
        else:
            model_plan = {
                model: [role for role in roles if role in args.roles]
                for model, roles in roles_by_model.items()
                if any(role in args.roles for role in roles)
            }

        for model, roles in model_plan.items():
            task_ids = []
            for role in roles:
                task_ids.extend(ROLE_TASKS[role])
            task_ids = list(dict.fromkeys(task_ids))
            print(f"MODEL {model}", flush=True)
            for task_id in task_ids:
                task = task_by_id[task_id]
                print(f"  TASK {task.task_id}", flush=True)
                row = run_task(
                    llama_cli=llama_cli,
                    model=model,
                    task=task,
                    ctx=args.ctx,
                    threads=args.threads,
                    temp=args.temp,
                    timeout=args.timeout,
                )
                row["roles"] = [role for role in roles if task_id in ROLE_TASKS[role]]
                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                print(
                    json.dumps(
                        {
                            "task_id": row["task_id"],
                            "returncode": row["returncode"],
                            "elapsed_sec": row["elapsed_sec"],
                            "score": row["score"]["points"],
                            "passed": row["score"]["passed"],
                        }
                    ),
                    flush=True,
                )

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results_path": str(results_path),
        "role_summary": summarize_by_role(rows),
        "task_leaderboards": task_leaderboards(rows, top_n=3),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
