#!/usr/bin/env python3
"""
Constrained JSON-schema benchmark for local llama.cpp planning outputs.

This isolates the agentic-planning boundary from prose quality. It uses
llama.cpp's --json-schema-file option so failures measure semantic planning
coverage after syntax/schema validity is handled by constrained decoding.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from model_quality_bench import clean_output, hf_env, model_args


DEFAULT_MODELS = [
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf",
    "unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf",
    "LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf",
    "unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf",
    "unsloth/Qwen3.6-35B-A3B-GGUF:Qwen3.6-35B-A3B-UD-IQ1_M.gguf",
]


SYSTEM_PROMPT = """You are the planning specialist inside Brandon's local CPU news agent.
Return only the JSON object required by the schema. Keep values concise.
Do not include hidden reasoning, markdown, code fences, social-posting language, or unsupported claims."""


PLANNING_PROMPT = """Design the agent workflow for Brandon's daily AI news system.

The system runs on a Hetzner CPU VM with local GGUF models. It must produce
source-grounded news summaries, highlight finance-domain signals for Brandon's
new role, support a webchat with inline citations, and refuse unsupported claims.

The plan must include these workflow responsibilities:
- retrieval
- reranking
- answer_generation
- citation_extraction
- citation_verification
- refusal_gate

It must include these validation checks:
- coverage_gate
- citation_validity
- schema_validity
- unsupported_refusal

Use exactly six workflow items and exactly four checks.
Avoid Twitter, LinkedIn, influencer, hashtag, or engagement-growth wording."""


JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["brief_id", "audience", "workflow", "routing", "checks"],
    "properties": {
        "brief_id": {"type": "string", "maxLength": 80},
        "audience": {"type": "string", "maxLength": 120},
        "workflow": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "required": ["step_name", "role", "inputs", "outputs", "model_role", "validation"],
                "properties": {
                    "step_name": {"type": "string", "maxLength": 80},
                    "role": {"type": "string", "maxLength": 100},
                    "inputs": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 80}},
                    "outputs": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 80}},
                    "model_role": {"type": "string", "maxLength": 100},
                    "validation": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 80}},
                },
            },
        },
        "routing": {
            "type": "object",
            "required": ["default_generator", "finance_backup", "structured_planner", "retriever", "fallback_retriever"],
            "properties": {
                "default_generator": {"type": "string", "maxLength": 120},
                "finance_backup": {"type": "string", "maxLength": 120},
                "structured_planner": {"type": "string", "maxLength": 120},
                "retriever": {"type": "string", "maxLength": 120},
                "fallback_retriever": {"type": "string", "maxLength": 120},
            },
        },
        "checks": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "required": ["name", "purpose"],
                "properties": {
                    "name": {"type": "string", "maxLength": 80},
                    "purpose": {"type": "string", "maxLength": 180},
                },
            },
        },
    },
}


REQUIRED_STEPS = {
    "retrieval",
    "reranking",
    "answer_generation",
    "citation_extraction",
    "citation_verification",
    "refusal_gate",
}

REQUIRED_CHECKS = {
    "coverage_gate",
    "citation_validity",
    "schema_validity",
    "unsupported_refusal",
}

BANNED_TERMS = {"twitter", "linkedin", "influencer", "hashtag", "engagement growth"}


def parse_json(text: str) -> Any | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


def flattened_strings(value: Any) -> str:
    parts: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            parts.append(flattened_strings(item))
    elif isinstance(value, list):
        for item in value:
            parts.append(flattened_strings(item))
    elif isinstance(value, str):
        parts.append(value)
    return " ".join(parts).lower()


def covered_terms(text: str, required: set[str]) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower())
    return {term for term in required if term in normalized}


def score_output(output: str, returncode: int) -> dict[str, Any]:
    reasons: list[str] = []
    points = 0
    max_points = 10
    parsed = parse_json(output)

    if returncode == 0:
        points += 1
    else:
        reasons.append(f"nonzero return code {returncode}")
    if parsed is not None:
        points += 3
    else:
        reasons.append("invalid JSON")
        return {"points": points, "max_points": max_points, "passed": False, "reasons": reasons}

    required_top = {"brief_id", "audience", "workflow", "routing", "checks"}
    if isinstance(parsed, dict) and required_top.issubset(parsed):
        points += 1
    else:
        reasons.append("missing required top-level keys")

    workflow = parsed.get("workflow") if isinstance(parsed, dict) else None
    if isinstance(workflow, list) and len(workflow) >= 6:
        points += 1
    else:
        reasons.append("workflow has fewer than six steps")

    all_text = flattened_strings(parsed)
    step_hits = covered_terms(all_text, REQUIRED_STEPS)
    if step_hits == REQUIRED_STEPS:
        points += 2
    else:
        reasons.append(f"missing required workflow responsibilities: {', '.join(sorted(REQUIRED_STEPS - step_hits))}")

    check_hits = covered_terms(all_text, REQUIRED_CHECKS)
    if check_hits == REQUIRED_CHECKS:
        points += 1
    else:
        reasons.append(f"missing required validation checks: {', '.join(sorted(REQUIRED_CHECKS - check_hits))}")

    if not any(term in all_text for term in BANNED_TERMS):
        points += 1
    else:
        reasons.append("included banned social/influencer language")

    return {
        "points": min(points, max_points),
        "max_points": max_points,
        "passed": min(points, max_points) >= 8,
        "reasons": reasons,
    }


def run_model(
    llama_cli: Path,
    model: str,
    schema_file: Path,
    ctx: int,
    threads: int,
    temp: float,
    timeout: int,
) -> dict[str, Any]:
    full_prompt = f"{SYSTEM_PROMPT}\n\nTASK:\n{PLANNING_PROMPT}\n\nJSON:\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as prompt_tmp:
        prompt_tmp.write(full_prompt)
        prompt_file = Path(prompt_tmp.name)

    cmd = [
        str(llama_cli),
        *model_args(model),
        "-f",
        str(prompt_file),
        "-c",
        str(ctx),
        "-n",
        "900",
        "-t",
        str(threads),
        "--temp",
        str(temp),
        "--json-schema-file",
        str(schema_file),
        "--no-display-prompt",
        "-no-cnv",
    ]

    started = time.time()
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, env=hf_env())
        elapsed = time.time() - started
        output = clean_output(proc.stdout)
        return {
            "model": model,
            "returncode": proc.returncode,
            "elapsed_sec": round(elapsed, 3),
            "output": output,
            "stderr_tail": proc.stderr.strip()[-2000:],
            "score": score_output(output, proc.returncode),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "model": model,
            "returncode": "timeout",
            "elapsed_sec": timeout,
            "output": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
            "score": {"points": 0, "max_points": 10, "passed": False, "reasons": ["timeout"]},
        }
    finally:
        try:
            prompt_file.unlink()
        except OSError:
            pass


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        rows,
        key=lambda row: (
            row["score"]["points"] / max(1, row["score"]["max_points"]),
            row["score"]["passed"],
            -row["elapsed_sec"],
        ),
        reverse=True,
    )
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema": JSON_SCHEMA,
        "leaders": [
            {
                "rank": i + 1,
                "model": row["model"],
                "score_pct": round(100 * row["score"]["points"] / max(1, row["score"]["max_points"]), 1),
                "passed": row["score"]["passed"],
                "elapsed_sec": row["elapsed_sec"],
                "reasons": row["score"]["reasons"],
            }
            for i, row in enumerate(ranked)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-cli", default="~/opt/llama.cpp/llama-completion")
    parser.add_argument("--out-dir", default="output_data/model_bench/constrained_structured_v1")
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()

    llama_cli = Path(args.llama_cli).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    schema_path = out_dir / "planning_schema.json"
    results_path = out_dir / "constrained_structured_results.jsonl"
    summary_path = out_dir / "constrained_structured_summary.json"
    schema_path.write_text(json.dumps(JSON_SCHEMA, indent=2) + "\n", encoding="utf-8")

    rows = []
    with results_path.open("w", encoding="utf-8") as out:
        for model in args.models:
            print(f"MODEL {model}", flush=True)
            row = run_model(
                llama_cli=llama_cli,
                model=model,
                schema_file=schema_path,
                ctx=args.ctx,
                threads=args.threads,
                temp=args.temp,
                timeout=args.timeout,
            )
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(
                json.dumps(
                    {
                        "returncode": row["returncode"],
                        "elapsed_sec": row["elapsed_sec"],
                        "score": row["score"]["points"],
                        "passed": row["score"]["passed"],
                        "reasons": row["score"]["reasons"],
                    }
                ),
                flush=True,
            )

    summary_doc = summarize(rows)
    summary_doc["results_path"] = str(results_path)
    summary_doc["schema_path"] = str(schema_path)
    summary_path.write_text(json.dumps(summary_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary_doc, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
