#!/usr/bin/env python3
"""
Task-specific quality benchmark for local llama.cpp GGUF models.

This is intentionally rule-scored. It is meant to separate obvious misses from
usable candidates before spending time on longer distilled-log evals or SFT.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are running inside a background news agent for Brandon.
Answer only the requested task. Be concise, source-grounded, and useful.
Do not write hashtags, influencer bait, LinkedIn copy, or hidden reasoning.
If the sources do not support a claim, say that directly instead of guessing."""


DEFAULT_MODELS = [
    "LiquidAI/LFM2-1.2B-GGUF:LFM2-1.2B-Q4_K_M.gguf",
    "LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf",
    "LiquidAI/LFM2-8B-A1B-GGUF:LFM2-8B-A1B-Q4_0.gguf",
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf",
    "unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf",
    "unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-UD-IQ2_XXS.gguf",
    "unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf",
    "unsloth/gemma-4-E4B-it-GGUF:gemma-4-E4B-it-UD-IQ2_M.gguf",
    "unsloth/Qwen3.6-35B-A3B-GGUF:Qwen3.6-35B-A3B-UD-IQ1_M.gguf",
]

STRESS_MODELS = [
    "LiquidAI/LFM2-24B-A2B-GGUF:LFM2-24B-A2B-Q4_0.gguf",
]


@dataclass(frozen=True)
class Task:
    task_id: str
    kind: str
    max_tokens: int
    prompt: str


TASKS = [
    Task(
        task_id="brandon_brief",
        kind="brief",
        max_tokens=360,
        prompt="""Sources:
[1] Arize Phoenix added tracing and evaluation workflows that help teams inspect agent runs, compare prompts, and catch regressions before deploying model changes.
[2] NVIDIA published Data Flywheel and NeMo Curator guidance for collecting production traces, filtering low-quality samples, removing sensitive data, and turning logs into training and evaluation datasets.
[3] Unsloth released dynamic GGUF quantizations for Qwen, Gemma, Phi, and other open models, making small CPU inference more practical on constrained servers.
[4] Hetzner cx33 has 4 shared vCPUs and 8 GB RAM, so local models need small quantized GGUF files and strong instruction following.

Write Brandon's morning AI news brief in 5 bullets. Include at least 3 citations using [1]-[4]. Include one "second-order insight" bullet. Do not mention Twitter, LinkedIn, influencing, or hashtags.""",
    ),
    Task(
        task_id="finance_relevance",
        kind="finance",
        max_tokens=300,
        prompt="""Sources:
[1] J.P. Morgan expanded internal AI workflows for research summarization, controls, and developer productivity across regulated banking teams.
[2] Hedge funds and asset managers such as Acadian Asset Management, Balyasny, Arrowstreet Capital, and Citadel are evaluating agent workflows for research triage, signal discovery, and model-risk review.
[3] Mastercard and Visa are deploying AI in fraud detection, payments risk, identity, and compliance workflows.
[4] Open-source and domain-specific local models can help regulated firms prototype privately, but require strong evals, audit logs, citation checks, and data governance.

Write 4 bullets for Brandon on why these items matter for his finance-facing role. Mention at least 3 named companies from the sources. Include citations. Avoid unsupported claims.""",
    ),
    Task(
        task_id="source_refusal",
        kind="refusal",
        max_tokens=180,
        prompt="""Sources:
[1] Arize Phoenix can trace agent runs and evaluate prompt changes.
[2] llama.cpp can run GGUF models on CPU.

Question: Which Anthropic pricing tier changed on May 25, 2026, and what exact clause changed?

Answer in 2 sentences using only the sources.""",
    ),
    Task(
        task_id="workflow_json",
        kind="json",
        max_tokens=320,
        prompt="""Return strict JSON only, with keys:
"workflow", "roles", "model_requirements", "eval_checks".

Design a local CPU agent workflow for a Brandon news summary system. The workflow must include source collection, ranking, summarization, citation verification, and delivery. "roles" must be an array of objects with "name" and "responsibility".""",
    ),
    Task(
        task_id="workflow_xml",
        kind="xml",
        max_tokens=320,
        prompt="""Return XML only with this shape:
<workflow>
  <step name=""></step>
  <roles>
    <role name=""><responsibility></responsibility></role>
  </roles>
  <model_requirements></model_requirements>
  <eval_checks></eval_checks>
</workflow>

Design a local CPU agent workflow for a Brandon news summary system. The workflow must include source collection, ranking, summarization, citation verification, and delivery.""",
    ),
    Task(
        task_id="citation_discipline",
        kind="citation",
        max_tokens=260,
        prompt="""Sources:
[1] Qwen3.6-35B-A3B is a mixture-of-experts model with about 3B active parameters per token.
[2] Gemma 4 E4B is a compact instruction model option for local CPU testing.
[3] LiquidAI LFM2-2.6B passed the smoke test on the Hetzner VM.
[4] Phi-4-mini-instruct requires a cloud GPU and cannot run on CPU. 

Write 4 bullets comparing these model candidates for a CPU-only Hetzner VM. Cite every factual bullet. Do not cite [4] for any claim that says Phi can run locally.""",
    ),
]


def model_args(model: str) -> list[str]:
    if ":" in model and model.endswith(".gguf"):
        repo, hf_file = model.split(":", 1)
        return ["--hf-repo", repo, "--hf-file", hf_file]
    return ["-hf", model]


def hf_env() -> dict[str, str]:
    env = os.environ.copy()
    token_path = Path("~/.cache/huggingface/token").expanduser()
    if token_path.exists() and "HF_TOKEN" not in env:
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            env["HF_TOKEN"] = token
    return env


def clean_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    if "\n>" in text:
        text = text.rsplit("\n>", 1)[-1].strip()
    if " ... (truncated)\n\n" in text:
        text = text.split(" ... (truncated)\n\n", 1)[-1].strip()
    elif "ANSWER:" in text:
        text = text.rsplit("ANSWER:", 1)[-1].strip()
    elif "TASK:" in text:
        text = text.rsplit("TASK:", 1)[-1].strip()
    text = re.sub(r"^ASSISTANT RESPONSE:\s*", "", text, flags=re.I)
    text = re.sub(r"^Loading model\.\.\..*?available commands:.*?(?:\n\s*>|\Z)", "", text, flags=re.S)
    text = re.sub(r"\n?\[ Prompt: .*?Generation: .*?\]\s*", "\n", text, flags=re.S)
    text = re.sub(r"\n?Exiting\.\.\.\s*$", "", text)
    return text.strip()


def run_task(
    llama_cli: Path,
    model: str,
    task: Task,
    ctx: int,
    threads: int,
    temp: float,
    timeout: int,
) -> dict[str, Any]:
    full_prompt = f"{SYSTEM_PROMPT}\n\nTASK:\n{task.prompt}\n\nANSWER:\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tmp:
        tmp.write(full_prompt)
        tmp_path = Path(tmp.name)

    cmd = [
        str(llama_cli),
        *model_args(model),
        "-f",
        str(tmp_path),
        "-c",
        str(ctx),
        "-n",
        str(task.max_tokens),
        "-t",
        str(threads),
        "--temp",
        str(temp),
        "--no-display-prompt",
        "--log-disable",
        "--simple-io",
        "--no-show-timings",
        "--reasoning",
        "off",
        "--single-turn",
    ]
    started = time.time()
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, env=hf_env())
        elapsed = time.time() - started
        output = clean_output(proc.stdout)
        stderr = proc.stderr.strip()
        return {
            "task_id": task.task_id,
            "kind": task.kind,
            "model": model,
            "returncode": proc.returncode,
            "elapsed_sec": round(elapsed, 3),
            "output": output,
            "stderr_tail": stderr[-2000:],
            "score": score_task(task, output, proc.returncode),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "task_id": task.task_id,
            "kind": task.kind,
            "model": model,
            "returncode": "timeout",
            "elapsed_sec": timeout,
            "output": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
            "score": {"points": 0, "max_points": 10, "passed": False, "reasons": ["timeout"]},
        }
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def score_task(task: Task, output: str, returncode: int) -> dict[str, Any]:
    reasons: list[str] = []
    text = output.strip()
    lower = text.lower()
    points = 0
    max_points = 10

    if returncode == 0:
        points += 1
    else:
        reasons.append(f"nonzero return code {returncode}")
    if text:
        points += 1
    else:
        reasons.append("empty output")
    if 80 <= len(text) <= 2600:
        points += 1
    else:
        reasons.append("bad length")
    if "<think>" not in lower and "</think>" not in lower and "we need answer" not in lower:
        points += 1
    else:
        reasons.append("thinking trace leaked")

    if task.kind == "brief":
        citations = set(re.findall(r"\[(\d+)\]", text))
        if len(citations & {"1", "2", "3", "4", "5", "6"}) >= 3:
            points += 2
        else:
            reasons.append("missing required citations")
        if "second-order" in lower or "second order" in lower:
            points += 1
        else:
            reasons.append("missing second-order insight")
        banned = ["twitter", "linkedin", "hashtag", "#"]
        if not any(term in lower for term in banned):
            points += 1
        else:
            reasons.append("included banned social/influencer language")
        if all(term in lower for term in ["brandon", "citation"]) or "source" in lower:
            points += 1
        else:
            reasons.append("weak brandon/source framing")
        if any(term in lower for term in ["phoenix", "curator", "quant", "gguf", "hetzner"]):
            points += 1
        else:
            reasons.append("weak source detail")

    elif task.kind == "refusal":
        refusal_terms = [
            "not in the sources",
            "not supported",
            "do not specify",
            "doesn't specify",
            "cannot determine",
            "provided sources do not contain",
            "sources do not contain",
            "do not include",
            "do not mention",
            "no information",
            "not enough information",
        ]
        if any(term in lower for term in refusal_terms):
            points += 3
        else:
            reasons.append("did not refuse unsupported fact")
        hallucination_terms = ["claude pro", "claude max", "team plan", "enterprise plan", "opus", "sonnet", "$"]
        if not any(term in lower for term in hallucination_terms):
            points += 2
        else:
            reasons.append("invented likely pricing details")
        if len(re.split(r"[.!?]+", text)) <= 4:
            points += 1
        else:
            reasons.append("too many sentences")

    elif task.kind == "finance":
        citations = set(re.findall(r"\[(\d+)\]", text))
        if len(citations & {"1", "2", "3", "4"}) >= 3:
            points += 2
        else:
            reasons.append("missing finance citations")
        named_companies = [
            "j.p. morgan",
            "jp morgan",
            "acadian",
            "balyasny",
            "arrowstreet",
            "citadel",
            "mastercard",
            "visa",
        ]
        if sum(1 for company in named_companies if company in lower) >= 3:
            points += 2
        else:
            reasons.append("missing named finance companies")
        domain_terms = ["regulated", "compliance", "risk", "audit", "governance", "hedge fund", "asset manager"]
        if any(term in lower for term in domain_terms):
            points += 1
        else:
            reasons.append("weak regulated-finance framing")
        if any(term in lower for term in ["workflow", "agent", "model", "eval", "open-source", "local"]):
            points += 1
        else:
            reasons.append("weak AI workflow/model framing")

    elif task.kind == "json":
        parsed = parse_json_object(text)
        if parsed is not None:
            points += 3
            required = {"workflow", "roles", "model_requirements", "eval_checks"}
            if required.issubset(parsed):
                points += 2
            else:
                reasons.append("missing required JSON keys")
            roles = parsed.get("roles")
            if isinstance(roles, list) and roles and all(
                isinstance(role, dict) and {"name", "responsibility"}.issubset(role) for role in roles
            ):
                points += 1
            else:
                reasons.append("roles schema mismatch")
        else:
            reasons.append("invalid JSON")

    elif task.kind == "xml":
        root = parse_xml_object(text)
        if root is not None and root.tag == "workflow":
            points += 3
            if root.findall(".//step"):
                points += 1
            else:
                reasons.append("missing workflow steps")
            roles = root.findall(".//role")
            if roles and all(role.attrib.get("name") and role.findtext("responsibility") for role in roles):
                points += 2
            else:
                reasons.append("roles XML schema mismatch")
            serialized = text.lower()
            if all(term in serialized for term in ["source", "ranking", "summarization", "citation", "delivery"]):
                points += 1
            else:
                reasons.append("missing required workflow coverage")
        else:
            reasons.append("invalid XML")

    elif task.kind == "citation":
        citations = set(re.findall(r"\[(\d+)\]", text))
        if {"1", "2", "3"}.issubset(citations):
            points += 2
        else:
            reasons.append("missing core model citations")
        phi_can_run_claim = "phi" in lower and any(
            term in lower
            for term in [
                "phi can run",
                "phi runs",
                "phi is viable",
                "phi is suitable",
                "phi fits",
                "phi works",
            ]
        )
        if not (phi_can_run_claim and "[4]" in text):
            points += 2
        else:
            reasons.append("misused false Phi source")
        bullet_count = sum(
            1
            for line in text.splitlines()
            if line.strip().startswith(("-", "*")) or re.match(r"^\s*\d+[.)]", line)
        )
        if bullet_count >= 3:
            points += 1
        else:
            reasons.append("not enough bullets")
        if "hetzner" in lower or "8 gb" in lower or "cpu" in lower:
            points += 1
        else:
            reasons.append("weak vm fit discussion")

    return {
        "points": min(points, max_points),
        "max_points": max_points,
        "passed": min(points, max_points) >= 7,
        "reasons": reasons,
    }


def parse_json_object(text: str) -> Any | None:
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


def parse_xml_object(text: str) -> Any | None:
    import xml.etree.ElementTree as ET

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:xml)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("<workflow")
    end = cleaned.rfind("</workflow>")
    if start == -1 or end == -1:
        return None
    fragment = cleaned[start : end + len("</workflow>")]
    try:
        return ET.fromstring(fragment)
    except ET.ParseError:
        return None


def summarize(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(row["score"]["points"] for row in rows)
    max_total = sum(row["score"]["max_points"] for row in rows)
    completed = [row for row in rows if row["returncode"] == 0 and row["output"].strip()]
    return {
        "model": model,
        "tasks": len(rows),
        "completed": len(completed),
        "total_score": total,
        "max_score": max_total,
        "score_pct": round(100 * total / max(1, max_total), 1),
        "passed_tasks": sum(1 for row in rows if row["score"]["passed"]),
        "avg_elapsed_sec": round(sum(row["elapsed_sec"] for row in rows) / max(1, len(rows)), 3),
        "fail_reasons": {
            row["task_id"]: row["score"]["reasons"]
            for row in rows
            if row["score"]["reasons"]
        },
    }


def task_leaderboards(rows: list[dict[str, Any]], top_n: int = 3) -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row)

    leaderboards: dict[str, list[dict[str, Any]]] = {}
    for task_id, task_rows in by_task.items():
        ranked = sorted(
            task_rows,
            key=lambda row: (
                row["score"]["points"] / max(1, row["score"]["max_points"]),
                row["score"]["passed"],
                -row["elapsed_sec"],
            ),
            reverse=True,
        )
        leaderboards[task_id] = [
            {
                "rank": i + 1,
                "model": row["model"],
                "points": row["score"]["points"],
                "max_points": row["score"]["max_points"],
                "passed": row["score"]["passed"],
                "elapsed_sec": row["elapsed_sec"],
                "reasons": row["score"]["reasons"],
            }
            for i, row in enumerate(ranked[:top_n])
        ]
    return leaderboards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-cli", default="~/opt/llama.cpp/llama-cli")
    parser.add_argument("--out-dir", default="output_data/model_bench/quality")
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--temp", type=float, default=0.15)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--tasks", nargs="*", help="Optional task ids to run")
    parser.add_argument("--include-stress", action="store_true", help="Also run oversized stress-test candidates.")
    parser.add_argument("--top-n", type=int, default=3)
    args = parser.parse_args()

    llama_cli = Path(args.llama_cli).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "quality_results.jsonl"
    summary_path = out_dir / "quality_summary.json"

    selected_tasks = TASKS
    if args.tasks:
        requested = set(args.tasks)
        selected_tasks = [task for task in TASKS if task.task_id in requested]
        missing = requested - {task.task_id for task in selected_tasks}
        if missing:
            raise SystemExit(f"Unknown task id(s): {', '.join(sorted(missing))}")

    models = list(args.models)
    if args.include_stress:
        models.extend(STRESS_MODELS)

    summaries = []
    all_rows = []
    with results_path.open("w", encoding="utf-8") as out:
        for model in models:
            rows = []
            print(f"MODEL {model}", flush=True)
            for task in selected_tasks:
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
                rows.append(row)
                all_rows.append(row)
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
            summaries.append(summarize(model, rows))

    summaries.sort(key=lambda item: (item["score_pct"], item["passed_tasks"], -item["avg_elapsed_sec"]), reverse=True)
    summary_doc = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results_path": str(results_path),
        "summaries": summaries,
        "task_leaderboards": task_leaderboards(all_rows, top_n=args.top_n),
    }
    summary_path.write_text(json.dumps(summary_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary_doc, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
