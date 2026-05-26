#!/usr/bin/env python3
"""
Run local llama.cpp models against distilled Claude benchmark prompts.

Example on the VM:

    python3 tools/local_model_bench.py \
      --llama-cli ~/opt/llama.cpp/llama-cli \
      --model Qwen/Qwen3-1.7B-GGUF:Q8_0 \
      --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path


SYSTEM_PROMPT = """/no_think
You are a pragmatic engineering assistant. Answer the user's request directly.
Preserve concrete constraints, avoid invented facts, and prefer actionable implementation detail.
If the request is ambiguous, make the least risky assumption and state it briefly.
Do not include hidden reasoning, chain-of-thought, or thinking traces. Return the final answer only."""


def load_prompts(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(rows) >= limit:
                break
            rows.append(json.loads(line))
    return rows


def score_heuristics(output: str, reference: str) -> dict:
    out_words = set(output.lower().split())
    ref_words = set(reference.lower().split())
    overlap = len(out_words & ref_words) / max(1, len(ref_words))
    return {
        "output_chars": len(output),
        "reference_chars": len(reference),
        "word_overlap_vs_reference": round(overlap, 4),
        "empty": not bool(output.strip()),
        "mentions_uncertainty": any(x in output.lower() for x in ["i don't know", "not enough", "cannot", "can't"]),
    }


def run_one(llama_cli: Path, model: str, prompt: str, ctx: int, predict: int, threads: int, temp: float) -> tuple[str, float, int]:
    full_prompt = f"{SYSTEM_PROMPT}\n\nUSER REQUEST:\n{prompt}\n\nASSISTANT RESPONSE:\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tmp:
        tmp.write(full_prompt)
        tmp_path = Path(tmp.name)
    if ":" in model and model.endswith(".gguf"):
        repo, hf_file = model.split(":", 1)
        model_args = ["--hf-repo", repo, "--hf-file", hf_file]
    else:
        model_args = ["-hf", model]
    cmd = [
        str(llama_cli),
        *model_args,
        "-f",
        str(tmp_path),
        "-c",
        str(ctx),
        "-n",
        str(predict),
        "-t",
        str(threads),
        "--temp",
        str(temp),
        "--no-display-prompt",
        "--reasoning",
        "off",
        "--single-turn",
    ]
    started = time.time()
    env = os.environ.copy()
    token_path = Path("~/.cache/huggingface/token").expanduser()
    if token_path.exists() and "HF_TOKEN" not in env:
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            env["HF_TOKEN"] = token
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=1800, env=env)
        elapsed = time.time() - started
        output = proc.stdout.strip()
        if proc.returncode != 0:
            output = (output + "\n" + proc.stderr.strip()).strip()
        return output, elapsed, proc.returncode
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-cli", default="~/opt/llama.cpp/llama-cli")
    parser.add_argument("--bench", default="output_data/claude_distill/bench_prompts.jsonl")
    parser.add_argument("--out-dir", default="output_data/model_bench")
    parser.add_argument("--model", required=True, help="llama.cpp -hf model spec, e.g. Qwen/Qwen3-1.7B-GGUF:Q8_0")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--predict", type=int, default=768)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--temp", type=float, default=0.2)
    parser.add_argument("--max-prompt-chars", type=int, default=2500)
    args = parser.parse_args()

    llama_cli = Path(args.llama_cli).expanduser().resolve()
    bench = Path(args.bench).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_model = args.model.replace("/", "__").replace(":", "__")
    out_path = out_dir / f"{safe_model}.jsonl"
    summary_path = out_dir / f"{safe_model}.summary.json"

    prompts = load_prompts(bench, args.limit)
    results = []
    with out_path.open("w", encoding="utf-8") as out:
        for i, row in enumerate(prompts, 1):
            print(f"[{i}/{len(prompts)}] {row['id']} {args.model}", flush=True)
            prompt = row["prompt"][: args.max_prompt_chars]
            output, elapsed, rc = run_one(
                llama_cli,
                args.model,
                prompt,
                args.ctx,
                args.predict,
                args.threads,
                args.temp,
            )
            result = {
                "id": row["id"],
                "model": args.model,
                "source": row["source"],
                "project": row["project"],
                "elapsed_sec": round(elapsed, 3),
                "returncode": rc,
                "prompt_chars": len(prompt),
                "output": output,
                "heuristics": score_heuristics(output, row.get("reference", "")),
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            results.append(result)

    ok = [r for r in results if r["returncode"] == 0 and not r["heuristics"]["empty"]]
    summary = {
        "model": args.model,
        "attempted": len(results),
        "completed": len(ok),
        "avg_elapsed_sec": round(sum(r["elapsed_sec"] for r in ok) / max(1, len(ok)), 3),
        "avg_output_chars": round(sum(r["heuristics"]["output_chars"] for r in ok) / max(1, len(ok)), 1),
        "avg_word_overlap_vs_reference": round(
            sum(r["heuristics"]["word_overlap_vs_reference"] for r in ok) / max(1, len(ok)), 4
        ),
        "results_path": str(out_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
