#!/usr/bin/env python3
"""
Smoke-test llama.cpp GGUF models with one tiny prompt.

This verifies that a model can download, load, answer, and exit. It is not a
quality benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


DEFAULT_CANDIDATES = [
    "unsloth/Qwen3.5-0.8B-GGUF:Qwen3.5-0.8B-UD-IQ2_XXS.gguf",
    "LiquidAI/LFM2.5-350M-GGUF:LFM2.5-350M-Q4_K_M.gguf",
    "LiquidAI/LFM2.5-1.2B-Thinking-GGUF:LFM2.5-1.2B-Thinking-Q4_K_M.gguf",
    "unsloth/Qwen3.5-2B-GGUF:Qwen3.5-2B-UD-IQ2_XXS.gguf",
    "unsloth/Qwen3.5-4B-GGUF:Qwen3.5-4B-UD-IQ2_XXS.gguf",
]


def run_smoke(llama_cli: Path, model: str, timeout: int, prompt: str, ctx: int) -> dict:
    if ":" in model:
        repo, hf_file = model.split(":", 1)
        model_args = ["--hf-repo", repo, "--hf-file", hf_file]
    else:
        model_args = ["-hf", model]
    cmd = [
        str(llama_cli),
        *model_args,
        "-p",
        prompt,
        "-c",
        str(ctx),
        "-n",
        "64",
        "-t",
        "4",
        "--temp",
        "0.2",
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
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, env=env)
        elapsed = time.time() - started
        output = proc.stdout.strip()
        stderr = proc.stderr.strip()
        lower_output = output.lower()
        lower_stderr = stderr.lower()
        ok = (
            proc.returncode == 0
            and bool(output)
            and "exceeds the available context size" not in lower_output
            and "exceeds the available context size" not in lower_stderr
            and "error:" not in lower_output[-1000:]
        )
        return {
            "model": model,
            "ok": ok,
            "returncode": proc.returncode,
            "elapsed_sec": round(elapsed, 3),
            "output": output[-4000:],
            "stderr_tail": stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "model": model,
            "ok": False,
            "returncode": "timeout",
            "elapsed_sec": timeout,
            "output": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-cli", default="~/opt/llama.cpp/llama-cli")
    parser.add_argument("--out", default="output_data/model_bench/smoke_results.jsonl")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument(
        "--prompt",
        default="Answer in one concise sentence: what is this system being migrated to do?",
    )
    parser.add_argument("models", nargs="*")
    args = parser.parse_args()

    llama_cli = Path(args.llama_cli).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    models = args.models or DEFAULT_CANDIDATES

    with out.open("a", encoding="utf-8") as f:
        for model in models:
            print(f"SMOKE {model}", flush=True)
            result = run_smoke(llama_cli, model, args.timeout, args.prompt, args.ctx)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps({k: result[k] for k in ("model", "ok", "returncode", "elapsed_sec")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
