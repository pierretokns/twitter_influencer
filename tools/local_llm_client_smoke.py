#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pydantic>=2.0.0",
#     "python-dotenv>=0.19.0",
# ]
# ///
"""Smoke-test the shared local llm_client path and write a reusable artifact."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "output_data" / "model_bench" / "llm_client_local_smoke"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.llm_client import call_llm_json  # noqa: E402


class RouteDecision(BaseModel):
    route: Literal["answer_with_sources", "refuse_insufficient_sources"]
    confidence: float


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR.relative_to(ROOT)))
    parser.add_argument(
        "--prompt",
        default=(
            "Return JSON only. A user asks for a claim, but provided sources do not contain support. "
            "Choose route refuse_insufficient_sources with confidence 0.9."
        ),
    )
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "llm_client_local_smoke.json"

    started = time.monotonic()
    tests = []
    passed = False
    error = None
    try:
        result = call_llm_json(args.prompt, timeout=args.timeout, response_model=RouteDecision)
        passed = (
            result.get("route") == "refuse_insufficient_sources"
            and float(result.get("confidence", 0.0)) >= 0.5
        )
        tests.append({"name": "json_route_decision", "passed": passed, "result": result})
    except Exception as exc:  # noqa: BLE001 - smoke artifact should capture operational failures.
        error = f"{type(exc).__name__}: {exc}"
        tests.append({"name": "json_route_decision", "passed": False, "error": error})

    artifact = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "passed": passed,
        "backend": os.getenv("LLM_BACKEND", os.getenv("CHAT_BACKEND", "llama_cpp")),
        "prose_model": os.getenv(
            "LLM_LLAMA_MODEL",
            os.getenv("CHAT_LLAMA_MODEL", "LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf"),
        ),
        "json_model": os.getenv(
            "LLM_LLAMA_JSON_MODEL",
            "unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf",
        ),
        "elapsed_sec": round(time.monotonic() - started, 3),
        "tests": tests,
        "error": error,
    }
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(out_path), "passed": passed, "elapsed_sec": artifact["elapsed_sec"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
