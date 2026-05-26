#!/usr/bin/env python3
"""
Convert distilled instruction/output JSONL into OpenAI chat fine-tuning JSONL.

Input rows are produced by tools/claude_distill.py:
  {"instruction": "...", "output": "...", ...}

Output rows use the common OpenAI chat format:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}], "metadata": {...}}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_MESSAGE = (
    "You are a pragmatic engineering assistant. Answer directly, preserve concrete "
    "constraints, avoid invented facts, and provide actionable implementation detail."
)


def convert_file(src: Path, dst: Path) -> int:
    count = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            instruction = (row.get("instruction") or "").strip()
            output = (row.get("output") or "").strip()
            if not instruction or not output:
                continue
            out = {
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output},
                ],
                "metadata": {
                    "id": row.get("id"),
                    "source": row.get("source"),
                    "project": row.get("project"),
                    "session_id": row.get("session_id"),
                },
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default="output_data/claude_distill")
    parser.add_argument("--out-dir", default="output_data/claude_distill/openai")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "train": convert_file(in_dir / "sft_train.jsonl", out_dir / "train.jsonl"),
        "validation": convert_file(in_dir / "sft_validation.jsonl", out_dir / "validation.jsonl"),
    }
    (out_dir / "README.md").write_text(
        "# Claude Distill OpenAI Chat Format\n\n"
        "Private fine-tuning/evaluation dataset distilled from the Hetzner VM Claude Code logs.\n\n"
        "Files:\n"
        "- `train.jsonl`: OpenAI chat fine-tuning rows.\n"
        "- `validation.jsonl`: OpenAI chat validation rows.\n"
        "- `../bench_prompts.jsonl`: benchmark prompts with references and rubric.\n\n"
        "Rows use `messages` with `system`, `user`, and `assistant` roles plus metadata.\n",
        encoding="utf-8",
    )
    (out_dir / "manifest.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
