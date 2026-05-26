#!/usr/bin/env python3
"""
Extract redacted Claude Code logs into eval and SFT datasets.

This is intentionally dependency-free so it can run directly on the Hetzner VM:

    python3 tools/claude_distill.py \
      --claude-dir ~/.claude \
      --out-dir output_data/claude_distill
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"), "[REDACTED_ANTHROPIC_KEY]"),
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "[REDACTED_HF_TOKEN]"),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9\-]{20,}"), "[REDACTED_SLACK_TOKEN]"),
    (re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*['\"]?[^'\"\s]{8,}"), r"\1=[REDACTED_SECRET]"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[REDACTED_PRIVATE_KEY]"),
]


@dataclass
class Message:
    role: str
    content: str
    timestamp: str | None = None


@dataclass
class Conversation:
    source: str
    project: str
    session_id: str
    messages: list[Message] = field(default_factory=list)


def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def redact(text: str) -> str:
    for pattern, repl in SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif "content" in item:
                    parts.append(normalize_content(item["content"]))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if "content" in content:
            return normalize_content(content["content"])
    return ""


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def load_conversation(path: Path, claude_dir: Path) -> Conversation | None:
    rel = path.relative_to(claude_dir)
    project = rel.parts[1] if len(rel.parts) > 2 and rel.parts[0] == "projects" else rel.parts[0]
    conv = Conversation(source=str(rel), project=project, session_id=path.stem)

    for obj in iter_jsonl(path):
        msg = obj.get("message")
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = redact(normalize_content(msg.get("content", ""))).strip()
        if not content:
            continue
        if len(content) > 24000:
            content = content[:24000] + "\n[TRUNCATED]"
        conv.messages.append(Message(role=role, content=content, timestamp=obj.get("timestamp")))

    if len(conv.messages) < 2:
        return None
    return conv


def paired_examples(conv: Conversation) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pending_user: Message | None = None
    for msg in conv.messages:
        if msg.role == "user":
            pending_user = msg
        elif msg.role == "assistant" and pending_user:
            instruction = pending_user.content.strip()
            output = msg.content.strip()
            if len(instruction) >= 20 and len(output) >= 20:
                rows.append(
                    {
                        "id": stable_id(conv.source + instruction + output),
                        "source": conv.source,
                        "project": conv.project,
                        "session_id": conv.session_id,
                        "instruction": instruction,
                        "output": output,
                    }
                )
            pending_user = None
    return rows


def eval_prompts(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        prompt = row["instruction"]
        expected = row["output"]
        if len(prompt) < 40 or len(expected) < 80:
            continue
        if any(bad in prompt.lower() for bad in ("password", "token", "secret", "api key")):
            continue
        candidates.append(
            {
                "id": row["id"],
                "source": row["source"],
                "project": row["project"],
                "prompt": prompt[:6000],
                "reference": expected[:6000],
                "rubric": [
                    "Follows the user's concrete request",
                    "Preserves relevant technical constraints",
                    "Avoids invented facts",
                    "Gives actionable next steps",
                    "Uses concise, direct engineering prose",
                ],
            }
        )
    random.Random(17).shuffle(candidates)
    return candidates[:limit]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claude-dir", default="~/.claude")
    parser.add_argument("--out-dir", default="output_data/claude_distill")
    parser.add_argument("--eval-limit", type=int, default=200)
    args = parser.parse_args()

    claude_dir = Path(args.claude_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    conversations = []
    for path in claude_dir.glob("projects/**/*.jsonl"):
        conv = load_conversation(path, claude_dir)
        if conv:
            conversations.append(conv)

    rows: list[dict[str, Any]] = []
    for conv in conversations:
        rows.extend(paired_examples(conv))

    random.Random(17).shuffle(rows)
    split = max(1, int(len(rows) * 0.9)) if rows else 0
    train = rows[:split]
    validation = rows[split:]
    evals = eval_prompts(validation or rows, args.eval_limit)

    counts = {
        "conversations": len(conversations),
        "sft_train": write_jsonl(out_dir / "sft_train.jsonl", train),
        "sft_validation": write_jsonl(out_dir / "sft_validation.jsonl", validation),
        "bench_prompts": write_jsonl(out_dir / "bench_prompts.jsonl", evals),
    }
    (out_dir / "manifest.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
