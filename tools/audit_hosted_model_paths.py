#!/usr/bin/env python3
"""Audit remaining hosted-model paths and map them to local migration actions."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

SCAN_FILES = [
    "agents/chat_agent.py",
    "agents/llm_client.py",
    "deploy/agentcore/chat/main.py",
    "deploy/agentcore/generator_agent/main.py",
    "deploy/agentcore/qe_agent/main.py",
    "deploy/agentcore/evolution_agent/main.py",
    "deploy/agentcore/debate_agent/main.py",
    "deploy/agentcore/ranking/main.py",
    "deploy/agentcore/ranking_orchestrator/main.py",
    "deploy/agentcore/deploy.sh",
    "README.md",
    "DEPLOY_HETZNER.md",
]


ROLE_DECISIONS = {
    "agents/chat_agent.py": {
        "workflow": "main_flask_webchat",
        "status": "local_backend_available",
        "decision": "Use CHAT_BACKEND=llama_cpp with LFM2-2.6B, BGE-M3 retrieval, BGE reranking, and top-3 context compression.",
        "replacement": "LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf",
    },
    "agents/llm_client.py": {
        "workflow": "shared_llm_client",
        "status": "local_backend_available",
        "decision": "Uses local llama.cpp by default with LFM2-2.6B for prose and Phi-4-mini for JSON/control-plane; hosted Bedrock remains only as an explicit backend.",
        "replacement": "Phi-4-mini for JSON/control-plane; LFM2-2.6B for prose/RAG.",
    },
    "deploy/agentcore/chat/main.py": {
        "workflow": "agentcore_chat",
        "status": "legacy_runtime_guarded",
        "decision": "Runtime now returns disabled by default unless ALLOW_LEGACY_AGENTCORE_RUNTIME=1 is set; prefer Hetzner Flask ChatAgent local backend.",
        "replacement": "Hetzner ChatAgent local backend.",
    },
    "deploy/agentcore/generator_agent/main.py": {
        "workflow": "legacy_linkedin_generator",
        "status": "legacy_runtime_guarded",
        "decision": "Runtime now returns disabled by default unless ALLOW_LEGACY_AGENTCORE_RUNTIME=1 is set; old influencer-post generator conflicts with the Brandon news-summary goal.",
        "replacement": "LFM2-2.6B news-summary generator if this workflow is kept.",
    },
    "deploy/agentcore/evolution_agent/main.py": {
        "workflow": "legacy_linkedin_evolution",
        "status": "legacy_runtime_guarded",
        "decision": "Runtime now returns disabled by default unless ALLOW_LEGACY_AGENTCORE_RUNTIME=1 is set; old viral-post evolution conflicts with the Brandon digest goal.",
        "replacement": "LFM2-2.6B with citation verifier, or no model if deterministic QA is enough.",
    },
    "deploy/agentcore/qe_agent/main.py": {
        "workflow": "quality_evaluator",
        "status": "legacy_runtime_guarded",
        "decision": "Runtime now returns disabled by default unless ALLOW_LEGACY_AGENTCORE_RUNTIME=1 is set; replace with deterministic metrics plus Phi-4-mini structured QA when subjective review is needed.",
        "replacement": "Phi-4-mini constrained JSON.",
    },
    "deploy/agentcore/debate_agent/main.py": {
        "workflow": "tournament_judge",
        "status": "legacy_runtime_guarded",
        "decision": "Runtime now returns disabled by default unless ALLOW_LEGACY_AGENTCORE_RUNTIME=1 is set; replace with reranker/ELO metrics and Phi-4-mini only if still useful.",
        "replacement": "BGE reranker/ZeroEntropy-style reranker plus Phi-4-mini JSON judge.",
    },
    "deploy/agentcore/ranking/main.py": {
        "workflow": "legacy_all_in_one_agentcore_tournament",
        "status": "legacy_runtime_guarded",
        "decision": "Runtime now returns disabled by default unless ALLOW_LEGACY_AGENTCORE_RUNTIME=1 is set; old prompts are not aligned with Brandon news summaries.",
        "replacement": "Local news-summary workflow on Hetzner.",
    },
    "deploy/agentcore/ranking_orchestrator/main.py": {
        "workflow": "agentcore_orchestrator",
        "status": "legacy_runtime_guarded",
        "decision": "Runtime now returns disabled by default unless ALLOW_LEGACY_AGENTCORE_RUNTIME=1 is set; if kept, orchestrate local Hetzner endpoints rather than Bedrock leaf agents.",
        "replacement": "HTTP calls to local Hetzner services.",
    },
    "deploy/agentcore/deploy.sh": {
        "workflow": "agentcore_deployment",
        "status": "hosted_deployment_guarded",
        "decision": "Legacy hosted deployment is guarded and refuses normal deploys unless ALLOW_HOSTED_AGENTCORE_DEPLOY=1 is set; destroy remains available.",
        "replacement": "Hetzner-local service deployment.",
    },
    "README.md": {
        "workflow": "docs",
        "status": "local_docs_updated",
        "decision": "README now documents local llama.cpp defaults for text generation; residual Anthropic mentions are source/domain examples, not runtime instructions.",
        "replacement": "Document local llama.cpp defaults and current model constellation.",
    },
    "DEPLOY_HETZNER.md": {
        "workflow": "docs",
        "status": "local_docs_updated",
        "decision": "Hetzner deploy docs now configure local llama.cpp and smoke tests; residual Claude text is explicitly negative or historical branch naming.",
        "replacement": "Document local llama.cpp/HF cache and disabled hosted agents.",
    },
}


PATTERNS = {
    "bedrock": re.compile(r"\bBedrockModel\b|bedrock:InvokeModel|bedrock-agentcore|boto3\.client\(\"bedrock", re.I),
    "strands": re.compile(r"\bfrom strands\b|\bstrands-agents\b|\bAgent\(model=", re.I),
    "claude_anthropic": re.compile(r"anthropic|claude", re.I),
    "gemini_google_ai": re.compile(r"gemini|google-genai|pydantic-ai\[google\]", re.I),
}


def scan_file(path: Path) -> dict[str, Any]:
    rel = str(path.relative_to(ROOT))
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    hits = {}
    for name, pattern in PATTERNS.items():
        lines = []
        for idx, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                lines.append({"line": idx, "text": line.strip()[:220]})
        if lines:
            hits[name] = lines
    return {
        "path": rel,
        "exists": path.exists(),
        "hits": hits,
        **ROLE_DECISIONS.get(rel, {}),
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Hosted Model Migration Audit",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary["counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Paths", ""])
    for row in summary["files"]:
        if not row.get("hits"):
            continue
        lines.extend(
            [
                f"### {row['path']}",
                "",
                f"- Workflow: `{row.get('workflow', 'unknown')}`",
                f"- Status: `{row.get('status', 'needs_review')}`",
                f"- Decision: {row.get('decision', 'Review manually.')}",
                f"- Replacement: `{row.get('replacement', 'n/a')}`",
                f"- Hit types: `{', '.join(row['hits'])}`",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="output_data/model_bench/hosted_model_migration_audit")
    args = parser.parse_args()

    files = [scan_file(ROOT / rel) for rel in SCAN_FILES]
    status_counts = Counter(row.get("status", "needs_review") for row in files if row.get("hits"))
    hit_counts = Counter()
    for row in files:
        for name, lines in row.get("hits", {}).items():
            hit_counts[name] += len(lines)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "files_scanned": len(files),
            "files_with_hosted_hits": sum(1 for row in files if row.get("hits")),
            "statuses": dict(status_counts),
            "hit_types": dict(hit_counts),
        },
        "files": files,
        "migration_decision": (
            "Main Flask/webchat, shared llm_client, and migration docs now point to local llama.cpp. "
            "AgentCore leaf agents still contain hosted Claude/Bedrock code for historical reference, but the "
            "runtimes now return disabled responses by default unless ALLOW_LEGACY_AGENTCORE_RUNTIME=1 is set; "
            "the hosted deploy script also refuses normal deploys unless ALLOW_HOSTED_AGENTCORE_DEPLOY=1 is set."
        ),
    }

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "hosted_model_migration_audit.json"
    md_path = out_dir / "hosted_model_migration_audit.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summary, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "counts": summary["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
