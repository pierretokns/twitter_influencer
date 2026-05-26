#!/usr/bin/env python3
"""Focused tests for agents.llm_client local backend."""

from __future__ import annotations

import os
from unittest.mock import patch

from pydantic import BaseModel


def test_local_text_backend() -> None:
    os.environ["LLM_BACKEND"] = "llama_cpp"
    os.environ["LLM_LLAMA_MODEL"] = "fake/repo:model.gguf"
    os.environ["LLM_LLAMA_CLI"] = "/tmp/fake-llama-cli"

    import agents.llm_client as llm_client

    class Proc:
        returncode = 0
        stdout = "ANSWER: local response\nExiting..."
        stderr = ""

    with patch.object(llm_client.subprocess, "run", return_value=Proc()) as run:
        result = llm_client.call_llm("Say hello", timeout=5)

    assert result == "local response"
    cmd = run.call_args.args[0]
    assert "--hf-repo" in cmd
    assert "fake/repo" in cmd
    assert "--single-turn" in cmd


def test_local_json_backend_with_pydantic_validation() -> None:
    os.environ["LLM_BACKEND"] = "llama_cpp"
    os.environ["LLM_LLAMA_CLI"] = "/tmp/fake-llama-cli"

    import agents.llm_client as llm_client

    class Score(BaseModel):
        score: int
        passed: bool

    class Proc:
        returncode = 0
        stdout = 'JSON: {"score": 91, "passed": true}'
        stderr = ""

    with (
        patch.object(llm_client, "JSON_MODEL", "fake/json:model.gguf"),
        patch.object(llm_client.subprocess, "run", return_value=Proc()) as run,
    ):
        result = llm_client.call_llm_json("Score this", timeout=5, response_model=Score)

    assert result == {"score": 91, "passed": True}
    cmd = run.call_args.args[0]
    assert "--json-schema-file" in cmd
    assert "fake/json" in cmd


def main() -> int:
    test_local_text_backend()
    print("local text backend: PASS")
    test_local_json_backend_with_pydantic_validation()
    print("local json backend: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
