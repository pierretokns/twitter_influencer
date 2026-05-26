# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "python-dotenv>=0.19.0",
# ]
# ///
"""
Shared LLM Client
=================

Uses local llama.cpp by default, with explicit Bedrock fallback when requested.

ENVIRONMENT VARIABLES:
    LLM_BACKEND: llama_cpp/local or bedrock/strands/hosted (default: CHAT_BACKEND or llama_cpp)
    LLM_LLAMA_CLI: path to llama-cli (default: CHAT_LLAMA_CLI or ~/opt/llama.cpp/llama-cli)
    LLM_LLAMA_MODEL: prose model (default: CHAT_LLAMA_MODEL or LFM2-2.6B Q4_K_M)
    LLM_LLAMA_JSON_MODEL: structured model (default: Phi-4-mini Q3_K_M)
    LLM_LLAMA_CTX: context size (default: CHAT_LLAMA_CTX or 4096)
    LLM_LLAMA_THREADS: CPU threads (default: CHAT_LLAMA_THREADS or 4)
    LLM_LLAMA_TEMP: temperature (default: CHAT_LLAMA_TEMP or 0.1)
    AWS_PROFILE: AWS profile to use (default: uses default profile)
    AWS_DEFAULT_REGION: AWS region (default: us-east-1)
    STRANDS_MODEL_ID: Hosted model if LLM_BACKEND=bedrock
    STRANDS_FAST_MODEL_ID: Hosted fast model if LLM_BACKEND=bedrock

USAGE:
    from agents.llm_client import call_llm, call_llm_json

    response = call_llm("Evaluate this post...", timeout=60)
    data = call_llm_json("Return JSON: {score: 85}", timeout=45)
"""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Type

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

DEFAULT_TIMEOUT = 60
DEFAULT_BACKEND = os.getenv("LLM_BACKEND", os.getenv("CHAT_BACKEND", "llama_cpp")).strip().lower()
DEFAULT_MODEL = os.getenv(
    "LLM_LLAMA_MODEL",
    os.getenv("CHAT_LLAMA_MODEL", "LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf"),
)
FAST_MODEL = os.getenv("LLM_LLAMA_FAST_MODEL", DEFAULT_MODEL)
JSON_MODEL = os.getenv(
    "LLM_LLAMA_JSON_MODEL",
    "unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf",
)
HOSTED_MODEL = os.getenv("STRANDS_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
HOSTED_FAST_MODEL = os.getenv("STRANDS_FAST_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


class LLMError(Exception):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMRateLimitError(LLMError):
    pass


class LLMConnectionError(LLMError):
    pass


def _make_agent(model_id: str, system_prompt: Optional[str] = None):
    """Create a Strands Agent with Bedrock backend. Lazy import so startup is fast."""
    from strands import Agent
    from strands.models import BedrockModel

    model = BedrockModel(
        model_id=model_id,
        region_name=AWS_REGION,
        max_tokens=4096,
    )
    kwargs = {"model": model}
    if system_prompt:
        kwargs["system_prompt"] = system_prompt
    return Agent(**kwargs)


def _is_local_backend() -> bool:
    return DEFAULT_BACKEND in {"llama_cpp", "llama.cpp", "local"}


def _llama_model_args(model: str) -> list[str]:
    if ":" in model and model.endswith(".gguf"):
        repo, hf_file = model.split(":", 1)
        return ["--hf-repo", repo, "--hf-file", hf_file]
    return ["-hf", model]


def _clean_llama_output(text: str) -> str:
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text or "")
    clean = re.sub(r".\x08", "", clean)
    if " ... (truncated)\n\n" in clean:
        clean = clean.split(" ... (truncated)\n\n", 1)[-1]
    if "ANSWER:" in clean:
        clean = clean.rsplit("ANSWER:", 1)[-1]
    if "JSON:" in clean:
        clean = clean.rsplit("JSON:", 1)[-1]
    clean = re.sub(r"\n?Exiting\.\.\.\s*$", "", clean)
    clean = re.sub(r"\s*\[\s*Prompt:.*?Generation:.*?\]\s*$", "", clean, flags=re.S)
    return clean.strip()


def _run_llama_cli(
    prompt: str,
    *,
    timeout: int,
    model: str,
    json_schema: Optional[dict] = None,
) -> Optional[str]:
    llama_cli = Path(os.getenv("LLM_LLAMA_CLI", os.getenv("CHAT_LLAMA_CLI", "~/opt/llama.cpp/llama-cli"))).expanduser()
    ctx = int(os.getenv("LLM_LLAMA_CTX", os.getenv("CHAT_LLAMA_CTX", "4096")))
    threads = int(os.getenv("LLM_LLAMA_THREADS", os.getenv("CHAT_LLAMA_THREADS", "4")))
    temp = os.getenv("LLM_LLAMA_TEMP", os.getenv("CHAT_LLAMA_TEMP", "0.1"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", os.getenv("CHAT_MAX_TOKENS", "512")))

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as prompt_file:
        prompt_file.write(prompt)
        prompt_path = Path(prompt_file.name)
    schema_path: Optional[Path] = None
    try:
        cmd = [
            str(llama_cli),
            *_llama_model_args(model),
            "-f",
            str(prompt_path),
            "-c",
            str(ctx),
            "-n",
            str(max_tokens),
            "-t",
            str(threads),
            "--temp",
            temp,
            "--no-display-prompt",
            "--single-turn",
        ]
        if json_schema:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as schema_file:
                json.dump(json_schema, schema_file)
                schema_path = Path(schema_file.name)
            cmd.extend(["--json-schema-file", str(schema_path)])

        proc = subprocess.run(
            cmd,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            print(f"[LLMClient] llama.cpp exited with {proc.returncode}: {proc.stderr.strip()[-1000:]}")
            return None
        return _clean_llama_output(proc.stdout)
    except subprocess.TimeoutExpired:
        raise LLMTimeoutError(f"llama.cpp timed out after {timeout}s")
    finally:
        for path in (prompt_path, schema_path):
            if path:
                try:
                    path.unlink()
                except OSError:
                    pass


def _local_prompt(prompt: str, system_prompt: Optional[str], json_mode: bool) -> str:
    parts = []
    if system_prompt:
        parts.append(system_prompt.strip())
    if json_mode:
        parts.append("Return only one valid JSON object. Do not include markdown or commentary.")
    parts.append(prompt.strip())
    parts.append("JSON:" if json_mode else "ANSWER:")
    return "\n\n".join(parts)


def call_llm(
    prompt: str,
    timeout: int = DEFAULT_TIMEOUT,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    fast: bool = False,
) -> Optional[str]:
    """
    Call the configured LLM backend and return text response.

    Args:
        prompt: The prompt to send
        timeout: Unused (Bedrock handles retries with exponential backoff)
        model: Optional model ID override
        system_prompt: Optional system prompt
        fast: If True, use the cheaper/faster Haiku model

    Returns:
        Response text string, or None if call failed
    """
    if _is_local_backend():
        model_id = model or (FAST_MODEL if fast else DEFAULT_MODEL)
        return _run_llama_cli(
            _local_prompt(prompt, system_prompt, json_mode=False),
            timeout=timeout,
            model=model_id,
        )

    model_id = model or (HOSTED_FAST_MODEL if fast else HOSTED_MODEL)
    try:
        agent = _make_agent(model_id, system_prompt)
        result = agent(prompt)
        return str(result)
    except Exception as e:
        err = str(e)
        if "ThrottlingException" in err or "TooManyRequestsException" in err:
            raise LLMRateLimitError(f"Bedrock throttled: {e}")
        if "ExpiredTokenException" in err or "UnrecognizedClientException" in err:
            raise LLMConnectionError(f"AWS credentials error: {e}")
        print(f"[LLMClient] Error: {e}")
        return None


def call_llm_json(
    prompt: str,
    timeout: int = DEFAULT_TIMEOUT,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    response_model: Optional[Type[BaseModel]] = None,
    fast: bool = False,
) -> Optional[dict]:
    """
    Call the configured LLM backend and extract JSON from the response.

    Args:
        prompt: The prompt to send
        timeout: Unused (kept for backwards compatibility)
        model: Optional model ID override
        system_prompt: Optional system prompt
        response_model: Optional Pydantic model for structured output enforcement
        fast: If True, use the cheaper/faster Haiku model

    Returns:
        Parsed JSON dict, or None if call failed
    """
    if _is_local_backend():
        model_id = model or JSON_MODEL
        json_schema = response_model.model_json_schema() if response_model is not None else None
        response = _run_llama_cli(
            _local_prompt(prompt, system_prompt, json_mode=True),
            timeout=timeout,
            model=model_id,
            json_schema=json_schema,
        )
        data = extract_json(response) if response else None
        if data is not None and response_model is not None:
            try:
                return response_model.model_validate(data).model_dump()
            except Exception as e:
                print(f"[LLMClient] Pydantic validation error: {e}")
                return None
        return data

    model_id = model or (HOSTED_FAST_MODEL if fast else HOSTED_MODEL)
    try:
        if response_model is not None:
            from strands import Agent
            from strands.models import BedrockModel
            model_obj = BedrockModel(model_id=model_id, region_name=AWS_REGION, max_tokens=4096)
            agent = Agent(
                model=model_obj,
                system_prompt=system_prompt,
                structured_output_model=response_model,
            )
            result = agent(prompt)
            # structured_output returns a Pydantic model instance
            if hasattr(result, "model_dump"):
                return result.model_dump()
            return dict(result)

        response = call_llm(prompt, timeout=timeout, model=model_id, system_prompt=system_prompt)
        return extract_json(response) if response else None

    except Exception as e:
        print(f"[LLMClient] JSON call error: {e}")
        return None


def extract_json(text: str) -> Optional[dict]:
    """Extract first JSON object from text that may contain prose or markdown."""
    if not text:
        return None
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError as e:
        print(f"[LLMClient] JSON parse error: {e}")
    return None


__all__ = [
    'call_llm',
    'call_llm_json',
    'extract_json',
    'LLMError',
    'LLMTimeoutError',
    'LLMRateLimitError',
    'LLMConnectionError',
    'DEFAULT_TIMEOUT',
    'DEFAULT_BACKEND',
    'DEFAULT_MODEL',
    'FAST_MODEL',
    'JSON_MODEL',
]


if __name__ == "__main__":
    print(f"Testing LLM client (backend: {DEFAULT_BACKEND}, model: {DEFAULT_MODEL}, region: {AWS_REGION})...")
    try:
        result = call_llm("Say 'Hello from Bedrock!' in exactly those words.")
        print(f"Text response: {result}")

        json_result = call_llm_json('Respond with exactly this JSON: {"status": "ok", "number": 42}')
        print(f"JSON response: {json_result}")
    except LLMError as e:
        print(f"Error: {e}")
