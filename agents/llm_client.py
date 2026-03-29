# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "python-dotenv>=0.19.0",
# ]
# ///
"""
Claude CLI LLM Client
=====================

Uses Claude Code CLI for LLM calls with subprocess.
OTEL traces are sent automatically when configured via environment variables.

ENVIRONMENT VARIABLES:
    ANTHROPIC_API_KEY: Required for Claude CLI authentication

    # OTEL Configuration (optional - enables tracing)
    CLAUDE_CODE_ENABLE_TELEMETRY: Set to "1" to enable OTEL
    OTEL_EXPORTER_OTLP_PROTOCOL: "grpc" or "http"
    OTEL_EXPORTER_OTLP_ENDPOINT: e.g. "http://127.0.0.1:4317"

USAGE:
    from agents.llm_client import call_llm, call_llm_json

    # Simple text response
    response = call_llm("Evaluate this post...", timeout=60)

    # Auto-extract JSON from response
    data = call_llm_json("Return JSON: {score: 85}", timeout=45)
"""

import json
import subprocess
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

DEFAULT_TIMEOUT = 60


class LLMError(Exception):
    """Base class for LLM errors"""
    pass


class LLMTimeoutError(LLMError):
    """Raised when CLI call times out"""
    pass


class LLMRateLimitError(LLMError):
    """Raised on rate limit (for backwards compatibility)"""
    pass


class LLMConnectionError(LLMError):
    """Raised on connection failures (for backwards compatibility)"""
    pass


def call_llm(
    prompt: str,
    timeout: int = DEFAULT_TIMEOUT,
    model: str = None,  # Ignored - Claude CLI uses configured model
    system_prompt: str = None,
) -> Optional[str]:
    """
    Make a Claude CLI call and return the response.

    Args:
        prompt: The prompt to send
        timeout: Timeout in seconds (default: 60)
        model: Ignored - kept for backwards compatibility
        system_prompt: Optional system prompt (prepended to prompt)

    Returns:
        Response text string, or None if call failed

    Raises:
        LLMTimeoutError: If the CLI call times out
        LLMError: If Claude CLI is not installed
    """
    full_prompt = prompt
    if system_prompt:
        full_prompt = f"{system_prompt}\n\n{prompt}"

    try:
        result = subprocess.run(
            ['claude', '-p', full_prompt, '--output-format', 'text'],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            return result.stdout.strip()

        # Log error but don't raise for non-zero exit
        print(f"[LLMClient] CLI error (exit {result.returncode}): {result.stderr.strip()}")
        return None

    except subprocess.TimeoutExpired:
        print(f"[LLMClient] Timeout after {timeout}s")
        raise LLMTimeoutError(f"CLI timed out after {timeout}s")

    except FileNotFoundError:
        raise LLMError(
            "Claude CLI not found. Install with: sudo npm install -g @anthropic-ai/claude-code"
        )

    except Exception as e:
        print(f"[LLMClient] Unexpected error: {e}")
        return None


def call_llm_json(
    prompt: str,
    timeout: int = DEFAULT_TIMEOUT,
    model: str = None,
    system_prompt: str = None,
) -> Optional[dict]:
    """
    Make a Claude CLI call and extract JSON from the response.

    Uses the same JSON extraction pattern as existing agents:
    finds first '{' and last '}' to handle markdown/text around JSON.

    Args:
        prompt: The prompt to send
        timeout: Timeout in seconds (default: 60)
        model: Ignored - kept for backwards compatibility
        system_prompt: Optional system prompt

    Returns:
        Parsed JSON dict, or None if call failed or no valid JSON

    Raises:
        LLMTimeoutError: If the CLI call times out
        LLMError: If Claude CLI is not installed
    """
    response = call_llm(prompt, timeout=timeout, model=model, system_prompt=system_prompt)

    if not response:
        return None

    return extract_json(response)


def extract_json(text: str) -> Optional[dict]:
    """
    Extract JSON object from text that may contain markdown or prose.

    Uses the same pattern as existing agents: finds first '{' and last '}'.

    Args:
        text: Response text potentially containing JSON

    Returns:
        Parsed dict, or None if no valid JSON found
    """
    if not text:
        return None

    try:
        start = text.find('{')
        end = text.rfind('}') + 1

        if start >= 0 and end > start:
            json_str = text[start:end]
            return json.loads(json_str)

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
]


if __name__ == "__main__":
    # Quick test
    print("Testing Claude CLI client...")
    try:
        result = call_llm("Say 'Hello from Claude CLI!' in exactly those words.", timeout=30)
        print(f"Response: {result}")

        json_result = call_llm_json('Respond with exactly: {"status": "ok", "number": 42}', timeout=30)
        print(f"JSON Response: {json_result}")
    except LLMError as e:
        print(f"Error: {e}")
