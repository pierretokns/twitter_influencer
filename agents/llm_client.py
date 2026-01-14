# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai>=1.0.0",
#     "python-dotenv>=0.19.0",
# ]
# ///
"""
OpenRouter LLM Client
=====================

Centralized LLM client using OpenRouter API with OpenAI SDK.
Replaces subprocess calls to Claude CLI.

ENVIRONMENT VARIABLES:
    OPENROUTER_API_KEY: Required - OpenRouter API key
    OPENROUTER_MODEL: Optional - Default model (default: anthropic/claude-sonnet-4)

USAGE:
    from agents.llm_client import call_llm, call_llm_json

    # Simple text response
    response = call_llm("Evaluate this post...", timeout=60)

    # Auto-extract JSON from response
    data = call_llm_json("Return JSON: {score: 85}", timeout=45)
"""

import json
import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")
DEFAULT_TIMEOUT = 60

# Lazy-loaded client singleton
_client: Optional[OpenAI] = None


class LLMError(Exception):
    """Base class for LLM errors"""
    pass


class LLMTimeoutError(LLMError):
    """Raised when API call times out"""
    pass


class LLMRateLimitError(LLMError):
    """Raised on 429 rate limit"""
    pass


class LLMConnectionError(LLMError):
    """Raised on connection failures"""
    pass


def get_client() -> OpenAI:
    """
    Get or create the OpenRouter client (singleton).

    Returns:
        Configured OpenAI client for OpenRouter
    """
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMError("OPENROUTER_API_KEY environment variable not set")

        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/twitter_influencer",
                "X-Title": "Twitter Influencer Bot",
            }
        )
    return _client


def call_llm(
    prompt: str,
    timeout: int = DEFAULT_TIMEOUT,
    model: str = None,
    system_prompt: str = None,
) -> Optional[str]:
    """
    Make a simple LLM call and return the response text.

    Args:
        prompt: The user prompt to send
        timeout: Request timeout in seconds (default: 60)
        model: Model to use (default: OPENROUTER_MODEL env or anthropic/claude-sonnet-4)
        system_prompt: Optional system prompt

    Returns:
        Response text string, or None if call failed

    Raises:
        LLMTimeoutError: If the request times out
        LLMRateLimitError: If rate limited (429)
        LLMConnectionError: If connection fails
    """
    client = get_client()
    model = model or DEFAULT_MODEL

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=timeout,
        )

        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content

        return None

    except APITimeoutError as e:
        print(f"[LLMClient] Timeout after {timeout}s: {e}")
        raise LLMTimeoutError(f"Request timed out after {timeout}s") from e

    except RateLimitError as e:
        print(f"[LLMClient] Rate limited: {e}")
        raise LLMRateLimitError("Rate limited by OpenRouter") from e

    except APIConnectionError as e:
        print(f"[LLMClient] Connection error: {e}")
        raise LLMConnectionError(f"Failed to connect: {e}") from e

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
    Make an LLM call and extract JSON from the response.

    Uses the same JSON extraction pattern as existing agents:
    finds first '{' and last '}' to handle markdown/text around JSON.

    Args:
        prompt: The user prompt to send
        timeout: Request timeout in seconds (default: 60)
        model: Model to use (default: OPENROUTER_MODEL env or anthropic/claude-sonnet-4)
        system_prompt: Optional system prompt

    Returns:
        Parsed JSON dict, or None if call failed or no valid JSON

    Raises:
        LLMTimeoutError: If the request times out
        LLMRateLimitError: If rate limited (429)
        LLMConnectionError: If connection fails
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


def get_usage_info(response) -> dict:
    """
    Extract token usage info from a response.

    Args:
        response: OpenAI API response object

    Returns:
        Dict with prompt_tokens, completion_tokens, total_tokens
    """
    if hasattr(response, 'usage') and response.usage:
        return {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return {}


# For backwards compatibility - agents can catch these or let them propagate
__all__ = [
    'call_llm',
    'call_llm_json',
    'extract_json',
    'get_client',
    'LLMError',
    'LLMTimeoutError',
    'LLMRateLimitError',
    'LLMConnectionError',
    'DEFAULT_MODEL',
]


if __name__ == "__main__":
    # Quick test
    print(f"Testing LLM client with model: {DEFAULT_MODEL}")
    try:
        result = call_llm("Say 'Hello from OpenRouter!' in exactly those words.", timeout=30)
        print(f"Response: {result}")

        json_result = call_llm_json('Respond with exactly: {"status": "ok", "number": 42}', timeout=30)
        print(f"JSON Response: {json_result}")
    except LLMError as e:
        print(f"Error: {e}")
