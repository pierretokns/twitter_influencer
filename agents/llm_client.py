# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "python-dotenv>=0.19.0",
#     "strands-agents>=0.1.0",
#     "boto3>=1.34.0",
# ]
# ///
"""
Strands + Bedrock LLM Client
=============================

Replaces Claude CLI subprocess calls with Strands Agents SDK using Amazon Bedrock.

ENVIRONMENT VARIABLES:
    AWS_PROFILE: AWS profile to use (default: uses default profile)
    AWS_DEFAULT_REGION: AWS region (default: us-east-1)
    STRANDS_MODEL_ID: Override default model (default: anthropic.claude-sonnet-4-6)
    STRANDS_FAST_MODEL_ID: Faster/cheaper model for evals (default: anthropic.claude-haiku-4-5-20251001-v1:0)

USAGE:
    from agents.llm_client import call_llm, call_llm_json

    response = call_llm("Evaluate this post...", timeout=60)
    data = call_llm_json("Return JSON: {score: 85}", timeout=45)
"""

import json
import os
from typing import Optional, Type

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

DEFAULT_TIMEOUT = 60
DEFAULT_MODEL = os.getenv("STRANDS_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
FAST_MODEL = os.getenv("STRANDS_FAST_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
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


def call_llm(
    prompt: str,
    timeout: int = DEFAULT_TIMEOUT,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    fast: bool = False,
) -> Optional[str]:
    """
    Call Claude via Strands + Bedrock and return text response.

    Args:
        prompt: The prompt to send
        timeout: Unused (Bedrock handles retries with exponential backoff)
        model: Optional model ID override
        system_prompt: Optional system prompt
        fast: If True, use the cheaper/faster Haiku model

    Returns:
        Response text string, or None if call failed
    """
    model_id = model or (FAST_MODEL if fast else DEFAULT_MODEL)
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
    Call Claude via Strands + Bedrock and extract JSON from the response.

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
    model_id = model or (FAST_MODEL if fast else DEFAULT_MODEL)
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
    'DEFAULT_MODEL',
    'FAST_MODEL',
]


if __name__ == "__main__":
    print(f"Testing Strands + Bedrock client (model: {DEFAULT_MODEL}, region: {AWS_REGION})...")
    try:
        result = call_llm("Say 'Hello from Bedrock!' in exactly those words.")
        print(f"Text response: {result}")

        json_result = call_llm_json('Respond with exactly this JSON: {"status": "ok", "number": 42}')
        print(f"JSON response: {json_result}")
    except LLMError as e:
        print(f"Error: {e}")
