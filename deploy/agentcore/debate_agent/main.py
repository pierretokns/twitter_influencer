"""
Debate Agent — AgentCore Runtime Entrypoint
============================================
Judges which of two LinkedIn post variants is more likely to go viral.

POST /invocations  {"post_a": {"content": str, "style": str},
                    "post_b": {"content": str, "style": str}}
GET  /ping         health check
"""

import json
import logging
import os
import re
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
log = logging.getLogger("debate-agent")
log.info("Starting imports...")

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

log.info("Imports OK")

app = FastAPI()
LEGACY_AGENTCORE_ENABLED = os.getenv("ALLOW_LEGACY_AGENTCORE_RUNTIME") == "1"


def _legacy_disabled_response():
    return JSONResponse(
        {
            "error": "legacy_agentcore_disabled",
            "message": (
                "This hosted Bedrock/Strands AgentCore runtime is disabled by default. "
                "Use reranker/ELO scoring plus local Phi-4-mini JSON judging if needed, "
                "or set ALLOW_LEGACY_AGENTCORE_RUNTIME=1 to intentionally run this legacy stack."
            ),
        },
        status_code=410,
    )

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
MODEL = os.getenv("STRANDS_MODEL_ID", "us.anthropic.claude-sonnet-4-6")

_model = None

def _get_model():
    global _model
    if _model is None:
        from strands.models import BedrockModel

        _model = BedrockModel(model_id=MODEL, region_name=REGION, max_tokens=4096)
    return _model

def _call(prompt: str) -> str:
    from strands import Agent

    return str(Agent(model=_get_model())(prompt))

def _call_json(prompt: str) -> dict:
    text = _call(prompt)
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    # Fallback: try regex extraction for key fields
    winner_match = re.search(r'"winner"\s*:\s*"([AB])"', text)
    conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
    if winner_match:
        return {
            "winner": winner_match.group(1),
            "confidence": float(conf_match.group(1)) if conf_match else 0.6,
            "reasoning": reasoning_match.group(1) if reasoning_match else "Parsed via regex fallback",
        }
    return {}


DEBATE_PROMPT = """Which LinkedIn post is more likely to go viral?

POST A ({style_a}):
{content_a}

POST B ({style_b}):
{content_b}

Judge on: hook power(40%) authenticity(25%) value(20%) engagement(15%)
Respond JSON only: {{"winner":"A or B","reasoning":"why","confidence":0.5-1.0}}"""


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/ping")
async def ping():
    if not LEGACY_AGENTCORE_ENABLED:
        return {"status": "disabled", "reason": "legacy_agentcore_disabled"}
    return {"status": "healthy"}


@app.post("/invocations")
async def invoke(request: Request):
    if not LEGACY_AGENTCORE_ENABLED:
        return _legacy_disabled_response()

    body = await request.json()
    post_a: dict = body.get("post_a", {})
    post_b: dict = body.get("post_b", {})

    content_a = post_a.get("content", "")
    style_a = post_a.get("style", "unknown")
    content_b = post_b.get("content", "")
    style_b = post_b.get("style", "unknown")

    log.info(f"Debating: style_a={style_a} vs style_b={style_b}")

    try:
        result = _call_json(DEBATE_PROMPT.format(
            style_a=style_a,
            content_a=content_a[:800],
            style_b=style_b,
            content_b=content_b[:800],
        ))

        if not result:
            log.warning("Debate returned empty result, defaulting to A")
            result = {"winner": "A", "confidence": 0.5, "reasoning": "Could not evaluate"}

        winner = result.get("winner", "A")
        if winner not in ("A", "B"):
            log.warning(f"Invalid winner value '{winner}', defaulting to A")
            winner = "A"

        confidence = float(result.get("confidence", 0.7))
        confidence = max(0.5, min(1.0, confidence))

        log.info(f"Debate result: winner={winner} confidence={confidence:.2f}")
        return {
            "winner": winner,
            "confidence": confidence,
            "reasoning": result.get("reasoning", ""),
        }

    except Exception as e:
        log.exception("Debate failed")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    log.info("Starting uvicorn on 0.0.0.0:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
