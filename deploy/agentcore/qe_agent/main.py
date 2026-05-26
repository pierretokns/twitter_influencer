"""
QE Agent — AgentCore Runtime Entrypoint
========================================
Scores a LinkedIn post variant using quality evaluation criteria.

POST /invocations  {"content": str, "hook_style": str}
GET  /ping         health check
"""

import json
import logging
import os
import re
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
log = logging.getLogger("qe-agent")
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
                "Use deterministic metrics plus local Phi-4-mini constrained JSON for QA, "
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
    # Fallback: try to extract score via regex
    score_match = re.search(r'"score"\s*:\s*(\d+)', text)
    if score_match:
        return {"score": int(score_match.group(1)), "feedback": "Parsed via regex fallback",
                "strengths": [], "issues": []}
    return {}


QE_PROMPT = """Score this LinkedIn post 0-100 against best practices.

POST:
{content}

Scoring (100pts): hook(25) length(15) format(15) authenticity(15) CTA(10) hashtags(5) clarity(10) grammar(5)

Respond JSON only: {{"score":0-100,"feedback":"brief","strengths":["s1"],"issues":["i1"]}}"""


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
    content: str = body.get("content", "")
    hook_style: str = body.get("hook_style", "")

    log.info(f"QE scoring post: {len(content)} chars, style={hook_style}")

    try:
        result = _call_json(QE_PROMPT.format(content=content))

        if not result:
            log.warning("QE returned empty result, using defaults")
            result = {"score": 60, "feedback": "Could not evaluate", "strengths": [], "issues": []}

        score = result.get("score", 60)
        # Clamp score to valid range
        score = max(0, min(100, int(score)))

        log.info(f"QE score: {score}")
        return {
            "score": score,
            "feedback": result.get("feedback", ""),
            "strengths": result.get("strengths", []),
            "issues": result.get("issues", []),
        }

    except Exception as e:
        log.exception("QE scoring failed")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    log.info("Starting uvicorn on 0.0.0.0:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
