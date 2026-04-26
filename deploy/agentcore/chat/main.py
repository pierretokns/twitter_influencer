"""
Chat Agent — AgentCore Runtime Entrypoint
==========================================
RAG chatbot backed by Strands + Bedrock Claude.
Fetches context from Hetzner feed API — no local SQLite or ML deps.

POST /invocations  {"query": "...", "session_id": "...", "history": [...]}
GET  /ping         health check
"""

import logging
import os

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from strands import Agent
from strands.models import BedrockModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("chat-agent")

app = FastAPI()

HETZNER_FEED_URL = os.getenv("HETZNER_FEED_URL", "http://157.90.125.102:5002")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
MODEL = os.getenv("CHAT_MODEL", "us.anthropic.claude-sonnet-4-6")

SYSTEM_PROMPT = """You are an AI news analyst. Answer questions about recent AI developments
using the provided sources. Cite sources with [N] notation. Be concise and factual."""

_agent = None

def _get_agent():
    global _agent
    if _agent is None:
        model = BedrockModel(model_id=MODEL, region_name=REGION, max_tokens=2048)
        _agent = Agent(model=model, system_prompt=SYSTEM_PROMPT)
    return _agent


def _fetch_context(limit: int = 10) -> list[dict]:
    resp = requests.get(f"{HETZNER_FEED_URL}/api/feed", timeout=15)
    resp.raise_for_status()
    posts = resp.json().get("posts", [])
    return [
        {
            "id": str(p.get("run_id", i)),
            "text": p.get("winner_content", "")[:500],
            "hook_style": p.get("hook_style", ""),
            "completed_at": p.get("completed_at", ""),
        }
        for i, p in enumerate(posts[:limit])
        if p.get("winner_content")
    ]


@app.get("/ping")
async def ping():
    return {"status": "healthy"}


@app.post("/invocations")
async def invoke(request: Request):
    body = await request.json()
    query: str = body.get("query", "")
    session_id: str = body.get("session_id", "default")

    if not query:
        return JSONResponse({"error": "query is required"}, status_code=400)

    log.info(f"Chat [{session_id}]: {query[:80]}")

    try:
        sources = _fetch_context()
        context = "\n\n".join(f"[{i+1}] {s['text']}" for i, s in enumerate(sources))
        prompt = f"SOURCES:\n{context}\n\nQUESTION: {query}"

        answer = str(_get_agent()(prompt))

        return {"answer": answer, "sources": sources, "session_id": session_id}

    except Exception as e:
        log.exception("Chat agent failed")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
