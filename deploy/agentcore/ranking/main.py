"""
Ranking Agent — AgentCore Runtime Entrypoint
=============================================
Fetches recent AI news from Hetzner, runs Generate→QE→Debate→Rank
pipeline using pure Strands + Bedrock. No local ML deps needed.

POST /invocations  {"num_variants": 5, "elo_rounds": 3}
GET  /ping         health check
"""

import json
import logging
import os
import random
import re
import sys

# Force stdout logging before anything else so crashes are visible
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
log = logging.getLogger("ranking-agent")
log.info("Starting ranking-agent imports...")

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

log.info("fastapi OK")

from strands import Agent
from strands.models import BedrockModel

log.info("strands OK")

app = FastAPI()

HETZNER_FEED_URL = os.getenv("HETZNER_FEED_URL", "http://157.90.125.102:5002")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
MODEL = os.getenv("STRANDS_MODEL_ID", "us.anthropic.claude-sonnet-4-6")

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = BedrockModel(model_id=MODEL, region_name=REGION, max_tokens=4096)
    return _model


HOOK_STYLES = ["curiosity_gap", "bold_claim", "personal_story", "data_driven", "practical_value"]

GENERATION_PROMPT = """You are a LinkedIn content expert. Write a viral LinkedIn post about recent AI news.

Hook style: {hook_style}
News context:
{news_context}

Requirements:
- 1500-1900 characters
- Start with a scroll-stopping hook (first 2 lines are critical)
- One sentence per line, blank lines between thoughts
- End with a thought-provoking question
- 3-5 relevant hashtags at the end
- No markdown, max 2 emojis
- Cite specific news items inline

Write ONLY the post, no explanation:"""

QE_PROMPT = """Score this LinkedIn post 0-100 against best practices.

POST:
{content}

Scoring (100pts): hook(25) length(15) format(15) authenticity(15) CTA(10) hashtags(5) clarity(10) grammar(5)

Respond JSON only: {{"score":0-100,"feedback":"brief","strengths":["s1"],"issues":["i1"]}}"""

DEBATE_PROMPT = """Which LinkedIn post is more likely to go viral?

POST A ({style_a}):
{content_a}

POST B ({style_b}):
{content_b}

Judge on: hook power(40%) authenticity(25%) value(20%) engagement(15%)
Respond JSON only: {{"winner":"A or B","reasoning":"why","confidence":0.5-1.0}}"""


def _call(prompt: str) -> str:
    agent = Agent(model=_get_model())
    return str(agent(prompt))


def _call_json(prompt: str) -> dict:
    text = _call(prompt)
    start, end = text.find("{"), text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {}


def _fetch_news(limit: int = 15) -> list[dict]:
    resp = requests.get(f"{HETZNER_FEED_URL}/api/feed", timeout=15)
    resp.raise_for_status()
    posts = resp.json().get("posts", [])
    news = []
    for post in posts[:limit]:
        if post.get("winner_content"):
            news.append({
                "text": post["winner_content"][:300],
                "source": post.get("hook_style", "ai_news"),
                "run_id": post.get("run_id"),
            })
    return news[:limit]


def _clean(text: str) -> str:
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    return text.strip()


@app.get("/ping")
async def ping():
    return {"status": "healthy"}


@app.post("/invocations")
async def invoke(request: Request):
    body = await request.json()
    num_variants: int = min(body.get("num_variants", 5), 8)
    elo_rounds: int = min(body.get("elo_rounds", 3), 5)

    log.info(f"Ranking pipeline: variants={num_variants} rounds={elo_rounds}")

    try:
        news = _fetch_news()
        if not news:
            return JSONResponse({"error": "No news available from Hetzner"}, status_code=503)

        news_context = "\n".join(
            f"[{i+1}] {n['text'][:200]}" for i, n in enumerate(news[:10])
        )

        # Phase 1: Generate variants
        variants = []
        for i in range(num_variants):
            style = HOOK_STYLES[i % len(HOOK_STYLES)]
            content = _clean(_call(GENERATION_PROMPT.format(
                hook_style=style, news_context=news_context
            )))
            if content and len(content) > 200:
                variants.append({"id": f"v{i+1}", "content": content, "style": style,
                                  "qe_score": 0, "elo": 1000, "wins": 0, "losses": 0})
        log.info(f"Generated {len(variants)} variants")

        if not variants:
            return JSONResponse({"error": "Generation failed"}, status_code=500)

        # Phase 2: QE evaluation
        for v in variants:
            result = _call_json(QE_PROMPT.format(content=v["content"]))
            v["qe_score"] = result.get("score", 60)
            v["qe_feedback"] = result.get("feedback", "")
            v["strengths"] = result.get("strengths", [])
            v["issues"] = result.get("issues", [])
            log.info(f"  {v['id']} QE={v['qe_score']}")

        # Phase 3: ELO tournament
        K = 32
        for _ in range(elo_rounds):
            shuffled = variants.copy()
            random.shuffle(shuffled)
            pairs = [(shuffled[i], shuffled[i+1]) for i in range(0, len(shuffled)-1, 2)]
            for a, b in pairs:
                result = _call_json(DEBATE_PROMPT.format(
                    style_a=a["style"], content_a=a["content"][:800],
                    style_b=b["style"], content_b=b["content"][:800],
                ))
                winner_key = result.get("winner", "A")
                conf = float(result.get("confidence", 0.7))
                k = K * (0.5 + conf)
                exp_a = 1 / (1 + 10 ** ((b["elo"] - a["elo"]) / 400))
                actual_a = 1.0 if winner_key == "A" else 0.0
                a["elo"] += k * (actual_a - exp_a)
                b["elo"] += k * ((1 - actual_a) - (1 - exp_a))
                if winner_key == "A":
                    a["wins"] += 1; b["losses"] += 1
                else:
                    b["wins"] += 1; a["losses"] += 1

        ranked = sorted(variants, key=lambda v: v["elo"], reverse=True)
        winner = ranked[0]
        log.info(f"Winner: {winner['id']} ELO={winner['elo']:.0f} QE={winner['qe_score']}")

        return {
            "winner": {
                "content": winner["content"],
                "hook_style": winner["style"],
                "elo_rating": round(winner["elo"], 1),
                "qe_score": winner["qe_score"],
                "wins": winner["wins"],
                "losses": winner["losses"],
            },
            "ranking": [
                {"id": v["id"], "elo": round(v["elo"], 1),
                 "qe_score": v["qe_score"], "style": v["style"],
                 "wins": v["wins"], "losses": v["losses"]}
                for v in ranked
            ],
        }

    except Exception as e:
        log.exception("Ranking pipeline failed")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    log.info("Starting uvicorn on 0.0.0.0:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
