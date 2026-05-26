"""
Ranking Orchestrator — AgentCore Runtime Entrypoint
=====================================================
Pure orchestration: no Strands/BedrockModel imports.
Coordinates generator, qe, evolution, and debate agents via AgentCore invoke.

POST /invocations  {"num_variants": 5, "elo_rounds": 3, "evolve": true}
GET  /ping         health check
"""

import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
log = logging.getLogger("ranking-orchestrator")
log.info("Starting imports...")

import requests
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
                "This AgentCore orchestrator is disabled by default because it invokes "
                "legacy hosted Bedrock/Strands leaf agents. Use Hetzner-local services, "
                "or set ALLOW_LEGACY_AGENTCORE_RUNTIME=1 to intentionally run this legacy stack."
            ),
        },
        status_code=410,
    )

HETZNER_FEED_URL = os.getenv("HETZNER_FEED_URL", "http://157.90.125.102:5002")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
QE_THRESHOLD = int(os.getenv("QE_EVOLUTION_THRESHOLD", "70"))
DEDUP_THRESHOLD = float(os.getenv("DEDUP_THRESHOLD", "0.6"))

GENERATOR_ARN = os.getenv("GENERATOR_AGENT_ARN", "")
QE_ARN = os.getenv("QE_AGENT_ARN", "")
EVOLUTION_ARN = os.getenv("EVOLUTION_AGENT_ARN", "")
DEBATE_ARN = os.getenv("DEBATE_AGENT_ARN", "")

# ── AgentCore client ──────────────────────────────────────────────────────────

_agentcore = None

def _agentcore_client():
    global _agentcore
    if _agentcore is None:
        import boto3

        _agentcore = boto3.client("bedrock-agentcore", region_name=REGION)
    return _agentcore

def _invoke_agent(arn: str, payload: dict) -> dict:
    resp = _agentcore_client().invoke_agent_runtime(
        agentRuntimeArn=arn,
        payload=json.dumps(payload).encode(),
    )
    body = resp["response"].read()
    return json.loads(body)

def _parallel_invoke(arn: str, payloads: list[dict]) -> list[dict]:
    results = [None] * len(payloads)
    with ThreadPoolExecutor(max_workers=len(payloads)) as ex:
        futures = {ex.submit(_invoke_agent, arn, p): i for i, p in enumerate(payloads)}
        for f in as_completed(futures):
            results[futures[f]] = f.result()
    return results


# ── Hook styles ───────────────────────────────────────────────────────────────

HOOK_STYLES = ["curiosity_gap", "bold_claim", "personal_story", "data_driven", "practical_value"]

VIRAL_HOOKS = {
    "curiosity_gap": [
        "Most people don't realize this about {topic}...",
        "I discovered something unexpected about {topic}",
        "The hidden truth about {topic} that nobody talks about",
    ],
    "bold_claim": [
        "This will change everything we know about {topic}",
        "{topic} is dead. Here's what's replacing it.",
        "Forget everything you learned about {topic}",
    ],
    "personal_story": [
        "I was skeptical about {topic} until I saw this...",
        "I spent 6 months studying {topic}. Here's what I learned.",
        "When I first heard about {topic}, I didn't believe it",
    ],
    "data_driven": [
        "97% of professionals are missing this about {topic}",
        "New data reveals surprising truth about {topic}",
        "3 statistics about {topic} that will surprise you",
    ],
    "practical_value": [
        "How to leverage {topic} (step-by-step)",
        "Save 10 hours per week with {topic}",
        "Stop struggling with {topic}. Do this instead.",
    ],
}


# ── News fetching ─────────────────────────────────────────────────────────────

def _fetch_news(limit: int = 20) -> list[dict]:
    """
    Fetch recent AI news from Hetzner. Tries /api/news first for real tweets/articles;
    falls back to /api/feed if unavailable (with url=None to avoid broken citation links).
    """
    # Try /api/news first (real tweets and articles)
    try:
        resp = requests.get(f"{HETZNER_FEED_URL}/api/news", timeout=15)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", data.get("news", data.get("posts", [])))
        if items:
            log.info(f"Fetched {len(items)} items from /api/news")
            result = []
            for item in items[:limit]:
                result.append({
                    "text": (item.get("content") or item.get("text") or item.get("winner_content", ""))[:400],
                    "source": item.get("source_type") or item.get("author") or item.get("hook_style", "ai_news"),
                    "url": None,  # don't expose potentially broken links
                    "run_id": item.get("run_id") or item.get("id"),
                })
            result = [r for r in result if r["text"]]
            if result:
                return result[:limit]
    except Exception as e:
        log.warning(f"/api/news unavailable ({e}), falling back to /api/feed")

    # Fallback: /api/feed (our own generated posts)
    resp = requests.get(f"{HETZNER_FEED_URL}/api/feed", timeout=15)
    resp.raise_for_status()
    posts = resp.json().get("posts", [])

    by_style: dict[str, list] = defaultdict(list)
    for post in posts:
        if post.get("winner_content"):
            by_style[post.get("hook_style", "unknown")].append({
                "text": post["winner_content"][:400],
                "source": post.get("hook_style", "unknown"),
                "url": None,  # set url=None to avoid broken citation links
                "run_id": post.get("run_id"),
            })

    # Round-robin for diversity across styles
    result, styles = [], list(by_style.keys())
    random.shuffle(styles)
    idx = defaultdict(int)
    while len(result) < limit:
        added = False
        for style in styles:
            i = idx[style]
            if i < len(by_style[style]):
                result.append(by_style[style][i])
                idx[style] += 1
                added = True
                if len(result) >= limit:
                    break
        if not added:
            break
    return result[:limit]


def _format_news_context(news: list[dict]) -> str:
    """Format news for LLM context with [N] numbering."""
    lines = []
    for i, n in enumerate(news):
        source = n.get("source", "ai_news")
        lines.append(f"[{i+1}] ({source}): {n['text'][:350]}")
    return "\n\n".join(lines)


# ── Dedup ─────────────────────────────────────────────────────────────────────

def _bigrams(text: str) -> set:
    words = re.findall(r'\b\w+\b', text.lower())
    return set(zip(words, words[1:])) if len(words) > 1 else set()

def _jaccard(a: str, b: str) -> float:
    ba, bb = _bigrams(a), _bigrams(b)
    union = ba | bb
    return len(ba & bb) / len(union) if union else 0.0

def _dedup(variants: list[dict]) -> list[dict]:
    """Mark near-duplicate variants (lower QE score flagged)."""
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            if variants[i].get("is_duplicate") or variants[j].get("is_duplicate"):
                continue
            sim = _jaccard(variants[i]["content"], variants[j]["content"])
            if sim >= DEDUP_THRESHOLD:
                drop = i if variants[i]["qe_score"] <= variants[j]["qe_score"] else j
                variants[drop]["is_duplicate"] = True
                log.info(f"  Dedup: {variants[drop]['id']} ~ other (sim={sim:.2f})")
    return variants


# ── ELO math ──────────────────────────────────────────────────────────────────

def _elo_update(rating_a: float, rating_b: float, winner: str, confidence: float, k: float = 32) -> tuple[float, float]:
    """Update ELO ratings. Returns (new_rating_a, new_rating_b)."""
    k_scaled = k * (0.5 + confidence)
    exp_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    actual_a = 1.0 if winner == "A" else 0.0
    new_a = rating_a + k_scaled * (actual_a - exp_a)
    new_b = rating_b + k_scaled * ((1 - actual_a) - (1 - exp_a))
    return new_a, new_b


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
    num_variants: int = min(body.get("num_variants", 5), 8)
    elo_rounds: int = min(body.get("elo_rounds", 3), 5)
    evolve: bool = body.get("evolve", True)

    log.info(f"Tournament: variants={num_variants} rounds={elo_rounds} evolve={evolve}")

    try:
        # ── Phase 1: News selection ──────────────────────────────────────────
        log.info("Phase 1: Fetching news...")
        news = _fetch_news(limit=20)
        if not news:
            return JSONResponse({"error": "No news from Hetzner"}, status_code=503)

        news_context = _format_news_context(news[:12])
        log.info(f"News: {len(news)} items, {len(set(n['source'] for n in news))} sources")

        # ── Phase 2: Parallel generation ────────────────────────────────────
        log.info(f"Phase 2: Generating {num_variants} variants in parallel...")
        gen_payloads = []
        gen_meta = []
        for i in range(num_variants):
            style = HOOK_STYLES[i % len(HOOK_STYLES)]
            hook_example = random.choice(VIRAL_HOOKS[style]).format(topic="AI")
            focus_item = random.randint(1, min(10, len(news)))
            gen_payloads.append({
                "news_context": news_context,
                "hook_style": style,
                "hook_example": hook_example,
                "focus_item": focus_item,
            })
            gen_meta.append({"style": style, "focus_item": focus_item})

        gen_results = _parallel_invoke(GENERATOR_ARN, gen_payloads)

        variants = []
        for i, (res, meta) in enumerate(zip(gen_results, gen_meta)):
            if res and res.get("content") and len(res["content"]) > 200:
                variants.append({
                    "id": f"v{i+1}",
                    "content": res["content"],
                    "style": meta["style"],
                    "qe_score": 0,
                    "qe_feedback": "",
                    "strengths": [],
                    "issues": [],
                    "elo": 1000.0,
                    "wins": 0,
                    "losses": 0,
                    "generation": 1,
                    "evolved_from": None,
                    "is_duplicate": False,
                    "source_attributions": res.get("source_attributions", []),
                })
            else:
                log.warning(f"Variant v{i+1} generation failed or too short: {res}")

        log.info(f"Generated {len(variants)} valid variants")

        if not variants:
            return JSONResponse({"error": "Generation failed for all variants"}, status_code=500)

        # ── Phase 3: Parallel QE ─────────────────────────────────────────────
        log.info(f"Phase 3: QE scoring {len(variants)} variants in parallel...")
        qe_payloads = [{"content": v["content"], "hook_style": v["style"]} for v in variants]
        qe_results = _parallel_invoke(QE_ARN, qe_payloads)

        for v, qe_res in zip(variants, qe_results):
            if qe_res:
                v["qe_score"] = qe_res.get("score", 60)
                v["qe_feedback"] = qe_res.get("feedback", "")
                v["strengths"] = qe_res.get("strengths", [])
                v["issues"] = qe_res.get("issues", [])
            else:
                v["qe_score"] = 60
            log.info(f"  QE {v['id']}: {v['qe_score']}")

        # ── Phase 4: Dedup ───────────────────────────────────────────────────
        log.info("Phase 4: Deduplication...")
        variants = _dedup(variants)
        unique = [v for v in variants if not v["is_duplicate"]]
        dupes = len(variants) - len(unique)
        if dupes:
            log.info(f"Dedup: {dupes} near-duplicates removed, {len(unique)} unique")
        variants = unique if len(unique) >= 2 else variants

        # ── Phase 5: Sequential evolution ───────────────────────────────────
        evolved_count = 0
        if evolve:
            to_evolve = [v for v in variants if v["qe_score"] < QE_THRESHOLD]
            log.info(f"Phase 5: Evolving {len(to_evolve)} variants below QE {QE_THRESHOLD}...")
            for v in to_evolve:
                log.info(f"  Evolving {v['id']} (QE={v['qe_score']})...")
                evo_res = _invoke_agent(EVOLUTION_ARN, {
                    "content": v["content"],
                    "hook_style": v["style"],
                    "news_context": news_context[:1500],
                    "feedback": v["qe_feedback"] or "No specific feedback",
                    "issues": v["issues"],
                    "strengths": v["strengths"],
                    "qe_score": v["qe_score"],
                })

                if not evo_res or not evo_res.get("content") or len(evo_res["content"]) < 200:
                    log.warning(f"  Evolution of {v['id']} produced insufficient content")
                    continue

                # Re-score evolved content
                new_qe_res = _invoke_agent(QE_ARN, {
                    "content": evo_res["content"],
                    "hook_style": v["style"],
                })
                new_score = new_qe_res.get("score", v["qe_score"]) if new_qe_res else v["qe_score"]

                if new_score >= v["qe_score"]:
                    log.info(f"  Evolved {v['id']}: QE {v['qe_score']} → {new_score}")
                    v["content"] = evo_res["content"]
                    v["qe_score"] = new_score
                    v["qe_feedback"] = new_qe_res.get("feedback", "") if new_qe_res else ""
                    v["strengths"] = new_qe_res.get("strengths", []) if new_qe_res else []
                    v["issues"] = new_qe_res.get("issues", []) if new_qe_res else []
                    v["generation"] = 2
                    v["source_attributions"] = evo_res.get("source_attributions", [])
                    evolved_count += 1
                else:
                    log.info(f"  Evolution of {v['id']} did not improve ({new_score} < {v['qe_score']})")

        # ── Phase 6: Parallel ELO debates ───────────────────────────────────
        log.info(f"Phase 6: ELO tournament ({elo_rounds} rounds)...")
        total_debates = 0
        for rnd in range(elo_rounds):
            log.info(f"  ELO round {rnd+1}/{elo_rounds}...")
            shuffled = variants.copy()
            random.shuffle(shuffled)
            pairs = [(shuffled[i], shuffled[i+1]) for i in range(0, len(shuffled)-1, 2)]

            if not pairs:
                log.info("  Not enough variants for debates this round")
                continue

            # Build debate payloads for parallel invocation
            debate_payloads = [
                {
                    "post_a": {"content": a["content"], "style": a["style"]},
                    "post_b": {"content": b["content"], "style": b["style"]},
                }
                for a, b in pairs
            ]

            debate_results = _parallel_invoke(DEBATE_ARN, debate_payloads)

            # Apply ELO updates after all pairs resolve
            for (a, b), debate_res in zip(pairs, debate_results):
                if not debate_res:
                    log.warning(f"  Debate {a['id']} vs {b['id']} failed, skipping")
                    continue
                winner_key = debate_res.get("winner", "A")
                conf = float(debate_res.get("confidence", 0.7))
                new_elo_a, new_elo_b = _elo_update(a["elo"], b["elo"], winner_key, conf)
                a["elo"] = new_elo_a
                b["elo"] = new_elo_b
                if winner_key == "A":
                    a["wins"] += 1
                    b["losses"] += 1
                else:
                    b["wins"] += 1
                    a["losses"] += 1
                total_debates += 1
                log.info(f"  {a['id']} vs {b['id']}: winner={winner_key} conf={conf:.2f}")

        ranked = sorted(variants, key=lambda v: v["elo"], reverse=True)
        winner = ranked[0]
        log.info(
            f"Winner: {winner['id']} ELO={winner['elo']:.0f} "
            f"QE={winner['qe_score']} gen={winner['generation']}"
        )

        # ── Build result payload ─────────────────────────────────────────────
        winner_sources = winner.get("source_attributions", [])
        cited_sources = [s for s in winner_sources if s.get("is_referenced")]
        cited_nums = {s.get("citation_number") for s in cited_sources if s.get("citation_number")}

        # url=None to avoid broken citation links from feed posts
        news_sources = [
            {
                "citation_number": i + 1,
                "text": n.get("text", "")[:300],
                "source_type": "ai_news",
                "author": n.get("source", ""),
                "url": None,
                "is_cited": (i + 1) in cited_nums,
            }
            for i, n in enumerate(news[:12])
        ]

        result = {
            "winner": {
                "content": winner["content"],
                "hook_style": winner["style"],
                "elo_rating": round(winner["elo"], 1),
                "qe_score": winner["qe_score"],
                "wins": winner["wins"],
                "losses": winner["losses"],
                "generation": winner["generation"],
                "citations": len(cited_sources),
            },
            "news_sources": news_sources,
            "ranking": [
                {
                    "id": v["id"], "elo": round(v["elo"], 1),
                    "qe_score": v["qe_score"], "style": v["style"],
                    "wins": v["wins"], "losses": v["losses"],
                    "generation": v["generation"],
                }
                for v in ranked
            ],
            "total_debates": total_debates,
            "elo_rounds": elo_rounds,
            "evolved": evolved_count,
            "deduped": dupes,
            "pipeline": {
                "phases": ["select", "generate", "qe", "dedup", "evolve", "elo"],
                "news_items": len(news),
                "variants_generated": num_variants,
                "variants_after_dedup": len(unique) if dupes else num_variants,
                "variants_evolved": evolved_count,
                "winner_generation": winner["generation"],
                "winner_citations": len(cited_sources),
            },
        }

        # ── Persist to Hetzner feed API ──────────────────────────────────────
        log.info("Persisting winner to Hetzner feed...")
        try:
            save_resp = requests.post(
                f"{HETZNER_FEED_URL}/api/submit_result",
                json=result,
                timeout=10,
            )
            run_id = save_resp.json().get("run_id")
            log.info(f"Saved to feed: run_id={run_id}")
            result["run_id"] = run_id
        except Exception as e:
            log.warning(f"Failed to save to feed: {e}")

        return result

    except Exception as e:
        log.exception("Tournament failed")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    log.info("Starting uvicorn on 0.0.0.0:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
