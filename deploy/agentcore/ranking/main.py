"""
Ranking Agent — AgentCore Runtime Entrypoint
=============================================
Full 1:1 port of the local tournament pipeline:
  News Selection → Generate (with citations) → QE → Dedup → Evolve → ELO → Persist

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

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
log = logging.getLogger("ranking-agent")
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
                "This hosted Bedrock/Strands AgentCore runtime is disabled by default. "
                "The old all-in-one influencer tournament is not aligned with Brandon news summaries. "
                "Use the Hetzner-local news-summary workflow, or set "
                "ALLOW_LEGACY_AGENTCORE_RUNTIME=1 to intentionally run this legacy stack."
            ),
        },
        status_code=410,
    )

HETZNER_FEED_URL = os.getenv("HETZNER_FEED_URL", "http://157.90.125.102:5002")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
MODEL = os.getenv("STRANDS_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
QE_THRESHOLD = int(os.getenv("QE_EVOLUTION_THRESHOLD", "70"))
DEDUP_THRESHOLD = float(os.getenv("DEDUP_THRESHOLD", "0.6"))
SEMANTIC_THRESHOLD = float(os.getenv("CITATION_THRESHOLD", "0.3"))

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
    return {}

def _clean(text: str) -> str:
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r"^(Here'?s?|Certainly|Sure|Of course)\s.*?:\s*\n*", '', text, flags=re.IGNORECASE)
    # Remove stray "(item [N])" artifact patterns
    text = re.sub(r'\(item\s*\[\d+\]\)', '', text)
    return text.strip()


# ── Hook styles (1:1 from variant_generator.py) ──────────────────────────────

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

# ── Prompts (1:1 from variant_generator.py + evolution_agent.py) ─────────────

GENERATION_PROMPT = '''You are a LinkedIn content creator with 100K+ followers. Write a viral post about TODAY'S AI NEWS.

===== TODAY'S AI NEWS (NUMBERED SOURCES) =====
{news_context}

===== CITATION RULES =====
Add [N] citations when referencing specific information from source N.

Rules:
- Only cite when DIRECTLY using information from that specific source
- Do NOT cite for your own insights or generic observations
- Maximum 5 citations total (use them strategically)
- Place [N] at end of sentence, before punctuation
- Each source should only be cited ONCE (for its strongest claim)
- IMPORTANT: Ensure the source actually contains the information you\'re citing

Example: "OpenAI released GPT-5 with multimodal capabilities[1]. This represents a major shift..."

===== YOUR TASK =====
Write a LinkedIn post that:
1. MUST reference specific news from above (mention the actual development, company, or finding)
2. FOCUS primarily on news item [{focus_item}] but can reference others
3. Add YOUR unique insight, opinion, or takeaway - don\'t just summarize
4. Make it feel timely and current ("Just saw that...", "This week...", "Breaking:")
5. Include [N] citations for specific facts from sources (NOT for your opinions)

===== HOOK STYLE: {hook_style} =====
Example: "{hook_example}"

===== FORMAT REQUIREMENTS =====
- First 210 characters = CRITICAL hook (shows before "see more")
- One sentence per line with blank lines between paragraphs
- Short sentences under 12 words perform best
- End with a thought-provoking QUESTION
- Target 1,500-1,900 characters
- NO markdown symbols (no ** or #)
- 1-3 emojis strategically placed
- 3-5 hashtags at the very end

Write ONLY the post. Start directly with the hook:'''

QE_PROMPT = """Score this LinkedIn post 0-100 against best practices.

POST:
{content}

Scoring (100pts): hook(25) length(15) format(15) authenticity(15) CTA(10) hashtags(5) clarity(10) grammar(5)

Respond JSON only: {{"score":0-100,"feedback":"brief","strengths":["s1"],"issues":["i1"]}}"""

EVOLUTION_PROMPT = '''You are a LinkedIn content EVOLUTION expert. Improve this post based on specific feedback.

ORIGINAL POST (QE Score: {qe_score}/100):
{content}

ISSUES TO FIX:
{issues}

STRENGTHS TO KEEP:
{strengths}

QE FEEDBACK: {feedback}

NEWS CONTEXT (reference this with [N] citations!):
{news_context}

Rewrite to:
1. Fix all issues listed above
2. Keep all strengths
3. Reference specific news from context with [N] citations
4. Make the hook (first 2 lines) more scroll-stopping
5. Target 1,500-1,900 characters
6. NO markdown, max 2 emojis, end with question, 3-5 hashtags

Write ONLY the improved post, start directly with the hook:'''

DEBATE_PROMPT = """Which LinkedIn post is more likely to go viral?

POST A ({style_a}):
{content_a}

POST B ({style_b}):
{content_b}

Judge on: hook power(40%) authenticity(25%) value(20%) engagement(15%)
Respond JSON only: {{"winner":"A or B","reasoning":"why","confidence":0.5-1.0}}"""


# ── News fetching with source diversity ──────────────────────────────────────

def _fetch_news(limit: int = 20) -> list[dict]:
    """
    Fetch recent posts from Hetzner, round-robin across hook_styles for
    diversity (mirrors news_selector.py's MMR diversity approach without ML).
    """
    resp = requests.get(f"{HETZNER_FEED_URL}/api/feed", timeout=15)
    resp.raise_for_status()
    posts = resp.json().get("posts", [])

    by_style: dict[str, list] = defaultdict(list)
    for post in posts:
        if post.get("winner_content"):
            by_style[post.get("hook_style", "unknown")].append({
                "text": post["winner_content"][:400],
                "source": post.get("hook_style", "unknown"),
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
    """Format news for LLM context with [N] numbering (matches variant_generator format)."""
    lines = []
    for i, n in enumerate(news):
        source = n.get("source", "ai_news")
        lines.append(f"[{i+1}] ({source}): {n['text'][:350]}")
    return "\n\n".join(lines)


# ── Citation pipeline (3-stage, entity-overlap based) ────────────────────────

# Common words to exclude from entity extraction
_STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","this","that","these","those","is","are","was","were","be",
    "been","being","have","has","had","do","does","did","will","would","could",
    "should","may","might","shall","can","it","its","we","our","they","their",
    "he","she","his","her","you","your","i","my","me","us","them","what",
    "which","who","when","where","how","why","not","no","if","then","than",
    "so","as","up","out","about","into","through","during","before","after",
    "above","below","just","also","more","most","other","some","such","own",
    "same","few","both","all","any","each","every","much","now","here","there",
    "ai","linkedin","post","news","week","today","new","just","said","says",
}

def _extract_entities(text: str) -> set[str]:
    """Extract meaningful noun phrases / proper nouns (no ML needed)."""
    words = re.findall(r'\b[A-Za-z][A-Za-z0-9\-\.]+\b', text)
    entities = set()
    for w in words:
        lower = w.lower()
        if lower not in _STOPWORDS and len(w) >= 3:
            entities.add(lower)
        # Keep capitalised multi-word candidates (likely proper nouns)
        if w[0].isupper() and lower not in _STOPWORDS:
            entities.add(w)
    return entities

def _parse_llm_citations(content: str, news_items: list[dict]) -> tuple[str, list[dict]]:
    """Stage 1: Extract [N] markers placed by the LLM."""
    cited_nums = set(int(m) for m in re.findall(r'\[(\d+)\]', content))
    annotated = []
    for i, item in enumerate(news_items):
        item_copy = dict(item)
        num = i + 1
        item_copy['is_referenced'] = num in cited_nums
        item_copy['citation_number'] = num if num in cited_nums else None
        item_copy['attribution_score'] = 1.0 if num in cited_nums else 0.0
        annotated.append(item_copy)
    log.info(f"LLM generated citations: {sorted(cited_nums)}")
    return content, annotated

def _verify_citations(content: str, news_items: list[dict], threshold: float = SEMANTIC_THRESHOLD) -> list[dict]:
    """
    Stage 2: Verify each citation via entity overlap.
    (Local equivalent of verify_citations_hybrid — no BGE-M3 needed.)
    """
    results = []
    pattern = r'([^.!?\n]*\[(\d+)\][^.!?\n]*[.!?]?)'
    for match in re.finditer(pattern, content):
        sentence = match.group(1).strip()
        num = int(match.group(2))
        if num < 1 or num > len(news_items):
            results.append({'citation': num, 'sentence': sentence, 'status': 'invalid',
                            'reason': f'[{num}] out of range'})
            continue
        source = news_items[num - 1]
        source_text = source.get('text', '')[:500]
        sent_ents = _extract_entities(sentence)
        src_ents = _extract_entities(source_text)
        overlap = sent_ents & src_ents
        # Use Jaccard similarity as proxy for semantic similarity
        union = sent_ents | src_ents
        sim = len(overlap) / len(union) if union else 0.0
        status = 'verified' if (overlap and sim >= threshold) else 'weak'
        results.append({
            'citation': num, 'sentence': sentence,
            'source': source.get('source', 'unknown'),
            'similarity': sim, 'entity_overlap': list(overlap)[:5],
            'status': status,
            'reason': f'overlap={list(overlap)[:3]} sim={sim:.2f}',
        })
    return results

def _correct_citations(content: str, verification: list[dict],
                        news_items: list[dict], min_citations: int = 1) -> tuple[str, list[str]]:
    """Stage 3: Remove weak/invalid citations, preserve min_citations."""
    warnings = []
    verified_count = sum(1 for r in verification if r['status'] == 'verified')
    weak = [r for r in verification if r['status'] in ('weak', 'invalid')]
    corrected = content
    for result in weak:
        # Don't remove if it would drop below min_citations
        current = len(re.findall(r'\[\d+\]', corrected))
        if current - 1 < min_citations:
            warnings.append(f"Kept weak [{result['citation']}] to maintain minimum citations")
            continue
        # Remove the [N] marker
        corrected = re.sub(rf'\[{result["citation"]}\]', '', corrected)
        warnings.append(f"Removed [{result['citation']}]: {result['reason']}")
    return corrected.strip(), warnings

def _run_citation_pipeline(content: str, news_items: list[dict]) -> tuple[str, list[dict], list[str]]:
    """Full 3-stage citation pipeline. Returns (content, annotated_sources, warnings)."""
    content, annotated = _parse_llm_citations(content, news_items)
    verification = _verify_citations(content, annotated)
    content, warnings = _correct_citations(content, verification, annotated)
    # Re-parse after correction to get final state
    content, annotated = _parse_llm_citations(content, annotated)
    if warnings:
        log.info(f"Citation corrections: {warnings}")
    return content, annotated, warnings


# ── Dedup via bigram Jaccard (proxy for ProximityAgent cosine sim) ───────────

def _bigrams(text: str) -> set:
    words = re.findall(r'\b\w+\b', text.lower())
    return set(zip(words, words[1:])) if len(words) > 1 else set()

def _jaccard(a: str, b: str) -> float:
    ba, bb = _bigrams(a), _bigrams(b)
    union = ba | bb
    return len(ba & bb) / len(union) if union else 0.0

def _dedup(variants: list[dict]) -> list[dict]:
    """Mark near-duplicate variants (lower QE score flagged). Mirrors ProximityAgent."""
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


# ── Endpoints ────────────────────────────────────────────────────────────────

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
        # ── Phase 1: Diverse news selection ─────────────────────────────────
        news = _fetch_news(limit=20)
        if not news:
            return JSONResponse({"error": "No news from Hetzner"}, status_code=503)

        news_context = _format_news_context(news[:12])
        log.info(f"News: {len(news)} items, {len(set(n['source'] for n in news))} styles")

        # ── Phase 2: Generate variants with prompt-based citations ───────────
        variants = []
        for i in range(num_variants):
            style = HOOK_STYLES[i % len(HOOK_STYLES)]
            hook_example = random.choice(VIRAL_HOOKS[style]).format(topic="AI")
            focus_item = random.randint(1, min(10, len(news)))

            raw = _call(GENERATION_PROMPT.format(
                news_context=news_context,
                focus_item=focus_item,
                hook_style=style,
                hook_example=hook_example,
            ))
            content = _clean(raw)

            if content and len(content) > 200:
                # Run 3-stage citation pipeline on generated content
                content, annotated_sources, cite_warnings = _run_citation_pipeline(content, news[:12])
                variants.append({
                    "id": f"v{i+1}", "content": content, "style": style,
                    "qe_score": 0, "qe_feedback": "", "strengths": [], "issues": [],
                    "elo": 1000.0, "wins": 0, "losses": 0,
                    "generation": 1, "evolved_from": None, "is_duplicate": False,
                    "source_attributions": annotated_sources,
                    "citation_warnings": cite_warnings,
                })
        log.info(f"Generated {len(variants)} variants")

        if not variants:
            return JSONResponse({"error": "Generation failed"}, status_code=500)

        # ── Phase 3: QE evaluation ───────────────────────────────────────────
        for v in variants:
            result = _call_json(QE_PROMPT.format(content=v["content"]))
            v["qe_score"] = result.get("score", 60)
            v["qe_feedback"] = result.get("feedback", "")
            v["strengths"] = result.get("strengths", [])
            v["issues"] = result.get("issues", [])
            log.info(f"  QE {v['id']}: {v['qe_score']}")

        # ── Phase 4: Dedup (ProximityAgent equivalent) ───────────────────────
        variants = _dedup(variants)
        unique = [v for v in variants if not v["is_duplicate"]]
        dupes = len(variants) - len(unique)
        if dupes:
            log.info(f"Dedup: {dupes} near-duplicates removed, {len(unique)} unique")
        variants = unique if len(unique) >= 2 else variants  # need at least 2 for tournament

        # ── Phase 5: Evolution (EvolutionAgent equivalent) ───────────────────
        evolved_count = 0
        if evolve:
            to_evolve = [v for v in variants if v["qe_score"] < QE_THRESHOLD]
            log.info(f"Evolving {len(to_evolve)} variants below QE {QE_THRESHOLD}...")
            for v in to_evolve:
                issues_text = "\n".join(f"- {x}" for x in v["issues"]) or "- Generic, lacks specificity"
                strengths_text = "\n".join(f"- {x}" for x in v["strengths"]) or "- None identified"
                raw_evolved = _call(EVOLUTION_PROMPT.format(
                    qe_score=v["qe_score"],
                    content=v["content"],
                    issues=issues_text,
                    strengths=strengths_text,
                    feedback=v["qe_feedback"] or "No specific feedback",
                    news_context=news_context[:1500],
                ))
                evolved_content = _clean(raw_evolved)
                if not evolved_content or len(evolved_content) < 200:
                    continue

                # Re-run citation pipeline on evolved content
                evolved_content, evo_sources, _ = _run_citation_pipeline(evolved_content, news[:12])

                # Re-evaluate QE
                new_qe = _call_json(QE_PROMPT.format(content=evolved_content))
                new_score = new_qe.get("score", v["qe_score"])
                if new_score >= v["qe_score"]:
                    log.info(f"  Evolved {v['id']}: QE {v['qe_score']} → {new_score}")
                    v["content"] = evolved_content
                    v["qe_score"] = new_score
                    v["qe_feedback"] = new_qe.get("feedback", "")
                    v["strengths"] = new_qe.get("strengths", [])
                    v["issues"] = new_qe.get("issues", [])
                    v["generation"] = 2
                    v["source_attributions"] = evo_sources
                    evolved_count += 1
                else:
                    log.info(f"  Evolution of {v['id']} did not improve ({new_score} < {v['qe_score']})")

        # ── Phase 6: ELO tournament (DebateAgent + EloRanker equivalent) ─────
        K = 32
        total_debates = 0
        for rnd in range(elo_rounds):
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
                total_debates += 1

        ranked = sorted(variants, key=lambda v: v["elo"], reverse=True)
        winner = ranked[0]
        log.info(
            f"Winner: {winner['id']} ELO={winner['elo']:.0f} "
            f"QE={winner['qe_score']} gen={winner['generation']}"
        )

        # ── Build result payload ─────────────────────────────────────────────
        # Count cited sources in winner
        winner_sources = winner.get("source_attributions", [])
        cited_sources = [s for s in winner_sources if s.get("is_referenced")]

        # Build news sources for frontend citation rendering
        cited_nums = {s.get("citation_number") for s in cited_sources if s.get("citation_number")}
        news_sources = [
            {
                "citation_number": i + 1,
                "text": n.get("text", "")[:300],
                "source_type": "ai_news",
                "author": n.get("source", ""),
                "url": f"{HETZNER_FEED_URL}/post/{n.get('run_id', '')}" if n.get("run_id") else None,
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
