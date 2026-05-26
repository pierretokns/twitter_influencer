"""
Generator Agent — AgentCore Runtime Entrypoint
===============================================
Generates a LinkedIn post variant with prompt-based citations.

POST /invocations  {"news_context": str, "hook_style": str, "hook_example": str, "focus_item": int}
GET  /ping         health check
"""

import json
import logging
import os
import re
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
log = logging.getLogger("generator-agent")
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
                "The old influencer generator is not aligned with Brandon news summaries. "
                "Use the Hetzner-local LFM2-2.6B RAG/news generator, or set "
                "ALLOW_LEGACY_AGENTCORE_RUNTIME=1 to intentionally run this legacy stack."
            ),
        },
        status_code=410,
    )

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
MODEL = os.getenv("STRANDS_MODEL_ID", "us.anthropic.claude-sonnet-4-6")
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

def _clean(text: str) -> str:
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r"^(Here'?s?|Certainly|Sure|Of course)\s.*?:\s*\n*", '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(item\s*\[\d+\]\)', '', text)
    return text.strip()


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


# ── Citation pipeline ─────────────────────────────────────────────────────────

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
    """Stage 2: Verify each citation via entity overlap."""
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
    weak = [r for r in verification if r['status'] in ('weak', 'invalid')]
    corrected = content
    for result in weak:
        current = len(re.findall(r'\[\d+\]', corrected))
        if current - 1 < min_citations:
            warnings.append(f"Kept weak [{result['citation']}] to maintain minimum citations")
            continue
        corrected = re.sub(rf'\[{result["citation"]}\]', '', corrected)
        warnings.append(f"Removed [{result['citation']}]: {result['reason']}")
    return corrected.strip(), warnings

def _run_citation_pipeline(content: str, news_items: list[dict]) -> tuple[str, list[dict], list[str]]:
    """Full 3-stage citation pipeline. Returns (content, annotated_sources, warnings)."""
    content, annotated = _parse_llm_citations(content, news_items)
    verification = _verify_citations(content, annotated)
    content, warnings = _correct_citations(content, verification, annotated)
    content, annotated = _parse_llm_citations(content, annotated)
    if warnings:
        log.info(f"Citation corrections: {warnings}")
    return content, annotated, warnings


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
    news_context: str = body.get("news_context", "")
    hook_style: str = body.get("hook_style", "curiosity_gap")
    hook_example: str = body.get("hook_example", "")
    focus_item: int = body.get("focus_item", 1)

    log.info(f"Generating variant: style={hook_style} focus_item={focus_item}")

    try:
        raw = _call(GENERATION_PROMPT.format(
            news_context=news_context,
            focus_item=focus_item,
            hook_style=hook_style,
            hook_example=hook_example,
        ))
        content = _clean(raw)

        if not content or len(content) < 200:
            log.warning("Generated content too short or empty")
            return JSONResponse({"error": "Generated content too short"}, status_code=500)

        # Parse news items from context for citation pipeline
        news_items = []
        for line in news_context.split("\n\n"):
            line = line.strip()
            if line:
                m = re.match(r'^\[(\d+)\]\s*\(([^)]*)\):\s*(.*)', line, re.DOTALL)
                if m:
                    news_items.append({"text": m.group(3).strip(), "source": m.group(2)})
                else:
                    news_items.append({"text": line, "source": "unknown"})

        log.info(f"Running citation pipeline on {len(news_items)} news items")
        content, annotated_sources, cite_warnings = _run_citation_pipeline(content, news_items)

        log.info(f"Generation complete: {len(content)} chars, {sum(1 for s in annotated_sources if s.get('is_referenced'))} citations")
        return {
            "content": content,
            "style": hook_style,
            "source_attributions": annotated_sources,
        }

    except Exception as e:
        log.exception("Generation failed")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    log.info("Starting uvicorn on 0.0.0.0:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
