"""
Evolution Agent — AgentCore Runtime Entrypoint
===============================================
Evolves a LinkedIn post based on QE feedback and issues.

POST /invocations  {"content": str, "hook_style": str, "news_context": str,
                    "feedback": str, "issues": list, "qe_score": int}
GET  /ping         health check
"""

import logging
import os
import re
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
log = logging.getLogger("evolution-agent")
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
                "The old influencer evolution workflow is not aligned with Brandon news summaries. "
                "Use local digest editing/citation verification, or set "
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


# ── Citation pipeline (identical copy from generator_agent) ───────────────────

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
    content: str = body.get("content", "")
    hook_style: str = body.get("hook_style", "")
    news_context: str = body.get("news_context", "")
    feedback: str = body.get("feedback", "No specific feedback")
    issues: list = body.get("issues", [])
    strengths: list = body.get("strengths", [])
    qe_score: int = body.get("qe_score", 60)

    log.info(f"Evolving post: style={hook_style} qe_score={qe_score} issues={len(issues)}")

    try:
        issues_text = "\n".join(f"- {x}" for x in issues) or "- Generic, lacks specificity"
        strengths_text = "\n".join(f"- {x}" for x in strengths) or "- None identified"

        raw_evolved = _call(EVOLUTION_PROMPT.format(
            qe_score=qe_score,
            content=content,
            issues=issues_text,
            strengths=strengths_text,
            feedback=feedback,
            news_context=news_context[:1500],
        ))
        evolved_content = _clean(raw_evolved)

        if not evolved_content or len(evolved_content) < 200:
            log.warning("Evolved content too short, returning original")
            return JSONResponse({"error": "Evolution produced insufficient content"}, status_code=500)

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

        log.info(f"Running citation pipeline on evolved content ({len(news_items)} news items)")
        evolved_content, evo_sources, _ = _run_citation_pipeline(evolved_content, news_items)

        log.info(f"Evolution complete: {len(evolved_content)} chars")
        return {
            "content": evolved_content,
            "source_attributions": evo_sources,
        }

    except Exception as e:
        log.exception("Evolution failed")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    log.info("Starting uvicorn on 0.0.0.0:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
