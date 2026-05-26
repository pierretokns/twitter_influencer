"""
QEAgent - Quality Evaluation Agent
===================================

AGENT TYPE: Evaluation Agent (Single-turn reasoning with structured output)

PURPOSE:
    Scores Brandon news bulletins against source-grounded usefulness.
    Returns detailed breakdown with strengths and issues for evolution feedback.

PROMPT ENGINEERING:
    - Uses explicit scoring criteria with point allocation
    - Requires JSON output for structured parsing
    - Includes examples of things to penalize
    - Total score: 100 points across 8 dimensions

SCORING CRITERIA (100 points total):
    1. Missed-news value (25pts): Surfaces items Brandon likely did not see on x.com.
    2. Novelty/source quality (20pts): Prioritizes releases, papers, videos, repos, primary sources.
    3. Source-grounding/citations (20pts): Claims are cited and supported.
    4. Finance/workflow relevance (15pts): Connects to Brandon's finance-facing work when supported.
    5. Bulletin format (10pts): Terse scan-friendly bullets, no narrative filler.
    6. Actionability (10pts): Clear "check next" implications.

IMPLEMENTATION NOTES:
    - Uses Claude CLI for evaluation
    - Parses JSON response for structured feedback
    - Falls back to default score (60) if parsing fails
    - Rate-limited with 0.5s delay between evaluations

USAGE:
    qe_agent = QEAgent()
    variant = qe_agent.evaluate_post(variant)  # Updates qe_score, qe_feedback, etc.
    variants = qe_agent.evaluate_batch(variants)  # Batch evaluation
"""

import time
from typing import List

try:
    from strands import tool
except ImportError:
    def tool(func=None, **_kwargs):
        if func is None:
            return lambda wrapped: wrapped
        return func

from .post_variant import PostVariant
from .llm_client import call_llm_json, LLMError
from .models import QEResult


class QEAgent:
    """
    Quality Evaluation Agent - Scores Brandon news bulletins.

    This agent implements a rubric-based evaluation system that provides
    detailed feedback for the Evolution Agent to use when improving posts.
    """

    # Scoring criteria with point allocations for Brandon's news workflow.
    CRITERIA = {
        "missed_news_value": 25,
        "novelty_source_quality": 20,
        "source_grounding": 20,
        "finance_workflow_relevance": 15,
        "bulletin_format": 10,
        "actionability": 10,
    }

    # The evaluation prompt template for the Brandon news product.
    EVALUATION_PROMPT = '''You are Brandon's AI-news QA reviewer. Evaluate this candidate news bulletin for Brandon, not for LinkedIn virality.

BULLETIN ({char_count} characters):
{content}

===== SCORING CRITERIA (total 100 points) =====

1. MISSED-NEWS VALUE (25pts):
   - Prioritizes items Brandon likely did not already see on x.com.
   - Rewards YouTube/video drops, model releases, repos, papers, conference/CFP deadlines, Boston/NYC AI events, primary-source announcements, eval/tooling changes.
   - Penalizes generic X discourse and obvious summaries.

2. NOVELTY & SOURCE QUALITY (20pts):
   - Separates genuinely novel releases/papers/tools from commentary.
   - Names concrete models, papers, repos, videos, datasets, companies, or benchmark changes when supported.

3. SOURCE-GROUNDING & CITATIONS (20pts):
   - Every factual bullet has numeric citations.
   - No unsupported extrapolation.
   - Clearly admits when sources are thin.

4. FINANCE / WORKFLOW RELEVANCE (15pts):
   - Highlights finance-facing relevance when supported: payments, fraud/risk, compliance, banks, asset managers, hedge funds, regulated workflows.
   - Does not force finance relevance when the sources do not support it.

5. BULLETIN FORMAT (10pts):
   - 3-6 compact bullets.
   - Starts bullets with short labels.
   - No narrative opener, story arc, influencer tone, hashtags, or engagement CTA.

6. ACTIONABILITY (10pts):
   - Says what Brandon should inspect, test, save, or ignore next.

===== PENALIZE HEAVILY =====
- Narrative filler: "the deeper shift", "these developments underscore", "AI is transforming..."
- Social-post mechanics: hooks, hashtags, engagement questions, motivational framing.
- Any claim about a company, model, paper, or release without a citation.
- Repeating obvious x.com discourse instead of primary-source or missed items.

Respond in JSON format ONLY:
{{"score": 0-100, "breakdown": {{"missed_news_value": 0-25, "novelty_source_quality": 0-20, "source_grounding": 0-20, "finance_workflow_relevance": 0-15, "bulletin_format": 0-10, "actionability": 0-10}}, "feedback": "brief specific feedback", "strengths": ["strength1"], "issues": ["issue1"]}}'''

    def __init__(self):
        """Initialize the QE Agent"""
        pass

    def evaluate_post(self, variant: PostVariant) -> PostVariant:
        """
        Evaluate a single post variant and update its QE fields.

        Args:
            variant: The PostVariant to evaluate

        Returns:
            The same PostVariant with updated qe_* fields
        """
        prompt = self.EVALUATION_PROMPT.format(
            content=variant.content,
            char_count=len(variant.content)
        )

        try:
            data = call_llm_json(prompt, timeout=45)
            if data:
                # Update variant with evaluation results
                variant.qe_score = data.get('score', 50)
                variant.qe_feedback = data.get('feedback', '')
                variant.qe_breakdown = data.get('breakdown', {})
                variant.qe_strengths = data.get('strengths', [])
                variant.qe_issues = data.get('issues', [])

                print(f"  [QE] {variant.variant_id}: {variant.qe_score}/100 - {variant.qe_feedback[:60]}...")
                return variant
        except LLMError as e:
            print(f"  [QE] LLM error: {e}")

        # Fallback: assign default score
        variant.qe_score = 60
        variant.qe_feedback = "Evaluation completed with default score"
        return variant

    def evaluate_batch(self, variants: List[PostVariant]) -> List[PostVariant]:
        """
        Evaluate multiple variants with rate limiting.

        Args:
            variants: List of PostVariants to evaluate

        Returns:
            List of PostVariants with updated qe_* fields
        """
        print(f"[QEAgent] Evaluating {len(variants)} variants...")
        evaluated = []

        for i, variant in enumerate(variants):
            print(f"[QE] Evaluating {i+1}/{len(variants)} ({variant.hook_style})...")
            evaluated.append(self.evaluate_post(variant))
            time.sleep(0.5)  # Rate limiting

        return evaluated


# ---------------------------------------------------------------------------
# Strands tool interface — callable by an orchestrating Agent
# ---------------------------------------------------------------------------

_qe_agent_instance = QEAgent()


@tool
def evaluate_post_quality(content: str) -> dict:
    """Evaluate a Brandon news bulletin against the source-grounded usefulness rubric."""
    from .post_variant import PostVariant as _PV
    v = _PV(variant_id="tool_call", content=content, hook_style="unknown")
    result = _qe_agent_instance.evaluate_post(v)
    return QEResult(
        score=result.qe_score,
        breakdown=result.qe_breakdown or {},
        feedback=result.qe_feedback or "",
        strengths=result.qe_strengths or [],
        issues=result.qe_issues or [],
    ).model_dump()
