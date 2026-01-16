"""
QEAgent - Quality Evaluation Agent
===================================

AGENT TYPE: Evaluation Agent (Single-turn reasoning with structured output)

PURPOSE:
    Scores LinkedIn posts against best practices using an 8-criteria rubric.
    Returns detailed breakdown with strengths and issues for evolution feedback.

PROMPT ENGINEERING:
    - Uses explicit scoring criteria with point allocation
    - Requires JSON output for structured parsing
    - Includes examples of things to penalize
    - Total score: 100 points across 8 dimensions

SCORING CRITERIA (100 points total):
    1. Hook Strength (25pts): First 2-3 lines compelling?
    2. Single Focus (15pts): One clear idea, not cramming multiple tips?
    3. Mobile Format (15pts): Short paragraphs (1-3 lines), scannable?
    4. Authenticity (15pts): Personal, genuine voice, not generic?
    5. Engagement CTA (10pts): Ends with thought-provoking question?
    6. Hashtag Usage (5pts): 3-5 relevant hashtags at end?
    7. Clarity (10pts): Easy to understand, clear takeaway?
    8. Grammar (5pts): No spelling/grammar errors?

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

from .post_variant import PostVariant
from .llm_client import call_llm_json, LLMError


class QEAgent:
    """
    Quality Evaluation Agent - Scores posts against LinkedIn best practices.

    This agent implements a rubric-based evaluation system that provides
    detailed feedback for the Evolution Agent to use when improving posts.
    """

    # Scoring criteria with point allocations (updated for 2024/2025 algorithm)
    CRITERIA = {
        "hook_strength": 25,      # First 210 chars compelling?
        "length_depth": 15,       # 1500-1900 chars, insight fully developed?
        "format_whitespace": 15,  # One sentence/line, white space, short sentences?
        "authenticity": 15,       # Personal voice, not generic AI content?
        "question_cta": 10,       # Thought-provoking question at end?
        "hashtag_usage": 5,       # 3-5 relevant hashtags?
        "clarity": 10,            # Clear takeaway, easy to understand?
        "grammar": 5,             # No errors, no markdown symbols?
    }

    # The evaluation prompt template (updated with 2024/2025 algorithm research)
    EVALUATION_PROMPT = '''You are a LinkedIn content QA expert. Evaluate this post against 2024/2025 research-backed best practices.

POST ({char_count} characters):
{content}

===== SCORING CRITERIA (total 100 points) =====

1. HOOK STRENGTH (25pts):
   - First 210 characters CRITICAL (shows before "see more")
   - Would it stop someone scrolling?
   - Curiosity gap, bold claim, or value promise?

2. LENGTH & DEPTH (15pts):
   - Sweet spot: 1,500-1,900 characters
   - Under 1,000 chars = -25% reach penalty
   - Is the insight fully developed, not surface-level?

3. FORMAT & WHITE SPACE (15pts):
   - One sentence per line with blank lines between thoughts
   - Short sentences (<12 words ideal = +20% reach)
   - Uses white space liberally (+57% engagement)
   - NOT dense wall of text

4. AUTHENTICITY (15pts):
   - Personal voice, specific details
   - Not generic AI-sounding content
   - Unique angle or insight

5. QUESTION CTA (10pts):
   - Ends with thought-provoking question (+35% engagement)
   - NOT engagement bait ("Comment YES!")
   - Invites genuine discussion

6. HASHTAGS (5pts): 3-5 relevant hashtags at end

7. CLARITY (10pts): Clear takeaway, easy to understand

8. GRAMMAR/POLISH (5pts): No errors, clean formatting, no markdown symbols

===== PENALIZE HEAVILY =====
- Generic intros ("Here's my thoughts...")
- Engagement bait ("Like if you agree!")
- Dense paragraphs (no line breaks)
- Too many emojis (>3)
- Markdown symbols (** or # - LinkedIn doesn't render)
- Multiple unrelated topics crammed in
- Posts under 1,000 characters

Respond in JSON format ONLY:
{{"score": 0-100, "breakdown": {{"hook": 0-25, "length": 0-15, "format": 0-15, "authenticity": 0-15, "cta": 0-10, "hashtags": 0-5, "clarity": 0-10, "grammar": 0-5}}, "feedback": "brief specific feedback", "strengths": ["strength1"], "issues": ["issue1"]}}'''

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
