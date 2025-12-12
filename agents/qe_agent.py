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

import json
import subprocess
import time
from typing import List, Optional

from .post_variant import PostVariant


class QEAgent:
    """
    Quality Evaluation Agent - Scores posts against LinkedIn best practices.

    This agent implements a rubric-based evaluation system that provides
    detailed feedback for the Evolution Agent to use when improving posts.
    """

    # Scoring criteria with point allocations
    CRITERIA = {
        "hook_strength": 25,      # First 2-3 lines compelling?
        "single_focus": 15,       # One clear idea?
        "mobile_format": 15,      # Short paragraphs, scannable?
        "authenticity": 15,       # Personal, genuine voice?
        "engagement_cta": 10,     # Good question at end?
        "hashtag_usage": 5,       # 3-5 relevant hashtags?
        "clarity": 10,            # Easy to understand?
        "grammar": 5,             # No errors?
    }

    # The evaluation prompt template
    EVALUATION_PROMPT = '''You are a LinkedIn content QA expert. Evaluate this post against best practices.

POST:
{content}

SCORING CRITERIA (total 100 points):
1. HOOK STRENGTH (25pts): First 2-3 lines compelling? Would it stop scrolling?
2. SINGLE FOCUS (15pts): One clear idea, not cramming multiple tips?
3. MOBILE FORMAT (15pts): Short paragraphs (1-3 lines), scannable?
4. AUTHENTICITY (15pts): Personal, genuine voice, not generic?
5. ENGAGEMENT CTA (10pts): Ends with thought-provoking question (not engagement bait)?
6. HASHTAG USAGE (5pts): 3-5 relevant hashtags at end?
7. CLARITY (10pts): Easy to understand, clear takeaway?
8. GRAMMAR (5pts): No spelling/grammar errors?

THINGS TO PENALIZE:
- Generic intros ("Here's my thoughts on...")
- Engagement bait ("Comment YES if you agree!")
- Dense paragraphs
- Too many emojis (>3)
- Markdown symbols (** or #)
- Multiple topics crammed in

Respond in JSON format ONLY:
{{"score": 0-100, "breakdown": {{"hook": 0-25, "focus": 0-15, "format": 0-15, "authenticity": 0-15, "cta": 0-10, "hashtags": 0-5, "clarity": 0-10, "grammar": 0-5}}, "feedback": "brief specific feedback", "strengths": ["strength1"], "issues": ["issue1"]}}'''

    def __init__(self):
        """Initialize the QE Agent"""
        pass

    def _call_claude_cli(self, prompt: str) -> Optional[str]:
        """
        Call Claude CLI for evaluation.

        Args:
            prompt: The evaluation prompt with post content

        Returns:
            Claude's response string or None if failed
        """
        try:
            result = subprocess.run(
                ['claude', '-p', prompt],
                capture_output=True,
                text=True,
                timeout=45
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except subprocess.TimeoutExpired:
            print("[QEAgent] Claude CLI timed out")
            return None
        except Exception as e:
            print(f"[QEAgent] Claude call failed: {e}")
            return None

    def evaluate_post(self, variant: PostVariant) -> PostVariant:
        """
        Evaluate a single post variant and update its QE fields.

        Args:
            variant: The PostVariant to evaluate

        Returns:
            The same PostVariant with updated qe_* fields
        """
        prompt = self.EVALUATION_PROMPT.format(content=variant.content)
        result = self._call_claude_cli(prompt)

        if result:
            try:
                # Extract JSON from response
                start = result.find('{')
                end = result.rfind('}') + 1
                if start >= 0 and end > start:
                    data = json.loads(result[start:end])

                    # Update variant with evaluation results
                    variant.qe_score = data.get('score', 50)
                    variant.qe_feedback = data.get('feedback', '')
                    variant.qe_breakdown = data.get('breakdown', {})
                    variant.qe_strengths = data.get('strengths', [])
                    variant.qe_issues = data.get('issues', [])

                    print(f"  [QE] {variant.variant_id}: {variant.qe_score}/100 - {variant.qe_feedback[:60]}...")
                    return variant
            except json.JSONDecodeError as e:
                print(f"  [QE] JSON parse error: {e}")

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
