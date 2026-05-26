"""
DebateAgent - Self-Play Debate Agent
=====================================

AGENT TYPE: Self-Play Argumentative Agent (Google Co-Scientist inspired)

PURPOSE:
    Simulates debates between two candidate Brandon news bulletins to determine
    which is more useful, novel, source-grounded, and aligned with his workflow.

KEY INSIGHT (from Google Co-Scientist):
    The "self-play scientific debate" mechanism has the model argue BOTH sides
    of a comparison before deciding. This produces more nuanced evaluations
    than direct "which is better?" comparisons.

DEBATE FORMAT:
    1. Argue FOR Post A (as if you wrote it)
       - Which missed items does it surface?
       - How well does it cite primary sources?
       - Why is it useful for Brandon?

    2. Argue FOR Post B (as if you wrote it)
       - Same questions

    3. As neutral judge, declare winner based on:
       - Missed-news/novelty value (30% weight)
       - Source-grounding/citation quality (30% weight)
       - Finance/workflow relevance (20% weight)
       - Bulletin concision/actionability (20% weight)

OUTPUT STRUCTURE:
    {
        "argument_for_a": "2-3 sentence argument",
        "argument_for_b": "2-3 sentence argument",
        "winner": "A" or "B",
        "reasoning": "Why the winner is better",
        "confidence": 0.5-1.0
    }

IMPLEMENTATION NOTES:
    - Returns structured debate results for UI display
    - Confidence score affects ELO rating changes
    - Falls back to QE-based decision if parsing fails

USAGE:
    debate_agent = DebateAgent()
    result = debate_agent.conduct_debate(post_a, post_b)
    # result contains argument_for_a, argument_for_b, winner, reasoning, confidence
"""

from typing import Dict

try:
    from strands import tool
except ImportError:
    def tool(func=None, **_kwargs):
        if func is None:
            return lambda wrapped: wrapped
        return func

from .post_variant import PostVariant
from .llm_client import call_llm_json, LLMError
from .models import DebateResult


class DebateAgent:
    """
    Debate Agent - Simulates self-play debates between posts.

    Inspired by Google's AI Co-Scientist "self-play scientific debate"
    mechanism where the model argues both sides before deciding.
    """

    # Judging weights (should sum to 100%)
    JUDGING_WEIGHTS = {
        "missed_news_novelty": 30,
        "source_grounding": 30,
        "finance_workflow_relevance": 20,
        "concision_actionability": 20,
    }

    # The debate prompt template
    DEBATE_PROMPT = '''You are moderating a DEBATE between two candidate Brandon AI-news bulletins.

===== POST A ({style_a}) =====
{content_a}

===== POST B ({style_b}) =====
{content_b}

===== DEBATE FORMAT =====
First, argue FOR Post A (as if you wrote it):
- Which missed model releases, YouTube/video items, papers, repos, conference/CFP deadlines, Boston/NYC events, eval/tooling changes, or primary-source updates does it surface?
- How well are its factual claims cited and grounded?
- Why is it useful for Brandon's finance/workflow context?

Then, argue FOR Post B (as if you wrote it):
- Which missed model releases, YouTube/video items, papers, repos, conference/CFP deadlines, Boston/NYC events, eval/tooling changes, or primary-source updates does it surface?
- How well are its factual claims cited and grounded?
- Why is it useful for Brandon's finance/workflow context?

Finally, as a neutral judge, declare a WINNER based on:
1. Missed-news and novelty value (30% weight)
2. Source-grounding and citation quality (30% weight)
3. Finance/workflow relevance when supported (20% weight)
4. Bulletin concision and actionability (20% weight)

Penalize narrative filler, influencer hooks, hashtags, engagement questions, and generic "AI is transforming X" claims.

Respond in JSON format:
{{"argument_for_a": "2-3 sentence argument", "argument_for_b": "2-3 sentence argument", "winner": "A" or "B", "reasoning": "Why the winner is better", "confidence": 0.5-1.0}}'''

    def __init__(self):
        """Initialize the Debate Agent"""
        pass

    def conduct_debate(self, post_a: PostVariant, post_b: PostVariant) -> Dict:
        """
        Simulate a debate between two posts.

        The agent argues FOR each post before deciding, producing richer
        reasoning than direct comparison.

        Args:
            post_a: First PostVariant
            post_b: Second PostVariant

        Returns:
            Dict with debate results:
            {
                "post_a_id": str,
                "post_b_id": str,
                "argument_for_a": str,
                "argument_for_b": str,
                "winner": "A" or "B",
                "reasoning": str,
                "confidence": float
            }
        """
        prompt = self.DEBATE_PROMPT.format(
            style_a=post_a.hook_style,
            content_a=post_a.content,
            style_b=post_b.hook_style,
            content_b=post_b.content
        )

        try:
            data = call_llm_json(prompt, timeout=60)
            if data:
                return {
                    "post_a_id": post_a.variant_id,
                    "post_b_id": post_b.variant_id,
                    "argument_for_a": data.get("argument_for_a", ""),
                    "argument_for_b": data.get("argument_for_b", ""),
                    "winner": data.get("winner", "A"),
                    "reasoning": data.get("reasoning", ""),
                    "confidence": data.get("confidence", 0.5)
                }
        except LLMError as e:
            print(f"[DebateAgent] LLM error: {e}")

        # Fallback: decide based on QE scores
        print("[DebateAgent] Using fallback (QE scores)")
        winner = "A" if post_a.qe_score >= post_b.qe_score else "B"
        return {
            "post_a_id": post_a.variant_id,
            "post_b_id": post_b.variant_id,
            "argument_for_a": "Has potential based on content structure",
            "argument_for_b": "Has potential based on content structure",
            "winner": winner,
            "reasoning": f"Decided by QE scores ({post_a.qe_score} vs {post_b.qe_score})",
            "confidence": 0.5
        }

    def format_debate_summary(self, debate: Dict) -> str:
        """
        Format a debate result as human-readable summary.

        Args:
            debate: Debate result dict

        Returns:
            Formatted string summary
        """
        winner_id = debate["post_a_id"] if debate["winner"] == "A" else debate["post_b_id"]
        return f"""
DEBATE: {debate['post_a_id']} vs {debate['post_b_id']}
---
FOR A: {debate['argument_for_a']}
FOR B: {debate['argument_for_b']}
---
WINNER: {winner_id} ({debate['confidence']*100:.0f}% confidence)
REASON: {debate['reasoning']}
"""


# ---------------------------------------------------------------------------
# Strands tool interface
# ---------------------------------------------------------------------------

_debate_agent_instance = DebateAgent()


@tool
def debate_posts(post_a_content: str, post_b_content: str, post_a_style: str = "unknown", post_b_style: str = "unknown") -> dict:
    """Run a self-play debate between two Brandon news bulletins and return the more useful candidate."""
    from .post_variant import PostVariant as _PV
    a = _PV(variant_id="A", content=post_a_content, hook_style=post_a_style)
    b = _PV(variant_id="B", content=post_b_content, hook_style=post_b_style)
    result = _debate_agent_instance.conduct_debate(a, b)
    return DebateResult(
        argument_for_a=result.get("argument_for_a", ""),
        argument_for_b=result.get("argument_for_b", ""),
        winner=result.get("winner", "A"),
        reasoning=result.get("reasoning", ""),
        confidence=result.get("confidence", 0.5),
    ).model_dump()
