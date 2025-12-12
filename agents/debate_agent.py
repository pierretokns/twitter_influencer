"""
DebateAgent - Self-Play Debate Agent
=====================================

AGENT TYPE: Self-Play Argumentative Agent (Google Co-Scientist inspired)

PURPOSE:
    Simulates debates between two posts to determine which is more likely to
    go viral on LinkedIn. Unlike simple comparison, this agent argues FOR
    each post before making a decision, producing richer reasoning.

KEY INSIGHT (from Google Co-Scientist):
    The "self-play scientific debate" mechanism has the model argue BOTH sides
    of a comparison before deciding. This produces more nuanced evaluations
    than direct "which is better?" comparisons.

DEBATE FORMAT:
    1. Argue FOR Post A (as if you wrote it)
       - Why is its hook superior?
       - What unique value does it provide?
       - Why will it get more engagement?

    2. Argue FOR Post B (as if you wrote it)
       - Same questions

    3. As neutral judge, declare winner based on:
       - Hook stopping power (40% weight)
       - Authenticity & uniqueness (25% weight)
       - Clear value/insight (20% weight)
       - Engagement potential (15% weight)

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

import json
import subprocess
from typing import Dict, Optional

from .post_variant import PostVariant


class DebateAgent:
    """
    Debate Agent - Simulates self-play debates between posts.

    Inspired by Google's AI Co-Scientist "self-play scientific debate"
    mechanism where the model argues both sides before deciding.
    """

    # Judging weights (should sum to 100%)
    JUDGING_WEIGHTS = {
        "hook_stopping_power": 40,
        "authenticity_uniqueness": 25,
        "clear_value_insight": 20,
        "engagement_potential": 15,
    }

    # The debate prompt template
    DEBATE_PROMPT = '''You are moderating a DEBATE between two LinkedIn posts competing for virality.

===== POST A ({style_a}) =====
{content_a}

===== POST B ({style_b}) =====
{content_b}

===== DEBATE FORMAT =====
First, argue FOR Post A (as if you wrote it):
- Why is its hook superior?
- What unique value does it provide?
- Why will it get more engagement?

Then, argue FOR Post B (as if you wrote it):
- Why is its hook superior?
- What unique value does it provide?
- Why will it get more engagement?

Finally, as a neutral judge, declare a WINNER based on:
1. Hook stopping power (40% weight)
2. Authenticity & uniqueness (25% weight)
3. Clear value/insight (20% weight)
4. Engagement potential (15% weight)

Respond in JSON format:
{{"argument_for_a": "2-3 sentence argument", "argument_for_b": "2-3 sentence argument", "winner": "A" or "B", "reasoning": "Why the winner is better", "confidence": 0.5-1.0}}'''

    def __init__(self):
        """Initialize the Debate Agent"""
        pass

    def _call_claude_cli(self, prompt: str) -> Optional[str]:
        """
        Call Claude CLI for debate.

        Args:
            prompt: The debate prompt

        Returns:
            Claude's response string or None if failed
        """
        try:
            result = subprocess.run(
                ['claude', '-p', prompt],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except subprocess.TimeoutExpired:
            print("[DebateAgent] Claude CLI timed out")
            return None
        except Exception as e:
            print(f"[DebateAgent] Claude call failed: {e}")
            return None

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

        result = self._call_claude_cli(prompt)

        if result:
            try:
                # Extract JSON from response
                start = result.find('{')
                end = result.rfind('}') + 1
                if start >= 0 and end > start:
                    data = json.loads(result[start:end])
                    return {
                        "post_a_id": post_a.variant_id,
                        "post_b_id": post_b.variant_id,
                        "argument_for_a": data.get("argument_for_a", ""),
                        "argument_for_b": data.get("argument_for_b", ""),
                        "winner": data.get("winner", "A"),
                        "reasoning": data.get("reasoning", ""),
                        "confidence": data.get("confidence", 0.5)
                    }
            except json.JSONDecodeError as e:
                print(f"[DebateAgent] JSON parse error: {e}")

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
