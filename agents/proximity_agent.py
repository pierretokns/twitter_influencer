"""
ProximityAgent - Semantic Similarity Detection Agent
=====================================================

AGENT TYPE: Deduplication Agent (Pairwise comparison)

PURPOSE:
    Detects semantically similar posts to ensure diversity in the ranking pool.
    Marks redundant posts as duplicates so they can be filtered out before
    tournament competition.

KEY INSIGHT (from Google Co-Scientist):
    The "Proximity Agent" in Google's system identifies similarities between
    hypotheses to avoid redundancy. This ensures the tournament compares
    genuinely different approaches rather than minor variations.

PROMPT ENGINEERING:
    - Provides clear similarity scale (0.0 to 1.0)
    - Examples for each range (0.0-0.3, 0.3-0.5, etc.)
    - Requests single decimal number output

SIMILARITY SCALE:
    0.0-0.3: Completely different topics/angles
    0.3-0.5: Same general topic but different focus
    0.5-0.7: Similar topic and angle, but different execution
    0.7-0.9: Very similar - same main points (DUPLICATE THRESHOLD)
    0.9-1.0: Nearly identical - redundant content

IMPLEMENTATION NOTES:
    - Calculates pairwise similarities for all variants
    - Marks lower-QE-scoring post as duplicate when similarity >= 0.7
    - Has fallback using simple word overlap if Claude call fails
    - Rate-limited with 0.3s delay between comparisons

TRACKING:
    - similarity_scores: Dict mapping other variant IDs to similarity (0-1)
    - is_duplicate: Boolean flag for filtering

USAGE:
    proximity_agent = ProximityAgent()
    variants = proximity_agent.find_duplicates(variants)  # Marks duplicates
    unique_variants = proximity_agent.filter_duplicates(variants)  # Removes them
"""

import re
import subprocess
import time
from typing import List, Optional

from .post_variant import PostVariant


class ProximityAgent:
    """
    Proximity Agent - Detects similar/duplicate posts to ensure diversity.

    Inspired by Google's AI Co-Scientist "Proximity Agent" that identifies
    similarities between hypotheses to avoid redundancy.
    """

    # Threshold above which posts are considered duplicates
    SIMILARITY_THRESHOLD = 0.7

    # The similarity prompt template
    SIMILARITY_PROMPT = '''Rate the SEMANTIC SIMILARITY between these two LinkedIn posts on a scale of 0.0 to 1.0.

POST A:
{content_a}

POST B:
{content_b}

SIMILARITY CRITERIA:
- 0.0-0.3: Completely different topics/angles
- 0.3-0.5: Same general topic but different focus
- 0.5-0.7: Similar topic and angle, but different execution
- 0.7-0.9: Very similar - same main points
- 0.9-1.0: Nearly identical - redundant content

Respond with ONLY a decimal number between 0.0 and 1.0:'''

    def __init__(self, threshold: float = None):
        """
        Initialize the Proximity Agent.

        Args:
            threshold: Similarity threshold for marking duplicates (default: 0.7)
        """
        self.similarity_threshold = threshold or self.SIMILARITY_THRESHOLD

    def _call_claude_cli(self, prompt: str) -> Optional[str]:
        """
        Call Claude CLI for similarity calculation.

        Args:
            prompt: The similarity prompt

        Returns:
            Claude's response string or None if failed
        """
        try:
            result = subprocess.run(
                ['claude', '-p', prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except subprocess.TimeoutExpired:
            print("[ProximityAgent] Claude CLI timed out")
            return None
        except Exception as e:
            print(f"[ProximityAgent] Claude call failed: {e}")
            return None

    def _fallback_similarity(self, post_a: PostVariant, post_b: PostVariant) -> float:
        """
        Calculate similarity using simple word overlap (fallback method).

        Args:
            post_a: First post
            post_b: Second post

        Returns:
            Similarity score between 0.0 and 1.0
        """
        words_a = set(post_a.content.lower().split())
        words_b = set(post_b.content.lower().split())

        if not words_a or not words_b:
            return 0.0

        overlap = len(words_a & words_b)
        return overlap / max(len(words_a), len(words_b))

    def calculate_similarity(self, post_a: PostVariant, post_b: PostVariant) -> float:
        """
        Calculate semantic similarity between two posts.

        Uses Claude CLI for semantic analysis, with word overlap as fallback.

        Args:
            post_a: First post
            post_b: Second post

        Returns:
            Similarity score between 0.0 and 1.0
        """
        prompt = self.SIMILARITY_PROMPT.format(
            content_a=post_a.content[:500],
            content_b=post_b.content[:500]
        )

        result = self._call_claude_cli(prompt)

        if result:
            try:
                # Extract first number found
                match = re.search(r'(\d+\.?\d*)', result)
                if match:
                    score = float(match.group(1))
                    return min(1.0, max(0.0, score))
            except ValueError:
                pass

        # Fallback to word overlap
        return self._fallback_similarity(post_a, post_b)

    def find_duplicates(self, variants: List[PostVariant]) -> List[PostVariant]:
        """
        Calculate pairwise similarities and mark duplicates.

        For each pair with similarity >= threshold, marks the lower-QE-scoring
        post as a duplicate.

        Args:
            variants: List of PostVariants to check

        Returns:
            Same list with similarity_scores and is_duplicate updated
        """
        print(f"[ProximityAgent] Checking proximity for {len(variants)} variants...")
        comparisons = 0

        # Calculate pairwise similarities
        for i, post_a in enumerate(variants):
            for j, post_b in enumerate(variants):
                if i >= j:  # Only check each pair once
                    continue

                similarity = self.calculate_similarity(post_a, post_b)
                comparisons += 1

                # Store similarity scores in both directions
                post_a.similarity_scores[post_b.variant_id] = similarity
                post_b.similarity_scores[post_a.variant_id] = similarity

                if similarity >= self.similarity_threshold:
                    # Mark lower-scoring post as duplicate
                    if post_a.qe_score <= post_b.qe_score:
                        post_a.is_duplicate = True
                        print(f"  [Proximity] Marked {post_a.variant_id} as duplicate of {post_b.variant_id} (sim: {similarity:.2f})")
                    else:
                        post_b.is_duplicate = True
                        print(f"  [Proximity] Marked {post_b.variant_id} as duplicate of {post_a.variant_id} (sim: {similarity:.2f})")

                time.sleep(0.3)  # Rate limiting

        duplicate_count = sum(1 for v in variants if v.is_duplicate)
        print(f"[ProximityAgent] Complete: {comparisons} comparisons, {duplicate_count} duplicates found")
        return variants

    def filter_duplicates(self, variants: List[PostVariant]) -> List[PostVariant]:
        """
        Remove duplicates from the variant list.

        Args:
            variants: List of PostVariants (with is_duplicate set)

        Returns:
            Filtered list with duplicates removed
        """
        unique = [v for v in variants if not v.is_duplicate]
        print(f"[ProximityAgent] Filtered: {len(variants)} -> {len(unique)} unique variants")
        return unique
