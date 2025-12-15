"""
ELORanker - ELO-based Tournament Ranking System
================================================

AGENT TYPE: Tournament Manager (Orchestrates DebateAgent for comparisons)

PURPOSE:
    Manages the ELO tournament where posts compete head-to-head through debates.
    Uses the classic ELO rating system (from chess) with confidence-adjusted
    K-factors based on debate certainty.

ELO RATING SYSTEM:
    - All posts start at 1000 rating
    - After each match, ratings update based on:
        1. Expected outcome (based on current ratings)
        2. Actual outcome (from debate)
        3. Confidence adjustment (higher confidence = bigger swings)

    Formula:
        expected = 1 / (1 + 10^((opponent_rating - my_rating) / 400))
        new_rating = old_rating + K * (actual - expected)

    Where:
        - K = 32 * (0.5 + confidence)  # Higher confidence = bigger K
        - actual = 1 for win, 0 for loss

CONFIDENCE ADJUSTMENT (Google Co-Scientist inspired):
    The debate agent returns a confidence score (0.5-1.0). Higher confidence
    debates produce larger rating changes, similar to how Google's system
    weights more certain comparisons more heavily.

TOURNAMENT FORMAT:
    - Multiple rounds of random pairings
    - Each round: shuffle variants, pair up, run debates
    - Update ELO ratings after each debate
    - Track debate history for UI display

IMPLEMENTATION NOTES:
    - Uses DebateAgent for actual comparisons
    - Stores all debates for transparency
    - Backwards-compatible compare_posts() method

USAGE:
    ranker = ELORanker()
    ranked = ranker.run_tournament(variants, rounds=3)
    # ranked is sorted by ELO rating (highest first)
"""

import random
import time
from typing import Dict, List, Optional, Tuple

from .post_variant import PostVariant
from .debate_agent import DebateAgent


class ELORanker:
    """
    ELO-based ranking system with debate integration.

    Combines the classic ELO rating system with Google Co-Scientist inspired
    confidence-adjusted rating changes.
    """

    # Standard ELO K-factor (base value, adjusted by confidence)
    K_FACTOR = 32

    # Starting rating for all posts
    INITIAL_RATING = 1000.0

    def __init__(self):
        """Initialize the ELO Ranker with a Debate Agent"""
        self.debate_agent = DebateAgent()
        self.all_debates: List[Dict] = []  # Track all debate results

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player A using ELO formula.

        Args:
            rating_a: ELO rating of player A
            rating_b: ELO rating of player B

        Returns:
            Expected probability of A winning (0-1)
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def _update_ratings(
        self,
        winner: PostVariant,
        loser: PostVariant,
        confidence: float = 0.5
    ):
        """
        Update ELO ratings after a match.

        Uses confidence-adjusted K-factor: higher confidence debates
        produce larger rating changes.

        Args:
            winner: The winning PostVariant
            loser: The losing PostVariant
            confidence: Debate confidence (0.5-1.0)
        """
        # Adjust K-factor by confidence (higher confidence = bigger swings)
        k = self.K_FACTOR * (0.5 + confidence)

        # Calculate expected scores
        expected_winner = self._expected_score(winner.elo_rating, loser.elo_rating)
        expected_loser = self._expected_score(loser.elo_rating, winner.elo_rating)

        # Update ratings
        winner.elo_rating += k * (1 - expected_winner)
        loser.elo_rating += k * (0 - expected_loser)

        # Update match statistics
        winner.matches_played += 1
        loser.matches_played += 1
        winner.wins += 1
        loser.losses += 1

    def compare_posts_with_debate(
        self,
        post_a: PostVariant,
        post_b: PostVariant
    ) -> Tuple[Optional[PostVariant], Dict]:
        """
        Compare two posts using full debate mechanism.

        Args:
            post_a: First PostVariant
            post_b: Second PostVariant

        Returns:
            Tuple of (winner PostVariant, debate result dict)
        """
        # Conduct debate
        debate_result = self.debate_agent.conduct_debate(post_a, post_b)

        # Determine winner
        winner_letter = debate_result.get("winner", "A")
        if winner_letter == "A":
            winner = post_a
            loser = post_b
        else:
            winner = post_b
            loser = post_a

        # Record debate in winner's history
        winner_record = {
            "opponent": loser.variant_id,
            "reasoning": debate_result.get("reasoning", ""),
            "won": True,
            "confidence": debate_result.get("confidence", 0.5)
        }
        winner.debate_history.append(winner_record)

        # Record debate in loser's history
        loser_argument_key = "argument_for_b" if winner_letter == "A" else "argument_for_a"
        loser_record = {
            "opponent": winner.variant_id,
            "reasoning": debate_result.get(loser_argument_key, ""),
            "won": False,
            "confidence": debate_result.get("confidence", 0.5)
        }
        loser.debate_history.append(loser_record)

        # Store full debate for UI
        self.all_debates.append(debate_result)

        return winner, debate_result

    def compare_posts(
        self,
        post_a: PostVariant,
        post_b: PostVariant
    ) -> Optional[PostVariant]:
        """
        Compare two posts and return the winner (backwards compatible).

        Args:
            post_a: First PostVariant
            post_b: Second PostVariant

        Returns:
            The winning PostVariant
        """
        winner, _ = self.compare_posts_with_debate(post_a, post_b)
        return winner

    def run_tournament(
        self,
        variants: List[PostVariant],
        rounds: int = 3,
        callback=None
    ) -> Tuple[List[PostVariant], List[Dict]]:
        """
        Run ELO tournament with multiple rounds of comparisons.

        Args:
            variants: List of PostVariants to compete
            rounds: Number of tournament rounds
            callback: Optional callback(round, match, debate) for progress updates

        Returns:
            Tuple of (ranked variants list, all debates list)
        """
        if len(variants) < 2:
            return variants, []

        print(f"[ELORanker] Starting tournament: {len(variants)} variants, {rounds} rounds")
        self.all_debates = []  # Reset debates

        for round_num in range(1, rounds + 1):
            print(f"[ELO] Round {round_num}/{rounds}...")

            # Shuffle for random pairings
            shuffled = variants.copy()
            random.shuffle(shuffled)

            match_num = 0
            for i in range(0, len(shuffled) - 1, 2):
                post_a = shuffled[i]
                post_b = shuffled[i + 1]
                match_num += 1

                # Run debate
                winner, debate_result = self.compare_posts_with_debate(post_a, post_b)
                confidence = debate_result.get("confidence", 0.5)

                # Update ratings
                loser = post_b if winner == post_a else post_a
                self._update_ratings(winner, loser, confidence)

                print(f"  [Match {match_num}] {winner.variant_id} defeats {loser.variant_id} ({confidence*100:.0f}% conf)")

                # Optional progress callback
                if callback:
                    callback(round_num, match_num, debate_result)

            time.sleep(0.5)

        # Sort by ELO rating (highest first)
        variants.sort(key=lambda x: x.elo_rating, reverse=True)

        print(f"[ELORanker] Tournament complete!")
        for i, v in enumerate(variants[:3]):
            print(f"  #{i+1}: {v.variant_id} (ELO: {v.elo_rating:.0f}, W{v.wins}-L{v.losses})")

        return variants, self.all_debates

    def get_tournament_stats(self, variants: List[PostVariant]) -> Dict:
        """
        Get tournament statistics for display.

        Args:
            variants: List of PostVariants after tournament

        Returns:
            Dict with tournament stats
        """
        return {
            "total_debates": len(self.all_debates),
            "variants_count": len(variants),
            "top_rating": variants[0].elo_rating if variants else 0,
            "bottom_rating": variants[-1].elo_rating if variants else 0,
            "rating_spread": (variants[0].elo_rating - variants[-1].elo_rating) if variants else 0,
            "avg_confidence": sum(d.get("confidence", 0.5) for d in self.all_debates) / len(self.all_debates) if self.all_debates else 0,
        }
