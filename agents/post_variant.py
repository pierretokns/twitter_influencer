"""
PostVariant - Data class for LinkedIn post variants
====================================================

This dataclass represents a single LinkedIn post variant that flows through
the multi-agent ranking pipeline.

TRACKING FIELDS:
    - Basic: content, hook_style, variant_id
    - QE Evaluation: qe_score, qe_feedback, qe_breakdown, strengths, issues
    - ELO Tournament: elo_rating, matches_played, wins, losses
    - Evolution: generation, evolved_from, evolution_feedback
    - Debate: debate_history (records of wins/losses with reasoning)
    - Proximity: similarity_scores, is_duplicate

LIFECYCLE:
    1. Created by PostVariantGenerator with generation=1
    2. Scored by QEAgent (qe_score, qe_feedback, etc.)
    3. Checked by ProximityAgent (similarity_scores, is_duplicate)
    4. If low score: evolved by EvolutionAgent (generation++, evolved_from set)
    5. Competed in ELORanker tournament (debate_history updated)
    6. Final ranking by elo_rating
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PostVariant:
    """
    A single LinkedIn post variant for multi-agent ranking.

    This dataclass tracks a post through all stages of the ranking pipeline:
    generation, evaluation, evolution, and tournament competition.
    """
    # Core content
    variant_id: str
    content: str
    hook_style: str  # e.g., "curiosity_gap", "bold_claim", "data_driven"

    # ELO tournament tracking
    elo_rating: float = 1000.0
    matches_played: int = 0
    wins: int = 0
    losses: int = 0

    # QE Agent evaluation
    qe_score: float = 0.0
    qe_feedback: str = ""
    qe_breakdown: Dict = field(default_factory=dict)  # {hook: 20, focus: 15, ...}
    qe_strengths: List[str] = field(default_factory=list)
    qe_issues: List[str] = field(default_factory=list)

    # Evolution tracking (Google Co-Scientist inspired)
    generation: int = 1  # Which evolution generation (1 = original, 2+ = evolved)
    evolved_from: Optional[str] = None  # Parent variant ID if evolved
    evolution_feedback: str = ""  # Why/how it was evolved

    # Debate tracking
    debate_history: List[Dict] = field(default_factory=list)
    # Each entry: {opponent: str, reasoning: str, won: bool, confidence: float}

    # Proximity tracking
    similarity_scores: Dict[str, float] = field(default_factory=dict)  # {other_id: 0.0-1.0}
    is_duplicate: bool = False

    # Source attribution (Document Page Finder - Liang et al. 2024)
    source_attributions: List[Dict] = field(default_factory=list)  # [{idx, score, is_ref}]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.variant_id,
            "content": self.content,
            "hook_style": self.hook_style,
            "elo": self.elo_rating,
            "qe_score": self.qe_score,
            "qe_feedback": self.qe_feedback,
            "qe_breakdown": self.qe_breakdown,
            "qe_strengths": self.qe_strengths,
            "qe_issues": self.qe_issues,
            "generation": self.generation,
            "evolved_from": self.evolved_from,
            "evolution_feedback": self.evolution_feedback,
            "wins": self.wins,
            "losses": self.losses,
            "matches": self.matches_played,
            "debate_history": self.debate_history,
            "is_duplicate": self.is_duplicate,
            "source_attributions": self.source_attributions,
        }

    @property
    def is_evolved(self) -> bool:
        """Check if this variant is an evolution of another"""
        return self.generation > 1

    @property
    def win_rate(self) -> float:
        """Calculate win rate as percentage"""
        if self.matches_played == 0:
            return 0.0
        return (self.wins / self.matches_played) * 100
