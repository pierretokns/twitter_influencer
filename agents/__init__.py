"""
LinkedIn Post Ranking Agents
============================

A multi-agent system inspired by Google's AI Co-Scientist for generating
and ranking viral LinkedIn posts through tournament-based evolution.

ARCHITECTURE:
    This system implements a "Generate-Debate-Evolve" approach:

    1. GENERATION PHASE
       - PostVariantGenerator creates multiple post variants using different hook styles
       - Each variant focuses on a specific news item from the scraped sources

    2. EVALUATION PHASE
       - QEAgent scores posts against LinkedIn best practices (0-100)
       - ProximityAgent detects duplicate/similar posts to ensure diversity

    3. EVOLUTION PHASE (Google Co-Scientist inspired)
       - EvolutionAgent refines low-scoring posts using specific QE feedback
       - Evolved posts re-enter the tournament with improved content

    4. TOURNAMENT PHASE
       - DebateAgent conducts self-play debates between posts
       - ELORanker uses debate outcomes to update ELO ratings
       - Multiple rounds ensure robust ranking

AGENT TYPES:
    - QEAgent: Quality Evaluation - scores against best practices
    - EvolutionAgent: Improves posts based on feedback (recursive self-improvement)
    - ProximityAgent: Detects semantic similarity between posts
    - DebateAgent: Self-play debates for pairwise comparison
    - ELORanker: Tournament management with confidence-adjusted ratings
    - PostVariantGenerator: Initial content generation

AGENTIC PATTERN:
    This is a "Multi-Agent Collaborative System" using:
    - Tool-using agents (Claude CLI for reasoning)
    - Structured output (JSON responses for scoring/decisions)
    - Recursive refinement (evolution loops)
    - Self-play (posts argue against each other in debates)
    - ELO ranking (borrowed from game theory/chess)

References:
    - Google AI Co-Scientist: https://arxiv.org/abs/2502.18864
    - ELO Rating System: https://en.wikipedia.org/wiki/Elo_rating_system
"""

from .post_variant import PostVariant
from .qe_agent import QEAgent
from .evolution_agent import EvolutionAgent
from .proximity_agent import ProximityAgent
from .debate_agent import DebateAgent
from .elo_ranker import ELORanker
from .variant_generator import PostVariantGenerator

__all__ = [
    'PostVariant',
    'QEAgent',
    'EvolutionAgent',
    'ProximityAgent',
    'DebateAgent',
    'ELORanker',
    'PostVariantGenerator',
]
