"""
EvolutionAgent - Post Evolution/Refinement Agent
=================================================

AGENT TYPE: Recursive Self-Improvement Agent (Google Co-Scientist inspired)

PURPOSE:
    Takes low-scoring posts and evolves them into improved versions using
    specific QE feedback. This implements the "Evolution" part of the
    "Generate-Debate-Evolve" paradigm from Google's AI Co-Scientist paper.

KEY INSIGHT (from Google Co-Scientist):
    Instead of just ranking hypotheses (posts), the system IMPROVES low-scoring
    ones and re-enters them in competition. This leads to better overall quality
    than simple selection.

PROMPT ENGINEERING:
    - Provides original post with its QE score
    - Lists specific ISSUES TO FIX
    - Lists STRENGTHS TO KEEP
    - Includes fresh NEWS CONTEXT to ensure relevance
    - Explicit improvement instructions

EVOLUTION PROCESS:
    1. Identify posts with QE score < threshold (default: 70)
    2. Extract QE issues and strengths from evaluation
    3. Generate improved version with specific guidance
    4. Create new PostVariant with generation++ and evolved_from set
    5. New variant re-enters QE evaluation and tournament

TRACKING:
    - generation: Increments with each evolution (1 = original, 2+ = evolved)
    - evolved_from: Points to parent variant ID
    - evolution_feedback: Human-readable description of why evolved

IMPLEMENTATION NOTES:
    - Uses Claude CLI with 90s timeout (evolution takes longer)
    - Cleans output of markdown and intro phrases
    - Falls back to original if evolution fails

USAGE:
    evolution_agent = EvolutionAgent()
    evolved = evolution_agent.evolve_post(variant, news_context)
    evolved_batch = evolution_agent.evolve_batch(variants, news_context, threshold=70)
"""

import re
import subprocess
import time
from typing import List, Optional

from .post_variant import PostVariant


class EvolutionAgent:
    """
    Evolution Agent - Refines low-scoring posts using QE feedback.

    Inspired by Google's AI Co-Scientist "evolution" mechanism where
    hypotheses are iteratively improved rather than just selected.
    """

    # Default threshold for evolution (posts below this score get evolved)
    DEFAULT_THRESHOLD = 70

    # The evolution prompt template
    EVOLUTION_PROMPT = '''You are a LinkedIn content EVOLUTION expert. Your job is to IMPROVE this post based on specific feedback.

===== ORIGINAL POST (QE Score: {qe_score}/100) =====
{content}

===== ISSUES TO FIX =====
{issues}

===== STRENGTHS TO KEEP =====
{strengths}

===== QE FEEDBACK =====
{feedback}

===== NEWS CONTEXT (reference this!) =====
{news_context}

===== YOUR TASK =====
Rewrite this post to:
1. FIX all the issues listed above
2. KEEP all the strengths
3. REFERENCE specific news/data from the context above
4. Make the hook (first 2 lines) MORE scroll-stopping
5. Keep under 1300 characters
6. NO markdown formatting (no ** for bold, no # for headers)
7. Max 2 emojis
8. End with thought-provoking question
9. 3-5 hashtags at the end

CRITICAL FORMATTING: Use 2-3 paragraph breaks (empty lines) to create visual structure. LinkedIn posts need white space for readability. The hook should be its own paragraph.

===== OUTPUT =====
Write ONLY the improved post. No explanation. Start directly with the hook:'''

    def __init__(self, threshold: int = None):
        """
        Initialize the Evolution Agent.

        Args:
            threshold: QE score below which posts are evolved (default: 70)
        """
        self.evolution_threshold = threshold or self.DEFAULT_THRESHOLD

    def _call_claude_cli(self, prompt: str, timeout: int = 90) -> Optional[str]:
        """
        Call Claude CLI for evolution.

        Args:
            prompt: The evolution prompt
            timeout: Timeout in seconds (default: 90)

        Returns:
            Claude's response string or None if failed
        """
        try:
            result = subprocess.run(
                ['claude', '-p', prompt],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except subprocess.TimeoutExpired:
            print("[EvolutionAgent] Claude CLI timed out")
            return None
        except Exception as e:
            print(f"[EvolutionAgent] Claude call failed: {e}")
            return None

    def _clean_post(self, text: str) -> str:
        """
        Clean up generated post text.

        Removes:
            - Markdown formatting (**, #, >)
            - Common intro phrases ("Here's...", "Certainly...")

        Args:
            text: Raw generated text

        Returns:
            Cleaned post text
        """
        # Remove markdown
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)

        # Remove intro phrases
        intro_patterns = [
            r"^Here'?s?\s+(a|the|my)\s+.*?:\s*\n*",
            r"^(Certainly|Sure|Of course).*?:\s*\n*",
        ]
        for pattern in intro_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        return text.strip()

    def evolve_post(self, variant: PostVariant, news_context: str) -> PostVariant:
        """
        Evolve a low-scoring post into an improved version.

        Args:
            variant: The PostVariant to evolve
            news_context: Fresh news context to reference

        Returns:
            New evolved PostVariant (or original if evolution fails)
        """
        # Format issues and strengths
        issues_text = "\n".join([f"- {issue}" for issue in variant.qe_issues]) \
            if variant.qe_issues else "- Generic content, lacks specificity"
        strengths_text = "\n".join([f"- {s}" for s in variant.qe_strengths]) \
            if variant.qe_strengths else "- None identified"

        # Build prompt
        prompt = self.EVOLUTION_PROMPT.format(
            qe_score=variant.qe_score,
            content=variant.content,
            issues=issues_text,
            strengths=strengths_text,
            feedback=variant.qe_feedback or "No specific feedback",
            news_context=news_context[:1500]  # Limit context length
        )

        result = self._call_claude_cli(prompt)

        if result:
            content = self._clean_post(result)
            if content and len(content) > 100:
                # Create evolved variant
                evolved = PostVariant(
                    variant_id=f"{variant.variant_id}_evo{variant.generation + 1}",
                    content=content,
                    hook_style=variant.hook_style,
                    generation=variant.generation + 1,
                    evolved_from=variant.variant_id,
                    evolution_feedback=self._build_evolution_feedback(variant)
                )
                print(f"  [Evolution] {variant.variant_id} -> {evolved.variant_id}")
                return evolved

        # Evolution failed, return original
        print(f"  [Evolution] Failed to evolve {variant.variant_id}, keeping original")
        return variant

    def _build_evolution_feedback(self, variant: PostVariant) -> str:
        """Build human-readable evolution feedback"""
        issues_summary = ', '.join(variant.qe_issues[:2]) if variant.qe_issues else 'generic issues'
        return f"Evolved from QE {variant.qe_score} to fix: {issues_summary}"

    def evolve_batch(
        self,
        variants: List[PostVariant],
        news_context: str,
        threshold: int = None
    ) -> List[PostVariant]:
        """
        Evolve all low-scoring variants in a batch.

        Args:
            variants: List of PostVariants
            news_context: Fresh news context
            threshold: QE score threshold (uses instance default if not specified)

        Returns:
            List of PostVariants (evolved where applicable)
        """
        if threshold is None:
            threshold = self.evolution_threshold

        print(f"[EvolutionAgent] Evolving posts below QE {threshold}...")
        result = []
        evolved_count = 0

        for variant in variants:
            if variant.qe_score < threshold and not variant.is_duplicate:
                print(f"[Evolution] Evolving {variant.variant_id} (QE: {variant.qe_score})...")
                evolved = self.evolve_post(variant, news_context)
                result.append(evolved)
                if evolved.variant_id != variant.variant_id:
                    evolved_count += 1
                time.sleep(1)  # Rate limiting
            else:
                result.append(variant)

        print(f"[EvolutionAgent] Evolution complete: {evolved_count} posts evolved")
        return result
