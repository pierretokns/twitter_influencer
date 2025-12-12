"""
PostVariantGenerator - Content Generation Agent
===============================================

AGENT TYPE: Generation Agent (Multi-variant content creation)

PURPOSE:
    Generates multiple LinkedIn post variants from AI news sources, each using
    a different "hook style" to maximize diversity and viral potential.

HOOK STYLES:
    1. curiosity_gap: Creates information gap the reader needs to close
       Example: "Most people don't realize this about GPT-5..."

    2. bold_claim: Makes a strong, attention-grabbing statement
       Example: "This will change everything we know about AI"

    3. personal_story: Opens with relatable personal narrative
       Example: "I was skeptical until I tried this myself..."

    4. data_driven: Leads with compelling statistics
       Example: "97% of developers are missing this..."

    5. practical_value: Promises immediate actionable value
       Example: "Save 10 hours per week with this approach"

GENERATION STRATEGY:
    - Each variant focuses on a DIFFERENT news item from the sources
    - Uses different hook style for each variant
    - Formats news with source attribution for credibility
    - Forces specific references to actual developments/companies

PROMPT ENGINEERING:
    - Provides full news context with source attribution
    - Assigns specific news item focus per variant
    - Includes LinkedIn best practices inline
    - Requires specific format (1300 chars, 3-5 hashtags, etc.)

IMPLEMENTATION NOTES:
    - Cycles through hook styles for diversity
    - Each variant focuses on different news item index
    - Cleans output of markdown and intro phrases
    - Rate-limited with 1s delay between generations

USAGE:
    generator = PostVariantGenerator()
    variants = generator.generate_variants(news_items, num_variants=5)
"""

import hashlib
import random
import re
import subprocess
import time
from typing import Dict, List, Optional

from .post_variant import PostVariant


# Viral hook styles with example templates
VIRAL_HOOKS = {
    "curiosity_gap": [
        "Most people don't realize this about {topic}...",
        "I discovered something unexpected about {topic}",
        "The hidden truth about {topic} that nobody talks about",
        "What {topic} really means (and why it matters)",
        "Everyone's missing this about {topic}",
    ],
    "bold_claim": [
        "This will change everything we know about {topic}",
        "{topic} is dead. Here's what's replacing it.",
        "Forget everything you learned about {topic}",
        "The {topic} revolution is here",
        "{topic} just changed the game forever",
    ],
    "personal_story": [
        "I was skeptical about {topic} until I saw this...",
        "My experience with {topic} taught me something valuable",
        "I spent 6 months studying {topic}. Here's what I learned.",
        "The {topic} lesson that changed my perspective",
        "When I first heard about {topic}, I didn't believe it",
    ],
    "data_driven": [
        "97% of professionals are missing this about {topic}",
        "New data reveals surprising truth about {topic}",
        "The numbers don't lie: {topic} is transforming everything",
        "3 statistics about {topic} that will surprise you",
        "{topic}: The data tells a different story",
    ],
    "practical_value": [
        "How to leverage {topic} (step-by-step)",
        "Save 10 hours per week with {topic}",
        "The only {topic} guide you'll ever need",
        "Stop struggling with {topic}. Do this instead.",
        "My proven {topic} framework (works every time)",
    ],
}


class PostVariantGenerator:
    """
    Generate multiple post variants for ELO ranking.

    Creates diverse posts using different hook styles, each focusing on
    a specific news item from the scraped sources.
    """

    # LinkedIn post constraints
    MAX_LENGTH = 1300
    MAX_EMOJIS = 2
    HASHTAG_RANGE = (3, 5)

    # The generation prompt template
    GENERATION_PROMPT = '''You are a LinkedIn content creator with 100K+ followers. Write a viral post about TODAY'S AI NEWS.

===== TODAY'S AI NEWS (from Twitter/X and tech blogs) =====
{news_context}

===== YOUR TASK =====
Write a LinkedIn post that:
1. MUST reference specific news from above (mention the actual development, company, or finding)
2. FOCUS primarily on news item [{focus_item}] but can reference others
3. Add YOUR unique insight, opinion, or takeaway - don't just summarize
4. Make it feel timely and current ("Just saw that...", "This week...", "Breaking:")

===== HOOK STYLE: {hook_style} =====
Example: "{hook_example}"

===== FORMAT REQUIREMENTS =====
- First 2 lines = scroll-stopping hook (this shows before "see more")
- Short paragraphs (1-3 lines each) for mobile
- Include specific details from the news (names, numbers, companies)
- End with thought-provoking question
- 3-5 hashtags at the very end
- Max 1300 characters total
- NO markdown symbols (no ** or #)
- Max 2 emojis

===== OUTPUT =====
Write ONLY the post text. No intro, no explanation. Start directly with the hook:'''

    def __init__(self):
        """Initialize the generator with available hook styles"""
        self.hook_styles = list(VIRAL_HOOKS.keys())

    def _call_claude_cli(self, prompt: str, timeout: int = 90) -> Optional[str]:
        """
        Call Claude CLI for generation.

        Args:
            prompt: The generation prompt
            timeout: Timeout in seconds

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
            print("[Generator] Claude CLI timed out")
            return None
        except Exception as e:
            print(f"[Generator] Claude call failed: {e}")
            return None

    def _clean_post(self, text: str) -> str:
        """
        Clean up generated post text.

        Removes:
            - Markdown formatting (**, #, >)
            - Common intro phrases

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

    def _format_news_context(self, news_items: List[Dict]) -> str:
        """
        Format news items with source attribution.

        Args:
            news_items: List of news dicts with 'text', 'username' keys

        Returns:
            Formatted news context string
        """
        formatted = []
        for i, item in enumerate(news_items[:20], 1):
            text = item.get('text', '')
            source = item.get('username', item.get('source_name', 'Unknown'))
            formatted.append(f"[{i}] @{source}: {text}")
        return "\n\n".join(formatted)

    def _select_hook_styles(self, num_variants: int) -> List[str]:
        """
        Select hook styles for variants, ensuring diversity.

        Args:
            num_variants: Number of variants to generate

        Returns:
            List of hook style names
        """
        selected = random.sample(self.hook_styles, min(num_variants, len(self.hook_styles)))

        # If we need more variants than styles, repeat
        while len(selected) < num_variants:
            selected.extend(random.sample(self.hook_styles, min(num_variants - len(selected), len(self.hook_styles))))

        return selected[:num_variants]

    def generate_variants(
        self,
        news_items: List[Dict],
        num_variants: int = 5
    ) -> List[PostVariant]:
        """
        Generate multiple post variants from news items.

        Each variant uses a different hook style and focuses on a different
        news item for diversity.

        Args:
            news_items: List of news item dicts
            num_variants: Number of variants to generate

        Returns:
            List of PostVariant objects
        """
        if not news_items:
            print("[Generator] No news items provided")
            return []

        variants = []
        news_context = self._format_news_context(news_items)
        selected_styles = self._select_hook_styles(num_variants)

        print(f"[Generator] Creating {num_variants} variants from {len(news_items)} news items...")

        for i, hook_style in enumerate(selected_styles):
            # Get example hook for this style
            hook_examples = VIRAL_HOOKS.get(hook_style, VIRAL_HOOKS["curiosity_gap"])
            example_hook = random.choice(hook_examples)

            # Each variant focuses on a different news item
            focus_item = (i % len(news_items)) + 1

            prompt = self.GENERATION_PROMPT.format(
                news_context=news_context,
                focus_item=focus_item,
                hook_style=hook_style.replace('_', ' ').upper(),
                hook_example=example_hook
            )

            result = self._call_claude_cli(prompt)

            if result:
                content = self._clean_post(result)
                if content and len(content) > 100:
                    # Create variant with unique ID
                    variant_id = f"v{i+1}_{hashlib.md5(content[:50].encode()).hexdigest()[:8]}"
                    variant = PostVariant(
                        variant_id=variant_id,
                        content=content,
                        hook_style=hook_style,
                    )
                    variants.append(variant)
                    print(f"  [Gen] Created {variant_id} ({hook_style}) - {len(content)} chars")

            time.sleep(1)  # Rate limiting

        print(f"[Generator] Generated {len(variants)} variants")
        return variants
