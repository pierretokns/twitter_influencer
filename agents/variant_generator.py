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
import time
from typing import Dict, List, Tuple

from .post_variant import PostVariant
from .llm_client import call_llm, LLMError


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

    # LinkedIn post constraints (based on 2024/2025 algorithm research)
    # Sweet spot for text posts: 1,800-2,100 chars. Below 1000 = -25% reach
    MAX_LENGTH = 1900
    MIN_LENGTH = 1200  # Avoid the -25% penalty for short posts
    MAX_EMOJIS = 3  # 1-3 emojis = +25% engagement
    HASHTAG_RANGE = (3, 5)

    # The generation prompt template (research-backed formatting)
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

===== FORMAT REQUIREMENTS (research-backed for 2024/2025 algorithm) =====

STRUCTURE:
- First 210 characters = CRITICAL hook (this shows before "see more" button)
- One sentence per line with blank lines between paragraphs (gives eyes room to rest)
- Short sentences under 12 words perform best (+20% reach)
- 2-3 sentences max per paragraph block
- End with a thought-provoking QUESTION (+35% engagement, +20% reach)

LENGTH:
- Target 1,500-1,900 characters (sweet spot for text posts)
- Posts under 1,000 chars get -25% reach penalty
- Use the space to develop your insight fully

FORMATTING:
- NO markdown symbols (no ** or # - LinkedIn doesn't render them)
- Use blank lines liberally for white space (+57% engagement)
- 1-3 emojis strategically placed (+25% engagement)
- 3-5 hashtags at the very end

===== OUTPUT =====
Write ONLY the post text. No intro, no explanation. Start directly with the hook:'''

    def __init__(self):
        """Initialize the generator with available hook styles"""
        self.hook_styles = list(VIRAL_HOOKS.keys())

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

            try:
                result = call_llm(prompt, timeout=90)
            except LLMError as e:
                print(f"[Generator] LLM error: {e}")
                result = None

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

    def annotate_sources_with_attribution(
        self,
        content: str,
        news_items: List[Dict],
        threshold: float = 0.2
    ) -> List[Dict]:
        """
        Annotate news items with attribution scores based on generated content.

        Uses Document Page Finder (TF-IDF similarity) to identify which sources
        were actually referenced in the generated post.

        Args:
            content: The generated post content
            news_items: List of news item dicts used for generation
            threshold: Minimum similarity score to mark as referenced

        Returns:
            List of news items with added 'is_referenced' and 'attribution_score' keys
        """
        try:
            from agents.hybrid_retriever import find_supporting_sources

            # Get source texts for TF-IDF matching
            source_texts = [item.get('text', '')[:500] for item in news_items]

            # Find which sources are referenced
            citations = find_supporting_sources(content, source_texts, threshold=threshold)

            # Create a map of source index to score
            citation_scores = {idx: score for idx, score in citations}

            # Annotate each item
            annotated = []
            for i, item in enumerate(news_items):
                item_copy = dict(item)
                if i in citation_scores:
                    item_copy['is_referenced'] = True
                    item_copy['attribution_score'] = citation_scores[i]
                else:
                    item_copy['is_referenced'] = False
                    item_copy['attribution_score'] = 0.0
                annotated.append(item_copy)

            # Log attribution results
            referenced_count = len(citations)
            print(f"[Generator] Source attribution: {referenced_count}/{len(news_items)} sources referenced")
            if referenced_count > 0:
                top_sources = sorted(citations, key=lambda x: x[1], reverse=True)[:3]
                for idx, score in top_sources:
                    source = news_items[idx].get('username', news_items[idx].get('source_name', 'Unknown'))
                    print(f"  - @{source}: {score:.2f} similarity")

            return annotated

        except ImportError:
            print("[Generator] hybrid_retriever not available, skipping attribution")
            return news_items
        except Exception as e:
            print(f"[Generator] Attribution failed: {e}")
            return news_items

    def insert_citation_markers(
        self,
        content: str,
        annotated_sources: List[Dict],
        max_citations: int = 5
    ) -> Tuple[str, List[Dict]]:
        """
        Insert inline [1], [2] citation markers into post content.

        Uses sentence-level TF-IDF to identify which sentences reference
        which sources, then inserts markers at sentence boundaries.

        Args:
            content: The generated post content
            annotated_sources: Sources with is_referenced and attribution_score
            max_citations: Max inline markers (rest stay unlabeled in panel)

        Returns:
            Tuple of:
            - Content with [1], [2] markers inserted
            - Sources list with citation_number field added
        """
        from agents.hybrid_retriever import find_sentence_source_mapping

        # Get only referenced sources, sorted by score for selection
        cited = [
            (i, s) for i, s in enumerate(annotated_sources)
            if s.get('is_referenced')
        ]
        if not cited:
            return content, annotated_sources

        # Sort by score to pick best sources, limit to max_citations
        cited.sort(key=lambda x: x[1].get('attribution_score', 0), reverse=True)
        cited = cited[:max_citations]

        # Split content into paragraphs first (preserve structure)
        # Use regex to split on 2+ newlines OR single newlines
        paragraphs = re.split(r'(\n\n+|\n)', content)

        # Build flat list of sentences with paragraph boundary info
        # Each entry: (sentence_text, is_paragraph_end)
        all_sentences = []
        paragraph_breaks = []  # Track where paragraph breaks occur

        for part in paragraphs:
            if not part:
                continue
            if re.match(r'^\n+$', part):
                # This is a paragraph separator - mark the last sentence
                paragraph_breaks.append(len(all_sentences) - 1 if all_sentences else -1)
                continue

            # Split paragraph into sentences
            sentence_pattern = r'(?<=[.!?])\s+'
            sentences = re.split(sentence_pattern, part)
            sentences = [s.strip() for s in sentences if s.strip()]
            all_sentences.extend(sentences)

        if not all_sentences:
            return content, annotated_sources

        # Get source texts for matching
        source_texts = [s.get('text', '')[:500] for _, s in cited]

        # Find which sentence best matches which source
        # Returns: {sentence_idx: cited_source_idx}
        # Uses entity overlap validation to ensure citations are relevant
        mapping = find_sentence_source_mapping(all_sentences, source_texts, threshold=0.25)

        # First pass: find order of appearance in text
        # Track which sources appear and in what sentence order
        appearance_order = []  # List of (sentence_idx, cited_source_idx)
        for sent_idx in range(len(all_sentences)):
            if sent_idx in mapping:
                cited_source_idx = mapping[sent_idx]
                if cited_source_idx not in [x[1] for x in appearance_order]:
                    appearance_order.append((sent_idx, cited_source_idx))

        # Assign citation numbers in order of appearance (1, 2, 3...)
        # Also extract best matching quote for each cited source
        cited_idx_to_citation = {}
        for citation_num, (sent_idx, cited_source_idx) in enumerate(appearance_order, 1):
            cited_idx_to_citation[cited_source_idx] = citation_num
            # Update the source with its citation number
            orig_idx, src = cited[cited_source_idx]
            src['citation_number'] = citation_num
            if not src.get('source_url'):
                src['source_url'] = self._build_source_url(src)

            # Extract best matching quote from source content
            sentence = all_sentences[sent_idx] if sent_idx < len(all_sentences) else ''
            self._extract_citation_quote(src, sentence)

        # Insert markers into sentences
        marked_sentences = []
        used_citations = set()

        for sent_idx, sentence in enumerate(all_sentences):
            if sent_idx in mapping:
                cited_source_idx = mapping[sent_idx]
                citation_num = cited_idx_to_citation.get(cited_source_idx)
                if citation_num and citation_num not in used_citations:
                    # Add citation marker at end of sentence
                    if sentence and sentence[-1] in '.!?':
                        sentence = sentence[:-1] + f'[{citation_num}]' + sentence[-1]
                    else:
                        sentence = sentence + f'[{citation_num}]'
                    used_citations.add(citation_num)
            marked_sentences.append(sentence)

        # Reconstruct content preserving paragraph breaks
        result_parts = []
        current_para = []
        for sent_idx, sentence in enumerate(marked_sentences):
            current_para.append(sentence)
            if sent_idx in paragraph_breaks:
                result_parts.append(' '.join(current_para))
                current_para = []
        if current_para:
            result_parts.append(' '.join(current_para))

        marked_content = '\n\n'.join(result_parts)

        cited_count = len(used_citations)
        print(f"[Generator] Inserted {cited_count} citation markers")

        return marked_content, annotated_sources

    def _build_source_url(self, source: Dict) -> str:
        """Build URL for a source based on its type."""
        # Prefer existing URL if present (web articles and YouTube already have URLs)
        if source.get('url'):
            return source['url']

        source_type = source.get('source_type', '')
        source_id = source.get('id', '')

        if source_type == 'twitter':
            username = source.get('username', '')
            return f"https://x.com/{username}/status/{source_id}"
        elif source_type == 'youtube':
            return f"https://youtube.com/watch?v={source_id}"
        elif source_type == 'web':
            return ''  # Web articles should have URL

        return ''

    def _extract_citation_quote(self, source: Dict, sentence: str) -> None:
        """
        Extract the best matching quote from a source for citation display.

        For YouTube sources, also extracts timestamp for deep-linking.
        Updates source dict in-place with 'cited_quote' and 'start_time' fields.

        Args:
            source: Source dict with 'source_type', 'text', 'id', etc.
            sentence: The sentence from the post that references this source
        """
        source_type = source.get('source_type', '')

        # Default: use first 200 chars of source text
        source_text = source.get('text', '')
        source['cited_quote'] = source_text[:200] if source_text else None
        source['start_time'] = None

        if not sentence or len(sentence) < 10:
            return

        try:
            if source_type == 'youtube':
                # Use YouTube-specific function with transcript/timestamp support
                from youtube_channel_scraper import get_youtube_quote_with_timestamp
                from pathlib import Path

                video = {
                    'video_id': source.get('id', ''),
                    'description': source.get('text', ''),
                    'title': source.get('username', ''),  # channel name stored in username
                    'url': source.get('url', source.get('source_url', ''))
                }

                db_path = Path('output_data/ai_news.db')
                quote, url_with_timestamp = get_youtube_quote_with_timestamp(video, sentence, db_path)

                if quote:
                    source['cited_quote'] = quote

                # Extract timestamp from URL if present
                if url_with_timestamp and '&t=' in url_with_timestamp:
                    ts_match = re.search(r'[&?]t=(\d+)s', url_with_timestamp)
                    if ts_match:
                        source['start_time'] = float(ts_match.group(1))
                        # Update source URL to include timestamp
                        source['source_url'] = url_with_timestamp

            elif source_type in ('web', 'twitter'):
                # Use generic paragraph matching with BGE-M3
                from agents.hybrid_retriever import find_best_paragraph_match

                # Create paragraph chunks from source text
                paragraphs = []
                if source_text:
                    # Split on double newlines or sentences
                    parts = source_text.split('\n\n')
                    if len(parts) <= 1:
                        parts = re.split(r'(?<=[.!?])\s+', source_text)
                    paragraphs = [{'text': p.strip(), 'index': i} for i, p in enumerate(parts) if len(p.strip()) >= 30]

                if paragraphs:
                    match = find_best_paragraph_match(sentence, paragraphs, threshold=0.2)
                    if match:
                        source['cited_quote'] = match['text']

        except ImportError as e:
            print(f"[Generator] Quote extraction import failed: {e}")
        except Exception as e:
            print(f"[Generator] Quote extraction failed for {source_type}: {e}")
