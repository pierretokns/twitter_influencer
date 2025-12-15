"""
ProximityAgent - Vector-Based Semantic Similarity Detection
============================================================

AGENT TYPE: Deduplication Agent (Vector similarity)

PURPOSE:
    Detects semantically similar posts using vector embeddings to ensure
    diversity in the ranking pool. Uses sentence-transformers for fast,
    consistent similarity calculation without LLM API calls.

KEY INSIGHT (from Google Co-Scientist):
    The "Proximity Agent" in Google's system identifies similarities between
    hypotheses to avoid redundancy. This ensures the tournament compares
    genuinely different approaches rather than minor variations.

IMPLEMENTATION:
    - Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings
    - Cosine similarity for comparison (fast, no API calls)
    - Falls back to TF-IDF if sentence-transformers unavailable
    - Calculates all pairwise similarities in batch

SIMILARITY SCALE:
    0.0-0.3: Completely different topics/angles
    0.3-0.5: Same general topic but different focus
    0.5-0.7: Similar topic and angle, but different execution
    0.7-0.9: Very similar - same main points (DUPLICATE THRESHOLD)
    0.9-1.0: Nearly identical - redundant content

PERFORMANCE:
    - Old (LLM-based): ~30s per comparison (with API calls)
    - New (Vector-based): ~0.1s per comparison (local computation)
    - For 5 variants: 10 comparisons = ~300s old vs ~1s new

TRACKING:
    - similarity_scores: Dict mapping other variant IDs to similarity (0-1)
    - is_duplicate: Boolean flag for filtering

USAGE:
    proximity_agent = ProximityAgent()
    variants = proximity_agent.find_duplicates(variants)  # Marks duplicates
    unique_variants = proximity_agent.filter_duplicates(variants)  # Removes them
"""

import numpy as np
from typing import List, Optional

from .post_variant import PostVariant


# Lazy-loaded embedder (same pattern as ai_news_scraper.py)
_embedder = None


def get_embedder():
    """Get or initialize the sentence transformer model."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("[ProximityAgent] Loading sentence-transformers model...")
            _embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("[ProximityAgent] Model loaded successfully")
        except Exception as e:
            print(f"[ProximityAgent] Could not load sentence-transformers: {e}")
            print("[ProximityAgent] Using fallback TF-IDF embeddings")
            _embedder = FallbackEmbedder()
    return _embedder


class FallbackEmbedder:
    """TF-IDF based fallback when sentence-transformers unavailable."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=384)
        self.fitted = False

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to TF-IDF vectors."""
        if isinstance(texts, str):
            texts = [texts]

        if not self.fitted:
            self.vectorizer.fit(texts)
            self.fitted = True

        return self.vectorizer.transform(texts).toarray()


class ProximityAgent:
    """
    Proximity Agent - Detects similar/duplicate posts using vector embeddings.

    Uses sentence-transformers for fast semantic similarity without API calls.
    Inspired by Google's AI Co-Scientist "Proximity Agent".
    """

    # Threshold above which posts are considered duplicates
    SIMILARITY_THRESHOLD = 0.7

    def __init__(self, threshold: float = None):
        """
        Initialize the Proximity Agent.

        Args:
            threshold: Similarity threshold for marking duplicates (default: 0.7)
        """
        self.similarity_threshold = threshold or self.SIMILARITY_THRESHOLD
        self.embedder = None  # Lazy load

    def _get_embedder(self):
        """Get or initialize embedder."""
        if self.embedder is None:
            self.embedder = get_embedder()
        return self.embedder

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        # Normalize vectors
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def calculate_similarity(self, post_a: PostVariant, post_b: PostVariant) -> float:
        """
        Calculate semantic similarity between two posts using embeddings.

        Args:
            post_a: First post
            post_b: Second post

        Returns:
            Similarity score between 0.0 and 1.0
        """
        embedder = self._get_embedder()

        # Get embeddings (batch for efficiency)
        embeddings = embedder.encode([post_a.content, post_b.content])

        # Calculate cosine similarity
        similarity = self._cosine_similarity(embeddings[0], embeddings[1])

        # Clamp to valid range
        return min(1.0, max(0.0, similarity))

    def calculate_all_similarities(self, variants: List[PostVariant]) -> np.ndarray:
        """
        Calculate all pairwise similarities in batch (much faster).

        Args:
            variants: List of PostVariants

        Returns:
            NxN similarity matrix
        """
        embedder = self._get_embedder()

        # Batch encode all posts at once
        contents = [v.content for v in variants]
        embeddings = embedder.encode(contents)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = embeddings / norms

        # Calculate all cosine similarities at once (matrix multiplication)
        similarity_matrix = np.dot(normalized, normalized.T)

        return similarity_matrix

    def find_duplicates(self, variants: List[PostVariant]) -> List[PostVariant]:
        """
        Calculate pairwise similarities and mark duplicates.

        Uses batch embedding calculation for speed.
        For each pair with similarity >= threshold, marks the lower-QE-scoring
        post as a duplicate.

        Args:
            variants: List of PostVariants to check

        Returns:
            Same list with similarity_scores and is_duplicate updated
        """
        if len(variants) < 2:
            return variants

        print(f"[ProximityAgent] Checking proximity for {len(variants)} variants (vector-based)...")

        # Calculate all similarities at once
        similarity_matrix = self.calculate_all_similarities(variants)

        comparisons = 0
        duplicates_found = 0

        # Process similarity matrix
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                similarity = float(similarity_matrix[i, j])
                comparisons += 1

                post_a = variants[i]
                post_b = variants[j]

                # Store similarity scores in both directions
                post_a.similarity_scores[post_b.variant_id] = similarity
                post_b.similarity_scores[post_a.variant_id] = similarity

                if similarity >= self.similarity_threshold:
                    duplicates_found += 1
                    # Mark lower-scoring post as duplicate
                    if post_a.qe_score <= post_b.qe_score:
                        post_a.is_duplicate = True
                        print(f"  [Proximity] {post_a.variant_id} duplicate of {post_b.variant_id} (sim: {similarity:.2f})")
                    else:
                        post_b.is_duplicate = True
                        print(f"  [Proximity] {post_b.variant_id} duplicate of {post_a.variant_id} (sim: {similarity:.2f})")

        duplicate_count = sum(1 for v in variants if v.is_duplicate)
        print(f"[ProximityAgent] Complete: {comparisons} comparisons, {duplicate_count} duplicates marked")
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

    def get_diversity_score(self, variants: List[PostVariant]) -> float:
        """
        Calculate overall diversity score for a set of variants.

        Higher score = more diverse content.

        Args:
            variants: List of PostVariants

        Returns:
            Diversity score (0.0 to 1.0)
        """
        if len(variants) < 2:
            return 1.0

        similarity_matrix = self.calculate_all_similarities(variants)

        # Average of off-diagonal elements (excluding self-similarity)
        n = len(variants)
        total_sim = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_sim += similarity_matrix[i, j]
                count += 1

        avg_similarity = total_sim / count if count > 0 else 0
        diversity = 1.0 - avg_similarity

        return diversity
