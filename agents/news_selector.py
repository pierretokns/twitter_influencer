"""
NewsSelector - Vector-Based Diverse News Selection
===================================================

AGENT TYPE: Data Selection Agent (Vector diversity)

PURPOSE:
    Selects a diverse set of news items from the database using vector
    embeddings. Instead of random/chronological selection, this ensures
    the generated posts cover different topics and angles.

KEY INSIGHT:
    Random news selection often returns semantically similar items (e.g.,
    multiple tweets about the same announcement). Vector-based diversity
    selection ensures each news item represents a distinct topic.

ALGORITHM: Maximal Marginal Relevance (MMR)
    1. Start with the most recent/relevant item
    2. For each slot, select the item that is:
       - Relevant to AI (already filtered)
       - Most DIFFERENT from already selected items
    3. This maximizes diversity while maintaining relevance

IMPLEMENTATION:
    - Uses sentence-transformers for embeddings
    - Batch encodes all candidates for efficiency
    - Selects via MMR algorithm
    - Falls back to random selection if embeddings fail

USAGE:
    selector = NewsSelector(db_path)
    diverse_news = selector.select_diverse_news(limit=20)
"""

import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# Lazy-loaded embedder (shared with proximity_agent)
_embedder = None


def get_embedder():
    """Get or initialize the sentence transformer model."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("[NewsSelector] Loading sentence-transformers model...")
            _embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("[NewsSelector] Model loaded successfully")
        except Exception as e:
            print(f"[NewsSelector] Could not load sentence-transformers: {e}")
            _embedder = None
    return _embedder


class NewsSelector:
    """
    Selects diverse news items using vector embeddings.

    Uses Maximal Marginal Relevance (MMR) algorithm to select items
    that are both relevant and diverse from each other.
    """

    def __init__(self, db_path: Path):
        """
        Initialize the news selector.

        Args:
            db_path: Path to the ai_news.db database
        """
        self.db_path = db_path
        self.embedder = None

    def _get_embedder(self):
        """Get or initialize embedder."""
        if self.embedder is None:
            self.embedder = get_embedder()
        return self.embedder

    def _get_all_news(self, limit: int = 200) -> List[Dict]:
        """
        Get recent AI-relevant news from database.

        Args:
            limit: Maximum items to fetch (before diversity filtering)

        Returns:
            List of news item dicts
        """
        if not self.db_path or not self.db_path.exists():
            print(f"[NewsSelector] Database not found: {self.db_path}")
            return []

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row

        # Get tweets
        tweets = []
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT tweet_id as id, text, username, timestamp,
                       likes_count, 'twitter' as source_type
                FROM tweets
                WHERE is_ai_relevant = TRUE
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            tweets = [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"[NewsSelector] Error fetching tweets: {e}")

        # Get web articles
        articles = []
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT article_id as id, title as text, source_name as username,
                       published_at as timestamp, 0 as likes_count,
                       'web' as source_type, url
                FROM web_articles
                WHERE is_ai_relevant = TRUE
                ORDER BY scraped_at DESC
                LIMIT ?
            ''', (limit,))
            articles = [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"[NewsSelector] Error fetching articles: {e}")

        # Get YouTube videos
        youtube = []
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT video_id as id, title as text, channel_name as username,
                       published_at as timestamp, COALESCE(view_count, 0) as likes_count,
                       'youtube' as source_type, url
                FROM youtube_videos
                WHERE is_ai_relevant = TRUE
                ORDER BY published_at DESC
                LIMIT ?
            ''', (limit,))
            youtube = [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            print(f"[NewsSelector] Error fetching YouTube videos: {e}")

        conn.close()

        # Combine all sources (Discord links excluded - they're just pointers to web content)
        combined = tweets + articles + youtube
        print(f"[NewsSelector] Fetched {len(tweets)} tweets + {len(articles)} articles + {len(youtube)} YouTube = {len(combined)} total")
        return combined

    def _cosine_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate pairwise cosine similarities."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = embeddings / norms
        return np.dot(normalized, normalized.T)

    def _mmr_select(
        self,
        embeddings: np.ndarray,
        items: List[Dict],
        k: int,
        lambda_param: float = 0.5
    ) -> List[int]:
        """
        Select items using Maximal Marginal Relevance.

        MMR balances relevance and diversity:
        - lambda_param = 1.0: Pure relevance (most recent/engaging)
        - lambda_param = 0.0: Pure diversity (most different from selected)
        - lambda_param = 0.5: Balance of both

        Args:
            embeddings: Item embeddings (N x D)
            items: Original items list
            k: Number of items to select
            lambda_param: Balance parameter

        Returns:
            List of selected indices
        """
        n = len(items)
        if n <= k:
            return list(range(n))

        # Calculate similarity matrix
        sim_matrix = self._cosine_similarity_matrix(embeddings)

        # Relevance scores (combine recency and engagement)
        relevance = np.zeros(n)
        for i, item in enumerate(items):
            # Recency: items earlier in list (more recent) get higher score
            recency_score = 1.0 - (i / n)
            # Engagement: normalize likes
            likes = item.get('likes_count', 0) or 0
            engagement_score = min(1.0, likes / 1000)  # Cap at 1000 likes
            relevance[i] = 0.7 * recency_score + 0.3 * engagement_score

        selected = []
        remaining = set(range(n))

        # Start with most relevant item
        first_idx = np.argmax(relevance)
        selected.append(first_idx)
        remaining.remove(first_idx)

        # Iteratively select using MMR
        while len(selected) < k and remaining:
            mmr_scores = []

            for idx in remaining:
                # Relevance component
                rel = relevance[idx]

                # Diversity component (max similarity to any selected item)
                max_sim = max(sim_matrix[idx, sel_idx] for sel_idx in selected)

                # MMR score
                mmr = lambda_param * rel - (1 - lambda_param) * max_sim
                mmr_scores.append((idx, mmr))

            # Select item with highest MMR score
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(best_idx)
            remaining.remove(best_idx)

        return selected

    def select_diverse_news(
        self,
        limit: int = 20,
        lambda_param: float = 0.5
    ) -> List[Dict]:
        """
        Select diverse news items using vector embeddings.

        Args:
            limit: Number of diverse items to select
            lambda_param: Balance between relevance (1.0) and diversity (0.0)

        Returns:
            List of diverse news items
        """
        # Get all recent news
        all_news = self._get_all_news(limit=limit * 10)

        if not all_news:
            return []

        # Try vector-based selection
        embedder = self._get_embedder()
        if embedder is not None:
            try:
                print(f"[NewsSelector] Encoding {len(all_news)} items for diversity selection...")

                # Encode all items
                texts = [item.get('text', '')[:500] for item in all_news]
                embeddings = embedder.encode(texts)

                # Select using MMR
                selected_indices = self._mmr_select(
                    embeddings, all_news, limit, lambda_param
                )

                selected = [all_news[i] for i in selected_indices]
                print(f"[NewsSelector] Selected {len(selected)} diverse items via MMR")

                return selected

            except Exception as e:
                print(f"[NewsSelector] Vector selection failed: {e}")

        # Fallback: simple interleaving of sources
        print("[NewsSelector] Falling back to source-based selection")
        by_source = {
            'twitter': [n for n in all_news if n.get('source_type') == 'twitter'],
            'web': [n for n in all_news if n.get('source_type') == 'web'],
            'youtube': [n for n in all_news if n.get('source_type') == 'youtube'],
        }

        selected = []
        indices = {k: 0 for k in by_source}
        source_order = ['twitter', 'youtube', 'web']

        while len(selected) < limit:
            added = False
            for source in source_order:
                items = by_source[source]
                idx = indices[source]
                if idx < len(items) and len(selected) < limit:
                    selected.append(items[idx])
                    indices[source] += 1
                    added = True
            if not added:
                break

        return selected[:limit]

    def get_news_by_topic(self, topic: str, limit: int = 10) -> List[Dict]:
        """
        Get news items similar to a topic query.

        Args:
            topic: Topic query string (e.g., "LLM benchmarks", "AI regulation")
            limit: Number of items to return

        Returns:
            List of news items most similar to the topic
        """
        all_news = self._get_all_news(limit=200)
        if not all_news:
            return []

        embedder = self._get_embedder()
        if embedder is None:
            # Fallback: keyword matching
            topic_words = set(topic.lower().split())
            scored = []
            for item in all_news:
                text_words = set(item.get('text', '').lower().split())
                overlap = len(topic_words & text_words)
                scored.append((item, overlap))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [item for item, _ in scored[:limit]]

        # Vector similarity search
        try:
            texts = [item.get('text', '')[:500] for item in all_news]
            embeddings = embedder.encode(texts)
            topic_embedding = embedder.encode([topic])[0]

            # Calculate similarities
            norms = np.linalg.norm(embeddings, axis=1)
            topic_norm = np.linalg.norm(topic_embedding)
            similarities = np.dot(embeddings, topic_embedding) / (norms * topic_norm + 1e-8)

            # Get top items
            top_indices = np.argsort(similarities)[-limit:][::-1]
            return [all_news[i] for i in top_indices]

        except Exception as e:
            print(f"[NewsSelector] Topic search failed: {e}")
            return all_news[:limit]

    def get_trending_topics(self) -> List[Dict]:
        """
        Get trending topics from the database.

        Returns:
            List of topic dicts with name, count, trend
        """
        if not self.db_path or not self.db_path.exists():
            return []

        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('''
                SELECT name, tweet_count, avg_engagement, trend_direction
                FROM topics
                ORDER BY tweet_count DESC
                LIMIT 10
            ''')
            topics = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return topics
        except Exception as e:
            print(f"[NewsSelector] Error fetching topics: {e}")
            return []
