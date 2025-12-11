# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "undetected-chromedriver>=3.5.0",
#     "python-dotenv>=0.19.0",
#     "setuptools>=65.0.0",
#     "requests>=2.31.0",
#     "sentence-transformers>=2.2.0",
#     "sqlite-vec>=0.1.0",
#     "numpy>=1.24.0",
#     "scikit-learn>=1.3.0",
#     "hdbscan>=0.8.33",
# ]
# ///

"""
AI News Scraper - Twitter/X AI Content Aggregator with Vector Similarity Search

Scrapes AI-related content from curated influencers, stores in SQLite with VSS,
detects emerging trends, and discovers new high-signal accounts dynamically.
"""

import os
import sys
import json
import time
import random
import sqlite3
import ssl
import re
import struct
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib

from dotenv import load_dotenv
import numpy as np

# Lazy imports for heavy ML libraries
_sentence_transformer = None
_hdbscan = None
_sklearn_umap = None


def get_sentence_transformer():
    """Lazy load sentence transformer model with offline fallback"""
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            import os

            # Try to use offline mode if model is cached
            os.environ.setdefault('HF_HUB_OFFLINE', '0')

            print("[INFO] Loading sentence-transformers model (first time may download ~90MB)...")
            _sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            print("[INFO] Model loaded successfully")
        except Exception as e:
            print(f"[WARN] Could not load sentence-transformers: {e}")
            print("[INFO] Using fallback TF-IDF based embeddings")
            _sentence_transformer = FallbackEmbedder()
    return _sentence_transformer


class FallbackEmbedder:
    """Fallback embedder using TF-IDF when sentence-transformers unavailable"""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.svd = TruncatedSVD(n_components=384, random_state=42)
        self._fitted = False
        self._corpus = []

    def encode(self, sentences, show_progress_bar=False, batch_size=32):
        """Generate embeddings using TF-IDF + SVD"""
        if isinstance(sentences, str):
            sentences = [sentences]

        # Add to corpus and refit
        self._corpus.extend(sentences)

        # Need enough samples to fit SVD
        if len(self._corpus) < 10:
            # Return random embeddings for small corpus
            return np.random.randn(len(sentences), 384).astype(np.float32)

        # Fit on corpus
        try:
            tfidf_matrix = self.vectorizer.fit_transform(self._corpus)
            if tfidf_matrix.shape[1] >= 384:
                reduced = self.svd.fit_transform(tfidf_matrix)
            else:
                # Pad if not enough features
                reduced = np.zeros((len(self._corpus), 384))
                temp = tfidf_matrix.toarray()
                reduced[:, :temp.shape[1]] = temp

            # Return only the embeddings for requested sentences
            result = reduced[-len(sentences):].astype(np.float32)
            return result if len(sentences) > 1 else result[0]
        except Exception:
            return np.random.randn(len(sentences), 384).astype(np.float32)

    def similarity(self, emb1, emb2):
        """Compute cosine similarity"""
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0, 0]


def get_hdbscan():
    """Lazy load HDBSCAN"""
    global _hdbscan
    if _hdbscan is None:
        import hdbscan
        _hdbscan = hdbscan
    return _hdbscan


# Fix SSL certificate verification for macOS/Python 3.13
ssl._create_default_https_context = ssl._create_unverified_context


# =============================================================================
# CURATED AI INFLUENCER LIST - High Signal Sources
# =============================================================================

# Categories of AI influencers for comprehensive coverage
AI_INFLUENCERS = {
    # -------------------------------------------------------------------------
    # TIER 1: Core AI Researchers & Scientists (Highest Signal)
    # -------------------------------------------------------------------------
    "researchers": [
        "ylecun",           # Yann LeCun - Meta Chief AI Scientist, Turing Award
        "demaborsa",        # Demis Hassabis - DeepMind CEO, Nobel 2024
        "drfeifei",         # Fei-Fei Li - Stanford HAI, computer vision pioneer
        "AndrewYNg",        # Andrew Ng - Coursera, deeplearning.ai founder
        "karpathy",         # Andrej Karpathy - Tesla AI, OpenAI founding team
        "jeremyphoward",    # Jeremy Howard - fast.ai co-founder
        "GaryMarcus",       # Gary Marcus - AI critic, cognitive scientist
        "EMostaque",        # Emad Mostaque - Stability AI founder
        "iaborsa",          # Ilya Sutskever - OpenAI co-founder (if active)
        "goodfellow_ian",   # Ian Goodfellow - GAN inventor
        "geoffreyhinton",   # Geoffrey Hinton - Godfather of Deep Learning
    ],

    # -------------------------------------------------------------------------
    # TIER 2: AI Companies & Labs (Official Announcements)
    # -------------------------------------------------------------------------
    "organizations": [
        "OpenAI",           # ChatGPT, GPT-4, DALL-E
        "GoogleDeepMind",   # Gemini, AlphaFold
        "AnthropicAI",      # Claude
        "xaborsa",          # xAI - Grok
        "MetaAI",           # Llama, open source AI
        "MistralAI",        # European AI leader
        "huggingface",      # ML platform & community
        "GoogleAI",         # Google AI research
        "MSFTResearch",     # Microsoft Research
        "nvidia",           # GPU & AI infrastructure
        "StabilityAI",      # Stable Diffusion
        "CohereAI",         # Cohere AI
        "scale_AI",         # Scale AI - data labeling
        "peraborsa",        # Perplexity AI - search
    ],

    # -------------------------------------------------------------------------
    # TIER 3: AI Journalists & Commentators (News & Analysis)
    # -------------------------------------------------------------------------
    "journalists": [
        "sama",             # Sam Altman - OpenAI CEO (major announcements)
        "eaborsa",          # Elad Gil - investor, AI commentary
        "benedictevans",    # Benedict Evans - tech analyst
        "svpino",           # Santiago - ML engineering content
        "DrJimFan",         # Jim Fan - NVIDIA AI research
        "oaborsa",          # Elvis Saravia - ML papers & news
        "_akaborsa",        # Aakash Kumar - AI news aggregator
        "ai_breakfast",     # AI Breakfast - daily AI news
    ],

    # -------------------------------------------------------------------------
    # TIER 4: AI Engineering & Practical (Tutorials & Tools)
    # -------------------------------------------------------------------------
    "engineering": [
        "llama_index",      # LlamaIndex - RAG framework
        "LangChainAI",      # LangChain - LLM apps
        "weights_biases",   # Weights & Biases - ML ops
        "modal_labs",       # Modal - serverless ML
        "replicate",        # Replicate - ML inference
        "roboaborsa",       # Roboflow - computer vision
        "streamlit",        # Streamlit - ML apps
        "Gradio",           # Gradio - ML interfaces
    ],

    # -------------------------------------------------------------------------
    # TIER 5: AI Safety & Policy (Important Context)
    # -------------------------------------------------------------------------
    "safety_policy": [
        "AISafetyInst",     # AI Safety Institute
        "ArtificialAnalys", # AI policy analysis
        "DarioAmodei",      # Anthropic CEO - safety focus
        "janaborsa",        # Jan Leike - alignment research
    ],
}

# Flatten for easy iteration
def get_all_seed_influencers() -> List[str]:
    """Get all seed influencers as a flat list"""
    all_influencers = []
    for category, handles in AI_INFLUENCERS.items():
        all_influencers.extend(handles)
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for h in all_influencers:
        if h.lower() not in seen:
            seen.add(h.lower())
            unique.append(h)
    return unique


# AI-related keywords for content filtering
AI_KEYWORDS = [
    # Core AI/ML terms
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "neural network", "llm", "large language model", "gpt", "chatgpt",
    "transformer", "attention mechanism", "embedding", "vector",

    # Specific models & companies
    "openai", "anthropic", "claude", "gemini", "llama", "mistral",
    "stable diffusion", "midjourney", "dall-e", "sora", "grok",

    # Techniques & concepts
    "fine-tuning", "rag", "retrieval augmented", "prompt engineering",
    "chain of thought", "reasoning", "alignment", "rlhf", "dpo",
    "multimodal", "vision language", "agents", "agentic",

    # Infrastructure
    "gpu", "cuda", "inference", "training", "parameters", "weights",
    "quantization", "distillation", "benchmark",

    # Applications
    "text-to-image", "text-to-video", "speech-to-text", "coding assistant",
    "ai safety", "alignment", "interpretability", "hallucination",

    # Research terms
    "paper", "arxiv", "research", "sota", "state of the art",
    "breakthrough", "benchmark", "evaluation",
]


@dataclass
class InfluencerScore:
    """Track influencer quality metrics for discovery"""
    username: str
    total_tweets: int = 0
    ai_relevant_tweets: int = 0
    total_engagement: int = 0  # likes + retweets + replies
    avg_engagement: float = 0.0
    mention_count: int = 0  # How often mentioned by other influencers
    follower_estimate: int = 0
    last_seen: Optional[str] = None
    is_seed: bool = False
    discovery_source: str = ""  # How we found them
    quality_score: float = 0.0

    def calculate_quality_score(self) -> float:
        """Calculate overall quality score for ranking"""
        if self.total_tweets == 0:
            return 0.0

        relevance_ratio = self.ai_relevant_tweets / max(self.total_tweets, 1)
        engagement_factor = min(self.avg_engagement / 100, 10)  # Cap at 10x
        mention_bonus = min(self.mention_count * 0.1, 2)  # Cap at 2x
        seed_bonus = 2.0 if self.is_seed else 1.0

        self.quality_score = (
            relevance_ratio * 40 +  # 40% weight on relevance
            engagement_factor * 30 +  # 30% weight on engagement
            mention_bonus * 20 +  # 20% weight on network effects
            seed_bonus * 10  # 10% boost for seed accounts
        )
        return self.quality_score


class Logger:
    """Simple console logger"""

    @staticmethod
    def info(msg: str):
        print(f"[INFO] {msg}")

    @staticmethod
    def success(msg: str):
        print(f"[OK] {msg}")

    @staticmethod
    def warning(msg: str):
        print(f"[WARN] {msg}")

    @staticmethod
    def error(msg: str):
        print(f"[ERROR] {msg}")

    @staticmethod
    def debug(msg: str):
        if os.getenv('DEBUG'):
            print(f"[DEBUG] {msg}")


# =============================================================================
# DATABASE MANAGER WITH VECTOR SIMILARITY SEARCH
# =============================================================================

class AINewsDatabase:
    """SQLite database with sqlite-vec for vector similarity search"""

    EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None
        self._init_database()

    def _init_database(self):
        """Initialize database with VSS support"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # Load sqlite-vec extension
        try:
            import sqlite_vec
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            Logger.success("sqlite-vec extension loaded")
            self._has_vec = True
        except Exception as e:
            Logger.warning(f"sqlite-vec not available: {e}")
            Logger.info("Falling back to basic storage without VSS")
            self._has_vec = False

        cursor = self.conn.cursor()

        # Influencers table - track sources
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS influencers (
                username TEXT PRIMARY KEY,
                display_name TEXT,
                category TEXT,
                is_seed BOOLEAN DEFAULT FALSE,
                discovery_source TEXT,
                total_tweets INTEGER DEFAULT 0,
                ai_relevant_tweets INTEGER DEFAULT 0,
                total_engagement INTEGER DEFAULT 0,
                avg_engagement REAL DEFAULT 0.0,
                mention_count INTEGER DEFAULT 0,
                quality_score REAL DEFAULT 0.0,
                last_scraped TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        ''')

        # Tweets table - main content storage
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tweets (
                tweet_id TEXT PRIMARY KEY,
                username TEXT,
                display_name TEXT,
                text TEXT,
                timestamp TEXT,
                url TEXT,
                replies_count INTEGER DEFAULT 0,
                retweets_count INTEGER DEFAULT 0,
                likes_count INTEGER DEFAULT 0,
                has_media BOOLEAN DEFAULT FALSE,
                media_type TEXT DEFAULT 'none',
                is_reply BOOLEAN DEFAULT FALSE,
                is_ai_relevant BOOLEAN DEFAULT FALSE,
                ai_relevance_score REAL DEFAULT 0.0,
                scraped_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES influencers(username)
            )
        ''')

        # Tweet embeddings - for vector search (if sqlite-vec available)
        if self._has_vec:
            try:
                cursor.execute(f'''
                    CREATE VIRTUAL TABLE IF NOT EXISTS tweet_embeddings USING vec0(
                        tweet_id TEXT PRIMARY KEY,
                        embedding float[{self.EMBEDDING_DIM}]
                    )
                ''')
                Logger.success("Vector embeddings table created")
            except Exception as e:
                Logger.warning(f"Could not create vector table: {e}")
                self._has_vec = False

        # Topics/Themes table - for trend detection
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS topics (
                topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                keywords TEXT,  -- JSON array of keywords
                tweet_count INTEGER DEFAULT 0,
                avg_engagement REAL DEFAULT 0.0,
                first_seen TEXT,
                last_seen TEXT,
                trend_direction TEXT DEFAULT 'stable',  -- rising, stable, declining
                is_emerging BOOLEAN DEFAULT FALSE,
                centroid_embedding BLOB  -- Average embedding for the cluster
            )
        ''')

        # Topic-Tweet mapping
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS topic_tweets (
                topic_id INTEGER,
                tweet_id TEXT,
                similarity_score REAL,
                PRIMARY KEY (topic_id, tweet_id),
                FOREIGN KEY (topic_id) REFERENCES topics(topic_id),
                FOREIGN KEY (tweet_id) REFERENCES tweets(tweet_id)
            )
        ''')

        # Mentioned accounts - for influencer discovery
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mentioned_accounts (
                username TEXT PRIMARY KEY,
                mention_count INTEGER DEFAULT 1,
                mentioned_by TEXT,  -- JSON array of who mentioned them
                first_seen TEXT,
                last_seen TEXT,
                is_promoted BOOLEAN DEFAULT FALSE  -- Promoted to influencer?
            )
        ''')

        # Scrape history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scrape_history (
                scrape_id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT,
                completed_at TEXT,
                tweets_scraped INTEGER DEFAULT 0,
                influencers_scraped INTEGER DEFAULT 0,
                new_influencers_discovered INTEGER DEFAULT 0,
                topics_detected INTEGER DEFAULT 0,
                status TEXT DEFAULT 'running'
            )
        ''')

        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tweets_timestamp ON tweets(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tweets_username ON tweets(username)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tweets_ai_relevant ON tweets(is_ai_relevant)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_influencers_score ON influencers(quality_score DESC)')

        self.conn.commit()
        Logger.success(f"Database initialized: {self.db_path}")

    def save_influencer(self, username: str, display_name: str = None,
                        category: str = None, is_seed: bool = False,
                        discovery_source: str = None):
        """Save or update an influencer"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO influencers (username, display_name, category, is_seed, discovery_source)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
                display_name = COALESCE(excluded.display_name, display_name),
                category = COALESCE(excluded.category, category)
        ''', (username.lower(), display_name, category, is_seed, discovery_source))
        self.conn.commit()

    def save_tweet(self, tweet_data: Dict[str, Any], embedding: np.ndarray = None):
        """Save a tweet with optional embedding"""
        cursor = self.conn.cursor()

        tweet_id = tweet_data.get('tweet_id')
        if not tweet_id:
            return

        # Check AI relevance
        text = tweet_data.get('text', '').lower()
        is_ai_relevant = any(kw.lower() in text for kw in AI_KEYWORDS)
        relevance_score = sum(1 for kw in AI_KEYWORDS if kw.lower() in text) / len(AI_KEYWORDS)

        cursor.execute('''
            INSERT OR REPLACE INTO tweets (
                tweet_id, username, display_name, text, timestamp, url,
                replies_count, retweets_count, likes_count,
                has_media, media_type, is_reply,
                is_ai_relevant, ai_relevance_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            tweet_id,
            tweet_data.get('username', '').lower().lstrip('@'),
            tweet_data.get('display_name'),
            tweet_data.get('text'),
            tweet_data.get('timestamp'),
            tweet_data.get('url'),
            tweet_data.get('replies_count', 0),
            tweet_data.get('retweets_count', 0),
            tweet_data.get('likes_count', 0),
            tweet_data.get('has_media', False),
            tweet_data.get('media_type', 'none'),
            tweet_data.get('is_reply', False),
            is_ai_relevant,
            relevance_score
        ))

        # Save embedding if available
        if embedding is not None and self._has_vec:
            try:
                # Convert numpy array to bytes for sqlite-vec
                embedding_list = embedding.tolist()
                cursor.execute('''
                    INSERT OR REPLACE INTO tweet_embeddings (tweet_id, embedding)
                    VALUES (?, ?)
                ''', (tweet_id, json.dumps(embedding_list)))
            except Exception as e:
                Logger.debug(f"Could not save embedding: {e}")

        self.conn.commit()

        # Track mentions for influencer discovery
        self._extract_and_save_mentions(tweet_data)

        return is_ai_relevant

    def _extract_and_save_mentions(self, tweet_data: Dict[str, Any]):
        """Extract @mentions from tweet for influencer discovery"""
        text = tweet_data.get('text', '')
        mentions = re.findall(r'@(\w+)', text)

        if not mentions:
            return

        cursor = self.conn.cursor()
        author = tweet_data.get('username', '').lower()

        for mention in mentions:
            mention = mention.lower()
            if mention == author:
                continue

            cursor.execute('''
                INSERT INTO mentioned_accounts (username, mention_count, mentioned_by, first_seen, last_seen)
                VALUES (?, 1, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(username) DO UPDATE SET
                    mention_count = mention_count + 1,
                    last_seen = CURRENT_TIMESTAMP
            ''', (mention, json.dumps([author])))

        self.conn.commit()

    def similarity_search(self, query_embedding: np.ndarray, limit: int = 20) -> List[Dict]:
        """Find similar tweets using vector similarity search"""
        if not self._has_vec:
            Logger.warning("VSS not available, falling back to keyword search")
            return []

        cursor = self.conn.cursor()

        try:
            embedding_json = json.dumps(query_embedding.tolist())
            # sqlite-vec requires k=? constraint in WHERE clause for KNN queries
            cursor.execute(f'''
                SELECT
                    e.tweet_id,
                    e.distance,
                    t.text,
                    t.username,
                    t.timestamp,
                    t.likes_count,
                    t.retweets_count
                FROM tweet_embeddings e
                JOIN tweets t ON e.tweet_id = t.tweet_id
                WHERE e.embedding MATCH ? AND k = ?
                ORDER BY e.distance
            ''', (embedding_json, limit))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'tweet_id': row['tweet_id'],
                    'distance': row['distance'],
                    'text': row['text'],
                    'username': row['username'],
                    'timestamp': row['timestamp'],
                    'engagement': row['likes_count'] + row['retweets_count']
                })
            return results
        except Exception as e:
            Logger.error(f"Similarity search failed: {e}")
            return []

    def get_discovered_influencers(self, min_mentions: int = 3) -> List[Dict]:
        """Get accounts that should be considered for promotion to influencer"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT
                m.username,
                m.mention_count,
                m.first_seen,
                m.last_seen
            FROM mentioned_accounts m
            LEFT JOIN influencers i ON m.username = i.username
            WHERE m.mention_count >= ?
                AND m.is_promoted = FALSE
                AND i.username IS NULL
            ORDER BY m.mention_count DESC
            LIMIT 50
        ''', (min_mentions,))

        return [dict(row) for row in cursor.fetchall()]

    def promote_to_influencer(self, username: str):
        """Promote a discovered account to tracked influencer"""
        cursor = self.conn.cursor()

        # Mark as promoted in mentions table
        cursor.execute('''
            UPDATE mentioned_accounts SET is_promoted = TRUE WHERE username = ?
        ''', (username,))

        # Add to influencers table
        cursor.execute('''
            INSERT OR IGNORE INTO influencers (username, is_seed, discovery_source)
            VALUES (?, FALSE, 'auto_discovered')
        ''', (username,))

        self.conn.commit()
        Logger.success(f"Promoted @{username} to tracked influencer")

    def get_recent_tweets(self, hours: int = 24, ai_only: bool = True) -> List[Dict]:
        """Get recent tweets for trend analysis"""
        cursor = self.conn.cursor()

        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        query = '''
            SELECT * FROM tweets
            WHERE timestamp > ?
        '''
        if ai_only:
            query += ' AND is_ai_relevant = TRUE'
        query += ' ORDER BY timestamp DESC'

        cursor.execute(query, (cutoff,))
        return [dict(row) for row in cursor.fetchall()]

    def get_influencers_to_scrape(self, limit: int = 50) -> List[str]:
        """Get list of influencers to scrape, prioritized by quality score"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT username FROM influencers
            WHERE is_active = TRUE
            ORDER BY
                is_seed DESC,
                quality_score DESC,
                last_scraped ASC NULLS FIRST
            LIMIT ?
        ''', (limit,))

        return [row['username'] for row in cursor.fetchall()]

    def update_influencer_stats(self, username: str):
        """Update aggregated stats for an influencer"""
        cursor = self.conn.cursor()

        cursor.execute('''
            UPDATE influencers SET
                total_tweets = (SELECT COUNT(*) FROM tweets WHERE username = ?),
                ai_relevant_tweets = (SELECT COUNT(*) FROM tweets WHERE username = ? AND is_ai_relevant = TRUE),
                total_engagement = (SELECT COALESCE(SUM(likes_count + retweets_count + replies_count), 0) FROM tweets WHERE username = ?),
                avg_engagement = (SELECT COALESCE(AVG(likes_count + retweets_count + replies_count), 0) FROM tweets WHERE username = ?),
                last_scraped = CURRENT_TIMESTAMP
            WHERE username = ?
        ''', (username, username, username, username, username))

        # Calculate quality score
        cursor.execute('SELECT * FROM influencers WHERE username = ?', (username,))
        row = cursor.fetchone()
        if row:
            score = InfluencerScore(
                username=username,
                total_tweets=row['total_tweets'],
                ai_relevant_tweets=row['ai_relevant_tweets'],
                total_engagement=row['total_engagement'],
                avg_engagement=row['avg_engagement'],
                is_seed=row['is_seed']
            )
            quality = score.calculate_quality_score()
            cursor.execute('UPDATE influencers SET quality_score = ? WHERE username = ?',
                          (quality, username))

        self.conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        cursor = self.conn.cursor()

        stats = {}
        cursor.execute('SELECT COUNT(*) FROM tweets')
        stats['total_tweets'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM tweets WHERE is_ai_relevant = TRUE')
        stats['ai_relevant_tweets'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM influencers')
        stats['total_influencers'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM influencers WHERE is_seed = TRUE')
        stats['seed_influencers'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM mentioned_accounts WHERE mention_count >= 3')
        stats['potential_influencers'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM topics')
        stats['topics_detected'] = cursor.fetchone()[0]

        return stats

    def close(self):
        if self.conn:
            self.conn.close()


# =============================================================================
# TREND DETECTION & THEME ANALYSIS
# =============================================================================

class TrendDetector:
    """Detect emerging trends and themes using clustering"""

    def __init__(self, db: AINewsDatabase):
        self.db = db
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = get_sentence_transformer()
        return self._model

    def _cluster_embeddings(self, embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
        """Cluster embeddings using HDBSCAN with KMeans fallback"""
        try:
            hdbscan = get_hdbscan()
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=2,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            return clusterer.fit_predict(embeddings)
        except Exception as e:
            Logger.warning(f"HDBSCAN failed ({e}), using KMeans fallback")
            from sklearn.cluster import KMeans

            # Estimate number of clusters (sqrt of n_samples is a heuristic)
            n_clusters = max(2, min(int(np.sqrt(len(embeddings))), 10))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            return kmeans.fit_predict(embeddings)

    def analyze_trends(self, hours: int = 24, min_cluster_size: int = 3) -> List[Dict]:
        """Analyze recent tweets to detect emerging trends"""
        tweets = self.db.get_recent_tweets(hours=hours, ai_only=True)

        if len(tweets) < min_cluster_size * 2:
            Logger.info(f"Not enough tweets for trend analysis ({len(tweets)} tweets)")
            return []

        Logger.info(f"Analyzing {len(tweets)} tweets for trends...")

        # Get embeddings
        texts = [t['text'] for t in tweets]
        embeddings = self.model.encode(texts, show_progress_bar=False)

        # Cluster embeddings
        cluster_labels = self._cluster_embeddings(embeddings, min_cluster_size)

        # Analyze each cluster
        trends = []
        unique_labels = set(cluster_labels)
        unique_labels.discard(-1)  # Remove noise label

        for label in unique_labels:
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_tweets = [tweets[i] for i in cluster_indices]
            cluster_embeddings = embeddings[cluster_indices]

            # Extract keywords from cluster
            keywords = self._extract_keywords(cluster_tweets)

            # Calculate cluster metrics
            total_engagement = sum(
                t['likes_count'] + t['retweets_count'] + t['replies_count']
                for t in cluster_tweets
            )
            avg_engagement = total_engagement / len(cluster_tweets)

            # Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0)

            trend = {
                'cluster_id': int(label),
                'keywords': keywords[:10],  # Top 10 keywords
                'tweet_count': len(cluster_tweets),
                'total_engagement': total_engagement,
                'avg_engagement': avg_engagement,
                'sample_tweets': [t['text'][:200] for t in cluster_tweets[:3]],
                'centroid': centroid,
                'is_emerging': len(cluster_tweets) >= 5 and avg_engagement > 50
            }
            trends.append(trend)

        # Sort by engagement
        trends.sort(key=lambda x: x['total_engagement'], reverse=True)

        Logger.success(f"Detected {len(trends)} topic clusters")
        return trends

    def _extract_keywords(self, tweets: List[Dict], top_n: int = 20) -> List[str]:
        """Extract significant keywords from a cluster of tweets"""
        # Combine all text
        all_text = ' '.join(t['text'] for t in tweets).lower()

        # Simple keyword extraction (could be enhanced with TF-IDF)
        words = re.findall(r'\b[a-z]{3,}\b', all_text)

        # Remove common stop words
        stop_words = {
            'the', 'and', 'for', 'that', 'this', 'with', 'are', 'was', 'will',
            'have', 'has', 'been', 'were', 'they', 'their', 'what', 'when',
            'where', 'which', 'who', 'how', 'more', 'some', 'than', 'them',
            'these', 'just', 'but', 'not', 'you', 'your', 'can', 'get', 'all',
            'out', 'about', 'from', 'into', 'over', 'such', 'very', 'now',
            'new', 'like', 'use', 'using', 'one', 'two', 'also', 'any',
            'http', 'https', 'com', 'www'
        }

        words = [w for w in words if w not in stop_words]

        # Count and return most common
        word_counts = Counter(words)
        return [word for word, count in word_counts.most_common(top_n)]

    def save_trends_to_db(self, trends: List[Dict]):
        """Save detected trends to database"""
        cursor = self.db.conn.cursor()

        for trend in trends:
            cursor.execute('''
                INSERT INTO topics (
                    name, keywords, tweet_count, avg_engagement,
                    first_seen, last_seen, is_emerging, centroid_embedding
                ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?)
            ''', (
                ', '.join(trend['keywords'][:3]),  # Name from top 3 keywords
                json.dumps(trend['keywords']),
                trend['tweet_count'],
                trend['avg_engagement'],
                trend['is_emerging'],
                trend['centroid'].tobytes() if 'centroid' in trend else None
            ))

        self.db.conn.commit()


# =============================================================================
# EMBEDDING GENERATOR
# =============================================================================

class EmbeddingGenerator:
    """Generate embeddings for tweets"""

    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = get_sentence_transformer()
        return self._model

    def generate(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.model.encode(text, show_progress_bar=False)

    def generate_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)


# =============================================================================
# TWITTER SCRAPER (Browser Automation)
# =============================================================================

class TwitterAuth:
    """Handles Twitter authentication"""

    def __init__(self, driver, username: str, password: str):
        self.driver = driver
        self.username = username
        self.password = password

    def login(self) -> bool:
        """Login to Twitter"""
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.common.keys import Keys
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            Logger.info("Logging into Twitter...")

            self.driver.get("https://twitter.com")
            time.sleep(random.uniform(2, 4))
            self.driver.get("https://twitter.com/i/flow/login")
            time.sleep(random.uniform(3, 5))

            # Username
            username_input = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'input[autocomplete="username"]'))
            )
            for char in self.username:
                username_input.send_keys(char)
                if random.random() < 0.1:
                    time.sleep(random.uniform(0.2, 0.4))
                else:
                    time.sleep(random.uniform(0.08, 0.18))
            time.sleep(random.uniform(0.8, 2.0))
            username_input.send_keys(Keys.RETURN)
            time.sleep(random.uniform(2.5, 5.0))

            # Password
            password_input = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'input[name="password"]'))
            )
            for char in self.password:
                password_input.send_keys(char)
                if random.random() < 0.1:
                    time.sleep(random.uniform(0.15, 0.35))
                else:
                    time.sleep(random.uniform(0.06, 0.16))
            time.sleep(random.uniform(0.6, 1.8))
            password_input.send_keys(Keys.RETURN)
            time.sleep(random.uniform(3.5, 7.0))

            # Verify login
            try:
                WebDriverWait(self.driver, 15).until(
                    lambda d: "home" in d.current_url.lower() or
                             len(d.find_elements(By.CSS_SELECTOR, '[data-testid="AppTabBar_Home_Link"]')) > 0
                )
                Logger.success("Login successful!")
                return True
            except:
                Logger.warning("Login status unclear, continuing...")
                return True
        except Exception as e:
            Logger.error(f"Login failed: {str(e)}")
            return False


class TweetParser:
    """Parses tweet elements into structured data"""

    @staticmethod
    def parse(tweet_element) -> Optional[Dict[str, Any]]:
        """Extract data from a tweet element"""
        from selenium.webdriver.common.by import By

        try:
            data = {}

            # User info
            try:
                user_elem = tweet_element.find_element(By.CSS_SELECTOR, '[data-testid="User-Name"]')
                lines = user_elem.text.split('\n')
                data['display_name'] = lines[0] if lines else 'N/A'
                data['username'] = lines[1].lstrip('@') if len(lines) > 1 else 'N/A'
            except:
                data['display_name'] = 'N/A'
                data['username'] = 'N/A'

            # Tweet URL and ID
            try:
                link = tweet_element.find_element(By.CSS_SELECTOR, 'a[href*="/status/"]')
                data['url'] = link.get_attribute('href')
                data['tweet_id'] = data['url'].split('/status/')[-1].split('?')[0].split('/')[0]
            except:
                return None

            # Reply detection
            data['is_reply'] = False
            try:
                tweet_element.find_element(By.XPATH, ".//*[contains(text(), 'Replying to')]")
                data['is_reply'] = True
            except:
                pass

            # Text
            try:
                text_elem = tweet_element.find_element(By.CSS_SELECTOR, '[data-testid="tweetText"]')
                data['text'] = text_elem.text
            except:
                data['text'] = ''

            # Timestamp
            try:
                time_elem = tweet_element.find_element(By.CSS_SELECTOR, 'time')
                data['timestamp'] = time_elem.get_attribute('datetime')
            except:
                data['timestamp'] = None

            # Engagement metrics
            try:
                reply_elem = tweet_element.find_element(By.CSS_SELECTOR, '[data-testid="reply"]')
                label = reply_elem.get_attribute('aria-label') or ''
                match = re.search(r'(\d+)', label)
                data['replies_count'] = int(match.group(1)) if match else 0
            except:
                data['replies_count'] = 0

            try:
                rt_elem = tweet_element.find_element(By.CSS_SELECTOR, '[data-testid="retweet"]')
                label = rt_elem.get_attribute('aria-label') or ''
                match = re.search(r'(\d+)', label)
                data['retweets_count'] = int(match.group(1)) if match else 0
            except:
                data['retweets_count'] = 0

            try:
                like_elem = tweet_element.find_element(By.CSS_SELECTOR, '[data-testid="like"]')
                label = like_elem.get_attribute('aria-label') or ''
                match = re.search(r'(\d+)', label)
                data['likes_count'] = int(match.group(1)) if match else 0
            except:
                data['likes_count'] = 0

            # Media
            has_photo = len(tweet_element.find_elements(By.CSS_SELECTOR, '[data-testid="tweetPhoto"]')) > 0
            has_video = len(tweet_element.find_elements(By.CSS_SELECTOR, '[data-testid="videoPlayer"]')) > 0
            data['has_media'] = has_photo or has_video
            data['media_type'] = 'photo' if has_photo else ('video' if has_video else 'none')

            return data

        except Exception as e:
            Logger.debug(f"Tweet parse error: {e}")
            return None


class AINewsScraper:
    """Main scraper for AI news from Twitter"""

    def __init__(self, username: str, password: str, db: AINewsDatabase):
        self.username = username
        self.password = password
        self.db = db
        self.driver = None
        self.parser = TweetParser()
        self.embedding_gen = EmbeddingGenerator()
        self.seen_ids: Set[str] = set()
        self.tweets_scraped = 0
        self.ai_tweets_found = 0

    def setup_driver(self):
        """Setup Chrome driver with undetected-chromedriver"""
        import undetected_chromedriver as uc

        options = uc.ChromeOptions()
        options.add_argument('--start-maximized')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')

        try:
            self.driver = uc.Chrome(options=options, version_main=131)
        except Exception:
            try:
                self.driver = uc.Chrome(options=options)
            except Exception as e:
                Logger.error(f"Could not start Chrome: {e}")
                raise

        return self.driver

    def scrape_user_timeline(self, username: str, max_tweets: int = 50,
                            max_scrolls: int = 20) -> int:
        """Scrape recent tweets from a user's timeline"""
        from selenium.webdriver.common.by import By

        Logger.info(f"Scraping @{username}...")

        try:
            # Navigate to user's timeline
            self.driver.get(f"https://twitter.com/{username}")
            time.sleep(random.uniform(3, 5))

            # Check if account exists/is accessible
            if "This account doesn" in self.driver.page_source or \
               "Account suspended" in self.driver.page_source:
                Logger.warning(f"@{username} is not accessible")
                return 0

            user_tweets = []
            scroll_count = 0
            no_new_count = 0

            while scroll_count < max_scrolls and len(user_tweets) < max_tweets and no_new_count < 3:
                elements = self.driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')

                new_count = 0
                for elem in elements:
                    try:
                        parsed = self.parser.parse(elem)
                        if not parsed or parsed['tweet_id'] in self.seen_ids:
                            continue

                        self.seen_ids.add(parsed['tweet_id'])

                        # Generate embedding for AI-relevant tweets
                        embedding = None
                        text = parsed.get('text', '')
                        if text and any(kw.lower() in text.lower() for kw in AI_KEYWORDS[:20]):
                            try:
                                embedding = self.embedding_gen.generate(text)
                            except Exception:
                                pass

                        # Save to database
                        is_ai = self.db.save_tweet(parsed, embedding)

                        user_tweets.append(parsed)
                        new_count += 1
                        self.tweets_scraped += 1

                        if is_ai:
                            self.ai_tweets_found += 1

                        if len(user_tweets) >= max_tweets:
                            break

                    except Exception:
                        continue

                if new_count == 0:
                    no_new_count += 1
                else:
                    no_new_count = 0

                # Scroll
                self.driver.execute_script("window.scrollBy(0, 800);")
                time.sleep(random.uniform(1.5, 2.5))
                scroll_count += 1

            Logger.info(f"  Scraped {len(user_tweets)} tweets from @{username}")

            # Update influencer stats
            self.db.update_influencer_stats(username.lower())

            return len(user_tweets)

        except Exception as e:
            Logger.error(f"Error scraping @{username}: {e}")
            return 0

    def scrape_search(self, query: str, max_tweets: int = 100, max_scrolls: int = 30) -> int:
        """Scrape tweets from a search query"""
        from selenium.webdriver.common.by import By
        import urllib.parse

        Logger.info(f"Searching: {query}")

        try:
            encoded_query = urllib.parse.quote(query)
            search_url = f"https://twitter.com/search?q={encoded_query}&src=typed_query&f=live"
            self.driver.get(search_url)
            time.sleep(random.uniform(3, 5))

            search_tweets = []
            scroll_count = 0
            no_new_count = 0

            while scroll_count < max_scrolls and len(search_tweets) < max_tweets and no_new_count < 3:
                elements = self.driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')

                new_count = 0
                for elem in elements:
                    try:
                        parsed = self.parser.parse(elem)
                        if not parsed or parsed['tweet_id'] in self.seen_ids:
                            continue

                        self.seen_ids.add(parsed['tweet_id'])

                        # Generate embedding
                        embedding = None
                        text = parsed.get('text', '')
                        if text:
                            try:
                                embedding = self.embedding_gen.generate(text)
                            except Exception:
                                pass

                        # Save to database
                        is_ai = self.db.save_tweet(parsed, embedding)

                        search_tweets.append(parsed)
                        new_count += 1
                        self.tweets_scraped += 1

                        if is_ai:
                            self.ai_tweets_found += 1

                        if len(search_tweets) >= max_tweets:
                            break

                    except Exception:
                        continue

                if new_count == 0:
                    no_new_count += 1
                else:
                    no_new_count = 0

                # Scroll
                self.driver.execute_script("window.scrollBy(0, 800);")
                time.sleep(random.uniform(1.5, 2.5))
                scroll_count += 1

            Logger.info(f"  Found {len(search_tweets)} tweets for '{query}'")
            return len(search_tweets)

        except Exception as e:
            Logger.error(f"Search error: {e}")
            return 0

    def run_full_scrape(self, max_influencers: int = 20,
                       tweets_per_user: int = 30,
                       search_queries: List[str] = None):
        """Run a complete scraping session"""

        try:
            # Setup
            self.setup_driver()

            # Auth
            auth = TwitterAuth(self.driver, self.username, self.password)
            if not auth.login():
                Logger.error("Login failed, aborting")
                return

            # Get influencers to scrape
            influencers = self.db.get_influencers_to_scrape(limit=max_influencers)
            Logger.info(f"Will scrape {len(influencers)} influencers")

            # Scrape each influencer
            for i, username in enumerate(influencers):
                Logger.info(f"[{i+1}/{len(influencers)}] Processing @{username}")
                self.scrape_user_timeline(username, max_tweets=tweets_per_user)

                # Random delay between users
                time.sleep(random.uniform(3, 7))

            # Run search queries for additional coverage
            if search_queries:
                Logger.info("Running search queries...")
                for query in search_queries:
                    self.scrape_search(query, max_tweets=50)
                    time.sleep(random.uniform(5, 10))

            # Discover new influencers
            self._discover_new_influencers()

            Logger.success(f"Scrape complete! {self.tweets_scraped} tweets ({self.ai_tweets_found} AI-relevant)")

        except KeyboardInterrupt:
            Logger.warning("Interrupted by user")
        except Exception as e:
            Logger.error(f"Scrape error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.driver:
                self.driver.quit()

    def _discover_new_influencers(self):
        """Automatically discover and promote new high-quality influencers"""
        discovered = self.db.get_discovered_influencers(min_mentions=3)

        if not discovered:
            Logger.info("No new influencers to discover")
            return

        Logger.info(f"Found {len(discovered)} potential new influencers")

        # Promote top candidates (those mentioned most by existing influencers)
        for account in discovered[:5]:  # Top 5
            Logger.info(f"  Promoting @{account['username']} (mentioned {account['mention_count']} times)")
            self.db.promote_to_influencer(account['username'])


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class AINewsOrchestrator:
    """Main orchestrator for the AI news scraping system"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.output_dir / 'ai_news.db'
        self.db = None

    def initialize(self):
        """Initialize the system"""
        Logger.info("Initializing AI News Scraper...")

        # Create database
        self.db = AINewsDatabase(self.db_path)

        # Seed initial influencers
        self._seed_influencers()

        Logger.success("System initialized")
        return self

    def _seed_influencers(self):
        """Seed the database with initial influencer list"""
        seed_count = 0
        for category, handles in AI_INFLUENCERS.items():
            for handle in handles:
                self.db.save_influencer(
                    username=handle.lower(),
                    category=category,
                    is_seed=True,
                    discovery_source='initial_seed'
                )
                seed_count += 1

        Logger.success(f"Seeded {seed_count} influencers across {len(AI_INFLUENCERS)} categories")

    def run_scrape(self, username: str, password: str,
                  max_influencers: int = 15,
                  tweets_per_user: int = 25):
        """Run a scraping session"""

        # Default search queries for additional coverage
        search_queries = [
            "AI breakthrough",
            "LLM release",
            "GPT announcement",
            "machine learning research",
            "neural network paper",
        ]

        scraper = AINewsScraper(username, password, self.db)
        scraper.run_full_scrape(
            max_influencers=max_influencers,
            tweets_per_user=tweets_per_user,
            search_queries=search_queries
        )

        return scraper.tweets_scraped, scraper.ai_tweets_found

    def analyze_trends(self, hours: int = 48):
        """Analyze trends from recent tweets"""
        detector = TrendDetector(self.db)
        trends = detector.analyze_trends(hours=hours)

        if trends:
            detector.save_trends_to_db(trends)

            # Print trend summary
            print("\n" + "=" * 60)
            print("EMERGING TRENDS")
            print("=" * 60)

            for i, trend in enumerate(trends[:10], 1):
                status = "[EMERGING]" if trend.get('is_emerging') else ""
                print(f"\n{i}. {', '.join(trend['keywords'][:5])} {status}")
                print(f"   Tweets: {trend['tweet_count']} | Engagement: {trend['total_engagement']}")
                if trend.get('sample_tweets'):
                    print(f"   Sample: {trend['sample_tweets'][0][:100]}...")

        return trends

    def search_similar(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for tweets similar to a query"""
        embedding_gen = EmbeddingGenerator()
        query_embedding = embedding_gen.generate(query)

        results = self.db.similarity_search(query_embedding, limit=limit)

        if results:
            print(f"\nSimilar tweets to: '{query}'")
            print("-" * 40)
            for i, r in enumerate(results, 1):
                print(f"{i}. @{r['username']}: {r['text'][:100]}...")
                print(f"   Distance: {r['distance']:.4f} | Engagement: {r['engagement']}")

        return results

    def export_report(self, filename: str = None) -> Path:
        """Export a summary report"""
        if filename is None:
            filename = f"ai_news_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        stats = self.db.get_stats()

        # Get recent high-engagement tweets
        cursor = self.db.conn.cursor()
        cursor.execute('''
            SELECT * FROM tweets
            WHERE is_ai_relevant = TRUE
            ORDER BY (likes_count + retweets_count) DESC
            LIMIT 50
        ''')
        top_tweets = [dict(row) for row in cursor.fetchall()]

        # Get top influencers
        cursor.execute('''
            SELECT * FROM influencers
            ORDER BY quality_score DESC
            LIMIT 20
        ''')
        top_influencers = [dict(row) for row in cursor.fetchall()]

        report = {
            'generated_at': datetime.now().isoformat(),
            'statistics': stats,
            'top_influencers': top_influencers,
            'top_tweets': top_tweets,
        }

        report_path = self.output_dir / filename
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        Logger.success(f"Report exported to: {report_path}")
        return report_path

    def print_stats(self):
        """Print current statistics"""
        stats = self.db.get_stats()

        print("\n" + "=" * 60)
        print("AI NEWS SCRAPER STATISTICS")
        print("=" * 60)
        print(f"Total tweets:           {stats['total_tweets']}")
        print(f"AI-relevant tweets:     {stats['ai_relevant_tweets']}")
        print(f"Tracked influencers:    {stats['total_influencers']}")
        print(f"  - Seed influencers:   {stats['seed_influencers']}")
        print(f"Potential new sources:  {stats['potential_influencers']}")
        print(f"Topics detected:        {stats['topics_detected']}")
        print("=" * 60)

    def close(self):
        """Cleanup resources"""
        if self.db:
            self.db.close()


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='AI News Scraper with Vector Search')
    parser.add_argument('--scrape', action='store_true', help='Run scraping session')
    parser.add_argument('--trends', action='store_true', help='Analyze trends')
    parser.add_argument('--search', type=str, help='Semantic search query')
    parser.add_argument('--report', action='store_true', help='Generate report')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--max-users', type=int, default=15, help='Max influencers to scrape')
    parser.add_argument('--tweets-per-user', type=int, default=25, help='Tweets per user')
    parser.add_argument('--hours', type=int, default=48, help='Hours for trend analysis')

    args = parser.parse_args()

    # Load environment
    script_dir = Path(__file__).parent
    load_dotenv(script_dir / '.env')

    username = os.getenv('TWITTER_USERNAME')
    password = os.getenv('TWITTER_PASSWORD')

    # Initialize orchestrator
    output_dir = script_dir / 'output_data'
    orchestrator = AINewsOrchestrator(output_dir)
    orchestrator.initialize()

    try:
        if args.scrape:
            if not username or not password:
                Logger.error("Set TWITTER_USERNAME and TWITTER_PASSWORD in .env")
                sys.exit(1)
            orchestrator.run_scrape(
                username, password,
                max_influencers=args.max_users,
                tweets_per_user=args.tweets_per_user
            )

        if args.trends:
            orchestrator.analyze_trends(hours=args.hours)

        if args.search:
            orchestrator.search_similar(args.search)

        if args.report:
            orchestrator.export_report()

        if args.stats or not any([args.scrape, args.trends, args.search, args.report]):
            orchestrator.print_stats()

    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()
