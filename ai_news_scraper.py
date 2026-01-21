# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "undetected-chromedriver>=3.5.0",
#     "python-dotenv>=0.19.0",
#     "setuptools>=65.0.0",
#     "requests>=2.31.0",
#     "sentence-transformers>=2.2.0",
#     "FlagEmbedding>=1.2.0",
#     "sqlite-vec>=0.1.0",
#     "numpy>=1.24.0",
#     "scikit-learn>=1.3.0",
#     "hdbscan>=0.8.33",
#     "beautifulsoup4>=4.12.0",
# ]
# ///

"""
AI News Scraper - Multi-Source AI Content Aggregator with Vector Similarity Search

Scrapes AI-related content from:
- Twitter/X: Curated influencers and AI researchers
- Web Sources: RSS feeds and HTML news sites (paddo.dev, ainativedev.io, vectorlab.dev, accenture newsroom)

Stores in SQLite with VSS, detects emerging trends, and discovers new high-signal sources.
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
_hybrid_embedder = None
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


def get_hybrid_embedder():
    """Lazy load BGE-M3 hybrid embedder with fallback to sentence-transformers"""
    global _hybrid_embedder
    if _hybrid_embedder is None:
        try:
            from FlagEmbedding import BGEM3FlagModel
            import torch

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"[INFO] Loading BGE-M3 model on {device} (first time may download ~2GB)...")

            _hybrid_embedder = BGEM3FlagModel(
                'BAAI/bge-m3',
                use_fp16=True,
                device=device
            )
            print("[INFO] BGE-M3 model loaded successfully")
        except ImportError:
            print("[WARN] FlagEmbedding not available, using sentence-transformers fallback")
            _hybrid_embedder = None
        except Exception as e:
            print(f"[WARN] Could not load BGE-M3: {e}")
            _hybrid_embedder = None
    return _hybrid_embedder


def encode_texts_hybrid(texts: List[str], max_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode texts using BGE-M3 hybrid embeddings (dense + sparse).

    Returns:
        Tuple of (dense_embeddings, sparse_embeddings) as numpy arrays.
        - dense_embeddings: (N, 1024)
        - sparse_embeddings: (N, 256) - top-K representation

        Falls back to (384 dim, None) if BGE-M3 not available.
    """
    embedder = get_hybrid_embedder()

    if embedder is not None:
        try:
            # Use BGE-M3 for hybrid embeddings
            embedding_results = embedder.encode(
                texts,
                batch_size=32,
                max_length=max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False
            )

            dense_embeddings = np.array(embedding_results['dense_vecs'], dtype=np.float32)

            # Convert sparse embeddings to fixed-size dense
            sparse_list = []
            for sparse_dict in embedding_results['lexical_weights']:
                # Sort by weight, take top-256
                sorted_items = sorted(sparse_dict.items(), key=lambda x: x[1], reverse=True)[:256]
                sparse_dense = np.zeros(256, dtype=np.float32)
                for i, (token_id, weight) in enumerate(sorted_items):
                    sparse_dense[i] = weight
                sparse_list.append(sparse_dense)

            sparse_embeddings = np.array(sparse_list, dtype=np.float32)
            return dense_embeddings, sparse_embeddings

        except Exception as e:
            print(f"[WARN] BGE-M3 encoding failed: {e}, falling back to sentence-transformers")

    # Fallback to sentence-transformers (384 dim, no sparse)
    embedder = get_sentence_transformer()
    dense_embeddings = embedder.encode(texts, show_progress_bar=False)
    return np.array(dense_embeddings, dtype=np.float32), None


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


# =============================================================================
# WEB NEWS SOURCES - RSS/HTML Feeds
# =============================================================================

WEB_SOURCES = {
    # -------------------------------------------------------------------------
    # RSS Sources (most reliable, server-side rendered)
    # -------------------------------------------------------------------------
    "paddo_dev": {
        "name": "Emergent Minds (Paddo.dev)",
        "url": "https://paddo.dev/",
        "rss_url": "https://paddo.dev/rss.xml",
        "type": "rss",
        "category": "ai_engineering",
        "description": "AI coding tools, LLM quality, agent design",
    },

    # -------------------------------------------------------------------------
    # HTML Sources (server-side rendered, BeautifulSoup compatible)
    # -------------------------------------------------------------------------
    "vectorlab": {
        "name": "Vector Lab",
        "url": "https://vectorlab.dev/news/",
        "type": "html",
        "category": "ai_news",
        "selectors": {
            "article": ".blog-article, article, [class*='article']",
            "title": "h1, h2, h3, .title, [class*='title']",
            "link": "a[href*='/news/']",
            "date": ".date, time, [class*='date']",
            "description": "p, .subtitle, [class*='description']",
        },
        "description": "Weekly AI updates, model comparisons",
    },

    # -------------------------------------------------------------------------
    # JavaScript-rendered sites (require browser automation - slower)
    # NOTE: These sites use React/Next.js and require Selenium to scrape
    # -------------------------------------------------------------------------
    "ainativedev": {
        "name": "AI Native Dev (Tessl)",
        "url": "https://tessl.io/blog/",
        "type": "html",  # Can parse statically now
        "category": "ai_news",
        "selectors": {
            "article": "article, .article-card, [class*='article'], [class*='post'], [class*='card']",
            "title": "h1, h2, h3, .title, [class*='title'], [class*='headline']",
            "link": "a[href*='/blog/']",
            "date": "time, .date, [class*='date'], [datetime]",
            "description": "p, .excerpt, .summary, [class*='excerpt'], [class*='summary']",
        },
        "description": "AI development news, coding agents, models",
    },
    "accenture_ai": {
        "name": "Accenture Newsroom",
        "url": "https://newsroom.accenture.com/news",
        "type": "js",  # Requires browser automation
        "category": "enterprise_ai",
        "filter_keywords": ["ai", "artificial intelligence", "machine learning", "generative", "automation"],
        "selectors": {
            "article": ".news-item, article, [class*='news'], [class*='article']",
            "title": "h1, h2, h3, .title, [class*='title'], [class*='headline']",
            "link": "a[href*='/news/']",
            "date": ".date, time, [class*='date']",
            "description": "p, .excerpt, [class*='excerpt'], [class*='summary']",
        },
        "description": "Enterprise AI news and announcements",
    },
}


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

# Blocked keywords - content containing these will be filtered out
# Avoid military/political content that could be sensitive
BLOCKED_KEYWORDS = [
    "military", "armed forces", "troops", "defense contractor",
    "air force", "navy", "usmc", "marines",
    "trump", "president",
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

    EMBEDDING_DIM_DENSE = 1024  # BGE-M3 dense dimension
    EMBEDDING_DIM_SPARSE = 256  # BGE-M3 sparse (top-K) dimension
    EMBEDDING_DIM = 384  # Fallback all-MiniLM-L6-v2 dimension

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None
        self._has_vec = False
        self._init_database()

    def _init_database(self):
        """Initialize database with migrations and VSS support"""
        from db_migrations import migrate_ai_news_db

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row

        # Run schema migrations
        migrations_applied = migrate_ai_news_db(self.conn)
        if migrations_applied > 0:
            Logger.success(f"Applied {migrations_applied} migration(s)")

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

        # Vector embeddings tables (handled separately - not in migrations due to extension dependency)
        # We create both legacy (384-dim) and new hybrid tables (1024-dim dense + 256-dim sparse)
        if self._has_vec:
            try:
                # Legacy table for backward compatibility
                self.conn.execute(f'''
                    CREATE VIRTUAL TABLE IF NOT EXISTS tweet_embeddings USING vec0(
                        tweet_id TEXT PRIMARY KEY,
                        embedding float[{self.EMBEDDING_DIM}]
                    )
                ''')

                # New hybrid embedding tables for BGE-M3
                for source_type in ['tweet', 'web_article', 'youtube_video']:
                    # Dense embeddings (1024 dimensions)
                    self.conn.execute(f'''
                        CREATE VIRTUAL TABLE IF NOT EXISTS {source_type}_embeddings_dense USING vec0(
                            id TEXT PRIMARY KEY,
                            embedding float[{self.EMBEDDING_DIM_DENSE}]
                        )
                    ''')
                    # Sparse embeddings (256 dimensions - top-K representation)
                    self.conn.execute(f'''
                        CREATE VIRTUAL TABLE IF NOT EXISTS {source_type}_embeddings_sparse USING vec0(
                            id TEXT PRIMARY KEY,
                            embedding float[{self.EMBEDDING_DIM_SPARSE}]
                        )
                    ''')

                Logger.success("Vector embeddings tables ready (legacy + hybrid)")
            except Exception as e:
                Logger.warning(f"Could not create vector table: {e}")
                self._has_vec = False

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

    def save_tweet(self, tweet_data: Dict[str, Any], embedding: np.ndarray = None,
                   dense_embedding: np.ndarray = None, sparse_embedding: np.ndarray = None):
        """Save a tweet with optional embeddings (legacy or hybrid)"""
        cursor = self.conn.cursor()

        tweet_id = tweet_data.get('tweet_id')
        if not tweet_id:
            return

        # Check for blocked content
        text = tweet_data.get('text', '').lower()
        if any(blocked.lower() in text for blocked in BLOCKED_KEYWORDS):
            print(f"[FILTER] Skipping tweet {tweet_id} - contains blocked keyword")
            return

        # Check AI relevance
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

        # Save legacy embedding if available (384 dim)
        if embedding is not None and self._has_vec:
            try:
                # sqlite-vec expects binary blob format
                embedding_blob = np.asarray(embedding, dtype=np.float32).tobytes()
                cursor.execute('''
                    INSERT OR REPLACE INTO tweet_embeddings (tweet_id, embedding)
                    VALUES (?, ?)
                ''', (tweet_id, embedding_blob))
            except Exception as e:
                Logger.debug(f"Could not save legacy embedding: {e}")

        # Save hybrid embeddings if available (BGE-M3)
        if self._has_vec and dense_embedding is not None:
            try:
                # Save dense embedding (1024 dim) - sqlite-vec expects binary blob
                dense_blob = np.asarray(dense_embedding, dtype=np.float32).tobytes()
                cursor.execute('''
                    INSERT OR REPLACE INTO tweet_embeddings_dense (id, embedding)
                    VALUES (?, ?)
                ''', (tweet_id, dense_blob))

                # Save sparse embedding if available (256 dim)
                if sparse_embedding is not None:
                    sparse_blob = np.asarray(sparse_embedding, dtype=np.float32).tobytes()
                    cursor.execute('''
                        INSERT OR REPLACE INTO tweet_embeddings_sparse (id, embedding)
                        VALUES (?, ?)
                    ''', (tweet_id, sparse_blob))
            except Exception as e:
                Logger.debug(f"Could not save hybrid embeddings: {e}")

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

    def save_web_source(self, source_id: str, name: str, url: str,
                        source_type: str, category: str):
        """Save or update a web source"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO web_sources (source_id, name, url, source_type, category)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
                name = excluded.name,
                url = excluded.url
        ''', (source_id, name, url, source_type, category))
        self.conn.commit()

    def save_web_article(self, article_data: Dict[str, Any], embedding: np.ndarray = None,
                         dense_embedding: np.ndarray = None, sparse_embedding: np.ndarray = None) -> bool:
        """Save a web article with optional embeddings (legacy or hybrid)"""
        cursor = self.conn.cursor()

        url = article_data.get('url')
        if not url:
            return False

        # Generate article_id from URL hash
        article_id = hashlib.md5(url.encode()).hexdigest()[:16]

        # Check for blocked content
        text_to_check = f"{article_data.get('title', '')} {article_data.get('content', '')}".lower()
        if any(blocked.lower() in text_to_check for blocked in BLOCKED_KEYWORDS):
            print(f"[FILTER] Skipping article {url[:50]}... - contains blocked keyword")
            return False

        # Check AI relevance
        text = f"{article_data.get('title', '')} {article_data.get('description', '')}".lower()
        is_ai_relevant = any(kw.lower() in text for kw in AI_KEYWORDS)
        relevance_score = sum(1 for kw in AI_KEYWORDS if kw.lower() in text) / len(AI_KEYWORDS)

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO web_articles (
                    article_id, source_id, source_name, title, url,
                    description, content, author, published_at, category,
                    is_ai_relevant, ai_relevance_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                article_id,
                article_data.get('source_id'),
                article_data.get('source_name'),
                article_data.get('title'),
                url,
                article_data.get('description'),
                article_data.get('content'),
                article_data.get('author'),
                article_data.get('published_at'),
                article_data.get('category'),
                is_ai_relevant,
                relevance_score
            ))
            self.conn.commit()

            # Save hybrid embeddings if available (BGE-M3)
            if self._has_vec and dense_embedding is not None:
                try:
                    # Save dense embedding (1024 dim) - sqlite-vec expects binary blob
                    dense_blob = np.asarray(dense_embedding, dtype=np.float32).tobytes()
                    cursor.execute('''
                        INSERT OR REPLACE INTO web_article_embeddings_dense (id, embedding)
                        VALUES (?, ?)
                    ''', (article_id, dense_blob))

                    # Save sparse embedding if available (256 dim)
                    if sparse_embedding is not None:
                        sparse_blob = np.asarray(sparse_embedding, dtype=np.float32).tobytes()
                        cursor.execute('''
                            INSERT OR REPLACE INTO web_article_embeddings_sparse (id, embedding)
                            VALUES (?, ?)
                        ''', (article_id, sparse_blob))
                    self.conn.commit()
                except Exception as e:
                    Logger.debug(f"Could not save web article embeddings: {e}")

            # Update source article count
            cursor.execute('''
                UPDATE web_sources SET
                    articles_count = (SELECT COUNT(*) FROM web_articles WHERE source_id = ?),
                    last_scraped = CURRENT_TIMESTAMP
                WHERE source_id = ?
            ''', (article_data.get('source_id'), article_data.get('source_id')))
            self.conn.commit()

            return is_ai_relevant
        except sqlite3.IntegrityError:
            # Article already exists
            return False

    def get_recent_web_articles(self, hours: int = 24, ai_only: bool = True) -> List[Dict]:
        """Get recent web articles for analysis"""
        cursor = self.conn.cursor()

        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        query = '''
            SELECT * FROM web_articles
            WHERE scraped_at > ?
        '''
        if ai_only:
            query += ' AND is_ai_relevant = TRUE'
        query += ' ORDER BY published_at DESC'

        cursor.execute(query, (cutoff,))
        return [dict(row) for row in cursor.fetchall()]

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

        # Web article stats
        cursor.execute('SELECT COUNT(*) FROM web_articles')
        stats['total_web_articles'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM web_articles WHERE is_ai_relevant = TRUE')
        stats['ai_relevant_web_articles'] = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM web_sources WHERE is_active = TRUE')
        stats['active_web_sources'] = cursor.fetchone()[0]

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
# WEB SOURCE SCRAPER (RSS/HTML)
# =============================================================================

class WebSourceScraper:
    """Scraper for web news sources (RSS and HTML)"""

    def __init__(self, db: AINewsDatabase):
        self.db = db
        self.session = None
        self.articles_scraped = 0
        self.ai_articles_found = 0

    def _get_session(self):
        """Get or create requests session"""
        if self.session is None:
            import requests
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            })
        return self.session

    def _parse_rss(self, source_id: str, source_config: Dict) -> List[Dict]:
        """Parse RSS feed and extract articles"""
        import xml.etree.ElementTree as ET

        articles = []
        rss_url = source_config.get('rss_url')
        if not rss_url:
            Logger.warning(f"No RSS URL for {source_id}")
            return articles

        try:
            session = self._get_session()
            response = session.get(rss_url, timeout=30)
            response.raise_for_status()

            # Parse XML
            root = ET.fromstring(response.content)

            # Handle both RSS 2.0 and Atom feeds
            items = root.findall('.//item') or root.findall('.//{http://www.w3.org/2005/Atom}entry')

            for item in items:
                try:
                    # RSS 2.0 format
                    title = item.findtext('title') or item.findtext('{http://www.w3.org/2005/Atom}title')
                    link = item.findtext('link') or item.findtext('{http://www.w3.org/2005/Atom}id')
                    description = item.findtext('description') or item.findtext('{http://www.w3.org/2005/Atom}summary') or ''
                    pub_date = item.findtext('pubDate') or item.findtext('{http://www.w3.org/2005/Atom}published')
                    author = item.findtext('author') or item.findtext('{http://purl.org/dc/elements/1.1/}creator')

                    # For Atom, get link from href attribute
                    if not link:
                        link_elem = item.find('{http://www.w3.org/2005/Atom}link')
                        if link_elem is not None:
                            link = link_elem.get('href')

                    if not title or not link:
                        continue

                    # Clean HTML from description
                    description = re.sub(r'<[^>]+>', '', description).strip()

                    articles.append({
                        'source_id': source_id,
                        'source_name': source_config.get('name'),
                        'title': title.strip(),
                        'url': link.strip(),
                        'description': description[:500] if description else '',
                        'author': author,
                        'published_at': pub_date,
                        'category': source_config.get('category'),
                    })
                except Exception as e:
                    Logger.debug(f"Error parsing RSS item: {e}")
                    continue

            Logger.info(f"  Parsed {len(articles)} articles from RSS: {source_config.get('name')}")

        except Exception as e:
            Logger.error(f"RSS fetch error for {source_id}: {e}")

        return articles

    def _parse_html(self, source_id: str, source_config: Dict) -> List[Dict]:
        """Parse HTML page and extract articles"""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            Logger.warning("BeautifulSoup not installed. Run: pip install beautifulsoup4")
            return []

        articles = []
        url = source_config.get('url')
        selectors = source_config.get('selectors', {})
        filter_keywords = source_config.get('filter_keywords', [])

        try:
            session = self._get_session()
            response = session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find article containers
            article_selector = selectors.get('article', 'article')
            article_elements = []

            # Try multiple selectors
            for selector in article_selector.split(','):
                selector = selector.strip()
                found = soup.select(selector)
                if found:
                    article_elements.extend(found)
                    break

            # Fallback: if no article containers found, try link elements directly
            # This handles sites where <a> tags are the article containers (e.g., tessl.io)
            if not article_elements:
                link_selector = selectors.get('link', 'a')
                for sel in link_selector.split(','):
                    sel = sel.strip()
                    found = soup.select(sel)
                    if found:
                        # Filter to only links with headings inside (actual articles)
                        article_elements = [a for a in found if a.select_one('h1, h2, h3, h4')]
                        if article_elements:
                            break

            # Deduplicate
            seen_articles = set()

            for article_elem in article_elements[:50]:  # Limit to 50 articles
                try:
                    # Extract title
                    title = None
                    for title_sel in selectors.get('title', 'h2, h3').split(','):
                        title_elem = article_elem.select_one(title_sel.strip())
                        if title_elem:
                            title = title_elem.get_text(strip=True)
                            break

                    # Extract link
                    link = None
                    # First check if the article element itself is a link
                    if article_elem.name == 'a' and article_elem.get('href'):
                        href = article_elem.get('href', '')
                        if href:
                            if href.startswith('/'):
                                from urllib.parse import urljoin
                                link = urljoin(url, href)
                            else:
                                link = href

                    # Otherwise look for link inside the article
                    if not link:
                        for link_sel in selectors.get('link', 'a').split(','):
                            link_elem = article_elem.select_one(link_sel.strip())
                            if link_elem:
                                href = link_elem.get('href', '')
                                if href:
                                    # Handle relative URLs
                                    if href.startswith('/'):
                                        from urllib.parse import urljoin
                                        link = urljoin(url, href)
                                    else:
                                        link = href
                                    break

                    # If no link found in selectors, try the title element itself
                    if not link and title:
                        title_links = article_elem.select('a')
                        for a in title_links:
                            href = a.get('href', '')
                            if href and any(p in href for p in ['/news/', '/blog/', '/post/', '/article/']):
                                if href.startswith('/'):
                                    from urllib.parse import urljoin
                                    link = urljoin(url, href)
                                else:
                                    link = href
                                break

                    if not title or not link:
                        continue

                    # Deduplicate by link
                    if link in seen_articles:
                        continue
                    seen_articles.add(link)

                    # Extract description
                    description = ''
                    for desc_sel in selectors.get('description', 'p').split(','):
                        desc_elem = article_elem.select_one(desc_sel.strip())
                        if desc_elem:
                            description = desc_elem.get_text(strip=True)
                            break

                    # Extract date
                    published_at = None
                    for date_sel in selectors.get('date', 'time').split(','):
                        date_elem = article_elem.select_one(date_sel.strip())
                        if date_elem:
                            published_at = date_elem.get('datetime') or date_elem.get_text(strip=True)
                            break

                    # Apply keyword filter if specified
                    if filter_keywords:
                        combined_text = f"{title} {description}".lower()
                        if not any(kw.lower() in combined_text for kw in filter_keywords):
                            continue

                    articles.append({
                        'source_id': source_id,
                        'source_name': source_config.get('name'),
                        'title': title,
                        'url': link,
                        'description': description[:500] if description else '',
                        'published_at': published_at,
                        'category': source_config.get('category'),
                    })

                except Exception as e:
                    Logger.debug(f"Error parsing article element: {e}")
                    continue

            Logger.info(f"  Parsed {len(articles)} articles from HTML: {source_config.get('name')}")

        except Exception as e:
            Logger.error(f"HTML fetch error for {source_id}: {e}")

        return articles

    def _parse_js(self, source_id: str, source_config: Dict) -> List[Dict]:
        """Parse JavaScript-rendered page using Selenium"""
        try:
            from bs4 import BeautifulSoup
            import undetected_chromedriver as uc
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
        except ImportError as e:
            Logger.warning(f"Required package not installed: {e}")
            return []

        articles = []
        url = source_config.get('url')
        selectors = source_config.get('selectors', {})
        filter_keywords = source_config.get('filter_keywords', [])

        driver = None
        try:
            Logger.info(f"  Using browser for JS-rendered site: {url}")

            # Setup headless Chrome
            options = uc.ChromeOptions()
            options.add_argument('--headless=new')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')

            driver = uc.Chrome(options=options)
            driver.get(url)

            # Wait for content to load
            time.sleep(5)  # Give JS time to render

            # Get page source after JS execution
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')

            # Find article containers (same logic as HTML parser)
            article_selector = selectors.get('article', 'article')
            article_elements = []

            for selector in article_selector.split(','):
                selector = selector.strip()
                found = soup.select(selector)
                if found:
                    article_elements.extend(found)
                    break

            seen_articles = set()

            for article_elem in article_elements[:50]:
                try:
                    # Extract title
                    title = None
                    for title_sel in selectors.get('title', 'h2, h3').split(','):
                        title_elem = article_elem.select_one(title_sel.strip())
                        if title_elem:
                            title = title_elem.get_text(strip=True)
                            break

                    # Extract link
                    link = None
                    for link_sel in selectors.get('link', 'a').split(','):
                        link_elem = article_elem.select_one(link_sel.strip())
                        if link_elem:
                            href = link_elem.get('href', '')
                            if href:
                                if href.startswith('/'):
                                    from urllib.parse import urljoin
                                    link = urljoin(url, href)
                                else:
                                    link = href
                                break

                    if not link and title:
                        title_links = article_elem.select('a')
                        for a in title_links:
                            href = a.get('href', '')
                            if href and any(p in href for p in ['/news/', '/blog/', '/post/', '/article/']):
                                if href.startswith('/'):
                                    from urllib.parse import urljoin
                                    link = urljoin(url, href)
                                else:
                                    link = href
                                break

                    if not title or not link:
                        continue

                    if link in seen_articles:
                        continue
                    seen_articles.add(link)

                    # Extract description
                    description = ''
                    for desc_sel in selectors.get('description', 'p').split(','):
                        desc_elem = article_elem.select_one(desc_sel.strip())
                        if desc_elem:
                            description = desc_elem.get_text(strip=True)
                            break

                    # Extract date
                    published_at = None
                    for date_sel in selectors.get('date', 'time').split(','):
                        date_elem = article_elem.select_one(date_sel.strip())
                        if date_elem:
                            published_at = date_elem.get('datetime') or date_elem.get_text(strip=True)
                            break

                    # Apply keyword filter if specified
                    if filter_keywords:
                        combined_text = f"{title} {description}".lower()
                        if not any(kw.lower() in combined_text for kw in filter_keywords):
                            continue

                    articles.append({
                        'source_id': source_id,
                        'source_name': source_config.get('name'),
                        'title': title,
                        'url': link,
                        'description': description[:500] if description else '',
                        'published_at': published_at,
                        'category': source_config.get('category'),
                    })

                except Exception as e:
                    Logger.debug(f"Error parsing JS article element: {e}")
                    continue

            Logger.info(f"  Parsed {len(articles)} articles from JS: {source_config.get('name')}")

        except Exception as e:
            Logger.error(f"JS/Browser fetch error for {source_id}: {e}")
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass

        return articles

    def scrape_source(self, source_id: str, source_config: Dict) -> int:
        """Scrape a single source and save articles"""
        Logger.info(f"Scraping: {source_config.get('name')}...")

        # Save source to database
        self.db.save_web_source(
            source_id=source_id,
            name=source_config.get('name'),
            url=source_config.get('url'),
            source_type=source_config.get('type'),
            category=source_config.get('category')
        )

        # Parse based on type
        source_type = source_config.get('type', 'html')
        if source_type == 'rss':
            articles = self._parse_rss(source_id, source_config)
        elif source_type == 'js':
            articles = self._parse_js(source_id, source_config)
        else:
            articles = self._parse_html(source_id, source_config)

        # Save articles
        new_articles = 0
        for article in articles:
            is_ai = self.db.save_web_article(article)
            self.articles_scraped += 1
            new_articles += 1
            if is_ai:
                self.ai_articles_found += 1

        return new_articles

    def scrape_all_sources(self, sources: Dict[str, Dict] = None) -> Tuple[int, int]:
        """Scrape all configured web sources"""
        if sources is None:
            sources = WEB_SOURCES

        Logger.info(f"Scraping {len(sources)} web sources...")

        total_articles = 0
        for source_id, source_config in sources.items():
            try:
                count = self.scrape_source(source_id, source_config)
                total_articles += count
                time.sleep(random.uniform(1, 3))  # Be polite
            except Exception as e:
                Logger.error(f"Error scraping {source_id}: {e}")
                continue

        Logger.success(f"Web scrape complete! {self.articles_scraped} articles ({self.ai_articles_found} AI-relevant)")
        return self.articles_scraped, self.ai_articles_found


# =============================================================================
# TWITTER SCRAPER (Browser Automation)
# =============================================================================

class TwitterAuth:
    """Handles Twitter authentication"""

    def __init__(self, driver, username: str = None, password: str = None,
                 google_email: str = None):
        self.driver = driver
        self.username = username
        self.password = password
        self.google_email = google_email

    def login(self) -> bool:
        """Login to Twitter - uses Google auth if google_email is set"""
        if self.google_email:
            return self.login_with_google()
        else:
            return self.login_with_password()

    def login_with_google(self) -> bool:
        """Login to Twitter using Google authentication"""
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            Logger.info(f"Logging into Twitter via Google ({self.google_email})...")

            self.driver.get("https://twitter.com")
            time.sleep(random.uniform(2, 4))
            self.driver.get("https://twitter.com/i/flow/login")
            time.sleep(random.uniform(3, 5))

            # Look for "Sign in with Google" button
            try:
                # Twitter's Google sign-in button
                google_btn = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH,
                        "//button[contains(., 'Google')]|//div[contains(., 'Sign in with Google')]|"
                        "//span[contains(text(), 'Google')]/ancestor::div[@role='button']|"
                        "//img[contains(@src, 'google')]/ancestor::div[@role='button']"
                    ))
                )
                google_btn.click()
                Logger.info("Clicked Google sign-in button")
                time.sleep(random.uniform(3, 5))
            except Exception as e:
                Logger.warning(f"Could not find Google button: {e}")
                # Try alternative: look for Google iframe or redirect
                try:
                    # Sometimes it's in an iframe
                    iframes = self.driver.find_elements(By.TAG_NAME, 'iframe')
                    for iframe in iframes:
                        src = iframe.get_attribute('src') or ''
                        if 'google' in src.lower():
                            self.driver.switch_to.frame(iframe)
                            break
                except:
                    pass

            # Wait for Google account picker / sign-in page
            time.sleep(random.uniform(2, 4))

            # Handle Google account selection
            # First, check if we're on a Google domain
            current_url = self.driver.current_url
            Logger.info(f"Current URL: {current_url}")

            if 'accounts.google.com' in current_url or 'google.com' in current_url:
                # Look for the account with the specified email
                try:
                    # Wait for account list to load
                    time.sleep(random.uniform(2, 3))

                    # Try multiple selectors for the account
                    account_selectors = [
                        f"//div[contains(text(), '{self.google_email}')]",
                        f"//div[@data-email='{self.google_email}']",
                        f"//*[contains(text(), '{self.google_email}')]",
                        f"//div[contains(@data-identifier, '{self.google_email}')]",
                    ]

                    account_elem = None
                    for selector in account_selectors:
                        try:
                            account_elem = WebDriverWait(self.driver, 5).until(
                                EC.element_to_be_clickable((By.XPATH, selector))
                            )
                            if account_elem:
                                break
                        except:
                            continue

                    if account_elem:
                        account_elem.click()
                        Logger.success(f"Selected Google account: {self.google_email}")
                        time.sleep(random.uniform(3, 5))
                    else:
                        Logger.warning(f"Could not find account {self.google_email}, waiting for manual selection...")
                        # Wait longer for user to manually select
                        time.sleep(15)

                except Exception as e:
                    Logger.warning(f"Account selection issue: {e}")
                    Logger.info("Waiting for manual Google authentication...")
                    time.sleep(15)

            # Wait for redirect back to Twitter
            Logger.info("Waiting for Twitter redirect...")
            try:
                WebDriverWait(self.driver, 30).until(
                    lambda d: 'twitter.com' in d.current_url or 'x.com' in d.current_url
                )
            except:
                Logger.warning("Still waiting for redirect...")
                time.sleep(10)

            # Verify login success
            time.sleep(random.uniform(3, 5))
            try:
                WebDriverWait(self.driver, 15).until(
                    lambda d: "home" in d.current_url.lower() or
                             len(d.find_elements(By.CSS_SELECTOR, '[data-testid="AppTabBar_Home_Link"]')) > 0
                )
                Logger.success("Google login successful!")
                return True
            except:
                Logger.warning("Login status unclear, continuing...")
                return True

        except Exception as e:
            Logger.error(f"Google login failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def login_with_password(self) -> bool:
        """Login to Twitter using username/password"""
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

    def __init__(self, db: AINewsDatabase, username: str = None, password: str = None,
                 google_email: str = None, output_dir: Path = None):
        self.username = username
        self.password = password
        self.google_email = google_email
        self.db = db
        self.driver = None
        self.parser = TweetParser()
        self.embedding_gen = EmbeddingGenerator()
        self.seen_ids: Set[str] = set()
        self.tweets_scraped = 0
        self.ai_tweets_found = 0

        # Session persistence
        self.output_dir = output_dir or Path('output_data')
        self.cookies_path = self.output_dir / 'twitter_cookies.json'
        self.profile_dir = self.output_dir / 'chrome_profile_twitter'

    def setup_driver(self):
        """Setup Chrome driver with undetected-chromedriver and persistent profile"""
        import undetected_chromedriver as uc

        options = uc.ChromeOptions()
        options.add_argument('--start-maximized')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')

        # Use a persistent profile directory for session persistence
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        options.add_argument(f'--user-data-dir={self.profile_dir}')

        # Let undetected-chromedriver auto-detect Chrome version
        try:
            self.driver = uc.Chrome(options=options)
        except Exception as e:
            Logger.error(f"Could not start Chrome: {e}")
            raise

        return self.driver

    def save_cookies(self):
        """Save browser cookies to file"""
        if self.driver:
            try:
                cookies = self.driver.get_cookies()
                with open(self.cookies_path, 'w') as f:
                    json.dump(cookies, f)
                Logger.success(f"Saved {len(cookies)} Twitter cookies")
            except Exception as e:
                Logger.warning(f"Could not save cookies: {e}")

    def load_cookies(self) -> bool:
        """Load cookies from file"""
        if not self.cookies_path.exists():
            return False

        try:
            with open(self.cookies_path, 'r') as f:
                cookies = json.load(f)

            # Navigate to Twitter first (required to set cookies for domain)
            self.driver.get("https://twitter.com")
            time.sleep(2)

            for cookie in cookies:
                # Remove expiry if it's in the past
                if 'expiry' in cookie:
                    del cookie['expiry']
                try:
                    self.driver.add_cookie(cookie)
                except:
                    pass

            Logger.success(f"Loaded {len(cookies)} Twitter cookies")
            return True
        except Exception as e:
            Logger.warning(f"Could not load cookies: {e}")
            return False

    def is_logged_in(self) -> bool:
        """Check if already logged into Twitter"""
        from selenium.webdriver.common.by import By

        try:
            self.driver.get("https://twitter.com/home")
            time.sleep(3)
            # Check if we're on the home timeline (logged in)
            current_url = self.driver.current_url
            if 'login' in current_url or 'flow' in current_url:
                return False
            # Check for home timeline elements
            home_elements = self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="AppTabBar_Home_Link"]')
            return len(home_elements) > 0
        except:
            return False

    def _do_login(self):
        """Perform login and save cookies"""
        auth = TwitterAuth(
            self.driver,
            username=self.username,
            password=self.password,
            google_email=self.google_email
        )
        if auth.login():
            # Save cookies for future sessions
            self.save_cookies()
        else:
            Logger.error("Login failed")

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

            # Try to use existing session first
            Logger.info("Checking for existing Twitter session...")
            if self.is_logged_in():
                Logger.success("Already logged in via persistent session!")
            else:
                # Try loading cookies
                if self.load_cookies():
                    if self.is_logged_in():
                        Logger.success("Logged in via saved cookies!")
                    else:
                        # Need fresh login
                        self._do_login()
                else:
                    # Need fresh login
                    self._do_login()

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

    def run_scrape(self, username: str = None, password: str = None,
                  google_email: str = None,
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

        scraper = AINewsScraper(
            self.db,
            username=username,
            password=password,
            google_email=google_email,
            output_dir=self.output_dir
        )
        scraper.run_full_scrape(
            max_influencers=max_influencers,
            tweets_per_user=tweets_per_user,
            search_queries=search_queries
        )

        return scraper.tweets_scraped, scraper.ai_tweets_found

    def run_web_scrape(self, sources: Dict[str, Dict] = None) -> Tuple[int, int]:
        """Run web source scraping (RSS/HTML)"""
        Logger.info("Starting web source scraping...")

        scraper = WebSourceScraper(self.db)
        articles_scraped, ai_articles = scraper.scrape_all_sources(sources)

        return articles_scraped, ai_articles

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
        print("\nTwitter/X:")
        print(f"  Total tweets:           {stats['total_tweets']}")
        print(f"  AI-relevant tweets:     {stats['ai_relevant_tweets']}")
        print(f"  Tracked influencers:    {stats['total_influencers']}")
        print(f"    - Seed influencers:   {stats['seed_influencers']}")
        print(f"  Potential new sources:  {stats['potential_influencers']}")
        print("\nWeb Sources:")
        print(f"  Total web articles:     {stats.get('total_web_articles', 0)}")
        print(f"  AI-relevant articles:   {stats.get('ai_relevant_web_articles', 0)}")
        print(f"  Active web sources:     {stats.get('active_web_sources', 0)}")
        print("\nAnalysis:")
        print(f"  Topics detected:        {stats['topics_detected']}")
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
    parser.add_argument('--scrape', action='store_true', help='Run Twitter/X scraping session')
    parser.add_argument('--web', action='store_true', help='Scrape web sources (RSS/HTML news sites)')
    parser.add_argument('--all', action='store_true', help='Scrape both Twitter and web sources')
    parser.add_argument('--trends', action='store_true', help='Analyze trends')
    parser.add_argument('--search', type=str, help='Semantic search query')
    parser.add_argument('--report', action='store_true', help='Generate report')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--max-users', type=int, default=15, help='Max influencers to scrape')
    parser.add_argument('--tweets-per-user', type=int, default=25, help='Tweets per user')
    parser.add_argument('--hours', type=int, default=48, help='Hours for trend analysis')
    parser.add_argument('--google-auth', type=str, help='Google email for OAuth login (e.g., user@gmail.com)')

    args = parser.parse_args()

    # Load environment
    script_dir = Path(__file__).parent
    load_dotenv(script_dir / '.env')

    username = os.getenv('TWITTER_USERNAME')
    password = os.getenv('TWITTER_PASSWORD')
    google_email = args.google_auth or os.getenv('GOOGLE_EMAIL')

    # Initialize orchestrator
    output_dir = script_dir / 'output_data'
    orchestrator = AINewsOrchestrator(output_dir)
    orchestrator.initialize()

    try:
        # Handle scraping options
        run_twitter = args.scrape or args.all
        run_web = args.web or args.all

        if run_twitter:
            # Check we have either Google auth or username/password
            if not google_email and (not username or not password):
                Logger.error("Set GOOGLE_EMAIL or TWITTER_USERNAME/TWITTER_PASSWORD in .env")
                Logger.info("Or use --google-auth=your@email.com")
                sys.exit(1)
            orchestrator.run_scrape(
                username=username,
                password=password,
                google_email=google_email,
                max_influencers=args.max_users,
                tweets_per_user=args.tweets_per_user
            )

        if run_web:
            # Web scraping doesn't require authentication
            orchestrator.run_web_scrape()

        if args.trends:
            orchestrator.analyze_trends(hours=args.hours)

        if args.search:
            orchestrator.search_similar(args.search)

        if args.report:
            orchestrator.export_report()

        if args.stats or not any([args.scrape, args.web, args.all, args.trends, args.search, args.report]):
            orchestrator.print_stats()

    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()
