# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "undetected-chromedriver>=3.5.0",
#     "python-dotenv>=0.19.0",
#     "setuptools>=65.0.0",
#     "beautifulsoup4>=4.12.0",
#     "requests>=2.31.0",
#     "sentence-transformers>=2.2.0",
#     "sqlite-vec>=0.1.0",
#     "numpy>=1.24.0",
#     "youtube-transcript-api>=1.0.0",
#     "dateparser>=1.2.0",
# ]
# ///

"""
Discord Link Scraper - Extracts links from specific users in a Discord server

Uses Discord's REST API with user token for reliable message retrieval.
Scrapes discovered links and stores content in vector database for semantic search.

Based on patterns from:
- https://github.com/LAION-AI/Discord-Scrapers
- https://github.com/lorenz234/Discord-Data-Scraping
"""

import os
import sys
import json
import time
import random
import sqlite3
import re
import struct
import base64
import sqlite3 as sqlite
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin
import hashlib

from dotenv import load_dotenv
import numpy as np

load_dotenv()


# =============================================================================
# LOGGING
# =============================================================================

class Logger:
    """Simple colored logging"""
    COLORS = {
        'RED': '\033[91m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'BLUE': '\033[94m',
        'MAGENTA': '\033[95m',
        'CYAN': '\033[96m',
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
    }

    @classmethod
    def _log(cls, color: str, prefix: str, msg: str):
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"{cls.COLORS[color]}[{timestamp}] {prefix}{cls.COLORS['RESET']} {msg}")

    @classmethod
    def info(cls, msg): cls._log('BLUE', 'ℹ', msg)
    @classmethod
    def success(cls, msg): cls._log('GREEN', '✓', msg)
    @classmethod
    def warning(cls, msg): cls._log('YELLOW', '⚠', msg)
    @classmethod
    def error(cls, msg): cls._log('RED', '✗', msg)
    @classmethod
    def debug(cls, msg):
        if os.getenv('DEBUG'):
            cls._log('MAGENTA', '⚙', msg)


# =============================================================================
# EMBEDDINGS
# =============================================================================

_sentence_transformer = None

def get_sentence_transformer():
    """Lazy load sentence transformer model"""
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("[INFO] Loading sentence-transformers model...")
            _sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            print("[INFO] Model loaded successfully")
        except Exception as e:
            print(f"[WARN] Could not load sentence-transformers: {e}")
            _sentence_transformer = FallbackEmbedder()
    return _sentence_transformer


class FallbackEmbedder:
    """Fallback embedder using TF-IDF when sentence-transformers unavailable"""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.svd = TruncatedSVD(n_components=384, random_state=42)
        self._corpus = []

    def encode(self, sentences, show_progress_bar=False, batch_size=32):
        """Generate embeddings using TF-IDF + SVD"""
        if isinstance(sentences, str):
            sentences = [sentences]

        self._corpus.extend(sentences)

        if len(self._corpus) < 10:
            return np.random.randn(len(sentences), 384).astype(np.float32)

        try:
            tfidf_matrix = self.vectorizer.fit_transform(self._corpus)
            if tfidf_matrix.shape[1] >= 384:
                reduced = self.svd.fit_transform(tfidf_matrix)
            else:
                reduced = np.zeros((len(self._corpus), 384))
                temp = tfidf_matrix.toarray()
                reduced[:, :temp.shape[1]] = temp

            result = reduced[-len(sentences):].astype(np.float32)
            return result if len(sentences) > 1 else result[0]
        except Exception:
            return np.random.randn(len(sentences), 384).astype(np.float32)


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers"""
    EMBEDDING_DIM = 384

    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = get_sentence_transformer()
        return self._model

    def generate(self, text: str) -> np.ndarray:
        return self.model.encode(text, show_progress_bar=False)

    def generate_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)


# =============================================================================
# DATABASE WITH VECTOR SEARCH
# =============================================================================

class DiscordLinksDatabase:
    """SQLite database for storing Discord links with vector embeddings"""
    EMBEDDING_DIM = 384

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None
        self._has_vec = False
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
            self._has_vec = False

        cursor = self.conn.cursor()

        # Discord servers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS discord_servers (
                server_id TEXT PRIMARY KEY,
                server_name TEXT,
                first_scraped TEXT DEFAULT CURRENT_TIMESTAMP,
                last_scraped TEXT
            )
        ''')

        # Discord channels table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS discord_channels (
                channel_id TEXT PRIMARY KEY,
                server_id TEXT,
                channel_name TEXT,
                channel_type TEXT,
                first_scraped TEXT DEFAULT CURRENT_TIMESTAMP,
                last_scraped TEXT,
                FOREIGN KEY (server_id) REFERENCES discord_servers(server_id)
            )
        ''')

        # Discord users being tracked
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS discord_users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE,
                display_name TEXT,
                discriminator TEXT,
                messages_scraped INTEGER DEFAULT 0,
                links_found INTEGER DEFAULT 0,
                first_seen TEXT DEFAULT CURRENT_TIMESTAMP,
                last_seen TEXT
            )
        ''')

        # Discord messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS discord_messages (
                message_id TEXT PRIMARY KEY,
                channel_id TEXT,
                server_id TEXT,
                user_id TEXT,
                username TEXT,
                content TEXT,
                timestamp TEXT,
                has_links BOOLEAN DEFAULT FALSE,
                link_count INTEGER DEFAULT 0,
                scraped_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (channel_id) REFERENCES discord_channels(channel_id),
                FOREIGN KEY (server_id) REFERENCES discord_servers(server_id)
            )
        ''')

        # Links extracted from messages
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS discord_links (
                link_id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT,
                url TEXT,
                domain TEXT,
                username TEXT,
                server_id TEXT,
                channel_id TEXT,
                found_at TEXT DEFAULT CURRENT_TIMESTAMP,
                -- Scraped content fields
                is_scraped BOOLEAN DEFAULT FALSE,
                scraped_at TEXT,
                page_title TEXT,
                page_description TEXT,
                page_content TEXT,
                content_type TEXT,
                scrape_error TEXT,
                -- Temporal metadata
                published_at TEXT,
                author_name TEXT,
                -- Content-specific metadata
                link_type TEXT,  -- 'youtube', 'github', 'article', 'social', 'other'
                duration_seconds INTEGER,  -- for video content
                FOREIGN KEY (message_id) REFERENCES discord_messages(message_id),
                UNIQUE(url, message_id)
            )
        ''')

        # Add columns if they don't exist (for existing databases)
        try:
            cursor.execute('ALTER TABLE discord_links ADD COLUMN published_at TEXT')
        except: pass
        try:
            cursor.execute('ALTER TABLE discord_links ADD COLUMN author_name TEXT')
        except: pass
        try:
            cursor.execute('ALTER TABLE discord_links ADD COLUMN link_type TEXT')
        except: pass
        try:
            cursor.execute('ALTER TABLE discord_links ADD COLUMN duration_seconds INTEGER')
        except: pass

        # Vector embeddings for scraped content
        if self._has_vec:
            try:
                cursor.execute(f'''
                    CREATE VIRTUAL TABLE IF NOT EXISTS link_embeddings USING vec0(
                        link_id INTEGER PRIMARY KEY,
                        embedding float[{self.EMBEDDING_DIM}]
                    )
                ''')
                Logger.success("Vector embeddings table created")
            except Exception as e:
                Logger.warning(f"Could not create vector table: {e}")
                self._has_vec = False

        # Scrape history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS discord_scrape_history (
                scrape_id INTEGER PRIMARY KEY AUTOINCREMENT,
                server_id TEXT,
                usernames TEXT,
                started_at TEXT,
                completed_at TEXT,
                messages_found INTEGER DEFAULT 0,
                links_found INTEGER DEFAULT 0,
                links_scraped INTEGER DEFAULT 0,
                status TEXT DEFAULT 'running'
            )
        ''')

        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_discord_links_url ON discord_links(url)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_discord_links_domain ON discord_links(domain)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_discord_links_username ON discord_links(username)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_discord_links_scraped ON discord_links(is_scraped)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_discord_messages_username ON discord_messages(username)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_discord_messages_timestamp ON discord_messages(timestamp)')

        self.conn.commit()
        Logger.success(f"Database initialized: {self.db_path}")

    def save_server(self, server_id: str, server_name: str = None):
        """Save or update a server"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO discord_servers (server_id, server_name, last_scraped)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(server_id) DO UPDATE SET
                server_name = COALESCE(excluded.server_name, server_name),
                last_scraped = CURRENT_TIMESTAMP
        ''', (server_id, server_name))
        self.conn.commit()

    def save_channel(self, channel_id: str, server_id: str, channel_name: str = None, channel_type: str = None):
        """Save or update a channel"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO discord_channels (channel_id, server_id, channel_name, channel_type, last_scraped)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(channel_id) DO UPDATE SET
                channel_name = COALESCE(excluded.channel_name, channel_name),
                channel_type = COALESCE(excluded.channel_type, channel_type),
                last_scraped = CURRENT_TIMESTAMP
        ''', (channel_id, server_id, channel_name, channel_type))
        self.conn.commit()

    def save_user(self, user_id: str, username: str, display_name: str = None, discriminator: str = None):
        """Save or update a tracked user"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO discord_users (user_id, username, display_name, discriminator, last_seen)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(username) DO UPDATE SET
                user_id = COALESCE(excluded.user_id, user_id),
                display_name = COALESCE(excluded.display_name, display_name),
                discriminator = COALESCE(excluded.discriminator, discriminator),
                last_seen = CURRENT_TIMESTAMP
        ''', (user_id, username.lower(), display_name, discriminator))
        self.conn.commit()

    def save_message(self, message_data: Dict[str, Any]) -> bool:
        """Save a message and return True if it's new"""
        cursor = self.conn.cursor()
        message_id = message_data.get('message_id')
        if not message_id:
            return False

        # Check if exists
        cursor.execute('SELECT message_id FROM discord_messages WHERE message_id = ?', (message_id,))
        if cursor.fetchone():
            return False

        content = message_data.get('content', '')
        links = self._extract_links(content)

        cursor.execute('''
            INSERT INTO discord_messages (
                message_id, channel_id, server_id, user_id, username,
                content, timestamp, has_links, link_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            message_id,
            message_data.get('channel_id'),
            message_data.get('server_id'),
            message_data.get('user_id'),
            message_data.get('username', '').lower(),
            content,
            message_data.get('timestamp'),
            len(links) > 0,
            len(links)
        ))

        # Update user stats
        cursor.execute('''
            UPDATE discord_users
            SET messages_scraped = messages_scraped + 1,
                links_found = links_found + ?
            WHERE username = ?
        ''', (len(links), message_data.get('username', '').lower()))

        self.conn.commit()
        return True

    def save_link(self, link_data: Dict[str, Any]) -> Optional[int]:
        """Save a link and return link_id if new, None otherwise"""
        cursor = self.conn.cursor()
        url = link_data.get('url')
        message_id = link_data.get('message_id')

        # Check if exists
        cursor.execute('SELECT link_id FROM discord_links WHERE url = ? AND message_id = ?', (url, message_id))
        existing = cursor.fetchone()
        if existing:
            return None

        domain = urlparse(url).netloc if url else None

        cursor.execute('''
            INSERT INTO discord_links (
                message_id, url, domain, username, server_id, channel_id
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            message_id,
            url,
            domain,
            link_data.get('username'),
            link_data.get('server_id'),
            link_data.get('channel_id')
        ))
        self.conn.commit()
        return cursor.lastrowid

    def update_link_content(self, link_id: int, content_data: Dict[str, Any], embedding: np.ndarray = None):
        """Update a link with scraped content and embedding"""
        cursor = self.conn.cursor()

        cursor.execute('''
            UPDATE discord_links SET
                is_scraped = TRUE,
                scraped_at = CURRENT_TIMESTAMP,
                page_title = ?,
                page_description = ?,
                page_content = ?,
                content_type = ?,
                scrape_error = ?,
                published_at = ?,
                author_name = ?,
                link_type = ?,
                duration_seconds = ?
            WHERE link_id = ?
        ''', (
            content_data.get('title'),
            content_data.get('description'),
            content_data.get('content'),
            content_data.get('content_type'),
            content_data.get('error'),
            content_data.get('published_at'),
            content_data.get('author_name'),
            content_data.get('link_type'),
            content_data.get('duration_seconds'),
            link_id
        ))

        # Save embedding if available
        if embedding is not None and self._has_vec:
            try:
                embedding_list = embedding.tolist()
                cursor.execute('''
                    INSERT OR REPLACE INTO link_embeddings (link_id, embedding)
                    VALUES (?, ?)
                ''', (link_id, json.dumps(embedding_list)))
            except Exception as e:
                Logger.debug(f"Could not save embedding: {e}")

        self.conn.commit()

    def _extract_links(self, text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = r'https?://[^\s<>"\')\]]+[^\s<>"\')\].,;:!?]'
        urls = re.findall(url_pattern, text)
        # Filter out Discord internal links
        external_urls = [
            url for url in urls
            if not url.startswith('https://discord.com')
            and not url.startswith('https://cdn.discordapp.com')
            and not url.startswith('https://media.discordapp.net')
        ]
        return external_urls

    def get_unscraped_links(self, limit: int = 100) -> List[Dict]:
        """Get links that haven't been scraped yet"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT link_id, url, domain, username, message_id
            FROM discord_links
            WHERE is_scraped = FALSE
            ORDER BY found_at DESC
            LIMIT ?
        ''', (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def get_all_links(self, username: str = None, domain: str = None,
                      is_scraped: bool = None) -> List[Dict]:
        """Get links with optional filters"""
        cursor = self.conn.cursor()
        query = 'SELECT * FROM discord_links WHERE 1=1'
        params = []

        if username:
            query += ' AND username = ?'
            params.append(username.lower())
        if domain:
            query += ' AND domain LIKE ?'
            params.append(f'%{domain}%')
        if is_scraped is not None:
            query += ' AND is_scraped = ?'
            params.append(is_scraped)

        query += ' ORDER BY found_at DESC'
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def similarity_search(self, query_embedding: np.ndarray, limit: int = 20) -> List[Dict]:
        """Find similar content using vector similarity search"""
        if not self._has_vec:
            Logger.warning("VSS not available")
            return []

        cursor = self.conn.cursor()

        try:
            embedding_json = json.dumps(query_embedding.tolist())
            cursor.execute(f'''
                SELECT
                    e.link_id,
                    e.distance,
                    l.url,
                    l.domain,
                    l.page_title,
                    l.page_description,
                    l.username
                FROM link_embeddings e
                JOIN discord_links l ON e.link_id = l.link_id
                WHERE e.embedding MATCH ?
                  AND k = ?
                ORDER BY e.distance
            ''', (embedding_json, limit))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'link_id': row[0],
                    'distance': row[1],
                    'url': row[2],
                    'domain': row[3],
                    'title': row[4],
                    'description': row[5],
                    'username': row[6],
                    'similarity': 1 - row[1]  # Convert distance to similarity
                })
            return results

        except Exception as e:
            Logger.error(f"Similarity search error: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get scraping statistics"""
        cursor = self.conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM discord_messages')
        messages = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM discord_links')
        links = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM discord_links WHERE is_scraped = TRUE')
        scraped = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(DISTINCT domain) FROM discord_links')
        domains = cursor.fetchone()[0]

        cursor.execute('''
            SELECT username, COUNT(*) as count
            FROM discord_links
            GROUP BY username
            ORDER BY count DESC
        ''')
        by_user = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute('''
            SELECT domain, COUNT(*) as count
            FROM discord_links
            GROUP BY domain
            ORDER BY count DESC
            LIMIT 20
        ''')
        top_domains = {row[0]: row[1] for row in cursor.fetchall()}

        return {
            'total_messages': messages,
            'total_links': links,
            'scraped_links': scraped,
            'unique_domains': domains,
            'links_by_user': by_user,
            'top_domains': top_domains
        }

    def start_scrape(self, server_id: str, usernames: List[str]) -> int:
        """Record start of a scrape session"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO discord_scrape_history (server_id, usernames, started_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (server_id, json.dumps(usernames)))
        self.conn.commit()
        return cursor.lastrowid

    def end_scrape(self, scrape_id: int, messages: int, links: int, links_scraped: int = 0, status: str = 'completed'):
        """Record end of a scrape session"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE discord_scrape_history
            SET completed_at = CURRENT_TIMESTAMP,
                messages_found = ?,
                links_found = ?,
                links_scraped = ?,
                status = ?
            WHERE scrape_id = ?
        ''', (messages, links, links_scraped, status, scrape_id))
        self.conn.commit()


# =============================================================================
# DISCORD API CLIENT
# =============================================================================

class DiscordAPI:
    """Discord REST API client using user token"""

    BASE_URL = "https://discord.com/api/v10"

    def __init__(self, token: str):
        self.token = token
        self.session = None

    def _get_session(self):
        """Get or create requests session"""
        if self.session is None:
            import requests
            self.session = requests.Session()
            self.session.headers.update({
                'Authorization': self.token,
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Content-Type': 'application/json',
            })
        return self.session

    def get_user_info(self) -> Optional[Dict]:
        """Get current user info to verify token"""
        try:
            resp = self._get_session().get(f"{self.BASE_URL}/users/@me")
            if resp.status_code == 200:
                return resp.json()
            Logger.error(f"Token validation failed: {resp.status_code}")
            return None
        except Exception as e:
            Logger.error(f"API error: {e}")
            return None

    def get_guild(self, guild_id: str) -> Optional[Dict]:
        """Get server/guild info"""
        try:
            resp = self._get_session().get(f"{self.BASE_URL}/guilds/{guild_id}")
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            Logger.error(f"Get guild error: {e}")
            return None

    def get_guild_channels(self, guild_id: str) -> List[Dict]:
        """Get all channels in a guild"""
        try:
            resp = self._get_session().get(f"{self.BASE_URL}/guilds/{guild_id}/channels")
            if resp.status_code == 200:
                return resp.json()
            return []
        except Exception as e:
            Logger.error(f"Get channels error: {e}")
            return []

    def search_messages(self, guild_id: str, author_id: str = None, has_link: bool = False,
                       limit: int = 25, offset: int = 0) -> Dict:
        """Search messages in a guild"""
        params = {
            'limit': min(limit, 25),  # Discord max is 25
            'offset': offset,
        }
        if author_id:
            params['author_id'] = author_id
        if has_link:
            params['has'] = 'link'

        try:
            resp = self._get_session().get(
                f"{self.BASE_URL}/guilds/{guild_id}/messages/search",
                params=params
            )
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                # Rate limited
                retry_after = resp.json().get('retry_after', 5)
                Logger.warning(f"Rate limited, waiting {retry_after}s...")
                time.sleep(retry_after)
                return self.search_messages(guild_id, author_id, has_link, limit, offset)
            else:
                Logger.debug(f"Search failed: {resp.status_code} - {resp.text}")
                return {'messages': [], 'total_results': 0}
        except Exception as e:
            Logger.error(f"Search error: {e}")
            return {'messages': [], 'total_results': 0}

    def get_channel_messages(self, channel_id: str, limit: int = 100, before: str = None) -> List[Dict]:
        """Get messages from a channel"""
        params = {'limit': min(limit, 100)}
        if before:
            params['before'] = before

        try:
            resp = self._get_session().get(
                f"{self.BASE_URL}/channels/{channel_id}/messages",
                params=params
            )
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                retry_after = resp.json().get('retry_after', 5)
                Logger.warning(f"Rate limited, waiting {retry_after}s...")
                time.sleep(retry_after)
                return self.get_channel_messages(channel_id, limit, before)
            return []
        except Exception as e:
            Logger.error(f"Get messages error: {e}")
            return []

    def get_user_by_username(self, guild_id: str, username: str) -> Optional[Dict]:
        """Search for a user by username in a guild"""
        try:
            resp = self._get_session().get(
                f"{self.BASE_URL}/guilds/{guild_id}/members/search",
                params={'query': username, 'limit': 10}
            )
            if resp.status_code == 200:
                members = resp.json()
                for member in members:
                    user = member.get('user', {})
                    if user.get('username', '').lower() == username.lower():
                        return user
                    # Also check global_name
                    if user.get('global_name', '').lower() == username.lower():
                        return user
            return None
        except Exception as e:
            Logger.error(f"User search error: {e}")
            return None


# =============================================================================
# CHROME COOKIE EXTRACTOR
# =============================================================================

def get_discord_token_from_chrome() -> Optional[str]:
    """Extract Discord token from Chrome's local storage"""
    import platform

    system = platform.system()

    if system == "Darwin":  # macOS
        chrome_path = Path.home() / "Library/Application Support/Google/Chrome"
    elif system == "Windows":
        chrome_path = Path(os.environ.get('LOCALAPPDATA', '')) / "Google/Chrome/User Data"
    else:  # Linux
        chrome_path = Path.home() / ".config/google-chrome"

    # Try different profile folders
    profiles = ["Default", "Profile 1", "Profile 2"]

    for profile in profiles:
        leveldb_path = chrome_path / profile / "Local Storage" / "leveldb"
        if not leveldb_path.exists():
            continue

        try:
            # Read LevelDB files for Discord token
            for ldb_file in leveldb_path.glob("*.ldb"):
                try:
                    with open(ldb_file, 'rb') as f:
                        content = f.read().decode('utf-8', errors='ignore')
                        # Look for Discord token pattern
                        # Tokens are usually in format: mfa.xxx or just a long base64 string
                        patterns = [
                            r'[\"\']([a-zA-Z0-9_-]{24}\.[a-zA-Z0-9_-]{6}\.[a-zA-Z0-9_-]{27})[\"\']',  # User token
                            r'[\"\']mfa\.[a-zA-Z0-9_-]{84}[\"\']',  # MFA token
                            r'[\"\']([a-zA-Z0-9_-]{26}\.[a-zA-Z0-9_-]{6}\.[a-zA-Z0-9_-]{38})[\"\']',  # New format
                        ]
                        for pattern in patterns:
                            matches = re.findall(pattern, content)
                            if matches:
                                token = matches[0] if isinstance(matches[0], str) else matches[0][0]
                                Logger.success(f"Found Discord token in {profile}")
                                return token
                except:
                    continue

            # Also try .log files
            for log_file in leveldb_path.glob("*.log"):
                try:
                    with open(log_file, 'rb') as f:
                        content = f.read().decode('utf-8', errors='ignore')
                        for pattern in [
                            r'[\"\']([a-zA-Z0-9_-]{24}\.[a-zA-Z0-9_-]{6}\.[a-zA-Z0-9_-]{27})[\"\']',
                            r'[\"\']([a-zA-Z0-9_-]{26}\.[a-zA-Z0-9_-]{6}\.[a-zA-Z0-9_-]{38})[\"\']',
                        ]:
                            matches = re.findall(pattern, content)
                            if matches:
                                token = matches[0]
                                Logger.success(f"Found Discord token in {profile}")
                                return token
                except:
                    continue

        except Exception as e:
            Logger.debug(f"Error reading {profile}: {e}")
            continue

    return None


def get_discord_token_interactive() -> Optional[str]:
    """Get Discord token interactively using browser"""
    import undetected_chromedriver as uc
    from selenium.webdriver.common.by import By

    Logger.info("Opening browser to extract Discord token...")
    Logger.info("Please ensure you're logged into Discord...")

    options = uc.ChromeOptions()
    options.add_argument('--start-maximized')

    # Use default Chrome profile to inherit login
    chrome_path = Path.home() / "Library/Application Support/Google/Chrome"
    if chrome_path.exists():
        options.add_argument(f'--user-data-dir={chrome_path}')

    try:
        driver = uc.Chrome(options=options)
        driver.get("https://discord.com/app")
        time.sleep(5)

        # Try to extract token from local storage
        token = driver.execute_script('''
            return (webpackChunkdiscord_app.push([[''],{},e=>{m=[];for(let c in e.c)m.push(e.c[c])}]),m).find(m=>m?.exports?.default?.getToken!==void 0).exports.default.getToken()
        ''')

        if token:
            Logger.success("Extracted Discord token from browser!")
            driver.quit()
            return token

        driver.quit()
    except Exception as e:
        Logger.warning(f"Browser extraction failed: {e}")

    return None


# =============================================================================
# CONTENT SCRAPER WITH BESPOKE HANDLERS
# =============================================================================

class ContentScraper:
    """Smart content scraper that routes to bespoke scrapers based on URL type"""

    # URL patterns for routing
    YOUTUBE_DOMAINS = ['youtube.com', 'www.youtube.com', 'youtu.be', 'm.youtube.com']
    GITHUB_DOMAINS = ['github.com', 'gist.github.com', 'raw.githubusercontent.com']
    SOCIAL_DOMAINS = ['twitter.com', 'x.com', 'linkedin.com', 'www.linkedin.com']

    def __init__(self, embedding_gen: EmbeddingGenerator = None):
        self.embedding_gen = embedding_gen or EmbeddingGenerator()
        self.session = None

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

    def _detect_link_type(self, url: str) -> str:
        """Detect the type of content based on URL"""
        domain = urlparse(url).netloc.lower()

        if any(d in domain for d in self.YOUTUBE_DOMAINS):
            return 'youtube'
        if any(d in domain for d in self.GITHUB_DOMAINS):
            return 'github'
        if any(d in domain for d in self.SOCIAL_DOMAINS):
            return 'social'
        return 'article'

    def _extract_youtube_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL"""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})',
            r'youtube\.com/shorts/([a-zA-Z0-9_-]{11})',  # YouTube Shorts
            r'youtu\.be/([a-zA-Z0-9_-]{11})',  # Short URL with query params
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def scrape_youtube(self, url: str) -> Dict[str, Any]:
        """Scrape YouTube video - get transcript and metadata"""
        result = {
            'url': url,
            'title': None,
            'description': None,
            'content': None,
            'content_type': 'video/youtube',
            'link_type': 'youtube',
            'error': None,
            'published_at': None,
            'author_name': None,
            'duration_seconds': None,
        }

        video_id = self._extract_youtube_id(url)
        if not video_id:
            result['error'] = 'Could not extract video ID'
            return result

        # Get transcript using youtube-transcript-api
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api.formatters import TextFormatter

            ytt_api = YouTubeTranscriptApi()
            # Try to get transcript in order of preference
            transcript = None
            for lang in ['en', 'en-US', 'en-GB']:
                try:
                    transcript = ytt_api.fetch(video_id, languages=[lang])
                    break
                except:
                    continue

            if not transcript:
                # Try auto-generated
                try:
                    transcript_list = ytt_api.list(video_id)
                    transcript = transcript_list.find_generated_transcript(['en']).fetch()
                except:
                    pass

            if transcript:
                # Format transcript as plain text
                formatter = TextFormatter()
                transcript_text = formatter.format_transcript(transcript)
                result['content'] = transcript_text[:15000]  # Limit to 15k chars for longer videos

                # Calculate approximate duration from transcript
                if transcript:
                    last_entry = transcript[-1] if isinstance(transcript, list) else None
                    if last_entry and 'start' in last_entry and 'duration' in last_entry:
                        result['duration_seconds'] = int(last_entry['start'] + last_entry['duration'])

                Logger.success(f"Got transcript for YouTube video: {video_id}")
            else:
                result['error'] = 'No transcript available'
                Logger.warning(f"No transcript for YouTube video: {video_id}")

        except ImportError:
            result['error'] = 'youtube-transcript-api not installed'
            Logger.warning("youtube-transcript-api not installed. Install with: pip install youtube-transcript-api")
        except Exception as e:
            result['error'] = f'Transcript error: {str(e)}'
            Logger.debug(f"YouTube transcript error: {e}")

        # Get video metadata from oEmbed API (no API key needed)
        try:
            oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
            resp = self._get_session().get(oembed_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                result['title'] = data.get('title')
                result['author_name'] = data.get('author_name')
        except Exception as e:
            Logger.debug(f"YouTube oEmbed error: {e}")

        # Try to get more metadata from page
        if not result['title']:
            try:
                resp = self._get_session().get(url, timeout=15)
                if resp.status_code == 200:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(resp.content, 'html.parser')

                    title_tag = soup.find('meta', property='og:title')
                    if title_tag:
                        result['title'] = title_tag.get('content')

                    desc_tag = soup.find('meta', property='og:description')
                    if desc_tag:
                        result['description'] = desc_tag.get('content')

                    # Try to get publish date
                    date_tag = soup.find('meta', itemprop='datePublished')
                    if date_tag:
                        result['published_at'] = date_tag.get('content')
            except:
                pass

        return result

    def scrape_github(self, url: str) -> Dict[str, Any]:
        """Scrape GitHub repository or gist"""
        result = {
            'url': url,
            'title': None,
            'description': None,
            'content': None,
            'content_type': 'text/html',
            'link_type': 'github',
            'error': None,
            'published_at': None,
            'author_name': None,
        }

        try:
            from bs4 import BeautifulSoup
            session = self._get_session()

            # For raw content, fetch directly
            if 'raw.githubusercontent.com' in url:
                resp = session.get(url, timeout=30)
                if resp.status_code == 200:
                    result['content'] = resp.text[:15000]
                    result['content_type'] = resp.headers.get('Content-Type', 'text/plain')
                    # Extract repo name from URL
                    parts = url.split('/')
                    if len(parts) >= 5:
                        result['author_name'] = parts[3]
                        result['title'] = f"{parts[3]}/{parts[4]}"
                return result

            # Regular GitHub page
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, 'html.parser')

            # Title from page
            title_tag = soup.find('title')
            if title_tag:
                result['title'] = title_tag.get_text(strip=True)

            # Try to get repo description
            about = soup.select_one('.f4.my-3')
            if about:
                result['description'] = about.get_text(strip=True)

            # For repo main page, get README content
            readme = soup.select_one('article.markdown-body')
            if readme:
                result['content'] = readme.get_text(separator=' ', strip=True)[:10000]

            # For issue/PR pages
            issue_body = soup.select_one('.js-comment-body')
            if issue_body and not result['content']:
                result['content'] = issue_body.get_text(separator=' ', strip=True)[:10000]

            # Get author
            author_link = soup.select_one('.author a') or soup.select_one('[rel="author"]')
            if author_link:
                result['author_name'] = author_link.get_text(strip=True)

            # Get date
            time_tag = soup.select_one('relative-time')
            if time_tag:
                result['published_at'] = time_tag.get('datetime')

        except Exception as e:
            result['error'] = str(e)
            Logger.debug(f"GitHub scrape error: {e}")

        return result

    def scrape_social(self, url: str) -> Dict[str, Any]:
        """Handle social media links - limited scraping possible"""
        result = {
            'url': url,
            'title': None,
            'description': None,
            'content': None,
            'content_type': 'text/html',
            'link_type': 'social',
            'error': None,
            'published_at': None,
            'author_name': None,
        }

        try:
            from bs4 import BeautifulSoup
            session = self._get_session()

            # Social sites often block scraping, but we can try
            resp = session.get(url, timeout=15)

            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, 'html.parser')

                # Get OpenGraph metadata (works for most social sites)
                og_title = soup.find('meta', property='og:title')
                if og_title:
                    result['title'] = og_title.get('content')

                og_desc = soup.find('meta', property='og:description')
                if og_desc:
                    result['description'] = og_desc.get('content')
                    result['content'] = result['description']  # Use description as content

                # Try to get author
                author_meta = soup.find('meta', attrs={'name': 'author'})
                if author_meta:
                    result['author_name'] = author_meta.get('content')

                # Extract username from Twitter/X URLs
                if 'twitter.com' in url or 'x.com' in url:
                    match = re.search(r'(?:twitter|x)\.com/([^/]+)', url)
                    if match:
                        result['author_name'] = match.group(1)

            else:
                result['error'] = f'HTTP {resp.status_code}'

        except Exception as e:
            result['error'] = str(e)
            Logger.debug(f"Social scrape error: {e}")

        return result

    def scrape_article(self, url: str, timeout: int = 30) -> Dict[str, Any]:
        """Scrape generic article/webpage with publication date extraction"""
        from bs4 import BeautifulSoup

        result = {
            'url': url,
            'title': None,
            'description': None,
            'content': None,
            'content_type': None,
            'link_type': 'article',
            'error': None,
            'published_at': None,
            'author_name': None,
        }

        try:
            session = self._get_session()
            resp = session.get(url, timeout=timeout, allow_redirects=True)
            resp.raise_for_status()

            content_type = resp.headers.get('Content-Type', '')
            result['content_type'] = content_type

            if 'text/html' not in content_type.lower():
                result['error'] = f'Not HTML: {content_type}'
                return result

            soup = BeautifulSoup(resp.content, 'html.parser')

            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                result['title'] = title_tag.get_text(strip=True)

            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                result['description'] = meta_desc.get('content', '')

            if not result['description']:
                og_desc = soup.find('meta', attrs={'property': 'og:description'})
                if og_desc:
                    result['description'] = og_desc.get('content', '')

            # Extract publication date - try multiple sources
            date_selectors = [
                ('meta', {'property': 'article:published_time'}),
                ('meta', {'name': 'pubdate'}),
                ('meta', {'name': 'publishdate'}),
                ('meta', {'name': 'date'}),
                ('meta', {'property': 'og:published_time'}),
                ('time', {'itemprop': 'datePublished'}),
                ('time', {'class': re.compile(r'publish|post|article', re.I)}),
            ]

            for tag_name, attrs in date_selectors:
                date_elem = soup.find(tag_name, attrs=attrs)
                if date_elem:
                    date_str = date_elem.get('content') or date_elem.get('datetime') or date_elem.get_text(strip=True)
                    if date_str:
                        # Try to parse the date
                        try:
                            import dateparser
                            parsed_date = dateparser.parse(date_str)
                            if parsed_date:
                                result['published_at'] = parsed_date.isoformat()
                                break
                        except ImportError:
                            result['published_at'] = date_str
                            break
                        except:
                            continue

            # Extract author
            author_selectors = [
                ('meta', {'name': 'author'}),
                ('meta', {'property': 'article:author'}),
                ('a', {'rel': 'author'}),
                ('span', {'class': re.compile(r'author', re.I)}),
            ]

            for tag_name, attrs in author_selectors:
                author_elem = soup.find(tag_name, attrs=attrs)
                if author_elem:
                    author = author_elem.get('content') or author_elem.get_text(strip=True)
                    if author:
                        result['author_name'] = author
                        break

            # Extract main content
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']):
                tag.decompose()

            main_content = None
            for selector in ['article', 'main', '[role="main"]', '.content', '.post', '.entry', '.post-content']:
                main_content = soup.select_one(selector)
                if main_content:
                    break

            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
            else:
                body = soup.find('body')
                text = body.get_text(separator=' ', strip=True) if body else ''

            text = re.sub(r'\s+', ' ', text).strip()
            result['content'] = text[:10000]

            Logger.debug(f"Scraped article: {result['title'][:50] if result['title'] else url}")

        except Exception as e:
            result['error'] = str(e)
            Logger.debug(f"Scrape error for {url}: {e}")

        return result

    def scrape_url(self, url: str, timeout: int = 30) -> Dict[str, Any]:
        """Smart scraper that routes to appropriate handler based on URL"""
        link_type = self._detect_link_type(url)

        Logger.info(f"  [{link_type.upper()}] {url[:60]}...")

        if link_type == 'youtube':
            return self.scrape_youtube(url)
        elif link_type == 'github':
            return self.scrape_github(url)
        elif link_type == 'social':
            return self.scrape_social(url)
        else:
            return self.scrape_article(url, timeout)

    def scrape_and_embed(self, url: str) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        """Scrape URL and generate embedding for the content"""
        content_data = self.scrape_url(url)

        embedding = None
        if content_data.get('content') and not content_data.get('error'):
            # Create text for embedding: title + description + truncated content
            embed_text = ' '.join(filter(None, [
                content_data.get('title', ''),
                content_data.get('description', ''),
                content_data.get('content', '')[:3000]  # First 3k chars for better context
            ]))

            if embed_text.strip():
                try:
                    embedding = self.embedding_gen.generate(embed_text)
                except Exception as e:
                    Logger.debug(f"Embedding error: {e}")

        return content_data, embedding


# =============================================================================
# MAIN SCRAPER
# =============================================================================

class DiscordLinkScraper:
    """Discord link scraper with API and content scraping"""

    def __init__(self, output_dir: Path = None, token: str = None):
        self.output_dir = output_dir or Path('output_data')
        self.db = DiscordLinksDatabase(self.output_dir / 'discord_links.db')

        self.token = token or os.getenv('DISCORD_TOKEN')
        self.api = None

        self.content_scraper = ContentScraper()
        self.embedding_gen = EmbeddingGenerator()

        # Stats
        self.messages_scraped = 0
        self.links_found = 0
        self.links_content_scraped = 0

    def setup_api(self) -> bool:
        """Setup Discord API with token"""
        if not self.token:
            Logger.info("No Discord token provided, attempting to extract from Chrome...")
            self.token = get_discord_token_from_chrome()

        if not self.token:
            Logger.info("Trying interactive browser extraction...")
            self.token = get_discord_token_interactive()

        if not self.token:
            Logger.error("Could not obtain Discord token")
            Logger.info("Please set DISCORD_TOKEN environment variable")
            Logger.info("To get your token:")
            Logger.info("1. Open Discord in browser")
            Logger.info("2. Open Developer Tools (F12)")
            Logger.info("3. Go to Network tab")
            Logger.info("4. Filter by '/api'")
            Logger.info("5. Find 'Authorization' header in any request")
            return False

        self.api = DiscordAPI(self.token)

        # Verify token
        user = self.api.get_user_info()
        if user:
            Logger.success(f"Logged in as: {user.get('username')}#{user.get('discriminator')}")
            return True
        else:
            Logger.error("Token verification failed")
            return False

    def scrape_user_messages(self, guild_id: str, username: str, max_results: int = 500) -> int:
        """Scrape messages from a specific user in a guild"""
        Logger.info(f"Searching for messages from @{username}...")

        # First, find the user's ID
        user = self.api.get_user_by_username(guild_id, username)
        if not user:
            Logger.warning(f"Could not find user: {username}")
            # Try searching with author_id as string directly
            return self._scrape_by_username_search(guild_id, username, max_results)

        user_id = user.get('id')
        user_display = user.get('global_name') or user.get('username')
        Logger.info(f"Found user: {user_display} (ID: {user_id})")

        # Save user to database
        self.db.save_user(
            user_id=user_id,
            username=username,
            display_name=user_display,
            discriminator=user.get('discriminator')
        )

        # Search for messages from this user
        messages_found = 0
        links_found = 0
        offset = 0

        while messages_found < max_results:
            result = self.api.search_messages(
                guild_id=guild_id,
                author_id=user_id,
                has_link=True,  # Only get messages with links
                limit=25,
                offset=offset
            )

            messages = result.get('messages', [])
            total = result.get('total_results', 0)

            if not messages:
                break

            Logger.info(f"  Processing batch of {len(messages)} messages (total: {total})...")

            for msg_group in messages:
                # Discord returns messages in groups (context)
                for msg in msg_group if isinstance(msg_group, list) else [msg_group]:
                    msg_author = msg.get('author', {})

                    # Only process messages from our target user
                    if msg_author.get('id') != user_id:
                        continue

                    msg_data = {
                        'message_id': msg.get('id'),
                        'channel_id': msg.get('channel_id'),
                        'server_id': guild_id,
                        'user_id': user_id,
                        'username': username,
                        'content': msg.get('content', ''),
                        'timestamp': msg.get('timestamp'),
                    }

                    if self.db.save_message(msg_data):
                        messages_found += 1
                        self.messages_scraped += 1

                        # Extract and save links
                        content = msg.get('content', '')
                        urls = self._extract_urls(content)

                        # Also check embeds
                        for embed in msg.get('embeds', []):
                            if embed.get('url'):
                                urls.append(embed['url'])

                        for url in urls:
                            link_data = {
                                'url': url,
                                'message_id': msg.get('id'),
                                'username': username,
                                'server_id': guild_id,
                                'channel_id': msg.get('channel_id'),
                            }
                            if self.db.save_link(link_data):
                                links_found += 1
                                self.links_found += 1

            offset += 25
            time.sleep(random.uniform(0.5, 1.5))  # Rate limit protection

            if offset >= total:
                break

        Logger.success(f"@{username}: {messages_found} messages, {links_found} links")
        return messages_found

    def _scrape_by_username_search(self, guild_id: str, username: str, max_results: int) -> int:
        """Fallback: Search messages containing username mention"""
        Logger.info(f"Trying fallback search for {username}...")

        # This is less reliable but can work
        # We'll scan channels instead
        channels = self.api.get_guild_channels(guild_id)
        text_channels = [c for c in channels if c.get('type') == 0]  # Type 0 = text channel

        messages_found = 0
        links_found = 0

        for channel in text_channels[:10]:  # Limit to 10 channels
            channel_id = channel.get('id')
            channel_name = channel.get('name')

            Logger.info(f"  Scanning #{channel_name}...")
            self.db.save_channel(channel_id, guild_id, channel_name, 'text')

            before = None
            channel_messages = 0

            while channel_messages < 500:  # Limit per channel
                msgs = self.api.get_channel_messages(channel_id, limit=100, before=before)
                if not msgs:
                    break

                for msg in msgs:
                    author = msg.get('author', {})
                    author_name = author.get('username', '').lower()
                    author_global = (author.get('global_name') or '').lower()

                    if username.lower() not in author_name and username.lower() not in author_global:
                        continue

                    msg_data = {
                        'message_id': msg.get('id'),
                        'channel_id': channel_id,
                        'server_id': guild_id,
                        'user_id': author.get('id'),
                        'username': author.get('username', ''),
                        'content': msg.get('content', ''),
                        'timestamp': msg.get('timestamp'),
                    }

                    if self.db.save_message(msg_data):
                        messages_found += 1
                        channel_messages += 1
                        self.messages_scraped += 1  # Fix: increment instance counter

                        urls = self._extract_urls(msg.get('content', ''))
                        for embed in msg.get('embeds', []):
                            if embed.get('url'):
                                urls.append(embed['url'])

                        for url in urls:
                            link_data = {
                                'url': url,
                                'message_id': msg.get('id'),
                                'username': author.get('username', ''),
                                'server_id': guild_id,
                                'channel_id': channel_id,
                            }
                            if self.db.save_link(link_data):
                                links_found += 1
                                self.links_found += 1  # Fix: increment instance counter

                before = msgs[-1]['id']
                time.sleep(0.5)

        Logger.success(f"Fallback search for @{username}: {messages_found} messages, {links_found} links")
        return messages_found

    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = r'https?://[^\s<>"\')\]]+[^\s<>"\')\].,;:!?]'
        urls = re.findall(url_pattern, text)
        # Filter Discord internal links
        return [
            url for url in urls
            if not any(d in url for d in ['discord.com', 'discordapp.com', 'discord.gg'])
        ]

    def scrape_link_content(self, limit: int = 100) -> int:
        """Scrape content from discovered links and store with embeddings"""
        Logger.info(f"Scraping content from links...")

        unscraped = self.db.get_unscraped_links(limit=limit)
        Logger.info(f"Found {len(unscraped)} unscraped links")

        scraped_count = 0
        for link in unscraped:
            url = link['url']
            link_id = link['link_id']

            Logger.info(f"  Scraping: {url[:60]}...")

            try:
                content_data, embedding = self.content_scraper.scrape_and_embed(url)
                self.db.update_link_content(link_id, content_data, embedding)

                if not content_data.get('error'):
                    scraped_count += 1
                    self.links_content_scraped += 1

                time.sleep(random.uniform(0.5, 1.5))  # Be nice to servers

            except Exception as e:
                Logger.debug(f"Error scraping {url}: {e}")
                self.db.update_link_content(link_id, {'error': str(e)})

        Logger.success(f"Scraped content from {scraped_count} links")
        return scraped_count

    def run(self, server_id: str, usernames: List[str], max_messages: int = 500,
            scrape_content: bool = True, max_content_scrape: int = 100) -> Dict[str, Any]:
        """Main entry point for scraping"""
        Logger.info("Discord Link Scraper Starting...")
        Logger.info(f"Server: {server_id}")
        Logger.info(f"Users: {', '.join(usernames)}")

        if not self.setup_api():
            return {'error': 'Could not setup Discord API'}

        # Save server
        guild = self.api.get_guild(server_id)
        server_name = guild.get('name') if guild else None
        self.db.save_server(server_id, server_name)
        Logger.info(f"Server name: {server_name}")

        # Start scrape session
        scrape_id = self.db.start_scrape(server_id, usernames)

        results = {
            'server_id': server_id,
            'server_name': server_name,
            'users': {},
            'total_messages': 0,
            'total_links': 0,
            'links_content_scraped': 0,
        }

        try:
            # Scrape each user
            for username in usernames:
                Logger.info(f"\n{'='*50}")
                Logger.info(f"Scraping user: {username}")
                Logger.info(f"{'='*50}")

                before_msgs = self.messages_scraped
                before_links = self.links_found

                self.scrape_user_messages(server_id, username, max_messages)

                results['users'][username] = {
                    'messages': self.messages_scraped - before_msgs,
                    'links': self.links_found - before_links,
                }

                time.sleep(random.uniform(2, 4))

            results['total_messages'] = self.messages_scraped
            results['total_links'] = self.links_found

            # Scrape content from links
            if scrape_content and self.links_found > 0:
                Logger.info(f"\n{'='*50}")
                Logger.info("Scraping link content...")
                Logger.info(f"{'='*50}")
                self.scrape_link_content(limit=max_content_scrape)
                results['links_content_scraped'] = self.links_content_scraped

            # End scrape session
            self.db.end_scrape(scrape_id, self.messages_scraped, self.links_found, self.links_content_scraped)

            # Print summary
            self._print_summary(results)

            return results

        except Exception as e:
            Logger.error(f"Scraping error: {e}")
            import traceback
            traceback.print_exc()
            self.db.end_scrape(scrape_id, self.messages_scraped, self.links_found, self.links_content_scraped, 'error')
            raise

    def _print_summary(self, results: Dict):
        """Print scraping summary"""
        print("\n" + "="*60)
        print("SCRAPING COMPLETE")
        print("="*60)
        print(f"Server: {results.get('server_name', results['server_id'])}")
        print(f"Total messages: {results['total_messages']}")
        print(f"Total links found: {results['total_links']}")
        print(f"Links content scraped: {results['links_content_scraped']}")
        print("\nBy user:")
        for user, stats in results['users'].items():
            print(f"  @{user}: {stats['messages']} messages, {stats['links']} links")

        stats = self.db.get_stats()
        print(f"\nTop domains:")
        for domain, count in list(stats['top_domains'].items())[:10]:
            print(f"  {domain}: {count}")

    def search_content(self, query: str, limit: int = 10) -> List[Dict]:
        """Semantic search over scraped content"""
        Logger.info(f"Searching for: {query}")

        query_embedding = self.embedding_gen.generate(query)
        results = self.db.similarity_search(query_embedding, limit=limit)

        for i, r in enumerate(results, 1):
            print(f"\n{i}. [{r['similarity']:.2f}] {r['title'] or r['url']}")
            print(f"   Domain: {r['domain']}")
            print(f"   From: @{r['username']}")
            if r['description']:
                print(f"   {r['description'][:100]}...")

        return results

    def export_links(self, output_file: str = None, format: str = 'json'):
        """Export all scraped links"""
        links = self.db.get_all_links()

        if not output_file:
            output_file = self.output_dir / f'discord_links.{format}'

        if format == 'json':
            with open(output_file, 'w') as f:
                json.dump(links, f, indent=2, default=str)
        elif format == 'txt':
            with open(output_file, 'w') as f:
                for link in links:
                    f.write(f"{link['url']}\n")
        elif format == 'csv':
            import csv
            with open(output_file, 'w', newline='') as f:
                if links:
                    writer = csv.DictWriter(f, fieldnames=links[0].keys())
                    writer.writeheader()
                    writer.writerows(links)

        Logger.success(f"Exported {len(links)} links to {output_file}")
        return output_file


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Discord Link Scraper - Extract and scrape links from specific users'
    )
    parser.add_argument(
        '--server', '-s',
        required=True,
        help='Discord server ID (from URL: discord.com/channels/SERVER_ID/...)'
    )
    parser.add_argument(
        '--users', '-u',
        required=True,
        nargs='+',
        help='Usernames to scrape (space-separated)'
    )
    parser.add_argument(
        '--token', '-t',
        help='Discord user token (or set DISCORD_TOKEN env var)'
    )
    parser.add_argument(
        '--max-messages', '-m',
        type=int,
        default=500,
        help='Max messages per user (default: 500)'
    )
    parser.add_argument(
        '--scrape-content',
        action='store_true',
        default=True,
        help='Scrape content from discovered links (default: True)'
    )
    parser.add_argument(
        '--no-scrape-content',
        action='store_true',
        help='Skip scraping link content'
    )
    parser.add_argument(
        '--max-content', '-c',
        type=int,
        default=100,
        help='Max links to scrape content from (default: 100)'
    )
    parser.add_argument(
        '--export', '-e',
        choices=['json', 'txt', 'csv'],
        help='Export links to file after scraping'
    )
    parser.add_argument(
        '--search', '-q',
        help='Search scraped content (run after scraping)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='output_data',
        help='Output directory (default: output_data)'
    )

    args = parser.parse_args()

    scraper = DiscordLinkScraper(
        output_dir=Path(args.output_dir),
        token=args.token
    )

    # Search mode
    if args.search:
        scraper.setup_api()
        scraper.search_content(args.search)
        return

    # Scraping mode
    scrape_content = not args.no_scrape_content

    results = scraper.run(
        server_id=args.server,
        usernames=args.users,
        max_messages=args.max_messages,
        scrape_content=scrape_content,
        max_content_scrape=args.max_content
    )

    if args.export and results:
        scraper.export_links(format=args.export)


if __name__ == '__main__':
    main()
