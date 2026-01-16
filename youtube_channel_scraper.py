# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests>=2.31.0",
#     "python-dotenv>=0.19.0",
#     "feedparser>=6.0.0",
# ]
# ///
#
# YOUTUBE CONTENT CHUNKING
# ========================
# This module also provides chunking functions for citation quote extraction.
# See: chunk_youtube_description(), fetch_transcript_if_needed(), get_youtube_quote_with_timestamp()
#
# Architecture: RSS-first discovery with TranscriptAPI.com enrichment
# Based on Pinecone best practices for YouTube semantic search (~75 word chunks)
#
"""
YouTube Channel RSS Scraper
===========================

Monitors YouTube channels via RSS feeds for new AI-related videos.
No API key required - uses free RSS feeds.

Usage:
    uv run python youtube_channel_scraper.py              # Scrape all channels
    uv run python youtube_channel_scraper.py --add-channel "UCxxxxxx" "Channel Name"
    uv run python youtube_channel_scraper.py --list       # List monitored channels
    uv run python youtube_channel_scraper.py --recent 24  # Videos from last 24 hours

RSS Feed Format:
    https://www.youtube.com/feeds/videos.xml?channel_id=CHANNEL_ID
"""

import os
import sys
import sqlite3
import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs

import feedparser
import requests
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path("output_data")
DB_PATH = OUTPUT_DIR / "ai_news.db"

# Seed YouTube channels for AI news monitoring
# Format: (channel_id, channel_name, category)
# All channel IDs verified via RSS feed URLs (youtube.com/feeds/videos.xml?channel_id=...)
SEED_CHANNELS = [
    # Official AI Company Channels - VERIFIED
    ("UCXZCJLdBC09xxGZ6gcdrc6A", "OpenAI", "official"),  # @OpenAI - 1.9M+ subs
    ("UCP7jMXSY2xbc3KCAE0MHQ-A", "Google DeepMind", "official"),  # @GoogleDeepMind
    ("UCrDwWp7EBBv4NwvScIpBDOA", "Anthropic", "official"),  # @anthropic-ai - 352K+ subs

    # Top-Tier AI Research & Education Channels
    ("UCbfYPyITQ-7l4upoX8nvctg", "Two Minute Papers", "research"),  # Péter Sólyom - Weekly AI papers
    ("UCZHmQk67mSJgfCCTn7xBfew", "Yannic Kilcher", "research"),  # ML/AI research papers & NNs explained
    ("UCYO_jab_esuFRV4b17AJtAw", "3Blue1Brown", "education"),  # Grant Sanderson - Math/AI visualization - 7.9M subs
    ("UCSHZKyawb77ixDdsGog4iWA", "Lex Fridman", "interviews"),  # Long-form AI researcher interviews - 3.6M+ subs
    ("UCX7Y2qWriXpqocG97SFW2OQ", "Jeremy Howard", "education"),  # @howardjeremyp - Practical deep learning
    ("UCNJ1Ymd5yFuUPtn21xtRbbw", "AI Explained", "education"),  # Accessible AI concepts & news

    # Tech & Innovation Coverage
    ("UCddiUEpeqJcYeBxX1IVBKvQ", "The Verge", "news"),  # @theverge - Tech news & reviews
    ("UC5WjFrtBdufl6CZojX3D8dQ", "Tesla", "innovation"),  # @Tesla - AI/robotics development
    ("UCsBjURrPoezykLs9EqgamOA", "Fireship", "education"),  # @Fireship - Concise coding & tech tutorials
    ("UChpleBmo18P08aKCIgti38g", "Matt Wolfe", "news"),  # @mreflow - AI tools & news

    # AI/ML Community & Education Platforms
    ("UCHlNU7kIZhRgSbhHvFoy72w", "Hugging Face", "community"),  # AI models, datasets, community
    ("UCcIXc5mJsHVYTZR1maL5l9w", "DeepLearning.AI", "education"),  # @DeepLearningAI - Andrew Ng's platform
    ("UCBa5G_ESCn8Yd4vw5U-gIcg", "Stanford Online", "education"),  # University AI/ML course lectures
]

# Keywords to identify AI-relevant videos
AI_KEYWORDS = [
    'ai', 'artificial intelligence', 'machine learning', 'ml', 'gpt', 'chatgpt',
    'llm', 'large language model', 'claude', 'gemini', 'openai', 'anthropic',
    'deepmind', 'neural', 'deep learning', 'transformer', 'diffusion',
    'stable diffusion', 'midjourney', 'dall-e', 'sora', 'copilot',
    'agent', 'agi', 'generative', 'gen ai', 'foundation model',
    'disco', 'gentabs',  # Google's new product
]


# =============================================================================
# DATABASE
# =============================================================================

@dataclass
class YouTubeVideo:
    video_id: str
    channel_id: str
    channel_name: str
    title: str
    description: str
    url: str
    published_at: str
    thumbnail_url: str
    is_ai_relevant: bool = False
    ai_relevance_score: float = 0.0
    scraped_at: str = ""


class YouTubeDatabase:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        """Create YouTube-specific tables"""
        cursor = self.conn.cursor()

        # YouTube channels table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS youtube_channels (
                channel_id TEXT PRIMARY KEY,
                channel_name TEXT,
                channel_url TEXT,
                category TEXT,
                subscriber_count INTEGER,
                video_count INTEGER DEFAULT 0,
                ai_relevant_count INTEGER DEFAULT 0,
                last_scraped TEXT,
                first_scraped TEXT DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        """)

        # YouTube videos table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS youtube_videos (
                video_id TEXT PRIMARY KEY,
                channel_id TEXT,
                channel_name TEXT,
                title TEXT,
                description TEXT,
                url TEXT,
                published_at TEXT,
                thumbnail_url TEXT,
                duration_seconds INTEGER,
                view_count INTEGER,
                like_count INTEGER,
                comment_count INTEGER,
                is_ai_relevant BOOLEAN DEFAULT FALSE,
                ai_relevance_score REAL DEFAULT 0.0,
                scraped_at TEXT DEFAULT CURRENT_TIMESTAMP,
                content_scraped BOOLEAN DEFAULT FALSE,
                transcript TEXT,
                FOREIGN KEY (channel_id) REFERENCES youtube_channels(channel_id)
            )
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_yt_videos_channel ON youtube_videos(channel_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_yt_videos_published ON youtube_videos(published_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_yt_videos_ai_relevant ON youtube_videos(is_ai_relevant)")

        self.conn.commit()

    def add_channel(self, channel_id: str, name: str, category: str = "news"):
        """Add a channel to monitor"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO youtube_channels (channel_id, channel_name, channel_url, category)
            VALUES (?, ?, ?, ?)
        """, (channel_id, name, f"https://www.youtube.com/channel/{channel_id}", category))
        self.conn.commit()
        print(f"[+] Added channel: {name} ({channel_id})")

    def get_channels(self, active_only: bool = True) -> List[Dict]:
        """Get all monitored channels"""
        cursor = self.conn.cursor()
        query = "SELECT * FROM youtube_channels"
        if active_only:
            query += " WHERE is_active = TRUE"
        cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]

    def save_video(self, video: YouTubeVideo) -> bool:
        """Save a video, returns True if new"""
        cursor = self.conn.cursor()

        # Check if exists
        cursor.execute("SELECT video_id FROM youtube_videos WHERE video_id = ?", (video.video_id,))
        exists = cursor.fetchone() is not None

        if not exists:
            cursor.execute("""
                INSERT INTO youtube_videos (
                    video_id, channel_id, channel_name, title, description,
                    url, published_at, thumbnail_url, is_ai_relevant,
                    ai_relevance_score, scraped_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                video.video_id, video.channel_id, video.channel_name,
                video.title, video.description, video.url, video.published_at,
                video.thumbnail_url, video.is_ai_relevant, video.ai_relevance_score,
                video.scraped_at or datetime.now().isoformat()
            ))
            self.conn.commit()
            return True
        return False

    def update_channel_stats(self, channel_id: str):
        """Update channel video counts"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE youtube_channels SET
                video_count = (SELECT COUNT(*) FROM youtube_videos WHERE channel_id = ?),
                ai_relevant_count = (SELECT COUNT(*) FROM youtube_videos WHERE channel_id = ? AND is_ai_relevant = TRUE),
                last_scraped = ?
            WHERE channel_id = ?
        """, (channel_id, channel_id, datetime.now().isoformat(), channel_id))
        self.conn.commit()

    def get_recent_videos(self, hours: int = 24, ai_only: bool = False) -> List[Dict]:
        """Get videos from the last N hours"""
        cursor = self.conn.cursor()
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        query = """
            SELECT * FROM youtube_videos
            WHERE scraped_at > ?
        """
        if ai_only:
            query += " AND is_ai_relevant = TRUE"
        query += " ORDER BY published_at DESC"

        cursor.execute(query, (cutoff,))
        return [dict(row) for row in cursor.fetchall()]

    def close(self):
        self.conn.close()


# =============================================================================
# RSS SCRAPER
# =============================================================================

class YouTubeRSSScraper:
    """Scrape YouTube channels via RSS feeds (no API key needed)"""

    RSS_URL_TEMPLATE = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

    def __init__(self, db: YouTubeDatabase):
        self.db = db

    def _calculate_ai_relevance(self, title: str, description: str) -> tuple[bool, float]:
        """Calculate AI relevance score based on keywords"""
        text = f"{title} {description}".lower()

        matches = 0
        for keyword in AI_KEYWORDS:
            if keyword.lower() in text:
                matches += 1

        # Score is percentage of keywords found (capped at 1.0)
        score = min(matches / 5, 1.0)  # 5+ keywords = 100%
        is_relevant = score >= 0.2 or matches >= 1

        return is_relevant, score

    def _parse_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        parsed = urlparse(url)
        if 'youtube.com' in parsed.netloc:
            query = parse_qs(parsed.query)
            return query.get('v', [None])[0]
        elif 'youtu.be' in parsed.netloc:
            return parsed.path.lstrip('/')
        return None

    def scrape_channel(self, channel_id: str, channel_name: str = "") -> List[YouTubeVideo]:
        """Scrape videos from a channel's RSS feed"""
        url = self.RSS_URL_TEMPLATE.format(channel_id=channel_id)

        try:
            feed = feedparser.parse(url)

            if feed.bozo and not feed.entries:
                print(f"[!] Failed to parse feed for {channel_name or channel_id}")
                return []

            videos = []
            for entry in feed.entries:
                video_id = entry.get('yt_videoid', self._parse_video_id(entry.link))
                if not video_id:
                    continue

                title = entry.get('title', '')
                description = entry.get('summary', '')

                is_relevant, score = self._calculate_ai_relevance(title, description)

                # Get thumbnail
                thumbnail = ""
                if 'media_thumbnail' in entry:
                    thumbnail = entry.media_thumbnail[0].get('url', '')

                video = YouTubeVideo(
                    video_id=video_id,
                    channel_id=channel_id,
                    channel_name=channel_name or feed.feed.get('title', ''),
                    title=title,
                    description=description[:1000] if description else "",
                    url=entry.link,
                    published_at=entry.get('published', ''),
                    thumbnail_url=thumbnail,
                    is_ai_relevant=is_relevant,
                    ai_relevance_score=score,
                    scraped_at=datetime.now().isoformat()
                )
                videos.append(video)

            return videos

        except Exception as e:
            print(f"[!] Error scraping {channel_name or channel_id}: {e}")
            return []

    def scrape_all_channels(self) -> Dict[str, int]:
        """Scrape all monitored channels"""
        channels = self.db.get_channels()

        if not channels:
            print("[!] No channels to scrape. Adding seed channels...")
            for channel_id, name, category in SEED_CHANNELS:
                self.db.add_channel(channel_id, name, category)
            channels = self.db.get_channels()

        results = {"total": 0, "new": 0, "ai_relevant": 0}

        for channel in channels:
            channel_id = channel['channel_id']
            channel_name = channel['channel_name']

            print(f"[*] Scraping {channel_name}...")
            videos = self.scrape_channel(channel_id, channel_name)

            new_count = 0
            ai_count = 0
            for video in videos:
                if self.db.save_video(video):
                    new_count += 1
                    if video.is_ai_relevant:
                        ai_count += 1
                        print(f"    [AI] {video.title[:60]}...")

            self.db.update_channel_stats(channel_id)

            results["total"] += len(videos)
            results["new"] += new_count
            results["ai_relevant"] += ai_count

            if new_count > 0:
                print(f"    Found {new_count} new videos ({ai_count} AI-relevant)")

        return results


# =============================================================================
# CHANNEL DISCOVERY
# =============================================================================

def extract_channel_id_from_url(url: str) -> Optional[str]:
    """Extract channel ID from various YouTube URL formats"""
    # Direct channel ID URL
    match = re.search(r'youtube\.com/channel/([a-zA-Z0-9_-]{24})', url)
    if match:
        return match.group(1)

    # Handle URL - need to fetch page to get channel ID
    if '/c/' in url or '/@' in url or '/user/' in url:
        try:
            resp = requests.get(url, timeout=10)
            match = re.search(r'"channelId":"([a-zA-Z0-9_-]{24})"', resp.text)
            if match:
                return match.group(1)
        except:
            pass

    return None


# =============================================================================
# YOUTUBE CONTENT CHUNKING FOR CITATION QUOTES
# =============================================================================
# Based on Pinecone best practices: https://www.pinecone.io/learn/youtube-search/
# ~75 word chunks with timestamps for deep-linking

# TranscriptAPI.com configuration
TRANSCRIPT_API_KEY = os.getenv('TRANSCRIPT_API_KEY', 'sk_HX7Vjy6TH8kR5fe60f9SAhcCWbK9kZgvQqthmTBG2iE')
TRANSCRIPT_API_URL = "https://api.transcriptapi.com/v1/transcript"


def chunk_youtube_description(description: str, min_chunk: int = 30) -> List[Dict[str, Any]]:
    """
    Chunk YouTube description into semantic segments.

    Descriptions typically contain:
    - Summary paragraph(s)
    - Timestamps/chapters
    - Links and credits

    We chunk on double newlines (natural paragraph breaks) first,
    then fall back to sentence splitting.

    Args:
        description: YouTube video description text
        min_chunk: Minimum characters per chunk

    Returns:
        List of chunk dicts with 'text', 'index', 'source' keys
    """
    if not description or len(description) < min_chunk:
        return []

    # First try: Split on double newlines (paragraph breaks)
    segments = [s.strip() for s in description.split('\n\n') if len(s.strip()) >= min_chunk]

    # If no good paragraphs, try sentence splitting
    if not segments:
        sentences = re.split(r'(?<=[.!?])\s+', description)
        segments = [s.strip() for s in sentences if len(s.strip()) >= min_chunk]

    return [{'text': s, 'index': i, 'source': 'description'} for i, s in enumerate(segments)]


def fetch_transcript_if_needed(video_id: str, db_path: Path = DB_PATH) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch transcript with timestamps only if we don't already have it.
    Fails gracefully when API quota exceeded.

    Pipeline: RSS Feed (discovery) → TranscriptAPI (enrichment)

    Args:
        video_id: YouTube video ID
        db_path: Path to ai_news.db

    Returns:
        List of transcript segments with timestamps, or None
    """
    import json

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Check if we already have transcript
    cursor.execute(
        "SELECT transcript FROM youtube_videos WHERE video_id = ? AND transcript IS NOT NULL AND transcript != ''",
        (video_id,)
    )
    existing = cursor.fetchone()
    if existing and existing[0]:
        conn.close()
        # Parse stored JSON transcript
        try:
            return json.loads(existing[0])
        except:
            return [{'text': existing[0], 'start': 0}]

    # Fetch from API
    try:
        response = requests.get(
            TRANSCRIPT_API_URL,
            params={"video_id": video_id},
            headers={"Authorization": f"Bearer {TRANSCRIPT_API_KEY}"},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            # TranscriptAPI returns segments with timestamps
            segments = data.get("segments", [])

            # Save to database as JSON
            if segments:
                cursor.execute(
                    "UPDATE youtube_videos SET transcript = ? WHERE video_id = ?",
                    (json.dumps(segments), video_id)
                )
                conn.commit()
                print(f"[TranscriptAPI] Fetched {len(segments)} segments for {video_id}")

            conn.close()
            return segments if segments else None

        elif response.status_code == 429:
            print("[TranscriptAPI] Quota exceeded, falling back to description")
            conn.close()
            return None

        elif response.status_code == 404:
            print(f"[TranscriptAPI] No transcript available for {video_id}")
            conn.close()
            return None

        else:
            print(f"[TranscriptAPI] Error {response.status_code}")
            conn.close()
            return None

    except requests.RequestException as e:
        print(f"[TranscriptAPI] Request failed: {e}")
        conn.close()
        return None


def chunk_transcript_with_timestamps(
    segments: List[Dict[str, Any]],
    target_words: int = 75
) -> List[Dict[str, Any]]:
    """
    Chunk transcript segments into ~100 token chunks with timestamps.

    TranscriptAPI returns segments like:
    [{"text": "Hello world", "start": 0.0, "duration": 2.5}, ...]

    We merge these into larger semantic chunks while tracking start times.

    Args:
        segments: Raw transcript segments from API
        target_words: Target words per chunk (~75 = ~100 tokens)

    Returns:
        List of chunks with 'text', 'start_time', 'index', 'source' keys
    """
    if not segments:
        return []

    chunks = []
    current_chunk = {'text': '', 'start_time': 0.0, 'word_count': 0}

    for seg in segments:
        text = seg.get('text', '')
        start = seg.get('start', 0.0)

        # Start new chunk if empty
        if not current_chunk['text']:
            current_chunk['start_time'] = start

        current_chunk['text'] += ' ' + text
        current_chunk['word_count'] += len(text.split())

        # Chunk boundary at target word count
        if current_chunk['word_count'] >= target_words:
            chunks.append({
                'text': current_chunk['text'].strip(),
                'start_time': current_chunk['start_time'],
                'index': len(chunks),
                'source': 'transcript'
            })
            current_chunk = {'text': '', 'start_time': 0.0, 'word_count': 0}

    # Don't forget the last chunk
    if current_chunk['text'].strip():
        chunks.append({
            'text': current_chunk['text'].strip(),
            'start_time': current_chunk['start_time'],
            'index': len(chunks),
            'source': 'transcript'
        })

    return chunks


def get_youtube_quote_with_timestamp(
    video: Dict[str, Any],
    sentence: str,
    db_path: Path = DB_PATH
) -> Tuple[Optional[str], Optional[str]]:
    """
    Get best matching quote from YouTube video.

    Returns both the quote AND the timestamped URL for deep-linking.

    Priority:
    1. Transcript segment → URL with &t=XXs
    2. Description chunk → URL without timestamp
    3. Title → URL without timestamp

    Args:
        video: Video dict with 'video_id', 'description', 'title', 'url' keys
        sentence: The sentence from generated post to match against
        db_path: Path to ai_news.db

    Returns:
        Tuple of (quote_text, url_with_timestamp)
    """
    # Import here to avoid circular dependency
    from agents.hybrid_retriever import find_best_paragraph_match

    video_id = video.get('video_id', video.get('id', ''))
    base_url = video.get('url', f'https://youtube.com/watch?v={video_id}')

    # Try transcript first (enriched data)
    transcript_segments = fetch_transcript_if_needed(video_id, db_path)
    if transcript_segments:
        chunks = chunk_transcript_with_timestamps(transcript_segments)
        if chunks:
            match = find_best_paragraph_match(sentence, chunks)
            if match and match.get('start_time') is not None:
                # Build timestamped URL
                start_seconds = int(match['start_time'])
                # Handle URLs that already have query params
                if '?' in base_url:
                    timestamped_url = f"{base_url}&t={start_seconds}s"
                else:
                    timestamped_url = f"{base_url}?t={start_seconds}s"
                return (match['text'][:200], timestamped_url)

    # Fallback to description (always available from RSS)
    description = video.get('description', '')
    desc_chunks = chunk_youtube_description(description)
    if desc_chunks:
        match = find_best_paragraph_match(sentence, desc_chunks)
        if match:
            return (match['text'][:200], base_url)  # No timestamp

    # Final fallback to title
    title = video.get('title', '')
    return (title[:200] if title else None, base_url)


def get_all_youtube_chunks(video: Dict[str, Any], db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    """
    Get all available chunks for a YouTube video (transcript + description).

    Useful for pre-computing all searchable content.

    Args:
        video: Video dict with 'video_id', 'description' keys
        db_path: Path to database

    Returns:
        Combined list of all chunks, transcript chunks first (if available)
    """
    video_id = video.get('video_id', video.get('id', ''))
    all_chunks = []

    # Try transcript first
    transcript_segments = fetch_transcript_if_needed(video_id, db_path)
    if transcript_segments:
        transcript_chunks = chunk_transcript_with_timestamps(transcript_segments)
        all_chunks.extend(transcript_chunks)

    # Always add description chunks as fallback/supplement
    description = video.get('description', '')
    desc_chunks = chunk_youtube_description(description)
    all_chunks.extend(desc_chunks)

    return all_chunks


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='YouTube Channel RSS Scraper')
    parser.add_argument('--add-channel', nargs=2, metavar=('ID', 'NAME'),
                       help='Add a channel to monitor')
    parser.add_argument('--add-url', type=str,
                       help='Add channel by URL (auto-extracts ID)')
    parser.add_argument('--list', action='store_true',
                       help='List monitored channels')
    parser.add_argument('--recent', type=int, metavar='HOURS',
                       help='Show videos from last N hours')
    parser.add_argument('--ai-only', action='store_true',
                       help='Only show AI-relevant videos')
    parser.add_argument('--seed', action='store_true',
                       help='Add seed channels')

    args = parser.parse_args()

    db = YouTubeDatabase()

    try:
        if args.add_channel:
            channel_id, name = args.add_channel
            db.add_channel(channel_id, name)
            return

        if args.add_url:
            channel_id = extract_channel_id_from_url(args.add_url)
            if channel_id:
                name = input("Channel name: ").strip() or "Unknown"
                db.add_channel(channel_id, name)
            else:
                print("[!] Could not extract channel ID from URL")
            return

        if args.list:
            channels = db.get_channels(active_only=False)
            print(f"\nMonitored Channels ({len(channels)}):")
            print("-" * 60)
            for ch in channels:
                status = "active" if ch['is_active'] else "inactive"
                print(f"  {ch['channel_name']:<30} {ch['category']:<10} [{status}]")
                print(f"    ID: {ch['channel_id']}")
                print(f"    Videos: {ch['video_count']} total, {ch['ai_relevant_count']} AI-relevant")
            return

        if args.recent:
            videos = db.get_recent_videos(hours=args.recent, ai_only=args.ai_only)
            print(f"\nVideos from last {args.recent} hours ({len(videos)} found):")
            print("-" * 60)
            for v in videos:
                ai_tag = "[AI] " if v['is_ai_relevant'] else ""
                print(f"  {ai_tag}{v['title'][:50]}...")
                print(f"    Channel: {v['channel_name']}")
                print(f"    URL: {v['url']}")
                print()
            return

        if args.seed:
            print("Adding seed channels...")
            for channel_id, name, category in SEED_CHANNELS:
                db.add_channel(channel_id, name, category)
            return

        # Default: scrape all channels
        print("=" * 60)
        print(f"YouTube RSS Scraper - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        scraper = YouTubeRSSScraper(db)
        results = scraper.scrape_all_channels()

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Total videos checked: {results['total']}")
        print(f"  New videos found: {results['new']}")
        print(f"  AI-relevant: {results['ai_relevant']}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
