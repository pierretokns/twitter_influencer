# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests>=2.31.0",
#     "python-dotenv>=0.19.0",
#     "trafilatura>=1.6.0",
#     "FlagEmbedding>=1.2.0",
#     "scikit-learn>=1.3.0",
#     "numpy>=1.24.0",
# ]
# ///
"""
Backfill Content Chunks for Citation Quote Extraction
======================================================

Re-ingests historical web articles and YouTube videos with proper semantic chunking.
Populates article_paragraphs and youtube_segments tables for granular citation quotes.

Usage:
    uv run python backfill_content_chunks.py              # Process all
    uv run python backfill_content_chunks.py --web-only   # Only web articles
    uv run python backfill_content_chunks.py --youtube-only  # Only YouTube
    uv run python backfill_content_chunks.py --limit 50   # Limit items processed
    uv run python backfill_content_chunks.py --dry-run    # Preview without saving

Based on:
- Meta-Chunking (arXiv 2410.12788): Semantic boundary detection
- Pinecone best practices: ~75 word chunks for YouTube
- TranscriptAPI.com for YouTube transcript enrichment
"""

import os
import re
import json
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import requests
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# Configuration
OUTPUT_DIR = Path("output_data")
DB_PATH = OUTPUT_DIR / "ai_news.db"

# TranscriptAPI.com configuration
TRANSCRIPT_API_KEY = os.getenv('TRANSCRIPT_API_KEY', 'sk_HX7Vjy6TH8kR5fe60f9SAhcCWbK9kZgvQqthmTBG2iE')
TRANSCRIPT_API_URL = "https://api.transcriptapi.com/v1/transcript"

# Chunking parameters
MIN_CHUNK_CHARS = 50
MAX_CHUNK_CHARS = 500
TARGET_WORDS_YOUTUBE = 75  # ~100 tokens per Pinecone best practices


@dataclass
class ChunkStats:
    """Track chunking statistics"""
    articles_processed: int = 0
    articles_skipped: int = 0
    article_chunks_created: int = 0
    videos_processed: int = 0
    videos_skipped: int = 0
    video_chunks_created: int = 0
    transcripts_fetched: int = 0
    transcript_errors: int = 0


# =============================================================================
# WEB ARTICLE CHUNKING
# =============================================================================

def chunk_article_content(content: str, use_semantic: bool = True) -> List[Dict[str, Any]]:
    """
    Chunk article content into semantic paragraphs.

    Uses semantic chunking with BGE-M3 if available, falls back to
    paragraph-based chunking.

    Args:
        content: Full article text
        use_semantic: Whether to use BGE-M3 semantic chunking

    Returns:
        List of chunk dicts with 'text', 'index', 'char_start' keys
    """
    if not content or len(content) < MIN_CHUNK_CHARS:
        return []

    # First try paragraph-based chunking (natural boundaries)
    paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) >= MIN_CHUNK_CHARS]

    # If no good paragraphs, try sentence splitting
    if not paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', content)

        # Merge short sentences into chunks
        chunks = []
        current = {'text': '', 'char_start': 0}
        char_pos = 0

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 20:
                continue

            if not current['text']:
                current['char_start'] = char_pos

            current['text'] += ' ' + sent
            char_pos += len(sent) + 1

            if len(current['text']) >= MIN_CHUNK_CHARS:
                chunks.append({
                    'text': current['text'].strip()[:MAX_CHUNK_CHARS],
                    'index': len(chunks),
                    'char_start': current['char_start']
                })
                current = {'text': '', 'char_start': 0}

        # Don't forget the last chunk
        if current['text'].strip() and len(current['text'].strip()) >= MIN_CHUNK_CHARS // 2:
            chunks.append({
                'text': current['text'].strip()[:MAX_CHUNK_CHARS],
                'index': len(chunks),
                'char_start': current['char_start']
            })

        return chunks

    # Use paragraph-based chunks
    chunks = []
    char_pos = 0

    for para in paragraphs:
        # Truncate long paragraphs
        text = para[:MAX_CHUNK_CHARS]
        chunks.append({
            'text': text,
            'index': len(chunks),
            'char_start': char_pos
        })
        char_pos += len(para) + 2  # +2 for \n\n

    return chunks


def fetch_article_content(url: str) -> Optional[str]:
    """
    Fetch and extract article content using trafilatura.

    Args:
        url: Article URL

    Returns:
        Extracted article text or None
    """
    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None

        content = trafilatura.extract(downloaded, include_comments=False)
        return content

    except ImportError:
        print("[WARN] trafilatura not available, using stored content only")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None


def process_web_articles(
    conn: sqlite3.Connection,
    limit: Optional[int] = None,
    dry_run: bool = False,
    refetch: bool = False
) -> ChunkStats:
    """
    Process all web articles and create chunks.

    Args:
        conn: Database connection
        limit: Max articles to process
        dry_run: If True, don't save to database
        refetch: If True, re-fetch content from URLs

    Returns:
        ChunkStats with processing results
    """
    stats = ChunkStats()
    cursor = conn.cursor()

    # Get articles that don't have chunks yet
    query = """
        SELECT a.article_id, a.url, a.title, a.content, a.description
        FROM web_articles a
        LEFT JOIN article_paragraphs p ON a.article_id = p.article_id
        WHERE p.id IS NULL
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    articles = cursor.fetchall()

    print(f"\n[WEB] Processing {len(articles)} articles...")

    for row in articles:
        article_id = row['article_id']
        url = row['url']
        title = row['title'] or ''
        content = row['content'] or ''
        description = row['description'] or ''

        # Use best available content
        text_to_chunk = content

        # If no content stored, try fetching
        if (not text_to_chunk or len(text_to_chunk) < 100) and refetch and url:
            print(f"  [FETCH] {url[:60]}...")
            fetched = fetch_article_content(url)
            if fetched:
                text_to_chunk = fetched
                # Update stored content
                if not dry_run:
                    cursor.execute(
                        "UPDATE web_articles SET content = ? WHERE article_id = ?",
                        (fetched[:10000], article_id)
                    )

        # Fall back to description if no content
        if not text_to_chunk or len(text_to_chunk) < 50:
            text_to_chunk = description

        if not text_to_chunk or len(text_to_chunk) < MIN_CHUNK_CHARS:
            print(f"  [SKIP] {title[:40]}... (no content)")
            stats.articles_skipped += 1
            continue

        # Create chunks
        chunks = chunk_article_content(text_to_chunk)

        if not chunks:
            stats.articles_skipped += 1
            continue

        # Save chunks
        if not dry_run:
            for chunk in chunks:
                cursor.execute("""
                    INSERT INTO article_paragraphs (article_id, paragraph_index, text, char_start)
                    VALUES (?, ?, ?, ?)
                """, (article_id, chunk['index'], chunk['text'], chunk.get('char_start', 0)))

        stats.articles_processed += 1
        stats.article_chunks_created += len(chunks)

        print(f"  [OK] {title[:40]}... ({len(chunks)} chunks)")

    if not dry_run:
        conn.commit()

    return stats


# =============================================================================
# YOUTUBE CHUNKING
# =============================================================================

def chunk_youtube_description(description: str) -> List[Dict[str, Any]]:
    """
    Chunk YouTube description into segments.

    Args:
        description: YouTube video description

    Returns:
        List of chunk dicts with 'text', 'index', 'source' keys
    """
    if not description or len(description) < MIN_CHUNK_CHARS:
        return []

    # Split on double newlines (natural paragraph breaks)
    segments = [s.strip() for s in description.split('\n\n') if len(s.strip()) >= 30]

    # If no good paragraphs, try sentence splitting
    if not segments:
        sentences = re.split(r'(?<=[.!?])\s+', description)
        segments = [s.strip() for s in sentences if len(s.strip()) >= 30]

    return [
        {'text': s[:MAX_CHUNK_CHARS], 'index': i, 'source': 'description', 'start_time': None}
        for i, s in enumerate(segments)
    ]


def fetch_transcript(video_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch transcript from TranscriptAPI.com.

    Args:
        video_id: YouTube video ID

    Returns:
        List of transcript segments with timestamps, or None
    """
    try:
        response = requests.get(
            TRANSCRIPT_API_URL,
            params={"video_id": video_id},
            headers={"Authorization": f"Bearer {TRANSCRIPT_API_KEY}"},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("segments", [])
        elif response.status_code == 429:
            print("    [QUOTA] TranscriptAPI quota exceeded")
            return None
        elif response.status_code == 404:
            return None  # No transcript available
        else:
            return None

    except requests.RequestException as e:
        print(f"    [ERROR] Transcript fetch failed: {e}")
        return None


def chunk_transcript_segments(
    segments: List[Dict[str, Any]],
    target_words: int = TARGET_WORDS_YOUTUBE
) -> List[Dict[str, Any]]:
    """
    Chunk transcript segments into ~100 token chunks with timestamps.

    Args:
        segments: Raw transcript segments from API
        target_words: Target words per chunk

    Returns:
        List of chunks with 'text', 'start_time', 'end_time', 'index', 'source'
    """
    if not segments:
        return []

    chunks = []
    current = {'text': '', 'start_time': 0.0, 'end_time': 0.0, 'word_count': 0}

    for seg in segments:
        text = seg.get('text', '')
        start = seg.get('start', 0.0)
        duration = seg.get('duration', 0.0)

        if not current['text']:
            current['start_time'] = start

        current['text'] += ' ' + text
        current['end_time'] = start + duration
        current['word_count'] += len(text.split())

        if current['word_count'] >= target_words:
            chunks.append({
                'text': current['text'].strip()[:MAX_CHUNK_CHARS],
                'start_time': current['start_time'],
                'end_time': current['end_time'],
                'index': len(chunks),
                'source': 'transcript'
            })
            current = {'text': '', 'start_time': 0.0, 'end_time': 0.0, 'word_count': 0}

    # Don't forget the last chunk
    if current['text'].strip():
        chunks.append({
            'text': current['text'].strip()[:MAX_CHUNK_CHARS],
            'start_time': current['start_time'],
            'end_time': current['end_time'],
            'index': len(chunks),
            'source': 'transcript'
        })

    return chunks


def process_youtube_videos(
    conn: sqlite3.Connection,
    limit: Optional[int] = None,
    dry_run: bool = False,
    fetch_transcripts: bool = True
) -> ChunkStats:
    """
    Process all YouTube videos and create chunks.

    Args:
        conn: Database connection
        limit: Max videos to process
        dry_run: If True, don't save to database
        fetch_transcripts: If True, fetch transcripts from API

    Returns:
        ChunkStats with processing results
    """
    stats = ChunkStats()
    cursor = conn.cursor()

    # Get videos that don't have chunks yet
    query = """
        SELECT v.video_id, v.title, v.description, v.transcript
        FROM youtube_videos v
        LEFT JOIN youtube_segments s ON v.video_id = s.video_id
        WHERE s.id IS NULL
    """
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    videos = cursor.fetchall()

    print(f"\n[YOUTUBE] Processing {len(videos)} videos...")

    transcript_quota_exceeded = False

    for row in videos:
        video_id = row['video_id']
        title = row['title'] or ''
        description = row['description'] or ''
        stored_transcript = row['transcript']

        chunks = []

        # Try to use stored transcript first
        if stored_transcript:
            try:
                segments = json.loads(stored_transcript)
                if segments:
                    chunks = chunk_transcript_segments(segments)
                    print(f"  [CACHED] {title[:40]}... ({len(chunks)} transcript chunks)")
            except (json.JSONDecodeError, TypeError):
                pass

        # Fetch transcript if not cached and quota not exceeded
        if not chunks and fetch_transcripts and not transcript_quota_exceeded:
            print(f"  [FETCH] {title[:40]}...")
            segments = fetch_transcript(video_id)

            if segments:
                # Save transcript to database
                if not dry_run:
                    cursor.execute(
                        "UPDATE youtube_videos SET transcript = ? WHERE video_id = ?",
                        (json.dumps(segments), video_id)
                    )

                chunks = chunk_transcript_segments(segments)
                stats.transcripts_fetched += 1
                print(f"    [OK] {len(chunks)} transcript chunks")

            elif segments is None:
                # Check if quota exceeded (indicated by empty return after 429)
                stats.transcript_errors += 1

        # Fall back to description chunking
        if not chunks:
            chunks = chunk_youtube_description(description)
            if chunks:
                print(f"  [DESC] {title[:40]}... ({len(chunks)} description chunks)")

        if not chunks:
            print(f"  [SKIP] {title[:40]}... (no content)")
            stats.videos_skipped += 1
            continue

        # Save chunks
        if not dry_run:
            for chunk in chunks:
                cursor.execute("""
                    INSERT INTO youtube_segments (
                        video_id, segment_index, text, start_time, end_time, source
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    video_id,
                    chunk['index'],
                    chunk['text'],
                    chunk.get('start_time'),
                    chunk.get('end_time'),
                    chunk.get('source', 'description')
                ))

        stats.videos_processed += 1
        stats.video_chunks_created += len(chunks)

    if not dry_run:
        conn.commit()

    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Backfill content chunks for citation quotes')
    parser.add_argument('--web-only', action='store_true', help='Only process web articles')
    parser.add_argument('--youtube-only', action='store_true', help='Only process YouTube videos')
    parser.add_argument('--limit', type=int, help='Limit items to process')
    parser.add_argument('--dry-run', action='store_true', help='Preview without saving')
    parser.add_argument('--refetch', action='store_true', help='Re-fetch web article content from URLs')
    parser.add_argument('--no-transcripts', action='store_true', help='Skip transcript fetching')
    parser.add_argument('--db', type=str, default=str(DB_PATH), help='Database path')

    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[ERROR] Database not found: {db_path}")
        return

    print("=" * 60)
    print("Content Chunk Backfill")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Dry run: {args.dry_run}")
    print(f"Limit: {args.limit or 'none'}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    total_stats = ChunkStats()

    try:
        # Process web articles
        if not args.youtube_only:
            web_stats = process_web_articles(
                conn,
                limit=args.limit,
                dry_run=args.dry_run,
                refetch=args.refetch
            )
            total_stats.articles_processed = web_stats.articles_processed
            total_stats.articles_skipped = web_stats.articles_skipped
            total_stats.article_chunks_created = web_stats.article_chunks_created

        # Process YouTube videos
        if not args.web_only:
            yt_stats = process_youtube_videos(
                conn,
                limit=args.limit,
                dry_run=args.dry_run,
                fetch_transcripts=not args.no_transcripts
            )
            total_stats.videos_processed = yt_stats.videos_processed
            total_stats.videos_skipped = yt_stats.videos_skipped
            total_stats.video_chunks_created = yt_stats.video_chunks_created
            total_stats.transcripts_fetched = yt_stats.transcripts_fetched
            total_stats.transcript_errors = yt_stats.transcript_errors

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        if not args.youtube_only:
            print(f"\nWeb Articles:")
            print(f"  Processed: {total_stats.articles_processed}")
            print(f"  Skipped: {total_stats.articles_skipped}")
            print(f"  Chunks created: {total_stats.article_chunks_created}")

        if not args.web_only:
            print(f"\nYouTube Videos:")
            print(f"  Processed: {total_stats.videos_processed}")
            print(f"  Skipped: {total_stats.videos_skipped}")
            print(f"  Chunks created: {total_stats.video_chunks_created}")
            print(f"  Transcripts fetched: {total_stats.transcripts_fetched}")
            if total_stats.transcript_errors:
                print(f"  Transcript errors: {total_stats.transcript_errors}")

        total_chunks = total_stats.article_chunks_created + total_stats.video_chunks_created
        print(f"\nTotal chunks created: {total_chunks}")

        if args.dry_run:
            print("\n[DRY RUN] No changes saved to database")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
