# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24.0",
#     "sqlite-vec>=0.1.0",
#     "FlagEmbedding>=1.2.0",
#     "sentence-transformers>=2.2.0",
#     "torch>=2.0.0",
# ]
# ///
"""
Regenerate Hybrid Embeddings from Historical Data
==================================================

This script regenerates BGE-M3 hybrid embeddings (dense + sparse) for all
existing tweets, web articles, and YouTube videos in the database.

Usage:
    uv run python regenerate_embeddings.py              # Regenerate all
    uv run python regenerate_embeddings.py --tweets     # Tweets only
    uv run python regenerate_embeddings.py --articles   # Web articles only
    uv run python regenerate_embeddings.py --youtube    # YouTube videos only
    uv run python regenerate_embeddings.py --batch 50   # Custom batch size
    uv run python regenerate_embeddings.py --dry-run    # Preview only
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# Lazy-loaded embedder
_hybrid_embedder = None


def get_hybrid_embedder():
    """Lazy load BGE-M3 hybrid embedder"""
    global _hybrid_embedder
    if _hybrid_embedder is None:
        try:
            import torch
            from FlagEmbedding import BGEM3FlagModel

            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            print(f"[Embedder] Loading BGE-M3 model on {device}...")
            _hybrid_embedder = BGEM3FlagModel(
                'BAAI/bge-m3',
                use_fp16=True,
                device=device
            )
            print("[Embedder] BGE-M3 loaded successfully")
        except Exception as e:
            print(f"[Embedder] Failed to load BGE-M3: {e}")
            _hybrid_embedder = None
    return _hybrid_embedder


def sparse_to_dense(sparse_dict: dict, top_k: int = 256) -> np.ndarray:
    """Convert sparse {token_id: weight} to dense top-k vector"""
    if not sparse_dict:
        return np.zeros(top_k, dtype=np.float32)
    sorted_items = sorted(sparse_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
    dense = np.zeros(top_k, dtype=np.float32)
    for i, (_, weight) in enumerate(sorted_items):
        dense[i] = weight
    return dense


def encode_texts_hybrid(texts: List[str], batch_size: int = 32) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Encode texts using BGE-M3 hybrid model.

    Returns:
        Tuple of (dense_embeddings, sparse_embeddings) or (None, None) if failed
    """
    embedder = get_hybrid_embedder()
    if embedder is None:
        return None, None

    try:
        # Truncate very long texts
        truncated = [t[:8000] if t else "" for t in texts]

        result = embedder.encode(
            truncated,
            batch_size=batch_size,
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )

        dense_embeddings = result['dense_vecs']  # (N, 1024)
        sparse_weights = result['lexical_weights']  # List of dicts

        # Convert sparse to fixed-size dense
        sparse_embeddings = np.array([
            sparse_to_dense(sw) for sw in sparse_weights
        ], dtype=np.float32)

        return dense_embeddings, sparse_embeddings

    except Exception as e:
        print(f"[Embedder] Encoding failed: {e}")
        return None, None


def regenerate_tweets(conn: sqlite3.Connection, batch_size: int = 32, dry_run: bool = False) -> int:
    """Regenerate embeddings for all tweets"""
    cursor = conn.cursor()

    # Get all tweets that need embeddings
    cursor.execute("""
        SELECT t.tweet_id, t.text
        FROM tweets t
        LEFT JOIN tweet_embeddings_dense e ON t.tweet_id = e.id
        WHERE e.id IS NULL AND t.text IS NOT NULL AND t.text != ''
    """)
    rows = cursor.fetchall()

    if not rows:
        print("[Tweets] No tweets need embedding regeneration")
        return 0

    print(f"[Tweets] Found {len(rows)} tweets needing embeddings")

    if dry_run:
        print("[Tweets] Dry run - no changes made")
        return len(rows)

    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        ids = [row[0] for row in batch]
        texts = [row[1] or "" for row in batch]

        dense, sparse = encode_texts_hybrid(texts, batch_size)

        if dense is None:
            print(f"[Tweets] Batch {i//batch_size + 1} failed - skipping")
            continue

        # Insert embeddings - sqlite-vec expects binary blob format
        for j, tweet_id in enumerate(ids):
            dense_blob = np.asarray(dense[j], dtype=np.float32).tobytes()
            sparse_blob = np.asarray(sparse[j], dtype=np.float32).tobytes()

            cursor.execute(
                "INSERT OR REPLACE INTO tweet_embeddings_dense (id, embedding) VALUES (?, ?)",
                (tweet_id, dense_blob)
            )
            cursor.execute(
                "INSERT OR REPLACE INTO tweet_embeddings_sparse (id, embedding) VALUES (?, ?)",
                (tweet_id, sparse_blob)
            )

        conn.commit()
        total += len(batch)
        print(f"[Tweets] Processed {total}/{len(rows)} ({100*total//len(rows)}%)")

    return total


def regenerate_articles(conn: sqlite3.Connection, batch_size: int = 32, dry_run: bool = False) -> int:
    """Regenerate embeddings for all web articles"""
    cursor = conn.cursor()

    # Get all articles that need embeddings
    cursor.execute("""
        SELECT a.article_id, COALESCE(a.title, '') || ' ' || COALESCE(a.content, '') as text
        FROM web_articles a
        LEFT JOIN web_article_embeddings_dense e ON a.article_id = e.id
        WHERE e.id IS NULL
    """)
    rows = cursor.fetchall()

    if not rows:
        print("[Articles] No articles need embedding regeneration")
        return 0

    print(f"[Articles] Found {len(rows)} articles needing embeddings")

    if dry_run:
        print("[Articles] Dry run - no changes made")
        return len(rows)

    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        ids = [row[0] for row in batch]
        texts = [row[1] or "" for row in batch]

        dense, sparse = encode_texts_hybrid(texts, batch_size)

        if dense is None:
            print(f"[Articles] Batch {i//batch_size + 1} failed - skipping")
            continue

        # Insert embeddings - sqlite-vec expects binary blob format
        for j, article_id in enumerate(ids):
            dense_blob = np.asarray(dense[j], dtype=np.float32).tobytes()
            sparse_blob = np.asarray(sparse[j], dtype=np.float32).tobytes()

            cursor.execute(
                "INSERT OR REPLACE INTO web_article_embeddings_dense (id, embedding) VALUES (?, ?)",
                (article_id, dense_blob)
            )
            cursor.execute(
                "INSERT OR REPLACE INTO web_article_embeddings_sparse (id, embedding) VALUES (?, ?)",
                (article_id, sparse_blob)
            )

        conn.commit()
        total += len(batch)
        print(f"[Articles] Processed {total}/{len(rows)} ({100*total//len(rows)}%)")

    return total


def regenerate_youtube(conn: sqlite3.Connection, batch_size: int = 32, dry_run: bool = False) -> int:
    """Regenerate embeddings for all YouTube videos"""
    cursor = conn.cursor()

    # Get all videos that need embeddings
    cursor.execute("""
        SELECT v.video_id, COALESCE(v.title, '') || ' ' || COALESCE(v.description, '') as text
        FROM youtube_videos v
        LEFT JOIN youtube_video_embeddings_dense e ON v.video_id = e.id
        WHERE e.id IS NULL
    """)
    rows = cursor.fetchall()

    if not rows:
        print("[YouTube] No videos need embedding regeneration")
        return 0

    print(f"[YouTube] Found {len(rows)} videos needing embeddings")

    if dry_run:
        print("[YouTube] Dry run - no changes made")
        return len(rows)

    total = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        ids = [row[0] for row in batch]
        texts = [row[1] or "" for row in batch]

        dense, sparse = encode_texts_hybrid(texts, batch_size)

        if dense is None:
            print(f"[YouTube] Batch {i//batch_size + 1} failed - skipping")
            continue

        # Insert embeddings - sqlite-vec expects binary blob format
        for j, video_id in enumerate(ids):
            dense_blob = np.asarray(dense[j], dtype=np.float32).tobytes()
            sparse_blob = np.asarray(sparse[j], dtype=np.float32).tobytes()

            cursor.execute(
                "INSERT OR REPLACE INTO youtube_video_embeddings_dense (id, embedding) VALUES (?, ?)",
                (video_id, dense_blob)
            )
            cursor.execute(
                "INSERT OR REPLACE INTO youtube_video_embeddings_sparse (id, embedding) VALUES (?, ?)",
                (video_id, sparse_blob)
            )

        conn.commit()
        total += len(batch)
        print(f"[YouTube] Processed {total}/{len(rows)} ({100*total//len(rows)}%)")

    return total


def main():
    parser = argparse.ArgumentParser(description='Regenerate hybrid embeddings from historical data')
    parser.add_argument('--tweets', action='store_true', help='Regenerate tweet embeddings only')
    parser.add_argument('--articles', action='store_true', help='Regenerate article embeddings only')
    parser.add_argument('--youtube', action='store_true', help='Regenerate YouTube video embeddings only')
    parser.add_argument('--batch', type=int, default=32, help='Batch size for encoding')
    parser.add_argument('--dry-run', action='store_true', help='Preview what would be regenerated')
    parser.add_argument('--db', type=str, default='output_data/ai_news.db', help='Database path')

    args = parser.parse_args()

    # Default to all if none specified
    do_tweets = args.tweets or (not args.tweets and not args.articles and not args.youtube)
    do_articles = args.articles or (not args.tweets and not args.articles and not args.youtube)
    do_youtube = args.youtube or (not args.tweets and not args.articles and not args.youtube)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[Error] Database not found: {db_path}")
        sys.exit(1)

    print("=" * 60)
    print("Hybrid Embeddings Regeneration")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Batch size: {args.batch}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Connect and load sqlite-vec
    conn = sqlite3.connect(str(db_path))
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        print("[OK] sqlite-vec extension loaded")
    except Exception as e:
        print(f"[Error] Could not load sqlite-vec: {e}")
        sys.exit(1)

    # Ensure tables exist
    try:
        from ai_news_scraper import AINewsDatabase
        # Just init to create tables if missing
        temp_db = AINewsDatabase(db_path)
        temp_db.close()
    except Exception as e:
        print(f"[Warning] Could not init AINewsDatabase: {e}")

    # Reconnect after table creation
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)

    results = {}

    if do_tweets:
        results['tweets'] = regenerate_tweets(conn, args.batch, args.dry_run)

    if do_articles:
        results['articles'] = regenerate_articles(conn, args.batch, args.dry_run)

    if do_youtube:
        results['youtube'] = regenerate_youtube(conn, args.batch, args.dry_run)

    conn.close()

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for source, count in results.items():
        print(f"  {source}: {count} embeddings {'would be ' if args.dry_run else ''}generated")

    total = sum(results.values())
    print(f"\nTotal: {total} embeddings")


if __name__ == "__main__":
    main()
