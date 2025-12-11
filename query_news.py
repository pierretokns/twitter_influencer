#!/usr/bin/env python3
"""Simple CLI to query the AI news vector database."""

import sqlite3
import sqlite_vec
import sys
from sentence_transformers import SentenceTransformer

DB_PATH = "output_data/ai_news.db"

def query_similar(query: str, limit: int = 10):
    """Find tweets similar to the query."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(query).tolist()

    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)

    results = db.execute("""
        SELECT t.username, t.text, t.likes_count, t.url,
               vec_distance_cosine(e.embedding, ?) as distance
        FROM tweet_embeddings e
        JOIN tweets t ON e.tweet_id = t.tweet_id
        ORDER BY distance ASC
        LIMIT ?
    """, [str(embedding), limit]).fetchall()

    print(f"\n🔍 Top {limit} results for: '{query}'\n" + "="*60)
    for i, (user, text, likes, url, dist) in enumerate(results, 1):
        similarity = 1 - dist
        print(f"\n{i}. @{user} (❤️ {likes}) - {similarity:.1%} match")
        print(f"   {text[:200]}...")
        print(f"   {url}")

    db.close()

def list_topics():
    """List all detected topics."""
    db = sqlite3.connect(DB_PATH)
    topics = db.execute("""
        SELECT name, tweet_count, avg_engagement, trend_direction
        FROM topics ORDER BY tweet_count DESC
    """).fetchall()

    print("\n📊 Topics:\n" + "="*40)
    for name, count, eng, trend in topics:
        print(f"  • {name}: {count} tweets, {eng:.0f} avg engagement ({trend})")
    db.close()

def recent_tweets(limit: int = 20):
    """Show recent AI-relevant tweets."""
    db = sqlite3.connect(DB_PATH)
    tweets = db.execute("""
        SELECT username, text, likes_count, timestamp, url
        FROM tweets WHERE is_ai_relevant = 1
        ORDER BY timestamp DESC LIMIT ?
    """, [limit]).fetchall()

    print(f"\n🕐 Recent AI tweets:\n" + "="*60)
    for user, text, likes, ts, url in tweets:
        print(f"\n@{user} (❤️ {likes}) - {ts}")
        print(f"  {text[:200]}...")
    db.close()

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python query_news.py search '<query>'  - Semantic search")
        print("  python query_news.py topics            - List topics")
        print("  python query_news.py recent            - Recent tweets")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "search" and len(sys.argv) > 2:
        query_similar(" ".join(sys.argv[2:]))
    elif cmd == "topics":
        list_topics()
    elif cmd == "recent":
        recent_tweets()
    else:
        print("Unknown command. Run without args for help.")

if __name__ == "__main__":
    main()
