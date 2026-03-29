# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Clean up YouTube scrape data and seed channels.

This script deletes:
1. All YouTube channels from youtube_channels table
2. All YouTube videos from youtube_videos table
3. All YouTube video embeddings
"""

import sqlite3
from pathlib import Path


def cleanup_youtube_data(db_path: Path) -> None:
    """Delete all YouTube scrape data from database."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        print("[Cleanup] Starting YouTube data cleanup...")

        # Get counts before deletion
        cursor.execute("SELECT COUNT(*) FROM youtube_channels")
        channel_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM youtube_videos")
        video_count = cursor.fetchone()[0]

        print(f"[Cleanup] Found {channel_count} channels and {video_count} videos")

        # Delete in order of foreign key dependencies
        # Note: sqlite-vec virtual tables don't need explicit deletion, they're tied to the rowid

        print("[Cleanup] Deleting YouTube videos...")
        cursor.execute("DELETE FROM youtube_videos")

        print("[Cleanup] Deleting YouTube segments...")
        cursor.execute("DELETE FROM youtube_segments")

        print("[Cleanup] Deleting YouTube channels...")
        cursor.execute("DELETE FROM youtube_channels")

        conn.commit()

        # Verify deletion
        cursor.execute("SELECT COUNT(*) FROM youtube_channels")
        remaining_channels = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM youtube_videos")
        remaining_videos = cursor.fetchone()[0]

        print(f"[Cleanup] Deleted {channel_count} channels")
        print(f"[Cleanup] Deleted {video_count} videos")
        print(f"[Cleanup] Remaining channels: {remaining_channels}")
        print(f"[Cleanup] Remaining videos: {remaining_videos}")

        if remaining_channels == 0 and remaining_videos == 0:
            print("[Cleanup] ✓ All YouTube data cleaned up successfully!")
        else:
            print("[Cleanup] ⚠ Warning: Some data remains")

    except Exception as e:
        print(f"[Cleanup] ERROR: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    db_path = Path(__file__).parent / 'output_data' / 'ai_news.db'

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        exit(1)

    print(f"[Cleanup] Cleaning up YouTube data in {db_path}")
    cleanup_youtube_data(db_path)
