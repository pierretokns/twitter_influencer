# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Database Migration System for SQLite
=====================================

Simple version-based migrations using SQLite's PRAGMA user_version.
Supports multiple databases with separate migration tracks.

CRITICAL: READ BEFORE MODIFYING SCHEMA
--------------------------------------
1. NEVER modify existing migrations - only add new ones
2. ALWAYS bump the version number when adding migrations
3. TEST migrations on a copy before deploying

HOW TO ADD A NEW MIGRATION
--------------------------
1. Increment the version constant:

   DISCORD_DB_VERSION = 3  # was 2

2. Add SQL statements to the migrations dict:

   DISCORD_MIGRATIONS[3] = [
       "ALTER TABLE discord_links ADD COLUMN new_field TEXT",
       "CREATE INDEX IF NOT EXISTS idx_new ON discord_links(new_field)",
   ]

3. Test:

   python db_migrations.py --check  # verify pending
   python db_migrations.py          # apply migrations

SQLITE ALTER TABLE LIMITATIONS
------------------------------
- CAN: ADD COLUMN, RENAME COLUMN/TABLE
- CANNOT: DROP COLUMN, CHANGE TYPE, ADD CONSTRAINT

For complex changes, use the copy-and-replace pattern:
    CREATE TABLE new_table AS SELECT ... FROM old_table;
    DROP TABLE old_table;
    ALTER TABLE new_table RENAME TO old_table;

CURRENT VERSIONS
----------------
- discord_links: v2 (base schema + temporal metadata)
- ai_news: v6 (base + web_articles + tournaments + likes + sources + youtube)

USAGE
-----
    # In code (auto-runs on class init):
    from db_migrations import migrate_discord_db
    migrate_discord_db(conn)

    # CLI:
    python db_migrations.py --check   # status
    python db_migrations.py           # migrate
"""

import sqlite3
from typing import List, Dict, Callable, Optional
from datetime import datetime


class MigrationError(Exception):
    """Raised when a migration fails"""
    pass


def get_db_version(conn: sqlite3.Connection) -> int:
    """Get current schema version from database"""
    return conn.execute("PRAGMA user_version").fetchone()[0]


def set_db_version(conn: sqlite3.Connection, version: int):
    """Set schema version in database"""
    conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()


def run_migrations(
    conn: sqlite3.Connection,
    migrations: Dict[int, List[str]],
    target_version: int,
    db_name: str = "database"
) -> int:
    """
    Run pending migrations on a database connection.

    Args:
        conn: SQLite connection
        migrations: Dict mapping version numbers to lists of SQL statements
        target_version: The version to migrate to
        db_name: Name for logging purposes

    Returns:
        Number of migrations applied
    """
    current_version = get_db_version(conn)

    if current_version >= target_version:
        return 0

    migrations_applied = 0

    print(f"[MIGRATE] {db_name}: v{current_version} -> v{target_version}")

    for version in range(current_version + 1, target_version + 1):
        if version not in migrations:
            print(f"[WARN] Migration v{version} not found, skipping")
            continue

        print(f"[MIGRATE] Applying v{version}...")

        for sql in migrations[version]:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError as e:
                # Handle "column already exists" and similar idempotent errors
                error_msg = str(e).lower()
                if any(x in error_msg for x in [
                    "already exists",
                    "duplicate column",
                    "table .* already exists",
                ]):
                    print(f"  [SKIP] {e}")
                else:
                    print(f"  [ERROR] {e}")
                    raise MigrationError(f"Migration v{version} failed: {e}")

        set_db_version(conn, version)
        migrations_applied += 1
        print(f"  [OK] v{version} applied")

    return migrations_applied


# =============================================================================
# DISCORD LINKS DATABASE MIGRATIONS
# =============================================================================

DISCORD_DB_VERSION = 2

DISCORD_MIGRATIONS: Dict[int, List[str]] = {
    # Version 1: Initial schema
    1: [
        # Discord servers
        """
        CREATE TABLE IF NOT EXISTS discord_servers (
            server_id TEXT PRIMARY KEY,
            server_name TEXT,
            first_scraped TEXT DEFAULT CURRENT_TIMESTAMP,
            last_scraped TEXT
        )
        """,

        # Discord channels
        """
        CREATE TABLE IF NOT EXISTS discord_channels (
            channel_id TEXT PRIMARY KEY,
            server_id TEXT,
            channel_name TEXT,
            channel_type TEXT,
            first_scraped TEXT DEFAULT CURRENT_TIMESTAMP,
            last_scraped TEXT,
            FOREIGN KEY (server_id) REFERENCES discord_servers(server_id)
        )
        """,

        # Discord users being tracked
        """
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
        """,

        # Discord messages
        """
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
        """,

        # Links extracted from messages
        """
        CREATE TABLE IF NOT EXISTS discord_links (
            link_id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT,
            url TEXT,
            domain TEXT,
            username TEXT,
            server_id TEXT,
            channel_id TEXT,
            found_at TEXT DEFAULT CURRENT_TIMESTAMP,
            is_scraped BOOLEAN DEFAULT FALSE,
            scraped_at TEXT,
            page_title TEXT,
            page_description TEXT,
            page_content TEXT,
            content_type TEXT,
            scrape_error TEXT,
            FOREIGN KEY (message_id) REFERENCES discord_messages(message_id),
            UNIQUE(url, message_id)
        )
        """,

        # Scrape history
        """
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
        """,

        # Indexes
        "CREATE INDEX IF NOT EXISTS idx_discord_links_url ON discord_links(url)",
        "CREATE INDEX IF NOT EXISTS idx_discord_links_domain ON discord_links(domain)",
        "CREATE INDEX IF NOT EXISTS idx_discord_links_username ON discord_links(username)",
        "CREATE INDEX IF NOT EXISTS idx_discord_links_scraped ON discord_links(is_scraped)",
        "CREATE INDEX IF NOT EXISTS idx_discord_messages_username ON discord_messages(username)",
        "CREATE INDEX IF NOT EXISTS idx_discord_messages_timestamp ON discord_messages(timestamp)",
    ],

    # Version 2: Add temporal metadata and content type tracking
    2: [
        "ALTER TABLE discord_links ADD COLUMN published_at TEXT",
        "ALTER TABLE discord_links ADD COLUMN author_name TEXT",
        "ALTER TABLE discord_links ADD COLUMN link_type TEXT",
        "ALTER TABLE discord_links ADD COLUMN duration_seconds INTEGER",
    ],
}


def migrate_discord_db(conn: sqlite3.Connection) -> int:
    """Run migrations for discord_links database"""
    return run_migrations(
        conn,
        DISCORD_MIGRATIONS,
        DISCORD_DB_VERSION,
        "discord_links"
    )


# =============================================================================
# AI NEWS DATABASE MIGRATIONS
# =============================================================================

AI_NEWS_DB_VERSION = 6

AI_NEWS_MIGRATIONS: Dict[int, List[str]] = {
    # Version 1: Initial schema
    1: [
        # Influencers table
        """
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
        """,

        # Tweets table
        """
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
        """,

        # Topics table
        """
        CREATE TABLE IF NOT EXISTS topics (
            topic_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            keywords TEXT,
            tweet_count INTEGER DEFAULT 0,
            avg_engagement REAL DEFAULT 0.0,
            first_seen TEXT,
            last_seen TEXT,
            trend_direction TEXT DEFAULT 'stable',
            is_emerging BOOLEAN DEFAULT FALSE,
            centroid_embedding BLOB
        )
        """,

        # Topic-Tweet mapping
        """
        CREATE TABLE IF NOT EXISTS topic_tweets (
            topic_id INTEGER,
            tweet_id TEXT,
            similarity_score REAL,
            PRIMARY KEY (topic_id, tweet_id),
            FOREIGN KEY (topic_id) REFERENCES topics(topic_id),
            FOREIGN KEY (tweet_id) REFERENCES tweets(tweet_id)
        )
        """,

        # Mentioned accounts for discovery
        """
        CREATE TABLE IF NOT EXISTS mentioned_accounts (
            username TEXT PRIMARY KEY,
            mention_count INTEGER DEFAULT 1,
            mentioned_by TEXT,
            first_seen TEXT,
            last_seen TEXT,
            is_promoted BOOLEAN DEFAULT FALSE
        )
        """,

        # Web articles table
        """
        CREATE TABLE IF NOT EXISTS web_articles (
            article_id TEXT PRIMARY KEY,
            source_id TEXT,
            source_name TEXT,
            title TEXT,
            url TEXT UNIQUE,
            description TEXT,
            content TEXT,
            author TEXT,
            published_at TEXT,
            category TEXT,
            is_ai_relevant BOOLEAN DEFAULT FALSE,
            ai_relevance_score REAL DEFAULT 0.0,
            scraped_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_id) REFERENCES web_sources(source_id)
        )
        """,

        # Web sources table
        """
        CREATE TABLE IF NOT EXISTS web_sources (
            source_id TEXT PRIMARY KEY,
            name TEXT,
            url TEXT,
            source_type TEXT,
            category TEXT,
            last_scraped TEXT,
            articles_count INTEGER DEFAULT 0,
            is_active BOOLEAN DEFAULT TRUE
        )
        """,

        # Scrape history
        """
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
        """,

        # Indexes
        "CREATE INDEX IF NOT EXISTS idx_tweets_timestamp ON tweets(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_tweets_username ON tweets(username)",
        "CREATE INDEX IF NOT EXISTS idx_tweets_ai_relevant ON tweets(is_ai_relevant)",
        "CREATE INDEX IF NOT EXISTS idx_influencers_score ON influencers(quality_score DESC)",
        "CREATE INDEX IF NOT EXISTS idx_web_articles_published ON web_articles(published_at)",
        "CREATE INDEX IF NOT EXISTS idx_web_articles_source ON web_articles(source_id)",
        "CREATE INDEX IF NOT EXISTS idx_web_articles_ai_relevant ON web_articles(is_ai_relevant)",
    ],

    # Version 2: Add enhanced metadata for web articles
    2: [
        "ALTER TABLE web_articles ADD COLUMN link_type TEXT",
        "ALTER TABLE web_articles ADD COLUMN author_name TEXT",
        "ALTER TABLE web_articles ADD COLUMN duration_seconds INTEGER",
    ],

    # Version 3: Tournament results tracking
    3: [
        # Tournament runs table - one row per tournament execution
        """
        CREATE TABLE IF NOT EXISTS tournament_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT DEFAULT CURRENT_TIMESTAMP,
            completed_at TEXT,
            num_variants INTEGER,
            num_rounds INTEGER,
            total_debates INTEGER DEFAULT 0,
            winner_variant_id TEXT,
            winner_content TEXT,
            winner_elo REAL,
            winner_qe_score INTEGER,
            was_published BOOLEAN DEFAULT FALSE,
            published_at TEXT,
            status TEXT DEFAULT 'running'
        )
        """,

        # Tournament variants - all posts generated for a tournament
        """
        CREATE TABLE IF NOT EXISTS tournament_variants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            variant_id TEXT,
            hook_style TEXT,
            content TEXT,
            elo_rating REAL DEFAULT 1000,
            qe_score INTEGER,
            qe_feedback TEXT,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            generation INTEGER DEFAULT 1,
            evolved_from TEXT,
            evolution_feedback TEXT,
            is_duplicate BOOLEAN DEFAULT FALSE,
            final_rank INTEGER,
            FOREIGN KEY (run_id) REFERENCES tournament_runs(run_id)
        )
        """,

        # Tournament debates - individual matchups
        """
        CREATE TABLE IF NOT EXISTS tournament_debates (
            debate_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            round_num INTEGER,
            variant_a_id TEXT,
            variant_b_id TEXT,
            winner_id TEXT,
            reasoning TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES tournament_runs(run_id)
        )
        """,

        # Indexes for tournament tables
        "CREATE INDEX IF NOT EXISTS idx_tournament_variants_run ON tournament_variants(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_tournament_debates_run ON tournament_debates(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_tournament_runs_status ON tournament_runs(status)",
    ],

    # Version 4: Post likes and user engagement tracking
    4: [
        # Post likes table - tracks user likes on tournament winners
        """
        CREATE TABLE IF NOT EXISTS post_likes (
            like_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            user_id TEXT DEFAULT 'anonymous',
            user_name TEXT,
            liked_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES tournament_runs(run_id),
            UNIQUE(run_id, user_id)
        )
        """,

        # Add like_count to tournament_runs for quick access
        "ALTER TABLE tournament_runs ADD COLUMN like_count INTEGER DEFAULT 0",

        # Index for quick like lookups
        "CREATE INDEX IF NOT EXISTS idx_post_likes_run ON post_likes(run_id)",
    ],

    # Version 5: Source traceability - track which news items were used for each post
    5: [
        # Tournament sources table - links posts to source news items
        """
        CREATE TABLE IF NOT EXISTS tournament_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            source_type TEXT,
            source_id TEXT,
            source_text TEXT,
            source_url TEXT,
            source_author TEXT,
            source_timestamp TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES tournament_runs(run_id)
        )
        """,

        # Index for quick source lookups by run
        "CREATE INDEX IF NOT EXISTS idx_tournament_sources_run ON tournament_sources(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_tournament_sources_type ON tournament_sources(source_type)",
    ],

    # Version 6: YouTube channel/video tracking + source relevance
    6: [
        # YouTube channels table
        """
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
        """,

        # YouTube videos table
        """
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
        """,

        # Add is_referenced flag to track which sources were actually used in the post
        "ALTER TABLE tournament_sources ADD COLUMN is_referenced BOOLEAN DEFAULT FALSE",

        # Indexes for YouTube tables
        "CREATE INDEX IF NOT EXISTS idx_yt_videos_channel ON youtube_videos(channel_id)",
        "CREATE INDEX IF NOT EXISTS idx_yt_videos_published ON youtube_videos(published_at)",
        "CREATE INDEX IF NOT EXISTS idx_yt_videos_ai_relevant ON youtube_videos(is_ai_relevant)",
    ],
}


def migrate_ai_news_db(conn: sqlite3.Connection) -> int:
    """Run migrations for ai_news database"""
    return run_migrations(
        conn,
        AI_NEWS_MIGRATIONS,
        AI_NEWS_DB_VERSION,
        "ai_news"
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_migration_status(conn: sqlite3.Connection, target_version: int, db_name: str = "database"):
    """Print migration status without applying changes"""
    current = get_db_version(conn)
    if current >= target_version:
        print(f"[OK] {db_name}: v{current} (up to date)")
    else:
        print(f"[PENDING] {db_name}: v{current} -> v{target_version} ({target_version - current} migrations pending)")


def init_discord_db(db_path: str) -> sqlite3.Connection:
    """Initialize and migrate discord_links database"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    migrate_discord_db(conn)
    return conn


def init_ai_news_db(db_path: str) -> sqlite3.Connection:
    """Initialize and migrate ai_news database"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    migrate_ai_news_db(conn)
    return conn


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for checking/running migrations"""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Database Migration Tool')
    parser.add_argument('--discord', type=str, help='Path to discord_links.db')
    parser.add_argument('--ai-news', type=str, help='Path to ai_news.db')
    parser.add_argument('--check', action='store_true', help='Check status without migrating')
    parser.add_argument('--output-dir', type=str, default='output_data', help='Output directory')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Discord DB
    discord_path = args.discord or output_dir / 'discord_links.db'
    if Path(discord_path).exists():
        conn = sqlite3.connect(str(discord_path))
        if args.check:
            check_migration_status(conn, DISCORD_DB_VERSION, "discord_links")
        else:
            migrate_discord_db(conn)
        conn.close()
    else:
        print(f"[SKIP] {discord_path} not found")

    # AI News DB
    ai_news_path = args.ai_news or output_dir / 'ai_news.db'
    if Path(ai_news_path).exists():
        conn = sqlite3.connect(str(ai_news_path))
        if args.check:
            check_migration_status(conn, AI_NEWS_DB_VERSION, "ai_news")
        else:
            migrate_ai_news_db(conn)
        conn.close()
    else:
        print(f"[SKIP] {ai_news_path} not found")


if __name__ == "__main__":
    main()
