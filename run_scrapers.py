# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests>=2.31.0",
#     "python-dotenv>=0.19.0",
# ]
# ///
"""
Unified Scraper Runner
======================

Runs all scrapers in sequence, deduplicates content, and sends alerts.

Usage:
    uv run python run_scrapers.py              # Run all scrapers
    uv run python run_scrapers.py --discord    # Discord only
    uv run python run_scrapers.py --twitter    # Twitter/news only
    uv run python run_scrapers.py --dedup      # Just run deduplication
    uv run python run_scrapers.py --dry-run    # Show what would run

Alerting:
    Set ALERT_WEBHOOK_URL in .env for Discord/Slack webhook notifications
    Set ALERT_EMAIL for email notifications (requires SMTP config)
"""

import os
import sys
import sqlite3
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path("output_data")
DISCORD_DB = OUTPUT_DIR / "discord_links.db"
AI_NEWS_DB = OUTPUT_DIR / "ai_news.db"

# Alert thresholds
HIGH_VALUE_ENGAGEMENT_THRESHOLD = 1000  # likes + retweets
HIGH_VALUE_SIMILARITY_THRESHOLD = 0.9   # duplicate detection


# =============================================================================
# LOGGING
# =============================================================================

class Logger:
    COLORS = {
        'RED': '\033[91m', 'GREEN': '\033[92m', 'YELLOW': '\033[93m',
        'BLUE': '\033[94m', 'MAGENTA': '\033[95m', 'RESET': '\033[0m',
    }

    @classmethod
    def _log(cls, color: str, prefix: str, msg: str):
        ts = datetime.now().strftime('%H:%M:%S')
        print(f"{cls.COLORS[color]}[{ts}] {prefix}{cls.COLORS['RESET']} {msg}")

    @classmethod
    def info(cls, msg): cls._log('BLUE', 'INFO', msg)
    @classmethod
    def success(cls, msg): cls._log('GREEN', 'OK', msg)
    @classmethod
    def warning(cls, msg): cls._log('YELLOW', 'WARN', msg)
    @classmethod
    def error(cls, msg): cls._log('RED', 'ERROR', msg)


# =============================================================================
# ALERTING
# =============================================================================

@dataclass
class Alert:
    level: str  # info, warning, error
    title: str
    message: str
    data: Optional[Dict] = None


class AlertManager:
    """Send alerts via webhook or email"""

    def __init__(self):
        self.webhook_url = os.getenv("ALERT_WEBHOOK_URL")
        self.alerts: List[Alert] = []

    def add(self, level: str, title: str, message: str, data: Dict = None):
        self.alerts.append(Alert(level, title, message, data))

    def send_all(self):
        if not self.alerts:
            return

        if self.webhook_url:
            self._send_webhook()
        else:
            self._print_alerts()

        self.alerts = []

    def _send_webhook(self):
        """Send alerts to Discord/Slack webhook"""
        import requests

        # Format for Discord
        embeds = []
        for alert in self.alerts:
            color = {"info": 3447003, "warning": 16776960, "error": 15158332}.get(alert.level, 3447003)
            embed = {
                "title": f"{'🔵' if alert.level == 'info' else '🟡' if alert.level == 'warning' else '🔴'} {alert.title}",
                "description": alert.message,
                "color": color,
                "timestamp": datetime.utcnow().isoformat(),
            }
            if alert.data:
                embed["fields"] = [{"name": k, "value": str(v), "inline": True} for k, v in alert.data.items()]
            embeds.append(embed)

        payload = {"embeds": embeds[:10]}  # Discord limit

        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=10)
            if resp.status_code == 204:
                Logger.success(f"Sent {len(self.alerts)} alerts to webhook")
            else:
                Logger.warning(f"Webhook returned {resp.status_code}")
        except Exception as e:
            Logger.error(f"Webhook failed: {e}")

    def _print_alerts(self):
        """Print alerts to console when no webhook configured"""
        print("\n" + "="*60)
        print("ALERTS")
        print("="*60)
        for alert in self.alerts:
            icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}.get(alert.level, "•")
            print(f"{icon} [{alert.level.upper()}] {alert.title}")
            print(f"   {alert.message}")
            if alert.data:
                for k, v in alert.data.items():
                    print(f"   {k}: {v}")
            print()


# =============================================================================
# SCRAPER RUNNER
# =============================================================================

@dataclass
class ScrapeResult:
    scraper: str
    success: bool
    duration_seconds: float
    items_scraped: int
    items_new: int
    error: Optional[str] = None


def run_scraper(name: str, command: List[str], timeout: int = 600) -> ScrapeResult:
    """Run a scraper subprocess and capture results"""
    Logger.info(f"Starting {name}...")
    start = datetime.now()

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(__file__).parent),
        )

        duration = (datetime.now() - start).total_seconds()

        if result.returncode == 0:
            Logger.success(f"{name} completed in {duration:.1f}s")
            # Try to parse item counts from output
            items = 0
            for line in result.stdout.split('\n'):
                if 'scraped' in line.lower() or 'found' in line.lower():
                    import re
                    nums = re.findall(r'\d+', line)
                    if nums:
                        items = max(items, int(nums[0]))

            return ScrapeResult(name, True, duration, items, items)
        else:
            Logger.error(f"{name} failed: {result.stderr[:200]}")
            return ScrapeResult(name, False, duration, 0, 0, result.stderr[:500])

    except subprocess.TimeoutExpired:
        duration = (datetime.now() - start).total_seconds()
        Logger.error(f"{name} timed out after {timeout}s")
        return ScrapeResult(name, False, duration, 0, 0, f"Timeout after {timeout}s")
    except Exception as e:
        duration = (datetime.now() - start).total_seconds()
        Logger.error(f"{name} error: {e}")
        return ScrapeResult(name, False, duration, 0, 0, str(e))


# =============================================================================
# DEDUPLICATION
# =============================================================================

def get_all_urls(db_path: Path, table: str, url_column: str = "url") -> Dict[str, Dict]:
    """Get all URLs from a database table"""
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute(f"SELECT * FROM {table} WHERE {url_column} IS NOT NULL")
        rows = cursor.fetchall()
        return {row[url_column]: dict(row) for row in rows}
    except Exception as e:
        Logger.warning(f"Could not read {table} from {db_path}: {e}")
        return {}
    finally:
        conn.close()


def find_duplicates() -> List[Dict]:
    """Find URLs that appear in both Discord and AI News databases"""
    duplicates = []

    # Get URLs from discord_links
    discord_urls = get_all_urls(DISCORD_DB, "discord_links", "url")

    # Get URLs from web_articles
    article_urls = get_all_urls(AI_NEWS_DB, "web_articles", "url")

    # Get URLs from tweets (they might link to same content)
    tweet_urls = {}
    if AI_NEWS_DB.exists():
        conn = sqlite3.connect(str(AI_NEWS_DB))
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT url, tweet_id, text FROM tweets WHERE url IS NOT NULL")
            for row in cursor.fetchall():
                if row[0]:
                    tweet_urls[row[0]] = {"tweet_id": row[1], "text": row[2]}
        except:
            pass
        conn.close()

    # Find overlaps
    all_urls = set(discord_urls.keys()) | set(article_urls.keys()) | set(tweet_urls.keys())

    for url in all_urls:
        sources = []
        if url in discord_urls:
            sources.append("discord")
        if url in article_urls:
            sources.append("web_articles")
        if url in tweet_urls:
            sources.append("tweets")

        if len(sources) > 1:
            duplicates.append({
                "url": url,
                "sources": sources,
                "discord_data": discord_urls.get(url),
                "article_data": article_urls.get(url),
                "tweet_data": tweet_urls.get(url),
            })

    return duplicates


def mark_duplicates(duplicates: List[Dict]):
    """Mark duplicate content in databases (add is_duplicate flag)"""
    if not duplicates:
        return

    # Add is_duplicate column if not exists, then mark
    for db_path, table in [(DISCORD_DB, "discord_links"), (AI_NEWS_DB, "web_articles")]:
        if not db_path.exists():
            continue

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN is_duplicate BOOLEAN DEFAULT FALSE")
        except:
            pass  # Column already exists

        dup_urls = [d["url"] for d in duplicates if table.replace("_", "") in str(d.get("sources", []))]
        if dup_urls:
            placeholders = ",".join("?" * len(dup_urls))
            cursor.execute(f"UPDATE {table} SET is_duplicate = TRUE WHERE url IN ({placeholders})", dup_urls)
            conn.commit()
            Logger.info(f"Marked {cursor.rowcount} duplicates in {table}")

        conn.close()


# =============================================================================
# HIGH-VALUE CONTENT DETECTION
# =============================================================================

def find_high_value_content() -> List[Dict]:
    """Find high-engagement or trending content"""
    high_value = []

    # Check tweets for high engagement
    if AI_NEWS_DB.exists():
        conn = sqlite3.connect(str(AI_NEWS_DB))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT tweet_id, username, text, likes_count, retweets_count, url,
                       (likes_count + retweets_count) as engagement
                FROM tweets
                WHERE is_ai_relevant = 1
                  AND (likes_count + retweets_count) > ?
                  AND scraped_at > datetime('now', '-1 day')
                ORDER BY engagement DESC
                LIMIT 10
            """, (HIGH_VALUE_ENGAGEMENT_THRESHOLD,))

            for row in cursor.fetchall():
                high_value.append({
                    "type": "high_engagement_tweet",
                    "source": f"@{row['username']}",
                    "text": row['text'][:200],
                    "engagement": row['engagement'],
                    "url": row['url'],
                })
        except Exception as e:
            Logger.warning(f"Could not check high-value tweets: {e}")

        conn.close()

    # Check discord links for interesting content
    if DISCORD_DB.exists():
        conn = sqlite3.connect(str(DISCORD_DB))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            # Find YouTube videos (potentially viral)
            cursor.execute("""
                SELECT url, page_title, author_name, username
                FROM discord_links
                WHERE link_type = 'youtube'
                  AND page_title IS NOT NULL
                  AND scraped_at > datetime('now', '-1 day')
                LIMIT 5
            """)

            for row in cursor.fetchall():
                high_value.append({
                    "type": "youtube_video",
                    "source": f"Discord (@{row['username']})",
                    "text": row['page_title'],
                    "author": row['author_name'],
                    "url": row['url'],
                })
        except Exception as e:
            Logger.warning(f"Could not check discord content: {e}")

        conn.close()

    return high_value


# =============================================================================
# MAIN
# =============================================================================

def run_all_scrapers(
    run_discord: bool = True,
    run_twitter: bool = True,
    dry_run: bool = False
) -> List[ScrapeResult]:
    """Run all scrapers and return results"""
    results = []

    scrapers = []
    if run_discord:
        scrapers.append(("Discord Links", ["uv", "run", "python", "discord_link_scraper.py"]))
    if run_twitter:
        scrapers.append(("AI News (Twitter)", ["uv", "run", "python", "ai_news_scraper.py"]))

    if dry_run:
        Logger.info("DRY RUN - would execute:")
        for name, cmd in scrapers:
            print(f"  {name}: {' '.join(cmd)}")
        return []

    for name, cmd in scrapers:
        result = run_scraper(name, cmd)
        results.append(result)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Unified Scraper Runner")
    parser.add_argument("--discord", action="store_true", help="Run Discord scraper only")
    parser.add_argument("--twitter", action="store_true", help="Run Twitter/news scraper only")
    parser.add_argument("--dedup", action="store_true", help="Run deduplication only")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--no-alert", action="store_true", help="Disable alerts")

    args = parser.parse_args()

    alerts = AlertManager()
    start_time = datetime.now()

    print("="*60)
    print(f"UNIFIED SCRAPER - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Determine what to run
    run_discord = args.discord or (not args.discord and not args.twitter and not args.dedup)
    run_twitter = args.twitter or (not args.discord and not args.twitter and not args.dedup)

    # Run scrapers
    results = []
    if not args.dedup:
        results = run_all_scrapers(run_discord, run_twitter, args.dry_run)

        # Generate alerts for failures
        for r in results:
            if not r.success:
                alerts.add("error", f"{r.scraper} Failed", r.error or "Unknown error", {
                    "Duration": f"{r.duration_seconds:.1f}s"
                })

    # Run deduplication
    if not args.dry_run:
        Logger.info("Checking for duplicates...")
        duplicates = find_duplicates()
        if duplicates:
            Logger.warning(f"Found {len(duplicates)} duplicate URLs across databases")
            mark_duplicates(duplicates)
            alerts.add("info", "Duplicates Found", f"{len(duplicates)} URLs appear in multiple sources", {
                "Count": len(duplicates)
            })

        # Find high-value content
        Logger.info("Checking for high-value content...")
        high_value = find_high_value_content()
        if high_value:
            Logger.success(f"Found {len(high_value)} high-value items")
            for item in high_value[:3]:  # Alert top 3
                alerts.add("info", f"High Value: {item['type']}", item['text'][:100], {
                    "Source": item.get('source', 'Unknown'),
                    "URL": item.get('url', 'N/A')[:50]
                })

    # Summary
    duration = (datetime.now() - start_time).total_seconds()
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Duration: {duration:.1f}s")
    for r in results:
        status = "✓" if r.success else "✗"
        print(f"  {status} {r.scraper}: {r.items_scraped} items in {r.duration_seconds:.1f}s")

    # Send alerts
    if not args.no_alert:
        alerts.send_all()


if __name__ == "__main__":
    main()
