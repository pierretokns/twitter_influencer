# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "discord.py>=2.3.0",
#     "Mastodon.py>=2.0.0",
#     "python-dotenv>=0.19.0",
# ]
# ///
"""
Discord Bot for Tournament Rankings & Mastodon Approval
========================================================

Commands:
    /rankings - Show top 10 posts by ELO rating
    /pending  - List posts in mastodon_queue with status='pending'
    /approve <id> - Approve post for Mastodon
    /reject <id>  - Reject post
    /post <id>    - Immediately post to Mastodon
    /stats        - Show OTEL trace stats

ENV VARS:
    DISCORD_BOT_TOKEN: Discord bot token
    MASTODON_ACCESS_TOKEN: Mastodon API token
    MASTODON_API_BASE: e.g. https://mastodon.social
    DISCORD_APPROVAL_CHANNEL: Channel ID for approval notifications

USAGE:
    uv run python discord_bot.py
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from mastodon import Mastodon

load_dotenv()

# Configuration
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
MASTODON_ACCESS_TOKEN = os.getenv("MASTODON_ACCESS_TOKEN")
MASTODON_API_BASE = os.getenv("MASTODON_API_BASE", "https://mastodon.social")
DISCORD_APPROVAL_CHANNEL = os.getenv("DISCORD_APPROVAL_CHANNEL")
DB_PATH = Path("output_data/ai_news.db")


class MastodonClient:
    """Mastodon API client wrapper"""

    def __init__(self):
        self.client = None
        if MASTODON_ACCESS_TOKEN:
            self.client = Mastodon(
                access_token=MASTODON_ACCESS_TOKEN,
                api_base_url=MASTODON_API_BASE
            )

    def post_status(self, content: str) -> Optional[dict]:
        """Post a status to Mastodon"""
        if not self.client:
            return None
        try:
            status = self.client.status_post(content, visibility="public")
            return {"id": status["id"], "url": status["url"]}
        except Exception as e:
            print(f"[Mastodon] Error posting: {e}")
            return None

    def verify_credentials(self) -> bool:
        """Check if credentials are valid"""
        if not self.client:
            return False
        try:
            self.client.account_verify_credentials()
            return True
        except Exception:
            return False


class Database:
    """Database operations for the Discord bot"""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def get_top_rankings(self, limit: int = 10) -> list:
        """Get top posts by ELO rating"""
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT
                tv.variant_id,
                tv.hook_style,
                tv.elo_rating,
                tv.qe_score,
                tv.wins,
                tv.losses,
                tv.content,
                tr.run_id,
                tr.started_at
            FROM tournament_variants tv
            JOIN tournament_runs tr ON tv.run_id = tr.run_id
            WHERE tr.status = 'completed'
            ORDER BY tv.elo_rating DESC
            LIMIT ?
        """, (limit,))
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def get_pending_posts(self) -> list:
        """Get posts pending approval"""
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT
                mq.id,
                mq.tournament_run_id,
                mq.content,
                mq.status,
                mq.created_at,
                tr.winner_elo
            FROM mastodon_queue mq
            LEFT JOIN tournament_runs tr ON mq.tournament_run_id = tr.run_id
            WHERE mq.status = 'pending'
            ORDER BY mq.created_at DESC
        """)
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def get_post_by_id(self, post_id: int) -> Optional[dict]:
        """Get a specific post from the queue"""
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT * FROM mastodon_queue WHERE id = ?
        """, (post_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def update_post_status(
        self,
        post_id: int,
        status: str,
        approved_by: str = None,
        mastodon_status_id: str = None,
        rejected_reason: str = None
    ):
        """Update post status in the queue"""
        conn = self._get_conn()
        now = datetime.utcnow().isoformat()

        if status == "approved":
            conn.execute("""
                UPDATE mastodon_queue
                SET status = ?, approved_at = ?, approved_by = ?
                WHERE id = ?
            """, (status, now, approved_by, post_id))
        elif status == "posted":
            conn.execute("""
                UPDATE mastodon_queue
                SET status = ?, posted_at = ?, mastodon_status_id = ?
                WHERE id = ?
            """, (status, now, mastodon_status_id, post_id))
        elif status == "rejected":
            conn.execute("""
                UPDATE mastodon_queue
                SET status = ?, rejected_reason = ?
                WHERE id = ?
            """, (status, rejected_reason, post_id))
        else:
            conn.execute("""
                UPDATE mastodon_queue SET status = ? WHERE id = ?
            """, (status, post_id))

        conn.commit()
        conn.close()

    def get_otel_stats(self) -> dict:
        """Get OTEL trace statistics"""
        conn = self._get_conn()

        stats = {}

        # Tournament count
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM otel_spans WHERE name = 'elo_tournament'
        """)
        stats["total_tournaments"] = cursor.fetchone()["count"]

        # Debate count
        cursor = conn.execute("""
            SELECT COUNT(*) as count FROM otel_spans WHERE name = 'debate_match'
        """)
        stats["total_debates"] = cursor.fetchone()["count"]

        # Average debate duration
        cursor = conn.execute("""
            SELECT AVG(duration_ms) as avg_ms FROM otel_spans WHERE name = 'debate_match'
        """)
        row = cursor.fetchone()
        stats["avg_debate_duration_ms"] = row["avg_ms"] or 0

        # Average confidence
        cursor = conn.execute("""
            SELECT AVG(json_extract(attributes, '$.match.confidence')) as avg_conf
            FROM otel_spans WHERE name = 'debate_match'
        """)
        row = cursor.fetchone()
        stats["avg_confidence"] = row["avg_conf"] or 0

        # Recent tournaments
        cursor = conn.execute("""
            SELECT
                json_extract(attributes, '$.tournament.variant_count') as variants,
                json_extract(attributes, '$.tournament.rounds') as rounds,
                duration_ms,
                created_at
            FROM otel_spans
            WHERE name = 'elo_tournament'
            ORDER BY created_at DESC
            LIMIT 5
        """)
        stats["recent_tournaments"] = [dict(row) for row in cursor.fetchall()]

        conn.close()
        return stats


# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Initialize clients
db = Database(DB_PATH)
mastodon = MastodonClient()


@bot.event
async def on_ready():
    print(f"[Discord] Logged in as {bot.user}")
    print(f"[Discord] Mastodon connected: {mastodon.verify_credentials()}")

    # Sync slash commands
    try:
        synced = await bot.tree.sync()
        print(f"[Discord] Synced {len(synced)} commands")
    except Exception as e:
        print(f"[Discord] Failed to sync commands: {e}")


@bot.tree.command(name="rankings", description="Show top posts by ELO rating")
async def rankings(interaction: discord.Interaction, limit: int = 10):
    """Show top posts by ELO rating"""
    await interaction.response.defer()

    rankings = db.get_top_rankings(limit)

    if not rankings:
        await interaction.followup.send("No tournament results found.")
        return

    embed = discord.Embed(
        title="Top Posts by ELO Rating",
        color=discord.Color.gold()
    )

    for i, post in enumerate(rankings, 1):
        content_preview = post["content"][:100] + "..." if len(post["content"]) > 100 else post["content"]
        embed.add_field(
            name=f"#{i} - ELO: {post['elo_rating']:.0f}",
            value=f"**{post['hook_style']}** ({post['wins']}W-{post['losses']}L)\n{content_preview}",
            inline=False
        )

    await interaction.followup.send(embed=embed)


@bot.tree.command(name="pending", description="List posts awaiting Mastodon approval")
async def pending(interaction: discord.Interaction):
    """List pending posts"""
    await interaction.response.defer()

    posts = db.get_pending_posts()

    if not posts:
        await interaction.followup.send("No pending posts.")
        return

    embed = discord.Embed(
        title="Pending Posts",
        description=f"{len(posts)} posts awaiting approval",
        color=discord.Color.blue()
    )

    for post in posts[:10]:
        content_preview = post["content"][:150] + "..." if len(post["content"]) > 150 else post["content"]
        embed.add_field(
            name=f"ID: {post['id']} (ELO: {post.get('winner_elo', 'N/A')})",
            value=f"{content_preview}\n`/approve {post['id']}` or `/reject {post['id']}`",
            inline=False
        )

    await interaction.followup.send(embed=embed)


@bot.tree.command(name="approve", description="Approve a post for Mastodon")
async def approve(interaction: discord.Interaction, post_id: int):
    """Approve a post"""
    await interaction.response.defer()

    post = db.get_post_by_id(post_id)
    if not post:
        await interaction.followup.send(f"Post {post_id} not found.")
        return

    if post["status"] != "pending":
        await interaction.followup.send(f"Post {post_id} is not pending (status: {post['status']}).")
        return

    db.update_post_status(post_id, "approved", approved_by=str(interaction.user))

    await interaction.followup.send(
        f"Post {post_id} approved by {interaction.user.mention}.\n"
        f"Use `/post {post_id}` to publish to Mastodon."
    )


@bot.tree.command(name="reject", description="Reject a post")
async def reject(interaction: discord.Interaction, post_id: int, reason: str = "No reason provided"):
    """Reject a post"""
    await interaction.response.defer()

    post = db.get_post_by_id(post_id)
    if not post:
        await interaction.followup.send(f"Post {post_id} not found.")
        return

    if post["status"] not in ("pending", "approved"):
        await interaction.followup.send(f"Post {post_id} cannot be rejected (status: {post['status']}).")
        return

    db.update_post_status(post_id, "rejected", rejected_reason=reason)

    await interaction.followup.send(f"Post {post_id} rejected. Reason: {reason}")


@bot.tree.command(name="post", description="Post to Mastodon")
async def post_to_mastodon(interaction: discord.Interaction, post_id: int):
    """Post to Mastodon"""
    await interaction.response.defer()

    post = db.get_post_by_id(post_id)
    if not post:
        await interaction.followup.send(f"Post {post_id} not found.")
        return

    if post["status"] == "posted":
        await interaction.followup.send(f"Post {post_id} has already been posted.")
        return

    # Post to Mastodon
    result = mastodon.post_status(post["content"])

    if result:
        db.update_post_status(post_id, "posted", mastodon_status_id=str(result["id"]))
        await interaction.followup.send(
            f"Posted to Mastodon!\n{result['url']}"
        )
    else:
        await interaction.followup.send(
            "Failed to post to Mastodon. Check bot logs for details."
        )


@bot.tree.command(name="stats", description="Show OTEL trace statistics")
async def stats(interaction: discord.Interaction):
    """Show OTEL statistics"""
    await interaction.response.defer()

    stats = db.get_otel_stats()

    embed = discord.Embed(
        title="Tournament Statistics",
        color=discord.Color.green()
    )

    embed.add_field(
        name="Total Tournaments",
        value=str(stats["total_tournaments"]),
        inline=True
    )
    embed.add_field(
        name="Total Debates",
        value=str(stats["total_debates"]),
        inline=True
    )
    embed.add_field(
        name="Avg Debate Duration",
        value=f"{stats['avg_debate_duration_ms']:.0f}ms",
        inline=True
    )
    embed.add_field(
        name="Avg Confidence",
        value=f"{stats['avg_confidence']:.1%}",
        inline=True
    )

    if stats["recent_tournaments"]:
        recent = "\n".join([
            f"- {t['variants']} variants, {t['rounds']} rounds, {t['duration_ms']:.0f}ms"
            for t in stats["recent_tournaments"]
        ])
        embed.add_field(
            name="Recent Tournaments",
            value=recent or "None",
            inline=False
        )

    await interaction.followup.send(embed=embed)


@bot.tree.command(name="preview", description="Preview a post before posting")
async def preview(interaction: discord.Interaction, post_id: int):
    """Preview a post's full content"""
    await interaction.response.defer()

    post = db.get_post_by_id(post_id)
    if not post:
        await interaction.followup.send(f"Post {post_id} not found.")
        return

    embed = discord.Embed(
        title=f"Post Preview (ID: {post_id})",
        description=post["content"],
        color=discord.Color.purple()
    )
    embed.add_field(name="Status", value=post["status"], inline=True)
    embed.add_field(name="Created", value=post["created_at"], inline=True)

    if post.get("mastodon_status_id"):
        embed.add_field(
            name="Mastodon",
            value=f"Posted (ID: {post['mastodon_status_id']})",
            inline=False
        )

    await interaction.followup.send(embed=embed)


def main():
    if not DISCORD_BOT_TOKEN:
        print("Error: DISCORD_BOT_TOKEN not set")
        return

    if not DB_PATH.exists():
        print(f"Warning: Database not found at {DB_PATH}")

    print("[Discord] Starting bot...")
    bot.run(DISCORD_BOT_TOKEN)


if __name__ == "__main__":
    main()
