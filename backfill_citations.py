# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "sqlite-vec>=0.1.0",
# ]
# ///
"""
Backfill script to insert citation markers into existing tournament posts.

This fixes the issue where tournaments had citations stored in the database
but not actually displayed in the post content.
"""

import sqlite3
import sys
import re
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from agents.variant_generator import PostVariantGenerator


def get_cited_sources(conn: sqlite3.Connection, run_id: int) -> dict:
    """Get only cited sources with their citation numbers."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT citation_number, cited_quote
        FROM tournament_sources
        WHERE run_id = ? AND citation_number IS NOT NULL
        ORDER BY citation_number
    """, (run_id,))

    citations = {}
    for row in cursor.fetchall():
        citation_num = row['citation_number']
        cited_quote = row['cited_quote']
        if citation_num and cited_quote:
            citations[citation_num] = cited_quote

    return citations


def insert_citation_markers_simple(content: str, citations: dict) -> str:
    """
    Insert citation markers based on cited_quote matches.

    This is simpler than sentence matching and works better for backfill
    because it uses exact quotes from the database.
    """
    if not citations:
        return content

    # Sort by citation number in reverse to avoid position shifts
    for citation_num in sorted(citations.keys(), reverse=True):
        cited_quote = citations[citation_num]
        if not cited_quote:
            continue

        # Try to find the quoted text in the content (case-insensitive)
        quote_lower = cited_quote.lower()
        content_lower = content.lower()

        idx = content_lower.find(quote_lower)
        if idx >= 0:
            # Found the quote - add marker at the end of the sentence
            end_idx = idx + len(cited_quote)

            # Find the sentence boundary (period, exclamation, question mark)
            while end_idx < len(content) and content[end_idx] not in '.!?\n':
                end_idx += 1

            if end_idx < len(content) and content[end_idx] in '.!?':
                # Insert marker before the punctuation
                new_content = (
                    content[:end_idx] +
                    f'[{citation_num}]' +
                    content[end_idx:]
                )
                content = new_content

    return content


def backfill_citations(db_path: Path) -> None:
    """Backfill citation markers into existing tournament posts."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Get all completed tournaments
        cursor.execute("""
            SELECT run_id, winner_content
            FROM tournament_runs
            WHERE status = 'complete' AND winner_content IS NOT NULL
            ORDER BY run_id DESC
        """)

        tournaments = cursor.fetchall()
        print(f"[Backfill] Found {len(tournaments)} completed tournaments")

        if not tournaments:
            print("[Backfill] No tournaments to backfill")
            return

        generator = PostVariantGenerator()
        updated_count = 0

        for tournament in tournaments:
            run_id = tournament['run_id']
            original_content = tournament['winner_content']

            # Check if already has citation markers
            if re.search(r'\[\d+\]', original_content):
                print(f"[Backfill] Run #{run_id}: Already has citations, skipping")
                continue

            try:
                # Get citations from database
                citations = get_cited_sources(conn, run_id)

                if not citations:
                    print(f"[Backfill] Run #{run_id}: No citation data found, skipping")
                    continue

                # Insert citation markers using quoted text matching
                new_content = insert_citation_markers_simple(original_content, citations)

                # Check if content actually changed
                if new_content == original_content:
                    print(f"[Backfill] Run #{run_id}: Could not match citations to content, skipping")
                    continue

                # Update the tournament with new content
                cursor.execute("""
                    UPDATE tournament_runs
                    SET winner_content = ?
                    WHERE run_id = ?
                """, (new_content, run_id))

                conn.commit()
                updated_count += 1
                marker_count = len(re.findall(r'\[\d+\]', new_content))
                print(f"[Backfill] Run #{run_id}: Added {marker_count} citation markers ✓")

            except Exception as e:
                print(f"[Backfill] Run #{run_id}: ERROR - {e}")
                conn.rollback()
                continue

        print(f"\n[Backfill] Done! Updated {updated_count} tournaments")

    finally:
        conn.close()


if __name__ == '__main__':
    db_path = Path(__file__).parent / 'output_data' / 'ai_news.db'

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)

    print(f"[Backfill] Starting citation backfill for {db_path}")
    backfill_citations(db_path)
    print("[Backfill] Complete!")
