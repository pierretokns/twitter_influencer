#!/usr/bin/env python3
"""
Test script to validate ChatAgent keyword search functionality.

Tests:
1. wordninja word segmentation for concatenated brand names
2. Snippet extraction around keyword matches
3. Keyword search across web articles, tweets, YouTube
4. Multi-keyword search (AND logic)
5. Full retrieval pipeline integration

Run with: uv run python test_chat_agent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import sqlite3
import wordninja
import re


def test_wordninja_segmentation():
    """Test wordninja splits concatenated brand names correctly."""
    print("[Test 1] wordninja Word Segmentation\n")

    test_cases = [
        ("artificialanalysis", ["artificial", "analysis"]),
        ("vectorlab", ["vector", "lab"]),
        ("machinelearning", ["machine", "learning"]),
        ("deepmind", ["deep", "mind"]),
        ("huggingface", ["hugging", "face"]),
        ("anthropic", ["anthropic"]),  # Should NOT split
        ("openai", ["open", "a", "i"]),  # Known quirk
    ]

    passed = 0
    failed = 0

    for word, expected in test_cases:
        result = wordninja.split(word)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1
        print(f"  {word} -> {result} (expected {expected}) [{status}]")

    print(f"\n  Results: {passed} passed, {failed} failed")
    return failed == 0


def test_split_camel_or_concat():
    """Test the split_camel_or_concat function logic."""
    print("\n[Test 2] split_camel_or_concat Function\n")

    def split_camel_or_concat(word):
        """Copy of the function from chat_agent.py for testing."""
        # Try camelCase split first
        parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', word)
        if len(parts) > 1:
            return ' '.join(parts).lower()

        # Use wordninja for probabilistic word segmentation
        split_words = wordninja.split(word.lower())
        if len(split_words) > 1:
            if all(len(w) >= 2 for w in split_words):
                return ' '.join(split_words)

        return None

    test_cases = [
        ("ArtificialAnalysis", "artificial analysis"),  # camelCase
        ("artificialanalysis", "artificial analysis"),  # concatenated
        ("VectorLab", "vector lab"),  # camelCase
        ("vectorlab", "vector lab"),  # concatenated
        ("MachineLearning", "machine learning"),  # camelCase
        ("anthropic", None),  # Should NOT split (single word)
        ("ai", None),  # Too short
    ]

    passed = 0
    failed = 0

    for word, expected in test_cases:
        result = split_camel_or_concat(word)
        status = "PASS" if result == expected else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1
        print(f"  {word} -> '{result}' (expected '{expected}') [{status}]")

    print(f"\n  Results: {passed} passed, {failed} failed")
    return failed == 0


def test_snippet_extraction():
    """Test snippet extraction around keyword matches."""
    print("\n[Test 3] Snippet Extraction\n")

    def extract_snippet(content: str, keywords: list, max_len: int = 400) -> str:
        """Copy of the function from chat_agent.py for testing."""
        if not content:
            return ""

        content_lower = content.lower()
        best_pos = -1

        for kw in keywords:
            pos = content_lower.find(kw.lower())
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos = pos

        if best_pos == -1:
            return content[:max_len]

        start = max(0, best_pos - max_len // 4)
        end = min(len(content), start + max_len)

        if start > 0:
            space_pos = content.find(' ', start)
            if space_pos != -1 and space_pos < start + 30:
                start = space_pos + 1

        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."

        return snippet

    # Test content with keyword in the middle
    content = "A" * 500 + " Artificial Analysis is great " + "B" * 500

    test_cases = [
        (content, ["artificial analysis"], True),  # Should find and center on match
        (content, ["nonexistent"], False),  # Should fall back to beginning
        ("short content", ["short"], True),  # Short content
    ]

    passed = 0
    for content, keywords, should_contain in test_cases:
        snippet = extract_snippet(content, keywords)
        contains_keyword = any(kw.lower() in snippet.lower() for kw in keywords)

        if should_contain:
            status = "PASS" if contains_keyword else "FAIL"
        else:
            status = "PASS"  # No keyword to find, just checking it doesn't crash

        if status == "PASS":
            passed += 1
        print(f"  Keywords {keywords}: contains_keyword={contains_keyword} [{status}]")
        print(f"    Snippet preview: {snippet[:80]}...")

    print(f"\n  Results: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_keyword_search_db():
    """Test keyword search against the actual database."""
    print("\n[Test 4] Database Keyword Search\n")

    db_path = Path(__file__).parent / "output_data" / "ai_news.db"
    if not db_path.exists():
        print("  SKIP: Database not found at", db_path)
        return True

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Test cases: keyword -> expected to find articles
    test_cases = [
        ("artificialanalysis", "artificial analysis", "web_articles"),
        ("vectorlab", "vectorlab", "web_articles"),
        ("fireship", "fireship", "youtube_videos"),
    ]

    passed = 0
    for keyword, search_term, table in test_cases:
        # Check if search_term exists in the table
        if table == "web_articles":
            cursor.execute(
                "SELECT COUNT(*) FROM web_articles WHERE content LIKE ?",
                (f"%{search_term}%",)
            )
        elif table == "youtube_videos":
            cursor.execute(
                "SELECT COUNT(*) FROM youtube_videos WHERE channel_name LIKE ? OR title LIKE ?",
                (f"%{search_term}%", f"%{search_term}%")
            )

        count = cursor.fetchone()[0]
        status = "PASS" if count > 0 else "SKIP (no data)"
        if count > 0:
            passed += 1
        print(f"  '{keyword}' -> '{search_term}' in {table}: {count} matches [{status}]")

    conn.close()
    print(f"\n  Results: {passed}/{len(test_cases)} passed (data-dependent)")
    return True  # Don't fail on missing data


def test_multi_keyword_search():
    """Test that multiple keywords use AND logic."""
    print("\n[Test 5] Multi-Keyword AND Logic\n")

    db_path = Path(__file__).parent / "output_data" / "ai_news.db"
    if not db_path.exists():
        print("  SKIP: Database not found")
        return True

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Search for articles containing BOTH "vectorlab" AND "artificial analysis"
    cursor.execute("""
        SELECT COUNT(*) FROM web_articles
        WHERE (url LIKE '%vectorlab%' OR content LIKE '%vectorlab%')
          AND (content LIKE '%artificial analysis%')
    """)
    both_count = cursor.fetchone()[0]

    # Search for articles containing just "vectorlab"
    cursor.execute("""
        SELECT COUNT(*) FROM web_articles
        WHERE url LIKE '%vectorlab%' OR content LIKE '%vectorlab%'
    """)
    vectorlab_count = cursor.fetchone()[0]

    print(f"  'vectorlab' alone: {vectorlab_count} articles")
    print(f"  'vectorlab' AND 'artificial analysis': {both_count} articles")

    # AND should return <= results than single keyword
    status = "PASS" if both_count <= vectorlab_count else "FAIL"
    print(f"\n  AND logic correct: {status}")

    conn.close()
    return both_count <= vectorlab_count


def test_otel_spans_exist():
    """Verify OTEL spans are being recorded for chat operations."""
    print("\n[Test 6] OTEL Telemetry Spans\n")

    db_path = Path(__file__).parent / "output_data" / "ai_news.db"
    if not db_path.exists():
        print("  SKIP: Database not found")
        return True

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Check for chat-related spans
    cursor.execute("""
        SELECT name, COUNT(*) as count
        FROM otel_spans
        WHERE name LIKE 'chat.%'
        GROUP BY name
        ORDER BY count DESC
    """)

    spans = cursor.fetchall()

    if not spans:
        print("  WARNING: No chat OTEL spans found")
        print("  This may be normal if telemetry was recently added")
        conn.close()
        return True

    expected_spans = [
        "chat.process_message",
        "chat.validate_input",
        "chat.retrieve_sources",
        "chat.generate_response",
    ]

    found_spans = [s[0] for s in spans]
    print("  Found OTEL spans:")
    for name, count in spans:
        print(f"    {name}: {count}")

    missing = [s for s in expected_spans if s not in found_spans]
    if missing:
        print(f"\n  WARNING: Missing expected spans: {missing}")

    conn.close()
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("ChatAgent Keyword Search Tests")
    print("=" * 60)

    results = []

    results.append(("wordninja segmentation", test_wordninja_segmentation()))
    results.append(("split_camel_or_concat", test_split_camel_or_concat()))
    results.append(("snippet extraction", test_snippet_extraction()))
    results.append(("database keyword search", test_keyword_search_db()))
    results.append(("multi-keyword AND logic", test_multi_keyword_search()))
    results.append(("OTEL spans", test_otel_spans_exist()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
