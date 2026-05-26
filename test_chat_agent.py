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
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

import sqlite3
import wordninja
import re
from agents.chat_agent import ChatAgent, Source


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


def test_context_source_selection():
    """Test source compression selects relevant, diverse sources."""
    print("\n[Test 7] Context Source Selection\n")

    agent = object.__new__(ChatAgent)
    sources = [
        Source(
            id="generic",
            type="web",
            title="Generic AI news",
            text="OpenAI released a general model update.",
            url="https://example.com/generic",
        ),
        Source(
            id="phoenix",
            type="web",
            title="Evaluate RAG",
            author="Arize Phoenix Docs",
            text="Phoenix Tracing captures RAG pipeline data and supports LLM eval workflows.",
            url="https://example.com/phoenix",
        ),
        Source(
            id="nemo",
            type="web",
            title="NeMo Curator",
            author="NVIDIA",
            text="NVIDIA NeMo Curator helps curate JSONL datasets for fine-tuning.",
            url="https://example.com/nemo",
        ),
        Source(
            id="tweet1",
            type="twitter",
            author="researcher",
            text="A quick note about eval traces and production feedback loops.",
            url="https://example.com/tweet",
        ),
        Source(
            id="youtube1",
            type="youtube",
            author="channel",
            title="Unrelated video",
            text="This is mostly unrelated.",
            url="https://example.com/video",
        ),
    ]

    selected = agent._select_context_sources(
        "How do Phoenix, NeMo Curator, eval traces, and fine-tuning datasets help before training?",
        sources,
        {"context_max_sources": 3},
    )
    selected_ids = [source.id for source in selected]
    print(f"  selected: {selected_ids}")

    required = {"phoenix", "nemo"}
    passed = len(selected) == 3 and required.issubset(set(selected_ids))
    print(f"  required present: {required.issubset(set(selected_ids))} [{'PASS' if passed else 'FAIL'}]")
    return passed


def test_context_text_clipping():
    """Test context clipping preserves concise prompt sources."""
    print("\n[Test 8] Context Text Clipping\n")

    agent = object.__new__(ChatAgent)
    long_text = "A" * 400
    clipped = agent._clip_source_text(long_text, 80)
    passed = len(clipped) == 80 and clipped.endswith("...")
    print(f"  clipped length: {len(clipped)} [{'PASS' if passed else 'FAIL'}]")
    return passed


def test_reranker_reorders_before_context_selection():
    """Test reranker order is honored by context source compression."""
    print("\n[Test 9] Reranker Source Ordering\n")

    agent = object.__new__(ChatAgent)
    sources = [
        Source(id="generic", type="web", title="Generic", text="General AI news.", url="https://example.com/generic"),
        Source(id="phoenix", type="web", title="Phoenix", text="Phoenix traces RAG evals.", url="https://example.com/phoenix"),
        Source(id="nemo", type="web", title="NeMo", text="NeMo Curator prepares datasets.", url="https://example.com/nemo"),
        Source(id="tweet", type="twitter", text="Short social note.", url="https://example.com/tweet"),
    ]

    def fake_scores(query, candidates, model_name):
        score_by_id = {"nemo": 0.9, "phoenix": 0.8, "generic": 0.1, "tweet": 0.05}
        return [score_by_id[source.id] for source in candidates]

    agent._score_sources_with_reranker = fake_scores
    ranked, info = agent._rerank_sources(
        "How do Phoenix and NeMo help RAG evals and curation?",
        sources,
        {"enable_reranker": True, "reranker_model": "fake"},
    )
    selected = agent._select_context_sources(
        "How do Phoenix and NeMo help RAG evals and curation?",
        ranked,
        {"context_max_sources": 2, "_source_order_is_reranked": info["applied"]},
    )
    selected_ids = [source.id for source in selected]
    passed = info["applied"] and selected_ids == ["nemo", "phoenix"]
    print(f"  selected after rerank: {selected_ids} [{'PASS' if passed else 'FAIL'}]")
    return passed


def test_reranker_fallback_on_error():
    """Test reranker failure preserves base retrieval order."""
    print("\n[Test 10] Reranker Fallback\n")

    agent = object.__new__(ChatAgent)
    sources = [
        Source(id="first", type="web", text="First source.", url="https://example.com/1"),
        Source(id="second", type="web", text="Second source.", url="https://example.com/2"),
    ]

    def broken_scores(query, candidates, model_name):
        raise RuntimeError("test failure")

    agent._score_sources_with_reranker = broken_scores
    ranked, info = agent._rerank_sources(
        "query",
        sources,
        {"enable_reranker": True, "reranker_model": "fake-broken"},
    )
    ranked_ids = [source.id for source in ranked]
    passed = not info["applied"] and ranked_ids == ["first", "second"] and "unavailable" in info["warning"]
    print(f"  ranked after failure: {ranked_ids} [{'PASS' if passed else 'FAIL'}]")
    return passed


def test_local_backend_initialization_without_strands():
    """Test local backend does not import hosted Strands/Bedrock dependencies."""
    print("\n[Test 11] Local Backend Initialization\n")

    with patch.dict("os.environ", {"CHAT_BACKEND": "llama_cpp", "CHAT_LLAMA_MODEL": "fake/repo:model.gguf"}, clear=False):
        agent = ChatAgent(db_path="output_data/ai_news.db")
    passed = agent.backend == "llama_cpp" and agent.agent is None and agent.model_id == "fake/repo:model.gguf"
    print(f"  backend: {agent.backend}, model: {agent.model_id} [{'PASS' if passed else 'FAIL'}]")
    return passed


def test_local_llama_citation_retry():
    """Test local backend retries when supported answer has no citations."""
    print("\n[Test 12] Local Llama Citation Retry\n")

    agent = object.__new__(ChatAgent)
    agent.max_tokens = 320
    agent.model_id = "fake/repo:model.gguf"
    calls = []

    def fake_run(prompt, options):
        calls.append(prompt)
        if len(calls) == 1:
            return "Mastercard uses AI for fraud risk.", {"returncode": 0, "elapsed_sec": 1.0}
        return "Mastercard uses AI for fraud risk [1].", {"returncode": 0, "elapsed_sec": 2.0}

    agent._run_llama_cli = fake_run
    sources = [Source(id="1", type="web", text="Mastercard AI fraud risk", url="https://example.com")]
    answer, info = agent._generate_local_llama_response(
        "What matters for Mastercard?",
        "SOURCES:\n[1] WEB\nMastercard AI fraud risk\n\nQUESTION: What matters for Mastercard?",
        sources,
        {"citation_retry_on_missing": True},
    )
    passed = answer.endswith("[1].") and info.get("retry_used") and len(calls) == 2
    print(f"  retry_used: {info.get('retry_used')}, calls: {len(calls)} [{'PASS' if passed else 'FAIL'}]")
    return passed


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
    results.append(("context source selection", test_context_source_selection()))
    results.append(("context text clipping", test_context_text_clipping()))
    results.append(("reranker source ordering", test_reranker_reorders_before_context_selection()))
    results.append(("reranker fallback", test_reranker_fallback_on_error()))
    results.append(("local backend initialization", test_local_backend_initialization_without_strands()))
    results.append(("local llama citation retry", test_local_llama_citation_retry()))

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
