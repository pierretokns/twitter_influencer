# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "FlagEmbedding>=1.2.0",
#     "sqlite-vec>=0.1.0",
#     "numpy>=1.24.0",
#     "wordninja>=2.0.0",
# ]
# ///
"""
Hybrid Search Retrieval Benchmark
==================================

Evaluates retrieval quality for the RAG chatbot by running queries
against the local ai_news.db and measuring:

1. Recall@K - How many expected sources appear in top-K results
2. Recency ranking - Are recent items ranked higher than old ones
3. Source diversity - Mix of tweets, articles, YouTube
4. Latency - Time per query

Each benchmark case has:
- query: The user question
- expected_ids: Source IDs that MUST appear in results (hard recall)
- expected_keywords: Keywords that should appear in result text (soft recall)
- expected_types: Source types expected (twitter, web, youtube)
- recency_expected: Whether recent content should dominate

Run with: uv run python benchmark_retrieval.py
"""

import json
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

DB_PATH = Path(__file__).parent / "output_data" / "ai_news.db"


@dataclass
class BenchmarkCase:
    name: str
    query: str
    expected_ids: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    expected_types: list[str] = field(default_factory=list)
    max_age_days: Optional[int] = None  # Results should be within this many days of newest content


@dataclass
class BenchmarkResult:
    name: str
    query: str
    recall_ids: float  # Fraction of expected_ids found
    recall_keywords: float  # Fraction of expected_keywords found in result text
    type_coverage: float  # Fraction of expected_types present
    recency_score: float  # 1.0 if results are within max_age_days, else fraction
    latency_ms: float
    num_results: int
    result_types: dict  # Count per type
    top_sources: list[str]  # Brief descriptions of top results
    missing_ids: list[str]
    missing_keywords: list[str]


# Benchmark cases based on actual data in the DB (as of Dec 2025)
BENCHMARK_CASES = [
    # --- Recency queries (chatbot's main use case) ---
    BenchmarkCase(
        name="latest_ai_news",
        query="What's the latest AI news?",
        expected_keywords=["GPT-5.2", "Gemini 3", "OpenAI", "Google"],
        expected_types=["twitter", "web", "youtube"],
        max_age_days=7,
    ),
    BenchmarkCase(
        name="today_ai_news",
        query="What happened in AI today?",
        expected_keywords=["GPT-5.2", "Gemini"],
        expected_types=["twitter"],
        max_age_days=3,
    ),

    # --- Specific model queries ---
    BenchmarkCase(
        name="gpt52_news",
        query="Tell me about GPT-5.2",
        expected_ids=["1999182104362668275", "1999182098859700363"],  # OpenAI's tweets
        expected_keywords=["GPT-5.2", "OpenAI"],
        expected_types=["twitter"],
    ),
    BenchmarkCase(
        name="gemini3_news",
        query="What's new with Gemini 3?",
        expected_ids=["KzB0ywf4V-E"],  # YouTube: What's new in Gemini 3?
        expected_keywords=["Gemini 3", "Google"],
        expected_types=["twitter", "youtube"],
    ),
    BenchmarkCase(
        name="claude_news",
        query="Tell me about Claude and Anthropic updates",
        expected_keywords=["Claude", "Anthropic"],
        expected_types=["twitter", "web"],
    ),

    # --- Topic queries ---
    BenchmarkCase(
        name="ai_safety",
        query="What are experts saying about AI safety and risks?",
        expected_keywords=["safety", "risk"],
        expected_types=["twitter"],
    ),
    BenchmarkCase(
        name="ai_agents",
        query="What's happening with AI agents?",
        expected_keywords=["agent"],
        expected_types=["twitter", "web"],
    ),

    # --- Person queries ---
    BenchmarkCase(
        name="karpathy_posts",
        query="What has Karpathy been saying?",
        expected_ids=["1998806260783919434", "1998803709468487877"],  # karpathy tweets
        expected_keywords=["karpathy"],
        expected_types=["twitter"],
    ),
    BenchmarkCase(
        name="andrew_ng",
        query="Andrew Ng's latest thoughts on AI",
        expected_ids=["1999174188259770795"],  # andrewyng tweet
        expected_keywords=["andrewyng"],  # username in source.author
        expected_types=["twitter"],
    ),

    # --- Brand/keyword queries (tests keyword search) ---
    BenchmarkCase(
        name="mcp_protocol",
        query="Model Context Protocol MCP tools",
        expected_ids=["75db09cb9e2ff3e4", "894d674e93b1a9cb"],  # MCP articles
        expected_keywords=["MCP", "Model Context Protocol"],
        expected_types=["web"],
    ),

    # --- Broad topic queries ---
    BenchmarkCase(
        name="open_source_ai",
        query="Open source AI models and releases",
        expected_keywords=["open source", "model"],
    ),
    BenchmarkCase(
        name="llm_benchmarks",
        query="LLM benchmark results and comparisons",
        expected_keywords=["benchmark", "performance"],
    ),
]


class RetrievalOnlyAgent:
    """Lightweight wrapper that only initializes retrieval (no LLM needed)."""

    def __init__(self, db_path: str, alpha: float = 0.5):
        from agents.chat_agent import ChatAgent, ChatSecurity
        from agents.telemetry import get_tracer

        self.db_path = db_path
        self.alpha = alpha
        self.tracer = get_tracer("chat")
        self.security = ChatSecurity()

        # Bind the retrieval methods from ChatAgent without calling __init__
        self._retrieve_sources = ChatAgent._retrieve_sources.__get__(self, ChatAgent)
        self._search_table = ChatAgent._search_table.__get__(self, ChatAgent)
        self._search_chunks = ChatAgent._search_chunks.__get__(self, ChatAgent)
        self._keyword_search = ChatAgent._keyword_search.__get__(self, ChatAgent)
        self._get_connection = ChatAgent._get_connection.__get__(self, ChatAgent)
        self._connection = ChatAgent._connection.__get__(self, ChatAgent)


def run_benchmark(db_path: str = str(DB_PATH), alpha: float = 0.5) -> list[BenchmarkResult]:
    """Run all benchmark cases and return results."""
    from agents.chat_agent import Source

    agent = RetrievalOnlyAgent(db_path=db_path, alpha=alpha)
    results = []

    # Find the most recent timestamp in the DB for recency scoring
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(timestamp) FROM tweets")
    max_ts = cursor.fetchone()[0]
    conn.close()

    from datetime import datetime
    if max_ts:
        db_latest = datetime.fromisoformat(max_ts.replace('Z', '+00:00').replace('+00:00', ''))
    else:
        db_latest = datetime.now()

    print(f"DB latest content: {max_ts}")
    print(f"Testing with alpha={alpha}\n")
    print("=" * 80)

    for case in BENCHMARK_CASES:
        print(f"\n[{case.name}] {case.query}")

        start = time.time()
        sources, warning = agent._retrieve_sources(case.query, {
            "max_sources": 10,
            "alpha": alpha,
            "recency_boost": True,
        })
        latency_ms = (time.time() - start) * 1000

        if warning:
            print(f"  WARNING: {warning}")

        # Recall on expected IDs
        found_ids = {s.id for s in sources}
        matched_ids = [eid for eid in case.expected_ids if eid in found_ids]
        missing_ids = [eid for eid in case.expected_ids if eid not in found_ids]
        recall_ids = len(matched_ids) / len(case.expected_ids) if case.expected_ids else 1.0

        # Recall on expected keywords (check if keyword appears in any result text)
        all_text = " ".join([
            f"{s.text or ''} {s.title or ''} {s.author or ''}"
            for s in sources
        ]).lower()
        matched_kw = [kw for kw in case.expected_keywords if kw.lower() in all_text]
        missing_kw = [kw for kw in case.expected_keywords if kw.lower() not in all_text]
        recall_keywords = len(matched_kw) / len(case.expected_keywords) if case.expected_keywords else 1.0

        # Type coverage
        result_types_set = {s.type for s in sources}
        type_matches = [t for t in case.expected_types if t in result_types_set]
        type_coverage = len(type_matches) / len(case.expected_types) if case.expected_types else 1.0

        # Type counts
        type_counts = {}
        for s in sources:
            type_counts[s.type] = type_counts.get(s.type, 0) + 1

        # Recency score
        recency_score = 1.0
        if case.max_age_days and sources:
            recent_count = 0
            for s in sources:
                if s.published_at:
                    try:
                        ts = s.published_at
                        if isinstance(ts, str):
                            ts = datetime.fromisoformat(ts.replace('Z', '+00:00').replace('+00:00', ''))
                        days_from_latest = (db_latest - ts).days
                        if days_from_latest <= case.max_age_days:
                            recent_count += 1
                    except Exception:
                        pass
            recency_score = recent_count / len(sources) if sources else 0.0

        # Top source descriptions
        top_sources = []
        for s in sources[:5]:
            desc = f"[{s.type}] @{s.author or '?'}: {(s.title or s.text or '')[:60]}"
            if s.published_at:
                desc += f" ({str(s.published_at)[:10]})"
            top_sources.append(desc)

        result = BenchmarkResult(
            name=case.name,
            query=case.query,
            recall_ids=recall_ids,
            recall_keywords=recall_keywords,
            type_coverage=type_coverage,
            recency_score=recency_score,
            latency_ms=latency_ms,
            num_results=len(sources),
            result_types=type_counts,
            top_sources=top_sources,
            missing_ids=missing_ids,
            missing_keywords=missing_kw,
        )
        results.append(result)

        # Print per-case results
        print(f"  Results: {len(sources)} sources, {latency_ms:.0f}ms")
        print(f"  ID Recall: {recall_ids:.0%} | Keyword Recall: {recall_keywords:.0%} | Type Coverage: {type_coverage:.0%} | Recency: {recency_score:.0%}")
        if missing_ids:
            print(f"  Missing IDs: {missing_ids}")
        if missing_kw:
            print(f"  Missing keywords: {missing_kw}")
        print(f"  Types: {type_counts}")
        for desc in top_sources[:3]:
            print(f"    {desc}")

    return results


def print_summary(results: list[BenchmarkResult]):
    """Print aggregate benchmark summary."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    avg_recall_ids = sum(r.recall_ids for r in results) / len(results)
    avg_recall_kw = sum(r.recall_keywords for r in results) / len(results)
    avg_type_cov = sum(r.type_coverage for r in results) / len(results)
    avg_latency = sum(r.latency_ms for r in results) / len(results)

    recency_cases = [r for r in results if r.recency_score < 1.0 or any(
        c.max_age_days for c in BENCHMARK_CASES if c.name == r.name
    )]
    avg_recency = sum(r.recency_score for r in recency_cases) / len(recency_cases) if recency_cases else 1.0

    print(f"\n  Avg ID Recall:      {avg_recall_ids:.0%}")
    print(f"  Avg Keyword Recall: {avg_recall_kw:.0%}")
    print(f"  Avg Type Coverage:  {avg_type_cov:.0%}")
    print(f"  Avg Recency Score:  {avg_recency:.0%}")
    print(f"  Avg Latency:        {avg_latency:.0f}ms")
    print(f"  Total queries:      {len(results)}")

    # Per-case table
    print(f"\n  {'Case':<25} {'IDs':>6} {'KW':>6} {'Type':>6} {'Rec':>6} {'ms':>6} {'#':>3}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*3}")
    for r in results:
        print(f"  {r.name:<25} {r.recall_ids:>5.0%} {r.recall_keywords:>5.0%} {r.type_coverage:>5.0%} {r.recency_score:>5.0%} {r.latency_ms:>5.0f} {r.num_results:>3}")

    # Identify worst cases
    print("\n  Worst performing queries:")
    scored = [(r, (r.recall_ids + r.recall_keywords + r.type_coverage) / 3) for r in results]
    scored.sort(key=lambda x: x[1])
    for r, score in scored[:3]:
        issues = []
        if r.missing_ids:
            issues.append(f"missing IDs: {r.missing_ids[:2]}")
        if r.missing_keywords:
            issues.append(f"missing KW: {r.missing_keywords}")
        if r.recency_score < 0.5:
            issues.append(f"poor recency ({r.recency_score:.0%})")
        print(f"    {r.name} (score={score:.0%}): {'; '.join(issues) or 'low coverage'}")

    # Overall grade
    overall = (avg_recall_ids + avg_recall_kw + avg_type_cov + avg_recency) / 4
    if overall >= 0.8:
        grade = "A"
    elif overall >= 0.6:
        grade = "B"
    elif overall >= 0.4:
        grade = "C"
    else:
        grade = "D"
    print(f"\n  Overall Grade: {grade} ({overall:.0%})")


def sweep_alpha(db_path: str = str(DB_PATH)):
    """Test different alpha values to find optimal dense/sparse balance."""
    alphas = [0.3, 0.5, 0.7, 0.9]
    print("\n" + "=" * 80)
    print("ALPHA SWEEP (dense/sparse balance)")
    print("=" * 80)

    # Use a subset of cases for speed
    test_cases = [c for c in BENCHMARK_CASES if c.name in {
        "gpt52_news", "gemini3_news", "karpathy_posts", "mcp_protocol", "latest_ai_news"
    }]

    best_alpha = 0.5
    best_score = 0.0

    for alpha in alphas:
        agent = RetrievalOnlyAgent(db_path=db_path, alpha=alpha)
        total_recall = 0
        total_kw = 0
        count = 0

        for case in test_cases:
            sources, _ = agent._retrieve_sources(case.query, {
                "max_sources": 10, "alpha": alpha, "recency_boost": True
            })
            found_ids = {s.id for s in sources}
            all_text = " ".join([f"{s.text or ''} {s.title or ''} {s.author or ''}" for s in sources]).lower()

            id_recall = sum(1 for eid in case.expected_ids if eid in found_ids) / len(case.expected_ids) if case.expected_ids else 1.0
            kw_recall = sum(1 for kw in case.expected_keywords if kw.lower() in all_text) / len(case.expected_keywords) if case.expected_keywords else 1.0

            total_recall += id_recall
            total_kw += kw_recall
            count += 1

        avg_recall = total_recall / count
        avg_kw = total_kw / count
        combined = (avg_recall + avg_kw) / 2

        print(f"  alpha={alpha:.1f}: ID Recall={avg_recall:.0%}, KW Recall={avg_kw:.0%}, Combined={combined:.0%}")

        if combined > best_score:
            best_score = combined
            best_alpha = alpha

    print(f"\n  Best alpha: {best_alpha} (score={best_score:.0%})")
    return best_alpha


if __name__ == "__main__":
    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(description="Benchmark hybrid search retrieval")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dense/sparse balance (0=sparse, 1=dense)")
    parser.add_argument("--sweep", action="store_true", help="Sweep alpha values to find optimal")
    parser.add_argument("--json", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    if args.sweep:
        best = sweep_alpha()
        print(f"\nRunning full benchmark with best alpha={best}...")
        results = run_benchmark(alpha=best)
    else:
        results = run_benchmark(alpha=args.alpha)

    print_summary(results)

    if args.json:
        out = []
        for r in results:
            out.append({
                "name": r.name,
                "query": r.query,
                "recall_ids": r.recall_ids,
                "recall_keywords": r.recall_keywords,
                "type_coverage": r.type_coverage,
                "recency_score": r.recency_score,
                "latency_ms": r.latency_ms,
                "num_results": r.num_results,
                "result_types": r.result_types,
                "missing_ids": r.missing_ids,
                "missing_keywords": r.missing_keywords,
            })
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nResults saved to {args.json}")
