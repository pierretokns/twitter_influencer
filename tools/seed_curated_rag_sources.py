#!/usr/bin/env python3
"""
Seed curated RAG sources for benchmark slices that normal feeds miss.

This is intentionally small and explicit: it adds known finance-AI and
data-curation pages into the same web article/chunk tables used by ChatAgent so
retrieval benchmarks can distinguish model behavior from missing-source gaps.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_news_scraper import AINewsDatabase, WebSourceScraper, encode_texts_hybrid


CURATED_SOURCES: list[dict[str, str]] = [
    {
        "source_id": "openai_balyasny_case",
        "source_name": "OpenAI Customer Story: Balyasny Asset Management",
        "title": "How Balyasny Asset Management built an AI research engine",
        "url": "https://openai.com/index/balyasny-asset-management/",
        "category": "finance_ai",
    },
    {
        "source_id": "balyasny_openai_feature",
        "source_name": "Balyasny Asset Management",
        "title": "Balyasny Applied AI Team Featured in OpenAI Customer Story",
        "url": "https://www.bamfunds.com/news-and-insights/balyasny-openai-feature",
        "category": "finance_ai",
    },
    {
        "source_id": "citadel_ai_reuters",
        "source_name": "Reuters via Investing.com",
        "title": "Citadel debuts new AI tool for equities investors",
        "url": "https://www.investing.com/news/stock-market-news/citadel-debuts-new-ai-tool-for-equities-investors-cto-subramanian-says-4396760",
        "category": "finance_ai",
    },
    {
        "source_id": "citadel_ai_efc",
        "source_name": "eFinancialCareers",
        "title": "Citadel's CTO says AI won't generate lasting alpha for hedge funds",
        "url": "https://www.efinancialcareers.com/news/citadel-ai",
        "category": "finance_ai",
    },
    {
        "source_id": "balyasny_ai_reuters",
        "source_name": "Reuters via Investing.com",
        "title": "Balyasny names AI as a 2026 tail-risk theme",
        "url": "https://www.investing.com/news/economy-news/hedge-fund-managing-partner-dmitry-balyasny-taps-ai-as-largest-tail-risk-for-2026-4397804",
        "category": "finance_ai",
    },
    {
        "source_id": "jpmorgan_ai_research",
        "source_name": "J.P. Morgan AI Research",
        "title": "Artificial Intelligence Research",
        "url": "https://www.jpmorgan.com/US/en/technology/artificial-intelligence",
        "category": "finance_ai",
    },
    {
        "source_id": "jpmorgan_asset_ai",
        "source_name": "J.P. Morgan Asset Management",
        "title": "Artificial Intelligence",
        "url": "https://am.jpmorgan.com/us/en/asset-management/liq/insights/market-themes/artificial-intelligence/",
        "category": "finance_ai",
    },
    {
        "source_id": "mastercard_ai",
        "source_name": "Mastercard Newsroom",
        "title": "Artificial intelligence",
        "url": "https://newsroom.mastercard.com/news/perspectives/featured-topics/artificial-intelligence/",
        "category": "finance_ai",
    },
    {
        "source_id": "visa_newsroom_ai",
        "source_name": "Visa Newsroom",
        "title": "Visa Newsroom",
        "url": "https://usa.visa.com/about-visa/newsroom.html",
        "category": "finance_ai",
    },
    {
        "source_id": "acadian_ai",
        "source_name": "Acadian Asset Management",
        "title": "Our Edge",
        "url": "https://www.acadian-asset.com/our-edge",
        "category": "finance_ai",
    },
    {
        "source_id": "arrowstreet_ai_search",
        "source_name": "Arrowstreet Capital",
        "title": "Arrowstreet Capital AI and machine learning search",
        "url": "https://www.arrowstreetcapital.com/",
        "category": "finance_ai",
    },
    {
        "source_id": "nvidia_nemo_curator_docs",
        "source_name": "NVIDIA NeMo Curator Docs",
        "title": "NeMo Curator",
        "url": "https://docs.nvidia.com/nemo/curator/latest/home/welcome",
        "category": "data_curation",
    },
    {
        "source_id": "nvidia_nemo_curator_blog",
        "source_name": "NVIDIA Technical Blog",
        "title": "Curating Custom Datasets for LLM Parameter-Efficient Fine-Tuning with NVIDIA NeMo Curator",
        "url": "https://developer.nvidia.com/blog/curating-custom-datasets-for-llm-parameter-efficient-fine-tuning-with-nvidia-nemo-curator/",
        "category": "data_curation",
    },
    {
        "source_id": "arize_phoenix_rag_eval",
        "source_name": "Arize Phoenix Docs",
        "title": "Evaluate RAG",
        "url": "https://arize.com/docs/phoenix/cookbook/evaluation/evaluate-rag",
        "category": "rag_eval",
    },
    {
        "source_id": "llama_factory_data_prep",
        "source_name": "LLaMA Factory Docs",
        "title": "Data Preparation",
        "url": "https://llamafactory.readthedocs.io/en/latest/getting_started/data_preparation.html",
        "category": "fine_tuning",
    },
]


def clean_html_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if len(line) >= 20]
    return "\n".join(lines)[:12000]


def fetch_text(url: str) -> str:
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ai-news-rag-bench/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    urls = [url, f"https://r.jina.ai/{url}"]
    last_error: Exception | None = None
    for fetch_url in urls:
        try:
            response = requests.get(fetch_url, timeout=30, headers=headers)
            response.raise_for_status()
            text = response.text
            if fetch_url.startswith("https://r.jina.ai/"):
                return text[:12000]
            return clean_html_text(text)
        except Exception as exc:
            last_error = exc
    raise last_error or RuntimeError(f"Could not fetch {url}")


def save_seed(db: AINewsDatabase, chunker: WebSourceScraper, source: dict[str, str]) -> tuple[bool, int]:
    content = fetch_text(source["url"])
    article = {
        **source,
        "description": content[:500],
        "content": content,
        "published_at": None,
    }

    dense = sparse = None
    try:
        dense_all, sparse_all = encode_texts_hybrid([f"{source['title']}\n{content[:3000]}"])
        dense = dense_all[0]
        sparse = sparse_all[0]
    except Exception as exc:
        print(f"[WARN] article embedding skipped for {source['source_id']}: {exc}")

    article_id = hashlib.md5(source["url"].encode()).hexdigest()[:16]
    is_ai = db.save_web_article(article, dense_embedding=dense, sparse_embedding=sparse)
    exists = db.conn.execute(
        "SELECT is_ai_relevant FROM web_articles WHERE article_id = ?",
        (article_id,),
    ).fetchone()
    if not exists:
        return False, 0

    chunks = chunker._chunk_content(content)
    saved_chunks = chunker._save_article_chunks(article_id, chunks) if chunks else 0
    return bool(exists["is_ai_relevant"]), saved_chunks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="output_data/ai_news.db")
    parser.add_argument("--only", nargs="*", help="Optional source_id allow-list")
    args = parser.parse_args()

    db = AINewsDatabase(Path(args.db_path))
    chunker = WebSourceScraper(db)
    wanted = set(args.only or [])
    sources = [s for s in CURATED_SOURCES if not wanted or s["source_id"] in wanted]

    seeded = 0
    chunks = 0
    for source in sources:
        print(f"SEED {source['source_id']} {source['url']}", flush=True)
        try:
            is_ai, saved_chunks = save_seed(db, chunker, source)
            seeded += int(is_ai)
            chunks += saved_chunks
            print(f"  ok ai_relevant={is_ai} chunks={saved_chunks}", flush=True)
        except Exception as exc:
            print(f"  failed: {exc}", flush=True)

    print({"sources": len(sources), "ai_relevant": seeded, "chunks": chunks})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
