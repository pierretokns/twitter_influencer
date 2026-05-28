#!/usr/bin/env python3
"""
Build a production-shaped seed gold dataset from the ai_news SQLite database.

The goal is not to synthesize a huge benchmark. It is to create traceable seed
cases tied to real rows so local-model evals stop depending only on toy prompts.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FINANCE_TERMS = [
    "j.p. morgan",
    "jp morgan",
    "balyasny",
    "arrowstreet",
    "acadian",
    "citadel",
    "mastercard",
    "visa",
    "hedge fund",
    "asset manager",
    "payments",
]

FINANCE_PAYMENTS_TERMS = [
    "mastercard",
    "visa",
    "payments",
    "fraud",
    "risk",
    "compliance",
    "regulated",
    "financial services",
]

FINANCE_HEDGE_FUND_TERMS = [
    "j.p. morgan",
    "jp morgan",
    "jpmorgan",
    "balyasny",
    "citadel",
    "hedge fund",
    "asset manager",
    "investment research",
    "alpha",
]

CURATION_TERMS = [
    "phoenix",
    "arize",
    "nemo curator",
    "data curation",
    "fine-tuning",
    "jsonl",
    "tracing",
    "eval",
]

PROMPT_OPTIMIZATION_TERMS = [
    "dspy",
    "gepa",
    "prompt",
    "few-shot",
    "eval",
    "phoenix",
    "tracing",
]

LOCAL_MODEL_TERMS = [
    "gguf",
    "llama.cpp",
    "qwen",
    "gemma",
    "liquid",
    "lfm",
    "phi",
    "nvidia",
    "local",
    "cpu",
    "quant",
]

STRUCTURED_OUTPUT_TERMS = [
    "json",
    "schema",
    "structured output",
    "function calling",
    "constrained",
    "xml",
    "validator",
]


@dataclass(frozen=True)
class Source:
    source_uid: str
    source_type: str
    db_table: str
    db_id: str
    title: str
    author: str
    url: str
    published_at: str
    text: str
    matched_terms: list[str]


def normalize_space(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def has_term(text: str, term: str) -> bool:
    lower = text.lower()
    if term == "jp morgan":
        return "jp morgan" in lower or "jpmorgan" in lower
    if re.search(r"[a-z0-9]", term):
        return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", lower) is not None
    return term in lower


def matched_terms(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if has_term(text, term)]


def excerpt(text: str, terms: list[str], max_chars: int = 1400) -> str:
    clean = normalize_space(text)
    if len(clean) <= max_chars:
        return clean
    lower = clean.lower()
    positions = [lower.find(term) for term in terms if lower.find(term) >= 0]
    start = max(0, min(positions) - 240) if positions else 0
    end = min(len(clean), start + max_chars)
    snippet = clean[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(clean):
        snippet += "..."
    return snippet


def article_sources(conn: sqlite3.Connection, terms: list[str], limit: int = 24) -> list[Source]:
    rows = conn.execute(
        """
        SELECT article_id, source_name, title, url, description, content, author, author_name,
               published_at, scraped_at
        FROM web_articles
        WHERE coalesce(content, '') != ''
        ORDER BY coalesce(published_at, scraped_at, '') DESC
        """
    ).fetchall()
    sources: list[Source] = []
    for row in rows:
        body = " ".join(
            normalize_space(row[key])
            for key in ("source_name", "title", "description", "content", "author", "author_name")
            if row[key]
        )
        hits = matched_terms(body, terms)
        if not hits:
            continue
        content = excerpt(row["content"], hits)
        if "scheduled maintenance" in content.lower() and len(hits) <= 1:
            continue
        sources.append(
            Source(
                source_uid=f"web:{row['article_id']}",
                source_type="web_article",
                db_table="web_articles",
                db_id=row["article_id"],
                title=normalize_space(row["title"] or row["source_name"]),
                author=normalize_space(row["author"] or row["author_name"] or row["source_name"]),
                url=normalize_space(row["url"]),
                published_at=normalize_space(row["published_at"] or row["scraped_at"]),
                text=content,
                matched_terms=hits,
            )
        )
    sources.sort(key=lambda item: (len(item.matched_terms), len(item.text)), reverse=True)
    return sources[:limit]


def paragraph_sources(conn: sqlite3.Connection, terms: list[str], limit: int = 24) -> list[Source]:
    rows = conn.execute(
        """
        SELECT p.id, p.article_id, p.paragraph_index, p.text, a.source_name, a.title, a.url,
               a.author, a.author_name, a.published_at, a.scraped_at
        FROM article_paragraphs p
        JOIN web_articles a ON a.article_id = p.article_id
        ORDER BY coalesce(a.published_at, a.scraped_at, '') DESC, p.paragraph_index ASC
        """
    ).fetchall()
    sources: list[Source] = []
    for row in rows:
        body = " ".join(
            normalize_space(row[key])
            for key in ("source_name", "title", "text", "author", "author_name")
            if row[key]
        )
        hits = matched_terms(body, terms)
        if not hits:
            continue
        sources.append(
            Source(
                source_uid=f"paragraph:{row['id']}",
                source_type="article_paragraph",
                db_table="article_paragraphs",
                db_id=str(row["id"]),
                title=normalize_space(row["title"] or row["source_name"]),
                author=normalize_space(row["author"] or row["author_name"] or row["source_name"]),
                url=normalize_space(row["url"]),
                published_at=normalize_space(row["published_at"] or row["scraped_at"]),
                text=excerpt(row["text"], hits, max_chars=900),
                matched_terms=hits,
            )
        )
    sources.sort(key=lambda item: (len(item.matched_terms), len(item.text)), reverse=True)
    return sources[:limit]


def tweet_sources(conn: sqlite3.Connection, terms: list[str], limit: int = 20) -> list[Source]:
    rows = conn.execute(
        """
        SELECT tweet_id, username, display_name, text, timestamp, url, scraped_at
        FROM tweets
        WHERE coalesce(text, '') != ''
        ORDER BY coalesce(timestamp, scraped_at, '') DESC
        """
    ).fetchall()
    sources: list[Source] = []
    for row in rows:
        body = " ".join(normalize_space(row[key]) for key in ("username", "display_name", "text") if row[key])
        hits = matched_terms(body, terms)
        if not hits:
            continue
        sources.append(
            Source(
                source_uid=f"tweet:{row['tweet_id']}",
                source_type="tweet",
                db_table="tweets",
                db_id=row["tweet_id"],
                title=f"@{row['username']}",
                author=normalize_space(row["display_name"] or row["username"]),
                url=normalize_space(row["url"]),
                published_at=normalize_space(row["timestamp"] or row["scraped_at"]),
                text=excerpt(row["text"], hits, max_chars=900),
                matched_terms=hits,
            )
        )
    sources.sort(key=lambda item: (len(item.matched_terms), item.published_at), reverse=True)
    return sources[:limit]


def source_json(source: Source) -> dict[str, Any]:
    return {
        "source_uid": source.source_uid,
        "source_type": source.source_type,
        "db_table": source.db_table,
        "db_id": source.db_id,
        "title": source.title,
        "author": source.author,
        "url": source.url,
        "published_at": source.published_at,
        "text": source.text,
        "matched_terms": source.matched_terms,
    }


def first_by_term(sources: list[Source], terms: list[str], max_sources: int) -> list[Source]:
    selected: list[Source] = []
    seen: set[str] = set()
    for term in terms:
        for source in sources:
            if source.source_uid in seen:
                continue
            if term in source.matched_terms:
                selected.append(source)
                seen.add(source.source_uid)
                break
    for source in sources:
        if len(selected) >= max_sources:
            break
        if source.source_uid not in seen:
            selected.append(source)
            seen.add(source.source_uid)
    return selected[:max_sources]


def make_retrieval_case(
    case_id: str,
    slice_name: str,
    query: str,
    sources: list[Source],
    required_terms: list[str],
    notes: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "case_type": "retrieval",
        "slice": slice_name,
        "query": query,
        "required_terms": required_terms,
        "gold_source_uids": [source.source_uid for source in sources],
        "gold_sources": [source_json(source) for source in sources],
        "metrics": ["recall_at_10", "mrr", "required_term_coverage"],
        "notes": notes,
    }


def make_rag_case(
    case_id: str,
    slice_name: str,
    query: str,
    sources: list[Source],
    expected_terms: list[str],
    unsupported: bool,
    notes: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "case_type": "rag_generation",
        "slice": slice_name,
        "query": query,
        "provided_sources": [source_json(source) for source in sources],
        "expected_terms": expected_terms,
        "unsupported": unsupported,
        "required_behavior": (
            "refuse or state that sources do not support the claim"
            if unsupported
            else "answer only from provided sources with inline [N] citations"
        ),
        "metrics": ["answer_term_coverage", "citation_validity", "citation_support", "unsupported_refusal"],
        "notes": notes,
    }


def make_citation_case(
    case_id: str,
    slice_name: str,
    claim: str,
    source: Source,
    label: str,
    expected_reason_terms: list[str],
    notes: str,
    support_type: str | None = None,
    claim_entities: list[str] | None = None,
) -> dict[str, Any]:
    source_data = source_json(source)
    return {
        "case_id": case_id,
        "case_type": "citation_pair",
        "slice": slice_name,
        "claim": claim,
        "cited_source": source_data,
        "label": label,
        "expected_status": "valid" if label == "supported" else "invalid",
        "support_type": support_type or label,
        "claim_entities": claim_entities or [],
        "source_entities": source.matched_terms,
        "expected_reason_terms": expected_reason_terms,
        "supporting_quote": excerpt(source.text, source.matched_terms, max_chars=420),
        "notes": notes,
        "human_review_required": True,
    }


def first_matching_source(sources: list[Source], terms: list[str]) -> Source | None:
    for term in terms:
        for source in sources:
            if term in source.matched_terms:
                return source
    return sources[0] if sources else None


def sources_covering_terms(sources: list[Source], terms: list[str], max_sources: int) -> list[Source]:
    """Order sources so compact RAG prompts preserve expected evidence diversity."""
    selected: list[Source] = []
    seen: set[str] = set()
    covered: set[str] = set()
    for term in terms:
        if term in covered:
            continue
        best: Source | None = None
        best_new_hits: set[str] = set()
        for source in sources:
            if source.source_uid in seen:
                continue
            hits = {hit for hit in source.matched_terms if hit in terms}
            new_hits = hits - covered
            if term in hits and (best is None or len(new_hits) > len(best_new_hits)):
                best = source
                best_new_hits = new_hits
        if best is not None:
            selected.append(best)
            seen.add(best.source_uid)
            covered.update(best_new_hits)
            if len(selected) >= max_sources:
                return selected
    for source in sources:
        if len(selected) >= max_sources:
            break
        if source.source_uid not in seen:
            selected.append(source)
            seen.add(source.source_uid)
    return selected


def build_cases(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    finance_sources = first_by_term(
        article_sources(conn, FINANCE_TERMS, limit=40) + tweet_sources(conn, FINANCE_TERMS, limit=12),
        FINANCE_TERMS,
        max_sources=10,
    )
    finance_payment_sources = first_by_term(
        article_sources(conn, FINANCE_PAYMENTS_TERMS, limit=30)
        + paragraph_sources(conn, FINANCE_PAYMENTS_TERMS, limit=24)
        + tweet_sources(conn, FINANCE_PAYMENTS_TERMS, limit=10),
        FINANCE_PAYMENTS_TERMS,
        max_sources=8,
    )
    finance_hedge_sources = first_by_term(
        article_sources(conn, FINANCE_HEDGE_FUND_TERMS, limit=30)
        + paragraph_sources(conn, FINANCE_HEDGE_FUND_TERMS, limit=24)
        + tweet_sources(conn, FINANCE_HEDGE_FUND_TERMS, limit=10),
        FINANCE_HEDGE_FUND_TERMS,
        max_sources=8,
    )
    curation_sources = first_by_term(
        paragraph_sources(conn, CURATION_TERMS, limit=40) + article_sources(conn, CURATION_TERMS, limit=20),
        CURATION_TERMS,
        max_sources=10,
    )
    prompt_optimization_sources = first_by_term(
        article_sources(conn, PROMPT_OPTIMIZATION_TERMS, limit=25)
        + paragraph_sources(conn, PROMPT_OPTIMIZATION_TERMS, limit=30),
        PROMPT_OPTIMIZATION_TERMS,
        max_sources=8,
    )
    local_model_sources = first_by_term(
        article_sources(conn, LOCAL_MODEL_TERMS, limit=35)
        + paragraph_sources(conn, LOCAL_MODEL_TERMS, limit=20)
        + tweet_sources(conn, LOCAL_MODEL_TERMS, limit=15),
        LOCAL_MODEL_TERMS,
        max_sources=10,
    )
    structured_output_sources = first_by_term(
        article_sources(conn, STRUCTURED_OUTPUT_TERMS, limit=25)
        + paragraph_sources(conn, STRUCTURED_OUTPUT_TERMS, limit=30)
        + tweet_sources(conn, STRUCTURED_OUTPUT_TERMS, limit=10),
        STRUCTURED_OUTPUT_TERMS,
        max_sources=8,
    )
    mixed_unsupported_sources = first_by_term(
        article_sources(conn, ["anthropic", "claude", "agent", "sdk"], limit=12)
        + paragraph_sources(conn, ["anthropic", "claude", "agent", "sdk"], limit=12),
        ["anthropic", "claude", "agent", "sdk"],
        max_sources=5,
    )
    local_deployment_terms = ["llama.cpp", "gguf", "quant", "cpu", "qwen", "gemma", "lfm", "nvidia"]

    cases: list[dict[str, Any]] = [
        make_retrieval_case(
            "retrieval_finance_firms_v1",
            "finance_domain_signal",
            (
                "Find AI news relevant to Brandon's finance role mentioning banks, asset managers, "
                "hedge funds, payments, J.P. Morgan, Balyasny, Arrowstreet, Acadian, Citadel, Mastercard, or Visa."
            ),
            finance_sources,
            FINANCE_TERMS,
            "Gold sources are real VM DB rows with finance company/domain terms.",
        ),
        make_retrieval_case(
            "retrieval_curation_eval_v1",
            "data_curation_eval",
            (
                "Find sources about Phoenix, Arize, NVIDIA NeMo Curator, production traces, evals, "
                "JSONL conversion, and fine-tuning dataset curation."
            ),
            curation_sources,
            CURATION_TERMS,
            "Gold sources emphasize eval/tracing and data-curation workflows before fine-tuning.",
        ),
        make_retrieval_case(
            "retrieval_curation_prompt_traces_v1",
            "data_curation_eval",
            (
                "Find sources about using Phoenix, Arize, DSPy, GEPA, prompt optimization, few-shot examples, "
                "and traces to improve local-model evals before fine-tuning."
            ),
            first_by_term(prompt_optimization_sources + curation_sources, PROMPT_OPTIMIZATION_TERMS, max_sources=8),
            PROMPT_OPTIMIZATION_TERMS,
            "Gold sources emphasize prompt optimization and trace-based evals as pre-fine-tuning curation work.",
        ),
        make_retrieval_case(
            "retrieval_local_models_v1",
            "local_model_ops",
            "Find sources about local open models, GGUF, llama.cpp, Qwen, Gemma, LiquidAI, Phi, NVIDIA, CPU, and quants.",
            local_model_sources,
            LOCAL_MODEL_TERMS,
            "Gold sources support local-model selection and CPU deployment decisions.",
        ),
        make_retrieval_case(
            "retrieval_finance_payments_risk_v1",
            "finance_domain_signal",
            (
                "Find AI news for regulated finance and payments: Mastercard, Visa, payment networks, fraud, "
                "risk, compliance, and financial services workflows."
            ),
            finance_payment_sources,
            FINANCE_PAYMENTS_TERMS,
            "Gold sources emphasize regulated payments and risk/compliance finance signals.",
        ),
        make_retrieval_case(
            "retrieval_finance_hedge_fund_research_v1",
            "finance_domain_signal",
            (
                "Find sources about AI in hedge funds, asset managers, investment research, alpha, "
                "J.P. Morgan, Balyasny, Citadel, and similar finance firms."
            ),
            finance_hedge_sources,
            FINANCE_HEDGE_FUND_TERMS,
            "Gold sources emphasize hedge-fund and asset-management AI use cases.",
        ),
        make_retrieval_case(
            "retrieval_structured_output_v1",
            "structured_control_plane",
            (
                "Find sources about JSON schemas, structured output, function calling, constrained decoding, "
                "XML alternatives, and validators for local models."
            ),
            structured_output_sources,
            STRUCTURED_OUTPUT_TERMS,
            "Gold sources support the structured-output/control-plane slice before fine-tuning.",
        ),
        make_retrieval_case(
            "retrieval_inline_citation_support_v1",
            "rag_webchat_inline_citations",
            (
                "Find source passages for a citation-heavy answer about finance AI signals, local models, "
                "retrieval, reranking, evals, and source-grounded citation support."
            ),
            sources_covering_terms(
                finance_sources + finance_payment_sources + curation_sources + local_model_sources,
                ["finance", "ai", "model", "retrieval", "rerank", "citation", "eval"],
                max_sources=8,
            ),
            ["finance", "ai", "model", "retrieval", "rerank", "citation", "eval"],
            "Gold sources exercise retrieval for inline-citation webchat answers before generation.",
        ),
        make_rag_case(
            "rag_finance_brief_v1",
            "finance_domain_signal",
            (
                "Write Brandon terse finance-facing AI news bulletins from these sources. "
                "Prioritize genuinely useful items he may have missed on x.com, and avoid narrative filler."
            ),
            finance_sources[:8],
            ["j.p. morgan", "balyasny", "arrowstreet", "acadian", "mastercard", "visa"],
            unsupported=False,
            notes="Tests sourced finance synthesis, named entities, and citation behavior.",
        ),
        make_rag_case(
            "rag_curation_before_finetune_v1",
            "data_curation_eval",
            "What can we do to improve evals and datasets before fine-tuning tiny local models?",
            curation_sources[:8],
            ["phoenix", "nemo curator", "jsonl", "fine-tuning", "eval"],
            unsupported=False,
            notes="Tests whether the model recommends curation/eval improvements before training.",
        ),
        make_rag_case(
            "rag_curation_prompt_traces_v1",
            "pre_finetune_system_improvements",
            "What prompt/eval improvements should we try before fine-tuning tiny local models?",
            (prompt_optimization_sources[:5] + curation_sources[:3])[:8],
            ["dspy", "gepa", "few-shot", "phoenix", "tracing", "eval"],
            unsupported=False,
            notes="Tests whether prompt optimization and tracing concepts are grounded before fine-tuning.",
        ),
        make_rag_case(
            "rag_finance_payments_risk_v1",
            "finance_domain_signal",
            (
                "Write Brandon terse bulletins on AI signals from regulated payments and financial services. "
                "Highlight Mastercard, Visa, risk, fraud, or compliance only when the sources support them. "
                "Avoid broad narrative claims."
            ),
            finance_payment_sources[:7],
            ["mastercard", "visa", "payments", "fraud", "risk", "compliance"],
            unsupported=False,
            notes="Tests finance-domain synthesis for regulated payments and compliance/risk signals.",
        ),
        make_rag_case(
            "rag_local_model_deployment_tradeoffs_v1",
            "local_model_ops",
            (
                "Summarize the local model deployment tradeoffs for this Hetzner CPU workflow. "
                "Mention llama.cpp, GGUF, quantization, CPU, and model-family choices only when supported."
            ),
            sources_covering_terms(local_model_sources, local_deployment_terms, max_sources=8),
            local_deployment_terms,
            unsupported=False,
            notes="Tests local-model operational synthesis for the migration away from hosted Claude.",
        ),
        make_rag_case(
            "rag_inline_citation_finance_model_brief_v1",
            "rag_webchat_inline_citations",
            (
                "Write concise Brandon-style bullets on the AI-and-finance items in these sources. "
                "Every factual sentence must have an inline numeric citation, and unsupported finance/model "
                "claims must be omitted."
            ),
            sources_covering_terms(
                finance_sources + finance_payment_sources + curation_sources + local_model_sources,
                ["finance", "ai", "model", "retrieval", "rerank", "citation", "eval"],
                max_sources=8,
            ),
            ["finance", "ai", "model", "retrieval", "citation", "eval"],
            unsupported=False,
            notes="Tests the actual webchat workload: concise sourced bullets with sentence-level inline citations.",
        ),
        make_rag_case(
            "rag_prompt_optimization_before_finetune_v1",
            "pre_finetune_system_improvements",
            (
                "Before fine-tuning, what prompt/eval improvements should we try for local models? "
                "Cover DSPy, GEPA, few-shot examples, Phoenix traces, or evals only if supported."
            ),
            prompt_optimization_sources[:7] or curation_sources[:7],
            ["dspy", "gepa", "few-shot", "phoenix", "tracing", "eval"],
            unsupported=False,
            notes="Tests whether prompt optimization and tracing concepts are grounded before fine-tuning.",
        ),
        make_rag_case(
            "rag_unsupported_anthropic_terms_v1",
            "source_grounded_refusal",
            "Which exact Anthropic subscription clause changed on May 25, 2026, and what price tier was affected?",
            mixed_unsupported_sources,
            ["not support", "do not contain", "insufficient", "cannot determine"],
            unsupported=True,
            notes="Hard negative: sources may mention Anthropic/Claude but not the requested exact subscription change.",
        ),
        make_rag_case(
            "rag_unsupported_finance_roi_v1",
            "source_grounded_refusal",
            (
                "Which Citadel desk deployed Qwen3.6 in production, what was the ROI, and which executive approved it?"
            ),
            (finance_hedge_sources[:3] + local_model_sources[:3])[:6],
            ["not support", "do not contain", "insufficient", "cannot determine"],
            unsupported=True,
            notes="Hard negative: mixes finance and local-model sources but asks for an unsupported exact deployment claim.",
        ),
        make_rag_case(
            "rag_unsupported_vendor_contract_v1",
            "source_grounded_refusal",
            (
                "What private NVIDIA Data Flywheel contract terms did Balyasny sign, including price and term length?"
            ),
            (curation_sources[:3] + finance_hedge_sources[:3])[:6],
            ["not support", "do not contain", "insufficient", "cannot determine"],
            unsupported=True,
            notes="Hard negative for vendor-contract hallucination under finance/data-curation context.",
        ),
    ]

    if finance_sources and local_model_sources:
        cases.append(
            make_citation_case(
                "citation_negative_cross_source_v1",
                "rag_webchat_inline_citations",
                "J.P. Morgan is using Gemma GGUF for analyst assistants.",
                local_model_sources[0],
                "unsupported",
                ["j.p. morgan", "gemma", "missing", "source"],
                "Negative pair: entity/company claim is cited to a local-model source.",
                support_type="entity_mismatch",
                claim_entities=["J.P. Morgan", "Gemma", "GGUF"],
            )
        )
    if curation_sources:
        phoenix_source = next(
            (
                source
                for source in curation_sources
                if "phoenix" in source.matched_terms or "arize" in source.matched_terms
            ),
            curation_sources[0],
        )
        cases.append(
            make_citation_case(
                "citation_positive_curation_v1",
                "rag_webchat_inline_citations",
                "Phoenix Tracing can capture data for evaluating a RAG pipeline with LLM evals.",
                phoenix_source,
                "supported",
                ["phoenix", "tracing", "rag", "eval"],
                "Positive pair from a real curation paragraph/article.",
                support_type="direct_support",
                claim_entities=["Phoenix", "RAG", "evals"],
            )
        )
    payment_source = first_matching_source(finance_payment_sources, ["mastercard", "visa", "payments"])
    if payment_source:
        cases.append(
            make_citation_case(
                "citation_positive_payments_ai_v1",
                "rag_webchat_inline_citations",
                "Mastercard or Visa publish AI-related material relevant to payment-network risk and financial services workflows.",
                payment_source,
                "supported",
                ["mastercard", "visa", "ai", "payments", "risk"],
                "Positive pair for regulated payments and finance-domain AI signals.",
                support_type="direct_or_domain_support",
                claim_entities=["Mastercard", "Visa", "AI", "payments"],
            )
        )
    hedge_source = first_matching_source(finance_hedge_sources, ["balyasny", "j.p. morgan", "jp morgan", "jpmorgan"])
    if hedge_source:
        cases.append(
            make_citation_case(
                "citation_positive_finance_ai_research_v1",
                "rag_webchat_inline_citations",
                "Balyasny or J.P. Morgan have public material about AI for investment research or financial services.",
                hedge_source,
                "supported",
                ["balyasny", "j.p. morgan", "ai", "research", "financial"],
                "Positive pair for finance-firm AI research and services material.",
                support_type="direct_or_domain_support",
                claim_entities=["Balyasny", "J.P. Morgan", "AI", "research"],
            )
        )
        cases.append(
            make_citation_case(
                "citation_negative_finance_action_mismatch_v1",
                "rag_webchat_inline_citations",
                "Balyasny deployed Qwen3.6 dense models on Hetzner CPU servers for production trading signals.",
                hedge_source,
                "unsupported",
                ["balyasny", "qwen3.6", "hetzner", "missing", "source"],
                "Negative pair: finance-firm AI material does not support a specific local-model deployment claim.",
                support_type="action_mismatch",
                claim_entities=["Balyasny", "Qwen3.6", "Hetzner"],
            )
        )
    if payment_source and local_model_sources:
        cases.append(
            make_citation_case(
                "citation_negative_payment_model_mismatch_v1",
                "rag_webchat_inline_citations",
                "Mastercard selected LiquidAI LFM2-24B as its production fraud-detection model.",
                local_model_sources[0],
                "unsupported",
                ["mastercard", "liquidai", "lfm2", "missing", "source"],
                "Negative pair: local-model source does not support a Mastercard deployment claim.",
                support_type="entity_and_action_mismatch",
                claim_entities=["Mastercard", "LiquidAI", "LFM2-24B"],
            )
        )
    return cases


def write_openai_messages(cases: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            if case["case_type"] != "rag_generation":
                continue
            sources = case["provided_sources"]
            source_text = "\n\n".join(
                f"[{idx}] {source['title']}\n{source['text']}"
                for idx, source in enumerate(sources, start=1)
            )
            if case["unsupported"]:
                assistant = "The provided sources do not support the requested exact claim. I cannot determine it from these sources."
            else:
                terms = ", ".join(case["expected_terms"][:5])
                assistant = (
                    "Answer from the numbered sources only, cite each factual sentence inline, "
                    f"and cover these expected concepts when supported: {terms}."
                )
            row = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a source-grounded AI news assistant for Brandon. Use only provided sources.",
                    },
                    {
                        "role": "user",
                        "content": f"SOURCES:\n{source_text}\n\nQUESTION: {case['query']}",
                    },
                    {"role": "assistant", "content": assistant},
                ],
                "metadata": {
                    "case_id": case["case_id"],
                    "slice": case["slice"],
                    "source": "production_seed_gold_eval",
                    "requires_human_review_before_training": True,
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="output_data/ai_news.vm.db")
    parser.add_argument("--out-dir", default="output_data/gold_eval")
    parser.add_argument("--name", default="production_seed_v1")
    args = parser.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cases = build_cases(conn)

    jsonl_path = out_dir / f"{args.name}.jsonl"
    summary_path = out_dir / f"{args.name}_summary.json"
    openai_path = out_dir / f"{args.name}_openai_messages.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    write_openai_messages(cases, openai_path)

    by_type: dict[str, int] = {}
    by_slice: dict[str, int] = {}
    source_uids: set[str] = set()
    for case in cases:
        by_type[case["case_type"]] = by_type.get(case["case_type"], 0) + 1
        by_slice[case["slice"]] = by_slice.get(case["slice"], 0) + 1
        for key in ("gold_sources", "provided_sources"):
            for source in case.get(key, []):
                source_uids.add(source["source_uid"])
        if "cited_source" in case:
            source_uids.add(case["cited_source"]["source_uid"])

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(db_path),
        "case_count": len(cases),
        "by_type": by_type,
        "by_slice": by_slice,
        "unique_source_count": len(source_uids),
        "outputs": {
            "gold_cases_jsonl": str(jsonl_path),
            "openai_messages_jsonl": str(openai_path),
            "summary_json": str(summary_path),
        },
        "training_status": "seed data only; human review required before using as SFT targets",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
