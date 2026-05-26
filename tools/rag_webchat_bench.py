#!/usr/bin/env python3
"""
End-to-end RAG/webchat benchmark using the real ChatAgent retrieval path.

The deployed Python chat agent still generates with hosted Strands/Bedrock.
This benchmark keeps the same retrieval/context construction path and swaps the
generator to local llama.cpp GGUF models so we can test CPU-only webchat
feasibility without changing production behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SYSTEM_PROMPT = """You are a source-grounded AI news webchat assistant for Brandon.
Answer using only the numbered sources. Cite factual claims inline with numeric citations such as [1] or [2].
Never write placeholder citations like [N], [source], or [citation].
If the retrieved sources do not contain the answer, say that the sources do not support the claim.
Keep answers concise and practical. Do not mention Twitter/X influencing or social posting."""


DEFAULT_MODELS = [
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf",
    "unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf",
    "LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf",
]

TERM_ALIASES = {
    "jpmorgan": ("jpmorgan", "jp morgan", "j.p. morgan"),
    "j.p. morgan": ("j.p. morgan", "jp morgan", "jpmorgan"),
    "hedge fund": ("hedge fund", "hedge funds"),
    "asset manager": ("asset manager", "asset managers", "asset management"),
    "payments": ("payments", "payment", "payment network", "payment networks"),
    "data flywheel": ("data flywheel", "data flywheels"),
    "llama-factory": ("llama-factory", "llama factory", "llamafactory"),
    "fine-tuning": ("fine-tuning", "fine tuning", "finetuning"),
    "trace": ("trace", "traces", "tracing"),
    "liquid": ("liquid", "liquidai", "lfm"),
    "cpu": ("cpu", "local cpu"),
    "model": ("model", "models"),
    "retrieval": ("retrieval", "retrieve", "retriever", "rag"),
    "citation": ("citation", "citations", "cite", "citing"),
    "source": ("source", "sources", "sourced"),
    "eval": ("eval", "evals", "evaluation", "evaluations", "evaluator", "evaluators"),
}


@dataclass(frozen=True)
class RagCase:
    case_id: str
    slice: str
    query: str
    required_terms: tuple[str, ...]
    answer_terms: tuple[str, ...]
    unsupported: bool = False
    min_retrieval_hit_rate: float = 0.2
    max_tokens: int = 260


CASES = [
    RagCase(
        case_id="finance_firms",
        slice="finance",
        query="For Brandon's finance-facing role, what AI news mentions hedge funds, asset managers, banks, payments, or firms like J.P. Morgan, Citadel, Mastercard, Visa, Balyasny, Arrowstreet, and Acadian?",
        required_terms=("jpmorgan", "j.p. morgan", "citadel", "mastercard", "visa", "balyasny", "arrowstreet", "acadian", "hedge fund", "asset manager", "payments"),
        answer_terms=("finance", "risk", "compliance", "bank", "payment", "hedge", "asset"),
        min_retrieval_hit_rate=0.25,
        max_tokens=320,
    ),
    RagCase(
        case_id="local_models",
        slice="local_models",
        query="What do the sources say about running local open models, GGUF, llama.cpp, Qwen, Gemma, LiquidAI, Phi, or NVIDIA models on CPU?",
        required_terms=("gguf", "llama.cpp", "qwen", "gemma", "liquid", "phi", "nvidia", "local", "cpu"),
        answer_terms=("local", "cpu", "model", "gguf", "quant"),
        min_retrieval_hit_rate=0.3,
        max_tokens=300,
    ),
    RagCase(
        case_id="data_flywheel_eval",
        slice="curation",
        query="What tools or workflows help turn production traces into eval or fine-tuning data, including Phoenix, Arize, NVIDIA data flywheel, NeMo Curator, or LLaMA-Factory?",
        required_terms=("phoenix", "arize", "nemo", "curator", "data flywheel", "llama-factory", "fine-tuning", "eval", "trace"),
        answer_terms=("eval", "trace", "curat", "train", "fine"),
        min_retrieval_hit_rate=0.25,
        max_tokens=320,
    ),
    RagCase(
        case_id="unsupported_exact_claim",
        slice="refusal",
        query="Which exact clause changed in Anthropic's subscription terms on May 25, 2026, and what price tier was affected?",
        required_terms=("anthropic", "subscription", "terms", "price", "tier", "clause"),
        answer_terms=("not support", "do not support", "do not contain", "don't have", "not enough", "cannot determine"),
        unsupported=True,
        min_retrieval_hit_rate=0.0,
        max_tokens=180,
    ),
    RagCase(
        case_id="citation_webchat",
        slice="citation",
        query="Give a concise answer about the most relevant AI model and retrieval items in the sources, with inline citations for every sentence.",
        required_terms=("model", "retrieval", "citation", "source", "eval"),
        answer_terms=("model", "retrieval", "citation", "source"),
        min_retrieval_hit_rate=0.2,
        max_tokens=260,
    ),
]


def hf_env() -> dict[str, str]:
    env = os.environ.copy()
    token_path = Path("~/.cache/huggingface/token").expanduser()
    if token_path.exists() and "HF_TOKEN" not in env:
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            env["HF_TOKEN"] = token
    return env


def model_args(model: str) -> list[str]:
    if ":" in model and model.endswith(".gguf"):
        repo, hf_file = model.split(":", 1)
        return ["--hf-repo", repo, "--hf-file", hf_file]
    return ["-hf", model]


def clean_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    if "\n>" in text:
        text = text.rsplit("\n>", 1)[-1].strip()
    if " ... (truncated)\n\n" in text:
        text = text.split(" ... (truncated)\n\n", 1)[-1].strip()
    elif "ANSWER:" in text:
        text = text.rsplit("ANSWER:", 1)[-1].strip()
    text = re.sub(r"\n?\[ Prompt: .*?Generation: .*?\]\s*", "\n", text, flags=re.S)
    text = re.sub(r"\n?Exiting\.\.\.\s*$", "", text)
    return text.strip()


def make_retrieval_agent(db_path: str, alpha: float) -> Any:
    from agents.chat_agent import ChatAgent

    agent = object.__new__(ChatAgent)
    agent.db_path = db_path
    agent.alpha = alpha
    return agent


def source_blob(source: Any) -> str:
    return " ".join(
        part
        for part in [source.id, source.type, source.author or "", source.title or "", source.text, source.url]
        if part
    ).lower()


def term_aliases(term: str) -> tuple[str, ...]:
    return TERM_ALIASES.get(term, (term,))


def text_has_term(text: str, term: str) -> bool:
    lower = text.lower()
    for alias in term_aliases(term):
        alias_lower = alias.lower()
        if re.search(r"[a-z0-9]", alias_lower):
            pattern = r"(?<![a-z0-9])" + re.escape(alias_lower) + r"(?![a-z0-9])"
            if re.search(pattern, lower):
                return True
        elif alias_lower in lower:
            return True
    return False


def retrieve_case(
    agent: ChatAgent,
    case: RagCase,
    max_sources: int,
    context_max_sources: int,
    enable_reranker: bool,
    reranker_model: str,
) -> dict[str, Any]:
    started = time.time()
    options = {
        "max_sources": max_sources,
        "context_max_sources": context_max_sources,
        "use_chunks": True,
        "enable_reranker": enable_reranker,
        "reranker_model": reranker_model,
    }
    sources, warning = agent._retrieve_sources(case.query, options)
    ranked_sources, reranker_info = agent._rerank_sources(case.query, sources, options)
    context_sources = agent._select_context_sources(
        case.query,
        ranked_sources,
        {
            **options,
            "_source_order_is_reranked": reranker_info["applied"],
        },
    )
    elapsed = time.time() - started
    blobs = [source_blob(source) for source in context_sources]
    required_hits = {
        term: [i + 1 for i, blob in enumerate(blobs) if text_has_term(blob, term)]
        for term in case.required_terms
    }
    hit_terms = [term for term, hits in required_hits.items() if hits]
    hit_rate = len(hit_terms) / max(1, len(case.required_terms))
    missing_terms = [term for term in case.required_terms if term not in hit_terms]
    gate_reason = None
    if hit_rate < case.min_retrieval_hit_rate:
        gate_reason = (
            f"retrieval term coverage {hit_rate:.3f} below "
            f"minimum {case.min_retrieval_hit_rate:.3f}"
        )
    return {
        "case_id": case.case_id,
        "slice": case.slice,
        "query": case.query,
        "warning": warning,
        "elapsed_sec": round(elapsed, 3),
        "retrieved_source_count": len(sources),
        "ranked_source_count": len(ranked_sources),
        "context_source_count": len(context_sources),
        "reranker": reranker_info,
        "sources": [
            {
                "index": i + 1,
                "id": source.id,
                "type": source.type,
                "author": source.author,
                "title": source.title,
                "text": source.text,
                "url": source.url,
                "published_at": source.published_at,
            }
            for i, source in enumerate(context_sources)
        ],
        "retrieved_sources": [
            {
                "index": i + 1,
                "id": source.id,
                "type": source.type,
                "author": source.author,
                "title": source.title,
                "url": source.url,
                "published_at": source.published_at,
            }
            for i, source in enumerate(sources)
        ],
        "required_terms": list(case.required_terms),
        "required_term_aliases": {term: list(term_aliases(term)) for term in case.required_terms},
        "hit_terms": hit_terms,
        "missing_terms": missing_terms,
        "hit_rate": round(hit_rate, 3),
        "min_retrieval_hit_rate": case.min_retrieval_hit_rate,
        "coverage_passed": gate_reason is None,
        "gate_reason": gate_reason,
        "required_hits": required_hits,
    }


def build_prompt(case: RagCase, sources: list[dict[str, Any]]) -> str:
    context = ""
    for source in sources:
        label = source["index"]
        heading = source.get("title") or source.get("author") or source.get("type") or "source"
        context += f"[{label}] {heading}\n{source.get('text') or ''}\n{source.get('url') or ''}\n\n"
    return f"{SYSTEM_PROMPT}\n\nSOURCES:\n{context}\nQUESTION: {case.query}\n\nANSWER:"


def run_generation(
    llama_cli: Path,
    model: str,
    case: RagCase,
    sources: list[dict[str, Any]],
    ctx: int,
    threads: int,
    temp: float,
    timeout: int,
) -> dict[str, Any]:
    prompt = build_prompt(case, sources)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tmp:
        tmp.write(prompt)
        tmp_path = Path(tmp.name)

    cmd = [
        str(llama_cli),
        *model_args(model),
        "-f",
        str(tmp_path),
        "-c",
        str(ctx),
        "-n",
        str(case.max_tokens),
        "-t",
        str(threads),
        "--temp",
        str(temp),
        "--no-display-prompt",
        "--log-disable",
        "--simple-io",
        "--no-show-timings",
        "--reasoning",
        "off",
        "--single-turn",
    ]
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            env=hf_env(),
        )
        output = clean_output(proc.stdout)
        return {
            "model": model,
            "case_id": case.case_id,
            "slice": case.slice,
            "returncode": proc.returncode,
            "elapsed_sec": round(time.time() - started, 3),
            "output": output,
            "stderr_tail": proc.stderr.strip()[-2000:],
            "score": score_answer(case, output, sources, proc.returncode),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "model": model,
            "case_id": case.case_id,
            "slice": case.slice,
            "returncode": "timeout",
            "elapsed_sec": timeout,
            "output": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
            "score": {"points": 0, "max_points": 10, "passed": False, "reasons": ["timeout"]},
        }
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def retrieval_gated_result(model: str, case: RagCase, retrieved: dict[str, Any]) -> dict[str, Any]:
    missing = ", ".join(retrieved.get("missing_terms", [])[:8])
    output = (
        "The retrieved sources do not have enough coverage to answer this reliably. "
        f"Coverage hit rate was {retrieved.get('hit_rate')} for the required "
        f"{case.slice} terms; missing examples include: {missing}."
    )
    return {
        "model": model,
        "case_id": case.case_id,
        "slice": case.slice,
        "returncode": "retrieval_gate",
        "elapsed_sec": 0,
        "output": output,
        "stderr_tail": "",
        "retrieval_gate": {
            "coverage_passed": False,
            "hit_rate": retrieved.get("hit_rate"),
            "min_retrieval_hit_rate": retrieved.get("min_retrieval_hit_rate"),
            "gate_reason": retrieved.get("gate_reason"),
            "missing_terms": retrieved.get("missing_terms", []),
        },
        "score": {
            "points": 0,
            "max_points": 10,
            "passed": False,
            "reasons": [retrieved.get("gate_reason") or "retrieval coverage gate failed"],
            "citations": [],
            "valid_citations": [],
            "invalid_citations": [],
        },
    }


def parse_citations(text: str) -> tuple[list[int], list[Any]]:
    cited: list[int] = []
    invalid: list[Any] = []
    for match in re.finditer(r"\[([^\]]+)\]", text):
        content = match.group(1).strip()
        if re.search(r"[A-Za-z]", content):
            invalid.append(content)
            continue
        nums = [int(num) for num in re.findall(r"\d+", content)]
        if nums:
            cited.extend(nums)
        elif content:
            invalid.append(content)
    return cited, invalid


def sentence_split(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", text.strip())
        if len(part.strip()) > 20
    ]


def extract_support_terms(text: str) -> set[str]:
    terms: set[str] = set()
    lower = text.lower()
    aliases = {
        "jpmorgan": ("jpmorgan", "jp morgan", "j.p. morgan"),
        "citadel": ("citadel",),
        "mastercard": ("mastercard",),
        "visa": ("visa",),
        "balyasny": ("balyasny",),
        "arrowstreet": ("arrowstreet",),
        "acadian": ("acadian",),
        "hedge fund": ("hedge fund", "hedge funds"),
        "asset manager": ("asset manager", "asset managers", "asset management"),
        "payments": ("payments", "payment", "payment network", "payment networks"),
        "anthropic": ("anthropic",),
        "subscription": ("subscription", "subscriptions"),
        "terms": ("terms", "term", "clause", "clauses"),
        "price": ("price", "pricing", "tier", "tiers"),
        "gguf": ("gguf",),
        "llama.cpp": ("llama.cpp",),
        "qwen": ("qwen",),
        "gemma": ("gemma", "functiongemma"),
        "liquid": ("liquid", "liquidai", "lfm"),
        "phi": ("phi",),
        "nvidia": ("nvidia", "nemotron"),
        "retrieval": ("retrieval", "retrieve", "retriever", "rag"),
        "citation": ("citation", "citations", "cite", "citing"),
        "eval": ("eval", "evals", "evaluation", "evaluations"),
        "phoenix": ("phoenix", "arize"),
        "curator": ("curator", "nemo curator"),
    }
    for canonical, variants in aliases.items():
        if any(text_has_term(lower, variant) for variant in variants):
            terms.add(canonical)

    for match in re.findall(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*){0,3}\b", text):
        if len(match) > 2:
            terms.add(match.lower())
    for match in re.findall(r"\b[A-Z]{2,8}\b", text):
        terms.add(match.lower())
    return terms


def citation_support(output: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    supported = 0
    checked = 0
    unsupported: list[dict[str, Any]] = []
    for sentence in sentence_split(output):
        cited, malformed = parse_citations(sentence)
        if malformed or not cited:
            continue
        sentence_terms = extract_support_terms(re.sub(r"\[[^\]]+\]", "", sentence))
        if not sentence_terms:
            continue
        for citation_num in cited:
            if not (1 <= citation_num <= len(sources)):
                continue
            checked += 1
            source = sources[citation_num - 1]
            source_text = " ".join(
                str(source.get(key) or "")
                for key in ("author", "title", "text", "url")
            )
            source_terms = extract_support_terms(source_text)
            overlap = sorted(sentence_terms & source_terms)
            if overlap:
                supported += 1
            else:
                unsupported.append(
                    {
                        "citation": citation_num,
                        "sentence": sentence[:240],
                        "source_id": source.get("id"),
                        "sentence_terms": sorted(sentence_terms)[:10],
                    }
                )
    precision = supported / checked if checked else 0.0
    return {
        "checked": checked,
        "supported": supported,
        "precision": round(precision, 3),
        "unsupported": unsupported[:5],
    }


def score_answer(case: RagCase, output: str, sources: list[dict[str, Any]], returncode: int) -> dict[str, Any]:
    reasons: list[str] = []
    text = output.strip()
    lower = text.lower()
    points = 0
    max_points = 10

    if returncode == 0:
        points += 1
    else:
        reasons.append(f"nonzero return code {returncode}")
    if text:
        points += 1
    else:
        reasons.append("empty output")
    if 40 <= len(text) <= 2200:
        points += 1
    else:
        reasons.append("bad length")
    if "<think>" not in lower and "</think>" not in lower:
        points += 1
    else:
        reasons.append("thinking trace leaked")

    cited, malformed_cites = parse_citations(text)
    valid_cites = [idx for idx in cited if 1 <= idx <= len(sources)]
    invalid_cites = [idx for idx in cited if idx < 1 or idx > len(sources)]
    if case.unsupported:
        if valid_cites or not cited:
            points += 2
        else:
            reasons.append("no valid inline citations")
    elif valid_cites:
        points += 2
    else:
        reasons.append("no valid inline citations")
    if not invalid_cites and not malformed_cites:
        points += 1
    else:
        reasons.append(f"invalid citations {invalid_cites + malformed_cites}")

    support = citation_support(text, sources)
    if case.unsupported:
        should_check_support = False
    else:
        should_check_support = True

    if should_check_support and valid_cites and support["checked"]:
        if support["precision"] >= 0.75:
            points += 1
        else:
            reasons.append(f"weak citation support precision {support['precision']}")
    elif should_check_support and valid_cites and case.slice in {"finance", "citation", "local_models"}:
        reasons.append("no citation support terms checked")

    if case.unsupported:
        refusal_terms = (
            "not support",
            "do not support",
            "do not contain",
            "do not specify",
            "does not specify",
            "don't have",
            "not enough",
            "cannot determine",
            "not in the sources",
            "do not identify",
            "does not identify",
        )
        if any(term in lower for term in refusal_terms):
            points += 2
        else:
            reasons.append("did not refuse unsupported question")
        if not any(term in lower for term in ("pro tier", "team tier", "enterprise", "opus", "sonnet", "$")):
            points += 1
        else:
            reasons.append("invented unsupported pricing/model details")
    else:
        answer_hits = [term for term in case.answer_terms if term in lower]
        if len(answer_hits) >= max(1, min(3, len(case.answer_terms))):
            points += 2
        else:
            reasons.append("weak answer term coverage")
        if case.slice == "finance":
            finance_terms = ("j.p. morgan", "jpmorgan", "citadel", "mastercard", "visa", "hedge", "asset", "payment", "risk", "compliance")
            if any(term in lower for term in finance_terms):
                points += 1
            else:
                reasons.append("weak finance framing")
        else:
            points += 1

    return {
        "points": min(points, max_points),
        "max_points": max_points,
        "passed": min(points, max_points) >= 7,
        "reasons": reasons,
        "citations": cited,
        "valid_citations": valid_cites,
        "invalid_citations": invalid_cites,
        "malformed_citations": malformed_cites,
        "citation_support": support,
    }


def summarize(retrieval_rows: list[dict[str, Any]], generation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in generation_rows:
        by_model.setdefault(row["model"], []).append(row)
    model_summaries = []
    for model, rows in by_model.items():
        total = sum(row["score"]["points"] for row in rows)
        max_total = sum(row["score"]["max_points"] for row in rows)
        model_summaries.append(
            {
                "model": model,
                "cases": len(rows),
                "retrieval_gated_cases": sum(1 for row in rows if row.get("returncode") == "retrieval_gate"),
                "passed_cases": sum(1 for row in rows if row["score"]["passed"]),
                "score_pct": round(100 * total / max(1, max_total), 1),
                "avg_elapsed_sec": round(sum(row["elapsed_sec"] for row in rows) / max(1, len(rows)), 3),
                "fail_reasons": {
                    row["case_id"]: row["score"]["reasons"]
                    for row in rows
                    if row["score"]["reasons"]
                },
            }
        )
    model_summaries.sort(key=lambda item: (item["score_pct"], item["passed_cases"], -item["avg_elapsed_sec"]), reverse=True)

    by_slice: dict[str, list[dict[str, Any]]] = {}
    for row in generation_rows:
        by_slice.setdefault(row["slice"], []).append(row)
    slice_leaders = {}
    for slice_name, rows in by_slice.items():
        ranked = sorted(
            rows,
            key=lambda row: (row["score"]["points"] / row["score"]["max_points"], row["score"]["passed"], -row["elapsed_sec"]),
            reverse=True,
        )
        slice_leaders[slice_name] = [
            {
                "rank": i + 1,
                "model": row["model"],
                "case_id": row["case_id"],
                "points": row["score"]["points"],
                "max_points": row["score"]["max_points"],
                "passed": row["score"]["passed"],
                "elapsed_sec": row["elapsed_sec"],
                "reasons": row["score"]["reasons"],
            }
            for i, row in enumerate(ranked[:3])
        ]

    return {
        "retrieval": {
            "cases": len(retrieval_rows),
            "avg_hit_rate": round(sum(row["hit_rate"] for row in retrieval_rows) / max(1, len(retrieval_rows)), 3),
            "coverage_passed_cases": sum(1 for row in retrieval_rows if row.get("coverage_passed")),
            "coverage_failed_cases": sum(1 for row in retrieval_rows if not row.get("coverage_passed")),
            "rows": [
                {
                    "case_id": row["case_id"],
                    "slice": row["slice"],
                    "hit_rate": row["hit_rate"],
                    "min_retrieval_hit_rate": row.get("min_retrieval_hit_rate"),
                    "coverage_passed": row.get("coverage_passed"),
                    "gate_reason": row.get("gate_reason"),
                    "hit_terms": row["hit_terms"],
                    "missing_terms": row.get("missing_terms", []),
                    "required_term_aliases": row.get("required_term_aliases", {}),
                    "source_count": len(row["sources"]),
                    "retrieved_source_count": row.get("retrieved_source_count", len(row["sources"])),
                    "context_source_count": row.get("context_source_count", len(row["sources"])),
                    "reranker": row.get("reranker", {}),
                    "warning": row["warning"],
                    "elapsed_sec": row["elapsed_sec"],
                }
                for row in retrieval_rows
            ],
        },
        "generation": model_summaries,
        "slice_leaders": slice_leaders,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="output_data/ai_news.db")
    parser.add_argument("--llama-cli", default="~/opt/llama.cpp/llama-cli")
    parser.add_argument("--out-dir", default="output_data/model_bench/rag_webchat_v1")
    parser.add_argument("--max-sources", type=int, default=10)
    parser.add_argument("--context-max-sources", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--enable-reranker", action="store_true")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--cases", nargs="*", help="Optional case_id allow-list")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--skip-coverage-gate", action="store_true", help="Run generation even when retrieval coverage is below the case minimum")
    parser.add_argument("--reuse-retrieval-dir", help="Use retrieval_results.jsonl from this directory and run generation only")
    parser.add_argument("--rescore-from-dir", help="Re-score an existing rag_webchat output directory without rerunning retrieval or generation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    retrieval_path = out_dir / "retrieval_results.jsonl"
    generation_path = out_dir / "generation_results.jsonl"
    summary_path = out_dir / "rag_webchat_summary.json"

    case_by_id = {case.case_id: case for case in CASES}

    if args.rescore_from_dir:
        source_dir = Path(args.rescore_from_dir).expanduser().resolve()
        retrieval_rows = [
            json.loads(line)
            for line in (source_dir / "retrieval_results.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        generation_rows = [
            json.loads(line)
            for line in (source_dir / "generation_results.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        retrieval_by_id = {row["case_id"]: row for row in retrieval_rows}
        for row in generation_rows:
            case = case_by_id[row["case_id"]]
            row["score"] = score_answer(
                case,
                row.get("output", ""),
                retrieval_by_id[row["case_id"]].get("sources", []),
                0 if row.get("returncode") == 0 else row.get("returncode"),
            )
        retrieval_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in retrieval_rows) + "\n",
            encoding="utf-8",
        )
        generation_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in generation_rows) + "\n",
            encoding="utf-8",
        )
        summary = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rescored_from_dir": str(source_dir),
            "retrieval_path": str(retrieval_path),
            "generation_path": str(generation_path),
            **summarize(retrieval_rows, generation_rows),
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    llama_cli = Path(args.llama_cli).expanduser().resolve()

    cases = [case for case in CASES if not args.cases or case.case_id in set(args.cases)]
    if not cases:
        raise SystemExit(f"No benchmark cases matched: {args.cases}")

    retrieval_rows = []
    if args.reuse_retrieval_dir:
        source_dir = Path(args.reuse_retrieval_dir).expanduser().resolve()
        selected_case_ids = {case.case_id for case in cases}
        retrieval_rows = [
            row
            for row in (
                json.loads(line)
                for line in (source_dir / "retrieval_results.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if row["case_id"] in selected_case_ids
        ]
        found_case_ids = {row["case_id"] for row in retrieval_rows}
        missing = selected_case_ids - found_case_ids
        if missing:
            raise SystemExit(f"Missing retrieval rows for cases: {sorted(missing)}")
        retrieval_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in retrieval_rows) + "\n",
            encoding="utf-8",
        )
    else:
        agent = make_retrieval_agent(args.db_path, args.alpha)
        with retrieval_path.open("w", encoding="utf-8") as out:
            for case in cases:
                print(f"RETRIEVE {case.case_id}", flush=True)
                row = retrieve_case(
                    agent,
                    case,
                    args.max_sources,
                    args.context_max_sources,
                    args.enable_reranker,
                    args.reranker_model,
                )
                retrieval_rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                print(
                    json.dumps(
                        {
                            "case_id": row["case_id"],
                            "hit_rate": row["hit_rate"],
                            "source_count": len(row["sources"]),
                            "retrieved_source_count": row.get("retrieved_source_count"),
                            "reranked": row.get("reranker", {}).get("applied"),
                            "warning": row["warning"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    generation_rows = []
    if not args.retrieval_only:
        retrieval_by_id = {row["case_id"]: row for row in retrieval_rows}
        with generation_path.open("w", encoding="utf-8") as out:
            for model in args.models:
                print(f"MODEL {model}", flush=True)
                for case in cases:
                    retrieved = retrieval_by_id[case.case_id]
                    print(f"  CASE {case.case_id}", flush=True)
                    if not args.skip_coverage_gate and not retrieved.get("coverage_passed", True):
                        row = retrieval_gated_result(model, case, retrieved)
                    else:
                        row = run_generation(
                            llama_cli=llama_cli,
                            model=model,
                            case=case,
                            sources=retrieved["sources"],
                            ctx=args.ctx,
                            threads=args.threads,
                            temp=args.temp,
                            timeout=args.timeout,
                        )
                    generation_rows.append(row)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    print(json.dumps({"case_id": row["case_id"], "score": row["score"]["points"], "passed": row["score"]["passed"], "elapsed_sec": row["elapsed_sec"]}, ensure_ascii=False), flush=True)

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "retrieval_path": str(retrieval_path),
        "generation_path": str(generation_path),
        **summarize(retrieval_rows, generation_rows),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
