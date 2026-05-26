#!/usr/bin/env python3
"""
Constrained JSON benchmark for local GGUF models via llama.cpp.

This tests whether schema-constrained decoding fixes the structured-output
slice before we decide that fine-tuning is necessary.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


DEFAULT_MODELS = [
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF:NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf",
    "unsloth/Phi-4-mini-instruct-GGUF:Phi-4-mini-instruct-Q3_K_M.gguf",
    "unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-UD-IQ2_M.gguf",
    "LiquidAI/LFM2-2.6B-GGUF:LFM2-2.6B-Q4_K_M.gguf",
]


@dataclass(frozen=True)
class Case:
    case_id: str
    prompt: str
    schema: dict[str, Any]
    required_values: dict[str, tuple[str, ...]]
    max_tokens: int = 260


LEGACY_CASES = [
    Case(
        case_id="workflow_plan",
        prompt=(
            "Create a local CPU news-agent workflow plan for Brandon. "
            "It must include retrieval, summarization, citation verification, "
            "finance relevance, and delivery. Return only JSON."
        ),
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["workflow_name", "steps", "requires_verifier", "primary_model_role"],
            "properties": {
                "workflow_name": {"type": "string"},
                "steps": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["name", "purpose"],
                        "properties": {
                            "name": {"type": "string"},
                            "purpose": {"type": "string"},
                        },
                    },
                },
                "requires_verifier": {"type": "boolean"},
                "primary_model_role": {"type": "string"},
            },
        },
        required_values={
            "all": ("retriev", "summar", "citation", "finance", "deliver"),
            "primary_model_role": ("summar", "brief", "writer", "generation"),
        },
    ),
    Case(
        case_id="citation_decision",
        prompt=(
            "Given a source says 'J.P. Morgan is testing AI assistants for analysts' "
            "and another source says 'Gemma models are available as local GGUF files', "
            "emit a citation decision for the claim 'J.P. Morgan is using Gemma GGUF "
            "for analyst assistants'. Return only JSON."
        ),
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["claim_supported", "support_level", "reason", "needed_evidence"],
            "properties": {
                "claim_supported": {"type": "boolean"},
                "support_level": {"type": "string", "enum": ["supported", "partial", "unsupported"]},
                "reason": {"type": "string"},
                "needed_evidence": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {"type": "string"},
                },
            },
        },
        required_values={
            "support_level": ("partial", "unsupported"),
            "all": ("j.p. morgan", "gemma", "evidence"),
        },
    ),
]


PRODUCTION_CASES = [
    Case(
        case_id="qa_review_result",
        prompt=(
            "Review this Brandon AI-news brief draft for production delivery.\n\n"
            "POST:\n"
            "J.P. Morgan is piloting analyst assistants, while Gemma GGUF models are "
            "useful for local CPU workflows. The draft has clear citations and no "
            "social engagement bait.\n\n"
            "Return the QA review object used by the app. The score is a 1-10 rating, "
            "not a percentage."
        ),
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["approved", "score", "issues", "summary"],
            "properties": {
                "approved": {"type": "boolean"},
                "score": {"type": "number", "minimum": 1, "maximum": 10},
                "issues": {"type": "array", "maxItems": 5, "items": {"type": "string"}},
                "summary": {"type": "string"},
            },
        },
        required_values={
            "summary": ("clear", "approved", "professional", "citation", "brief"),
        },
    ),
    Case(
        case_id="citation_verification_result",
        prompt=(
            "Emit one citation verification result for this generated sentence and cited source.\n\n"
            "Sentence: J.P. Morgan is using Gemma GGUF for analyst assistants [2].\n"
            "Cited source 2: Gemma models are available as local GGUF files.\n\n"
            "The cited source alone does not mention J.P. Morgan or analyst assistants, "
            "so the citation must not be marked verified. The result must follow the "
            "app's citation verifier fields."
        ),
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["citation", "sentence", "source", "similarity", "entity_overlap", "status", "reason"],
            "properties": {
                "citation": {"type": "integer", "minimum": 1, "maximum": 10},
                "sentence": {"type": "string"},
                "source": {"type": "string"},
                "similarity": {"type": "number", "minimum": 0, "maximum": 1},
                "entity_overlap": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
                "status": {"type": "string", "enum": ["verified", "weak", "invalid"]},
                "reason": {"type": "string"},
            },
        },
        required_values={
            "status": ("weak", "invalid"),
            "all": ("j.p. morgan", "gemma", "source", "evidence", "support"),
        },
    ),
    Case(
        case_id="retrieval_gate_result",
        prompt=(
            "The query asks: What AI news mentions Balyasny, Citadel, and Acadian?\n"
            "Retrieved sources mention OpenAI DevDay, Gemma local models, and generic payment networks. "
            "They do not mention Balyasny, Citadel, or Acadian.\n\n"
            "Emit a retrieval gate result for deciding whether generation should proceed. "
            "Because all required named entities are missing, query_supported and coverage_passed "
            "must be false."
        ),
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["query_supported", "coverage_passed", "hit_rate", "missing_terms", "gate_reason"],
            "properties": {
                "query_supported": {"type": "boolean"},
                "coverage_passed": {"type": "boolean"},
                "hit_rate": {"type": "number", "minimum": 0, "maximum": 1},
                "missing_terms": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 10,
                    "items": {"type": "string"},
                },
                "gate_reason": {"type": "string"},
            },
        },
        required_values={
            "gate_reason": ("missing", "coverage", "insufficient", "below"),
            "all": ("balyasny", "citadel", "acadian"),
        },
    ),
    Case(
        case_id="route_decision",
        prompt=(
            "A user asks: Which exact clause changed in Anthropic's subscription terms on May 25, 2026?\n"
            "The retrieved sources mention Anthropic generally but do not contain subscription terms, "
            "clauses, prices, or affected tiers.\n\n"
            "Emit the route decision for the local news agent."
        ),
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["route", "confidence", "reason", "required_sources"],
            "properties": {
                "route": {
                    "type": "string",
                    "enum": [
                        "answer_from_sources",
                        "refuse_insufficient_sources",
                        "run_retrieval",
                        "verify_citations",
                        "qa_review",
                    ],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
                "required_sources": {
                    "type": "array",
                    "maxItems": 6,
                    "items": {"type": "string"},
                },
            },
        },
        required_values={
            "route": ("refuse_insufficient_sources",),
            "all": ("subscription", "terms", "clause", "source"),
        },
    ),
    Case(
        case_id="finance_relevance_result",
        prompt=(
            "Classify this source for Brandon's finance-domain news slice.\n\n"
            "Source: Balyasny Asset Management built an AI research engine to help investment teams "
            "search internal and external research. The article mentions investment research, analysts, "
            "and asset management workflows.\n\n"
            "Return the finance relevance result. Because this mentions a priority finance firm and "
            "investment research workflow, finance_relevant must be true."
        ),
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["finance_relevant", "priority_entities", "domain_tags", "reason"],
            "properties": {
                "finance_relevant": {"type": "boolean"},
                "priority_entities": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {"type": "string"},
                },
                "domain_tags": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {
                        "type": "string",
                        "enum": [
                            "hedge_fund",
                            "asset_manager",
                            "bank",
                            "payments",
                            "fintech",
                            "risk",
                            "investment_research",
                            "regulated_finance",
                        ],
                    },
                },
                "reason": {"type": "string"},
            },
        },
        required_values={
            "finance_relevant": ("true",),
            "priority_entities": ("balyasny",),
            "all": ("investment", "research", "asset"),
        },
    ),
    Case(
        case_id="citation_correction_action",
        prompt=(
            "A generated sentence says: Mastercard selected LiquidAI LFM2-24B as its production "
            "fraud-detection model [3].\n"
            "Citation [3] source says only: LiquidAI released LFM2-24B-A2B GGUF models for local inference.\n"
            "There is no Mastercard deployment evidence in the cited source.\n\n"
            "Emit the citation correction action used by the app. The invalid citation should be removed "
            "unless a better source exists; here no replacement source exists."
        ),
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "citation", "replacement_citation", "corrected_sentence", "warning"],
            "properties": {
                "action": {"type": "string", "enum": ["keep", "remove", "replace", "warn"]},
                "citation": {"type": "integer", "minimum": 1, "maximum": 10},
                "replacement_citation": {"type": "integer", "minimum": 0, "maximum": 10},
                "corrected_sentence": {"type": "string"},
                "warning": {"type": "string"},
            },
        },
        required_values={
            "action": ("remove",),
            "warning": ("mastercard", "deployment", "unsupported", "invalid", "source"),
            "all": ("liquidai", "lfm2", "mastercard"),
        },
    ),
    Case(
        case_id="delivery_payload",
        prompt=(
            "Create the final delivery payload for Brandon's background news summary after QA passed.\n"
            "The summary is finance-facing, has 3 verified citations, and should be delivered to webchat "
            "and the daily digest. Do not include social posting or influencer language."
        ),
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["audience", "channels", "citation_count", "include_social_copy", "status", "notes"],
            "properties": {
                "audience": {"type": "string"},
                "channels": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {"type": "string", "enum": ["webchat", "daily_digest", "email", "discord", "linkedin"]},
                },
                "citation_count": {"type": "integer", "minimum": 0, "maximum": 20},
                "include_social_copy": {"type": "boolean"},
                "status": {"type": "string", "enum": ["ready", "hold_for_review", "blocked"]},
                "notes": {"type": "string"},
            },
        },
        required_values={
            "audience": ("brandon",),
            "channels": ("webchat", "daily_digest"),
            "include_social_copy": ("false",),
            "status": ("ready",),
            "all": ("finance", "citation"),
        },
    ),
]

CASES = PRODUCTION_CASES


CITATION_VERIFIER_FEW_SHOT = """\

Hard-negative examples for citation status:

Example A:
Sentence: Citadel deployed Qwen GGUF for portfolio research [1].
Cited source 1: Citadel is evaluating AI tools for research triage.
Correct status: weak
Reason: The source mentions Citadel and AI research triage, but not Qwen or GGUF.

Example B:
Sentence: Mastercard uses Gemma local models for fraud detection [2].
Cited source 2: Gemma models are available as local GGUF files.
Correct status: invalid
Reason: The source supports Gemma local model availability only; it does not mention Mastercard or fraud detection.

Example C:
Sentence: NeMo Curator helps prepare fine-tuning datasets [3].
Cited source 3: NVIDIA NeMo Curator helps filter, deduplicate, and prepare datasets for model training.
Correct status: verified
Reason: The source directly supports the claim.

Rule: only use status verified when the cited source directly supports the full sentence, including named entities, tools, and action. If the source supports only part of the sentence, use weak. If it supports the wrong entity or no material part, use invalid.
"""


def apply_prompt_variant(cases: list[Case], verifier_few_shot: bool) -> list[Case]:
    if not verifier_few_shot:
        return cases
    updated: list[Case] = []
    for case in cases:
        if case.case_id == "citation_verification_result":
            updated.append(
                Case(
                    case_id=case.case_id,
                    prompt=f"{case.prompt}\n\n{CITATION_VERIFIER_FEW_SHOT}",
                    schema=case.schema,
                    required_values=case.required_values,
                    max_tokens=case.max_tokens,
                )
            )
        else:
            updated.append(case)
    return updated


def load_external_cases(path: Path) -> list[Case]:
    cases: list[Case] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("case_type") and row.get("case_type") != "structured_contract":
                continue
            required_values = {
                key: tuple(str(value) for value in values)
                for key, values in row.get("required_values", {}).items()
            }
            cases.append(
                Case(
                    case_id=row.get("case_id") or row["trace_id"],
                    prompt=row["prompt"],
                    schema=row["schema"],
                    required_values=required_values,
                    max_tokens=int(row.get("max_tokens", 260)),
                )
            )
    return cases


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
    text = re.sub(r"\x1b\[[0-9;]*m", "", text.strip())
    if " ... (truncated)\n\n" in text:
        text = text.split(" ... (truncated)\n\n", 1)[-1].strip()
    if "JSON:" in text:
        text = text.rsplit("JSON:", 1)[-1].strip()
    text = re.sub(r"\n?Exiting\.\.\.\s*$", "", text)
    return text.strip()


def extract_json(text: str) -> Any:
    text = clean_output(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\})", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(1))


def flattened(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).lower()


def validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Small JSON Schema subset validator to keep the VM runner dependency-free."""
    errors: list[str] = []
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(value, dict):
            return [f"{path}: expected object"]
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{path}: missing required key {key}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}: unexpected key {key}")
        for key, child_schema in properties.items():
            if key in value:
                errors.extend(validate_schema(value[key], child_schema, f"{path}.{key}"))
        return errors

    if schema_type == "array":
        if not isinstance(value, list):
            return [f"{path}: expected array"]
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(value) < min_items:
            errors.append(f"{path}: expected at least {min_items} items")
        if max_items is not None and len(value) > max_items:
            errors.append(f"{path}: expected at most {max_items} items")
        item_schema = schema.get("items")
        if item_schema:
            for idx, item in enumerate(value):
                errors.extend(validate_schema(item, item_schema, f"{path}[{idx}]"))
        return errors

    if schema_type == "string":
        if not isinstance(value, str):
            return [f"{path}: expected string"]
    elif schema_type == "boolean":
        if not isinstance(value, bool):
            return [f"{path}: expected boolean"]
    elif schema_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            return [f"{path}: expected integer"]
    elif schema_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return [f"{path}: expected number"]

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: expected one of {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum {schema['maximum']}")
    return errors


def score(case: Case, output: str, returncode: int) -> dict[str, Any]:
    reasons: list[str] = []
    points = 0
    parsed: Any = None
    sampler_error = "failed to initialize samplers" in output.lower()
    if returncode == 0 and not sampler_error:
        points += 1
    else:
        if sampler_error:
            reasons.append("llama.cpp sampler initialization error")
        else:
            reasons.append(f"nonzero return code {returncode}")
    if "<think>" not in output.lower():
        points += 1
    else:
        reasons.append("thinking trace leaked")
    try:
        parsed = extract_json(output)
        points += 2
    except Exception as exc:
        reasons.append(f"invalid JSON: {type(exc).__name__}")
        parsed = None

    schema_errors: list[str] = []
    critical_semantic_misses: list[str] = []
    if isinstance(parsed, dict):
        schema_errors = validate_schema(parsed, case.schema)
        if not schema_errors:
            points += 3
        else:
            reasons.append(f"schema errors: {schema_errors[:5]}")
        text = flattened(parsed)
        for field, terms in case.required_values.items():
            field_text = text if field == "all" else str(parsed.get(field, "")).lower()
            if any(term in field_text for term in terms):
                points += 1
            else:
                reason = f"semantic miss for {field}: expected one of {terms}"
                reasons.append(reason)
                if field != "all":
                    critical_semantic_misses.append(reason)
    else:
        reasons.append("parsed output is not an object")

    max_points = 7 + len(case.required_values)
    passed = (
        min(points, max_points) >= max_points - 1
        and not schema_errors
        and not critical_semantic_misses
        and parsed is not None
    )
    return {
        "points": min(points, max_points),
        "max_points": max_points,
        "score_pct": round(100 * min(points, max_points) / max_points, 1),
        "passed": passed,
        "reasons": reasons,
        "parsed": parsed,
        "sampler_error": sampler_error,
        "schema_errors": schema_errors,
        "critical_semantic_misses": critical_semantic_misses,
    }


def run_case(
    llama_cli: Path,
    model: str,
    case: Case,
    ctx: int,
    threads: int,
    timeout: int,
    template_mode: str,
    chat_template: str | None,
    validation_retry: bool = False,
) -> dict[str, Any]:
    system_prompt = "You are a precise JSON generator. Follow the schema exactly. Do not include markdown or prose."
    user_prompt = f"TASK: {case.prompt}\n\nReturn only the JSON object."
    raw_prompt = (
        "You are a precise JSON generator. Follow the schema exactly. "
        "Do not include markdown or prose.\n\n"
        f"TASK: {case.prompt}\n\nJSON:"
    )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as schema_file:
        json.dump(case.schema, schema_file)
        schema_path = Path(schema_file.name)

    cmd = [
        str(llama_cli),
        *model_args(model),
        "--json-schema-file",
        str(schema_path),
        "-c",
        str(ctx),
        "-n",
        str(case.max_tokens),
        "-t",
        str(threads),
        "--temp",
        "0",
        "--no-display-prompt",
        "--log-disable",
        "--simple-io",
        "--no-show-timings",
        "--reasoning",
        "off",
        "--single-turn",
    ]
    if chat_template:
        cmd.extend(["--chat-template", chat_template])
    if template_mode == "chat":
        cmd.extend(["--conversation", "--system-prompt", system_prompt, "--prompt", user_prompt])
    else:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as prompt_file:
            prompt_file.write(raw_prompt)
            prompt_path = Path(prompt_file.name)
        cmd.extend(["-f", str(prompt_path)])
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
        row = {
            "model": model,
            "case_id": case.case_id,
            "returncode": proc.returncode,
            "elapsed_sec": round(time.time() - started, 3),
            "output": output,
            "stderr_tail": proc.stderr.strip()[-2000:],
        }
        row["score"] = score(case, output, proc.returncode)
        if validation_retry and not row["score"]["passed"] and not row["score"].get("sampler_error"):
            retry_prompt = (
                f"{case.prompt}\n\n"
                "Previous constrained JSON attempt failed validation.\n"
                f"Validation errors: {json.dumps(row['score']['reasons'][:8], ensure_ascii=False)}\n"
                "Return a corrected JSON object. Preserve the required schema exactly and fix the semantic fields."
            )
            retry_row = run_case(
                llama_cli,
                model,
                replace(case, prompt=retry_prompt),
                ctx,
                threads,
                timeout,
                template_mode,
                chat_template,
                validation_retry=False,
            )
            retry_row["case_id"] = case.case_id
            retry_row["validation_retry_used"] = True
            retry_row["initial_output"] = row["output"]
            retry_row["initial_score"] = row["score"]
            retry_row["elapsed_sec"] = round(row["elapsed_sec"] + retry_row["elapsed_sec"], 3)
            if retry_row["score"]["passed"] or retry_row["score"]["points"] > row["score"]["points"]:
                return retry_row
            row["validation_retry_used"] = True
            row["retry_output"] = retry_row["output"]
            row["retry_score"] = retry_row["score"]
            row["elapsed_sec"] = round(row["elapsed_sec"] + retry_row["elapsed_sec"], 3)
        else:
            row["validation_retry_used"] = False
        return row
    except subprocess.TimeoutExpired:
        return {
            "model": model,
            "case_id": case.case_id,
            "returncode": "timeout",
            "elapsed_sec": timeout,
            "output": "",
            "stderr_tail": "",
            "score": {
                "points": 0,
                "max_points": 10,
                "score_pct": 0,
                "passed": False,
                "reasons": ["timeout"],
                "parsed": None,
            },
        }
    finally:
        paths = [schema_path]
        if template_mode != "chat":
            paths.append(prompt_path)
        for path in paths:
            try:
                path.unlink()
            except OSError:
                pass


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    models = []
    for model, model_rows in by_model.items():
        total = sum(row["score"]["points"] for row in model_rows)
        max_total = sum(row["score"]["max_points"] for row in model_rows)
        models.append(
            {
                "model": model,
                "cases": len(model_rows),
                "passed_cases": sum(1 for row in model_rows if row["score"]["passed"]),
                "score_pct": round(100 * total / max(1, max_total), 1),
                "avg_elapsed_sec": round(sum(row["elapsed_sec"] for row in model_rows) / max(1, len(model_rows)), 3),
                "fail_reasons": {
                    row["case_id"]: row["score"]["reasons"]
                    for row in model_rows
                    if row["score"]["reasons"]
                },
            }
        )
    models.sort(key=lambda item: (item["score_pct"], item["passed_cases"], -item["avg_elapsed_sec"]), reverse=True)
    return {"models": models}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-cli", default="~/opt/llama.cpp/llama-cli")
    parser.add_argument("--out-dir", default="output_data/model_bench/constrained_json_v1")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--template-mode", choices=["raw", "chat"], default="raw")
    parser.add_argument("--chat-template", help="Optional llama.cpp built-in chat template name, e.g. phi4, gemma, chatml")
    parser.add_argument("--case-set", choices=["production", "legacy"], default="production")
    parser.add_argument("--cases-jsonl", help="External structured-contract cases JSONL")
    parser.add_argument("--verifier-few-shot", action="store_true", help="Add hard-negative few-shot examples to the citation verifier contract")
    parser.add_argument("--validation-retry", action="store_true", help="Retry failed constrained outputs once with validation errors included in the prompt")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "constrained_results.jsonl"
    summary_path = out_dir / "constrained_summary.json"
    llama_cli = Path(args.llama_cli).expanduser().resolve()
    cases = (
        load_external_cases(Path(args.cases_jsonl).expanduser().resolve())
        if args.cases_jsonl
        else PRODUCTION_CASES if args.case_set == "production" else LEGACY_CASES
    )
    cases = apply_prompt_variant(cases, args.verifier_few_shot)

    rows = []
    with results_path.open("w", encoding="utf-8") as out:
        for model in args.models:
            print(f"MODEL {model}", flush=True)
            for case in cases:
                print(f"  CASE {case.case_id}", flush=True)
                row = run_case(
                    llama_cli,
                    model,
                    case,
                    args.ctx,
                    args.threads,
                    args.timeout,
                    args.template_mode,
                    args.chat_template,
                    args.validation_retry,
                )
                rows.append(row)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                print(json.dumps({"case_id": row["case_id"], "score_pct": row["score"]["score_pct"], "passed": row["score"]["passed"], "elapsed_sec": row["elapsed_sec"]}, ensure_ascii=False), flush=True)

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results_path": str(results_path),
        **summarize(rows),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
