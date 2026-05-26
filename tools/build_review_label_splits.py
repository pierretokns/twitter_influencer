#!/usr/bin/env python3
"""Convert completed human-review labels into holdout and training split files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


ANSWER_REQUIRED_LABELS = (
    "acceptable",
    "factual_consistency",
    "usefulness",
    "missing_important_context",
)
CITATION_REQUIRED_LABELS = ("support_status",)
VALID_SUPPORT_STATUS = {"supported", "partial", "unsupported", "out_of_range"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def answer_label_errors(row: dict[str, Any]) -> list[str]:
    labels = row.get("human_labels", {})
    errors = []
    for key in ANSWER_REQUIRED_LABELS:
        if labels.get(key) is None:
            errors.append(f"missing answer label {key}")
    if labels.get("acceptable") is not None and not isinstance(labels.get("acceptable"), bool):
        errors.append("acceptable must be boolean")
    if labels.get("missing_important_context") is not None and not isinstance(labels.get("missing_important_context"), bool):
        errors.append("missing_important_context must be boolean")
    for key in ("factual_consistency", "usefulness"):
        value = labels.get(key)
        if value is not None and value not in {"good", "partial", "poor"}:
            errors.append(f"{key} must be good, partial, or poor")
    return errors


def citation_label_errors(row: dict[str, Any]) -> list[str]:
    labels = row.get("human_labels", {})
    status = labels.get("support_status")
    errors = []
    if status is None:
        errors.append("missing citation support_status")
    elif status not in VALID_SUPPORT_STATUS:
        errors.append("support_status must be supported, partial, unsupported, or out_of_range")
    if status in {"supported", "partial"}:
        if not labels.get("supporting_source_numbers"):
            errors.append("supported/partial rows need supporting_source_numbers")
        if not labels.get("supporting_spans"):
            errors.append("supported/partial rows need supporting_spans")
    return errors


def split_answer_rows(rows: list[dict[str, Any]], holdout_models: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    holdout: list[dict[str, Any]] = []
    train_candidates: list[dict[str, Any]] = []
    unlabeled: list[dict[str, Any]] = []
    for row in rows:
        errors = answer_label_errors(row)
        if errors:
            copy = dict(row)
            copy["label_errors"] = errors
            unlabeled.append(copy)
            continue
        labels = row["human_labels"]
        reviewed = {
            **row,
            "review_status": "reviewed",
            "training_status": "holdout_regression",
        }
        if row.get("model_label") in holdout_models or not labels.get("acceptable"):
            holdout.append(reviewed)
            continue
        trainable = labels.get("acceptable") is True and labels.get("factual_consistency") == "good"
        if trainable:
            train_candidates.append({**reviewed, "training_status": "candidate_train_pending_approval"})
        else:
            holdout.append(reviewed)
    return holdout, train_candidates, unlabeled


def split_citation_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    holdout: list[dict[str, Any]] = []
    train_candidates: list[dict[str, Any]] = []
    unlabeled: list[dict[str, Any]] = []
    for row in rows:
        errors = citation_label_errors(row)
        if errors:
            copy = dict(row)
            copy["label_errors"] = errors
            unlabeled.append(copy)
            continue
        reviewed = {
            **row,
            "review_status": "reviewed",
            "training_status": "holdout_regression",
        }
        status = row["human_labels"]["support_status"]
        if status in {"unsupported", "out_of_range"}:
            holdout.append(reviewed)
        elif status in {"supported", "partial"}:
            train_candidates.append({**reviewed, "training_status": "candidate_train_pending_approval"})
        else:
            holdout.append(reviewed)
    return holdout, train_candidates, unlabeled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", default="output_data/gold_eval/human_review_v1")
    parser.add_argument("--out-dir", default="output_data/gold_eval/human_review_v1_splits")
    parser.add_argument(
        "--holdout-models",
        default="NVIDIA-Nemotron-Nano",
        help="Comma-separated model labels to always keep as regression/negative holdout.",
    )
    args = parser.parse_args()

    review_dir = (ROOT / args.review_dir).resolve()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout_models = {item.strip() for item in args.holdout_models.split(",") if item.strip()}

    answer_rows = load_jsonl(review_dir / "answer_review.jsonl")
    citation_rows = load_jsonl(review_dir / "citation_span_review.jsonl")

    answer_holdout, answer_train, answer_unlabeled = split_answer_rows(answer_rows, holdout_models)
    citation_holdout, citation_train, citation_unlabeled = split_citation_rows(citation_rows)

    holdout_rows = answer_holdout + citation_holdout
    train_rows = answer_train + citation_train
    unlabeled_rows = answer_unlabeled + citation_unlabeled

    paths = {
        "holdout_regression": out_dir / "holdout_regression.jsonl",
        "candidate_train_pending_approval": out_dir / "candidate_train_pending_approval.jsonl",
        "unlabeled_or_invalid": out_dir / "unlabeled_or_invalid.jsonl",
        "summary": out_dir / "split_summary.json",
    }
    write_jsonl(paths["holdout_regression"], holdout_rows)
    write_jsonl(paths["candidate_train_pending_approval"], train_rows)
    write_jsonl(paths["unlabeled_or_invalid"], unlabeled_rows)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "review_dir": str(review_dir),
        "holdout_models": sorted(holdout_models),
        "counts": {
            "answer_rows": len(answer_rows),
            "citation_rows": len(citation_rows),
            "reviewed_rows": len(holdout_rows) + len(train_rows),
            "holdout_regression_rows": len(holdout_rows),
            "candidate_train_pending_approval_rows": len(train_rows),
            "unlabeled_or_invalid_rows": len(unlabeled_rows),
            "answer_unlabeled_or_invalid_rows": len(answer_unlabeled),
            "citation_unlabeled_or_invalid_rows": len(citation_unlabeled),
        },
        "unlabeled_by_model": dict(Counter(row.get("model_label", "") for row in unlabeled_rows)),
        "unlabeled_by_slice": dict(Counter(row.get("slice", "") for row in unlabeled_rows)),
        "training_rule": (
            "Rows remain excluded from training until human labels are complete and "
            "candidate_train_pending_approval rows are explicitly approved."
        ),
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
