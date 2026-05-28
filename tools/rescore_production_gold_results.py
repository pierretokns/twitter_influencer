#!/usr/bin/env python3
"""Rescore existing production gold generation outputs without rerunning models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from production_gold_bench import load_cases, score_answer, summarize


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    gold_path = Path(args.gold).expanduser().resolve()
    results_path = Path(args.results).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = {case["case_id"]: case for case in load_cases(gold_path)}
    rescored = []
    for row in load_jsonl(results_path):
        case = cases.get(row.get("case_id"))
        if not case or row.get("case_type") != "rag_generation":
            rescored.append(row)
            continue
        updated = dict(row)
        score = score_answer(
            case,
            row.get("answer", ""),
            row.get("source_limit"),
            row.get("source_char_limit"),
        )
        updated["score"] = score
        updated["passed"] = bool(score["passed"]) and row.get("error") is None
        rescored.append(updated)

    out_results = out_dir / "production_gold_results.jsonl"
    out_summary = out_dir / "production_gold_summary.json"
    with out_results.open("w", encoding="utf-8") as handle:
        for row in rescored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "gold_path": str(gold_path),
        "source_results_path": str(results_path),
        "rows": len(rescored),
        **summarize(rescored),
        "outputs": {"results": str(out_results), "summary": str(out_summary)},
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
