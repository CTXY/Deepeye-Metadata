#!/usr/bin/env python3
"""
Build a Historical QA file from BIRD dev set for memory augmentation.

Reads:
    dataset/bird/dev/dev.json

For each database (db_id), takes the first 50% questions (by original order)
and writes them as historical QA pairs to a JSONL file, one record per line:
    {
        "qa_id": <int>,           # source question_id
        "question": <str>,        # natural language question
        "sql": <str>,             # gold SQL
        "database_id": <str>      # BIRD db_id
    }

Usage (from project root):
    python script/build_historical_qa_from_bird_dev.py \\
        --input dataset/bird/dev/dev.json \\
        --output workspace/historical_qa/bird/dev_historical_qa.jsonl \\
        --ratio 0.5
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, DefaultDict
from collections import defaultdict


def build_historical_qa(
    input_path: str,
    output_path: str,
    ratio: float = 0.5,
) -> None:
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with input_file.open("r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    # Group by db_id in original order
    by_db: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in data:
        db_id = row.get("db_id")
        if db_id is None:
            continue
        by_db[db_id].append(row)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    num_written = 0
    with output_file.open("w", encoding="utf-8") as out_f:
        for db_id, rows in by_db.items():
            if not rows:
                continue
            # take first 50% (or user-specified ratio) by original order
            k = max(1, int(len(rows) * ratio))
            selected = rows[:k]
            for row in selected:
                qa = {
                    "qa_id": row.get("question_id"),
                    "question": row.get("question", ""),
                    "sql": row.get("SQL", ""),
                    "database_id": db_id,
                }
                out_f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                num_written += 1

    print(f"Written {num_written} historical QA records to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build historical QA JSONL from BIRD dev.json for memory augmentation."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/bird/dev/dev.json",
        help="Path to BIRD dev.json (default: dataset/bird/dev/dev.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="workspace/historical_qa/bird/dev_historical_qa.jsonl",
        help="Output JSONL path for historical QA pairs.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="Fraction of questions per database to use as historical QA (default: 0.5).",
    )
    args = parser.parse_args()

    if not (0.0 < args.ratio <= 1.0):
        raise ValueError(f"ratio must be in (0, 1], got {args.ratio}")

    build_historical_qa(args.input, args.output, ratio=args.ratio)


if __name__ == "__main__":
    main()

