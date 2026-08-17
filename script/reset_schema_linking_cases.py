#!/usr/bin/env python3
"""
Inspect or reset schema-linking cases from a saved schema_linking dataset.

Examples:
  python script/reset_schema_linking_cases.py --input workspace/schema_linking/bird/dev.pkl --list-failed

  python script/reset_schema_linking_cases.py \
    --input workspace/schema_linking/bird/dev.pkl \
    --filter reversed-parse-failure \
    --output workspace/schema_linking/bird/dev_reversed_failures_reset.pkl \
    --reset
"""

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.dataset import load_dataset, save_dataset  # noqa: E402


FILTER_CHOICES = [
    "reversed-parse-failure",
    "direct-parse-failure",
    "any-parse-failure",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or reset schema-linking cases by debug markers.")
    parser.add_argument("--input", required=True, help="Input schema_linking dataset pkl path.")
    parser.add_argument("--output", help="Output dataset pkl path. Required with --reset.")
    parser.add_argument("--filter", choices=FILTER_CHOICES, default="reversed-parse-failure")
    parser.add_argument(
        "--question-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit question_id list to inspect/reset. When provided, this overrides --filter.",
    )
    parser.add_argument("--list-failed", action="store_true", help="Print matching cases.")
    parser.add_argument("--reset", action="store_true", help="Clear schema-linking fields for matching cases.")
    parser.add_argument("--limit", type=int, default=50, help="Max rows to print with --list-failed.")
    return parser.parse_args()


def _get_debug_value(item: Any, linker_key: str) -> int:
    debug_info = getattr(item, "schema_linking_debug", None) or {}
    linker_debug = debug_info.get(linker_key, {}) or {}
    return int(linker_debug.get("parse_failure_count", 0) or 0)


def _matches(item: Any, filter_name: str) -> bool:
    reversed_count = _get_debug_value(item, "reversed_linking")
    direct_count = _get_debug_value(item, "direct_linking")
    if filter_name == "reversed-parse-failure":
        return reversed_count > 0
    if filter_name == "direct-parse-failure":
        return direct_count > 0
    if filter_name == "any-parse-failure":
        return reversed_count > 0 or direct_count > 0
    raise ValueError(f"Unsupported filter: {filter_name}")


def _reset_schema_linking_fields(item: Any) -> None:
    schema_time = getattr(item, "schema_linking_time", None) or 0.0
    total_time = getattr(item, "total_time", None)
    if total_time is not None:
        item.total_time = max(0.0, total_time - schema_time)

    schema_cost = getattr(item, "schema_linking_llm_cost", None) or {}
    total_cost = getattr(item, "total_llm_cost", None)
    if total_cost is not None:
        item.total_llm_cost = {
            "prompt_tokens": max(0, total_cost.get("prompt_tokens", 0) - schema_cost.get("prompt_tokens", 0)),
            "completion_tokens": max(0, total_cost.get("completion_tokens", 0) - schema_cost.get("completion_tokens", 0)),
            "total_tokens": max(0, total_cost.get("total_tokens", 0) - schema_cost.get("total_tokens", 0)),
        }

    item.direct_linked_tables_and_columns = None
    item.reversed_linked_tables_and_columns = None
    item.reversed_linking_sql_candidates = None
    item.value_linked_tables_and_columns = None
    item.final_linked_tables_and_columns = None
    item.database_schema_after_schema_linking = None
    item.direct_linking_recall = None
    item.reversed_linking_recall = None
    item.value_linking_recall = None
    item.final_linking_recall = None
    item.schema_linking_time = None
    item.schema_linking_llm_cost = None


def main() -> None:
    args = _parse_args()
    if args.reset and not args.output:
        raise ValueError("--output is required with --reset")

    dataset = load_dataset(args.input)
    if args.question_ids:
        question_id_set = set(args.question_ids)
        matches = [item for item in dataset if item.question_id in question_id_set]
        filter_label = f"question_ids={sorted(question_id_set)}"
    else:
        matches = [item for item in dataset if _matches(item, args.filter)]
        filter_label = args.filter

    print(f"matched_cases={len(matches)} filter={filter_label}")
    if args.list_failed:
        for item in matches[: args.limit]:
            print(
                f"question_id={item.question_id} db={item.database_id} "
                f"direct_failures={_get_debug_value(item, 'direct_linking')} "
                f"reversed_failures={_get_debug_value(item, 'reversed_linking')}"
            )

    if args.reset:
        for item in matches:
            _reset_schema_linking_fields(item)
        save_dataset(dataset, args.output)
        print(f"saved_reset_dataset={args.output}")


if __name__ == "__main__":
    main()
