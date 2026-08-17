#!/usr/bin/env python3
"""
Filter dataset by question ID range and save a subset for use in SQL generation etc.

Use cases:
  - Keep latter 50% of samples (by question_id order): --percentile 50-100
  - Per database: keep latter 50% within each database (e.g. California 44-88, card_games its own latter 50%%): --percentile 50-100 --per-database
  - Keep only a specific ID range (global): --id-min M --id-max N
  - Load from any pkl in the pipeline (dataset, schema_linking, etc.) and write filtered pkl

Examples:
  # Latter 50% per database (each DB keeps its own latter 50%% by question_id)
  python script/filter_dataset_by_question_id.py --input workspace/dataset/cleaned-mini-bird/schools_cards.pkl --percentile 50-100 --per-database -o workspace/dataset/cleaned-mini-bird/schools_cards_latter50.pkl

  # Global latter 50%%
  python script/filter_dataset_by_question_id.py --input workspace/dataset/bird/sub_dev_school_and_card.pkl --percentile 50-100 -o workspace/dataset/bird/sub_dev_school_and_card_latter50.pkl

  # Explicit ID range (global)
  python script/filter_dataset_by_question_id.py -i workspace/dataset/bird/sub_dev_school_and_card.pkl --id-range 500 999 -o workspace/dataset/bird/sub_dev_filtered.pkl
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

# project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.dataset import load_dataset, save_dataset
from app.dataset.dataset import BaseDataset
from app.logger import logger


def _parse_percentile(s: str) -> tuple[float, float]:
    low, high = s.strip().split("-")
    low_f = float(low.strip())
    high_f = float(high.strip())
    if not (0 <= low_f < high_f <= 100):
        raise ValueError(f"Invalid percentile range: {s}. Use e.g. 50-100 for latter 50%.")
    return low_f, high_f


def filter_by_percentile(dataset: BaseDataset, low_pct: float, high_pct: float, per_database: bool = False) -> list:
    """Return items in [low_pct, high_pct] percentile by question_id order. If per_database, apply within each database_id."""
    data = list(dataset._data)
    if not data:
        return []

    if per_database:
        by_db: dict[str, list] = defaultdict(list)
        for item in data:
            by_db[item.database_id].append(item)
        result = []
        for db_id in sorted(by_db.keys()):
            items = sorted(by_db[db_id], key=lambda x: x.question_id)
            n = len(items)
            low_idx = int(n * (low_pct / 100.0))
            high_idx = int(n * (high_pct / 100.0))
            if high_pct >= 100:
                high_idx = n
            result.extend(items[low_idx:high_idx])
        return sorted(result, key=lambda x: (x.database_id, x.question_id))
    else:
        data_sorted = sorted(data, key=lambda x: x.question_id)
        n = len(data_sorted)
        low_idx = int(n * (low_pct / 100.0))
        high_idx = int(n * (high_pct / 100.0))
        if high_idx < n and high_pct >= 100:
            high_idx = n
        return data_sorted[low_idx:high_idx]


def filter_by_id_range(dataset: BaseDataset, id_min: int | None, id_max: int | None) -> list:
    """Return items with question_id in [id_min, id_max] (inclusive)."""
    data = dataset._data
    if id_min is not None:
        data = [x for x in data if x.question_id >= id_min]
    if id_max is not None:
        data = [x for x in data if x.question_id <= id_max]
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Filter dataset by question ID range (percentile or explicit) and save subset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-i", "--input",
        default=None,
        help="Input dataset pkl path. Default: config dataset save_path.",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output dataset pkl path (filtered subset).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--percentile",
        metavar="LOW-HIGH",
        help="Keep items in this percentile range by question_id order. E.g. 50-100 for latter 50%%.",
    )
    parser.add_argument(
        "--per-database",
        action="store_true",
        help="Apply percentile within each database separately (e.g. each DB keeps its latter 50%%).",
    )
    group.add_argument(
        "--id-range",
        nargs=2,
        metavar=("ID_MIN", "ID_MAX"),
        type=int,
        help="Keep items with question_id in [ID_MIN, ID_MAX] (inclusive).",
    )
    args = parser.parse_args()
    if args.per_database and args.id_range is not None:
        parser.error("--per-database only applies to --percentile, not --id-range")

    input_path = args.input
    if input_path is None:
        from app.config import config
        input_path = config.dataset_config.save_path
        logger.info(f"Using config dataset save_path: {input_path}")

    dataset = load_dataset(input_path)
    n_orig = len(dataset)

    if args.percentile:
        low_pct, high_pct = _parse_percentile(args.percentile)
        filtered = filter_by_percentile(dataset, low_pct, high_pct, per_database=args.per_database)
        mode = "per-database" if args.per_database else "global"
        logger.info(f"Percentile {low_pct}-{high_pct} ({mode}): kept {len(filtered)} / {n_orig} items")
    else:
        id_min, id_max = args.id_range
        filtered = filter_by_id_range(dataset, id_min, id_max)
        logger.info(f"ID range [{id_min}, {id_max}]: kept {len(filtered)} / {n_orig} items")

    if not filtered:
        logger.warning("No items left after filter. Aborting.")
        sys.exit(1)

    qids = [x.question_id for x in filtered]
    logger.info(f"Question ID range (global): min={min(qids)}, max={max(qids)}")
    by_db = defaultdict(list)
    for x in filtered:
        by_db[x.database_id].append(x.question_id)
    for db_id in sorted(by_db.keys()):
        ids = sorted(by_db[db_id])
        logger.info(f"  {db_id}: {len(ids)} items, question_id range [{min(ids)}, {max(ids)}]")

    # Replace _data with filtered list and save (same type of dataset object)
    dataset._data = filtered
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(dataset, str(out_path))
    logger.info(f"Saved filtered dataset to {out_path} ({len(filtered)} items)")
    print("Next: point config dataset.save_path to this file and re-run pipeline from schema_linking (or preprocess) so SQL generation uses this subset.")


if __name__ == "__main__":
    main()
