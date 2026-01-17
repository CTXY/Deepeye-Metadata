#!/usr/bin/env python3
"""
Quick viewer for episodic memory records stored on disk.

Examples:
    python view_episodic_memory.py --list-databases
    python script/caf/view_episodic_memory.py --database-id toxicology --limit 3
    python view_episodic_memory.py --database-id financial --mode errors --as-json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Set
import pandas as pd

# Ensure project root is on sys.path so we can import `caf`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from caf.config.loader import CAFConfig
from caf.config.paths import PathConfig
from caf.memory.stores.episodic import EpisodicMemoryStore
from caf.memory.types import EpisodicRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect stored episodic memory records."
    )
    parser.add_argument(
        "--database-id",
        help="Target database ID (stored in unified parquet files).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of records to show (default: 5).",
    )
    parser.add_argument(
        "--mode",
        choices=["recent", "success", "errors", "all"],
        default="recent",
        help="Record filter: recent(all), success, or errors (default: recent).",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort timestamps ascending instead of descending.",
    )
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Print records as raw JSON instead of a compact summary.",
    )
    parser.add_argument(
        "--config",
        default=str(PathConfig.PROJECT_ROOT / "config" / "caf_config.yaml"),
        help="Optional path to CAF config (default: config/caf_config.yaml).",
    )
    parser.add_argument(
        "--list-databases",
        action="store_true",
        help="List available episodic memory databases and exit.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> CAFConfig:
    path = Path(config_path)
    if path.exists():
        return CAFConfig.from_file(str(path))
    # Fallback to defaults if file missing
    return CAFConfig.default()


def list_databases(storage_path: Path) -> List[str]:
    """
    List all available database IDs from parquet storage files.
    New storage format: all databases stored in unified records_*.parquet files.
    """
    if not storage_path.exists():
        return []
    
    database_ids: Set[str] = set()
    
    # Load from all parquet files in storage path
    try:
        for parquet_file in storage_path.glob("records_*.parquet"):
            try:
                df = pd.read_parquet(parquet_file)
                if df is not None and not df.empty and 'database_id' in df.columns:
                    # Get unique database_ids from this file
                    unique_ids = df['database_id'].dropna().unique()
                    database_ids.update(unique_ids)
            except Exception as e:
                print(f"Warning: Failed to read {parquet_file}: {e}", file=sys.stderr)
                continue
    except Exception as e:
        print(f"Warning: Error scanning storage path: {e}", file=sys.stderr)
    
    return sorted(list(database_ids))


def fetch_records(
    store: EpisodicMemoryStore, database_id: str, mode: str
) -> Iterable[EpisodicRecord]:
    """
    Fetch records from store and filter by mode.
    Note: _load_successful_records and _load_error_records don't exist,
    so we filter after loading all records.
    """
    # Load all records for the database
    all_records = store._load_all_records(database_id)  # type: ignore[attr-defined]
    
    mode = mode.lower()
    if mode == "success":
        # Filter for successful records (label=True or execution_success=True)
        return [
            r for r in all_records
            if (r.label is True) or 
               (r.execution_result is not None and 
                getattr(r.execution_result, 'execution_success', None) is True)
        ]
    elif mode == "errors":
        # Filter for error records (label=False or execution_success=False)
        return [
            r for r in all_records
            if (r.label is False) or 
               (r.execution_result is not None and 
                getattr(r.execution_result, 'execution_success', None) is False)
        ]
    else:
        # mode == "recent" or "all" - return all records
        return all_records


def sort_records(records: Iterable[EpisodicRecord], ascending: bool) -> List[EpisodicRecord]:
    def parse_ts(ts: str) -> datetime:
        normalized = ts.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return datetime.min

    return sorted(records, key=lambda r: parse_ts(r.timestamp), reverse=not ascending)


def summarize_record(record: EpisodicRecord, index: int) -> str:
    exec_status = (
        record.execution_result.execution_success
        if record.execution_result is not None
        else None
    )
    label = "✅" if record.label else ("❌" if record.label is False else "❔")
    sql_preview = (record.generated_sql or "").strip().replace("\n", " ")
    if len(sql_preview) > 140:
        sql_preview = sql_preview[:137] + "..."
    user_query = record.user_query.strip().replace("\n", " ")
    if len(user_query) > 120:
        user_query = user_query[:117] + "..."
    return (
        f"[{index}] session={record.session_id} round={record.round_id} "
        f"timestamp={record.timestamp} label={label} exec_success={exec_status}\n"
        f"    user_query: {user_query}\n"
        f"    generated_sql: {sql_preview}\n"
    )


def main() -> None:
    args = parse_args()
    storage_path = PathConfig.EPISODIC_MEMORY_PATH

    if args.list_databases or not args.database_id:
        dbs = list_databases(storage_path)
        if not dbs:
            print(f"No episodic memory data found in {storage_path}")
        else:
            print("Available episodic memory databases:")
            for db in dbs:
                print(f"  - {db}")
        if args.list_databases or not args.database_id:
            return

    config = load_config(args.config)
    store = EpisodicMemoryStore(config.memory)

    target_db = args.database_id
    
    # Verify database exists
    available_dbs = list_databases(storage_path)
    if target_db not in available_dbs:
        print(
            f"Database '{target_db}' not found in storage files. "
            "Use --list-databases to see available options."
        )
        if available_dbs:
            print(f"Available databases: {', '.join(available_dbs)}")
        return

    store.bind_database(target_db)
    records = fetch_records(store, target_db, args.mode)
    sorted_records = sort_records(records, args.ascending)

    if not sorted_records:
        print(f"No records found for database '{target_db}' (mode={args.mode}).")
        return

    print(
        f"Showing top {min(args.limit, len(sorted_records))} "
        f"{args.mode} record(s) for database '{target_db}':"
    )
    for idx, record in enumerate(sorted_records[: args.limit], start=1):
        if args.as_json:
            print(json.dumps(record.model_dump(), ensure_ascii=False, indent=2))
        else:
            print(summarize_record(record, idx))


if __name__ == "__main__":
    main()

