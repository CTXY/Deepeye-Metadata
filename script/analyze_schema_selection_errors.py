#!/usr/bin/env python3
"""
Analyze schema selection errors for a result JSON file.

Given:
  - A result JSON (question_id -> predicted SQL), e.g. qwen3-coder-30b-a3b.json
  - The BIRD dev dataset (dev.json)

This script:
  1. Reuses `calculate_ex_accuracy.calculate_ex_accuracy` to find EX-incorrect SQLs.
  2. Uses `SQLPreprocessor` (sqlglot-based) to normalize SQLs and remove aliases.
  3. Extracts the set of involved tables from gold and predicted SQL.
  4. Treats a prediction as a "schema selection error" if the set of tables
     used in the predicted SQL (after alias replacement) differs from that of
     the gold SQL.

Output:
  - A JSON file alongside the input result JSON:
      <stem>_schema_selection_errors.json
    with the following structure:
      {
        "total_incorrect": int,
        "schema_selection_error_count": int,
        "items": [
          {
            "question_id": int,
            "pred_sql": str,
            "gold_sql": str,
            "pred_tables": [str],
            "gold_tables": [str],
            "db_path": str
          },
          ...
        ]
      }
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import expressions
from importlib.machinery import SourceFileLoader
from importlib.util import spec_from_loader, module_from_spec
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from caf.preprocess.sql_preprocessor import SQLPreprocessor


def _load_execution_module():
    """
    Dynamically load app/db_utils/execution.py without importing app.db_utils,
    so that we do not trigger app/db_utils/__init__.py and its dependency on schema.py.
    """
    exec_path = PROJECT_ROOT / "app" / "db_utils" / "execution.py"
    loader = SourceFileLoader("de_execution", str(exec_path))
    spec = spec_from_loader(loader.name, loader)
    module = module_from_spec(spec)
    loader.exec_module(module)
    return module


_execution_module = _load_execution_module()
execute_sql = _execution_module.execute_sql


def _eval_ex_after_selection(pred_sql: str, gold_sql: str, db_path: str) -> Optional[int]:
    """
    Evaluate execution accuracy by comparing execution results of predicted SQL and gold SQL.
    Returns:
        1 if execution results match,
        0 if they don't,
        None if gold SQL execution failed.
    """
    pred_result = execute_sql(db_path, pred_sql)
    gold_result = execute_sql(db_path, gold_sql)

    if gold_result.result_rows is None:
        return None
    if pred_result.result_rows is None:
        return 0

    # Compare results ignoring order
    pred_result_set = set(map(frozenset, pred_result.result_rows))
    gold_result_set = set(map(frozenset, gold_result.result_rows))

    return 1 if pred_result_set == gold_result_set else 0


def _load_bird_dev_items(dataset_root_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Lightweight loader for BIRD dev data that avoids app.dataset.Dataset
    (and thus avoids importing app.db_utils.schema).

    Returns:
        dict: question_id -> {
            "gold_sql": str,
            "db_path": str,
            "question": str,
        }
    """
    dataset_root = Path(dataset_root_path)
    dev_json_path = dataset_root / "dev" / "dev.json"
    if not dev_json_path.exists():
        raise FileNotFoundError(f"BIRD dev.json not found: {dev_json_path}")

    with open(dev_json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    items: Dict[int, Dict[str, Any]] = {}
    for data_item in data_list:
        qid = int(data_item["question_id"])
        db_id = data_item["db_id"]
        gold_sql = data_item["SQL"]
        question = data_item.get("question", "")

        db_path = dataset_root / "dev" / "dev_databases" / db_id / f"{db_id}.sqlite"

        items[qid] = {
            "gold_sql": gold_sql,
            "db_path": str(db_path),
            "question": question,
        }

    return items


def _extract_tables_from_sql(sql: str) -> Set[str]:
    """
    Extract table names from a SQL string using sqlglot.

    The SQL is assumed to have aliases already removed / replaced
    (i.e., `FROM Player AS T1` -> `FROM Player`).
    """
    tables: Set[str] = set()

    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite", error_level=sqlglot.ErrorLevel.IGNORE)
        if not ast:
            return tables

        for node in ast.find_all(expressions.Table):
            # `node.name` returns unqualified table name for simple sqlite queries
            name = node.name
            if name:
                tables.add(name.lower())
    except Exception:
        # On any parsing error, just return whatever we have collected so far
        return tables

    return tables


def _analyze_schema_selection_errors(
    result_json_path: str,
    dataset_root_path: str | None,
    n_parallel: int,
) -> Dict:
    """
    Run EX evaluation and then analyze which incorrect SQLs are due to schema selection errors.
    """
    # 1) Load prediction result JSON
    result_path = Path(result_json_path)
    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_json_path}")

    print(f"Loading result file: {result_json_path}")
    with open(result_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)

    # 2) Determine dataset path (BIRD)
    if dataset_root_path is None:
        dataset_root_path = str(PROJECT_ROOT / "dataset" / "bird")

    print(f"Loading BIRD dev dataset from: {dataset_root_path}")
    dev_items = _load_bird_dev_items(dataset_root_path)

    # 3) Build evaluation tasks
    evaluation_tasks: List[Dict[str, Any]] = []
    missing_questions: List[int] = []

    for question_id_str, pred_sql in result_data.items():
        try:
            question_id = int(question_id_str)
        except ValueError:
            print(f"Warning: Invalid question_id '{question_id_str}', skipping...")
            continue

        if question_id not in dev_items:
            missing_questions.append(question_id)
            continue

        data_item = dev_items[question_id]
        evaluation_tasks.append(
            {
                "question_id": question_id,
                "pred_sql": pred_sql,
                "gold_sql": data_item["gold_sql"],
                "db_path": data_item["db_path"],
                "question": data_item["question"],
            }
        )

    if missing_questions:
        print(
            f"Warning: {len(missing_questions)} questions not found in dev.json: "
            f"{missing_questions[:10]}..."
        )

    if not evaluation_tasks:
        raise ValueError("No valid evaluation tasks found!")

    print(
        f"Evaluating {len(evaluation_tasks)} SQL queries with {n_parallel} parallel workers..."
    )

    # 4) Evaluate in parallel (same semantics as in calculate_ex_accuracy)
    executor = ProcessPoolExecutor(max_workers=n_parallel)
    future_to_task: Dict[Any, Dict[str, Any]] = {}
    for task in evaluation_tasks:
        future = executor.submit(
            _eval_ex_after_selection,
            task["pred_sql"],
            task["gold_sql"],
            task["db_path"],
        )
        future_to_task[future] = task

    incorrect_sqls: List[Dict[str, Any]] = []
    results: List[int] = []

    for future in tqdm(
        as_completed(future_to_task),
        total=len(future_to_task),
        desc="Evaluating SQL",
    ):
        task = future_to_task[future]
        result = future.result()
        if result is not None:
            results.append(result)
            if result == 0:
                incorrect_sqls.append(
                    {
                        "question_id": task["question_id"],
                        "pred_sql": task["pred_sql"],
                        "gold_sql": task["gold_sql"],
                        "db_path": task["db_path"],
                        "question": task["question"],
                    }
                )

    executor.shutdown(wait=True)

    sql_preprocessor = SQLPreprocessor(case_sensitive=False)

    schema_selection_error_items: List[Dict] = []

    for item in incorrect_sqls:
        pred_sql = item["pred_sql"]
        gold_sql = item["gold_sql"]

        # Normalize & remove aliases using SQLPreprocessor
        norm_pred_sql, _, _ = sql_preprocessor.preprocess(pred_sql)
        norm_gold_sql, _, _ = sql_preprocessor.preprocess(gold_sql)

        pred_tables = _extract_tables_from_sql(norm_pred_sql)
        gold_tables = _extract_tables_from_sql(norm_gold_sql)

        # If we cannot extract any tables from either side, we skip classification
        if not pred_tables or not gold_tables:
            continue

        # Schema selection error: table set mismatch (ignoring order / alias)
        if pred_tables != gold_tables:
            schema_selection_error_items.append(
                {
                    "question_id": item["question_id"],
                    "question": item.get("question", ""),
                    "pred_sql": pred_sql,
                    "gold_sql": gold_sql,
                    "pred_tables": sorted(list(pred_tables)),
                    "gold_tables": sorted(list(gold_tables)),
                    "db_path": str(item["db_path"]),
                }
            )

    result = {
        "total_incorrect": len(incorrect_sqls),
        "schema_selection_error_count": len(schema_selection_error_items),
        "items": schema_selection_error_items,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze schema selection errors among EX-incorrect SQLs."
    )
    parser.add_argument(
        "result_json_path",
        type=str,
        help="Path to the JSON file containing predicted SQL queries "
        "(e.g., results/bird-dev/qwen3-coder-30b-a3b.json)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Root path of the BIRD dataset (default: dataset/bird)",
    )
    parser.add_argument(
        "--n-parallel",
        type=int,
        default=16,
        help="Number of parallel workers for EX evaluation (default: 16)",
    )

    args = parser.parse_args()

    analysis_result = _analyze_schema_selection_errors(
        result_json_path=args.result_json_path,
        dataset_root_path=args.dataset_path,
        n_parallel=args.n_parallel,
    )

    result_path = Path(args.result_json_path)
    output_path = result_path.parent / f"{result_path.stem}_schema_selection_errors.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("Schema Selection Error Analysis")
    print("=" * 60)
    print(f"Result file: {args.result_json_path}")
    print(f"Total EX-incorrect SQLs: {analysis_result['total_incorrect']}")
    print(
        f"Schema selection errors (table set mismatch): "
        f"{analysis_result['schema_selection_error_count']}"
    )
    print(f"\nDetailed items saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()


