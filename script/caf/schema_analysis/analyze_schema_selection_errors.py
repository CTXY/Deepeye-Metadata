#!/usr/bin/env python3
"""
Analyze schema selection errors for a result JSON file.

Given:
  - A result JSON (question_id -> predicted SQL), e.g. qwen3-coder-30b-a3b.json
  - The BIRD dev dataset (dev.json)

This script:
  1. Reuses `calculate_ex_accuracy.calculate_ex_accuracy` to find EX-incorrect SQLs.
  2. Uses `SQLPreprocessor` (sqlglot-based) to normalize SQLs and remove aliases.
  3. Extracts the set of involved tables and columns from gold and predicted SQL.
  4. Treats a prediction as a "schema selection error" if:
     - The set of tables used in the predicted SQL differs from that of the gold SQL, OR
     - The set of columns used in the predicted SQL differs from that of the gold SQL.

Output:
  - A JSON file alongside the input result JSON:
      <stem>_schema_selection_errors.json
    with the following structure:
      {
        "total_incorrect": int,
        "schema_selection_error_count": int,
        "table_mismatch_count": int,
        "column_mismatch_count": int,
        "table_only_errors": int,
        "column_only_errors": int,
        "both_table_and_column_errors": int,
        "items": [
          {
            "question_id": int,
            "pred_sql": str,
            "gold_sql": str,
            "pred_tables": [str],
            "gold_tables": [str],
            "pred_columns": [str],
            "gold_columns": [str],
            "table_mismatch": bool,
            "column_mismatch": bool,
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
# __file__ is script/caf/analyze_schema_selection_errors.py
# parent = script/caf, parent.parent = script, parent.parent.parent = project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from caf.preprocess.sql_preprocessor import SQLPreprocessor
import sqlite3


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


def _load_database_schema(db_path: str) -> Dict[str, Set[str]]:
    """
    Load database schema: column_name -> set of table names that contain this column.
    
    Returns:
        Dict mapping column_name (lowercase) to set of table names (lowercase) that contain it.
    """
    schema: Dict[str, Set[str]] = {}
    
    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence';")
        table_names = [row[0] for row in cursor.fetchall()]
        
        # For each table, get its columns
        for table_name in table_names:
            cursor.execute(f"PRAGMA table_info(`{table_name}`)")
            columns = cursor.fetchall()
            
            for col_info in columns:
                col_name = col_info[1]  # Column name is at index 1
                col_name_lower = col_name.lower()
                
                if col_name_lower not in schema:
                    schema[col_name_lower] = set()
                schema[col_name_lower].add(table_name.lower())
        
        conn.close()
    except Exception as e:
        # On error, return empty schema
        print(f"Warning: Failed to load schema from {db_path}: {e}")
        return {}
    
    return schema


def _extract_columns_from_sql(sql: str, db_path: Optional[str] = None, schema: Optional[Dict[str, Set[str]]] = None, sql_tables: Optional[Set[str]] = None) -> Set[str]:
    """
    Extract column names from a SQL string using sqlglot.
    
    Extracts columns from:
    - SELECT clause
    - WHERE clause
    - JOIN conditions
    - ORDER BY clause
    - GROUP BY clause
    - HAVING clause
    
    Returns a set of column names (lowercased) in the format "table.column" or just "column"
    if table is not specified.
    
    The SQL is assumed to have aliases already removed / replaced.
    """
    columns: Set[str] = set()
    
    # Load schema if db_path provided but schema not provided
    if db_path and schema is None:
        schema = _load_database_schema(db_path)

    try:
        ast = sqlglot.parse_one(sql, dialect="sqlite", error_level=sqlglot.ErrorLevel.IGNORE)
        if not ast:
            return columns

        for node in ast.find_all(expressions.Column):
            # Extract column name
            col_name = node.name
            if not col_name:
                continue
            
            col_name_lower = col_name.lower()
            
            # Extract table name if present
            # node.table can be a string or an Identifier/Table object
            table_name = None
            if node.table:
                if isinstance(node.table, str):
                    table_name = node.table
                elif hasattr(node.table, 'name'):
                    table_name = node.table.name
                elif hasattr(node.table, 'this'):
                    # Handle Identifier objects
                    table_name = node.table.this if isinstance(node.table.this, str) else str(node.table.this)
                else:
                    # Fallback: convert to string
                    table_name = str(node.table)
            
            # If table name is not present, try to resolve from schema
            if not table_name and schema:
                possible_tables = schema.get(col_name_lower, set())
                
                if len(possible_tables) == 1:
                    # Only one table has this column
                    table_name = list(possible_tables)[0]
                elif len(possible_tables) > 1 and sql_tables:
                    # Multiple tables have this column - prefer tables used in SQL
                    matching_tables = possible_tables.intersection(sql_tables)
                    if len(matching_tables) == 1:
                        table_name = list(matching_tables)[0]
                    elif len(matching_tables) > 1:
                        # Still ambiguous, use first matching table
                        table_name = list(matching_tables)[0]
                    elif len(matching_tables) == 0 and len(possible_tables) > 0:
                        # No match with SQL tables, but we have possible tables from schema
                        # Use first possible table to ensure we have table.column format
                        table_name = list(possible_tables)[0]
                elif len(possible_tables) > 1:
                    # Multiple tables but no SQL context, use first to ensure format consistency
                    table_name = list(possible_tables)[0]
            
            # Always use "table.column" format if we have table_name
            # If no table_name, we'll handle it in normalization
            if table_name:
                col_key = f"{table_name.lower()}.{col_name_lower}"
                columns.add(col_key)
            else:
                # No table name found - add as "column" format
                # This will be normalized later in _normalize_column_set
                columns.add(col_name_lower)
    except Exception:
        # On any parsing error, just return whatever we have collected so far
        return columns

    return columns


def _normalize_column_set(columns: Set[str], schema: Optional[Dict[str, Set[str]]] = None, sql_tables: Optional[Set[str]] = None) -> Set[str]:
    """
    Normalize a set of columns to ensure consistent "table.column" format.
    
    For columns without table names (just "column"), try to resolve from schema.
    If a column appears in multiple tables and sql_tables is provided, prefer tables used in SQL.
    
    Args:
        columns: Set of column names (may include "table.column" or "column" format)
        schema: Optional dict mapping column_name -> set of table names
        sql_tables: Optional set of table names used in SQL (for disambiguation)
    
    Returns:
        Normalized set of columns in "table.column" format.
        If table cannot be determined, keeps "column" format (should be rare).
    """
    normalized: Set[str] = set()
    
    for col in columns:
        if '.' in col:
            # Already in "table.column" format
            normalized.add(col)
        else:
            # Column without table name - try to resolve
            col_lower = col.lower()
            table_name = None
            
            if schema:
                possible_tables = schema.get(col_lower, set())
                
                if len(possible_tables) == 1:
                    table_name = list(possible_tables)[0]
                elif len(possible_tables) > 1 and sql_tables:
                    # Multiple tables - prefer tables used in SQL
                    matching_tables = possible_tables.intersection(sql_tables)
                    if len(matching_tables) == 1:
                        table_name = list(matching_tables)[0]
                    elif len(matching_tables) > 1:
                        # Still ambiguous, use first matching table
                        table_name = list(matching_tables)[0]
                    elif len(matching_tables) == 0 and len(possible_tables) > 0:
                        # No match with SQL tables, use first possible table
                        table_name = list(possible_tables)[0]
                elif len(possible_tables) > 1:
                    # Multiple tables but no SQL context, use first
                    table_name = list(possible_tables)[0]
            
            if table_name:
                normalized.add(f"{table_name.lower()}.{col_lower}")
            else:
                # Cannot resolve table name, keep original format
                # This should be rare and indicates a potential issue
                normalized.add(col_lower)
    
    return normalized


def _compare_column_sets(pred_columns: Set[str], gold_columns: Set[str], 
                         schema: Optional[Dict[str, Set[str]]] = None,
                         pred_tables: Optional[Set[str]] = None,
                         gold_tables: Optional[Set[str]] = None) -> Tuple[bool, Set[str], Set[str]]:
    """
    Compare two column sets with proper normalization and equivalence checking.
    
    Handles cases where:
    - Some columns have table names, some don't
    - Same column might be represented differently (e.g., "gender" vs "account.gender")
    
    Args:
        pred_columns: Predicted columns set
        gold_columns: Gold columns set
        schema: Optional schema for column resolution
        pred_tables: Tables used in predicted SQL
        gold_tables: Tables used in gold SQL
    
    Returns:
        Tuple of (is_match, normalized_pred_columns, normalized_gold_columns)
    """
    # Normalize both sets to "table.column" format
    norm_pred = _normalize_column_set(pred_columns, schema=schema, sql_tables=pred_tables)
    norm_gold = _normalize_column_set(gold_columns, schema=schema, sql_tables=gold_tables)
    
    # Direct comparison
    if norm_pred == norm_gold:
        return True, norm_pred, norm_gold
    
    # If direct comparison fails, check for equivalence
    # A column "column" is equivalent to "table.column" if that's the only table with that column
    # or if the table is in the SQL's table set
    if schema:
        # Build expanded sets: for columns without table, add all possible "table.column" variants
        expanded_pred = set(norm_pred)
        expanded_gold = set(norm_gold)
        
        for col in norm_pred:
            if '.' not in col:
                # Column without table - add all possible table.column combinations
                possible_tables = schema.get(col, set())
                if pred_tables:
                    # Prefer tables used in SQL
                    matching = possible_tables.intersection(pred_tables)
                    if matching:
                        expanded_pred.update(f"{t}.{col}" for t in matching)
                    else:
                        expanded_pred.update(f"{t}.{col}" for t in possible_tables)
                else:
                    expanded_pred.update(f"{t}.{col}" for t in possible_tables)
        
        for col in norm_gold:
            if '.' not in col:
                # Column without table - add all possible table.column combinations
                possible_tables = schema.get(col, set())
                if gold_tables:
                    # Prefer tables used in SQL
                    matching = possible_tables.intersection(gold_tables)
                    if matching:
                        expanded_gold.update(f"{t}.{col}" for t in matching)
                    else:
                        expanded_gold.update(f"{t}.{col}" for t in possible_tables)
                else:
                    expanded_gold.update(f"{t}.{col}" for t in possible_tables)
        
        # Check if expanded sets match
        if expanded_pred == expanded_gold:
            return True, norm_pred, norm_gold
    
    # Not equivalent
    return False, norm_pred, norm_gold


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

        # Extract tables first
        pred_tables = _extract_tables_from_sql(norm_pred_sql)
        gold_tables = _extract_tables_from_sql(norm_gold_sql)
        
        # Load database schema for column resolution
        db_path = item["db_path"]
        schema = _load_database_schema(db_path)
        
        # Extract columns with schema information for better resolution
        pred_columns = _extract_columns_from_sql(norm_pred_sql, db_path=db_path, schema=schema, sql_tables=pred_tables)
        gold_columns = _extract_columns_from_sql(norm_gold_sql, db_path=db_path, schema=schema, sql_tables=gold_tables)

        # If we cannot extract any tables from either side, we skip classification
        if not pred_tables or not gold_tables:
            continue

        # Check for schema selection errors: table mismatch or column mismatch
        table_mismatch = pred_tables != gold_tables
        
        # Use normalized column comparison to handle format inconsistencies
        # (some columns may have "table.column" format, some may just be "column")
        columns_match, norm_pred_columns, norm_gold_columns = _compare_column_sets(
            pred_columns, gold_columns, 
            schema=schema,
            pred_tables=pred_tables,
            gold_tables=gold_tables
        )
        column_mismatch = not columns_match

        # Schema selection error: table set mismatch or column set mismatch
        if table_mismatch or column_mismatch:
            error_item = {
                "question_id": item["question_id"],
                "question": item.get("question", ""),
                "pred_sql": pred_sql,
                "gold_sql": gold_sql,
                "pred_tables": sorted(list(pred_tables)),
                "gold_tables": sorted(list(gold_tables)),
                "pred_columns": sorted(list(norm_pred_columns)),  # Use normalized columns
                "gold_columns": sorted(list(norm_gold_columns)),  # Use normalized columns
                "table_mismatch": table_mismatch,
                "column_mismatch": column_mismatch,
                "db_path": str(item["db_path"]),
            }
            schema_selection_error_items.append(error_item)

    # Calculate statistics
    table_only_errors = sum(1 for item in schema_selection_error_items if item["table_mismatch"] and not item["column_mismatch"])
    column_only_errors = sum(1 for item in schema_selection_error_items if item["column_mismatch"] and not item["table_mismatch"])
    both_errors = sum(1 for item in schema_selection_error_items if item["table_mismatch"] and item["column_mismatch"])

    result = {
        "total_incorrect": len(incorrect_sqls),
        "schema_selection_error_count": len(schema_selection_error_items),
        "table_mismatch_count": sum(1 for item in schema_selection_error_items if item["table_mismatch"]),
        "column_mismatch_count": sum(1 for item in schema_selection_error_items if item["column_mismatch"]),
        "table_only_errors": table_only_errors,
        "column_only_errors": column_only_errors,
        "both_table_and_column_errors": both_errors,
        "items": schema_selection_error_items,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze schema selection errors among EX-incorrect SQLs."
    )
    parser.add_argument(
        "--result_json_path",
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
        f"Schema selection errors (table or column mismatch): "
        f"{analysis_result['schema_selection_error_count']}"
    )
    print(f"  - Table mismatch: {analysis_result['table_mismatch_count']}")
    print(f"  - Column mismatch: {analysis_result['column_mismatch_count']}")
    print(f"  - Table only errors: {analysis_result['table_only_errors']}")
    print(f"  - Column only errors: {analysis_result['column_only_errors']}")
    print(f"  - Both table and column errors: {analysis_result['both_table_and_column_errors']}")
    print(f"\nDetailed items saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()


