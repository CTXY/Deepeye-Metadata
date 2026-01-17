#!/usr/bin/env python3
"""
Analyze schema linking recall rate.

Given:
  - A pickle file from schema linking stage (sub_dev.pkl) containing database_schema_after_schema_linking
  - The BIRD dev dataset (dev.json) containing gold SQL

This script:
  1. Extracts the retrieved schema from database_schema_after_schema_linking for each question
  2. Extracts the needed schema from gold SQL in dev.json
  3. Calculates recall rate for tables and columns
  4. Reports which needed schemas were not retrieved

Output:
  - A JSON file with recall statistics and missing schemas:
      <pickle_stem>_schema_linking_recall.json
    with the following structure:
      {
        "total_questions": int,
        "questions_with_errors": int,
        "table_recall": float,
        "column_recall": float,
        "table_level_stats": {
          "total_needed_tables": int,
          "total_retrieved_tables": int,
          "total_matched_tables": int
        },
        "column_level_stats": {
          "total_needed_columns": int,
          "total_retrieved_columns": int,
          "total_matched_columns": int
        },
        "items": [
          {
            "question_id": int,
            "question": str,
            "db_id": str,
            "needed_tables": [str],
            "retrieved_tables": [str],
            "missing_tables": [str],
            "needed_columns": [str],
            "retrieved_columns": [str],
            "missing_columns": [str],
            "table_recall": float,
            "column_recall": float
          },
          ...
        ]
      }
    Note: The "items" array only contains questions with recall < 1.0 
    (either table_recall < 1.0 or column_recall < 1.0).
"""

import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import expressions
import sqlite3

# Add project root to path
# __file__ is script/caf/schema_analysis/analyze_schema_linking_recall.py
# parent = script/caf/schema_analysis, parent.parent = script/caf, parent.parent.parent = script, parent.parent.parent.parent = project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from caf.preprocess.sql_preprocessor import SQLPreprocessor


def _load_bird_dev_items(dataset_root_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Load BIRD dev data.

    Returns:
        dict: question_id -> {
            "gold_sql": str,
            "db_path": str,
            "question": str,
            "db_id": str,
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
            "db_id": db_id,
        }

    return items


def _load_database_schema(db_path: str) -> Dict[str, Set[str]]:
    """
    Load database schema: column_name -> set of table names that contain this column.
    
    Returns:
        Dict mapping column_name (lowercase) to set of table names (lowercase) that contain it.
    """
    schema: Dict[str, Set[str]] = {}
    
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
    return schema


def _extract_tables_from_sql(sql: str) -> Set[str]:
    """
    Extract table names from a SQL string using sqlglot.

    The SQL is assumed to have aliases already removed / replaced
    (i.e., `FROM Player AS T1` -> `FROM Player`).
    """
    tables: Set[str] = set()

    ast = sqlglot.parse_one(sql, dialect="sqlite", error_level=sqlglot.ErrorLevel.IGNORE)
    if not ast:
        return tables

    for node in ast.find_all(expressions.Table):
        name = node.name
        if name:
            tables.add(name.lower())

    return tables


def _collect_column_aliases(ast: expressions.Expression) -> Set[str]:
    """
    Collect all column aliases defined in SELECT clauses.
    
    Returns:
        Set of alias names (lowercase) that are defined in SELECT clauses.
    """
    aliases: Set[str] = set()
    
    # Find all SELECT nodes
    for select_node in ast.find_all(expressions.Select):
        # Check each expression in SELECT list
        for expr in select_node.expressions:
            # Check if this expression has an alias
            if isinstance(expr, expressions.Alias):
                alias_name = expr.alias
                if alias_name:
                    if isinstance(alias_name, str):
                        aliases.add(alias_name.lower())
                    elif hasattr(alias_name, 'this'):
                        aliases.add(str(alias_name.this).lower())
                    else:
                        aliases.add(str(alias_name).lower())
    
    return aliases


def _extract_columns_from_sql(sql: str, db_path: Optional[str] = None, schema: Optional[Dict[str, Set[str]]] = None, sql_tables: Optional[Set[str]] = None) -> Set[str]:
    """
    Extract column names from a SQL string using sqlglot.
    
    Only extracts real database columns, skipping column aliases defined in SELECT clauses.
    Verifies that extracted columns exist in the database schema.
    
    Returns a set of column names (lowercased) in the format "table.column" or just "column"
    if table is not specified.
    
    The SQL is assumed to have aliases already removed / replaced.
    """
    columns: Set[str] = set()
    
    # Load schema if db_path provided but schema not provided
    if db_path and schema is None:
        schema = _load_database_schema(db_path)

    ast = sqlglot.parse_one(sql, dialect="sqlite", error_level=sqlglot.ErrorLevel.IGNORE)
    if not ast:
        return columns

    # Collect all column aliases defined in SELECT clauses
    column_aliases = _collect_column_aliases(ast)

    for node in ast.find_all(expressions.Column):
        col_name = node.name
        if not col_name:
            continue
        
        col_name_lower = col_name.lower()
        
        # Skip if this column name is actually an alias
        if col_name_lower in column_aliases:
            continue
        
        # Extract table name if present
        table_name = None
        if node.table:
            if isinstance(node.table, str):
                table_name = node.table
            elif hasattr(node.table, 'name'):
                table_name = node.table.name
            elif hasattr(node.table, 'this'):
                table_name = node.table.this if isinstance(node.table.this, str) else str(node.table.this)
            else:
                table_name = str(node.table)
        
        # If table name is not present, try to resolve from schema
        if not table_name and schema:
            possible_tables = schema.get(col_name_lower, set())
            
            if len(possible_tables) == 1:
                table_name = list(possible_tables)[0]
            elif len(possible_tables) > 1 and sql_tables:
                matching_tables = possible_tables.intersection(sql_tables)
                if len(matching_tables) == 1:
                    table_name = list(matching_tables)[0]
                elif len(matching_tables) > 1:
                    table_name = list(matching_tables)[0]
                elif len(matching_tables) == 0 and len(possible_tables) > 0:
                    table_name = list(possible_tables)[0]
            elif len(possible_tables) > 1:
                table_name = list(possible_tables)[0]
        
        # Verify that the column exists in the database schema
        # If schema is available, only add columns that exist in the schema
        if schema:
            if table_name:
                # Check if table.column exists in schema
                table_name_lower = table_name.lower()
                if col_name_lower in schema:
                    possible_tables_for_col = schema[col_name_lower]
                    if table_name_lower in possible_tables_for_col:
                        # Column exists in the specified table
                        col_key = f"{table_name_lower}.{col_name_lower}"
                        columns.add(col_key)
                    # else: column doesn't exist in the specified table, skip it
                # else: column doesn't exist in schema, skip it
            else:
                # No table name, check if column exists in any table
                if col_name_lower in schema:
                    # Column exists in schema, but we don't know which table
                    # Only add if there's exactly one table or if it's in one of the SQL tables
                    possible_tables = schema[col_name_lower]
                    if len(possible_tables) == 1:
                        table_name = list(possible_tables)[0]
                        col_key = f"{table_name.lower()}.{col_name_lower}"
                        columns.add(col_key)
                    elif sql_tables:
                        matching_tables = possible_tables.intersection(sql_tables)
                        if len(matching_tables) == 1:
                            table_name = list(matching_tables)[0]
                            col_key = f"{table_name.lower()}.{col_name_lower}"
                            columns.add(col_key)
        else:
            # No schema available, add column anyway (fallback behavior)
            if table_name:
                col_key = f"{table_name.lower()}.{col_name_lower}"
                columns.add(col_key)
            else:
                columns.add(col_name_lower)
    
    return columns


def _normalize_column_set(columns: Set[str], schema: Optional[Dict[str, Set[str]]] = None, sql_tables: Optional[Set[str]] = None) -> Set[str]:
    """
    Normalize a set of columns to ensure consistent "table.column" format.
    """
    normalized: Set[str] = set()
    
    for col in columns:
        if '.' in col:
            normalized.add(col)
        else:
            col_lower = col.lower()
            table_name = None
            
            if schema:
                possible_tables = schema.get(col_lower, set())
                
                if len(possible_tables) == 1:
                    table_name = list(possible_tables)[0]
                elif len(possible_tables) > 1 and sql_tables:
                    matching_tables = possible_tables.intersection(sql_tables)
                    if len(matching_tables) == 1:
                        table_name = list(matching_tables)[0]
                    elif len(matching_tables) > 1:
                        table_name = list(matching_tables)[0]
                    elif len(matching_tables) == 0 and len(possible_tables) > 0:
                        table_name = list(possible_tables)[0]
                elif len(possible_tables) > 1:
                    table_name = list(possible_tables)[0]
            
            if table_name:
                normalized.add(f"{table_name.lower()}.{col_lower}")
            else:
                normalized.add(col_lower)
    
    return normalized


def _extract_schema_from_database_schema_dict(database_schema_dict: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    """
    Extract tables and columns from database_schema_after_schema_linking.
    
    Args:
        database_schema_dict: The database_schema_after_schema_linking dict with structure:
            {
                "tables": {
                    "table_name": {
                        "columns": {
                            "column_name": {...}
                        }
                    }
                }
            }
    
    Returns:
        Tuple of (tables_set, columns_set) where:
        - tables_set: set of table names (lowercase)
        - columns_set: set of columns in "table.column" format (lowercase)
    """
    tables: Set[str] = set()
    columns: Set[str] = set()
    
    if not database_schema_dict or 'tables' not in database_schema_dict:
        return tables, columns
    
    for table_name, table_dict in database_schema_dict['tables'].items():
        table_name_lower = table_name.lower()
        tables.add(table_name_lower)
        
        if 'columns' in table_dict:
            for column_name in table_dict['columns'].keys():
                column_name_lower = column_name.lower()
                col_key = f"{table_name_lower}.{column_name_lower}"
                columns.add(col_key)
    
    return tables, columns


def _analyze_schema_linking_recall(
    pickle_path: str,
    dataset_root_path: str | None,
) -> Dict:
    """
    Analyze schema linking recall rate.
    """
    # 1) Load pickle file
    pickle_file = Path(pickle_path)
    if not pickle_file.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

    print(f"Loading pickle file: {pickle_path}")
    with open(pickle_file, "rb") as f:
        dataset = pickle.load(f)

    # 2) Determine dataset path (BIRD)
    if dataset_root_path is None:
        dataset_root_path = str(PROJECT_ROOT / "dataset" / "bird")

    print(f"Loading BIRD dev dataset from: {dataset_root_path}")
    dev_items = _load_bird_dev_items(dataset_root_path)

    # 3) Build a mapping from question_id to data_item in the dataset
    dataset_map: Dict[int, Any] = {}
    for data_item in dataset:
        dataset_map[data_item.question_id] = data_item

    # 4) Process each question
    sql_preprocessor = SQLPreprocessor(case_sensitive=False)
    
    items: List[Dict[str, Any]] = []
    total_needed_tables = 0
    total_retrieved_tables = 0
    total_matched_tables = 0
    total_needed_columns = 0
    total_retrieved_columns = 0
    total_matched_columns = 0

    for question_id, dev_item in dev_items.items():
        if question_id not in dataset_map:
            continue
        
        data_item = dataset_map[question_id]
        gold_sql = dev_item["gold_sql"]
        db_path = dev_item["db_path"]
        question = dev_item["question"]
        db_id = dev_item["db_id"]
        
        # Get retrieved schema from database_schema_after_schema_linking
        retrieved_schema_dict = data_item.database_schema_after_schema_linking
        if retrieved_schema_dict is None:
            retrieved_schema_dict = {}
        
        retrieved_tables, retrieved_columns = _extract_schema_from_database_schema_dict(retrieved_schema_dict)
        
        # Extract needed schema from gold SQL
        # Normalize & remove aliases using SQLPreprocessor
        norm_gold_sql, _, _ = sql_preprocessor.preprocess(gold_sql)
        
        # Load database schema for column resolution
        schema = _load_database_schema(db_path)
        
        # Extract tables and columns
        needed_tables = _extract_tables_from_sql(norm_gold_sql)
        needed_columns_raw = _extract_columns_from_sql(norm_gold_sql, db_path=db_path, schema=schema, sql_tables=needed_tables)
        needed_columns = _normalize_column_set(needed_columns_raw, schema=schema, sql_tables=needed_tables)
        
        # Skip if we cannot extract any needed tables
        if not needed_tables:
            continue
        
        # Calculate matches
        matched_tables = needed_tables.intersection(retrieved_tables)
        matched_columns = needed_columns.intersection(retrieved_columns)
        
        missing_tables = needed_tables - retrieved_tables
        missing_columns = needed_columns - retrieved_columns
        
        # Calculate recall for this question
        table_recall = len(matched_tables) / len(needed_tables) if needed_tables else 0.0
        column_recall = len(matched_columns) / len(needed_columns) if needed_columns else 0.0
        
        # Update totals
        total_needed_tables += len(needed_tables)
        total_retrieved_tables += len(retrieved_tables)
        total_matched_tables += len(matched_tables)
        total_needed_columns += len(needed_columns)
        total_retrieved_columns += len(retrieved_columns)
        total_matched_columns += len(matched_columns)
        
        items.append({
            "question_id": question_id,
            "question": question,
            "db_id": db_id,
            "needed_tables": sorted(list(needed_tables)),
            "retrieved_tables": sorted(list(retrieved_tables)),
            "missing_tables": sorted(list(missing_tables)),
            "needed_columns": sorted(list(needed_columns)),
            "retrieved_columns": sorted(list(retrieved_columns)),
            "missing_columns": sorted(list(missing_columns)),
            "table_recall": table_recall,
            "column_recall": column_recall,
        })
    
    # Calculate overall recall
    overall_table_recall = total_matched_tables / total_needed_tables if total_needed_tables > 0 else 0.0
    overall_column_recall = total_matched_columns / total_needed_columns if total_needed_columns > 0 else 0.0
    
    # Filter items with recall < 1.0 (either table_recall or column_recall)
    items_with_errors = [
        item for item in items 
        if item["table_recall"] < 1.0 or item["column_recall"] < 1.0
    ]
    
    result = {
        "total_questions": len(items),
        "questions_with_errors": len(items_with_errors),
        "table_recall": overall_table_recall,
        "column_recall": overall_column_recall,
        "table_level_stats": {
            "total_needed_tables": total_needed_tables,
            "total_retrieved_tables": total_retrieved_tables,
            "total_matched_tables": total_matched_tables,
        },
        "column_level_stats": {
            "total_needed_columns": total_needed_columns,
            "total_retrieved_columns": total_retrieved_columns,
            "total_matched_columns": total_matched_columns,
        },
        "items": items_with_errors,  # Only include items with recall < 1.0
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze schema linking recall rate."
    )
    parser.add_argument(
        "--pickle_path",
        type=str,
        required=True,
        help="Path to the pickle file from schema linking stage "
        "(e.g., workspace/schema_linking/bird/sub_dev.pkl)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default='dataset/bird',
        help="Root path of the BIRD dataset",
    )

    args = parser.parse_args()

    analysis_result = _analyze_schema_linking_recall(
        pickle_path=args.pickle_path,
        dataset_root_path=args.dataset_path,
    )

    pickle_file = Path(args.pickle_path)
    output_path = pickle_file.parent / f"{pickle_file.stem}_schema_linking_recall.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("Schema Linking Recall Analysis")
    print("=" * 60)
    print(f"Pickle file: {args.pickle_path}")
    print(f"Total questions analyzed: {analysis_result['total_questions']}")
    print(f"Questions with recall < 1.0: {analysis_result['questions_with_errors']}")
    print(f"\nOverall Recall:")
    print(f"  - Table recall: {analysis_result['table_recall']:.4f} ({analysis_result['table_level_stats']['total_matched_tables']}/{analysis_result['table_level_stats']['total_needed_tables']})")
    print(f"  - Column recall: {analysis_result['column_recall']:.4f} ({analysis_result['column_level_stats']['total_matched_columns']}/{analysis_result['column_level_stats']['total_needed_columns']})")
    print(f"\nTable Level Stats:")
    print(f"  - Total needed tables: {analysis_result['table_level_stats']['total_needed_tables']}")
    print(f"  - Total retrieved tables: {analysis_result['table_level_stats']['total_retrieved_tables']}")
    print(f"  - Total matched tables: {analysis_result['table_level_stats']['total_matched_tables']}")
    print(f"\nColumn Level Stats:")
    print(f"  - Total needed columns: {analysis_result['column_level_stats']['total_needed_columns']}")
    print(f"  - Total retrieved columns: {analysis_result['column_level_stats']['total_retrieved_columns']}")
    print(f"  - Total matched columns: {analysis_result['column_level_stats']['total_matched_columns']}")
    print(f"\nDetailed items (only recall < 1.0) saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

