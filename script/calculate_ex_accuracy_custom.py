#!/usr/bin/env python3
"""
Script to calculate EX Accuracy for DeepEye-SQL result JSON files using custom gold SQL file.

This script loads gold SQL from a custom JSON file (e.g., arcwise_plat_full_with_diff.json)
instead of the standard BIRD dataset. It evaluates all questions present in the result file.

Usage:
    python script/calculate_ex_accuracy_custom.py <result_json_path> --gold-path <gold_json_path> [--n-parallel N]

Example:
    python script/calculate_ex_accuracy_custom.py \
        workspace/sql_selection/cleaned-mini-bird/school_and_card_latter50.json \
        --gold-path data/cleaned-mini-bird/data/arcwise_plat_full_with_diff.json
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db_utils.execution import execute_sql


def _get_db_path(db_id: str, dataset_root: str) -> str:
    """
    Get the database file path from db_id.

    Args:
        db_id: Database ID (e.g., 'debit_card_specializing')
        dataset_root: Root path of the dataset

    Returns:
        Path to the SQLite database file
    """
    db_path = Path(dataset_root) / "databases" / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    return str(db_path)


def _eval_ex_after_selection(pred_sql: str, gold_sql: str, db_path: str) -> Optional[int]:
    """
    Evaluate execution accuracy by comparing execution results of predicted SQL and gold SQL.

    Args:
        pred_sql: Predicted SQL query
        gold_sql: Ground truth SQL query
        db_path: Path to the database file

    Returns:
        1 if execution results match, 0 if they don't, None if gold SQL execution failed
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


def calculate_ex_accuracy(
    result_json_path: str,
    gold_json_path: str,
    dataset_root_path: str,
    n_parallel: int = 16
) -> Dict[str, float]:
    """
    Calculate EX Accuracy for a result JSON file using custom gold SQL file.

    Args:
        result_json_path: Path to the JSON file containing predicted SQL queries
                         Format: {"question_id": "sql_query", ...}
        gold_json_path: Path to the JSON file containing gold SQL queries
                       Format: [{"question_id": "123", "SQL": "SELECT ...", "db_id": "xxx"}, ...]
        dataset_root_path: Root path of the dataset (contains database directories)
        n_parallel: Number of parallel workers for evaluation

    Returns:
        Dictionary containing accuracy metrics
    """
    # Load result JSON file
    result_path = Path(result_json_path)
    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_json_path}")

    print(f"Loading result file: {result_json_path}")
    with open(result_path, 'r', encoding='utf-8') as f:
        result_data = json.load(f)

    # Load gold JSON file
    gold_path = Path(gold_json_path)
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_json_path}")

    print(f"Loading gold file: {gold_json_path}")
    with open(gold_path, 'r', encoding='utf-8') as f:
        gold_data_list = json.load(f)

    # Convert gold data to dict keyed by question_id
    gold_data = {}
    for item in gold_data_list:
        qid = str(item.get('question_id', ''))
        if qid:
            gold_data[qid] = item

    print(f"Loaded {len(gold_data)} gold SQL entries")

    # Prepare evaluation tasks
    evaluation_tasks = []
    missing_questions = []
    missing_dbs = []

    for question_id_str, pred_sql in result_data.items():
        try:
            question_id = int(question_id_str)
        except ValueError:
            print(f"Warning: Invalid question_id '{question_id_str}', skipping...")
            continue

        if question_id_str not in gold_data:
            missing_questions.append(question_id)
            continue

        gold_item = gold_data[question_id_str]
        gold_sql = gold_item.get('SQL', '')
        db_id = gold_item.get('db_id', '')

        if not gold_sql:
            print(f"Warning: No gold SQL for question_id {question_id_str}, skipping...")
            continue

        if not db_id:
            print(f"Warning: No db_id for question_id {question_id_str}, skipping...")
            continue

        try:
            db_path = _get_db_path(db_id, dataset_root_path)
        except FileNotFoundError:
            missing_dbs.append((question_id, db_id))
            continue

        evaluation_tasks.append({
            'question_id': question_id,
            'pred_sql': pred_sql,
            'gold_sql': gold_sql,
            'db_path': db_path,
            'db_id': db_id
        })

    if missing_questions:
        print(f"Warning: {len(missing_questions)} questions not found in gold file: {missing_questions[:10]}...")

    if missing_dbs:
        print(f"Warning: {len(missing_dbs)} database files not found: {missing_dbs[:10]}...")

    if not evaluation_tasks:
        raise ValueError("No valid evaluation tasks found!")

    print(f"Evaluating {len(evaluation_tasks)} SQL queries with {n_parallel} parallel workers...")

    # Evaluate in parallel
    executor = ProcessPoolExecutor(max_workers=n_parallel)
    future_to_task = {}
    for task in evaluation_tasks:
        future = executor.submit(
            _eval_ex_after_selection,
            task['pred_sql'],
            task['gold_sql'],
            task['db_path']
        )
        future_to_task[future] = task

    results = []
    gold_sql_failures = []
    incorrect_sqls = []  # Store incorrect SQLs with their IDs

    for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Evaluating SQL"):
        task = future_to_task[future]
        result = future.result()
        if result is not None:
            results.append(result)
            # If result is 0, it means the SQL is incorrect
            if result == 0:
                incorrect_sqls.append({
                    'question_id': task['question_id'],
                    'pred_sql': task['pred_sql'],
                    'gold_sql': task['gold_sql'],
                    'db_path': task['db_path'],
                    'db_id': task['db_id']
                })
        else:
            gold_sql_failures.append(1)

    executor.shutdown(wait=True)

    # Calculate metrics
    if not results:
        raise ValueError("No valid evaluation results! All gold SQL queries may have failed.")

    correct_count = sum(results)
    total_count = len(results)
    accuracy = correct_count / total_count

    metrics = {
        'total_questions': len(result_data),
        'evaluated_questions': total_count,
        'missing_questions': len(missing_questions),
        'missing_dbs': len(missing_dbs),
        'gold_sql_failures': len(gold_sql_failures),
        'correct_count': correct_count,
        'incorrect_count': total_count - correct_count,
        'ex_accuracy': accuracy,
        'incorrect_sqls': incorrect_sqls
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Calculate EX Accuracy for DeepEye-SQL result JSON files using custom gold SQL file'
    )
    parser.add_argument(
        'result_json_path',
        type=str,
        help='Path to the JSON file containing predicted SQL queries'
    )
    parser.add_argument(
        '--gold-path',
        type=str,
        required=True,
        help='Path to the JSON file containing gold SQL queries (e.g., arcwise_plat_full_with_diff.json)'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='data/cleaned-mini-bird',
        help='Root path of the dataset containing database directories (default: data/cleaned-mini-bird)'
    )
    parser.add_argument(
        '--n-parallel',
        type=int,
        default=16,
        help='Number of parallel workers for evaluation (default: 16)'
    )

    args = parser.parse_args()

    metrics = calculate_ex_accuracy(
        result_json_path=args.result_json_path,
        gold_json_path=args.gold_path,
        dataset_root_path=args.dataset_path,
        n_parallel=args.n_parallel
    )

    # Save incorrect SQLs to a file
    result_path = Path(args.result_json_path)
    incorrect_output_path = result_path.parent / f"{result_path.stem}_incorrect.json"

    incorrect_data = {
        'total_incorrect': len(metrics['incorrect_sqls']),
        'incorrect_sqls': metrics['incorrect_sqls']
    }

    with open(incorrect_output_path, 'w', encoding='utf-8') as f:
        json.dump(incorrect_data, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("EX Accuracy Evaluation Results")
    print("="*60)
    print(f"Total questions in result file: {metrics['total_questions']}")
    print(f"Successfully evaluated: {metrics['evaluated_questions']}")
    print(f"Missing questions (not in gold file): {metrics['missing_questions']}")
    print(f"Missing database files: {metrics['missing_dbs']}")
    print(f"Gold SQL execution failures: {metrics['gold_sql_failures']}")
    print(f"Correct predictions: {metrics['correct_count']}")
    print(f"Incorrect predictions: {metrics['incorrect_count']}")
    print(f"\nEX Accuracy: {metrics['ex_accuracy']:.4f} ({metrics['ex_accuracy']*100:.2f}%)")
    print(f"\nIncorrect SQLs saved to: {incorrect_output_path}")
    print(f"Total incorrect SQLs: {len(metrics['incorrect_sqls'])}")
    print("="*60)


if __name__ == "__main__":
    main()
