#!/usr/bin/env python3
"""
Script to calculate EX Accuracy for DeepEye-SQL result JSON files.

Usage:
    python script/calculate_ex_accuracy.py <result_json_path> [--dataset-path DATASET_PATH] [--n-parallel N]
    python script/calculate_ex_accuracy.py <result_json_path> --question-id-ranges 44-88 435-529

Example:
    python script/calculate_ex_accuracy.py /home/yangchenyu/DeepEye-SQL/workspace/sql_selection/cleaned-mini-bird/school_and_card_latter50_context_graph_wo_user.json --question-id-ranges 435-530
    python script/calculate_ex_accuracy.py /home/yangchenyu/DeepEye-SQL-Metadata/results/bird-dev/qwen3-coder-30b-a3b.json --question-id-ranges 435-530 142-194 624-716
    python script/calculate_ex_accuracy.py workspace/sql_selection/bird/school_and_card_latter50_context_graph.json --question-id-ranges 44-88 435-530
""" 

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db_utils.execution import execute_sql
from app.dataset.dataset import DatasetFactory
from app.config import DatasetConfig


def _parse_question_id_ranges(ranges: list[str]) -> Set[int]:
    """
    Parse question ID ranges like "44-88" or "435-529" into a set of integers (inclusive).
    Single numbers like "100" are also supported.
    """
    ids = set()
    for s in ranges:
        s = s.strip()
        if "-" in s:
            parts = s.split("-", 1)
            low, high = int(parts[0].strip()), int(parts[1].strip())
            ids.update(range(low, high + 1))
        else:
            ids.add(int(s))
    return ids


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
    dataset_root_path: Optional[str] = None,
    n_parallel: int = 16,
    question_id_filter: Optional[Set[int]] = None
) -> Dict[str, float]:
    """
    Calculate EX Accuracy for a result JSON file.
    
    Args:
        result_json_path: Path to the JSON file containing predicted SQL queries
                         Format: {"question_id": "sql_query", ...}
        dataset_root_path: Root path of the BIRD dataset. If None, uses default path.
        n_parallel: Number of parallel workers for evaluation
        question_id_filter: If set, only evaluate and report EX for these question IDs.
                            Ranges like 44-88, 435-529 can be parsed via _parse_question_id_ranges.
        
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
    
    # Determine dataset path
    if dataset_root_path is None:
        # Default BIRD dataset path
        project_root = Path(__file__).resolve().parent.parent
        dataset_root_path = str(project_root / "dataset" / "bird")
    
    # Load BIRD dev dataset
    print(f"Loading BIRD dev dataset from: {dataset_root_path}")
    dataset_config = DatasetConfig(
        type="bird",
        split="dev",
        root_path=dataset_root_path
    )
    dataset = DatasetFactory.get_dataset(dataset_config)
    
    # Create a mapping from question_id to dataset item
    dataset_dict = {item.question_id: item for item in dataset}
    
    # Prepare evaluation tasks
    evaluation_tasks = []
    missing_questions = []
    parsed_result_ids = []
    
    for question_id_str, pred_sql in result_data.items():
        try:
            question_id = int(question_id_str)
        except ValueError:
            print(f"Warning: Invalid question_id '{question_id_str}', skipping...")
            continue
        parsed_result_ids.append(question_id)
        
        if question_id not in dataset_dict:
            missing_questions.append(question_id)
            continue
        
        if question_id_filter is not None and question_id not in question_id_filter:
            continue
        
        data_item = dataset_dict[question_id]
        evaluation_tasks.append({
            'question_id': question_id,
            'pred_sql': pred_sql,
            'gold_sql': data_item.gold_sql,
            'db_path': data_item.database_path
        })
    
    if missing_questions:
        print(f"Warning: {len(missing_questions)} questions not found in dataset: {missing_questions[:10]}...")
    
    if not evaluation_tasks:
        if question_id_filter is not None:
            if parsed_result_ids:
                min_id = min(parsed_result_ids)
                max_id = max(parsed_result_ids)
                overlap = len(set(parsed_result_ids) & question_id_filter)
                raise ValueError(
                    "No valid evaluation tasks found! "
                    "No question_id in result file matches the given --question-id-ranges. "
                    f"Result file question_id range: [{min_id}, {max_id}], "
                    f"parsed IDs: {len(parsed_result_ids)}, overlap with filter: {overlap}."
                )
            raise ValueError(
                "No valid evaluation tasks found! "
                "No parsable question_id found in result file keys."
            )
        raise ValueError("No valid evaluation tasks found!")
    
    if question_id_filter is not None:
        print(f"Filtering to {len(question_id_filter)} question IDs; {len(evaluation_tasks)} present in result file.")
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
                    'db_path': task['db_path']
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
        'gold_sql_failures': len(gold_sql_failures),
        'correct_count': correct_count,
        'incorrect_count': total_count - correct_count,
        'ex_accuracy': accuracy,
        'incorrect_sqls': incorrect_sqls
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Calculate EX Accuracy for DeepEye-SQL result JSON files'
    )
    parser.add_argument(
        'result_json_path',
        type=str,
        help='Path to the JSON file containing predicted SQL queries'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Root path of the BIRD dataset (default: dataset/bird)'
    )
    parser.add_argument(
        '--n-parallel',
        type=int,
        default=16,
        help='Number of parallel workers for evaluation (default: 16)'
    )
    parser.add_argument(
        '--question-id-ranges',
        type=str,
        nargs='+',
        default=None,
        metavar='RANGE',
        help='Only compute EX for these question ID ranges (inclusive). E.g.: --question-id-ranges 44-88 435-529'
    )
    
    args = parser.parse_args()
    
    question_id_filter = None
    if args.question_id_ranges is not None:
        question_id_filter = _parse_question_id_ranges(args.question_id_ranges)
        print(f"Question ID filter: {len(question_id_filter)} IDs from ranges: {args.question_id_ranges}")

    metrics = calculate_ex_accuracy(
        result_json_path=args.result_json_path,
        dataset_root_path=args.dataset_path,
        n_parallel=args.n_parallel,
        question_id_filter=question_id_filter
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
    print(f"Missing questions: {metrics['missing_questions']}")
    print(f"Gold SQL execution failures: {metrics['gold_sql_failures']}")
    print(f"Correct predictions: {metrics['correct_count']}")
    print(f"Incorrect predictions: {metrics['incorrect_count']}")
    print(f"\nEX Accuracy: {metrics['ex_accuracy']:.4f} ({metrics['ex_accuracy']*100:.2f}%)")
    print(f"\nIncorrect SQLs saved to: {incorrect_output_path}")
    print(f"Total incorrect SQLs: {len(metrics['incorrect_sqls'])}")
    print("="*60)
    


if __name__ == "__main__":
    main()

