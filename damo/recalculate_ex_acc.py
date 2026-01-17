#!/usr/bin/env python3
"""
Script to recalculate EX Accuracy for result files after dev.json ground truth changes.

This script:
1. Loads result JSON files
2. Updates ground truth SQL from the new dev.json
3. Recalculates EX ACC for specified databases (california_schools, financial)
4. Outputs the updated results

Usage:
    python recalculate_ex_acc.py <result_json_path> [--databases DB1 DB2 ...]
    
Example:
    python recalculate_ex_acc.py results/gpt-4o_damo_semantic_only_results_six_databases.json --databases california_schools financial
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import NL2SQLConfig, DataConfig
from data_handler import BirdDataHandler
from evaluation import NL2SQLEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dev_dataset(dev_json_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Load dev dataset and create a mapping from question_id to ground truth SQL
    
    Args:
        dev_json_path: Path to dev.json file
        
    Returns:
        Dictionary mapping question_id to dataset item
    """
    logger.info(f"Loading dev dataset from {dev_json_path}")
    with open(dev_json_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    # Create mapping from question_id to item
    dev_dict = {}
    for item in dev_data:
        question_id = item.get('question_id')
        if question_id is not None:
            dev_dict[question_id] = {
                'sql': item.get('SQL', ''),
                'db_id': item.get('db_id', ''),
                'question': item.get('question', ''),
                'evidence': item.get('evidence', '')
            }
    
    logger.info(f"Loaded {len(dev_dict)} items from dev dataset")
    return dev_dict


def filter_results_by_databases(results: List[Dict[str, Any]], target_databases: List[str]) -> List[Dict[str, Any]]:
    """
    Filter results to only include specified databases
    
    Args:
        results: List of result dictionaries
        target_databases: List of database IDs to include
        
    Returns:
        Filtered list of results
    """
    filtered = []
    for result in results:
        db_id = result.get('db_id', '')
        if db_id in target_databases:
            filtered.append(result)
    
    logger.info(f"Filtered {len(filtered)} results from {len(results)} total for databases: {target_databases}")
    return filtered


def update_ground_truth_sql(results: List[Dict[str, Any]], dev_dict: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Update ground truth SQL in results from dev_dict
    
    Args:
        results: List of result dictionaries
        dev_dict: Dictionary mapping question_id to dev dataset items
        
    Returns:
        Updated results with new ground truth SQL
    """
    updated_count = 0
    missing_count = 0
    
    for result in results:
        question_id = result.get('question_id')
        if question_id is None:
            logger.warning(f"Result missing question_id: {result}")
            continue
        
        if question_id in dev_dict:
            old_sql = result.get('ground_truth_sql', '')
            new_sql = dev_dict[question_id]['sql']
            result['ground_truth_sql'] = new_sql
            
            if old_sql != new_sql:
                updated_count += 1
                logger.debug(f"Updated ground truth SQL for question_id {question_id}")
        else:
            missing_count += 1
            logger.warning(f"Question ID {question_id} not found in dev dataset")
    
    logger.info(f"Updated {updated_count} ground truth SQLs, {missing_count} missing from dev dataset")
    return results


def recalculate_ex_accuracy(
    result_json_path: str,
    target_databases: List[str],
    dev_json_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Recalculate EX Accuracy for specified databases
    
    Args:
        result_json_path: Path to result JSON file
        target_databases: List of database IDs to recalculate
        dev_json_path: Path to dev.json file (default: dataset/bird/dev/dev.json)
        
    Returns:
        Dictionary containing updated evaluation results
    """
    # Determine dev.json path
    if dev_json_path is None:
        project_root = Path(__file__).resolve().parent.parent
        dev_json_path = str(project_root / "dataset" / "bird" / "dev" / "dev.json")
    
    # Load dev dataset
    dev_dict = load_dev_dataset(dev_json_path)
    
    # Load result file
    logger.info(f"Loading result file: {result_json_path}")
    
    # First, try to load JSON directly
    try:
        with open(result_json_path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing error: {e}. Attempting to fix common JSON errors...")
        # Read file content and fix common JSON errors
        with open(result_json_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix "execution_success": success, -> "execution_success": true/false,
        # Based on context: check if execution_result is null or has data
        import re
        
        def fix_execution_success(match):
            start_pos = match.start()
            # Look ahead to find execution_result
            lookahead = content[start_pos:start_pos+300]
            # If execution_result is null, execution_success is likely false
            if '"execution_result": null' in lookahead:
                return '"execution_success": false,'
            # If execution_result is an array (even if empty), execution_success is likely true
            elif '"execution_result": [' in lookahead or '"execution_result":[]' in lookahead:
                return '"execution_success": true,'
            else:
                # Default to true if we can't determine
                return '"execution_success": true,'
        
        # Replace "execution_success": success, with appropriate boolean
        content = re.sub(r'"execution_success":\s*success,', fix_execution_success, content)
        
        # Try to parse again
        try:
            result_data = json.loads(content)
            logger.info("Successfully fixed and loaded JSON file")
        except json.JSONDecodeError as e2:
            logger.error(f"Failed to fix JSON error: {e2}")
            # Try to get more context about the error
            error_pos = getattr(e2, 'pos', None)
            if error_pos:
                start = max(0, error_pos - 100)
                end = min(len(content), error_pos + 100)
                logger.error(f"Error context: ...{content[start:end]}...")
            raise e2
    
    # Get results array (could be "detailed_results" or "results")
    if 'detailed_results' in result_data:
        results = result_data['detailed_results']
    elif 'results' in result_data:
        results = result_data['results']
    else:
        raise ValueError("Result file must contain either 'detailed_results' or 'results' key")
    
    # Filter results by target databases
    filtered_results = filter_results_by_databases(results, target_databases)
    
    if len(filtered_results) == 0:
        logger.warning(f"No results found for databases: {target_databases}")
        return {
            'ex_accuracy': 0.0,
            'total': 0,
            'correct': 0,
            'incorrect': 0,
            'accuracy_percentage': 0.0,
            'databases': target_databases
        }
    
    # Update ground truth SQL from new dev.json
    filtered_results = update_ground_truth_sql(filtered_results, dev_dict)
    
    # Initialize data handler and evaluator
    logger.info("Initializing data handler and evaluator...")
    config = NL2SQLConfig.default()
    data_handler = BirdDataHandler(config.data)
    evaluator = NL2SQLEvaluator(data_handler)
    
    # Prepare results for evaluation (format expected by evaluator)
    evaluation_results = []
    for result in filtered_results:
        evaluation_results.append({
            'question_id': result.get('question_id'),
            'db_id': result.get('db_id', ''),
            'generated_sql': result.get('generated_sql', ''),
            'ground_truth_sql': result.get('ground_truth_sql', '')
        })
    
    # Recalculate EX Accuracy
    logger.info(f"Recalculating EX Accuracy for {len(evaluation_results)} queries...")
    # Temporarily reduce logging level to reduce output
    old_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)
    
    try:
        evaluation_result = evaluator.calculate_execution_accuracy(
            evaluation_results,
            show_individual=False  # Don't print individual results
        )
    finally:
        # Restore logging level
        logging.getLogger().setLevel(old_level)
    
    # Calculate per-database breakdown
    db_accuracies = evaluator.calculate_database_level_accuracy(evaluation_result)
    
    # Print summary
    print("\n" + "="*60)
    print(f"EX ACC RECALCULATION RESULTS")
    print(f"Result file: {result_json_path}")
    print(f"Target databases: {target_databases}")
    print("="*60)
    print(f"Overall EX Accuracy: {evaluation_result['correct']}/{evaluation_result['total']} = {evaluation_result['accuracy_percentage']:.2f}%")
    print(f"Correct: {evaluation_result['correct']}")
    print(f"Incorrect: {evaluation_result['incorrect']}")
    
    print(f"\nPer-Database Breakdown:")
    print("-" * 40)
    for db_id in target_databases:
        if db_id in db_accuracies:
            stats = db_accuracies[db_id]
            print(f"{db_id}: {stats['correct']}/{stats['total']} = {stats['accuracy_percentage']:.2f}%")
        else:
            print(f"{db_id}: No results found")
    
    print("="*60)
    
    return {
        **evaluation_result,
        'databases': target_databases,
        'database_breakdown': db_accuracies
    }


def main():
    parser = argparse.ArgumentParser(
        description='Recalculate EX Accuracy for result files after dev.json changes'
    )
    parser.add_argument(
        'result_json_path',
        type=str,
        help='Path to the result JSON file'
    )
    parser.add_argument(
        '--databases',
        type=str,
        nargs='+',
        default=['california_schools', 'financial'],
        help='List of database IDs to recalculate (default: california_schools financial)'
    )
    parser.add_argument(
        '--dev-json-path',
        type=str,
        default=None,
        help='Path to dev.json file (default: dataset/bird/dev/dev.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save updated results (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate result file exists
    result_path = Path(args.result_json_path)
    if not result_path.exists():
        logger.error(f"Result file not found: {args.result_json_path}")
        sys.exit(1)
    
    # Recalculate EX Accuracy
    try:
        evaluation_result = recalculate_ex_accuracy(
            result_json_path=args.result_json_path,
            target_databases=args.databases,
            dev_json_path=args.dev_json_path
        )
        
        # Save results if output path specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error recalculating EX Accuracy: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

