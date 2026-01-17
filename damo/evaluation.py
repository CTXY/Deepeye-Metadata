# NL2SQL Evaluation Module
# Implements common evaluation metrics for Text-to-SQL tasks

import logging
from typing import List, Dict, Any, Optional, Tuple
from data_handler import BirdDataHandler

logger = logging.getLogger(__name__)

class NL2SQLEvaluator:
    """
    NL2SQL evaluation module that implements standard metrics including Execution Accuracy (EX)
    """
    
    def __init__(self, data_handler: BirdDataHandler):
        self.data_handler = data_handler
        logger.info("NL2SQLEvaluator initialized")
    
    def calculate_execution_accuracy(self, results: List[Dict[str, Any]], show_individual: bool = True) -> Dict[str, Any]:
        """
        Calculate Execution Accuracy (EX) - the most common metric for NL2SQL evaluation
        
        EX measures whether the generated SQL produces the same execution result as ground truth SQL.
        This is considered the gold standard for Text-to-SQL evaluation.
        
        Args:
            results: List of result dictionaries with generated_sql, ground_truth_sql, db_id, etc.
            show_individual: Whether to print individual evaluation results
            
        Returns:
            Dictionary with EX metrics and detailed statistics
        """
        total_queries = len(results)
        if total_queries == 0:
            return {"ex_accuracy": 0.0, "total": 0, "correct": 0, "details": []}
        
        correct_count = 0
        evaluation_details = []
        
        logger.info(f"Calculating Execution Accuracy for {total_queries} queries...")
        
        for i, result in enumerate(results):
            question_id = result.get('question_id', f'query_{i+1}')
            logger.debug(f"Evaluating query {i+1}/{total_queries}: {question_id}")
            
            try:
                is_correct = self._compare_execution_results(
                    generated_sql=result.get('generated_sql', ''),
                    ground_truth_sql=result.get('ground_truth_sql', ''),
                    db_id=result.get('db_id', ''),
                    question_id=question_id
                )
                
                if is_correct:
                    correct_count += 1
                
                evaluation_details.append({
                    "question_id": question_id,
                    "db_id": result.get('db_id', ''),
                    "is_correct": is_correct,
                    "generated_sql": result.get('generated_sql', ''),
                    "ground_truth_sql": result.get('ground_truth_sql', '')
                })
                
                # Print individual result if requested
                if show_individual:
                    status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                    print(f"  {status} - {question_id} (DB: {result.get('db_id', 'unknown')})")
                    if not is_correct:
                        print(f"    Generated: {result.get('generated_sql', '')[:100]}...")
                        print(f"    Ground Truth: {result.get('ground_truth_sql', '')[:100]}...")
                
            except Exception as e:
                logger.error(f"Error evaluating query {question_id}: {e}")
                evaluation_details.append({
                    "question_id": question_id,
                    "db_id": result.get('db_id', ''),
                    "is_correct": False,
                    "error": str(e),
                    "generated_sql": result.get('generated_sql', ''),
                    "ground_truth_sql": result.get('ground_truth_sql', '')
                })
                
                if show_individual:
                    print(f"  ❌ ERROR - {question_id}: {e}")
        
        ex_accuracy = correct_count / total_queries
        
        evaluation_result = {
            "ex_accuracy": ex_accuracy,
            "total": total_queries,
            "correct": correct_count,
            "incorrect": total_queries - correct_count,
            "accuracy_percentage": ex_accuracy * 100,
            "details": evaluation_details
        }
        
        logger.info(f"Execution Accuracy: {correct_count}/{total_queries} = {ex_accuracy:.4f} ({ex_accuracy*100:.2f}%)")
        
        return evaluation_result
    
    def _compare_execution_results(self, generated_sql: str, ground_truth_sql: str, 
                                 db_id: str, question_id: str) -> bool:
        """
        Compare execution results of generated SQL vs ground truth SQL
        
        Returns True if both queries produce identical results, False otherwise
        """
        if not generated_sql or not ground_truth_sql:
            logger.debug(f"Empty SQL for {question_id}: generated={bool(generated_sql)}, ground_truth={bool(ground_truth_sql)}")
            return False
        
        try:
            # Execute generated SQL
            gen_success, gen_results, gen_error = self.data_handler.execute_sql(db_id, generated_sql)
            # Execute ground truth SQL
            gt_success, gt_results, gt_error = self.data_handler.execute_sql(db_id, ground_truth_sql)
            
            # Compare results
            return self._results_match(gen_results, gt_results)
            
        except Exception as e:
            logger.error(f"Exception during execution comparison for {question_id}: {e}")
            return False
    
    def _results_match(self, results1: List[Dict], results2: List[Dict]) -> bool:
        """
        Check if two query results are identical
        
        Handles different orderings, formats, and column names appropriately.
        Only compares the actual data values, ignoring column name differences.
        """
        print('====================Results Match===========================')
        print(results1)
        print(results2)
        print('===============================================')
        if results1 is None and results2 is None:
            return True
        
        if results1 is None or results2 is None:
            return False
        
        if len(results1) != len(results2):
            return False
        
        # Convert to sorted tuples for comparison (handles ordering differences)
        try:
            def normalize_row(row):
                """Normalize a row for comparison - only compare values, not column names or order"""
                if isinstance(row, dict):
                    # Extract only values, normalize them, and sort for consistent comparison
                    values = []
                    for v in row.values():
                        if v is None:
                            values.append('NULL')
                        elif isinstance(v, (int, float)):
                            # Normalize numeric values to avoid 507.0 vs 507 differences
                            values.append(str(float(v)))
                        else:
                            values.append(str(v))
                    return tuple(sorted(values))
                elif isinstance(row, (list, tuple)):
                    # Convert list/tuple to tuple of strings and sort for order-insensitive comparison
                    values = []
                    for v in row:
                        if v is None:
                            values.append('NULL')
                        elif isinstance(v, (int, float)):
                            # Normalize numeric values to avoid 507.0 vs 507 differences
                            values.append(str(float(v)))
                        else:
                            values.append(str(v))
                    return tuple(sorted(values))
                else:
                    if row is None:
                        return ('NULL',)
                    elif isinstance(row, (int, float)):
                        return (str(float(row)),)
                    else:
                        return (str(row),)
            
            normalized1 = sorted(normalize_row(row) for row in results1)
            normalized2 = sorted(normalize_row(row) for row in results2)
            
            return normalized1 == normalized2
            
        except Exception as e:
            logger.error(f"Error comparing results: {e}")
            # Fallback to string comparison
            return str(sorted(results1)) == str(sorted(results2))
    
    def calculate_database_level_accuracy(self, evaluation_result: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Calculate accuracy broken down by database
        
        Args:
            evaluation_result: Result from calculate_execution_accuracy
            
        Returns:
            Dictionary mapping database names to their accuracy metrics
        """
        db_stats = {}
        
        for detail in evaluation_result.get('details', []):
            db_id = detail.get('db_id', 'unknown')
            is_correct = detail.get('is_correct', False)
            
            if db_id not in db_stats:
                db_stats[db_id] = {'total': 0, 'correct': 0}
            
            db_stats[db_id]['total'] += 1
            if is_correct:
                db_stats[db_id]['correct'] += 1
        
        # Calculate accuracy for each database
        db_accuracies = {}
        for db_id, stats in db_stats.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            db_accuracies[db_id] = {
                'accuracy': accuracy,
                'accuracy_percentage': accuracy * 100,
                'correct': stats['correct'],
                'total': stats['total']
            }
        
        return db_accuracies
    
    def print_evaluation_summary(self, evaluation_result: Dict[str, Any], show_database_breakdown: bool = True):
        """
        Print a formatted summary of evaluation results
        """
        print("\n" + "="*60)
        print("EXECUTION ACCURACY (EX) EVALUATION RESULTS")
        print("="*60)
        
        print(f"Overall Execution Accuracy: {evaluation_result['correct']}/{evaluation_result['total']} = {evaluation_result['accuracy_percentage']:.2f}%")
        print(f"Correct: {evaluation_result['correct']}")
        print(f"Incorrect: {evaluation_result['incorrect']}")
        
        if show_database_breakdown:
            db_accuracies = self.calculate_database_level_accuracy(evaluation_result)
            
            if len(db_accuracies) > 1:  # Only show breakdown if multiple databases
                print(f"\nPer-Database Breakdown:")
                print("-" * 40)
                
                for db_id, stats in sorted(db_accuracies.items()):
                    print(f"{db_id}: {stats['correct']}/{stats['total']} = {stats['accuracy_percentage']:.2f}%")
        
        print("="*60)
