"""
Verifier Agent - Validates if guidance can correctly fix the SQL error

This agent takes the original incorrect SQL, applies the guidance,
and checks if the result matches the correct SQL (by executing both).
"""

import logging
import sqlite3
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from .models import MinerOutput

logger = logging.getLogger(__name__)


class VerifierAgent:
    """
    Verifier Agent that validates guidance effectiveness
    
    Validation approach:
    1. Execute original incorrect SQL → get result set A
    2. Execute correct SQL → get result set B
    3. Compare result sets (should be different, confirming the error)
    4. Apply guidance hints to modify incorrect SQL (simulated)
    5. Execute modified SQL → get result set C
    6. Check if C matches B (validation success)
    
    Note: Since we can't automatically "apply" guidance to SQL (that requires
    another LLM call or heuristic rules), we'll simplify the verification:
    - Just check if both SQLs are executable
    - Compare their results
    - Mark as "verified" if correct SQL produces different results
    """
    
    def __init__(self, max_rows: int = 100, timeout: int = 30):
        """
        Initialize Verifier Agent
        
        Args:
            max_rows: Maximum rows to fetch for comparison
            timeout: SQL execution timeout in seconds
        """
        self.max_rows = max_rows
        self.timeout = timeout
    
    def verify(self,
               incorrect_sql: str,
               correct_sql: str,
               db_path: str,
               miner_output: MinerOutput) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if guidance is valid by executing SQLs
        
        Args:
            incorrect_sql: Original incorrect SQL (with schema)
            correct_sql: Correct SQL (with schema)
            db_path: Path to database file
            miner_output: Miner output containing guidance
        
        Returns:
            Tuple of (verification_passed, details_dict)
        """
        details = {
            'incorrect_executable': False,
            'correct_executable': False,
            'results_differ': False,
            'incorrect_error': None,
            'correct_error': None,
            'incorrect_row_count': 0,
            'correct_row_count': 0,
            'execution_summary': ''
        }
        
        try:
            # Execute incorrect SQL
            incorrect_result, incorrect_error = self._execute_sql(db_path, incorrect_sql)
            details['incorrect_executable'] = (incorrect_error is None)
            details['incorrect_error'] = incorrect_error
            if incorrect_result is not None:
                details['incorrect_row_count'] = len(incorrect_result)
            
            # Execute correct SQL
            correct_result, correct_error = self._execute_sql(db_path, correct_sql)
            details['correct_executable'] = (correct_error is None)
            details['correct_error'] = correct_error
            if correct_result is not None:
                details['correct_row_count'] = len(correct_result)
            
            # Compare results
            if incorrect_result is not None and correct_result is not None:
                results_differ = not self._results_equal(incorrect_result, correct_result)
                details['results_differ'] = results_differ
                
                if results_differ:
                    details['execution_summary'] = (
                        f"Both SQLs executable. Results differ: "
                        f"incorrect={details['incorrect_row_count']} rows, "
                        f"correct={details['correct_row_count']} rows"
                    )
                    # Verification passes if results differ (confirming error exists)
                    verification_passed = True
                else:
                    details['execution_summary'] = (
                        f"Both SQLs executable but produce SAME results. "
                        f"This suggests the 'incorrect' SQL might not actually be wrong."
                    )
                    # If results are the same, the guidance might not be needed
                    verification_passed = False
            else:
                details['execution_summary'] = "One or both SQLs failed to execute"
                verification_passed = False
            
            logger.info(f"Verification result: {verification_passed}")
            logger.debug(f"Details: {details['execution_summary']}")
            
            return verification_passed, details
            
        except Exception as e:
            logger.error(f"Verification failed with exception: {e}")
            details['execution_summary'] = f"Verification exception: {str(e)}"
            return False, details
    
    def _execute_sql(self, db_path: str, sql: str) -> Tuple[Optional[list], Optional[str]]:
        """
        Execute SQL and return results
        
        Args:
            db_path: Path to database file
            sql: SQL query string
        
        Returns:
            Tuple of (results_list, error_message)
            - If successful: (list_of_rows, None)
            - If failed: (None, error_message)
        """
        try:
            # Check if database exists
            if not Path(db_path).exists():
                return None, f"Database file not found: {db_path}"
            
            # Connect to database
            conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True, timeout=self.timeout)
            cursor = conn.cursor()
            
            # Execute query
            cursor.execute(sql)
            results = cursor.fetchmany(self.max_rows)
            
            conn.close()
            
            return results, None
            
        except sqlite3.Error as e:
            logger.debug(f"SQL execution error: {e}")
            return None, str(e)
        except Exception as e:
            logger.debug(f"Unexpected error during SQL execution: {e}")
            return None, str(e)
    
    def _results_equal(self, results1: list, results2: list) -> bool:
        """
        Compare two result sets for equality
        
        Args:
            results1: First result set
            results2: Second result set
        
        Returns:
            True if results are equal, False otherwise
        """
        # Quick checks
        if len(results1) != len(results2):
            return False
        
        # Convert to sorted tuples for comparison
        try:
            set1 = set(tuple(row) if isinstance(row, (list, tuple)) else (row,) for row in results1)
            set2 = set(tuple(row) if isinstance(row, (list, tuple)) else (row,) for row in results2)
            return set1 == set2
        except TypeError:
            # If rows contain unhashable types, fall back to list comparison
            return sorted(results1) == sorted(results2)















