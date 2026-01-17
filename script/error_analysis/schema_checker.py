"""
Schema error checker

Determines if the error is purely due to incorrect schema selection.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Set, Any, Tuple

# Add project root to path to import SQLPreprocessor directly
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import directly from the module to avoid triggering caf/__init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "sql_preprocessor",
    project_root / "caf/preprocess/sql_preprocessor.py"
)
sql_preprocessor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sql_preprocessor_module)
SQLPreprocessor = sql_preprocessor_module.SQLPreprocessor

logger = logging.getLogger(__name__)


class SchemaChecker:
    """
    Check if SQL error is purely due to schema selection
    
    Strategy:
    1. Qualify both SQLs (resolve aliases, add table names to columns)
    2. Extract schema (tables, columns) from both SQLs
    3. Compare schema overlap
    4. If overlap is below threshold, consider it as pure schema error
    """
    
    def __init__(self, overlap_threshold: float = 0.3):
        """
        Initialize schema checker
        
        Args:
            overlap_threshold: If schema overlap < threshold, consider as pure schema error
        """
        self.overlap_threshold = overlap_threshold
    
    def check(self, 
              incorrect_sql: str, 
              correct_sql: str,
              db_path: str) -> Tuple[bool, float, Dict[str, Any], Dict[str, Any], str, str]:
        """
        Check if error is pure schema error
        
        Args:
            incorrect_sql: The incorrect SQL
            correct_sql: The correct SQL
            db_path: Path to database file (for schema loading)
        
        Returns:
            Tuple of:
            - is_pure_schema_error: Boolean indicating if it's pure schema error
            - overlap_score: Schema overlap score (0-1)
            - incorrect_schema: Extracted schema from incorrect SQL
            - correct_schema: Extracted schema from correct SQL
            - qualified_incorrect_sql: Qualified incorrect SQL
            - qualified_correct_sql: Qualified correct SQL
        """
        try:
            # Create preprocessor with database path for schema loading
            preprocessor = SQLPreprocessor(
                case_sensitive=False,
                db_path=db_path,
                use_schema_cache=True
            )
            
            # Step 1: Qualify SQLs (resolve aliases, add table names)
            logger.debug(f"Qualifying incorrect SQL...")
            qualified_incorrect_sql = preprocessor.qualify_sql(incorrect_sql)
            logger.debug(f"Qualified incorrect: {qualified_incorrect_sql}")
            
            logger.debug(f"Qualifying correct SQL...")
            qualified_correct_sql = preprocessor.qualify_sql(correct_sql)
            logger.debug(f"Qualified correct: {qualified_correct_sql}")
            
            # Step 2: Extract schema from qualified SQLs
            logger.debug(f"Extracting schema from incorrect SQL...")
            incorrect_schema = preprocessor.extract_sql_schema(qualified_incorrect_sql)
            
            logger.debug(f"Extracting schema from correct SQL...")
            correct_schema = preprocessor.extract_sql_schema(qualified_correct_sql)
            
            # Step 3: Calculate overlap
            overlap_score = self._calculate_schema_overlap(incorrect_schema, correct_schema)
            logger.debug(f"Schema overlap score: {overlap_score:.3f}")
            
            # Step 4: Determine if pure schema error
            is_pure_schema_error = overlap_score < self.overlap_threshold
            
            if is_pure_schema_error:
                logger.info(f"Classified as PURE SCHEMA ERROR (overlap={overlap_score:.3f})")
            else:
                logger.info(f"Classified as LOGIC/OPERATION ERROR (overlap={overlap_score:.3f})")
            
            return (
                is_pure_schema_error,
                overlap_score,
                incorrect_schema,
                correct_schema,
                qualified_incorrect_sql,
                qualified_correct_sql
            )
            
        except Exception as e:
            logger.error(f"Schema checking failed: {e}")
            # If checking fails, assume it's not a pure schema error (to be safe)
            return False, 0.0, {}, {}, incorrect_sql, correct_sql
    
    def _calculate_schema_overlap(self, 
                                   schema1: Dict[str, Any], 
                                   schema2: Dict[str, Any]) -> float:
        """
        Calculate schema overlap score between two schemas
        
        Args:
            schema1: First schema dict
            schema2: Second schema dict
        
        Returns:
            Overlap score (0-1), where:
            - 0 = completely different schemas
            - 1 = identical schemas
        """
        # Extract table sets
        tables1 = set(schema1.get('tables', []))
        tables2 = set(schema2.get('tables', []))
        
        # Extract column sets (as tuples of (table, column))
        columns1 = self._extract_column_set(schema1.get('columns', []))
        columns2 = self._extract_column_set(schema2.get('columns', []))
        
        # Calculate Jaccard similarity for tables
        if len(tables1) == 0 and len(tables2) == 0:
            table_overlap = 1.0
        elif len(tables1) == 0 or len(tables2) == 0:
            table_overlap = 0.0
        else:
            table_overlap = len(tables1 & tables2) / len(tables1 | tables2)
        
        # Calculate Jaccard similarity for columns
        if len(columns1) == 0 and len(columns2) == 0:
            column_overlap = 1.0
        elif len(columns1) == 0 or len(columns2) == 0:
            column_overlap = 0.0
        else:
            column_overlap = len(columns1 & columns2) / len(columns1 | columns2)
        
        # Weighted average (tables are more important)
        overlap_score = 0.6 * table_overlap + 0.4 * column_overlap
        
        logger.debug(f"Table overlap: {table_overlap:.3f}, Column overlap: {column_overlap:.3f}")
        
        return overlap_score
    
    def _extract_column_set(self, columns: list) -> Set[Tuple[str, str]]:
        """
        Extract set of (table, column) tuples from column list
        
        Args:
            columns: List of column dicts with 'table' and 'column' keys
        
        Returns:
            Set of (table, column) tuples
        """
        column_set = set()
        for col in columns:
            table = col.get('table')
            column = col.get('column')
            if table and column:
                column_set.add((table, column))
        return column_set

