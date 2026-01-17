"""
SQL Masker - Replace table names, column names, and values with placeholders

This module masks SQL queries to create abstract patterns by:
1. Replacing table names with T1, T2, T3, ...
2. Replacing column names with C1, C2, C3, ...
3. Replacing literal values with V1, V2, V3, ...
4. Ensuring consistent mapping across incorrect and correct SQL
"""

import logging
import re
from typing import Dict, Tuple, Set, Optional
import sqlglot
from sqlglot import expressions

logger = logging.getLogger(__name__)


class SQLMasker:
    """
    Mask SQL queries to create abstract patterns
    
    Key features:
    - Consistent mapping: same entity gets same mask across both SQLs
    - Removes aliases (AS xxx)
    - Preserves SQL structure and keywords
    """
    
    def __init__(self):
        self.table_counter = 0
        self.column_counter = 0
        self.value_counter = 0
        
        self.table_map: Dict[str, str] = {}
        self.column_map: Dict[str, str] = {}
        self.value_map: Dict[str, str] = {}
        
        # Combined mapping for output
        self.mapping_dict: Dict[str, str] = {}
    
    def mask_sql_pair(self, 
                      qualified_incorrect_sql: str,
                      qualified_correct_sql: str) -> Tuple[str, str, Dict[str, str]]:
        """
        Mask a pair of SQLs with consistent mapping
        
        Args:
            qualified_incorrect_sql: Qualified incorrect SQL
            qualified_correct_sql: Qualified correct SQL
        
        Returns:
            Tuple of (masked_incorrect, masked_correct, mapping_dict)
        """
        # Reset state
        self._reset()
        
        # Parse both SQLs first to build unified mapping
        try:
            ast_incorrect = sqlglot.parse_one(qualified_incorrect_sql, dialect='sqlite', 
                                             error_level=sqlglot.ErrorLevel.IGNORE)
            ast_correct = sqlglot.parse_one(qualified_correct_sql, dialect='sqlite',
                                           error_level=sqlglot.ErrorLevel.IGNORE)
            
            if not ast_incorrect or not ast_correct:
                logger.error("Failed to parse one or both SQLs for masking")
                return qualified_incorrect_sql, qualified_correct_sql, {}
            
            # Phase 1: Build unified mapping by traversing both ASTs
            self._build_unified_mapping(ast_incorrect, ast_correct)
            
            # Phase 2: Apply masking to both ASTs
            masked_ast_incorrect = self._mask_ast(ast_incorrect)
            masked_ast_correct = self._mask_ast(ast_correct)
            
            # Generate masked SQL strings
            masked_incorrect = masked_ast_incorrect.sql(dialect='sqlite')
            masked_correct = masked_ast_correct.sql(dialect='sqlite')
            
            logger.debug(f"Masked incorrect: {masked_incorrect}")
            logger.debug(f"Masked correct: {masked_correct}")
            logger.debug(f"Mapping: {self.mapping_dict}")
            
            return masked_incorrect, masked_correct, self.mapping_dict
            
        except Exception as e:
            logger.error(f"SQL masking failed: {e}")
            return qualified_incorrect_sql, qualified_correct_sql, {}
    
    def check_value_only_difference(self, 
                                    masked_incorrect: str,
                                    masked_correct: str) -> bool:
        """
        Check if any filter condition uses different values for the same column.
        
        Logic:
        - Extract all column-value pairs from filter conditions (WHERE/HAVING)
        - For each column that appears in both SQLs, check if values differ
        - If ANY column uses different values → True (value-only error, should filter)
        - If all common columns use same values → False (not value-only error)
        
        Example:
        Incorrect: WHERE T1.C4 = V2 AND T1.C5 = V3
        Correct:   WHERE T1.C4 = V4 AND T1.C5 = V3
        → Column C4 uses different values (V2 vs V4) → Return True
        
        Args:
            masked_incorrect: Masked incorrect SQL
            masked_correct: Masked correct SQL
        
        Returns:
            True if filter values differ for same columns, False otherwise
        """
        import re
        
        def extract_column_value_pairs(sql):
            """
            Extract column-value pairs from filter conditions.
            
            Matches patterns like:
            - T1.C4 = V2
            - T2.C3 < V1
            - T1.C5 IN (V1, V2)
            
            Returns:
                Dict mapping (column, operator) to list of values
            """
            # Pattern: (table.column) (operator) (value)
            # Handles =, <, >, <=, >=, !=, <>, LIKE, IN, etc.
            pattern = r'(T\d+\.C\d+)\s*([=<>!]+|LIKE|IN|BETWEEN|IS)\s*(V\d+)'
            matches = re.findall(pattern, sql, re.IGNORECASE)
            
            column_values = {}
            for column, operator, value in matches:
                # Normalize operator (treat = and == as same, etc.)
                normalized_op = operator.strip().upper()
                if normalized_op in ['==', 'IS']:
                    normalized_op = '='
                
                key = f"{column} {normalized_op}"
                if key not in column_values:
                    column_values[key] = set()
                column_values[key].add(value)
            
            return column_values
        
        cv_incorrect = extract_column_value_pairs(masked_incorrect)
        cv_correct = extract_column_value_pairs(masked_correct)
        
        logger.debug(f"Incorrect filter conditions: {cv_incorrect}")
        logger.debug(f"Correct filter conditions: {cv_correct}")
        
        # Find columns that appear in both SQLs
        common_columns = set(cv_incorrect.keys()) & set(cv_correct.keys())
        
        if not common_columns:
            # No common filtered columns
            logger.debug("No common filter columns found")
            return False
        
        # Check if any common column has different values
        for col_key in common_columns:
            values_incorrect = cv_incorrect[col_key]
            values_correct = cv_correct[col_key]
            
            if values_incorrect != values_correct:
                logger.info(f"✗ Detected value difference in filter condition:")
                logger.info(f"  Column: {col_key}")
                logger.info(f"  Incorrect values: {values_incorrect}")
                logger.info(f"  Correct values: {values_correct}")
                return True
        
        logger.debug(f"All {len(common_columns)} common filter columns use same values")
        return False
    
    def _reset(self):
        """Reset all counters and mappings"""
        self.table_counter = 0
        self.column_counter = 0
        self.value_counter = 0
        
        self.table_map = {}
        self.column_map = {}
        self.value_map = {}
        self.mapping_dict = {}
    
    def _build_unified_mapping(self, *asts):
        """
        Build unified mapping by traversing all ASTs
        
        This ensures the same entity gets the same mask across all SQLs
        """
        for ast in asts:
            for node in ast.walk():
                # Map tables
                if isinstance(node, expressions.Table):
                    table_name = str(node.name).lower()
                    if table_name not in self.table_map:
                        self.table_counter += 1
                        mask = f"T{self.table_counter}"
                        self.table_map[table_name] = mask
                        self.mapping_dict[f"table:{table_name}"] = mask
                
                # Map columns (table.column as key for uniqueness)
                elif isinstance(node, expressions.Column):
                    table_ref = str(node.table).lower() if node.table else ""
                    column_name = str(node.name).lower()
                    
                    # Use table.column as key if table is present
                    if table_ref:
                        column_key = f"{table_ref}.{column_name}"
                    else:
                        column_key = column_name
                    
                    if column_key not in self.column_map:
                        self.column_counter += 1
                        mask = f"C{self.column_counter}"
                        self.column_map[column_key] = mask
                        self.mapping_dict[f"column:{column_key}"] = mask
                
                # Map literal values
                elif isinstance(node, expressions.Literal):
                    # Skip NULL literals
                    if hasattr(node, 'is_null') and node.is_null:
                        continue
                    
                    value = str(node.this)
                    
                    # Check if it's NULL by value
                    if value.upper() == 'NULL':
                        continue
                    
                    # Normalize string values (remove quotes)
                    is_string = getattr(node, 'is_string', False) or (value.startswith("'") and value.endswith("'"))
                    if is_string:
                        value = value.strip("'\"")
                    
                    if value not in self.value_map:
                        self.value_counter += 1
                        mask = f"V{self.value_counter}"
                        self.value_map[value] = mask
                        self.mapping_dict[f"value:{value}"] = mask
    
    def _mask_ast(self, ast):
        """
        Apply masking transformations to AST
        
        Args:
            ast: sqlglot AST node
        
        Returns:
            Masked AST
        """
        def mask_node(node):
            # Mask table names
            if isinstance(node, expressions.Table):
                table_name = str(node.name).lower()
                masked_name = self.table_map.get(table_name, table_name)
                node.set('this', expressions.Identifier(this=masked_name, quoted=False))
                # Remove alias
                node.set('alias', None)
            
            # Mask column names
            elif isinstance(node, expressions.Column):
                table_ref = str(node.table).lower() if node.table else ""
                column_name = str(node.name).lower()
                
                # Get masked column name
                if table_ref:
                    column_key = f"{table_ref}.{column_name}"
                    masked_table = self.table_map.get(table_ref, table_ref)
                else:
                    column_key = column_name
                    masked_table = None
                
                masked_column = self.column_map.get(column_key, column_name)
                
                # Update node
                node.set('this', expressions.Identifier(this=masked_column, quoted=False))
                if masked_table:
                    node.set('table', expressions.Identifier(this=masked_table, quoted=False))
                
                # Remove alias
                if hasattr(node, 'alias') and node.alias:
                    node.set('alias', None)
            
            # Mask literal values
            elif isinstance(node, expressions.Literal):
                # Skip NULL
                if hasattr(node, 'is_null') and node.is_null:
                    return node
                
                value = str(node.this)
                
                # Check if it's NULL by value
                if value.upper() == 'NULL':
                    return node
                
                # Normalize string values
                is_string = getattr(node, 'is_string', False) or (value.startswith("'") and value.endswith("'"))
                if is_string:
                    value = value.strip("'\"")
                
                masked_value = self.value_map.get(value, value)
                
                # Create new literal with masked value
                return expressions.Literal(this=masked_value, is_string=False)
            
            return node
        
        # Transform AST
        masked_ast = ast.transform(mask_node, copy=True)
        return masked_ast

