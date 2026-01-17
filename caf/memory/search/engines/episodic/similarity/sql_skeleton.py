# SQL Skeleton Similarity Calculator (from DAIL-SQL and CAF)
# SQL Skeleton Generation - Adapted from DAIL-SQL
# This module generates SQL skeletons by replacing schema-specific content with placeholders

import re
import logging
import collections
from typing import Optional, Dict, Any, List
from sql_metadata import Parser
from caf.memory.types import MemoryType

logger = logging.getLogger(__name__)


# ============================================================================
# SQL Skeleton Generation Functions
# ============================================================================

def normalize_sql(sql: str) -> str:
    """
    Normalize SQL query by standardizing format, case, and whitespace.
    Adapted from DAIL-SQL utils/utils.py
    """
    sql = sql.strip()
    
    def white_space_fix(s):
        try:
            parsed_s = Parser(s)
            s = " ".join([token.value for token in parsed_s.tokens])
            return s
        except Exception:
            # Fallback to simple whitespace normalization
            return ' '.join(s.split())

    def lower_case_except_quotes(s):
        """Convert everything except text between single quotation marks to lower case"""
        in_quotation = False
        out_s = ""
        for char in s:
            if in_quotation:
                out_s += char
            else:
                out_s += char.lower()

            if char == "'":
                in_quotation = not in_quotation
        return out_s

    def remove_semicolon(s):
        return s.rstrip(";")

    def double_to_single_quotes(s):
        return s.replace('"', "'")

    def add_asc(s):
        """Add ASC to ORDER BY clauses that don't specify direction"""
        pattern = re.compile(r'order by (?:\w+ \( \S+ \)|\w+\.\w+|\w+)(?: (?:\+|\-|\<|\<\=|\>|\>\=) (?:\w+ \( \S+ \)|\w+\.\w+|\w+))*')
        if "order by" in s and "asc" not in s and "desc" not in s:
            for p_str in pattern.findall(s):
                s = s.replace(p_str, p_str + " asc")
        return s

    # Apply normalization pipeline
    processing_func = lambda x: add_asc(
        lower_case_except_quotes(
            white_space_fix(
                double_to_single_quotes(
                    remove_semicolon(x)
                )
            )
        )
    )
    
    return processing_func(sql)


def generate_sql_skeleton(sql: str, schema_info: Dict[str, Any]) -> str:
    """
    Generate SQL skeleton by replacing schema-specific content with placeholders.
    Adapted from DAIL-SQL utils/utils.py sql2skeleton function.
    
    Args:
        sql: SQL query string
        schema_info: Dictionary containing schema information with keys:
            - table_names_original: List of table names
            - column_names_original: List of [table_id, column_name] pairs
            
    Returns:
        SQL skeleton string with schema elements replaced by '_'
    """
    try:
        sql = normalize_sql(sql)
        
        # Extract schema information
        table_names_original = []
        table_dot_column_names_original = []
        column_names_original = ["*"]  # Always include wildcard
        
        # Process table names
        if "table_names_original" in schema_info:
            for table_name in schema_info["table_names_original"]:
                table_name_lower = table_name.lower()
                table_names_original.append(table_name_lower)
                table_dot_column_names_original.append(table_name_lower + ".*")
        
        # Process column names
        if "column_names_original" in schema_info:
            for column_info in schema_info["column_names_original"]:
                if len(column_info) >= 2:
                    table_id = column_info[0]
                    column_name = column_info[1].lower()
                    column_names_original.append(column_name)
                    
                    # Add table.column format if table_id is valid
                    if (table_id >= 0 and table_id < len(table_names_original)):
                        table_name = table_names_original[table_id]
                        table_dot_column_names_original.append(f"{table_name}.{column_name}")
        
        # Parse SQL and replace tokens
        # sql_metadata/sqlparse struggle with MySQL-style backticks (especially when
        # the identifier itself contains parentheses). Switch them to standard
        # double quotes before parsing so tokenizer does not crash with IndexError.
        sanitized_sql = sql.replace("`", '"')
        parsed_sql = Parser(sanitized_sql)
        new_sql_tokens = []
        
        for token in parsed_sql.tokens:
            token_value = token.value.lower()
            token_value_stripped = token_value.strip('`"')
            
            # Replace table names
            if (token_value in table_names_original or 
                token_value_stripped in table_names_original):
                new_sql_tokens.append("_")
            # Replace column names (including table.column format)
            elif (token_value in column_names_original or 
                  token_value in table_dot_column_names_original or
                  token_value_stripped in column_names_original or
                  token_value_stripped in table_dot_column_names_original):
                new_sql_tokens.append("_")
            # Replace string values (quoted)
            elif token.value.startswith("'") and token.value.endswith("'"):
                new_sql_tokens.append("_")
            # Replace positive integers
            elif token.value.isdigit():
                new_sql_tokens.append("_")
            # Replace negative integers
            elif _is_negative_int(token.value):
                new_sql_tokens.append("_")
            # Replace floating point numbers
            elif _is_float(token.value):
                new_sql_tokens.append("_")
            else:
                new_sql_tokens.append(token.value.strip())
        
        sql_skeleton = " ".join(new_sql_tokens)
        
        # Clean up skeleton patterns
        sql_skeleton = _clean_skeleton_patterns(sql_skeleton)
        
        return sql_skeleton
        
    except Exception as e:
        logger.warning(f"Failed to generate SQL skeleton: {e}. Returning normalized SQL.")
        return normalize_sql(sql)


def _is_negative_int(s: str) -> bool:
    """Check if string represents a negative integer"""
    return s.startswith("-") and s[1:].isdigit()


def _is_float(s: str) -> bool:
    """Check if string represents a floating point number"""
    if s.startswith("-"):
        s = s[1:]
    
    parts = s.split(".")
    if len(parts) > 2:
        return False
    
    return all(part.isdigit() for part in parts)


def _clean_skeleton_patterns(sql_skeleton: str) -> str:
    """
    Clean up skeleton patterns to make them more canonical.
    Adapted from DAIL-SQL skeleton cleaning logic.
    """
    # Remove JOIN ON patterns
    sql_skeleton = sql_skeleton.replace("on _ = _ and _ = _", "on _ = _")
    sql_skeleton = sql_skeleton.replace("on _ = _ or _ = _", "on _ = _")
    sql_skeleton = sql_skeleton.replace(" on _ = _", "")
    
    # Clean up multiple JOIN patterns
    join_pattern = re.compile(r"_ (?:join _ ?)+")
    sql_skeleton = re.sub(join_pattern, "_ ", sql_skeleton)
    
    # Collapse comma-separated placeholders: "_ , _ , ..., _" -> "_"
    while "_ , _" in sql_skeleton:
        sql_skeleton = sql_skeleton.replace("_ , _", "_")
    
    # Remove comparison operators in WHERE clauses
    ops = ["=", "!=", ">", ">=", "<", "<="]
    for op in ops:
        pattern = f"_ {op} _"
        if pattern in sql_skeleton:
            sql_skeleton = sql_skeleton.replace(pattern, "_")
    
    # Clean up WHERE conditions
    while ("where _ and _" in sql_skeleton or "where _ or _" in sql_skeleton):
        sql_skeleton = sql_skeleton.replace("where _ and _", "where _")
        sql_skeleton = sql_skeleton.replace("where _ or _", "where _")
    
    # Remove extra spaces
    while "  " in sql_skeleton:
        sql_skeleton = sql_skeleton.replace("  ", " ")
    
    # Final check for ORDER BY
    split_skeleton = sql_skeleton.split(" ")
    for i in range(2, len(split_skeleton)):
        if (split_skeleton[i-2] == "order" and 
            split_skeleton[i-1] == "by" and 
            split_skeleton[i] != "_"):
            split_skeleton[i] = "_"
    
    return " ".join(split_skeleton).strip()


def jaccard_similarity(skeleton1: str, skeleton2: str) -> float:
    """
    Calculate Jaccard similarity between two SQL skeletons.
    Adapted from DAIL-SQL utils/utils.py
    """
    if not skeleton1 or not skeleton2:
        return 0.0
    
    tokens1 = skeleton1.strip().split()
    tokens2 = skeleton2.strip().split()
    
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    
    def list_to_dict(tokens):
        token_dict = collections.defaultdict(int)
        for token in tokens:
            token_dict[token] += 1
        return token_dict
    
    token_dict1 = list_to_dict(tokens1)
    token_dict2 = list_to_dict(tokens2)
    
    # Calculate intersection
    intersection = 0
    for token in token_dict1:
        if token in token_dict2:
            intersection += min(token_dict1[token], token_dict2[token])
    
    # Calculate union
    union = len(tokens1) + len(tokens2) - intersection
    
    return float(intersection) / union if union > 0 else 0.0


def extract_schema_info_from_semantic_memory(semantic_store) -> Optional[Dict[str, Any]]:
    """
    Extract schema information from semantic memory store in the format expected by generate_sql_skeleton.
    
    Args:
        semantic_store: SemanticMemoryStore instance
        
    Returns:
        Dictionary with table_names_original and column_names_original keys
    """
    try:
        if not hasattr(semantic_store, 'dataframes') or not semantic_store.dataframes:
            logger.warning("Semantic memory dataframes not available")
            return None
        
        if 'table' not in semantic_store.dataframes or 'column' not in semantic_store.dataframes:
            logger.warning("Table or column metadata not found in semantic memory")
            return None
        
        table_df = semantic_store.dataframes['table']
        column_df = semantic_store.dataframes['column']
        
        if table_df.empty or column_df.empty:
            logger.warning("Empty table or column metadata in semantic memory")
            return None
        
        # Extract table names
        table_names_original = table_df['table_name'].unique().tolist()
        
        # Extract column names in the format expected by skeleton generation
        # Format: [[table_id, column_name], ...]
        column_names_original = []
        
        # Create table_name to id mapping
        table_name_to_id = {name: idx for idx, name in enumerate(table_names_original)}
        
        for _, row in column_df.iterrows():
            table_name = row.get('table_name')
            column_name = row.get('column_name')
            
            if table_name and column_name:
                table_id = table_name_to_id.get(table_name, -1)
                column_names_original.append([table_id, column_name])
        
        schema_info = {
            "table_names_original": table_names_original,
            "column_names_original": column_names_original
        }
        
        logger.debug(f"Extracted schema info: {len(table_names_original)} tables, {len(column_names_original)} columns")
        return schema_info
        
    except Exception as e:
        logger.error(f"Failed to extract schema info from semantic memory: {e}")
        return None


# ============================================================================
# SQLSkeletonSimilarity Class
# ============================================================================

class SQLSkeletonSimilarity:
    """
    SQL Skeleton similarity calculator using Jaccard similarity
    
    This uses the SQL skeleton generation and Jaccard similarity
    calculation functions defined in this module, maintaining compatibility with
    DAIL-SQL's approach.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('sql_skeleton', {})
        
        # Use the functions defined in this module
        self._generate_sql_skeleton = generate_sql_skeleton
        self._jaccard_similarity = jaccard_similarity 
        self._extract_schema_info = extract_schema_info_from_semantic_memory
        
        # Cache for generated skeletons
        self._skeleton_cache: Dict[str, str] = {}
        
        # Schema info for skeleton generation
        self._schema_info: Optional[Dict[str, Any]] = None
        self._memory_base = None  # Set by parent engine
        
        logger.debug("SQLSkeletonSimilarity initialized")
    
    def set_memory_base(self, memory_base) -> None:
        """Set memory base for schema extraction"""
        self._memory_base = memory_base
    
    def calculate_similarity(self, sql1: str, sql2: str) -> float:
        """
        Calculate Jaccard similarity between two SQL queries using their skeletons
        
        Args:
            sql1: First SQL query
            sql2: Second SQL query
            
        Returns:
            Jaccard similarity score between 0 and 1
        """
        if not sql1 or not sql2:
            return 0.0
        
        try:
            # Generate skeletons
            skeleton1 = self._get_skeleton_cached(sql1)
            skeleton2 = self._get_skeleton_cached(sql2)
            
            if not skeleton1 or not skeleton2:
                return 0.0
            
            # Calculate Jaccard similarity
            similarity = self._jaccard_similarity(skeleton1, skeleton2)
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.warning(f"SQL skeleton similarity calculation failed: {e}")
            return 0.0
    
    def _get_skeleton_cached(self, sql: str) -> Optional[str]:
        """Get SQL skeleton with caching"""
        cache_key = hash(sql.strip())
        
        if cache_key in self._skeleton_cache:
            return self._skeleton_cache[cache_key]
        
        skeleton = self._generate_skeleton_safe(sql)
        if skeleton:
            self._skeleton_cache[cache_key] = skeleton
        
        return skeleton
    
    def _generate_skeleton_safe(self, sql: str) -> Optional[str]:
        """Safely generate SQL skeleton with error handling"""
        try:
            # Extract schema info if not available
            if not self._schema_info and self._memory_base:
                self._extract_schema_info_from_memory()
            
            if not self._schema_info:
                logger.warning("No schema info available for skeleton generation")
                return None
            
            skeleton = self._generate_sql_skeleton(sql, self._schema_info)
            logger.debug(f"Generated SQL skeleton: {skeleton}")
            return skeleton
            
        except Exception as e:
            logger.warning(f"Failed to generate SQL skeleton: {e}")
            return None
    
    def _extract_schema_info_from_memory(self) -> None:
        """Extract schema info from semantic memory"""
        if not self._memory_base:
            return
        
        try:
            semantic_store = self._memory_base.memory_stores.get(MemoryType.SEMANTIC)
            if not semantic_store:
                logger.warning("No semantic memory store available")
                return
            
            self._schema_info = self._extract_schema_info(semantic_store)
            if self._schema_info:
                logger.debug("Extracted schema info for SQL skeleton generation")
            
        except Exception as e:
            logger.warning(f"Failed to extract schema info: {e}")
    
    def is_ready(self) -> bool:
        """Check if the similarity calculator is ready"""
        return (self._jaccard_similarity is not None and 
                self._generate_sql_skeleton is not None)
    
    def reset(self) -> None:
        """Reset internal state"""
        self._skeleton_cache.clear()
        self._schema_info = None
        logger.debug("SQLSkeletonSimilarity reset")
    
    def get_cache_size(self) -> int:
        """Get current cache size"""
        return len(self._skeleton_cache)
