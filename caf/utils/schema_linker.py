# Schema Linker - Extract NL-to-SQL mappings with Generate-Verify-Revise pipeline

import json
import logging
import re
from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass
import sqlglot
from sqlglot import expressions
from sqlglot.optimizer.scope import build_scope

from caf.llm.client import BaseLLMClient
from caf.preprocess.sql_preprocessor import SQLPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class Mapping:
    """Single NL-to-SQL mapping"""
    type: str  # "PROJECTION", "FILTER", "RELATIONSHIP"
    nl_span: str  # Exact text from question
    sql_fragment: str  # SQL fragment (e.g., "table.col = val" or "table.col")
    columns: List[str]  # List of columns involved (e.g., ["table.col"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "type": self.type,
            "nl_span": self.nl_span,
            "sql_fragment": self.sql_fragment,
            "columns": self.columns
        }


@dataclass
class SchemaLinkerResult:
    """Result of schema linking"""
    mappings: List[Mapping]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "mappings": [m.to_dict() for m in self.mappings]
        }


class SchemaLinker:
    """
    Schema Linker - Extract NL-to-SQL mappings using Generate-Verify-Revise pipeline
    
    This class implements a three-phase approach:
    1. Phase 1: Deterministic SQL Parsing - Extract structural metadata from SQL
    2. Phase 2: Initial Mapping Generation - Use LLM to generate initial mappings
    3. Phase 3: Coverage Verification & Self-Correction - Verify completeness and fix gaps
    """
    
    def __init__(self, 
                 llm_client: BaseLLMClient,
                 max_retries: int = 2,
                 config: Optional[Dict[str, Any]] = None,
                 database_id: Optional[str] = None):
        """
        Initialize Schema Linker
        
        Args:
            llm_client: LLM client for generating mappings
            max_retries: Maximum number of retry attempts for verification
            config: Optional configuration dictionary
            database_id: Optional database ID for SQL qualification
        """
        self.llm_client = llm_client
        self.max_retries = max_retries
        self.config = config or {}
        
        # LLM configuration
        self.temperature = self.config.get('temperature', 0.1)
        self.max_tokens = self.config.get('max_tokens', 4000)
        self.model = self.config.get('model', None)  # Use LLM client's default if None
        
        # Initialize SQL preprocessor for SQL normalization
        self.sql_preprocessor = SQLPreprocessor(
            case_sensitive=False,
            database_id=database_id,
            use_schema_cache=True
        )
        
        self.database_id = database_id

        logger.debug("SchemaLinker initialized")
    
    def _extract_structural_metadata(self, sql: str) -> Tuple[List[Mapping], Set[str]]:
        """
        Phase 1: Deterministic SQL Parsing (Python Rule-based)
        
        Extract structural metadata from SQL:
        1. JOIN relationships (as Relationship mappings)
        2. All columns (as Ground Truth for verification)
        3. Join key columns (to exclude from verification set)
        
        Args:
            sql: SQL query string
            
        Returns:
            (rule_mappings, target_columns): Rule-based mappings and columns to verify
        """
        try:
            # Parse SQL
            parsed = sqlglot.parse_one(sql, dialect='sqlite', error_level=sqlglot.ErrorLevel.IGNORE)
            if parsed is None:
                logger.warning(f"Failed to parse SQL: {sql[:100]}...")
                return [], set()
            
            # Build scope to resolve table names and aliases
            try:
                scope = build_scope(parsed)
                # Try to qualify columns (resolve table names and aliases)
                try:
                    from sqlglot.optimizer import qualify
                    parsed = qualify.qualify(parsed)
                except (ImportError, AttributeError):
                    # Fallback: manually resolve using scope if qualify is not available
                    logger.debug("qualify function not available, using scope-based resolution")
            except Exception as e:
                logger.debug(f"Scope building/qualification failed, continuing without: {e}")
            
            mappings = []
            all_columns = set()
            join_key_columns = set()
            
            # 1. Extract JOIN relationships
            for join in parsed.find_all(expressions.Join):
                on_condition = join.args.get('on')
                if on_condition:
                    # Extract columns in JOIN condition
                    cols_in_join = []
                    tables_in_join = set()  # Extract unique table names
                    for col in on_condition.find_all(expressions.Column):
                        col_str = self._format_column(col)
                        cols_in_join.append(col_str)
                        join_key_columns.add(col_str)
                        
                        # Extract table name from column
                        if col.table:
                            table_name = str(col.table).lower()
                            tables_in_join.add(table_name)
                    
                    if cols_in_join:
                        # Create nl_span as table names joined by underscore: table1_table2
                        if len(tables_in_join) >= 2:
                            tables_list = sorted(list(tables_in_join))  # Sort for consistency
                            nl_span = "_".join(tables_list)  # e.g., "frpm_satscores"
                        elif len(tables_in_join) == 1:
                            # Only one table found (shouldn't happen in JOIN, but handle gracefully)
                            nl_span = list(tables_in_join)[0]
                        else:
                            # Fallback if no tables found
                            nl_span = "IMPLICIT_RELATION"
                        
                        # Record as JOIN Mapping
                        mappings.append(Mapping(
                            type="RELATIONSHIP",
                            nl_span=nl_span,
                            sql_fragment=on_condition.sql(dialect='sqlite'),
                            columns=cols_in_join
                        ))
            
            # 2. Extract all columns from various clauses
            # SELECT clause
            for select_node in parsed.find_all(expressions.Select):
                for expr in select_node.args.get('expressions', []):
                    for col in expr.find_all(expressions.Column):
                        col_str = self._format_column(col)
                        all_columns.add(col_str)
            
            # WHERE clause
            where_node = parsed.find(expressions.Where)
            if where_node:
                for col in where_node.find_all(expressions.Column):
                    col_str = self._format_column(col)
                    all_columns.add(col_str)
            
            # GROUP BY clause
            group_node = parsed.find(expressions.Group)
            if group_node:
                for expr in group_node.expressions if hasattr(group_node, 'expressions') else []:
                    for col in expr.find_all(expressions.Column):
                        col_str = self._format_column(col)
                        all_columns.add(col_str)
            
            # HAVING clause
            having_node = parsed.find(expressions.Having)
            if having_node:
                for col in having_node.find_all(expressions.Column):
                    col_str = self._format_column(col)
                    all_columns.add(col_str)
            
            # ORDER BY clause
            order_node = parsed.find(expressions.Order)
            if order_node:
                for expr in order_node.expressions if hasattr(order_node, 'expressions') else []:
                    for col in expr.find_all(expressions.Column):
                        col_str = self._format_column(col)
                        all_columns.add(col_str)
            
            # 3. Calculate target verification set (exclude pure JOIN keys)
            # Strategy: If a column appears only in JOIN ON and nowhere else,
            # it's a structural join key and doesn't need NL mapping
            # For simplicity, we exclude all join key columns from verification
            # unless they also appear in SELECT/WHERE (which we handle by keeping them in all_columns)
            target_verification_set = all_columns - join_key_columns
            
            logger.info(f"Phase 1: Extracted {len(mappings)} rule-based mappings")
            logger.info(f"  All columns found: {sorted(all_columns)}")
            logger.info(f"  Join key columns (excluded): {sorted(join_key_columns)}")
            logger.info(f"  Target columns for verification: {sorted(target_verification_set)}")
            
            return mappings, target_verification_set
            
        except Exception as e:
            logger.error(f"Error extracting structural metadata: {e}")
            return [], set()
    
    def _format_column(self, col: expressions.Column) -> str:
        """Format column as 'table.column' (always qualified)"""
        if col.table:
            return f"{col.table}.{col.name}"
        # If no table, return just column name (shouldn't happen after qualification)
        return str(col.name)
    
    def _normalize_column_name(self, col_name: str) -> str:
        """
        Normalize column name to table.column format.
        
        If column is already in table.column format, return as is.
        If column is just 'column', try to resolve using schema.
        """
        # If already in table.column format, return as is
        if '.' in col_name:
            return col_name
        
        # Try to resolve using schema if available
        if self.schema:
            col_name_lower = col_name.lower()
            possible_tables = self.schema.get(col_name_lower, set())
            if len(possible_tables) == 1:
                table_name = list(possible_tables)[0]
                return f"{table_name}.{col_name}"
            elif len(possible_tables) > 1:
                # Ambiguous, return first one
                table_name = list(possible_tables)[0]
                logger.debug(f"Ambiguous column '{col_name}', using table '{table_name}'")
                return f"{table_name}.{col_name}"
        
        # Can't resolve, return as is
        return col_name
    
    def _construct_prompt(self, 
                         question: str, 
                         sql: str, 
                         evidence: Optional[str] = None,
                         missed_columns: Optional[List[str]] = None) -> Tuple[str, str]:
        """
        Construct prompt for LLM mapping generation
        
        Args:
            question: Natural language question
            sql: SQL query
            evidence: Optional evidence/context
            missed_columns: Optional list of columns that were missed in previous attempt
            
        Returns:
            (system_prompt, user_prompt): Prompt pair
        """
        system_prompt = """You are an expert in SQL-to-Text grounding. 

Task: Extract mappings between Natural Language phrases and SQL components.

Rules:
1. SQL Anchoring: Look at each SQL column/value. Find the exact word or phrase in the Question that triggered it.
2. Granularity: Map to specific values or entity names. (e.g., 'San Bernardino', not 'County').
3. One-to-Many Mapping (CRITICAL): If the Question uses a collective term (e.g., "email addresses", "valid e-mail addresses"), you MUST create a SINGLE mapping that includes ALL related SQL columns in the "columns" array. For example:
   - "valid e-mail addresses" -> {"columns": ["schools.AdmEmail1", "schools.AdmEmail2"]}
   - NOT two separate mappings for each column
4. Implicit Logic: Use Evidence to map codes (e.g., SOC=62) to their text descriptions.
5. Completeness: Ensure ALL business-relevant columns in SQL are mapped. Don't miss any columns that appear in SELECT or WHERE clauses.
6. Column Format: Always use "table.column" format (e.g., "schools.County", not just "County").

Output JSON Format:
[
  {
    "type": "PROJECTION" | "FILTER",
    "nl_span": "exact text from question",
    "sql_fragment": "table.col = val" or "table.col",
    "columns": ["table.col1", "table.col2"]  // For one-to-many, include ALL columns in ONE mapping
  }
]

Important Examples:
- One-to-Many: "valid e-mail addresses" should map to ONE mapping with columns: ["schools.AdmEmail1", "schools.AdmEmail2"]
- Single Column: "San Bernardino county" maps to ONE mapping with columns: ["schools.County"]

Note: 
- PROJECTION type is for columns in SELECT clause
- FILTER type is for columns/conditions in WHERE clause
- Do NOT include JOIN relationships (RELATIONSHIP type) - those are handled separately
- The "nl_span" should be the exact text from the question that corresponds to the SQL component"""

        user_prompt = f"""Question: {question}

SQL: {sql}
"""
        
        if evidence:
            user_prompt += f"\nEvidence: {evidence}"
        
        if missed_columns:
            user_prompt += f"\n\nCRITICAL WARNING: In your previous attempt, you missed the following SQL columns which are crucial for the query logic. You MUST find the text in the Question that corresponds to them:\nMissed Columns: {', '.join(missed_columns)}\n\nPlease provide mappings for ALL missed columns."
        
        return system_prompt, user_prompt
    
    def _parse_llm_response(self, response_str: str) -> List[Mapping]:
        """
        Parse LLM response into Mapping objects
        
        Args:
            response_str: LLM response string (may contain markdown code blocks)
            
        Returns:
            List of Mapping objects
        """
        print('--------------------------------')
        print(response_str)
        try:
            # 参考 column_retriever.py 的方式：使用正则表达式提取 JSON
            # 首先尝试提取 ```json ... ``` 中的内容
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', response_str, re.DOTALL)
            if not json_match:
                # 尝试提取 ``` ... ``` 中的内容
                json_match = re.search(r'```\s*(\[.*?\])\s*```', response_str, re.DOTALL)
            if not json_match:
                # 尝试直接匹配 JSON 数组
                json_match = re.search(r'(\[.*?\])', response_str, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
            else:
                # 如果找不到 JSON，尝试直接解析整个响应
                logger.warning("No JSON pattern found, trying to parse entire response as JSON")
                data = json.loads(response_str.strip())
            
            # Convert to Mapping objects and normalize column names
            mappings = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # Normalize all column names to table.column format
                        columns = item.get("columns", [])
                        normalized_columns = [self._normalize_column_name(col) for col in columns]
                        
                        mappings.append(Mapping(
                            type=item.get("type", "UNKNOWN"),
                            nl_span=item.get("nl_span", ""),
                            sql_fragment=item.get("sql_fragment", ""),
                            columns=normalized_columns
                        ))
            elif isinstance(data, dict) and "mappings" in data:
                # 如果返回的是 {"mappings": [...]} 格式
                for item in data.get("mappings", []):
                    if isinstance(item, dict):
                        # Normalize all column names to table.column format
                        columns = item.get("columns", [])
                        normalized_columns = [self._normalize_column_name(col) for col in columns]
                        
                        mappings.append(Mapping(
                            type=item.get("type", "UNKNOWN"),
                            nl_span=item.get("nl_span", ""),
                            sql_fragment=item.get("sql_fragment", ""),
                            columns=normalized_columns
                        ))
            
            logger.debug(f"Parsed {len(mappings)} mappings from LLM response")
            return mappings
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.error(f"Response (first 500 chars): {response_str[:500]}...")
            return []
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Response (first 500 chars): {response_str[:500]}...")
            import traceback
            logger.debug(traceback.format_exc())
            return []
    
    def _verify_coverage(self, 
                        llm_mappings: List[Mapping], 
                        target_columns: Set[str]) -> Tuple[bool, Set[str]]:
        """
        Verify coverage of LLM mappings against target columns
        
        Args:
            llm_mappings: Mappings generated by LLM 
            target_columns: Set of columns that should be covered
            
        Returns:
            (is_complete, missing_columns): Whether all columns are covered and list of missing ones
        """
        # Extract columns covered by LLM mappings
        # Note: mapping.columns is a list, so we need to flatten and convert to set
        # Also normalize to lowercase for case-insensitive comparison
        covered_columns = {col.lower() for mapping in llm_mappings for col in mapping.columns}
        
        # Normalize target columns to lowercase for case-insensitive comparison
        normalized_target_columns = {col.lower() for col in target_columns}
        
        # Calculate missing columns
        missing = normalized_target_columns - covered_columns
        
        is_complete = len(missing) == 0
        
        if not is_complete:
            logger.info(f"Coverage verification failed. Missing {len(missing)} columns:")
            logger.info(f"  Target columns ({len(normalized_target_columns)}): {sorted(normalized_target_columns)}")
            logger.info(f"  Covered columns ({len(covered_columns)}): {sorted(covered_columns)}")
            logger.info(f"  Missing columns ({len(missing)}): {sorted(missing)}")
        else:
            logger.info("Coverage verification passed! All columns are covered.")
        
        return is_complete, missing
    
    def run(self, 
            question: str, 
            sql: str, 
            evidence: Optional[str] = None) -> SchemaLinkerResult:
        """
        Main pipeline: Generate-Verify-Revise
        
        Args:
            question: Natural language question
            sql: SQL query
            evidence: Optional evidence/context (e.g., "Intermediate Schools refers to SOC = 62")
            
        Returns:
            SchemaLinkerResult containing all mappings
        """
        logger.info(f"Processing question: {question[:100]}...")
        
        # Normalize SQL using SQLPreprocessor
        normalized_sql = self.sql_preprocessor.qualify_sql(sql)
        
        rule_mappings, target_columns = self._extract_structural_metadata(normalized_sql)

        # Phase 2 & 3: Initial Generation + Verification Loop
        llm_mappings = []
        missed_columns = None
        
        for attempt in range(self.max_retries + 1):
            sys_prompt, user_prompt = self._construct_prompt(
                question, normalized_sql, evidence, missed_columns
            )
            
            try:
                full_prompt = f"{sys_prompt}\n\n{user_prompt}"
                
                # 直接使用 call_with_messages，简单直接
                response_str = self.llm_client.call_with_messages(
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                logger.debug(f"LLM response (first 500 chars): {response_str[:500]}...")
                
                # Parse response
                attempt_mappings = self._parse_llm_response(response_str)
                logger.info(f"Attempt {attempt + 1}: Parsed {len(attempt_mappings)} mappings from LLM")
                
                existing_columns = {col for m in llm_mappings for col in m.columns}
                new_mappings_count = 0
                
                for mapping in attempt_mappings:
                    # Columns are already normalized, so use them directly
                    mapping_cols = set(mapping.columns)
                    
                    # Check if this mapping adds new columns
                    if not mapping_cols.issubset(existing_columns):
                        # Check for duplicate mappings (same nl_span and columns)
                        is_duplicate = False
                        for existing_mapping in llm_mappings:
                            if (existing_mapping.nl_span == mapping.nl_span and 
                                set(existing_mapping.columns) == mapping_cols):
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            llm_mappings.append(mapping)
                            existing_columns.update(mapping_cols)
                            new_mappings_count += 1
                            logger.debug(f"  Added mapping: {mapping.type} - {mapping.nl_span} -> {mapping.columns}")
                        else:
                            logger.debug(f"  Skipped duplicate mapping: {mapping.nl_span} -> {mapping.columns}")
                    else:
                        logger.debug(f"  Skipped mapping (all columns already covered): {mapping.nl_span} -> {mapping.columns}")
                
                if new_mappings_count > 0:
                    logger.info(f"  Added {new_mappings_count} new mappings (total: {len(llm_mappings)})")
                else:
                    logger.info(f"  No new mappings added (all {len(attempt_mappings)} were duplicates or already covered)")
                
            except Exception as e:
                logger.error(f"LLM call failed on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries:
                    logger.warning("Max retries reached. Returning partial results.")
                    break
                continue
            
            # Phase 3: Verify Coverage
            is_complete, missing = self._verify_coverage(llm_mappings, target_columns)
            
            if is_complete:
                logger.info("Verification passed! All columns are covered.")
                break
            
            if attempt < self.max_retries:
                missed_columns = list(missing)
                logger.info(f"Verification failed. Retrying with {len(missed_columns)} missed columns...")
            else:
                logger.warning(f"Max retries reached. {len(missing)} columns still missing: {missing}")
        
        # Merge rule-based and LLM mappings
        final_mappings = rule_mappings + llm_mappings
        
        logger.info(f"Generated {len(final_mappings)} total mappings ({len(rule_mappings)} rule-based, {len(llm_mappings)} LLM-generated)")
        
        return SchemaLinkerResult(mappings=final_mappings)
