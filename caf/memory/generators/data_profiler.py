# Data Profiler - Extract statistical metadata from actual data
# Highest priority source for statistical information

import logging
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Set
from collections import Counter, defaultdict
import json
import re

logger = logging.getLogger(__name__)

class DataProfiler:
    """
    Data Profiler - Extract statistical metadata from database content
    
    This profiler analyzes actual data to generate:
    - Row counts, null counts, distinct counts
    - Min/max values, min/max lengths
    - Top-K values and patterns
    - Common default values
    - Data quality issues
    - Semantic similarity hints
    
    Priority: Highest (ddl_extract for statistics)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_sample_size = self.config.get('max_sample_size', -1)
        self.top_k_limit = self.config.get('top_k_limit', 10)
        self.min_distinct_for_top_k = self.config.get('min_distinct_for_top_k', 2)
        self.sample_data_limit = self.config.get('sample_data_limit', 10)
        self.query_timeout = self.config.get('query_timeout', 100)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.1) 

        self.distinct_values_threshold = self.config.get('distinct_values_threshold', 10000)
        
        logger.debug("DataProfiler initialized")
    
    def profile_database(self, database_path: str, tables: List[str] = None, 
                        columns: Dict[str, List[str]] = None,
                        relationships: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Profile database and generate statistical metadata
        
        Args:
            database_path: Path to database file
            tables: List of table names to profile (None = all tables)
            columns: Dict of {table_name: [column_names]} to profile (None = all columns)
            relationships: List of relationship dicts that need cardinality calculation
            
        Returns:
            Dictionary containing profiling results
        """
        
        db_path = Path(database_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {database_path}")
        
        logger.info(f"Starting data profiling of database: {database_path}")
        
        conn = sqlite3.connect(f'file:{database_path}?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        
        try:
            # Get table list - either specified tables or all tables
            if tables is None or len(tables) == 0:
                tables = self._get_table_list(conn)
            
            logger.info(f"Profiling {len(tables)} tables: {tables}")
            
            results = {
                'tables': {},
                'columns': {},
                'database': self._get_database_stats(conn, tables)
            }
            
            # --- 第一阶段：收集所有表和列的统计信息 ---
            for table_name in tables:
                logger.debug(f"Profiling table: {table_name}")
                
                table_stats = self._profile_table(conn, table_name)
                results['tables'][table_name] = table_stats
                
                columns_to_profile = columns.get(table_name) if columns else None
                column_stats = self._profile_table_columns(conn, table_name, columns_to_profile)
                if column_stats:
                    results['columns'][table_name] = column_stats
            
            
            if relationships:
                logger.info(f"Calculating cardinality for {len(relationships)} relationships")
                relationship_results = []
                for rel in relationships:
                    source_table = rel.get('source_table')
                    target_table = rel.get('target_table')
                    source_columns = rel.get('source_columns', [])
                    target_columns = rel.get('target_columns', [])
                    
                    if not source_table or not target_table or not source_columns or not target_columns:
                        logger.warning(f"Skipping invalid relationship: {rel}")
                        continue
                    
                    # Ensure source_columns and target_columns are lists
                    if not isinstance(source_columns, list):
                        source_columns = [source_columns]
                    if not isinstance(target_columns, list):
                        target_columns = [target_columns]
                    
                    logger.debug(f"Computing cardinality for {source_table} -> {target_table}")
                    cardinality = self._compute_relationship_cardinality(
                        database_path, source_table, source_columns, target_table, target_columns
                    )
                    
                    if cardinality:
                        # Create relationship dict with cardinality
                        rel_result = {
                            'source_table': source_table,
                            'target_table': target_table,
                            'source_columns': source_columns,
                            'target_columns': target_columns,
                            'cardinality': cardinality,
                            'relationship_type': rel.get('relationship_type'),
                            'source': rel.get('source', 'ddl_extract')
                        }
                        relationship_results.append(rel_result)
                        logger.debug(f"Computed cardinality {cardinality} for {source_table} -> {target_table}")
                    else:
                        logger.warning(f"Failed to compute cardinality for {source_table} -> {target_table}")
                
                if relationship_results:
                    results['relationships'] = relationship_results
                    logger.info(f"Computed cardinality for {len(relationship_results)} relationships")
        
            logger.info(f"Data profiling completed: {len(results['tables'])} tables profiled")
            return results
            
        finally:
            conn.close()
    
    def _get_table_list(self, conn: sqlite3.Connection) -> List[str]:
        """Get list of user tables"""
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        return [row['name'] for row in cursor]
    
    def _get_database_stats(self, conn: sqlite3.Connection, tables: List[str]) -> Dict[str, Any]:
        """Get database-level statistics"""
        total_tables = len(tables)
        total_rows = 0
        
        # Count total rows across all tables
        for table_name in tables:
            try:
                count_result = conn.execute(f"SELECT COUNT(*) as count FROM `{table_name}`").fetchone()
                total_rows += count_result['count']
            except Exception as e:
                logger.warning(f"Failed to count rows in table {table_name}: {e}")
        
        return {
            'total_tables': total_tables,
            'total_rows': total_rows
        }
    
    def _profile_table(self, conn: sqlite3.Connection, table_name: str) -> Dict[str, Any]:
        """Profile table-level statistics"""
        stats = {'table_name': table_name}
        
        try:
            # Row count
            count_result = conn.execute(f"SELECT COUNT(*) as count FROM `{table_name}`").fetchone()
            stats['row_count'] = count_result['count']
            
            # Sample data
            if stats['row_count'] > 0:
                sample_result = conn.execute(
                    f"SELECT * FROM `{table_name}` LIMIT {self.sample_data_limit}"
                ).fetchall()
                
                # Convert to dict format
                if sample_result:
                    stats['sample_data'] = {
                        col: [row[col] for row in sample_result if row[col] is not None]
                        for col in sample_result[0].keys()
                    }
            
            logger.debug(f"Table {table_name}: {stats['row_count']} rows")
            
        except Exception as e:
            logger.error(f"Failed to profile table {table_name}: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def _profile_table_columns(self, conn: sqlite3.Connection, table_name: str, 
                              column_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Profile columns in a table
        
        Args:
            conn: SQLite database connection
            table_name: Name of the table to profile
            column_names: Optional list of specific column names to profile. 
                         If None, profiles all columns in the table.
        
        Returns:
            Dictionary mapping column names to their statistics
        """
        columns_stats = {}
        
        try:
            # Get column information
            column_info = conn.execute(f"PRAGMA table_info(`{table_name}`)").fetchall()
            
            if column_names is None:
                # Profile all columns
                columns_to_profile = [col_info['name'] for col_info in column_info]
            else:
                # Profile only specified columns
                existing_columns = {col['name'] for col in column_info}
                columns_to_profile = []
                
                for column_name in column_names:
                    if column_name not in existing_columns:
                        logger.warning(f"Column '{column_name}' not found in table '{table_name}', skipping")
                        continue
                    columns_to_profile.append(column_name)
            
            # Create a mapping from column name to column info for quick lookup
            column_info_map = {col['name']: col for col in column_info}

            for column_name in columns_to_profile:
                logger.debug(f"Profiling column: {table_name}.{column_name}")
                
                col_info = column_info_map[column_name]
                column_stats = self._profile_column(conn, table_name, column_name, col_info['type'])

                columns_stats[column_name] = column_stats
                
        except Exception as e:
            logger.error(f"Failed to profile columns in table {table_name}: {e}")
        
        return columns_stats
    
    def _profile_column(self, conn: sqlite3.Connection, table_name: str, column_name: str, data_type: str) -> Dict[str, Any]:
        """Profile individual column statistics"""
        stats = {
            'column_name': column_name,
            'data_type': data_type
        }
        
        try:
            # Use sampling for large tables
            sample_clause = ""
            total_rows = conn.execute(f"SELECT COUNT(*) FROM `{table_name}`").fetchone()[0]
            
            if total_rows > self.max_sample_size and self.max_sample_size > 0:
                sample_rate = max(1, total_rows // self.max_sample_size)
                sample_clause = f" WHERE rowid % {sample_rate} = 0"
                stats['sampled'] = True
                stats['sample_rate'] = sample_rate
            
            # 1. Get distinct count and a sample of distinct values in one go
            distinct_stats = self._get_distinct_stats_and_values(conn, table_name, column_name, sample_clause, limit=self.max_sample_size)
            stats['distinct_count'] = distinct_stats['distinct_count']
            
            # 2. Get the rest of the basic statistics (now without distinct count)
            stats.update(self._get_basic_column_stats(conn, table_name, column_name, sample_clause))
            
            # 3. Value distribution and top-K (uses the already calculated distinct_count)
            # Always generate top_k_values if there are any distinct values, even if distinct_count < min_distinct_for_top_k
            # This ensures that columns with only 1 distinct value (e.g., State='CA') still get top_k_values populated
            if stats.get('distinct_count', 0) > 0:
                stats.update(self._get_column_value_distribution(conn, table_name, column_name, sample_clause))
            else:
                logger.debug(f"Skipping top_k_values for {table_name}.{column_name}: distinct_count is 0")
            
            # 4. Infer semantic type using the collected stats
            stats['semantic_type'] = self._infer_semantic_type(stats, data_type)

            # 5. Shape features for string-like / textual columns
            #    - length statistics (min/max/avg of value length)
            #    - stable prefix / suffix (Fixity)
            if self._is_string_type(data_type) or stats.get('semantic_type') in ['Identifier', 'Free_Text', 'Categorical']:
                sample_values = distinct_stats.get('distinct_values') or []
                # 保守起见，仅在有一定数量样本时才计算形状特征
                if sample_values:
                    lengths = [len(v) for v in sample_values if v is not None]
                    if lengths:
                        stats['min_length'] = min(lengths)
                        stats['max_length'] = max(lengths)
                        stats['avg_length'] = sum(lengths) / len(lengths)

                    # 根据样本检测稳定前缀和后缀
                    fixed_prefix, fixed_suffix = self._detect_fixity(sample_values)
                    if fixed_prefix:
                        stats['fixed_prefix'] = fixed_prefix
                    if fixed_suffix:
                        stats['fixed_suffix'] = fixed_suffix

            # NOTE:
            # We intentionally do NOT persist the full distinct_values list into
            # semantic metadata anymore to avoid large storage overhead.
            # distinct_stats['distinct_values'] is only used transiently here.
            # 之前这里会基于这些值生成 `minhash_sketch`，但当前版本已移除该特征，
            # 以避免无用字段的生成和存储开销。
                
        except Exception as e:
            logger.error(f"Failed to profile column {table_name}.{column_name}: {e}")
            stats['error'] = str(e)
        
        return stats

    def _get_distinct_stats_and_values(
        self, 
        conn: sqlite3.Connection, 
        table_name: str, 
        column_name: str, 
        sample_clause: str, 
        limit: int = 10000
    ) -> Dict[str, Any]:
        """
        In a single query, get the total distinct count and a limited sample of distinct non-null values.
        """
        # Construct the base WHERE clause for non-null values
        # The sample_clause already contains 'WHERE' if it exists
        if sample_clause:
            where_clause = sample_clause + f" AND `{column_name}` IS NOT NULL"
        else:
            where_clause = f"WHERE `{column_name}` IS NOT NULL"

        # Use a CTE to first find all distinct values, then count them and select a sample
        query = f"""
            WITH DistinctValues AS (
                SELECT DISTINCT `{column_name}` as value
                FROM `{table_name}` {where_clause}
            )
            SELECT
                (SELECT COUNT(*) FROM DistinctValues) as total_distinct_count,
                value
            FROM DistinctValues
            LIMIT {limit}
        """
        
        cursor = conn.execute(query)
        
        total_distinct_count = 0
        distinct_values_sample = []
        
        first_row = True
        for row in cursor:
            if first_row:
                # The total count is the same for all rows, so we only read it once
                total_distinct_count = row['total_distinct_count']
                first_row = False
            
            if row['value'] is not None:
                distinct_values_sample.append(str(row['value']))
                
        return {
            'distinct_count': total_distinct_count,
            'distinct_values': distinct_values_sample
        }

    
    def _get_basic_column_stats(self, conn: sqlite3.Connection, table_name: str, column_name: str, sample_clause: str = '') -> Dict[str, Any]:
        """Get basic column statistics (excluding distinct count)."""
        stats = {}
        
        # Null count and total count
        # REMOVED: COUNT(DISTINCT `{column_name}`) as distinct_count
        result = conn.execute(f"""
            SELECT 
                COUNT(*) as total_count,
                COUNT(`{column_name}`) as non_null_count
            FROM `{table_name}` {sample_clause}
        """).fetchone()
        
        stats['total_count'] = result['total_count']
        stats['non_null_count'] = result['non_null_count']
        stats['null_count'] = result['total_count'] - result['non_null_count']
        
        # Min/Max values (only for non-null values)
        if stats['non_null_count'] > 0:
            try:
                # Construct proper WHERE clause
                if sample_clause:
                    where_clause = sample_clause + f" AND `{column_name}` IS NOT NULL"
                else:
                    where_clause = f"WHERE `{column_name}` IS NOT NULL"
                
                min_max_result = conn.execute(f"""
                    SELECT 
                        MIN(`{column_name}`) as min_value,
                        MAX(`{column_name}`) as max_value
                    FROM `{table_name}` {where_clause}
                """).fetchone()
                
                stats['min_value'] = min_max_result['min_value']
                stats['max_value'] = min_max_result['max_value']
                
            except Exception as e:
                logger.debug(f"Could not get min/max for {table_name}.{column_name}: {e}")
        
        return stats

    def _get_column_value_distribution(self, conn: sqlite3.Connection, table_name: str, column_name: str, sample_clause: str) -> Dict[str, Any]:
        """Get value distribution and top-K values"""
        stats = {}
        
        try:
            # Get top-K most frequent values
            # Construct proper WHERE clause
            where_clause = "WHERE `{column_name}` IS NOT NULL".format(column_name=column_name)
            if sample_clause:
                where_clause = sample_clause + " AND `{column_name}` IS NOT NULL".format(column_name=column_name)
            
            top_k_result = conn.execute(f"""
                SELECT `{column_name}` as value, COUNT(*) as frequency
                FROM `{table_name}` {where_clause}
                GROUP BY `{column_name}`
                ORDER BY COUNT(*) DESC, `{column_name}`
                LIMIT {self.top_k_limit}
            """).fetchall()
            
            if top_k_result:
                # Convert to dictionary format
                top_k_values = {}
                for row in top_k_result:
                    # Convert value to string for JSON serialization
                    value_str = str(row['value']) if row['value'] is not None else 'NULL'
                    top_k_values[value_str] = row['frequency']
                
                stats['top_k_values'] = top_k_values
            
        except Exception as e:
            logger.debug(f"Could not get value distribution for {table_name}.{column_name}: {e}")
        
        return stats
    
    def _is_string_type(self, data_type: str) -> bool:
        """Check if data type is string-based"""
        string_types = ['TEXT', 'VARCHAR', 'CHAR', 'STRING', 'CLOB']
        return any(stype in data_type.upper() for stype in string_types)

    # -------------------------------------------------------------------------
    # Shape / Fixity helpers
    # -------------------------------------------------------------------------

    def _detect_fixity(self, values: List[str], min_fix_length: int = 3) -> (Optional[str], Optional[str]):
        """
        Detect stable prefix/suffix from a list of sample string values.
        
        - 返回 (fixed_prefix, fixed_suffix)，任何一个检测不到则为 None。
        - 仅当公共前缀/后缀长度 >= min_fix_length 时才认为是“固定”。
        """
        # 预清洗：仅保留非空字符串
        cleaned = [str(v) for v in values if v is not None and str(v) != ""]
        if len(cleaned) < 2:
            return None, None

        prefix = self._longest_common_prefix(cleaned)
        suffix = self._longest_common_suffix(cleaned)

        fixed_prefix = prefix if prefix and len(prefix) >= min_fix_length else None
        fixed_suffix = suffix if suffix and len(suffix) >= min_fix_length else None

        return fixed_prefix, fixed_suffix

    def _longest_common_prefix(self, values: List[str]) -> str:
        """Compute longest common prefix of a list of strings."""
        if not values:
            return ""
        prefix = values[0]
        for v in values[1:]:
            # 缩短前缀直到匹配
            while not v.startswith(prefix) and prefix:
                prefix = prefix[:-1]
            if not prefix:
                break
        return prefix

    def _longest_common_suffix(self, values: List[str]) -> str:
        """Compute longest common suffix of a list of strings."""
        if not values:
            return ""
        # 反转字符串，转化为前缀问题
        reversed_vals = [v[::-1] for v in values]
        rev_suffix = self._longest_common_prefix(reversed_vals)
        return rev_suffix[::-1]
    
    def _infer_semantic_type(self, stats: Dict[str, Any], data_type: str) -> str:
        """
        Infers the semantic type of a column based on its statistics.
        """
        distinct_count = stats.get('distinct_count', 0)
        total_count = stats.get('total_count', 0)
        
        if total_count == 0 or distinct_count is None:
            return 'Unknown'
        
        # Datetime check
        dt_upper = data_type.upper()
        if 'DATE' in dt_upper or 'TIME' in dt_upper:
            return 'Datetime'

        cardinality_ratio = distinct_count / total_count if total_count > 0 else 0

        # Numeric types (INTEGER, REAL, etc.)
        if any(t in dt_upper for t in ['INT', 'REAL', 'FLOAT', 'DOUBLE', 'NUMERIC', 'DECIMAL']):
            if distinct_count <= 20:  # Low number of distinct integers could be categories (e.g., status codes)
                return 'Categorical'
            if cardinality_ratio > 0.95:
                return 'Identifier'
            return 'Numerical'

        # Text types (TEXT, VARCHAR, etc.)
        if any(t in dt_upper for t in ['CHAR', 'TEXT', 'CLOB']):
            if distinct_count <= 100 and cardinality_ratio < 0.6: # Low distinct count and ratio
                 return 'Categorical'
            if cardinality_ratio > 0.95:
                return 'Identifier'
            return 'Free_Text'
            
        return 'Unknown'
    
    def _compute_relationship_cardinality(
        self,
        database_path: str,
        source_table: str,
        source_columns: List[str],
        target_table: str,
        target_columns: List[str]
    ) -> Optional[str]:
        """
        Compute relationship cardinality by executing actual JOIN queries.
        
        Returns:
            "1:1", "1:N", "N:1", "N:M", or "Lookup" based on actual data statistics
        """
        conn = sqlite3.connect(f'file:{database_path}?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        
        # Build JOIN condition
        join_conditions = []
        for src_col, tgt_col in zip(source_columns, target_columns):
            join_conditions.append(f"s.`{src_col}` = t.`{tgt_col}`")
        join_condition = " AND ".join(join_conditions)
        
        # Query 1: Count distinct source values that have matches
        query1 = f"""
            SELECT COUNT(DISTINCT {', '.join([f"s.`{col}`" for col in source_columns])}) as distinct_source_matched
            FROM `{source_table}` s
            INNER JOIN `{target_table}` t ON {join_condition}
            WHERE {' AND '.join([f"s.`{col}` IS NOT NULL" for col in source_columns])}
        """
        
        # Query 2: Count distinct target values that have matches
        query2 = f"""
            SELECT COUNT(DISTINCT {', '.join([f"t.`{col}`" for col in target_columns])}) as distinct_target_matched
            FROM `{source_table}` s
            INNER JOIN `{target_table}` t ON {join_condition}
            WHERE {' AND '.join([f"t.`{col}` IS NOT NULL" for col in target_columns])}
        """
        
        # Query 3: Count total rows in source table
        query3 = f"SELECT COUNT(*) as total_source FROM `{source_table}`"
        
        # Query 4: Count total rows in target table
        query4 = f"SELECT COUNT(*) as total_target FROM `{target_table}`"
        
        # Query 5: Check if source values can map to multiple targets (many-to-many detection)
        query5 = f"""
            SELECT MAX(match_count) as max_targets_per_source
            FROM (
                SELECT {', '.join([f"s.`{col}`" for col in source_columns])}, COUNT(DISTINCT {', '.join([f"t.`{col}`" for col in target_columns])}) as match_count
                FROM `{source_table}` s
                INNER JOIN `{target_table}` t ON {join_condition}
                WHERE {' AND '.join([f"s.`{col}` IS NOT NULL" for col in source_columns])}
                GROUP BY {', '.join([f"s.`{col}`" for col in source_columns])}
            )
        """
        
        # Query 6: Check if target values can map to multiple sources
        query6 = f"""
            SELECT MAX(match_count) as max_sources_per_target
            FROM (
                SELECT {', '.join([f"t.`{col}`" for col in target_columns])}, COUNT(DISTINCT {', '.join([f"s.`{col}`" for col in source_columns])}) as match_count
                FROM `{source_table}` s
                INNER JOIN `{target_table}` t ON {join_condition}
                WHERE {' AND '.join([f"t.`{col}` IS NOT NULL" for col in target_columns])}
                GROUP BY {', '.join([f"t.`{col}`" for col in target_columns])}
            )
        """
        
        result1 = conn.execute(query1).fetchone()
        result2 = conn.execute(query2).fetchone()
        result3 = conn.execute(query3).fetchone()
        result4 = conn.execute(query4).fetchone()
        result5 = conn.execute(query5).fetchone()
        result6 = conn.execute(query6).fetchone()
        
        conn.close()
        
        distinct_source_matched = result1[0] if result1 else 0
        distinct_target_matched = result2[0] if result2 else 0
        total_source = result3[0] if result3 else 0
        total_target = result4[0] if result4 else 0
        max_targets_per_source = result5[0] if result5 and result5[0] is not None else 0
        max_sources_per_target = result6[0] if result6 and result6[0] is not None else 0
        
        # Determine cardinality
        # Lookup: Small target table (usually < 100 rows) with low cardinality
        if total_target < 100 and distinct_target_matched < 50:
            return "Lookup"
        
        # 1:1: Each source maps to at most 1 target, and each target maps to at most 1 source
        if max_targets_per_source <= 1 and max_sources_per_target <= 1:
            return "1:1"
        
        # N:M: Both sides can map to multiple
        if max_targets_per_source > 1 and max_sources_per_target > 1:
            return "N:M"
        
        # 1:N: Source maps to multiple targets, but target maps to at most 1 source
        if max_targets_per_source > 1 and max_sources_per_target <= 1:
            return "1:N"
        
        # N:1: Target maps to multiple sources, but source maps to at most 1 target
        if max_sources_per_target > 1 and max_targets_per_source <= 1:
            return "N:1"
        
        # Default fallback
        if max_targets_per_source > 1:
            return "1:N"
        elif max_sources_per_target > 1:
            return "N:1"
        else:
            return "1:1"