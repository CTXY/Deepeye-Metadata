# DDL Analyzer - Extract metadata from database DDL
# Highest priority source for metadata extraction

import logging
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

class DDLAnalyzer:
    """
    DDL Analyzer - Extract metadata from database schema
    
    This analyzer extracts metadata directly from database DDL using:
    - PRAGMA commands for SQLite
    - INFORMATION_SCHEMA queries for other databases
    - Direct DDL parsing
    
    Priority: Highest (ddl_extract)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.timeout = self.config.get('query_timeout', 30)
        logger.debug("DDLAnalyzer initialized")
    
    def analyze_database(self, database_path: str, needs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze database DDL and extract metadata based on needs
        """
        db_path = Path(database_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {database_path}")
        
        logger.info(f"Starting DDL analysis of database: {database_path}")
        
        conn = sqlite3.connect(f'file:{database_path}?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        
        try:
            results = {
                'database': {},
                'tables': {},
                'columns': {},
                'relationships': []
            }
            
            if needs is None:
                logger.debug("No specific needs provided, extracting all metadata")
                results['tables'] = self._extract_table_info(conn)
                results['columns'] = self._extract_column_info(conn)
                results['relationships'] = self._extract_relationships(conn, database_path)
            else:
                logger.debug(f"Selective analysis based on needs: {needs}")
                
                if needs.get('database', False):
                    results['database'] = self._extract_database_info(conn, db_path)
                
                tables_needed = needs.get('tables')
                if tables_needed is not None:
                    results['tables'] = self._extract_table_info(conn, tables_needed if isinstance(tables_needed, list) else None)

                columns_needed = needs.get('columns')
                if columns_needed is not None:
                    # Logic for extracting specific columns remains the same, as it's a different concern
                    if isinstance(columns_needed, dict) and len(columns_needed) > 0:
                        for table_name, column_names in columns_needed.items():
                            results['columns'][table_name] = {}
                            for column_name in column_names:
                                column_info = self._extract_single_column_info(conn, table_name, column_name)
                                if column_info:
                                    results['columns'][table_name][column_name] = column_info
                    else:
                        results['columns'] = self._extract_column_info(conn)

                if needs.get('relationships', False):
                    results['relationships'] = self._extract_relationships(conn, database_path)
            
            # === NEW: Topology Analysis ===
            # Compute FK in-degree and out-degree for each table
            if results.get('relationships') and results.get('tables'):
                topology_features = self._analyze_topology(
                    relationships=results['relationships'],
                    tables=list(results['tables'].keys())
                )
                
                # Merge topology features into table metadata
                for table_name, features in topology_features.items():
                    if table_name in results['tables']:
                        results['tables'][table_name]['fk_in_degree'] = features['in_degree']
                        results['tables'][table_name]['fk_out_degree'] = features['out_degree']
            
            logger.info(f"DDL analysis completed: {len(results['tables'])} tables, "
                       f"{sum(len(cols) for cols in results['columns'].values())} columns, "
                       f"{len(results['relationships'])} relationships")
            
            return results
            
        finally:
            conn.close()
    
    def _extract_database_info(self, conn: sqlite3.Connection, db_path: Path) -> Dict[str, Any]:
        """Extract database-level information"""
        return {
            'path': str(db_path),
            'name': db_path.stem,
            'size_bytes': db_path.stat().st_size,
            'sqlite_version': conn.execute("SELECT sqlite_version()").fetchone()[0]
        }
    

    def _extract_table_info(self, conn: sqlite3.Connection, table_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Extracts metadata for all tables or a specific list of tables by calling _extract_single_table_info.
        This removes code duplication and ensures consistent results.
        
        Args:
            conn: The database connection.
            table_names: An optional list of table names to extract. If None, all tables are extracted.
            
        Returns:
            A dictionary mapping table names to their extracted metadata.
        """
        tables = {}
        
        if table_names is None:
            # If no specific tables are requested, get all table names from the database
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            table_names_to_process = [row['name'] for row in cursor]
        else:
            table_names_to_process = table_names

        for table_name in table_names_to_process:
            logger.debug(f"Analyzing table: {table_name}")
            table_info = self._extract_single_table_info(conn, table_name)
            if table_info:
                tables[table_name] = table_info
        
        return tables

    def _extract_single_table_info(self, conn: sqlite3.Connection, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Extracts information for a single table. This is now the single source of truth for table metadata extraction.
        """
        try:
            # Check if table exists
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (table_name,))
            if not cursor.fetchone():
                logger.warning(f"Table '{table_name}' not found in database")
                return None
            
            # Get column information
            col_cursor = conn.execute(f"PRAGMA table_info('{table_name}')")
            columns = []
            primary_keys = [] # Always initialize as a list
            
            for col_row in col_cursor:
                col_info = dict(col_row)
                columns.append(col_info['name'])
                # The 'pk' column from PRAGMA is > 0 for primary key columns.
                if col_info['pk'] > 0:
                    primary_keys.append(col_info['name'])
            
            # Sort composite primary keys by their 'pk' index to ensure correct order
            # This is good practice although append order is usually correct.
            if len(primary_keys) > 1:
                pk_order_map = {info['name']: info['pk'] for info in map(dict, col_cursor) if info['pk'] > 0}
                primary_keys.sort(key=lambda name: pk_order_map.get(name, 0))

            table_info = {
                'table_name': table_name,
                'column_count': len(columns),
                'primary_keys': primary_keys, # This is guaranteed to be a list
                'columns': columns,
            }
            
            # You can still extract foreign keys and other info here if needed
            fk_cursor = conn.execute(f"PRAGMA foreign_key_list('{table_name}')")
            table_info['foreign_keys'] = [dict(fk_row) for fk_row in fk_cursor]
            
            idx_cursor = conn.execute(f"PRAGMA index_list('{table_name}')")
            indexes = []
            for idx_row in idx_cursor:
                idx_info = dict(idx_row)
                idx_col_cursor = conn.execute(f"PRAGMA index_info('{idx_info['name']}')")
                idx_info['columns'] = [dict(col)['name'] for col in idx_col_cursor]
                indexes.append(idx_info)
            table_info['indexes'] = indexes
            
            return table_info
            
        except Exception as e:
            logger.error(f"Failed to extract table info for {table_name}: {e}")
            return None

    def _extract_column_info(self, conn: sqlite3.Connection) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Extract detailed column information"""
        columns = {}
        
        # Get all table names first
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        
        for row in cursor:
            table_name = row['name']
            columns[table_name] = {}
            
            # Get detailed column information using PRAGMA
            col_cursor = conn.execute(f"PRAGMA table_info('{table_name}')")
            
            for col_row in col_cursor:
                col_info = dict(col_row)
                column_name = col_info['name']
                
                column_metadata = {
                    'name': column_name,
                    'data_type': col_info['type'],
                    'is_nullable': not col_info['notnull'],
                    'default_value': col_info['dflt_value'],
                    'is_primary_key': bool(col_info['pk']),
                }

                
                columns[table_name][column_name] = column_metadata
        
        return columns
    
    def _extract_relationships(
        self, 
        conn: sqlite3.Connection, 
        database_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract relationship information from foreign keys and return as RelationshipMetadata objects."""
        relationships = []
        
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        
        for row in cursor:
            table_name = row['name']
            
            fk_cursor = conn.execute(f"PRAGMA foreign_key_list('{table_name}')")
            
            fk_groups: Dict[int, Dict[str, Any]] = {}

            for fk_row in fk_cursor:
                fk_info = dict(fk_row)
                fk_id = fk_info['id']
                
                if fk_id not in fk_groups:
                    fk_groups[fk_id] = {
                        'source_table': table_name,
                        'target_table': fk_info['table'],
                        'columns': []
                    }
                
                fk_groups[fk_id]['columns'].append(
                    (fk_info['seq'], fk_info['from'], fk_info['to'])
                )

            for fk_id in fk_groups:
                group = fk_groups[fk_id]
                
                sorted_columns = sorted(group['columns'], key=lambda x: x[0])
                
                source_cols = [col[1] for col in sorted_columns]
                target_cols = [col[2] for col in sorted_columns]

                relationship = {
                    'relationship_type': 'foreign_key',
                    'source_table': group['source_table'],
                    'target_table': group['target_table'],
                    'source_columns': source_cols,
                    'target_columns': target_cols,
                }
                
                # Compute cardinality by executing actual JOIN query
                if database_path:
                    cardinality = self._compute_relationship_cardinality(
                        database_path, 
                        group['source_table'], 
                        source_cols,
                        group['target_table'], 
                        target_cols
                    )
                    if cardinality:
                        relationship['cardinality'] = cardinality
                
                relationships.append(relationship)
                
        return relationships
    
    def _extract_single_column_info(self, conn: sqlite3.Connection, table_name: str, column_name: str) -> Optional[Dict[str, Any]]:
        """Extract information for a single column"""
        try:
            # Get detailed column information using PRAGMA
            col_cursor = conn.execute(f"PRAGMA table_info('{table_name}')")
            
            for col_row in col_cursor:
                col_info = dict(col_row)
                if col_info['name'] == column_name:
                    return {
                        'name': column_name,
                        'data_type': col_info['type'],
                        'is_nullable': not col_info['notnull'],
                        'default_value': col_info['dflt_value'],
                        'is_primary_key': bool(col_info['pk']),
                    }
            
            logger.warning(f"Column '{column_name}' not found in table '{table_name}'")
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract column info for {table_name}.{column_name}: {e}")
            return None
    
    def extract_table_schema(self, database_path: str, table_name: str) -> Dict[str, Any]:
        """Extract schema for specific table"""
        conn = sqlite3.connect(f'file:{database_path}?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name = ? AND name NOT LIKE 'sqlite_%'
            """, (table_name,))
            
            if not cursor.fetchone():
                raise ValueError(f"Table '{table_name}' not found in database")
            
            table_info = self._extract_single_table_info(conn, table_name)
            if not table_info:
                raise ValueError(f"Failed to extract table info for '{table_name}'")
            
            column_info = {}
            col_cursor = conn.execute(f"PRAGMA table_info('{table_name}')")
            
            for col_row in col_cursor:
                col_info = dict(col_row)
                column_name = col_info['name']
                
                column_metadata = {
                    'name': column_name,
                    'data_type': col_info['type'],
                    'is_nullable': not col_info['notnull'],
                    'default_value': col_info['dflt_value'],
                    'is_primary_key': bool(col_info['pk']),
                }
                
                column_info[column_name] = column_metadata
            
            all_relationships = self._extract_relationships(conn, database_path)
            table_relationships = [
                rel for rel in all_relationships 
                if rel['source_table'] == table_name or rel['target_table'] == table_name
            ]
            
            return {
                'table_info': table_info,
                'columns': column_info,
                'relationships': table_relationships
            }
            
        finally:
            conn.close()
    
    def get_create_table_ddl(self, database_path: str, table_name: str) -> str:
        """Get the original CREATE TABLE DDL statement"""
        conn = sqlite3.connect(f'file:{database_path}?mode=ro', uri=True)
        
        try:
            cursor = conn.execute("""
                SELECT sql FROM sqlite_master 
                WHERE type='table' AND name = ? AND name NOT LIKE 'sqlite_%'
            """, (table_name,))
            
            row = cursor.fetchone()
            if row:
                return row[0]
            else:
                raise ValueError(f"Table '{table_name}' not found")
                
        finally:
            conn.close()
    
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
        try:
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
                
        except Exception as e:
            logger.warning(f"Failed to compute cardinality for {source_table} -> {target_table}: {e}")
            return None
    
    def _analyze_topology(
        self, 
        relationships: List[Dict[str, Any]], 
        tables: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """
        Analyze table relationship topology and compute FK in-degree and out-degree
        
        Args:
            relationships: List of relationship dicts from _extract_relationships()
            tables: List of table names to analyze
        
        Returns:
            Dict mapping table_name to {'in_degree': int, 'out_degree': int}
            
        Example:
            {
                'orders': {'in_degree': 2, 'out_degree': 3},  # Referenced by 2 tables, references 3 tables
                'customers': {'in_degree': 5, 'out_degree': 0}  # Referenced by 5 tables, references nothing
            }
        """
        # Initialize topology counters for all tables
        topology = {table: {'in_degree': 0, 'out_degree': 0} for table in tables}
        
        # Count FK relationships
        for rel in relationships:
            source = rel.get('source_table')
            target = rel.get('target_table')
            
            # source_table references target_table via FK
            # So: source has out_degree++, target has in_degree++
            if source in topology:
                topology[source]['out_degree'] += 1
            if target in topology:
                topology[target]['in_degree'] += 1
        
        logger.info(f"Topology analysis complete: {len(topology)} tables analyzed")
        
        # Log some interesting patterns
        high_in_degree = [t for t, f in topology.items() if f['in_degree'] >= 3]
        high_out_degree = [t for t, f in topology.items() if f['out_degree'] >= 3]
        
        if high_in_degree:
            logger.debug(f"High in-degree tables (≥3, potential Dimensions): {high_in_degree}")
        if high_out_degree:
            logger.debug(f"High out-degree tables (≥3, potential Facts): {high_out_degree}")
        
        return topology
    
    def validate_database_access(self, database_path: str) -> Dict[str, Any]:
        """Validate database access and return basic information"""
        try:
            db_path = Path(database_path)
            if not db_path.exists():
                return {'valid': False, 'error': f'Database file not found: {database_path}'}
            
            conn = sqlite3.connect(f'file:{database_path}?mode=ro', uri=True)
            
            try:
                version = conn.execute("SELECT sqlite_version()").fetchone()[0]
                
                table_count = conn.execute("""
                    SELECT COUNT(*) FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """).fetchone()[0]
                
                return {
                    'valid': True,
                    'sqlite_version': version,
                    'table_count': table_count,
                    'file_size_mb': db_path.stat().st_size / (1024 * 1024)
                }
                
            finally:
                conn.close()
                
        except Exception as e:
            return {'valid': False, 'error': str(e)}