# Database Schema Extractor - Extract actual identifiers from database files
# Ensures metadata uses actual database names, table names, and column names

import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class DatabaseSchemaExtractor:
    """
    Extract actual schema information from database files
    
    Key features:
    - Extract database_id from database file using PRAGMA commands
    - Extract actual table and column names from DDL
    - Automatic name matching with normalization
    - Path to database_id mapping file management
    - Strict error handling for unmatched names
    """
    
    def __init__(self, mapping_file_path: str = "./memory/database_mapping.json"):
        self.mapping_file_path = Path(mapping_file_path)
        self.mapping_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_mapping()
    
    def _load_mapping(self) -> None:
        """Load path to database_id mapping file"""
        if self.mapping_file_path.exists():
            try:
                with open(self.mapping_file_path, 'r') as f:
                    self.path_to_id_mapping = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load mapping file: {e}")
                self.path_to_id_mapping = {}
        else:
            self.path_to_id_mapping = {}
    
    def _save_mapping(self) -> None:
        """Save mapping file"""
        try:
            with open(self.mapping_file_path, 'w') as f:
                json.dump(self.path_to_id_mapping, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save mapping file: {e}")
    
    def extract_database_id(self, database_path: str) -> str:
        """
        Extract database_id from database file
        
        Args:
            database_path: Path to database file
            
        Returns:
            Actual database_id extracted from the file
        """
        abs_path = os.path.abspath(database_path)
        
        # Check cache first
        if abs_path in self.path_to_id_mapping:
            return self.path_to_id_mapping[abs_path]
        
        # Extract from database file
        conn = sqlite3.connect(f'file:{database_path}?mode=ro', uri=True)
        try:
            # Try to get database name using PRAGMA database_list
            cursor = conn.execute("PRAGMA database_list")
            databases = cursor.fetchall()
            
            # For SQLite files, usually there's only one database
            if databases:
                db_name = databases[0][1]  # name field
                if db_name and db_name != 'main':
                    database_id = db_name
                else:
                    # If database name is empty or 'main', use filename
                    database_id = Path(database_path).stem
            else:
                database_id = Path(database_path).stem
            
            # Cache the result
            self.path_to_id_mapping[abs_path] = database_id
            self._save_mapping()
            
            return database_id
            
        finally:
            conn.close()
    
    def extract_actual_schema(self, database_path: str) -> Dict[str, Any]:
        """
        Extract actual schema from database DDL
        
        Args:
            database_path: Path to database file
            
        Returns:
            Dictionary containing actual table and column names
        """
        conn = sqlite3.connect(f'file:{database_path}?mode=ro', uri=True)
        
        try:
            # Extract table names
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            actual_tables = [row[0] for row in cursor]
            
            # Extract column names for each table
            actual_columns = {}
            for table_name in actual_tables:
                col_cursor = conn.execute(f"PRAGMA table_info('{table_name}')")
                actual_columns[table_name] = [row[1] for row in col_cursor]
            
            return {
                'tables': actual_tables,
                'columns': actual_columns
            }
        finally:
            conn.close()
    
    def normalize_name(self, name: str) -> str:
        """
        Normalize name for matching
        
        Args:
            name: Name to normalize
            
        Returns:
            Normalized name
        """
        if not name:
            return ""
        
        # Convert to lowercase and remove special characters
        normalized = re.sub(r'[^a-z0-9_]', '', name.lower())
        
        # Remove leading/trailing underscores
        normalized = normalized.strip('_')
        
        return normalized
    
    def find_best_match(self, target_name: str, candidates: List[str], threshold: float = 0.8) -> str:
        """
        Find best match in candidate names, raise exception if no match found
        
        Args:
            target_name: Name to match
            candidates: List of candidate names
            threshold: Minimum similarity threshold
            
        Returns:
            Best matching candidate name
            
        Raises:
            ValueError: If no suitable match is found
        """
        if not target_name or not candidates:
            raise ValueError(f"No candidates available for matching '{target_name}'")
        
        normalized_target = self.normalize_name(target_name)
        best_match = None
        best_score = 0
        
        for candidate in candidates:
            normalized_candidate = self.normalize_name(candidate)
            
            # Exact match
            if normalized_target == normalized_candidate:
                return candidate
            
            # Similarity match
            score = SequenceMatcher(None, normalized_target, normalized_candidate).ratio()
            if score > best_score:
                best_score = score
                best_match = candidate
        
        if best_score >= threshold:
            return best_match
        else:
            raise ValueError(f"No suitable match found for '{target_name}' in candidates {candidates} (best score: {best_score:.2f})")
    
    def validate_schema_consistency(self, database_path: str, expected_schema: Dict[str, Any]) -> None:
        """
        Validate schema consistency and raise exceptions for mismatches
        
        Args:
            database_path: Path to database file
            expected_schema: Expected schema structure
            
        Raises:
            ValueError: If schema inconsistencies are found
        """
        actual_schema = self.extract_actual_schema(database_path)
        errors = []
        
        # Check table name consistency
        for expected_table in expected_schema.get('tables', []):
            if expected_table not in actual_schema['tables']:
                errors.append(f"Table '{expected_table}' not found in actual database")
        
        # Check column name consistency
        for table_name, expected_columns in expected_schema.get('columns', {}).items():
            if table_name in actual_schema['columns']:
                actual_columns = actual_schema['columns'][table_name]
                for expected_col in expected_columns:
                    if expected_col not in actual_columns:
                        errors.append(f"Column '{table_name}.{expected_col}' not found in actual database")
        
        if errors:
            raise ValueError(f"Schema validation failed: {'; '.join(errors)}")
    
    def get_database_info(self, database_path: str) -> Dict[str, Any]:
        """
        Get comprehensive database information
        
        Args:
            database_path: Path to database file
            
        Returns:
            Dictionary containing database information
        """
        database_id = self.extract_database_id(database_path)
        schema = self.extract_actual_schema(database_path)
        
        return {
            'database_id': database_id,
            'database_path': os.path.abspath(database_path),
            'schema': schema,
            'table_count': len(schema['tables']),
            'total_column_count': sum(len(cols) for cols in schema['columns'].values())
        }
