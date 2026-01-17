# Data handler for BIRD dataset processing

import csv
import json
import os
import sqlite3
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel

from config import DataConfig

# Inline data models (previously in types.py)
class DatabaseSchema(BaseModel):
    """Database schema information"""
    db_id: str
    table_names: List[str]
    column_names: List[Dict[str, Any]]  # [{"table_id": int, "column_name": str}]
    column_types: List[str]
    primary_keys: List[Dict[str, Any]]   # [{"column_id": int}]
    foreign_keys: List[Dict[str, Any]]   # [{"column_id": int, "other_column_id": int}]
    db_path: str                         # Path to database file
    table_descriptions: Optional[Dict[str, Dict[str, str]]] = None  # Table descriptions from database_description files

class BirdDataItem(BaseModel):
    """BIRD dataset item"""
    question_id: int
    db_id: str
    question: str
    evidence: str
    sql: str
    difficulty: str                      # simple, moderate, challenging
    
    # Derived fields
    db_schema: Optional[DatabaseSchema] = None

logger = logging.getLogger(__name__)

class BirdDataHandler:
    """Handler for BIRD dataset operations"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.data_path = Path(config.bird_data_path)
        self.cache_dir = Path(config.cache_dir)
        
        # Create cache directory if needed
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Schema cache for databases
        self.schema_cache: Dict[str, DatabaseSchema] = {}
        
        logger.info(f"BirdDataHandler initialized with data path: {self.data_path}")
    
    def _preprocess_sql(self, sql: str, db_id: str) -> str:
        """
        Preprocess SQL to handle keyword conflicts with table names
        
        Args:
            sql: Original SQL query
            db_id: Database identifier
            
        Returns:
            Preprocessed SQL query
        """
        try:
            # Known problematic databases and their table names that might conflict with SQL keywords
            problematic_dbs = {
                'financial': ['account', 'card', 'client', 'disp', 'district', 'loan', 'order', 'trans'],
                'california_schools': ['schools', 'satscores', 'frpm'],
                'card_games': ['card_games'],
                'codebase_community': ['codebase_community'],
                'debit_card_specializing': ['debit_card_specializing'],
                'european_football_2': ['european_football_2'],
                'formula_1': ['formula_1'],
                'student_club': ['student_club'],
                'superhero': ['superhero'],
                'thrombosis_prediction': ['thrombosis_prediction'],
                'toxicology': ['toxicology']
            }
            
            # Get table names for this database
            table_names = problematic_dbs.get(db_id, [])
            
            # Handle SQL keyword conflicts - focus on table names that conflict with SQL keywords
            sql_keywords_that_conflict = ['order', 'group', 'select', 'from', 'where', 'having', 'limit', 'union', 'join']
            
            for table_name in table_names:
                if table_name.lower() in sql_keywords_that_conflict:
                    # Replace unquoted table names in various SQL contexts
                    # Use a simpler approach: replace table names that are not already quoted
                    
                    # Pattern to match unquoted table names in SQL contexts
                    # This matches table_name that:
                    # 1. Is not already quoted (no backticks, double quotes, or single quotes before)
                    # 2. Is preceded by SQL keywords that indicate table context
                    # 3. Is followed by word boundary (space, dot, comma, or other SQL keywords)
                    
                    patterns_to_replace = [
                        # FROM table_name (various cases: FROM order, FROM order AS, FROM order WHERE, etc.)
                        rf'\bFROM\s+(?![\"`\'])({re.escape(table_name)})\b',
                        # JOIN table_name  
                        rf'\bJOIN\s+(?![\"`\'])({re.escape(table_name)})\b',
                        # UPDATE table_name
                        rf'\bUPDATE\s+(?![\"`\'])({re.escape(table_name)})\b',
                        # INSERT INTO table_name
                        rf'\bINSERT\s+INTO\s+(?![\"`\'])({re.escape(table_name)})\b',
                        # table_name.column references (like order.account_id)
                        rf'\b(?![\"`\'])({re.escape(table_name)})(?=\.)',
                    ]
                    
                    # Special handling for ORDER BY vs order table
                    if table_name.lower() == 'order':
                        # First, protect ORDER BY keywords by temporarily replacing them
                        order_by_pattern = r'\bORDER\s+BY\b'
                        order_by_placeholder = '___ORDER_BY_PLACEHOLDER___'
                        sql = re.sub(order_by_pattern, order_by_placeholder, sql, flags=re.IGNORECASE)
                    
                    # Apply table name replacements
                    for pattern in patterns_to_replace:
                        sql = re.sub(pattern, lambda m: m.group(0).replace(m.group(1), f'`{m.group(1)}`'), sql, flags=re.IGNORECASE)
                    
                    # Restore ORDER BY keywords
                    if table_name.lower() == 'order':
                        sql = sql.replace(order_by_placeholder, 'ORDER BY')
                    
                    logger.debug(f"Processed table name '{table_name}' for SQL keyword conflicts")
            
            return sql
            
        except Exception as e:
            logger.warning(f"SQL preprocessing failed for {db_id}: {e}. Using original SQL.")
            return sql
    
    def load_dataset(self, split: str = "dev") -> List[BirdDataItem]:
        """
        Load BIRD dataset for specified split
        
        Args:
            split: Dataset split ("train" or "dev")
            
        Returns:
            List of BirdDataItem objects
        """
        if split not in ["train", "dev"]:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'dev'")
        
        json_file = self.data_path / split / f"{split}.json"
        
        if not json_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {json_file}")
        
        logger.info(f"Loading {split} dataset from {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = []
        for idx, item_data in enumerate(data):
            # Use index as question_id if not present in data
            question_id = item_data.get("question_id")
            if question_id is None:
                question_id = idx
            
            item = BirdDataItem(
                question_id=question_id,
                db_id=item_data["db_id"],
                question=item_data["question"],
                evidence=item_data.get("evidence", ""),
                sql=item_data.get("SQL", ""),
                difficulty=item_data.get("difficulty", "unknown")
            )
            items.append(item)
        
        logger.info(f"Loaded {len(items)} items from {split} dataset")
        return items
    
    def get_database_schema(self, db_id: str) -> DatabaseSchema:
        """
        Get schema for specified database
        
        Args:
            db_id: Database identifier
            
        Returns:
            DatabaseSchema object
        """
        if db_id in self.schema_cache:
            return self.schema_cache[db_id]
        
        # Determine database path based on split
        dev_db_path = self.data_path / "databases" / "dev_databases" / db_id / f"{db_id}.sqlite"
        train_db_path = self.data_path / "databases" / "train_databases" / db_id / f"{db_id}.sqlite"
        
        db_path = None
        if dev_db_path.exists():
            db_path = str(dev_db_path)
        elif train_db_path.exists():
            db_path = str(train_db_path)
        else:
            raise FileNotFoundError(f"Database not found for db_id: {db_id}")
        
        # Load database descriptions if enabled
        table_descriptions = self._load_database_descriptions(db_id)
        
        # Extract schema from database
        schema = self._extract_schema_from_db(db_id, db_path, table_descriptions)
        
        # Cache the schema
        self.schema_cache[db_id] = schema
        
        return schema
    
    def _extract_schema_from_db(self, db_id: str, db_path: str, table_descriptions: Optional[Dict[str, Dict[str, str]]] = None) -> DatabaseSchema:
        """
        Extract schema information from SQLite database
        
        Args:
            db_id: Database identifier
            db_path: Path to database file
            table_descriptions: Optional table and column descriptions
            
        Returns:
            DatabaseSchema object
        """
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        cursor = conn.cursor()
        
        try:
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence';")
            table_names = [row[0] for row in cursor.fetchall()]
            
            # Get column information
            column_names = []
            column_types = []
            primary_keys = []
            foreign_keys = []
            
            column_id = 0
            # Add a global column for "*"
            column_names.append({"table_id": -1, "column_name": "*"})
            column_types.append("text")
            column_id += 1
            
            for table_id, table_name in enumerate(table_names):
                # Get column info for this table
                # Use backticks to properly quote table names that might be SQL keywords
                cursor.execute(f"PRAGMA table_info(`{table_name}`)")
                table_columns = cursor.fetchall()
                
                for col_info in table_columns:
                    col_name = col_info[1]
                    col_type = col_info[2]
                    is_pk = col_info[5]
                    
                    column_names.append({"table_id": table_id, "column_name": col_name})
                    column_types.append(col_type.lower())
                    
                    if is_pk:
                        primary_keys.append({"column_id": column_id})
                    
                    column_id += 1
                
                # Get foreign key info for this table
                # Use backticks to properly quote table names that might be SQL keywords
                cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`)")
                fk_info = cursor.fetchall()
                
                for fk in fk_info:
                    from_col = fk[3]
                    to_table = fk[2]
                    to_col = fk[4]
                    
                    # Find column IDs
                    from_col_id = self._find_column_id(column_names, table_id, from_col)
                    to_table_id = table_names.index(to_table) if to_table in table_names else -1
                    to_col_id = self._find_column_id(column_names, to_table_id, to_col)
                    
                    if from_col_id is not None and to_col_id is not None:
                        foreign_keys.append({
                            "column_id": from_col_id,
                            "other_column_id": to_col_id
                        })
            
            schema = DatabaseSchema(
                db_id=db_id,
                table_names=table_names,
                column_names=column_names,
                column_types=column_types,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys,
                db_path=db_path,
                table_descriptions=table_descriptions
            )
            
            logger.debug(f"Extracted schema for {db_id}: {len(table_names)} tables, {len(column_names)} columns")
            return schema
            
        finally:
            conn.close()
    
    def _find_column_id(self, column_names: List[Dict], table_id: int, col_name: str) -> Optional[int]:
        """Find column ID by table ID and column name"""
        for i, col_info in enumerate(column_names):
            if col_info["table_id"] == table_id and col_info["column_name"] == col_name:
                return i
        return None
    
    def _load_database_descriptions(self, db_id: str) -> Optional[Dict[str, Dict[str, str]]]:
        """
        Load database descriptions from CSV files in database_description directory
        
        Args:
            db_id: Database identifier
            
        Returns:
            Dictionary mapping table names to column descriptions, or None if not found/enabled
        """
        if not self.config.use_database_description:
            return None
            
        # Try both dev and train databases
        dev_desc_path = self.data_path / "databases" / "dev_databases" / db_id / "database_description"
        train_desc_path = self.data_path / "databases" / "train_databases" / db_id / "database_description"
        
        desc_path = None
        if dev_desc_path.exists():
            desc_path = dev_desc_path
        elif train_desc_path.exists():
            desc_path = train_desc_path
        else:
            logger.debug(f"No database_description directory found for {db_id}")
            return None
        
        table_descriptions = {}
        
        try:
            # Read all CSV files in the database_description directory
            for csv_file in desc_path.glob("*.csv"):
                table_name = csv_file.stem  # Remove .csv extension
                
                column_descriptions = {}
                
                with open(csv_file, 'r', encoding='utf-8-sig') as f:  # utf-8-sig to handle BOM
                    reader = csv.DictReader(f)
                    
                    for row in reader:
                        # Always use original_column_name as it matches the actual database column names
                        col_name = row.get('original_column_name', '').strip()
                        
                        if not col_name:
                            continue
                        
                        # Build description parts
                        desc_parts = []
                        
                        # Add column description if available
                        col_desc = row.get('column_description', '').strip()
                        if col_desc and col_name != col_desc:
                            desc_parts.append(col_desc)
                            
                        
                        # # Add value description if available
                        # val_desc = row.get('value_description', '').strip()
                        # if val_desc:
                        #     desc_parts.append(f"Values: {val_desc}")
                        
                        # Combine descriptions
                        if desc_parts:
                            column_descriptions[col_name] = "; ".join(desc_parts)
                
                if column_descriptions:
                    table_descriptions[table_name] = column_descriptions
                    logger.debug(f"Loaded {len(column_descriptions)} column descriptions for table {table_name}")
            
            logger.info(f"Loaded database descriptions for {db_id}: {len(table_descriptions)} tables")
            return table_descriptions if table_descriptions else None
            
        except Exception as e:
            logger.warning(f"Failed to load database descriptions for {db_id}: {e}")
            return None
    
    def get_database_path(self, db_id: str) -> str:
        """
        Get path to database file
        
        Args:
            db_id: Database identifier
            
        Returns:
            Path to database file
        """
        schema = self.get_database_schema(db_id)
        return schema.db_path
    
    def execute_sql(self, db_id: str, sql: str) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[str]]:
        """
        Execute SQL query on specified database
        
        Args:
            db_id: Database identifier
            sql: SQL query to execute
            
        Returns:
            Tuple of (success, results, error_message)
        """
        try:
            # Preprocess SQL to handle keyword conflicts
            processed_sql = self._preprocess_sql(sql, db_id)
            
            # Add debug logging to see the actual SQL being executed
            if sql != processed_sql:
                logger.debug(f"SQL preprocessed for {db_id}:")
                logger.debug(f"  Original: {sql}")
                logger.debug(f"  Processed: {processed_sql}")
            
            db_path = self.get_database_path(db_id)
            conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()
            
            cursor.execute(processed_sql)
            results = cursor.fetchall()
            
            # Convert to list of dictionaries
            results_list = [dict(row) for row in results]
            
            conn.close()
            
            logger.debug(f"Executed SQL on {db_id}: {len(results_list)} rows returned")
            return True, results_list, None
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"SQL execution failed for {db_id}: {error_msg}")
            return False, None, error_msg
    
    def get_sample_data(self, db_id: str, table_name: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get sample data from a table
        
        Args:
            db_id: Database identifier
            table_name: Table name
            limit: Number of rows to return
            
        Returns:
            List of sample rows
        """
        try:
            # Handle tables with reserved names
            if table_name.lower() in ['order', 'by', 'group']:
                table_name = f"`{table_name}`"
            
            sql = f"SELECT * FROM {table_name} LIMIT {limit}"
            success, results, error = self.execute_sql(db_id, sql)
            
            if success:
                return results
            else:
                logger.warning(f"Failed to get sample data for {db_id}.{table_name}: {error}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting sample data for {db_id}.{table_name}: {e}")
            return []
