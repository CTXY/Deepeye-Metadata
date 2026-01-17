#!/usr/bin/env python3
"""
Import Historical Data Script

This script imports historical SQL generation data from NL2SQL-Bugs-Benchmark 
into the CAF memory system to test memory evolution functionality.

python script/caf/import_historical_data.py /home/yangchenyu/Text2SQL/NL2SQL-Bugs-Benchmark/NL2SQL-Bugs-with-evidence.json --data-source nl2sql_bugs

python script/caf/import_historical_data.py /home/yangchenyu/DeepEye-SQL-Metadata/dataset/bird/dev/dev.json --data-source bird_ground_truth

python script/caf/import_historical_data.py /home/yangchenyu/DeepEye-SQL-Metadata/dataset/bird/train/train.json --data-source bird_ground_truth

python script/caf/import_historical_data.py /home/yangchenyu/DeepEye-SQL-Metadata/output/bird_insights_qwen3-coder-30b-a3b_incorrect.jsonl --data-source insights
"""


import json
import sys
import logging
import csv
import sqlite3
import re
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel

# Add the project root directory to path so we can import caf
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import caf
from caf.memory.types import EpisodicRecord, SQLExecutionResult, MemoryType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SUPPORTED_DATA_SOURCES = {
    'nl2sql_bugs': 'NL2SQL-Bugs-Benchmark',
    'bird_ground_truth': 'BIRD-Dataset',
    'gpt4o_dev_results': 'GPT-4o',
    'gpt4o_mini_train_results': 'GPT-4o-mini',
    'insights': 'Insights-JSONL'
}


# =============================================================================
# Integrated Data Handler (from data_handler.py)
# =============================================================================

class DatabaseSchema(BaseModel):
    """Database schema information"""
    db_id: str
    table_names: List[str]
    column_names: List[Dict[str, Any]]
    column_types: List[str]
    primary_keys: List[Dict[str, Any]]
    foreign_keys: List[Dict[str, Any]]
    db_path: str
    table_descriptions: Optional[Dict[str, Dict[str, str]]] = None


class BirdDataHandler:
    """Handler for BIRD dataset operations"""
    
    def __init__(self, bird_data_path: str, cache_dir: str = None, use_database_description: bool = False):
        self.bird_data_path = Path(bird_data_path)
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent.parent.parent / "cache"
        self.use_database_description = use_database_description
        
        # Create cache directory if needed
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Schema cache for databases
        self.schema_cache: Dict[str, DatabaseSchema] = {}
        
        logger.info(f"BirdDataHandler initialized with data path: {self.bird_data_path}")
    
    def _preprocess_sql(self, sql: str, db_id: str) -> str:
        """Preprocess SQL to handle keyword conflicts with table names"""
        try:
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
            
            table_names = problematic_dbs.get(db_id, [])
            sql_keywords_that_conflict = ['order', 'group', 'select', 'from', 'where', 'having', 'limit', 'union', 'join']
            
            for table_name in table_names:
                if table_name.lower() in sql_keywords_that_conflict:
                    patterns_to_replace = [
                        rf'\bFROM\s+(?![\"`\'])({re.escape(table_name)})\b',
                        rf'\bJOIN\s+(?![\"`\'])({re.escape(table_name)})\b',
                        rf'\bUPDATE\s+(?![\"`\'])({re.escape(table_name)})\b',
                        rf'\bINSERT\s+INTO\s+(?![\"`\'])({re.escape(table_name)})\b',
                        rf'\b(?![\"`\'])({re.escape(table_name)})(?=\.)',
                    ]
                    
                    if table_name.lower() == 'order':
                        order_by_pattern = r'\bORDER\s+BY\b'
                        order_by_placeholder = '___ORDER_BY_PLACEHOLDER___'
                        sql = re.sub(order_by_pattern, order_by_placeholder, sql, flags=re.IGNORECASE)
                    
                    for pattern in patterns_to_replace:
                        sql = re.sub(pattern, lambda m: m.group(0).replace(m.group(1), f'`{m.group(1)}`'), sql, flags=re.IGNORECASE)
                    
                    if table_name.lower() == 'order':
                        sql = sql.replace(order_by_placeholder, 'ORDER BY')
            
            return sql
            
        except Exception as e:
            logger.warning(f"SQL preprocessing failed for {db_id}: {e}. Using original SQL.")
            return sql
    
    def get_database_path(self, db_id: str) -> str:
        """Get path to database file"""
        dev_db_path = self.bird_data_path / "databases" / "dev_databases" / db_id / f"{db_id}.sqlite"
        train_db_path = self.bird_data_path / "databases" / "train_databases" / db_id / f"{db_id}.sqlite"
        
        if dev_db_path.exists():
            return str(dev_db_path)
        elif train_db_path.exists():
            return str(train_db_path)
        else:
            raise FileNotFoundError(f"Database not found for db_id: {db_id}")
    
    def execute_sql(self, db_id: str, sql: str) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[str]]:
        """Execute SQL query on specified database"""
        try:
            processed_sql = self._preprocess_sql(sql, db_id)
            
            if sql != processed_sql:
                logger.debug(f"SQL preprocessed for {db_id}")
            
            db_path = self.get_database_path(db_id)
            conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(processed_sql)
            results = cursor.fetchall()
            
            results_list = [dict(row) for row in results]
            conn.close()
            
            logger.debug(f"Executed SQL on {db_id}: {len(results_list)} rows returned")
            return True, results_list, None
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"SQL execution failed for {db_id}: {error_msg}")
            return False, None, error_msg


# =============================================================================
# Historical Data Importer
# =============================================================================

class HistoricalDataImporter:
    """Import historical NL2SQL data into CAF memory system"""
    
    def __init__(self, config_path: str = None, bird_data_path: str = None, cache_dir: str = None):
        """Initialize the importer with CAF system"""
        try:
            # Initialize CAF system
            if config_path:
                self.caf_system = caf.initialize(config_path=config_path)
            else:
                default_config_path = Path(__file__).parent.parent.parent / "config" / "caf_config.yaml"
                if default_config_path.exists():
                    self.caf_system = caf.initialize(config_path=str(default_config_path))
                else:
                    logger.warning("No config file found, using default configuration")
                    self.caf_system = caf.initialize()
            
            logger.info("CAF system initialized successfully")
            
            # Initialize data handler for SQL execution
            if bird_data_path is None:
                bird_data_path = str(Path(__file__).parent.parent.parent / "dataset" / "bird")
            if cache_dir is None:
                cache_dir = str(Path(__file__).parent.parent.parent / "cache")
            
            self.data_handler = BirdDataHandler(bird_data_path, cache_dir)
            logger.info("Data handler initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CAF system: {e}")
            raise
    
    def _load_json_file(self, file_path: str, fix_unquoted_tokens: bool = False) -> Any:
        """Load JSON file, optionally fixing unquoted success/failure tokens."""
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
            if fix_unquoted_tokens:
                raw_content = raw_content.replace(': success', ': true').replace(': failure', ': false')
            return json.loads(raw_content)
    
    def _load_nl2sql_bugs_records(self, file_path: str) -> List[Dict[str, Any]]:
        """Load default NL2SQL Bugs Benchmark data."""
        data = self._load_json_file(file_path)
        logger.info(f"Loaded {len(data)} NL2SQL-Bugs records from {file_path}")
        return data
    
    def _load_bird_ground_truth_records(self, file_path: str) -> List[Dict[str, Any]]:
        """Load BIRD dev/train ground truth examples as episodic records."""
        raw_records = self._load_json_file(file_path)
        split_name = Path(file_path).parent.name
        normalized_records: List[Dict[str, Any]] = []
        
        for record in raw_records:
            sql_text = record.get('SQL') or record.get('sql')
            if not record.get('db_id') or not record.get('question') or not sql_text:
                logger.debug(f"Skipping malformed BIRD record: {record.get('question_id')}")
                continue
            
            normalized_records.append({
                'id': f"bird_{split_name}_{record.get('question_id')}",
                'db_id': record['db_id'],
                'question': record['question'],
                'sql': sql_text,
                'evidence': record.get('evidence'),
                'label': True,
                'source_model': SUPPORTED_DATA_SOURCES['bird_ground_truth'],
                'metadata': {
                    'dataset_split': split_name,
                    'difficulty': record.get('difficulty'),
                    'is_ground_truth': True,
                    'source_file': str(file_path)
                }
            })
        
        logger.info(f"Loaded {len(normalized_records)} BIRD ground truth records from {file_path}")
        return normalized_records
    
    def _load_gpt_results_records(self, file_path: str, records_key: str, source_model_key: str) -> List[Dict[str, Any]]:
        """Load GPT model execution results from DAMO evaluation outputs."""
        data = self._load_json_file(file_path, fix_unquoted_tokens=True)
        records = data.get(records_key)
        if records is None:
            raise ValueError(f"Field '{records_key}' not found in {file_path}")
        
        prefix = Path(file_path).stem
        split_name = data.get('experiment_info', {}).get('data_split')
        normalized_records: List[Dict[str, Any]] = []
        
        for entry in records:
            sql_text = entry.get('generated_sql')
            db_id = entry.get('db_id')
            question = entry.get('question')
            
            if not db_id or not question or not sql_text:
                logger.debug(f"Skipping malformed GPT record: {entry.get('question_id')}")
                continue
            
            metadata = {
                'ground_truth_sql': entry.get('ground_truth_sql'),
                'ex_correct': entry.get('ex_correct'),
                'generation_time_ms': entry.get('generation_time_ms'),
                'caf_used': entry.get('caf_used'),
                'memory_used': entry.get('memory_used'),
                'provided_execution_result': entry.get('execution_result'),
                'provided_error_message': entry.get('error_message'),
                'data_split': split_name,
                'source_file': str(file_path)
            }
            
            # For GPT results (gpt4o_dev_results, gpt4o_mini_train_results), use ex_correct to determine label
            # ex_correct=True means SQL is correct (should be skipped for non-ground-truth sources)
            # ex_correct=False/None means SQL is incorrect (should be imported)
            ex_correct = entry.get('ex_correct')
            # Use ex_correct if available, otherwise fall back to execution_success
            label = ex_correct if ex_correct is not None else entry.get('execution_success')
            
            normalized_records.append({
                'id': f"{prefix}_{entry.get('question_id')}",
                'db_id': db_id,
                'question': question,
                'sql': sql_text,
                'evidence': entry.get('evidence'),
                'label': label,
                'source_model': source_model_key,
                'metadata': metadata
            })
        
        logger.info(f"Loaded {len(normalized_records)} GPT result records from {file_path}")
        return normalized_records
    
    def _load_insights_records(self, file_path: str) -> List[Dict[str, Any]]:
        """Load line-delimited insights data where error_template is stored as SQL."""
        normalized_records: List[Dict[str, Any]] = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON line {idx} in {file_path}: {e}")
                    continue
                
                question = entry.get('question')
                db_id = entry.get('db_id')
                error_template = entry.get('error_template')
                
                if not (question and db_id and error_template):
                    logger.debug(f"Skipping incomplete insights record on line {idx}: missing required fields")
                    continue
                
                record_id = entry.get('question_id') or entry.get('id') or f"{Path(file_path).stem}_{idx}"
                
                metadata = {
                    'comparative_insight': entry.get('comparative_insight'),
                    'original_sqls': entry.get('original_sqls'),
                    'question_id': entry.get('question_id'),
                    'source_file': str(file_path)
                }
                
                normalized_records.append({
                    'id': f"insights_{record_id}",
                    'db_id': db_id,
                    'question': question,
                    'sql': error_template,
                    'label': False,  # error templates represent incorrect SQL variants
                    'source_model': SUPPORTED_DATA_SOURCES['insights'],
                    'metadata': metadata
                })
        
        logger.info(f"Loaded {len(normalized_records)} insights records from {file_path}")
        return normalized_records
    
    def load_historical_data(self, file_path: str, data_source: str = 'nl2sql_bugs') -> List[Dict[str, Any]]:
        """Load historical data from supported sources."""
        loaders = {
            'nl2sql_bugs': self._load_nl2sql_bugs_records,
            'bird_ground_truth': self._load_bird_ground_truth_records,
            'gpt4o_dev_results': lambda path: self._load_gpt_results_records(
                path,
                records_key='detailed_results',
                source_model_key=SUPPORTED_DATA_SOURCES['gpt4o_dev_results']
            ),
            'gpt4o_mini_train_results': lambda path: self._load_gpt_results_records(
                path,
                records_key='results',
                source_model_key=SUPPORTED_DATA_SOURCES['gpt4o_mini_train_results']
            ),
            'insights': self._load_insights_records
        }
        
        if data_source not in loaders:
            raise ValueError(f"Unsupported data source: {data_source}")
        
        try:
            data = loaders[data_source](file_path)
            logger.info(f"Loaded {len(data)} records from {file_path} using source '{data_source}'")
            return data
        except Exception as e:
            logger.error(f"Failed to load historical data from {file_path}: {e}")
            raise
    
    def convert_to_episodic_record(self, record: Dict[str, Any], base_timestamp: datetime = None) -> EpisodicRecord:
        """
        Convert a historical record to EpisodicRecord format (flattened structure)
        
        Args:
            record: Historical record from the dataset
            base_timestamp: Base timestamp to use (with some variation)
            
        Returns:
            EpisodicRecord object
        """
        try:
            # Use base timestamp with some variation to simulate different times
            if base_timestamp is None:
                base_timestamp = datetime.utcnow() - timedelta(days=30)
            
            # Add some variation (up to 1 hour) to make timestamps more realistic
            variation_minutes = random.randint(-60, 60)
            timestamp = base_timestamp + timedelta(minutes=variation_minutes)
            
            # Get SQL text (support multiple keys)
            sql_text = record.get('sql') or record.get('SQL') or record.get('generated_sql')
            if not sql_text:
                raise ValueError(f"No SQL text found for record {record.get('id')}")
            
            # Get question text
            question_text = record.get('question') or record.get('user_query')
            if not question_text:
                raise ValueError(f"No question found for record {record.get('id')}")
            
            # Note: session_id will be generated by the episodic store during batch storage
            
            # Execute SQL to get real execution results
            # execution_result = self._execute_sql_and_get_result(record['db_id'], sql_text)
            
            execution_result = None
            
            # Extract label from record (True if execution succeeded, False otherwise)
            label = record.get('label')
            if label is None:
                print(f"---------------Label is None for record {record.get('id')}---------------")
                # Infer label from execution result
                label = execution_result.execution_success if execution_result else False
            
            # Extract error types
            error_types = record.get('error_types')
            if error_types and isinstance(error_types, list):
                # Ensure error_types is in the correct format
                error_types = [
                    {"type": et.get('error_type', str(et)) if isinstance(et, dict) else str(et)}
                    for et in error_types
                ]
            
            source_model = record.get('source_model', SUPPORTED_DATA_SOURCES['nl2sql_bugs'])
            
            metadata = {}
            record_metadata = record.get('metadata')
            if isinstance(record_metadata, dict):
                metadata.update(record_metadata)
            
            metadata.setdefault('source', source_model)
            if record.get('id') is not None:
                metadata.setdefault('record_id', record.get('id'))
            
            # Create episodic record (flattened structure)
            # session_id is None and will be generated by store during batch storage
            episodic_record = EpisodicRecord(
                session_id=None,  # Will be generated by store
                database_id=record['db_id'],
                user_query=question_text,
                context=record.get('evidence'),
                round_id=1,
                generated_sql=sql_text,
                execution_result=execution_result,
                label=label,
                error_types=error_types,
                feedback_text=record.get('feedback'),
                source_model=source_model,
                timestamp=timestamp.isoformat(),
                metadata=metadata
            )
            
            return episodic_record
            
        except Exception as e:
            logger.error(f"Failed to convert record {record.get('id', 'unknown')} to episodic record: {e}")
            raise
    
# Removed _generate_session_id method - session_id is now generated by episodic store
    
    def _execute_sql_and_get_result(self, db_id: str, sql_text: str) -> SQLExecutionResult:
        """
        Execute SQL query and return SQLExecutionResult
        
        Args:
            db_id: Database identifier
            sql_text: SQL query to execute
            
        Returns:
            SQLExecutionResult object
        """
        start_time = time.time()
        
        try:
            # Execute SQL using data handler
            success, results, error_message = self.data_handler.execute_sql(db_id, sql_text)
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            if success:
                return SQLExecutionResult(
                    execution_success=True,
                    results=results,
                    error_message=None,
                    row_count=len(results) if results else 0,
                    execution_time_ms=execution_time_ms
                )
            else:
                return SQLExecutionResult(
                    execution_success=False,
                    results=None,
                    error_message=error_message,
                    row_count=0,
                    execution_time_ms=execution_time_ms
                )
                
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            error_msg = f"SQL execution failed: {str(e)}"
            logger.error(f"Failed to execute SQL for {db_id}: {error_msg}")
            
            return SQLExecutionResult(
                execution_success=False,
                results=None,
                error_message=error_msg,
                row_count=0,
                execution_time_ms=execution_time_ms
            )
    
    def import_records(self, records: List[Dict[str, Any]], 
                      database_id: str = None, 
                      limit: int = None,
                      test_evolution: bool = True,
                      data_source: str = 'nl2sql_bugs') -> Dict[str, Any]:
        """
        Import historical records into memory base
        
        Args:
            records: List of historical records
            database_id: Optional filter by database ID
            limit: Optional limit on number of records to import
            test_evolution: Whether to test evolution after import
            data_source: Data source type
                - 'bird_ground_truth': Import all records
                - Other sources: Skip records with label=True, import others
            
        Returns:
            Import statistics
        """
        stats = {
            'total_records': len(records),
            'imported_records': 0,
            'successful_records': 0,
            'failed_records': 0,
            'skipped_records': 0,
            'databases_processed': set(),
            'errors': []
        }
        
        try:
            # Filter records if database_id is specified
            if database_id:
                records = [r for r in records if r.get('db_id') == database_id]
                logger.info(f"Filtered to {len(records)} records for database: {database_id}")
            
            # Apply limit if specified
            if limit:
                records = records[:limit]
                logger.info(f"Limited to {limit} records")
            
            # Group records by database for efficient processing
            db_groups = {}
            for record in records:
                db_id = record['db_id']
                if db_id not in db_groups:
                    db_groups[db_id] = []
                db_groups[db_id].append(record)
            
            logger.info(f"Processing {len(records)} records across {len(db_groups)} databases")
            
            # Log filtering behavior
            if data_source == 'bird_ground_truth':
                logger.info(f"Importing mode: All records (data_source={data_source} is ground truth)")
            else:
                logger.info(f"Filtering mode: Skip records with label=True, import others (data_source={data_source})")
            
            # Get episodic memory store
            episodic_store = self.caf_system.get_memory_store(MemoryType.EPISODIC)
            if not episodic_store:
                raise ValueError("Episodic memory store not available")
            
            # Process each database group
            base_timestamp = datetime.utcnow() - timedelta(days=30)
            
            for db_id, db_records in db_groups.items():
                try:
                    logger.info(f"Processing {len(db_records)} records for database: {db_id}")
                    
                    # Bind database to episodic store
                    episodic_store.bind_database(db_id)
                    stats['databases_processed'].add(db_id)
                    
                    # Process records in batch for this database
                    valid_episodic_records = []
                    
                    for i, record in enumerate(db_records):
                        try:
                            # Simple filtering logic:
                            # - bird_ground_truth: import all records
                            # - other data sources: skip if label == True, import otherwise
                            if data_source != 'bird_ground_truth':
                                record_label = record.get('label')
                                if record_label is True:
                                    stats['skipped_records'] += 1
                                    continue
                            
                            # Convert to episodic record
                            episodic_record = self.convert_to_episodic_record(
                                record, 
                                base_timestamp + timedelta(minutes=i * 5)  # Space out timestamps
                            )
                            
                            # Add to batch for later storage
                            valid_episodic_records.append(episodic_record)
                            
                        except Exception as e:
                            error_msg = f"Failed to process record {record.get('id', 'unknown')}: {e}"
                            logger.error(error_msg)
                            stats['errors'].append(error_msg)
                            stats['skipped_records'] += 1
                    
                    # Batch store all valid records for this database
                    if valid_episodic_records:
                        logger.info(f"Batch storing {len(valid_episodic_records)} records for database: {db_id}")
                        try:
                            batch_stats = episodic_store.store_records_batch(valid_episodic_records)
                            
                            # Update stats from batch operation
                            stats['imported_records'] += batch_stats['stored']
                            stats['skipped_records'] += (batch_stats['skipped_duplicate_session_id'] + 
                                                        batch_stats['skipped_duplicate_content'])
                            
                            # Count successful/failed based on label (not execution result)
                            for record in valid_episodic_records[:batch_stats['stored']]:
                                if record.label is True:
                                    stats['successful_records'] += 1
                                else:
                                    stats['failed_records'] += 1
                            
                            logger.info(f"Batch storage completed: {batch_stats['stored']} stored, "
                                       f"{batch_stats['skipped_duplicate_session_id'] + batch_stats['skipped_duplicate_content']} duplicates skipped, "
                                       f"{batch_stats['errors']} errors")
                            
                            if batch_stats['error_details']:
                                stats['errors'].extend(batch_stats['error_details'])
                                
                        except Exception as e:
                            error_msg = f"Failed to batch store records for {db_id}: {e}"
                            logger.error(error_msg)
                            stats['errors'].append(error_msg)
                            stats['skipped_records'] += len(valid_episodic_records)
                    else:
                        logger.info(f"No valid records to store for database: {db_id}")
                    
                    logger.info(f"Completed importing {len(db_records)} records for database: {db_id}")
                    
                    # Rebuild search indexes for this database after importing all records
                    # This will batch compute and persist embeddings, improving search performance
                    try:
                        logger.info(f"Rebuilding search indexes for database: {db_id}...")
                        episodic_store.rebuild_search_indexes()
                        logger.info(f"Search indexes rebuilt successfully for database: {db_id}")
                    except Exception as e:
                        error_msg = f"Failed to rebuild search indexes for {db_id}: {e}"
                        logger.warning(error_msg)
                        stats['errors'].append(error_msg)
                    
                except Exception as e:
                    error_msg = f"Failed to process database {db_id}: {e}"
                    logger.error(error_msg)
                    stats['errors'].append(error_msg)
            
        except Exception as e:
            error_msg = f"Import process failed: {e}"
            logger.error(error_msg)
            stats['errors'].append(error_msg)
        
        finally:
            # Cleanup CAF system
            self.caf_system.cleanup()
        
        return stats
    
    def _compute_db_question_splits(self, dev_file: str, ratio: float) -> Dict[str, Dict[str, Any]]:
        """
        Build per-database split based on the BIRD dev set order.
        For each database, select the first ratio portion (by order in the file)
        of questions for injection; the rest are held out for evaluation.
        """
        if not dev_file:
            raise ValueError("dev_file must be provided when using ratio-based injection")

        if ratio < 0 or ratio > 1:
            raise ValueError("ratio must be within [0, 1]")

        with open(dev_file, 'r', encoding='utf-8') as f:
            dev_data = json.load(f)

        db_to_questions: Dict[str, List[str]] = {}
        for item in dev_data:
            db_id = item.get('db_id')
            q_text = item.get('question')
            if db_id is None or not q_text:
                continue
            db_to_questions.setdefault(db_id, []).append(q_text)

        splits: Dict[str, Dict[str, Any]] = {}
        for db_id, questions in db_to_questions.items():
            total = len(questions)
            cutoff = int(total * ratio)
            inject_questions = set(questions[:cutoff])
            eval_questions = set(questions[cutoff:])
            splits[db_id] = {
                'inject_questions': inject_questions,
                'eval_questions': eval_questions,
                'counts': {
                    'total': total,
                    'inject': len(inject_questions),
                    'eval': len(eval_questions)
                }
            }

        return splits

    def run_import(self, data_file: str, 
                   database_id: str = None, 
                   limit: int = None,
                   config_path: str = None,
                   test_evolution: bool = True,
                   ratio: float = None,
                   dev_file: str = None,
                   data_source: str = 'nl2sql_bugs') -> None:
        """
        Main import function
        
        Args:
            data_file: Path to historical data JSON file
            database_id: Optional database ID filter
            limit: Optional limit on records to import
            config_path: Optional CAF config file path
            test_evolution: Whether to test evolution functionality
            ratio: Per-database injection ratio
            dev_file: Path to BIRD dev set JSON (required when using ratio)
        """
        logger.info("Starting historical data import...")
        logger.info(f"Data file: {data_file}")
        logger.info(f"Database filter: {database_id}")
        logger.info(f"Limit: {limit}")
        logger.info(f"Test evolution: {test_evolution}")
        logger.info(f"Data source: {data_source}")
        if ratio is not None:
            logger.info(f"Ratio-based injection enabled: ratio={ratio}")
            logger.info(f"Dev file: {dev_file}")
        
        try:
            # Load historical data
            records = self.load_historical_data(data_file, data_source=data_source)

            # If ratio specified, compute per-db splits from dev set and filter
            if ratio is not None:
                splits = self._compute_db_question_splits(dev_file, ratio)

                candidate_records = records
                if database_id:
                    candidate_records = [r for r in candidate_records if r.get('db_id') == database_id]

                filtered: List[Dict[str, Any]] = []
                held_out_counts: Dict[str, int] = {}
                for rec in candidate_records:
                    db_id = rec.get('db_id')
                    question = rec.get('question')

                    split = splits.get(db_id)

                    if not split:
                        held_out_counts[db_id] = held_out_counts.get(db_id, 0) + 1
                        continue
                    if question in split['inject_questions']:
                        filtered.append(rec)
                    else:
                        held_out_counts[db_id] = held_out_counts.get(db_id, 0) + 1

                logger.info("Applying ratio-based filtering to historical records...")
                logger.info(f"Records before filtering: {len(records)}")
                logger.info(f"Records after filtering:  {len(filtered)}")

                for db_id, info in splits.items():
                    logger.info(
                        f"DB {db_id}: total={info['counts']['total']}, inject={info['counts']['inject']}, eval={info['counts']['eval']}"
                    )
                if held_out_counts:
                    logger.info("Held-out historical records (not injected) per DB due to split:")
                    for db_id, cnt in held_out_counts.items():
                        logger.info(f"  {db_id}: {cnt}")

                records = filtered
            else:
                if database_id:
                    original_count = len(records)
                    records = [r for r in records if r.get('db_id') == database_id]
                    logger.info(f"Filtered to {len(records)} records for database: {database_id} (from {original_count} total)")
            
            # Import records
            stats = self.import_records(
                records, 
                limit=limit,
                test_evolution=test_evolution,
                data_source=data_source
            )

            # Print summary
            logger.info("\n" + "="*60)
            logger.info("IMPORT SUMMARY")
            logger.info("="*60)
            logger.info(f"Total records in file: {stats['total_records']}")
            logger.info(f"Records imported: {stats['imported_records']}")
            logger.info(f"Successful records (label=True): {stats['successful_records']}")
            logger.info(f"Failed records (label=False/None): {stats['failed_records']}")
            logger.info(f"Skipped records: {stats['skipped_records']}")
            if data_source != 'bird_ground_truth':
                logger.info(f"  -> Skipped records are those with label=True (filtered out for non-ground-truth sources)")
            logger.info(f"Databases processed: {list(stats['databases_processed'])}")
            
            if stats.get('evolution_stats'):
                logger.info(f"Evolution sessions processed: {stats['evolution_stats'].get('sessions_processed', 0)}")
                logger.info(f"Semantic updates: {stats['evolution_stats'].get('semantic_updates', 0)}")
                logger.info(f"Procedural updates: {stats['evolution_stats'].get('procedural_updates', 0)}")
                logger.info(f"Evolution errors: {stats['evolution_stats'].get('errors', 0)}")
            
            if stats['errors']:
                logger.info(f"\nErrors encountered ({len(stats['errors'])}):")
                for error in stats['errors'][:5]:
                    logger.error(f"  - {error}")
                if len(stats['errors']) > 5:
                    logger.info(f"  ... and {len(stats['errors']) - 5} more errors")
            
            logger.info("="*60)
            logger.info("Import completed successfully!")
            
        except Exception as e:
            logger.error(f"Import failed: {e}")
            raise


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Import historical NL2SQL data into CAF memory system")
    parser.add_argument("data_file", help="Path to historical data JSON file")
    parser.add_argument("--database", "-d", help="Filter by specific database ID")
    parser.add_argument("--limit", "-l", type=int, help="Limit number of records to import")
    parser.add_argument("--config", "-c", help="Path to CAF config file")
    parser.add_argument("--no-evolution", action="store_true", help="Skip evolution testing")
    parser.add_argument("--ratio", type=float, default=None, help="Per-database injection ratio in [0,1]; holds out the rest for evaluation")
    parser.add_argument("--dev-file", type=str, default=None, help="Path to BIRD dev set JSON (required when using --ratio)")
    parser.add_argument("--bird-data-path", type=str, default=None, help="Path to BIRD dataset directory")
    parser.add_argument("--cache-dir", type=str, default=None, help="Path to cache directory")
    parser.add_argument(
        "--data-source",
        type=str,
        choices=list(SUPPORTED_DATA_SOURCES.keys()),
        default='nl2sql_bugs',
        help="Type of data to import (affects parsing and metadata)"
    )
    
    args = parser.parse_args()
    
    try:
        importer = HistoricalDataImporter(
            config_path=args.config,
            bird_data_path=args.bird_data_path,
            cache_dir=args.cache_dir
        )
        importer.run_import(
            data_file=args.data_file,
            database_id=args.database,
            limit=args.limit,
            config_path=args.config,
            test_evolution=not args.no_evolution,
            ratio=args.ratio,
            dev_file=args.dev_file,
            data_source=args.data_source
        )
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
