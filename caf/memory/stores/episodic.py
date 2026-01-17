# Episodic Memory Store - Historical interaction records storage (Refactored)

import json
import logging
import string
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd

from .base_store import BaseMemoryStore
from ..types import (
    MemoryQuery, MemoryResponse, MemoryItem, EpisodicRecord,
    InteractionRound, UserFeedback, SQLExecutionResult
)
from ..search.engines.episodic import EpisodicSearchEngine

logger = logging.getLogger(__name__)

class EpisodicMemoryStore(BaseMemoryStore):
    """
    Episodic Memory Store - stores complete interaction history (Refactored)
    
    Key Changes:
    - Uses EpisodicSearchEngine for all search operations
    - Simplified responsibilities: focus on storage and data management
    - Removed duplicate retrieval logic (delegated to search engine)
    - Maintains compatibility with existing storage format
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Storage configuration - set before calling super().__init__()
        self.storage_path = Path(config.get('episodic', {}).get('storage_path', './memory/episodic_memory'))
        
        super().__init__(config)
        
        # Memory cache for recent records
        self.recent_records_cache: Dict[str, List[EpisodicRecord]] = {}  # database_id -> List[EpisodicRecord]
        self.cache_size = config.get('episodic', {}).get('cache_size', 100)
        
        # Current state
        self.current_database_id = None
        
        # Session ID generation
        self._session_id_counter: Dict[str, int] = {}  # database_id -> counter
        
        # Search engine for retrieval operations
        self.search_engine = EpisodicSearchEngine(config, self.storage_path)
        
        # Memory base reference (set by MemoryBase after initialization)
        self._memory_base = None
        
        logger.info("EpisodicMemoryStore initialized with new search engine")
    
    def _setup_storage(self) -> None:
        """Setup episodic memory storage"""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Episodic memory storage setup at: {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to setup episodic memory storage: {e}")
            raise
    
    def bind_database(self, database_id: str) -> None:
        """Bind to specific database"""
        super().bind_database(database_id)
        self.current_database_id = database_id
        
        # Set memory base for search engine if available (before binding)
        if self._memory_base:
            self.search_engine.set_memory_base(self._memory_base)
        
        # Bind search engine to database
        self.search_engine.bind_database(database_id)
        
        logger.info(f"EpisodicMemoryStore bound to database: {database_id}")
    
    def set_memory_base(self, memory_base) -> None:
        """Set memory base reference for search engine"""
        self._memory_base = memory_base
        if self.search_engine:
            self.search_engine.set_memory_base(memory_base)
    
    def search(self, query: MemoryQuery) -> MemoryResponse:
        """
        Search episodic memory using EpisodicSearchEngine
        
        Args:
            query: Memory query object
            
        Returns:
            MemoryResponse with ranked results
        """
        if not self.search_engine:
            logger.error("Search engine not initialized")
            return MemoryResponse(items=[], total_count=0, query_time_ms=0)
        
        # Load all relevant records from ALL databases (not just current database)
        # This is necessary for cross-database retrieval in multi-layer retriever
        all_records = self._load_all_records(load_all_databases=True)
        
        if not all_records:
            logger.info("No records found for search")
            return MemoryResponse(items=[], total_count=0, query_time_ms=0)
        
        # Delegate to search engine
        return self.search_engine.search(query, all_records)
    
    def store(self, data: Any) -> None:
        """Store episodic data - accepts EpisodicRecord"""
        if isinstance(data, EpisodicRecord):
            self.store_record(data)
        else:
            logger.warning(f"Invalid data type for episodic storage: {type(data)}")
    
    def store_record(self, record: EpisodicRecord) -> None:
        """Store single episodic record with deduplication check
        
        Checks for duplicates based on:
        1. session_id (unique key) - if session_id already exists, reject
        2. database_id + user_query + generated_sql combination - if identical, reject
        """
        if not record.database_id:
            record.database_id = self.current_database_id
        
        # First check: session_id must be unique (primary key constraint)
        if self._is_duplicate_session_id(record.session_id):
            logger.warning(f"Duplicate session_id rejected: session_id={record.session_id}")
            return
        
        # Second check: database_id + user_query + generated_sql must be unique
        if self._is_duplicate_record(record):
            logger.warning(f"Duplicate episodic record rejected: database_id={record.database_id}, "
                         f"user_query={record.user_query[:50]}..., generated_sql={record.generated_sql[:50] if record.generated_sql else None}...")
            return
        
        try:
            # Store to file system (DataFrame/Parquet)
            storage_file = self._get_storage_file(record.database_id, record.timestamp)
            existing_df = self._load_records_from_file(storage_file)
            
            # Double-check session_id uniqueness in the target file before writing
            if existing_df is not None and not existing_df.empty:
                if 'session_id' in existing_df.columns:
                    if record.session_id in existing_df['session_id'].values:
                        logger.warning(f"Duplicate session_id found in storage file: session_id={record.session_id}")
                        return
            
            # Convert record to dict and add to DataFrame
            record_dict = self._record_to_dict(record)
            new_df = pd.DataFrame([record_dict])
            
            if existing_df is not None and not existing_df.empty:
                # Append to existing DataFrame
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            
            self._save_records_to_file(storage_file, combined_df)
            
            # Update cache
            self._update_cache(record)
            
            # Invalidate search engine cache since we added new records
            if hasattr(self.search_engine, 'embedding_cache'):
                self.search_engine.embedding_cache.invalidate_cache()
            
            logger.info(f"Episodic record stored: {record.session_id} (round {record.round_id})")
            
        except Exception as e:
            logger.error(f"Failed to store episodic record {record.session_id}: {e}")
            raise
    
    def store_records_batch(self, records: List[EpisodicRecord]) -> Dict[str, Any]:
        """
        Batch store multiple episodic records with optimized deduplication and I/O
        
        Args:
            records: List of EpisodicRecord objects (session_id can be None)
            
        Returns:
            Statistics dict with counts of stored/skipped records
        """
        if not records:
            return {'stored': 0, 'skipped_duplicate_session_id': 0, 'skipped_duplicate_content': 0, 'errors': 0}
        
        # Initialize statistics
        stats = {
            'stored': 0,
            'skipped_duplicate_session_id': 0,
            'skipped_duplicate_content': 0,
            'errors': 0,
            'error_details': []
        }
        
        # Ensure database is bound
        if not self.current_database_id:
            raise ValueError("No database bound for batch storage")
        
        # Set database_id for records that don't have it
        for record in records:
            if not record.database_id:
                record.database_id = self.current_database_id
        
        try:
            # Step 1: Load existing data for deduplication (one-time load)
            existing_session_ids, existing_content_keys = self._load_existing_keys_batch()
            
            # Step 2: Generate session_ids and perform deduplication
            valid_records = []
            for record in records:
                try:
                    # Generate session_id if not provided
                    if not record.session_id:
                        record.session_id = self._generate_next_session_id(record.database_id)
                    
                    # Check session_id uniqueness
                    if record.session_id in existing_session_ids:
                        stats['skipped_duplicate_session_id'] += 1
                        logger.debug(f"Skipped duplicate session_id: {record.session_id}")
                        continue
                    
                    # Check content uniqueness
                    content_key = self._generate_content_key(record)
                    if content_key in existing_content_keys:
                        stats['skipped_duplicate_content'] += 1
                        logger.debug(f"Skipped duplicate content: {content_key[:50]}...")
                        continue
                    
                    # Add to valid records and update tracking sets
                    valid_records.append(record)
                    existing_session_ids.add(record.session_id)
                    existing_content_keys.add(content_key)
                    
                except Exception as e:
                    stats['errors'] += 1
                    stats['error_details'].append(f"Record processing error: {e}")
                    logger.error(f"Error processing record: {e}")
            
            if not valid_records:
                logger.info("No valid records to store after deduplication")
                return stats
            
            # Step 3: Group records by storage file (year-based)
            file_groups = self._group_records_by_storage_file(valid_records)
            
            # Step 4: Batch write to each file
            for storage_file, file_records in file_groups.items():
                try:
                    self._batch_write_to_file(storage_file, file_records)
                    stats['stored'] += len(file_records)
                except Exception as e:
                    stats['errors'] += len(file_records)
                    error_msg = f"Failed to write {len(file_records)} records to {storage_file}: {e}"
                    stats['error_details'].append(error_msg)
                    logger.error(error_msg)
            
            # Step 5: Update cache with valid records
            for record in valid_records:
                self._update_cache(record)
            
            # Step 6: Invalidate search engine cache
            if hasattr(self.search_engine, 'embedding_cache'):
                self.search_engine.embedding_cache.invalidate_cache()
            
            logger.info(f"Batch stored {stats['stored']} records, "
                       f"skipped {stats['skipped_duplicate_session_id'] + stats['skipped_duplicate_content']} duplicates, "
                       f"{stats['errors']} errors")
            
        except Exception as e:
            logger.error(f"Batch storage failed: {e}")
            stats['errors'] += len(records)
            stats['error_details'].append(f"Batch storage error: {e}")
            raise
        
        return stats
    
    def _generate_next_session_id(self, database_id: str) -> str:
        """
        Generate next unique session_id for the database
        
        Format: {database_id}_{counter:06d}
        Example: college_2_000001, college_2_000002, etc.
        """
        # Initialize counter for database if not exists
        if database_id not in self._session_id_counter:
            # Find the current max counter from existing records
            self._session_id_counter[database_id] = self._get_max_session_counter(database_id)
        
        # Increment and generate session_id
        self._session_id_counter[database_id] += 1
        return f"{database_id}_{self._session_id_counter[database_id]:06d}"
    
    def _get_max_session_counter(self, database_id: str) -> int:
        """
        Get the maximum session counter for a database from existing records
        """
        max_counter = 0
        
        try:
            # Check cache
            if database_id in self.recent_records_cache:
                for record in self.recent_records_cache[database_id]:
                    if record.session_id and record.session_id.startswith(f"{database_id}_"):
                        try:
                            counter = int(record.session_id.split('_')[-1])
                            max_counter = max(max_counter, counter)
                        except (ValueError, IndexError):
                            pass
            
            # Check storage files
            for storage_file in self.storage_path.glob("records_*.parquet"):
                df = self._load_records_from_file(storage_file)
                if df is not None and not df.empty and 'session_id' in df.columns:
                    # Filter by database_id and extract counters
                    db_records = df[df['database_id'] == database_id]
                    for session_id in db_records['session_id'].dropna():
                        if session_id.startswith(f"{database_id}_"):
                            try:
                                counter = int(session_id.split('_')[-1])
                                max_counter = max(max_counter, counter)
                            except (ValueError, IndexError):
                                pass
                                
        except Exception as e:
            logger.warning(f"Error getting max session counter for {database_id}: {e}")
        
        logger.debug(f"Max session counter for {database_id}: {max_counter}")
        return max_counter
    
    def _generate_content_key(self, record: EpisodicRecord) -> str:
        """
        Generate content key for deduplication based on database_id + user_query + generated_sql
        """
        normalized_question = self._normalize_question_for_comparison(record.user_query)
        normalized_sql = self._normalize_sql_for_comparison(record.generated_sql) if record.generated_sql else ""
        return f"{record.database_id}||{normalized_question}||{normalized_sql}"
    
    def _load_existing_keys_batch(self) -> Tuple[set, set]:
        """
        Load all existing session_ids and content keys for deduplication
        
        Returns:
            (session_ids_set, content_keys_set)
        """
        session_ids = set()
        content_keys = set()
        
        try:
            # Load from cache
            for db_id, cache in self.recent_records_cache.items():
                for record in cache:
                    if record.session_id:
                        session_ids.add(record.session_id)
                    content_keys.add(self._generate_content_key(record))
            
            # Load from storage files
            for storage_file in self.storage_path.glob("records_*.parquet"):
                df = self._load_records_from_file(storage_file)
                if df is not None and not df.empty:
                    # Collect session_ids
                    if 'session_id' in df.columns:
                        session_ids.update(df['session_id'].dropna().tolist())
                    
                    # Generate content keys
                    for _, row in df.iterrows():
                        try:
                            # Create temporary record for content key generation
                            temp_record = EpisodicRecord(
                                session_id="temp",
                                database_id=row.get('database_id'),
                                user_query=row.get('user_query', ''),
                                round_id=1,
                                generated_sql=row.get('generated_sql')
                            )
                            content_keys.add(self._generate_content_key(temp_record))
                        except Exception:
                            pass  # Skip malformed rows
                            
        except Exception as e:
            logger.warning(f"Error loading existing keys: {e}")
        
        logger.debug(f"Loaded {len(session_ids)} session_ids and {len(content_keys)} content keys")
        return session_ids, content_keys
    
    def _group_records_by_storage_file(self, records: List[EpisodicRecord]) -> Dict[Path, List[EpisodicRecord]]:
        """
        Group records by their target storage file (year-based)
        """
        file_groups = {}
        for record in records:
            storage_file = self._get_storage_file(record.database_id, record.timestamp)
            if storage_file not in file_groups:
                file_groups[storage_file] = []
            file_groups[storage_file].append(record)
        return file_groups
    
    def _batch_write_to_file(self, storage_file: Path, records: List[EpisodicRecord]) -> None:
        """
        Batch write records to a single storage file
        """
        # Load existing DataFrame
        existing_df = self._load_records_from_file(storage_file)
        
        # Convert records to DataFrame
        record_dicts = [self._record_to_dict(record) for record in records]
        new_df = pd.DataFrame(record_dicts)
        
        # Combine and save
        if existing_df is not None and not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        self._save_records_to_file(storage_file, combined_df)
        logger.debug(f"Batch wrote {len(records)} records to {storage_file}")
    
    def rebuild_search_indexes(self) -> None:
        """Rebuild search indexes for current database"""
        if not self.current_database_id:
            raise ValueError("No database bound")
        
        logger.info(f"Rebuilding search indexes for database: {self.current_database_id}")
        
        # Load all records
        all_records = self._load_all_records()
        
        # Rebuild search engine indexes
        self.search_engine.rebuild_indexes(all_records)
        
        logger.info("Search indexes rebuilt successfully")
    
    def get_search_index_info(self) -> Dict[str, Any]:
        """Get information about current search indexes"""
        return self.search_engine.get_index_info()
    
    # === Storage Methods (unchanged from original) ===
    
    def _is_duplicate_session_id(self, session_id: str) -> bool:
        """
        Check if session_id already exists (session_id is unique key)
        
        Args:
            session_id: Session ID to check
            
        Returns:
            True if session_id already exists, False otherwise
        """
        # Check in cache first
        for db_id, cache in self.recent_records_cache.items():
            for existing_record in cache:
                if existing_record.session_id == session_id:
                    return True
        
        # Check in unified storage files
        try:
            for storage_file in self.storage_path.glob("records_*.parquet"):
                existing_df = self._load_records_from_file(storage_file)
                if existing_df is not None and not existing_df.empty:
                    if 'session_id' in existing_df.columns:
                        if session_id in existing_df['session_id'].values:
                            return True
        except Exception as e:
            logger.warning(f"Error checking for duplicate session_id: {e}")
            return False
        
        return False
    
    def _is_duplicate_record(self, record: EpisodicRecord) -> bool:
        """
        Check if a record is duplicate based on database_id, user_query, and generated_sql.
        
        This ensures that the same (question, SQL) combination is not stored multiple times,
        even if session_id is different.
        """
        database_id = record.database_id
        user_query = record.user_query
        generated_sql = record.generated_sql
        
        # Normalize inputs for comparison (lowercase, strip whitespace)
        normalized_question = self._normalize_question_for_comparison(user_query)
        normalized_sql = self._normalize_sql_for_comparison(generated_sql) if generated_sql else None
        
        # Check in cache first
        if database_id in self.recent_records_cache:
            for existing_record in self.recent_records_cache[database_id]:
                if (self._normalize_question_for_comparison(existing_record.user_query) == normalized_question and
                    self._normalize_sql_for_comparison(existing_record.generated_sql) == normalized_sql):
                    return True
        
        # Check in unified storage files
        try:
            for storage_file in self.storage_path.glob("records_*.parquet"):
                existing_df = self._load_records_from_file(storage_file)
                if existing_df is not None and not existing_df.empty:
                    # Filter by database_id and check for duplicates
                    db_records = existing_df[existing_df['database_id'] == database_id]
                    for _, row in db_records.iterrows():
                        row_question = row.get('user_query') if hasattr(row, 'get') else row['user_query']
                        if (self._normalize_question_for_comparison(row_question) == normalized_question and
                            self._normalize_sql_for_comparison(row.get('generated_sql')) == normalized_sql):
                            return True
        except Exception as e:
            logger.warning(f"Error checking for duplicate record: {e}")
            return False
        
        return False
    
    def _normalize_sql_for_comparison(self, sql: Optional[str]) -> Optional[str]:
        """Normalize SQL for comparison by removing extra whitespace and converting to lowercase."""
        if not sql:
            return None
        
        # Remove extra whitespace and convert to lowercase
        normalized = ' '.join(sql.strip().split()).lower()
        return normalized
    
    def _normalize_question_for_comparison(self, question: Optional[str]) -> Optional[str]:
        """Normalize user questions for duplicate comparison by lowercasing, trimming whitespace,
        and removing trailing punctuation."""
        if not question:
            return None
        
        # Normalize: trim whitespace, lowercase, and remove trailing punctuation
        normalized = ' '.join(question.strip().split()).lower()
        
        # Remove trailing punctuation marks
        while normalized and normalized[-1] in string.punctuation:
            normalized = normalized[:-1]
        
        return normalized
    
    def _get_storage_file(self, database_id: str, timestamp: str) -> Path:
        """Determine storage file based on timestamp (unified storage by year)"""
        try:
            date_obj = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            date_obj = datetime.utcnow()
        
        year = date_obj.strftime("%Y")
        return self.storage_path / f"records_{year}.parquet"
    
    def _load_records_from_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load records from parquet file"""
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_parquet(file_path)
            return df
        except Exception as e:
            logger.error(f"Failed to load records from {file_path}: {e}")
            return None
    
    def _save_records_to_file(self, file_path: Path, df: pd.DataFrame) -> None:
        """Save records to parquet file"""
        try:
            df.to_parquet(file_path, index=False, engine='pyarrow')
        except Exception as e:
            logger.error(f"Failed to save records to {file_path}: {e}")
            raise
    
    def _record_to_dict(self, record: EpisodicRecord) -> Dict[str, Any]:
        """Convert EpisodicRecord to dictionary for DataFrame storage"""
        record_dict = {
            'session_id': record.session_id,
            'database_id': record.database_id,
            'user_query': record.user_query,
            'context': record.context,
            'round_id': record.round_id,
            'generated_sql': record.generated_sql,
            'execution_result': json.dumps(record.execution_result.model_dump() if record.execution_result and hasattr(record.execution_result, 'model_dump') else (record.execution_result.dict() if record.execution_result and hasattr(record.execution_result, 'dict') else None)),
            'label': record.label,
            'error_types': json.dumps(record.error_types) if record.error_types else None,
            'feedback_text': record.feedback_text,
            'source_model': record.source_model,
            'timestamp': record.timestamp,
            'metadata': json.dumps(record.metadata) if record.metadata else None
        }
        return record_dict
    
    def _dict_to_record(self, record_dict: Dict[str, Any]) -> EpisodicRecord:
        """Convert dictionary from DataFrame to EpisodicRecord"""
        # Parse JSON fields
        execution_result = None
        if record_dict.get('execution_result'):
            try:
                exec_data = json.loads(record_dict['execution_result'])
                execution_result = SQLExecutionResult(**exec_data) if exec_data else None
            except:
                execution_result = None
        
        error_types = None
        if record_dict.get('error_types'):
            try:
                error_types = json.loads(record_dict['error_types'])
            except:
                error_types = None
        
        metadata = None
        if record_dict.get('metadata'):
            try:
                metadata = json.loads(record_dict['metadata'])
            except:
                metadata = None
        
        return EpisodicRecord(
            session_id=record_dict['session_id'],
            database_id=record_dict.get('database_id'),
            user_query=record_dict['user_query'],
            context=record_dict.get('context'),
            round_id=record_dict['round_id'],
            generated_sql=record_dict.get('generated_sql'),
            execution_result=execution_result,
            label=record_dict.get('label'),
            error_types=error_types,
            feedback_text=record_dict.get('feedback_text'),
            source_model=record_dict.get('source_model'),
            timestamp=record_dict['timestamp'],
            metadata=metadata
        )
    
    def _update_cache(self, record: EpisodicRecord) -> None:
        """Update recent records cache"""
        database_id = record.database_id
        if database_id not in self.recent_records_cache:
            self.recent_records_cache[database_id] = []
        
        cache = self.recent_records_cache[database_id]
        cache.append(record)
        
        if len(cache) > self.cache_size:
            cache.pop(0)
    
    def _load_all_records(self, database_id: str = None, load_all_databases: bool = False) -> List[EpisodicRecord]:
        """
        Load all unique records from cache and storage files, optionally filtered by database_id.
        
        Args:
            database_id: Database ID to filter by. If None and load_all_databases=False, uses current_database_id.
            load_all_databases: If True, load records from all databases regardless of database_id parameter.
                              This is needed for cross-database retrieval.
        """
        all_records: Dict[str, EpisodicRecord] = {}
        
        # Determine target database
        if load_all_databases:
            target_database_id = None  # Load all databases
        elif database_id is not None:
            target_database_id = database_id
        else:
            target_database_id = self.current_database_id
        
        # 1. Load from cache first (if database_id filter is provided)
        if target_database_id and target_database_id in self.recent_records_cache:
            for record in self.recent_records_cache[target_database_id]:
                key = f"{record.session_id}_{record.round_id}"
                all_records[key] = record
        elif not target_database_id:
            # Load from all caches if no database filter
            for db_id, cache in self.recent_records_cache.items():
                for record in cache:
                    key = f"{record.session_id}_{record.round_id}"
                    all_records[key] = record
        
        # 2. Load from unified storage files
        try:
            unified_files = sorted(self.storage_path.glob("records_*.parquet"), reverse=True)[:3]
            for storage_file in unified_files:
                df = self._load_records_from_file(storage_file)
                if df is not None and not df.empty:
                    # Filter by database_id if specified
                    if target_database_id:
                        df = df[df['database_id'] == target_database_id]
                    
                    for _, row in df.iterrows():
                        key = f"{row['session_id']}_{row['round_id']}"
                        if key not in all_records:
                            record = self._dict_to_record(row.to_dict())
                            all_records[key] = record
        except Exception as e:
            logger.warning(f"Failed to load some records from storage: {e}")
        
        loaded_records = list(all_records.values())
        db_filter_msg = f" for DB {target_database_id}" if target_database_id else " (all databases)"
        logger.debug(f"Loaded {len(loaded_records)} total unique records{db_filter_msg}")
        return loaded_records
    
    def cleanup(self) -> None:
        """Cleanup episodic memory resources"""
        super().cleanup()
        self.recent_records_cache.clear()
        logger.info("EpisodicMemoryStore cleanup completed")
