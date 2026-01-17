# Semantic Memory Store - Metadata and domain knowledge storage

import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path

from .base_store import BaseMemoryStore
from ..types import MemoryQuery, MemoryResponse, MemoryItem, DatabaseMetadata, TableMetadata, ColumnMetadata, RelationshipMetadata, TermDefinition, FieldVersion, SOURCE_PRIORITY, VERSIONED_FIELDS
from ..types_per_term import FlatMemoryResponse, PerTermMemoryResponse, JoinRelationship
from ..file_manager import FileManager
from ..search.engines.semantic.engine import SemanticSearchEngine

logger = logging.getLogger(__name__)

METADATA_MODELS = {
    'database': DatabaseMetadata,
    'table': TableMetadata,
    'column': ColumnMetadata,
    'relationship': RelationshipMetadata,
    'term': TermDefinition,
    'field_versions': FieldVersion,
}

class SemanticMemoryStore(BaseMemoryStore):
    """
    Semantic Memory Store - stores database metadata and domain knowledge
    
    Key features:
    - Metadata management with DataFrame storage
    - Support for different metadata types (table, column, relationship, term)
    - File-based persistence using pandas DataFrames
    - Batch import support for BIRD dataset
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Storage configuration - set before calling super().__init__()
        self.storage_path = Path(config.get('semantic', {}).get('storage_path', './memory/semantic_memory'))
        
        super().__init__(config)
        
        # Core components
        self.file_manager = FileManager(self.storage_path)
        self.search_engine = SemanticSearchEngine(config, self.storage_path)
        
        # Current state
        self.current_database_id = None
        self.dataframes: Dict[str, pd.DataFrame] = {}
        
        logger.info("SemanticMemoryStore initialized")
    
    def _initialize_dataframes(self) -> None:
        """
        Ensures that all required dataframes are initialized with the correct schema.
        It uses the Pydantic models from METADATA_MODELS as the source of truth for columns.
        If a dataframe already exists (e.g., loaded from a file), this function ensures
        it has all the columns from the schema, adding any that might be missing from
        older file versions.
        """
        logger.debug("Initializing and verifying dataframe schemas...")
        for metadata_type, model in METADATA_MODELS.items():
            # Get the full list of expected columns from the Pydantic model
            expected_columns = list(model.__fields__.keys())

            if metadata_type not in self.dataframes or self.dataframes[metadata_type] is None:
                # If the dataframe doesn't exist at all, create a new empty one
                logger.debug(f"Creating new empty dataframe for '{metadata_type}'")
                self.dataframes[metadata_type] = pd.DataFrame(columns=expected_columns)
            else:
                # If the dataframe was loaded, check for and add any missing columns
                df = self.dataframes[metadata_type]
                missing_columns = [col for col in expected_columns if col not in df.columns]
                if missing_columns:
                    logger.debug(f"Dataframe '{metadata_type}' is missing columns: {missing_columns}. Adding them.")
                    for col in missing_columns:
                        # Add missing columns with None and object dtype to handle any data type
                        df[col] = None
                        df[col] = df[col].astype('object')
                    # Ensure column order matches the model definition for consistency
                    self.dataframes[metadata_type] = df.reindex(columns=expected_columns)

        logger.debug("Dataframe schema initialization complete.")

    def _upsert_non_versioned_fields(self, metadata_type: str, row_data: Dict[str, Any]) -> None:
        """Upsert only non-versioned fields for a given metadata type using identity keys.
        Ensures a base row exists before versioned field updates occur.
        """
        # Determine identity mask construction per type
        if metadata_type == 'database':
            keys = {'database_id': self.current_database_id}
        elif metadata_type == 'table':
            keys = {'table_name': row_data.get('table_name')}
        elif metadata_type == 'column':
            keys = {'table_name': row_data.get('table_name'), 'column_name': row_data.get('column_name')}
        elif metadata_type == 'term':
            # IMPORTANT: include database_id to keep term definitions database-specific
            keys = {
                'database_id': self.current_database_id,
                'term_name': row_data.get('term_name')
            }
        else:
            return

        versioned = set(VERSIONED_FIELDS.get(metadata_type, []))
        non_versioned_data = {
            k: v for k, v in row_data.items() 
            if k not in versioned and k not in keys and v is not None
        }

        if not non_versioned_data and not keys:
            return

        self._upsert_row(metadata_type, keys, non_versioned_data)
    

    def _add_field_version(self, metadata_type: str, field_name: str, field_value: Any, 
                          source: str, table_name: str = None, column_name: str = None, 
                          term_name: str = None) -> bool:
        """
        Add a field version and update main table if priority is higher or equal.
        
        Returns True if main table was updated, False otherwise
        """
        if not self.current_database_id:
            raise ValueError("No database bound")

        # print(f"Adding field version: {metadata_type}.{field_name} = {field_value} from {source}")
        # Create field version entry
        version_data = {
            'database_id': self.current_database_id,
            'metadata_type': metadata_type,
            'table_name': table_name,
            'column_name': column_name,
            'term_name': term_name,
            'field_name': field_name,
            'field_value': field_value,
            'source': source
        }
        
        # Add to field_versions DataFrame
        if 'field_versions' not in self.dataframes:
            self.dataframes['field_versions'] = pd.DataFrame()
        
        self.dataframes['field_versions'] = pd.concat([
            self.dataframes['field_versions'], 
            pd.DataFrame([version_data])
        ], ignore_index=True)
        
        # Check if we need to update main table
        current_source = self._get_current_field_source(metadata_type, field_name, 
                                                       table_name, column_name, term_name)
        
        # CRITICAL FIX: Check if current value is empty/null
        # If current value is empty, allow any source to update regardless of priority
        current_value = self._get_current_field_value(metadata_type, field_name,
                                                      table_name, column_name, term_name)
        is_current_empty = self._is_value_empty(current_value)
        
        current_priority = SOURCE_PRIORITY.get(current_source, -1) # Use -1 to ensure first entry always updates
        new_priority = SOURCE_PRIORITY.get(source, 0)
        
        # Update if:
        # 1. New priority >= current priority (normal priority check), OR
        # 2. Current value is empty/null (allow any source to fill empty values)
        should_update = new_priority >= current_priority or is_current_empty
        
        if should_update:
            self._update_main_table(metadata_type, field_name, field_value,
                                  table_name, column_name, term_name)
            if is_current_empty:
                logger.debug(f"Updated {metadata_type}.{field_name} from {current_source} (empty value) to {source} (priority {new_priority})")
            else:
                logger.debug(f"Updated {metadata_type}.{field_name} from {current_source} (priority {current_priority}) to {source} (priority {new_priority})")
            return True
        
        logger.debug(f"Skipped update for {metadata_type}.{field_name}. Source {source} (priority {new_priority}) does not supersede {current_source} (priority {current_priority})")
        return False

    def _is_value_empty(self, value: Any) -> bool:
        """
        Check if a value is considered empty (None, NaN, empty string, or whitespace-only string).
        
        Args:
            value: The value to check
            
        Returns:
            True if value is empty, False otherwise
        """
        # Check None
        if value is None:
            return True
        
        # Check pandas NaN
        try:
            if pd.isna(value):
                return True
        except (TypeError, ValueError):
            pass
        
        # Check empty string or whitespace-only string
        if isinstance(value, str):
            return value.strip() == ''
        
        # Check empty list/dict (for composite fields)
        if isinstance(value, (list, dict)):
            return len(value) == 0
        
        return False
    
    def _get_current_field_value(self, metadata_type: str, field_name: str,
                                 table_name: str = None, column_name: str = None,
                                 term_name: str = None) -> Any:
        """
        Get current value of a field from main table.
        
        Args:
            metadata_type: Type of metadata ('column', 'table', etc.)
            field_name: Name of the field
            table_name: Table name (if applicable)
            column_name: Column name (if applicable)
            term_name: Term name (if applicable)
            
        Returns:
            Current field value or None if not found
        """
        if metadata_type not in self.dataframes or self.dataframes[metadata_type].empty:
            return None
        
        df = self.dataframes[metadata_type]
        
        # Build filter conditions
        if metadata_type == 'column' and table_name and column_name:
            mask = (
                (df['database_id'] == self.current_database_id) &
                (df['table_name'] == table_name) & 
                (df['column_name'] == column_name)
            )
        elif metadata_type == 'table' and table_name:
            mask = (
                (df['database_id'] == self.current_database_id) &
                (df['table_name'] == table_name)
            )
        elif metadata_type == 'term' and term_name:
            mask = (
                (df['database_id'] == self.current_database_id) &
                (df['term_name'] == term_name)
            )
        elif metadata_type == 'database':
            mask = df['database_id'] == self.current_database_id
        else:
            return None
        
        matching_rows = df[mask]
        
        if matching_rows.empty:
            return None
        
        # Get the value from first matching row
        if field_name in df.columns:
            return matching_rows.iloc[0][field_name]
        
        return None
    
    def _get_current_field_source(self, metadata_type: str, field_name: str,
                                 table_name: str = None, column_name: str = None,
                                 term_name: str = None) -> str:
        """Get current source for a field in main table"""
        if 'field_versions' not in self.dataframes or self.dataframes['field_versions'].empty:
            return 'unknown'
        
        # Build filter conditions
        df = self.dataframes['field_versions']
        mask = (
            (df['database_id'] == self.current_database_id) &
            (df['metadata_type'] == metadata_type) &
            (df['field_name'] == field_name)
        )
        
        if table_name:
            mask &= (df['table_name'] == table_name)
        if column_name:
            mask &= (df['column_name'] == column_name)
        if term_name:
            mask &= (df['term_name'] == term_name)
        
        matching_versions = df[mask]
        if matching_versions.empty:
            return 'unknown'
        
        # Find highest priority source
        best_source = 'unknown'
        best_priority = 0
        for _, row in matching_versions.iterrows():
            priority = SOURCE_PRIORITY.get(row['source'], 0)
            if priority > best_priority:
                best_priority = priority
                best_source = row['source']
        
        return best_source
    
    def _update_main_table(self, metadata_type: str, field_name: str, field_value: Any,
                          table_name: str = None, column_name: str = None, 
                          term_name: str = None) -> None:
        """Update field value in main table"""
        if metadata_type not in self.dataframes or self.dataframes[metadata_type].empty:
            return
        
        df = self.dataframes[metadata_type]
        
        # Build filter conditions for main table
        # IMPORTANT: Always include database_id filter to prevent cross-database updates
        if metadata_type == 'column' and table_name and column_name:
            mask = (
                (df['database_id'] == self.current_database_id) &
                (df['table_name'] == table_name) & 
                (df['column_name'] == column_name)
            )
        elif metadata_type == 'table' and table_name:
            mask = (
                (df['database_id'] == self.current_database_id) &
                (df['table_name'] == table_name)
            )
        elif metadata_type == 'term' and term_name:
            mask = (
                (df['database_id'] == self.current_database_id) &
                (df['term_name'] == term_name)
            )
        elif metadata_type == 'database':
            mask = df['database_id'] == self.current_database_id
        else:
            return
        
        matching_indices = df.index[mask]
        
        if not matching_indices.empty:
            # Skip None values to avoid overwriting existing data with None
            if field_value is None and not (metadata_type == 'term' and field_name == 'definition'):
                return
                
            for idx in matching_indices:
                # Special merge logic for term.definition: append new definitions instead of overwrite
                if metadata_type == 'term' and field_name == 'definition':
                    existing = self.dataframes[metadata_type].at[idx, field_name]

                    def _normalize_def(val):
                        """Normalize definition value to a list of clean strings, ignoring NaN/None."""
                        # Treat None/NaN as empty
                        try:
                            if val is None or pd.isna(val):
                                return []
                        except Exception:
                            if val is None:
                                return []
                        # List case
                        if isinstance(val, list):
                            out = []
                            for v in val:
                                try:
                                    if v is None or pd.isna(v):
                                        continue
                                except Exception:
                                    if v is None:
                                        continue
                                s = str(v).strip()
                                if s:
                                    out.append(s)
                            return out
                        # Scalar case
                        s = str(val).strip()
                        return [s] if s else []

                    existing_list = _normalize_def(existing)
                    incoming_list = _normalize_def(field_value)

                    merged: List[str] = []
                    for v in existing_list + incoming_list:
                        if v not in merged:
                            merged.append(v)

                    self.dataframes[metadata_type].at[idx, field_name] = merged if merged else None
                else:
                    self.dataframes[metadata_type].at[idx, field_name] = field_value
                

    def _setup_storage(self) -> None:
        """Setup semantic memory storage"""
        # Create storage directories
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Semantic memory storage setup at: {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to setup semantic memory storage: {e}")
            raise
    
    def bind_database(self, database_id: str) -> None:
        """Bind to specific database and load corresponding metadata"""
        super().bind_database(database_id)
        
        # Save current dataframes before switching
        if self.current_database_id and self.dataframes:
            self.save_all_metadata()
        
        self.current_database_id = database_id
        
        # Load dataframes for the new database
        self.dataframes = self.file_manager.load_dataframes(database_id)
        
        self._initialize_dataframes()
        
        # Bind search engine to database
        self.search_engine.bind_database(database_id)
        
        logger.info(f"SemanticMemoryStore bound to database: {database_id}")
    
    def search(self, query: MemoryQuery, return_per_term: bool = False) -> Union[FlatMemoryResponse, Tuple[PerTermMemoryResponse, List[JoinRelationship]]]:
        """
        Search semantic memory using two-stage retrieval architecture
        
        Stage 1: Multi-signal candidate generation (BM25, embeddings, value matching, domain knowledge)
        Stage 2: LLM-based refinement and explanation
        
        Returns:
            - FlatMemoryResponse if return_per_term=False
            - Tuple[PerTermMemoryResponse, List[JoinRelationship]] if return_per_term=True
        """

        if not self.dataframes:
            logger.warning("No metadata available for search")
            if return_per_term:
                return (PerTermMemoryResponse(query_term_results=[], all_tables=[], all_schemas=[]), [])
            else:
                return FlatMemoryResponse(column_items=[], term_items=[], all_tables=[], all_schemas=[], join_relationships=[], query_time_ms=0)

        try:

            response = self.search_engine.search(query, self.dataframes, return_per_term=return_per_term)
            
            return response
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            # Return empty response on failure
            return MemoryResponse(items=[], query_time_ms=0)
    
    def store(self, data: Any) -> None:
        """Store semantic memory data"""
        logger.warning("Use specific add_*_metadata methods instead of generic store()")
    
    def _upsert_row(self, metadata_type: str, keys: Dict[str, Any], update_data: Dict[str, Any]):
        """
        A helper function to find a row by keys, update it with update_data, or create it if it doesn't exist.
        It only updates fields present in update_data.
        """
        if metadata_type not in self.dataframes:
            logger.warning(f"'{metadata_type}' dataframe not found during upsert. Re-initializing.")
            self._initialize_dataframes()

        df = self.dataframes[metadata_type]

        # 直接构建掩码，因为我们确信所有key列都存在
        mask = pd.Series(True, index=df.index)
        for key, value in keys.items():
            if isinstance(value, list):
                # Compare list values by converting to string representation
                mask &= (df[key].astype(str) == str(value))
            else:
                mask &= (df[key] == value)

        matching_indices = df.index[mask]

        if not matching_indices.empty:
            idx = matching_indices.tolist()
            for col, value in update_data.items():
                # Skip None values to avoid overwriting existing data with None
                if value is None:
                    continue
                    
                if col == 'semantic_tags':
                    # Special handling: merge and deduplicate semantic_tags arrays of objects
                    for i in idx:
                        existing = df.at[i, col]

                        # Normalize new value into list, skipping None/NaN
                        if isinstance(value, list):
                            new_list = []
                            for v in value:
                                try:
                                    if v is None or pd.isna(v):
                                        continue
                                except Exception:
                                    if v is None:
                                        continue
                                new_list.append(v)
                        else:
                            try:
                                if value is None or pd.isna(value):
                                    new_list = []
                                else:
                                    new_list = [value]
                            except Exception:
                                new_list = [] if value is None else [value]

                        # If existing is None/NaN, just take new_list
                        use_existing = True
                        try:
                            if existing is None or pd.isna(existing):
                                use_existing = False
                        except Exception:
                            if existing is None:
                                use_existing = False

                        if not use_existing:
                            merged = new_list
                        else:
                            existing_list = existing if isinstance(existing, list) else [existing]
                            merged = existing_list + new_list

                        # Deduplicate by (type, content, source) tuple
                        seen = set()
                        deduped = []
                        for item in merged:
                            try:
                                t = (str(item.get('type')), str(item.get('content')), str(item.get('source')))
                            except Exception:
                                t = (str(item), '', '')
                            if t not in seen:
                                seen.add(t)
                                deduped.append(item)
                        df.at[i, col] = deduped
                else:
                    if isinstance(value, (list, dict)):
                        s = pd.Series([value] * len(idx), index=idx)
                        df.loc[s.index, col] = s
                    else:
                        df.loc[idx, col] = value
        else:
            # Only include non-None values when creating new row (except for keys and database_id)
            filtered_update_data = {k: v for k, v in update_data.items() if v is not None}
            new_row_data = {**keys, **filtered_update_data}
            
            # Ensure database_id is set for metadata types that require it
            if metadata_type in ['column', 'table', 'database', 'relationship', 'term']:
                if 'database_id' not in new_row_data and self.current_database_id:
                    new_row_data['database_id'] = self.current_database_id

            self.dataframes[metadata_type] = pd.concat([
                df,
                pd.DataFrame([new_row_data])
            ], ignore_index=True)

                        
    def add_database_metadata(self, database_metadata: DatabaseMetadata, source: str = 'manual') -> None:
        """Add or update database metadata, only processing non-null fields."""
        if not self.current_database_id:
            raise ValueError("No database bound")

        # --- MODIFICATION START ---
        # 1. Create a dictionary of only the fields provided with non-None values
        update_data = {k: v for k, v in database_metadata.dict().items() if v is not None}
        
        if not update_data:
            logger.debug("Skipping database metadata update; no non-null values provided.")
            return

        # 2. Separate versioned fields from non-versioned fields
        versioned_fields_to_update = {
            k: v for k, v in update_data.items() 
            if k in VERSIONED_FIELDS.get('database', [])
        }
        non_versioned_fields_to_update = {
            k: v for k, v in update_data.items() 
            if k not in VERSIONED_FIELDS.get('database', [])
        }

        # 3. Ensure the base row exists and update non-versioned fields
        # CRITICAL FIX: Always call _upsert_row even if non_versioned_fields_to_update is empty
        keys = {'database_id': self.current_database_id}
        # Always upsert to ensure base row exists
        self._upsert_row('database', keys, non_versioned_fields_to_update)

        # 4. Handle versioned fields via version control
        updated_fields = list(non_versioned_fields_to_update.keys())
        for field_name, field_value in versioned_fields_to_update.items():
            was_updated = self._add_field_version('database', field_name, field_value, source)
            if was_updated:
                updated_fields.append(field_name)

        logger.info(f"Database metadata for '{database_metadata.database_id}' updated (source: {source}, updated fields: {updated_fields})")

    def add_table_metadata(self, table_metadata: TableMetadata, source: str = 'manual') -> None:
        """Add or update table metadata, only processing non-null fields."""
        if not self.current_database_id:
            raise ValueError("No database bound")

        # 1. Create a dictionary of only the fields provided with non-None values
        update_data = {k: v for k, v in table_metadata.dict().items() if v is not None}

        if not update_data or not table_metadata.table_name:
            logger.debug(f"Skipping table metadata update for {table_metadata.table_name}; no data or table name provided.")
            return

        # 2. Separate versioned from non-versioned fields
        versioned_fields_to_update = {
            k: v for k, v in update_data.items() 
            if k in VERSIONED_FIELDS.get('table', [])
        }
        non_versioned_fields_to_update = {
            k: v for k, v in update_data.items() 
            if k not in VERSIONED_FIELDS.get('table', [])
        }

        # 3. Ensure base row exists and update non-versioned fields
        # IMPORTANT: Include database_id in keys to prevent cross-database data mixing
        # CRITICAL FIX: Always call _upsert_row even if non_versioned_fields_to_update is empty
        keys = {
            'database_id': self.current_database_id,
            'table_name': table_metadata.table_name
        }
        # Always upsert to ensure base row exists
        self._upsert_row('table', keys, non_versioned_fields_to_update)

        # 4. Handle versioned fields via version control
        updated_fields = list(non_versioned_fields_to_update.keys())
        for field_name, field_value in versioned_fields_to_update.items():
            was_updated = self._add_field_version(
                'table', field_name, field_value, source,
                table_name=table_metadata.table_name
            )
            if was_updated:
                updated_fields.append(field_name)

        logger.info(f"Table metadata for '{table_metadata.table_name}' updated (source: {source}, updated fields: {updated_fields})")
    
    def add_column_metadata(self, column_metadata: ColumnMetadata, source: str = 'manual') -> None:
        """Add or update column metadata, only processing non-null fields."""
        if not self.current_database_id:
            raise ValueError("No database bound")

        # 1. Create a dictionary of only the fields provided with non-None values
        update_data = {k: v for k, v in column_metadata.dict().items() if v is not None}

        if not update_data or not column_metadata.table_name or not column_metadata.column_name:
            logger.debug(f"Skipping column metadata update for {column_metadata.table_name}.{column_metadata.column_name}; incomplete data provided.")
            return

        # 2. Separate versioned from non-versioned fields
        versioned_fields_to_update = {
            k: v for k, v in update_data.items() 
            if k in VERSIONED_FIELDS.get('column', [])
        }
        non_versioned_fields_to_update = {
            k: v for k, v in update_data.items() 
            if k not in VERSIONED_FIELDS.get('column', [])
        }

        # 3. Ensure base row exists and update non-versioned fields
        # IMPORTANT: Include database_id in keys to prevent cross-database data mixing
        # CRITICAL FIX: Always call _upsert_row even if non_versioned_fields_to_update is empty
        # This ensures the base row exists so versioned fields can be properly updated
        keys = {
            'database_id': self.current_database_id,
            'table_name': column_metadata.table_name, 
            'column_name': column_metadata.column_name
        }
        # Always upsert, even with empty non_versioned_fields, to create base row
        self._upsert_row('column', keys, non_versioned_fields_to_update)
        
        # 4. Handle versioned fields via version control
        updated_fields = list(non_versioned_fields_to_update.keys())
        for field_name, field_value in versioned_fields_to_update.items():
            was_updated = self._add_field_version(
                'column', field_name, field_value, source,
                table_name=column_metadata.table_name,
                column_name=column_metadata.column_name
            )
            if was_updated:
                updated_fields.append(field_name)

        # logger.info(f"Column metadata for '{column_metadata.table_name}.{column_metadata.column_name}' updated (source: {source}, updated fields: {updated_fields})")

    def add_relationship_metadata(self, relationship_metadata: RelationshipMetadata, source: str = 'manual') -> None:
        """Add relationship metadata with source tagging for dedup/upsert"""
        if not self.current_database_id:
            raise ValueError("No database bound")
        
        # Convert to DataFrame row
        row_data = relationship_metadata.dict()

        if source:
            row_data['source'] = source
        
        # Upsert/deduplicate into relationship DataFrame
        if 'relationship' not in self.dataframes:
            self.dataframes['relationship'] = pd.DataFrame()
        
        rel_df = self.dataframes['relationship']
        if not rel_df.empty and {'source_table','source_columns','target_table','target_columns'}.issubset(rel_df.columns):
            # Build mask to match relationship: must include database_id to avoid cross-database matches
            mask = (
                (rel_df['source_table'] == relationship_metadata.source_table) &
                (rel_df['target_table'] == relationship_metadata.target_table)
            )
            # For columns (lists), compare via stringified representation to avoid list equality issues in pandas masks
            if not rel_df.empty and 'source_columns' in rel_df.columns and 'target_columns' in rel_df.columns:
                src_cols_str = str(relationship_metadata.source_columns)
                tgt_cols_str = str(relationship_metadata.target_columns)
                mask = mask & (rel_df['source_columns'].astype(str) == src_cols_str) & (rel_df['target_columns'].astype(str) == tgt_cols_str)
            
            if mask.any():
                idxs = rel_df.index[mask]
                # Only update non-None values to avoid overwriting existing data with None
                # Also skip database_id as it's an identifier and shouldn't be updated
                for key, value in row_data.items():
                    # Skip None values to preserve existing data
                    # Skip database_id as it's an identifier field
                    if value is not None and key != 'database_id':
                        # Ensure the column exists in the DataFrame
                        if key not in self.dataframes['relationship'].columns:
                            self.dataframes['relationship'][key] = None
                        for idx in idxs:
                            self.dataframes['relationship'].at[idx, key] = value
            else:
                # Only include non-None values when creating new row
                filtered_row_data = {k: v for k, v in row_data.items() if v is not None}
                self.dataframes['relationship'] = pd.concat([
                    self.dataframes['relationship'], 
                    pd.DataFrame([filtered_row_data])
                ], ignore_index=True)
        else:
            # Only include non-None values when creating new row
            filtered_row_data = {k: v for k, v in row_data.items() if v is not None}
            self.dataframes['relationship'] = pd.DataFrame([filtered_row_data])
        
        # Log with appropriate column information
        source_cols = relationship_metadata.source_columns
        target_cols = relationship_metadata.target_columns
        source_desc = f"{relationship_metadata.source_table}.[{', '.join(source_cols)}]"
        target_desc = f"{relationship_metadata.target_table}.[{', '.join(target_cols)}]"
        
        logger.info(f"Relationship metadata added: {source_desc} -> {target_desc}")
    
    def add_term_definition(self, term_definition: TermDefinition, source: str = 'manual') -> None:
        """Add term definition with version support"""
        if not self.current_database_id:
            raise ValueError("No database bound")

        # 1) Upsert non-versioned fields (ensure base row exists)
        row_data = term_definition.dict()
        self._upsert_non_versioned_fields('term', row_data)

        # 2) Handle versioned fields via version control
        updated_fields = []
        for field_name in VERSIONED_FIELDS['term']:
            field_value = getattr(term_definition, field_name)
            if field_value is not None:
                was_updated = self._add_field_version('term', field_name, field_value, source,
                                                     term_name=term_definition.term_name)
                if was_updated:
                    updated_fields.append(field_name)

        logger.info(f"Term definition added: {term_definition.term_name} (source: {source}, updated fields: {updated_fields})")
    
    def save_all_metadata(self) -> None:
        """Save all metadata to files"""
        if self.current_database_id:
            self.file_manager.save_dataframes(self.current_database_id, self.dataframes)
            logger.info(f"All metadata saved for database: {self.current_database_id}")
    
    
    def rebuild_search_indexes(self) -> None:
        """Manually rebuild search indexes for current database"""
        if not self.current_database_id:
            raise ValueError("No database bound")
        
        logger.info(f"Rebuilding search indexes for database: {self.current_database_id}")
        self.search_engine.rebuild_indexes(self.dataframes)
        logger.info("Search indexes rebuilt successfully")
    
    def get_search_index_info(self) -> Dict[str, Any]:
        """Get information about current search indexes"""
        return self.search_engine.get_index_info()
    
    def validate_search_capabilities(self) -> Dict[str, Any]:
        """Validate search engine capabilities and configuration"""
        index_info = self.get_search_index_info()
        
        # Check if LLM is configured for refinement
        llm_available = False
        try:
            llm_config = self.search_engine._get_llm_config()
            if llm_config.api_key or llm_config.base_url:
                llm_available = True
        except Exception:
            pass
        
        # Check if embedding model is available
        embedding_available = False
        try:
            embedding_config = self.search_engine._get_embedding_config()
            embedding_available = True
        except Exception:
            pass
        
        return {
            'database_bound': self.current_database_id is not None,
            'indexes_built': index_info.get('status') == 'built',
            'llm_available': llm_available,
            'embeddings_available': embedding_available,
            'matchers_ready': index_info.get('matchers', {}),
            'metadata_counts': {k: len(v) for k, v in self.dataframes.items() if not v.empty}
        }
    
    def get_schema_metadata_for_tables(
        self,
        relevant_schema: Dict[str, Optional[List[str]]],
        include_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get metadata for already-selected schema (from reasoning module)
        
        Args:
            relevant_schema: Dict mapping table_name to list of column_names
                           If column list is None, return all columns for that table
            include_fields: List of metadata field names to include
                          If None, defaults to ['long_description']
                          Available fields: 'long_description', 'short_description', 'description',
                                          'data_type', 'semantic_type', 'encoding_mapping',
                                          'semantic_tags', 'top_k_values', etc.
        
        Returns:
            Dict with keys: 'tables', 'columns', 'joins'
        """
        if not self.current_database_id:
            raise ValueError("No database bound")
        
        # Default to long_description if not specified
        if include_fields is None:
            include_fields = ['long_description']
        
        result = {
            'tables': {},
            'columns': {},
            'joins': []
        }
        
        # Get table metadata
        if 'table' in self.dataframes and not self.dataframes['table'].empty:
            table_df = self.dataframes['table']
            for table_name in relevant_schema.keys():
                # Filter by database_id and table_name
                mask = (
                    (table_df['database_id'] == self.current_database_id) &
                    (table_df['table_name'] == table_name)
                )
                table_rows = table_df[mask]
                
                if not table_rows.empty:
                    # Get first matching row (should only be one)
                    table_row = table_rows.iloc[0]
                    table_metadata = {}
                    
                    # Always include basic info
                    table_metadata['table_name'] = table_name
                    
                    # Include description if available
                    if 'description' in table_row:
                        val = table_row['description']
                        if val is not None and not (isinstance(val, float) and pd.isna(val)):
                            table_metadata['description'] = val
                    
                    # Include row_definition if available
                    if 'row_definition' in table_row:
                        val = table_row['row_definition']
                        if val is not None and not (isinstance(val, float) and pd.isna(val)):
                            table_metadata['row_definition'] = val
                    
                    # Include other fields if requested
                    for field in ['row_count', 'column_count', 'primary_keys', 'foreign_keys']:
                        if field in table_row:
                            val = table_row[field]
                            # Handle different types (scalars, lists, etc.)
                            if val is not None:
                                try:
                                    # For scalar values, check if NaN
                                    if isinstance(val, float) and pd.isna(val):
                                        continue
                                    table_metadata[field] = val
                                except (ValueError, TypeError):
                                    # For arrays/lists, include them directly
                                    table_metadata[field] = val
                    
                    result['tables'][table_name] = table_metadata
        
        # Get column metadata
        if 'column' in self.dataframes and not self.dataframes['column'].empty:
            column_df = self.dataframes['column']
            
            for table_name, column_names in relevant_schema.items():
                # If column_names is None, get all columns for this table
                if column_names is None:
                    mask = (
                        (column_df['database_id'] == self.current_database_id) &
                        (column_df['table_name'] == table_name)
                    )
                    column_rows = column_df[mask]
                else:
                    # Get specific columns
                    for column_name in column_names:
                        mask = (
                            (column_df['database_id'] == self.current_database_id) &
                            (column_df['table_name'] == table_name) &
                            (column_df['column_name'] == column_name)
                        )
                        column_rows = column_df[mask]
                        
                        if not column_rows.empty:
                            column_row = column_rows.iloc[0]
                            column_metadata = {}
                            
                            # Always include basic identifiers
                            column_metadata['table_name'] = table_name
                            column_metadata['column_name'] = column_name
                            
                            # Include requested fields
                            for field in include_fields:
                                if field in column_row:
                                    val = column_row[field]
                                    if val is not None:
                                        try:
                                            # For scalar values, check if NaN
                                            if isinstance(val, float) and pd.isna(val):
                                                continue
                                            column_metadata[field] = val
                                        except (ValueError, TypeError):
                                            # For arrays/lists/dicts, include them directly
                                            column_metadata[field] = val
                            
                            # Use "table.column" as key
                            key = f"{table_name}.{column_name}"
                            result['columns'][key] = column_metadata
                
                # Handle None case (all columns)
                if column_names is None:
                    for _, column_row in column_rows.iterrows():
                        column_name = column_row['column_name']
                        column_metadata = {}
                        
                        # Always include basic identifiers
                        column_metadata['table_name'] = table_name
                        column_metadata['column_name'] = column_name
                        
                        # Include requested fields
                        for field in include_fields:
                            if field in column_row:
                                val = column_row[field]
                                if val is not None:
                                    try:
                                        # For scalar values, check if NaN
                                        if isinstance(val, float) and pd.isna(val):
                                            continue
                                        column_metadata[field] = val
                                    except (ValueError, TypeError):
                                        # For arrays/lists/dicts, include them directly
                                        column_metadata[field] = val
                        
                        # Use "table.column" as key
                        key = f"{table_name}.{column_name}"
                        result['columns'][key] = column_metadata
        
        return result
    
    def get_direct_joins_for_tables(self, table_names: List[str]) -> List[Dict[str, Any]]:
        """
        Get all direct join relationships between given tables
        
        Args:
            table_names: List of table names
        
        Returns:
            List of join relationship dicts with keys:
            - source_table, target_table
            - source_columns, target_columns (lists)
            - relationship_type (if available)
        """
        if not self.current_database_id:
            raise ValueError("No database bound")
        
        joins = []
        
        if 'relationship' not in self.dataframes or self.dataframes['relationship'].empty:
            return joins
        
        rel_df = self.dataframes['relationship']
        table_set = set(table_names)
        
        # Find relationships where both source and target are in our table list
        for _, row in rel_df.iterrows():
            # Check database_id match
            if 'database_id' in row and row['database_id'] != self.current_database_id:
                continue
            
            source_table = row.get('source_table')
            target_table = row.get('target_table')
            
            # Check if both tables are in our list
            if source_table in table_set and target_table in table_set:
                join_info = {
                    'source_table': source_table,
                    'target_table': target_table,
                    'source_columns': row.get('source_columns', []),
                    'target_columns': row.get('target_columns', []),
                }
                
                # Add optional fields
                for field in ['relationship_type', 'cardinality', 'business_meaning']:
                    if field in row:
                        val = row[field]
                        if val is not None:
                            try:
                                if isinstance(val, float) and pd.isna(val):
                                    continue
                                join_info[field] = val
                            except (ValueError, TypeError):
                                join_info[field] = val
                
                joins.append(join_info)
        
        return joins
    
    def cleanup(self) -> None:
        """Cleanup semantic memory resources"""
        # Save any unsaved dataframes
        if self.current_database_id and self.dataframes:
            self.save_all_metadata()
        
        super().cleanup()
        logger.info("SemanticMemoryStore cleanup completed")

    # def delete_column_field(self, database_id: str, table_name: str, column_name: str,
    #                          field_name: str, save: bool = True, remove_versions: bool = False) -> bool:
    #     """Delete (clear) the value of a specific field for a column.

    #     This sets the field value in the main `column` metadata table to None.
    #     Optionally removes matching entries in `field_versions`.

    #     Returns True if a row was found and updated, False otherwise.
    #     """
    #     # Bind to the requested database (saving current if necessary happens inside)
    #     if self.current_database_id != database_id:
    #         self.bind_database(database_id)

    #     if 'column' not in self.dataframes or self.dataframes['column'].empty:
    #         return False

    #     df = self.dataframes['column']

    #     if field_name not in df.columns:
    #         logger.warning(f"Field '{field_name}' not found in column metadata schema; cannot delete.")
    #         return False

    #     # Locate the row for the specified column
    #     mask = (df['table_name'] == table_name) & (df['column_name'] == column_name)
    #     matching_indices = df.index[mask]

    #     if matching_indices.empty:
    #         logger.info(f"Column not found for deletion: {table_name}.{column_name}")
    #         return False

    #     # Clear the field value
    #     for idx in matching_indices:
    #         self.dataframes['column'].at[idx, field_name] = None

    #     # Optionally remove historical versions of this field
    #     if remove_versions and 'field_versions' in self.dataframes and not self.dataframes['field_versions'].empty:
    #         vdf = self.dataframes['field_versions']
    #         vmask = (
    #             (vdf['database_id'] == database_id) &
    #             (vdf['metadata_type'] == 'column') &
    #             (vdf['field_name'] == field_name) &
    #             (vdf['table_name'] == table_name) &
    #             (vdf['column_name'] == column_name)
    #         )
    #         self.dataframes['field_versions'] = vdf.loc[~vmask].copy()

    #     if save:
    #         self.save_all_metadata()

    #     logger.info(f"Deleted field '{field_name}' for {table_name}.{column_name} in database '{database_id}'")
    #     return True

    def update_column_field(self, database_id: str, table_name: str, column_name: str,
                            field_name: str, field_value: Any, source: str = 'manual',
                            save: bool = True) -> bool:
        """Update a specific field for a column using standard upsert logic.

        - Versioned fields are recorded via _add_field_version (priority-aware).
        - Non-versioned fields use _upsert_row. Special handling for `semantic_tags`
          is implemented inside _upsert_row (merge + dedup) and not here.

        Returns True if the column exists or a base row is created/updated.
        """
        if self.current_database_id != database_id:
            self.bind_database(database_id)

        if 'column' not in self.dataframes or self.dataframes['column'].empty:
            return False

        df = self.dataframes['column']
        if field_name not in df.columns:
            logger.warning(f"Field '{field_name}' not found in column metadata schema; cannot update.")
            return False

        mask = (df['table_name'] == table_name) & (df['column_name'] == column_name)
        matching_indices = df.index[mask]
        if matching_indices.empty:
            logger.info(f"Column not found for update: {table_name}.{column_name}")
            return False

        # Determine if field is versioned
        is_versioned = field_name in set(VERSIONED_FIELDS.get('column', []))

        if is_versioned:
            # Versioned field: record a new version; main table update handled by priority logic
            was_updated = self._add_field_version(
                'column', field_name, field_value, source,
                table_name=table_name, column_name=column_name
            )
            if not was_updated:
                logger.debug(f"Version recorded but main table not updated due to source priority: {table_name}.{column_name}.{field_name}")
        else:
            # Non-versioned: use _upsert_row (with semantic_tags special handling inside)
            self._upsert_row(
                'column',
                keys={'table_name': table_name, 'column_name': column_name},
                update_data={field_name: field_value}
            )

        if save:
            self.save_all_metadata()

        logger.info(f"Updated field '{field_name}' for {table_name}.{column_name} in database '{database_id}' (source: {source})")
        return True

    def delete_field(self, database_id: str, metadata_type: str, field_name: str,
                     table_name: str = None, column_name: str = None, term_name: str = None,
                     save: bool = True, remove_versions: bool = False) -> bool:
        """Delete (clear) the value of a specific field for a metadata item.

        Supports database, table, column, and term metadata types.
        This sets the field value in the main metadata table to None.
        Optionally removes matching entries in `field_versions`.

        Args:
            database_id: Database ID
            metadata_type: Type of metadata ('database', 'table', 'column', 'term')
            field_name: Field name to delete
            table_name: Table name (required for 'table' and 'column' types)
            column_name: Column name (required for 'column' type)
            term_name: Term name (required for 'term' type)
            save: Whether to save changes immediately
            remove_versions: Whether to remove historical versions

        Returns:
            True if a row was found and updated, False otherwise.
        """
        # Validate metadata type
        if metadata_type not in ['database', 'table', 'column', 'term']:
            logger.error(f"Invalid metadata_type: {metadata_type}. Must be one of: database, table, column, term")
            return False

        # Validate required parameters based on metadata type
        if metadata_type == 'table' and not table_name:
            logger.error("table_name is required for table metadata")
            return False
        if metadata_type == 'column' and (not table_name or not column_name):
            logger.error("table_name and column_name are required for column metadata")
            return False
        if metadata_type == 'term' and not term_name:
            logger.error("term_name is required for term metadata")
            return False

        # Bind to the requested database (saving current if necessary happens inside)
        if self.current_database_id != database_id:
            self.bind_database(database_id)

        # Check if dataframe exists
        if metadata_type not in self.dataframes or self.dataframes[metadata_type].empty:
            logger.info(f"No {metadata_type} metadata found for deletion")
            return False

        df = self.dataframes[metadata_type]

        if field_name not in df.columns:
            logger.warning(f"Field '{field_name}' not found in {metadata_type} metadata schema; cannot delete.")
            return False

        # Build mask to locate the row
        mask = pd.Series(True, index=df.index)
        
        # Always filter by database_id for all types
        if 'database_id' in df.columns:
            mask &= (df['database_id'] == database_id)
        
        # Add type-specific filters
        if metadata_type == 'table' and table_name:
            mask &= (df['table_name'] == table_name)
        elif metadata_type == 'column' and table_name and column_name:
            mask &= (df['table_name'] == table_name) & (df['column_name'] == column_name)
        elif metadata_type == 'term' and term_name:
            mask &= (df['term_name'] == term_name)
        # database type only needs database_id filter (already applied above)

        matching_indices = df.index[mask]

        if matching_indices.empty:
            if metadata_type == 'database':
                logger.info(f"Database metadata not found for deletion: {database_id}")
            elif metadata_type == 'table':
                logger.info(f"Table metadata not found for deletion: {table_name}")
            elif metadata_type == 'column':
                logger.info(f"Column metadata not found for deletion: {table_name}.{column_name}")
            elif metadata_type == 'term':
                logger.info(f"Term metadata not found for deletion: {term_name}")
            return False

        # Clear the field value
        for idx in matching_indices:
            self.dataframes[metadata_type].at[idx, field_name] = None

        # Optionally remove historical versions of this field
        if remove_versions and 'field_versions' in self.dataframes and not self.dataframes['field_versions'].empty:
            vdf = self.dataframes['field_versions']
            vmask = (
                (vdf['database_id'] == database_id) &
                (vdf['metadata_type'] == metadata_type) &
                (vdf['field_name'] == field_name)
            )
            
            # Add type-specific filters for field_versions
            if metadata_type == 'table' and table_name:
                vmask &= (vdf['table_name'] == table_name)
            elif metadata_type == 'column' and table_name and column_name:
                vmask &= (vdf['table_name'] == table_name) & (vdf['column_name'] == column_name)
            elif metadata_type == 'term' and term_name:
                vmask &= (vdf['term_name'] == term_name)
            
            self.dataframes['field_versions'] = vdf.loc[~vmask].copy()

        if save:
            self.save_all_metadata()

        # Log success message
        if metadata_type == 'database':
            logger.info(f"Deleted field '{field_name}' for database '{database_id}'")
        elif metadata_type == 'table':
            logger.info(f"Deleted field '{field_name}' for table '{table_name}' in database '{database_id}'")
        elif metadata_type == 'column':
            logger.info(f"Deleted field '{field_name}' for {table_name}.{column_name} in database '{database_id}'")
        elif metadata_type == 'term':
            logger.info(f"Deleted field '{field_name}' for term '{term_name}' in database '{database_id}'")
        
        return True
