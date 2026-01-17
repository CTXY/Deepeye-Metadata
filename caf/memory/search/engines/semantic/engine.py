# Semantic Search Engine - Core search functionality for semantic memory 

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import math

from ....types import MemoryQuery, MemoryResponse, MemoryItem
from ....types_per_term import (
    PerTermMemoryResponse, QueryTermResult, ColumnItem, TermItem, JoinRelationship, FlatMemoryResponse
)
from caf.llm.client import create_llm_client, LLMConfig
from caf.config.global_config import get_llm_config
from ...embedding.client import EmbeddingConfig
from ...embedding.client import create_embedding_client
from ...base import BaseSearchEngine
from .retrievers.column_retriever import AnchorColumnRetriever, PerQueryTermSelectionResult
from .retrievers.term_retriever import IndependentTermRetriever, PerQueryTermTermResult
from caf.preprocess.sql_preprocessor import SQLPreprocessor

logger = logging.getLogger(__name__)

class SemanticSearchEngine(BaseSearchEngine):
    """
    Semantic Search Engine for Memory Store (Refactored)
    
    New Architecture:
    - Focuses exclusively on anchor column retrieval
    - Independent term retrieval
    - No table metadata or score roll-up
    - LLM-driven query analysis
    - Multi-facet column indexing
    """
    
    def __init__(self, config: Dict[str, Any], storage_path: Path):
        super().__init__(config, storage_path)
        # Config structure: config.memory = {'semantic': {'search': {...}}, 'episodic': {...}}
        # So we need to access config['semantic']['search']
        self.config = config.get('semantic', {}).get('search', {})
        
        # Initialize specialized retrievers
        self.column_retriever = AnchorColumnRetriever(self.config)
        self.term_retriever = IndependentTermRetriever(self.config)
        self.sql_preprocessor = SQLPreprocessor(
            case_sensitive=self.config.get('case_sensitive', False)
        )
        
        # Read retriever-specific top_k limits from config
        retriever_limits = self.config.get('retriever_limits', {})
        self.column_top_k_per_term = retriever_limits.get('column_per_term', 10)  # Default: 10
        self.term_top_k_per_term = retriever_limits.get('term_per_term', 3)      # Default: 3
        
        logger.debug(f"SemanticSearchEngine initialized with new architecture (column_top_k={self.column_top_k_per_term}, term_top_k={self.term_top_k_per_term})")
    
    def bind_database(self, database_id: str) -> None:
        """Bind to specific database"""
        if self.current_database_id == database_id and self.indexes_built:
            return
            
        self.current_database_id = database_id
        self.indexes_built = False
        
        logger.info(f"SemanticSearchEngine bound to database: {database_id}")
    
    @staticmethod
    def _normalize_term_field(value: Any) -> Optional[str]:
        """
        Normalize term document fields to strings for TermItem compatibility
        
        Handles:
        - Lists: Convert to string (join with newline)
        - NaN/None: Return None
        - Other types: Convert to string
        """
        if value is None:
            return None
        
        # Handle NaN (float)
        if isinstance(value, float) and math.isnan(value):
            return None
        
        # Handle list (e.g., definition field can be List[str])
        if isinstance(value, list):
            # Filter out None/empty values and join
            valid_items = [str(item) for item in value if item is not None and str(item).strip()]
            return '\n'.join(valid_items) if valid_items else None
        
        # Convert to string
        str_value = str(value)
        return str_value if str_value.strip() else None
    
    def search(self, query: MemoryQuery, dataframes: Dict[str, pd.DataFrame], return_per_term: bool = False, top_k: Optional[int] = None) -> Union[FlatMemoryResponse, Tuple[PerTermMemoryResponse, List[JoinRelationship]]]:
        """
        Main search interface - implements anchor column focused search
        
        Args:
            query: Memory query with natural language question
            dataframes: Current database metadata dataframes
            return_per_term: If True, return (PerTermMemoryResponse, List[JoinRelationship]) grouped by query term;
                           If False, return FlatMemoryResponse with all schemas and metadata
            
        Returns:
            - Tuple[PerTermMemoryResponse, List[JoinRelationship]] if return_per_term=True
              (PerTermMemoryResponse contains query_term_results, all_tables, all_schemas;
               List[JoinRelationship] contains global join relationships)
            - FlatMemoryResponse if return_per_term=False (flat lists of column_items, term_items, 
              all_tables, all_schemas, join_relationships)
        """
        
        # Ensure indexes are built
        self._ensure_indexes_built(dataframes)
        
        relevant_terms = self.term_retriever.retrieve_terms(
            query.query_content,
            top_k=self.term_top_k_per_term,  # Use configured value instead of hardcoded calculation
            intent_analysis=query.intent_analysis,
            return_per_term=return_per_term
        )
        print(relevant_terms)

        retrieved_column_results = self.column_retriever.retrieve_anchor_columns(
            query.query_content,            
            return_per_term=return_per_term,
            top_k=self.column_top_k_per_term,  # Use configured value instead of query.limit_per_term
            intent_analysis=query.intent_analysis
        )
        print(retrieved_column_results)
        
        # Extract schema from generated SQL
        generated_schema = self.sql_preprocessor.extract_sql_schema(query.generated_sql) if query.generated_sql else None
            
        # Handle return_per_term for columns
        if return_per_term:
            per_term_column_result, required_tables = retrieved_column_results
            
            # Step 1: Collect all schemas from three sources
            all_schemas = set()
            
            # 1.1 Schemas from column retriever
            column_schemas = per_term_column_result.merged_schemas
            all_schemas.update(column_schemas)
            logger.debug(f"Column retriever schemas: {len(column_schemas)}")
            
            # 1.2 Schemas from term retriever
            term_schemas = set()
            if isinstance(relevant_terms, PerQueryTermTermResult):
                term_schemas.update(relevant_terms.merged_schemas)
            all_schemas.update(term_schemas)
            logger.debug(f"Term retriever schemas: {len(term_schemas)}")
            
            # 1.3 Schemas from generated SQL
            sql_schemas = set()
            if generated_schema:
                # Extract table.column format from generated_schema
                for column_info in generated_schema.get('columns', []):
                    table_name = column_info.get('table')
                    column_name = column_info.get('column')
                    if table_name and column_name:
                        sql_schemas.add(f"{table_name}.{column_name}")
            all_schemas.update(sql_schemas)

            logger.debug(f"Generated SQL schemas: {len(sql_schemas)}")
            
            logger.info(f"Total unique schemas from all sources: {len(all_schemas)}")
            
            # Step 2: Extract all tables from merged schemas
            all_tables = set()
            for schema in all_schemas:
                if '.' in schema:
                    # Handle both "table.column" and "db.table.column" formats
                    # Use the same logic as _extract_required_tables_from_columns
                    parts = schema.split('.')
                    if len(parts) >= 2:
                        # For "table.column": parts[:-1] = ["table"]
                        # For "db.table.column": parts[:-1] = ["db", "table"]
                        table_name = '.'.join(parts[:-1])
                        all_tables.add(table_name)
            
            all_tables = list(filter(None, all_tables))
            logger.info(f"Total unique tables from all sources: {len(all_tables)}")
            
            # Step 3: Get join candidates for all merged tables
            join_candidates = self.column_retriever.get_join_candidates_for_tables(all_tables)
            logger.info(f"Detected {len(join_candidates)} join relationships for {len(all_tables)} merged tables")
            
            # Step 4: Build QueryTermResult list grouped by query term
            query_term_results = []
            
            # Get all unique query terms from both column and term retrievers
            all_query_terms = set()
            
            # Collect query terms from column retriever
            for column_selection in per_term_column_result.query_term_selections:
                all_query_terms.add(column_selection.query_term)
            
            # Collect query terms from term retriever
            if isinstance(relevant_terms, PerQueryTermTermResult):
                for term_selection in relevant_terms.query_term_selections:
                    all_query_terms.add(term_selection.query_term)
            
            # Process each query term
            for query_term in sorted(all_query_terms):
                column_items = []
                term_items = []
                
                # 4.1: Find column items for this query term
                column_selection = None
                for col_sel in per_term_column_result.query_term_selections:
                    if col_sel.query_term == query_term:
                        column_selection = col_sel
                        break
                
                # Create ColumnItem objects for this query term
                if column_selection:
                    for schema in column_selection.selected_schemas:
                        # Get column metadata
                        if '.' in schema:
                            parts = schema.split('.')
                            if len(parts) >= 2:
                                # Extract table and column names
                                table_name = '.'.join(parts[:-1])
                                column_name = parts[-1]
                                
                                try:
                                    column_meta = self.column_retriever.column_provider.get_column_metadata(
                                        table_name, column_name
                                    )
                                    
                                    # Find value match info for this schema
                                    value_match_info = None
                                    for schema_with_values in column_selection.selected_schemas_with_values:
                                        if schema_with_values.schema.lower() == schema.lower():
                                            value_match_info = {
                                                'matched_values': schema_with_values.matched_values,
                                                'match_types': schema_with_values.match_types,
                                                'encoding_mappings': schema_with_values.encoding_mappings
                                            }
                                            break
                                    
                                    column_item = ColumnItem(
                                        schema_ref=schema,
                                        table_name=table_name,
                                        column_name=column_name,
                                        column_metadata=column_meta or {},
                                        value_matches=value_match_info
                                    )
                                    column_items.append(column_item)
                                except Exception as e:
                                    logger.warning(f"Failed to get metadata for column {schema}: {e}")
                
                # 4.2: Find term items for this query term
                if isinstance(relevant_terms, PerQueryTermTermResult):
                    for term_selection in relevant_terms.query_term_selections:
                        if term_selection.query_term == query_term:
                            # Create TermItem objects for this query term
                            for term_result in term_selection.selected_terms:
                                term_item = TermItem(
                                    term_id=term_result.term_document.term_id,
                                    term_name=term_result.term_document.term_name,
                                    definition=self._normalize_term_field(term_result.term_document.definition),
                                    formula=self._normalize_term_field(term_result.term_document.formula),
                                    example_usage=self._normalize_term_field(term_result.term_document.example_usage),
                                    related_tables=term_result.term_document.related_tables,
                                    related_columns=term_result.term_document.related_columns,
                                    schemas=term_result.term_document.schemas,
                                    signal_scores=term_result.signal_scores,
                                    explanation=term_result.explanation,
                                    score=term_result.final_score
                                )
                                term_items.append(term_item)
                            break
                
                # Create QueryTermResult for this query term
                query_term_result = QueryTermResult(
                    query_term=query_term,
                    column_items=column_items,
                    term_items=term_items
                )
                query_term_results.append(query_term_result)
            
            # Step 5: Build JoinRelationship list (separate from PerTermMemoryResponse)
            join_relationships = [
                JoinRelationship(
                    table1=jc.table1,
                    column1=jc.column1,
                    table2=jc.table2,
                    column2=jc.column2,
                    join_type=jc.join_type,
                    confidence=jc.confidence
                )
                for jc in join_candidates
            ]
            
            # Step 6: Build and return PerTermMemoryResponse and join_relationships separately
            total_column_items = sum(len(qtr.column_items) for qtr in query_term_results)
            total_term_items = sum(len(qtr.term_items) for qtr in query_term_results)
            
            logger.info(f"Search completed (per_term=True): {len(all_schemas)} schemas, "
                       f"{len(all_tables)} tables, {len(join_relationships)} joins, "
                       f"{total_column_items} column items, {total_term_items} term items")
            
            per_term_response = PerTermMemoryResponse(
                query_term_results=query_term_results,
                all_tables=all_tables,
                all_schemas=sorted(list(all_schemas))
            )
            
            return per_term_response, join_relationships

        else:
            # return_per_term=False: Return flat list of all schemas and metadata
            # This mirrors the logic from return_per_term=True but without query term grouping
            
            selected_columns, required_tables, value_matches = retrieved_column_results
            
            # Step 1: Collect all schemas from three sources (same as return_per_term=True)
            all_schemas = set()
            
            # 1.1 Schemas from column retriever (merged schemas from all query terms)
            column_schemas = set(selected_columns)
            all_schemas.update(column_schemas)
            logger.debug(f"Column retriever schemas: {len(column_schemas)}")
            
            # 1.2 Schemas from term retriever (merged results)
            term_schemas = set()
            if isinstance(relevant_terms, list):
                for term_result in relevant_terms:
                    if term_result.term_document.schemas:
                        term_schemas.update(term_result.term_document.schemas)
            all_schemas.update(term_schemas)
            logger.debug(f"Term retriever schemas: {len(term_schemas)}")
            
            # 1.3 Schemas from generated SQL
            sql_schemas = set()
            if generated_schema:
                for column_info in generated_schema.get('columns', []):
                    table_name = column_info.get('table')
                    column_name = column_info.get('column')
                    if table_name and column_name:
                        sql_schemas.add(f"{table_name}.{column_name}")
            all_schemas.update(sql_schemas)
            logger.debug(f"Generated SQL schemas: {len(sql_schemas)}")
            
            logger.info(f"Total unique schemas from all sources: {len(all_schemas)}")
            
            # Step 2: Extract all tables from merged schemas
            all_tables = set()
            for schema in all_schemas:
                if '.' in schema:
                    parts = schema.split('.')
                    if len(parts) >= 2:
                        table_name = '.'.join(parts[:-1])
                        all_tables.add(table_name)
            
            all_tables = list(filter(None, all_tables))
            logger.info(f"Total unique tables from all sources: {len(all_tables)}")
            
            # Step 3: Get join candidates for all merged tables
            join_candidates = self.column_retriever.get_join_candidates_for_tables(all_tables)
            logger.info(f"Detected {len(join_candidates)} join relationships for {len(all_tables)} merged tables")
            
            # Step 4: Build flat list of ColumnItem objects (not grouped by term)
            column_items = []
            for schema in sorted(all_schemas):
                if '.' in schema:
                    parts = schema.split('.')
                    if len(parts) >= 2:
                        table_name = '.'.join(parts[:-1])
                        column_name = parts[-1]
                        
                        try:
                            column_meta = self.column_retriever.column_provider.get_column_metadata(
                                table_name, column_name
                            )
                            
                            # Find value match info for this schema
                            value_match_info = None
                            if value_matches:
                                # Normalize schema for matching
                                normalized_schema = self._normalize_column_key(schema)
                                for vm_key, vm_data in value_matches.items():
                                    normalized_vm_key = self._normalize_column_key(vm_key)
                                    if normalized_vm_key == normalized_schema:
                                        value_match_info = {
                                            'matched_values': vm_data.get('matched_values', []),
                                            'match_types': vm_data.get('match_types', []),
                                            'encoding_mappings': vm_data.get('encoding_mappings', {})
                                        }
                                        break
                            
                            # Filter encoding_mapping if we have value matches
                            if column_meta and value_match_info and value_match_info.get('encoding_mappings'):
                                column_meta = column_meta.copy() if column_meta else {}
                                column_meta['encoding_mapping'] = value_match_info['encoding_mappings']
                            
                            column_item = ColumnItem(
                                schema_ref=schema,
                                table_name=table_name,
                                column_name=column_name,
                                column_metadata=column_meta or {},
                                value_matches=value_match_info
                            )
                            column_items.append(column_item)
                        except Exception as e:
                            logger.warning(f"Failed to get metadata for column {schema}: {e}")
            
            # Step 5: Build flat list of TermItem objects (not grouped by term)
            term_items = []
            if isinstance(relevant_terms, list):
                for term_result in relevant_terms:
                    term_item = TermItem(
                        term_id=term_result.term_document.term_id,
                        term_name=term_result.term_document.term_name,
                        definition=self._normalize_term_field(term_result.term_document.definition),
                        formula=self._normalize_term_field(term_result.term_document.formula),
                        example_usage=self._normalize_term_field(term_result.term_document.example_usage),
                        related_tables=term_result.term_document.related_tables,
                        related_columns=term_result.term_document.related_columns,
                        schemas=term_result.term_document.schemas,
                        signal_scores=term_result.signal_scores,
                        explanation=term_result.explanation,
                        score=term_result.final_score
                    )
                    term_items.append(term_item)
            
            # Step 6: Build JoinRelationship list
            join_relationships = [
                JoinRelationship(
                    table1=jc.table1,
                    column1=jc.column1,
                    table2=jc.table2,
                    column2=jc.column2,
                    join_type=jc.join_type,
                    confidence=jc.confidence
                )
                for jc in join_candidates
            ]
            
            logger.info(f"Search completed (per_term=False): {len(all_schemas)} schemas, "
                       f"{len(all_tables)} tables, {len(join_relationships)} joins, "
                       f"{len(column_items)} column items, {len(term_items)} term items")
            
            # Step 7: Return FlatMemoryResponse
            return FlatMemoryResponse(
                column_items=column_items,
                term_items=term_items,
                all_tables=all_tables,
                all_schemas=sorted(list(all_schemas)),
                join_relationships=join_relationships,
                query_time_ms=0  # Will be set by caller
            )
        
    def _ensure_indexes_built(self, dataframes: Dict[str, pd.DataFrame]) -> None:
        """Ensure search indexes are built for new architecture"""
        if self.indexes_built:
            return
            
        logger.info("Building anchor column and term indexes...")
        
        # Initialize clients
        llm_client = create_llm_client(self._get_llm_config())
        embedding_client = create_embedding_client(self._get_embedding_config())
        
        # Initialize retrievers with clients
        self.column_retriever.initialize(llm_client, embedding_client)
        self.term_retriever.initialize(llm_client, embedding_client)
        
        # Build indexes
        self.column_retriever.build_indexes(self.current_database_id, dataframes)
        self.term_retriever.build_indexes(self.current_database_id, dataframes)
        
        self.indexes_built = True
        logger.info("New architecture indexes built successfully")
    

    
    def _merge_schema_metadata(
        self,
        selected_columns: Optional[List[str]],
        required_tables: Optional[List[str]],
        generated_schema: Optional[Dict[str, Any]],
        dataframes: Dict[str, pd.DataFrame],
        base_tables_metadata: Optional[Dict[str, Any]] = None,
        base_columns_metadata: Optional[Dict[str, Any]] = None,
        value_matches: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Merge schema information from anchor selection with schema parsed from the generated SQL
        and enrich it with semantic memory metadata.
        
        Args:
            value_matches: Dict mapping column keys to matched values and encoding_mappings.
                          Used to filter encoding_mapping to only include matched entries.
        """
        combined_tables = dict(base_tables_metadata or {})
        combined_columns = dict(base_columns_metadata or {})
        
        # Normalize value_matches keys for lookup (handle case-insensitive matching and format variations)
        value_matches_lookup = {}
        if value_matches:
            for key, match_data in value_matches.items():
                # Normalize key: remove database_id prefix if present, then lowercase
                normalized_key = self._normalize_column_key(key)
                value_matches_lookup[normalized_key] = match_data
        
        anchor_tables = set(filter(None, required_tables or []))
        anchor_columns = set(filter(None, selected_columns or []))
        
        sql_tables = set()
        sql_columns = set()
        if generated_schema:
            sql_tables.update(filter(None, generated_schema.get('tables', [])))
            for column in generated_schema.get('columns', []):
                table_name = column.get('table')
                column_name = column.get('column')
                if table_name and column_name:
                    sql_columns.add(f"{table_name}.{column_name}")
        
        table_lookup = self._build_table_lookup(dataframes)
        column_lookup = self._build_column_lookup(dataframes)
        
        resolved_tables = set()
        for table in anchor_tables.union(sql_tables):
            resolved_table = self._resolve_table_name(table, table_lookup)
            if resolved_table:
                resolved_tables.add(resolved_table)
        
        resolved_columns = set()
        for column_ref in anchor_columns.union(sql_columns):
            if '.' not in column_ref:
                continue
            table_name, column_name = column_ref.split('.', 1)
            resolved_table = self._resolve_table_name(table_name, table_lookup)
            actual_table, actual_column = self._resolve_column_identifier(
                resolved_table or table_name,
                column_name,
                column_lookup
            )
            if actual_table and actual_column:
                resolved_columns.add((actual_table, actual_column))
        
        # Enrich tables metadata
        for table_name in resolved_tables:
            if table_name not in combined_tables:
                try:
                    table_meta = self.column_retriever.column_provider.get_table_metadata(table_name)
                    if table_meta:
                        combined_tables[table_name] = table_meta
                except Exception as exc:
                    logger.warning(f"Failed to fetch table metadata for {table_name}: {exc}")
        
        # Enrich columns metadata
        for table_name, column_name in resolved_columns:
            column_key = f"{table_name}.{column_name}"
            if column_key not in combined_columns:
                try:
                    column_meta = self.column_retriever.column_provider.get_column_metadata(table_name, column_name)
                    if column_meta:
                        combined_columns[column_key] = column_meta
                except Exception as exc:
                    logger.warning(f"Failed to fetch column metadata for {column_key}: {exc}")
        
        # Filter encoding_mapping for all columns based on value_matches
        # This ensures we only return matched encoding_mapping entries, not the entire dictionary
        for column_key in list(combined_columns.keys()):
            normalized_key = self._normalize_column_key(column_key)
            if normalized_key in value_matches_lookup:
                match_data = value_matches_lookup[normalized_key]
                encoding_mappings = match_data.get('encoding_mappings', {})
                
                # If we have matched encoding_mappings, replace the full encoding_mapping
                if encoding_mappings:
                    # Create a copy to avoid modifying the original
                    column_meta = combined_columns[column_key].copy()
                    # Replace encoding_mapping with only the matched entries
                    column_meta['encoding_mapping'] = encoding_mappings
                    combined_columns[column_key] = column_meta
                    logger.debug(f"Filtered encoding_mapping for {column_key}: {len(encoding_mappings)} matched entries")
        
        schema_components = generated_schema or {}
        return {
            'tables': combined_tables,
            'columns': combined_columns,
            'sql_conditions': schema_components.get('conditions', []),
            'sql_joins': schema_components.get('joins', []),
            'sources': {
                'anchor_tables': sorted(anchor_tables),
                'anchor_columns': sorted(anchor_columns),
                'generated_sql_tables': sorted(sql_tables),
                'generated_sql_columns': sorted(sql_columns)
            }
        }
    
    def _build_table_lookup(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Build case-insensitive lookup for table names"""
        lookup: Dict[str, str] = {}
        table_df = dataframes.get('table')
        if table_df is None or table_df.empty or 'table_name' not in table_df.columns:
            return lookup
        for table_name in table_df['table_name'].dropna().unique():
            name_str = str(table_name)
            lookup[name_str.lower()] = name_str
        return lookup
    
    def _build_column_lookup(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[Tuple[str, str], Tuple[str, str]]:
        """Build case-insensitive lookup for (table, column) combinations"""
        lookup: Dict[tuple, tuple] = {}
        column_df = dataframes.get('column')
        if column_df is None or column_df.empty:
            return lookup
        required_columns = {'table_name', 'column_name'}
        if not required_columns.issubset(column_df.columns):
            return lookup
        for _, row in column_df.iterrows():
            table_name = row.get('table_name')
            column_name = row.get('column_name')
            if not table_name or not column_name:
                continue
            key = (str(table_name).lower(), str(column_name).lower())
            lookup[key] = (str(table_name), str(column_name))
        return lookup
    
    def _resolve_table_name(self, table_name: Optional[str], table_lookup: Dict[str, str]) -> Optional[str]:
        """Resolve table name using case-insensitive lookup"""
        if not table_name:
            return None
        normalized = table_name.lower()
        return table_lookup.get(normalized, table_name)
    
    def _resolve_column_identifier(
        self,
        table_name: Optional[str],
        column_name: Optional[str],
        column_lookup: Dict[Tuple[str, str], Tuple[str, str]]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Resolve (table, column) pair using case-insensitive lookup"""
        if not table_name or not column_name:
            return None, None
        key = (table_name.lower(), column_name.lower())
        return column_lookup.get(key, (table_name, column_name))
    
    def _normalize_column_key(self, column_key: str) -> str:
        """
        Normalize column key for matching (handles database_id.table.column and table.column formats)
        
        Args:
            column_key: Column key in various formats (e.g., "database.table.column", "table.column")
            
        Returns:
            Normalized key in lowercase table.column format
        """
        if not column_key:
            return ""
        
        # Split by dots
        parts = column_key.split('.')
        if len(parts) >= 3:
            # database_id.table.column format - take last two parts
            return f"{parts[-2]}.{parts[-1]}".lower()
        elif len(parts) == 2:
            # table.column format - return as is, lowercase
            return column_key.lower()
        else:
            # Single part or empty - return as is, lowercase
            return column_key.lower()
    
    def _get_embedding_config(self) -> EmbeddingConfig:
        """Get embedding configuration"""
        from caf.config.paths import PathConfig
        
        embedding_cfg = self.config.get('embedding', {})
        return EmbeddingConfig(
            provider=embedding_cfg.get('provider', 'sentence_transformers'),
            model_name=embedding_cfg.get('model_name', 'all-MiniLM-L6-v2'),
            model_path=embedding_cfg.get('model_path'),  # Support local model path
            device=embedding_cfg.get('device', 'cpu'),
            batch_size=embedding_cfg.get('batch_size', 32),
            normalize_embeddings=embedding_cfg.get('normalize_embeddings', True),
            cache_dir=embedding_cfg.get('cache_dir') or str(PathConfig.get_model_cache_path())
        )
    
    def _get_llm_config(self) -> LLMConfig:
        """Get LLM configuration using global config manager"""
        try:
            # Try to get from global configuration first
            global_llm_config = get_llm_config('openai')
            return LLMConfig(
                provider=global_llm_config.provider,
                model_name=global_llm_config.model_name,
                api_key=global_llm_config.api_key,
                base_url=global_llm_config.base_url,
                temperature=global_llm_config.temperature,
                max_tokens=global_llm_config.max_tokens
            )
        except Exception as e:
            logger.warning(f"Failed to get global LLM config: {e}, falling back to local config")
            # Fallback to local configuration
            llm_cfg = self.config.get('llm', {})
            return LLMConfig(
                provider=llm_cfg.get('provider', 'openai'),
                model_name=llm_cfg.get('model_name', 'gpt-4o-mini'),
                api_key=llm_cfg.get('api_key'),
                base_url=llm_cfg.get('base_url'),
                temperature=llm_cfg.get('temperature', 0.1),
                max_tokens=llm_cfg.get('max_tokens', 4000)
            )
    
    def rebuild_indexes(self, dataframes: Dict[str, pd.DataFrame]) -> None:
        """Manually rebuild search indexes for new architecture"""
        logger.info("Manually rebuilding search indexes...")
        self.indexes_built = False
        
        # Force rebuild
        self._ensure_indexes_built(dataframes)
        logger.info("Search indexes rebuilt successfully")
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get information about current indexes"""
        if not self.indexes_built:
            return {'status': 'not_built'}
        
        info = {
            'status': 'built',
            'database_id': self.current_database_id,
            'anchor_columns': len(self.column_retriever.column_documents),
            'terms': len(self.term_retriever.term_documents)
        }
        
        # Add retriever-specific info
        if hasattr(self.column_retriever, 'bm25_indexes'):
            info['column_facet_indexes'] = list(self.column_retriever.bm25_indexes.keys())
        
        if hasattr(self.term_retriever, 'bm25_index') and self.term_retriever.bm25_index:
            info['term_index_ready'] = True
        
        return info
    
    def _apply_value_match_filtering_to_metadata(
        self, 
        columns_metadata: Dict[str, Any], 
        value_matches: Optional[Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Apply value match filtering to columns_metadata to ensure only matched encoding_mapping 
        entries are returned, preventing duplicate encoding_mapping data in the final result.
        
        Args:
            columns_metadata: Original column metadata with full encoding_mapping
            value_matches: Dict mapping column keys to matched values and encoding_mappings
            
        Returns:
            Filtered columns_metadata with only matched encoding_mapping entries
        """
        if not value_matches:
            return columns_metadata
        
        # Create a copy to avoid modifying the original
        filtered_metadata = {}
        
        # Normalize value_matches keys for lookup
        value_matches_lookup = {}
        for key, match_data in value_matches.items():
            normalized_key = self._normalize_column_key(key)
            value_matches_lookup[normalized_key] = match_data
        
        for column_key, column_meta in columns_metadata.items():
            # Create a copy of the column metadata
            filtered_column_meta = column_meta.copy() if column_meta else {}
            
            # Check if this column has value matches
            normalized_key = self._normalize_column_key(column_key)
            if normalized_key in value_matches_lookup:
                match_data = value_matches_lookup[normalized_key]
                encoding_mappings = match_data.get('encoding_mappings', {})
                
                # If we have matched encoding_mappings, replace the full encoding_mapping
                if encoding_mappings:
                    filtered_column_meta['encoding_mapping'] = encoding_mappings
                    logger.debug(f"Applied value match filtering for {column_key}: {len(encoding_mappings)} matched entries")
            
            filtered_metadata[column_key] = filtered_column_meta
        
        return filtered_metadata
    