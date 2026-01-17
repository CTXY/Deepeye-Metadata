# Anchor Column Retriever - Specialized retriever for finding anchor columns

import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..indexing.column_indexer import ColumnFacetProvider
from ..indexing.value_indexer import ValueIndexBuilder, ColumnValueIndex
from ..matchers.value_matcher import ValueMatcher, ValueMatchResult
from ..matchers.bm25_matcher import BM25Matcher, BM25MatchResult
from ..matchers.splade_matcher import SPLADEMatcher, SPLADEMatchResult
from ..matchers.vector_matcher import VectorMatcher, VectorMatchResult
from caf.memory.search.embedding.client import BaseEmbeddingClient
from caf.llm.client import BaseLLMClient
from caf.config.paths import PathConfig
from caf.utils.query_analyzer import QueryAnalyzer, IntentAnalysis, EntityGroup

        

logger = logging.getLogger(__name__)


@dataclass
class SlotCandidate:
    """Candidate result for a specific slot"""
    slot_type: str  # 'primary_entity', 'select_target', 'condition_field', 'condition_value'
    original_phrase: str
    candidate_metadata: str  # table_name or table_name.column_name
    score: float
    matched_value: Optional[str] = None
    match_type: Optional[str] = None
    encoding_mapping_info: Optional[Dict[str, Any]] = None

@dataclass
class JoinCandidate:
    """JOIN relationship candidate"""
    table1: str
    column1: str
    table2: str
    column2: str
    join_type: str  # 'foreign_key', 'semantic_similarity', 'value_match'
    confidence: float = 1.0

@dataclass
class SlotAggregationResult:
    slot_name: str
    slot_type: str  # 'primary_entity', 'select_target', 'condition_field', etc.
    original_phrases: List[str]
    has_candidates: bool  # 是否有候选schema
    selected_candidates: List[str]  # LLM选择的合适候选schema (table.column格式)
    reasoning: Optional[str] = None  # LLM的判断reasoning

@dataclass 
class SchemaAggregationResult:
    slot_results: List[SlotAggregationResult]  # 每个槽位的评估结果
    slots_with_candidates: List[str]  # 有候选者的槽位
    slots_without_candidates: List[str]  # 没有候选者的槽位

@dataclass
class SchemaWithValueMatches:
    """Schema及其对应的value matches"""
    schema: str  # table.column格式
    matched_values: List[str]  # 该schema匹配的值列表
    match_types: List[str]  # 匹配类型列表（如 'exact', 'fuzzy'等）
    encoding_mappings: Dict[str, Any]  # encoding mapping信息（如果有）

@dataclass
class QueryTermSchemaSelection:
    """每个检索词的schema选择结果"""
    query_term: str  # 检索词（entity或constraint）
    selected_schemas: List[str]  # LLM选择的top_k个schema (table.column格式)
    selected_schemas_with_values: List[SchemaWithValueMatches]  # 带value match信息的schemas
    all_candidates_count: int = 0  # 总候选数量

@dataclass
class PerQueryTermSelectionResult:
    """按检索词分组的schema选择结果"""
    query_term_selections: List[QueryTermSchemaSelection]  # 每个检索词的选择结果
    merged_schemas: List[str]  # 合并并去重后的所有schema



class AnchorColumnRetriever:
    """
    Anchor Column Retriever
    
    Specialized retriever focused on finding anchor columns that are most relevant
    to the user query. Uses multi-facet column indexing and hybrid retrieval.
    
    Core features:
    - Multi-facet column indexing (names, description, values, technical)
    - Hybrid retrieval (keyword + semantic + value matching)
    - LLM-driven query analysis
    - Unified scoring without table roll-up
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('anchor_column', {})
        
        # Scoring weights
        self.weights = self.config.get('weights', {
            'keyword': 0.3,
            'semantic': 0.4, 
            'value': 0.3
        })
        
        # Parameters
        self.candidate_limit = self.config.get('candidate_limit', 20)
        
        # OPTIMIZATION NOTE: Enhanced candidate filtering to reduce excessive candidates
        # - Updated RRF fusion with stricter thresholds (min_score: 0.001→0.05, max_count: 20→15)
        # - Slot-specific filtering: entity(12/0.08), aggregation(10/0.06), condition(15/0.04), value(10/0.07)
        
        # Components
        self.column_provider = ColumnFacetProvider(self.config)
        self.value_index_builder = ValueIndexBuilder(self.config.get("value_index", {}))

        # Legacy: Removed _slot_candidates as it's no longer used in EntityGroup approach
        
        # Initialize matchers
        self.value_matcher = ValueMatcher(
            cardinality_threshold=self.config.get('cardinality_threshold', 100000)
        )
        # Use SPLADE for more advanced sparse retrieval (optional due to CUDA issues)
        try:
            # Use unified cache path for SPLADE
            splade_cache_path = PathConfig.get_splade_cache_path()
            self.splade_matcher = SPLADEMatcher(
                facet_weights=self.config.get('facet_weights', {
                    'names': 3.0,
                    'description': 1.0
                }),
                model_name=self.config.get('splade_model', '/home/yangchenyu/pre-trained-models/splade-v3'),
                cache_path=splade_cache_path
            )
            self.splade_available = True
            logger.info("SPLADE matcher initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize SPLADE matcher: {e}")
            self.splade_matcher = None
            self.splade_available = False
        
        # Vector matcher with unified cache path
        vector_facets = self.config.get('vector_facets', ['description'])
        self.vector_matcher = VectorMatcher(
            cache_path=None,  # Use default unified cache path
            facets=vector_facets
        )
        
        # Data - removed: self.column_documents
        
        # Metadata dataframes storage
        self.dataframes: Optional[Dict[str, pd.DataFrame]] = None
        # Detected join relationships for current query lifecycle
        self.join_relationships: List[JoinCandidate] = []
        
        # Clients (will be set by engine)
        self.embedding_client: Optional[BaseEmbeddingClient] = None
        self.llm_client: Optional[BaseLLMClient] = None
        
        self.query_analyzer: Optional[QueryAnalyzer] = None
        
    
    def initialize(self, llm_client: BaseLLMClient, embedding_client: BaseEmbeddingClient):
        """Initialize with required clients"""
        self.llm_client = llm_client
        self.embedding_client = embedding_client
    
    def initialize_query_analyzer(self):
        # Initialize QueryAnalyzer with LLM client
        query_analyzer_config = {
            'dependency_parser_lang': 'en',
            'dependency_parser_use_gpu': False,
            'dependency_parser_model': self.config.get('dependency_parser_model', 'gpt-4o-mini'),
            'dependency_parser_temperature': self.config.get('dependency_parser_temperature', 0.1),
            'dependency_parser_max_tokens': self.config.get('dependency_parser_max_tokens', 2000)
        }
        
        self.query_analyzer = QueryAnalyzer(
            llm_client=self.llm_client,
            config=query_analyzer_config
        )
        
        logger.debug("QueryAnalyzer initialized")
        
    def build_indexes(self, database_id: str, dataframes: Dict[str, pd.DataFrame]) -> None:
        """Build all indexes for anchor column retrieval"""
        logger.info("Building anchor column indexes (provider-backed)...")
        logger.debug(f"Building indexes for database_id: {database_id}")
        logger.debug(f"Number of dataframes: {len(dataframes)}")
        for table_name, df in dataframes.items():
            logger.debug(f"Table '{table_name}': {len(df)} rows, {len(df.columns)} columns")
        
        # Store dataframes for future usage
        self.dataframes = dataframes
        self.column_provider.attach(database_id, dataframes)
        
        if not self.embedding_client:
            logger.warning("No embedding client available, skipping vector index build")
        
        # Build unified SPLADE indexes (if available)
        if self.splade_available:
            logger.debug("Building SPLADE indexes...")
            self.splade_matcher.build_indexes(self.column_provider)
            logger.debug("SPLADE indexes built")
        else:
            logger.debug("Skipping SPLADE index build - not available")
        
        # Build vector indexes using VectorMatcher
        if self.embedding_client:
            logger.debug("Building vector indexes...")
            self.vector_matcher.build_indexes(
                provider=self.column_provider,
                embedding_client=self.embedding_client,
                database_id=database_id
            )
            logger.debug("Vector indexes built")
        else:
            logger.debug("Skipping vector index build - no embedding client")
        
        # Build / load value index from real database and then unified value matching indexes
        logger.debug("Building column value index (distinct values from database)...")
        value_index: ColumnValueIndex = self.value_index_builder.build_or_load(database_id, self.column_provider)
        logger.debug("Column value index built with %d columns", len(value_index.values_by_column))

        logger.debug("Building value matching indexes...")
        self.value_matcher.build_indexes(self.column_provider, value_index=value_index)
        logger.debug("Value matching indexes built")
        
        logger.info("Anchor column indexes built (provider-backed)")
        logger.debug("All indexes built successfully")
    
    def _get_table_metadata(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get table metadata from dataframes"""
        if not self.dataframes:
            return None
        meta = self.column_provider.get_table_metadata(table_name)
        return meta or None
    
    def retrieve_anchor_columns(
        self, 
        user_question: str, 
        top_k: int = None,
        intent_analysis: Optional['IntentAnalysis'] = None,
        return_per_term: bool = False
    ):
        """
        Retrieve anchor columns using hybrid approach with LLM-based selection per query term
        
        Args:
            user_question: 用户问题
            top_k: 每个检索词选择的schema数量
            intent_analysis: 可选的意图分析结果（如果已存在）
            return_per_term: 如果为True，返回按检索词分组的结果；如果为False，返回合并去重的结果
        
        Returns:
            如果return_per_term=False:
                (selected_columns, required_tables, value_matches_selected)
            如果return_per_term=True:
                per_term_result
                其中per_term_result是PerQueryTermSelectionResult对象
        """
        if not self.llm_client:
            raise ValueError("AnchorColumnRetriever not properly initialized")
        
        logger.debug(f"Retrieving anchor columns for: {user_question[:100]}...")
        
        # Stage 1: LLM-driven query analysis
        if intent_analysis is None:
            intent_analysis = self._analyze_query(user_question)
        
        # Stage 2: Unified multi-signal retrieval for entity groups
        extended_candidates = self._multi_signal_retrieval(intent_analysis)
        
        # Stage 3: 使用LLM为每个检索词选择top_k个schema
        per_term_result = self._select_schemas_per_query_term_with_llm(
            user_question=user_question,
            extended_candidates=extended_candidates,
            top_k=top_k
        )

        selected_columns = per_term_result.merged_schemas
            
        required_tables = self._extract_required_tables_from_columns(selected_columns)

        
        if return_per_term:
            # 返回方式(b): 按检索词分组的结果

            return per_term_result, required_tables
        else:
            # 临时保留这种处理方式，后续需要进行优化
            # Aggregate matched values for candidates
            value_matches_all = self._aggregate_value_matches(extended_candidates)

            # 处理value_matches_selected
            value_matches_selected = self._process_value_matches_for_selected_schemas(
                selected_columns, value_matches_all
            )
            
            logger.info(f"Merged selection: {len(selected_columns)} unique schemas from "
                       f"{len(per_term_result.query_term_selections)} query terms")
            
            # print("******************Value Matches*******************")
            # print(value_matches_all)
            # print("*************************************************")
            
            # print("******************Retrieved Results from AnchorColumnRetriever  *******************")
            # print(selected_columns)
            # print(required_tables)
            # print("***********************************************************************************")
            
            return selected_columns, required_tables, value_matches_selected

  
    def _multi_signal_retrieval(self, intent_analysis: 'IntentAnalysis') -> Dict[str, List[SlotCandidate]]:
        """Unified multi-signal retrieval for entity groups using three retrieval methods"""
        extended_candidates = {}  # phrase -> List[SlotCandidate]
        
        logger.info(f"Starting unified retrieval for {len(intent_analysis.entity_groups)} entity groups")
        
        # Process each entity group
        for group_idx, entity_group in enumerate(intent_analysis.entity_groups):
            logger.info(f"Processing entity group {group_idx}: {entity_group.full_phrase}")
            
            # 1. Retrieve candidates for base_phrase (using base_entity property for backward compatibility)
            base_phrase = entity_group.base_phrase
            base_entity_candidates = self._retrieve_unified_candidates(
                phrase=base_phrase,
                original_phrase=entity_group.full_phrase,
                slot_type="entity_base"
            )
            extended_candidates[base_phrase] = base_entity_candidates
            
            logger.info(f'Base Entity "{base_phrase}" candidates: {len(base_entity_candidates)}')
            
            # 2. Retrieve candidates for each constraint
            for constraint_phrase in entity_group.constraints:
                if not constraint_phrase:
                    continue
                    
                constraint_candidates = self._retrieve_unified_candidates(
                    phrase=constraint_phrase,
                    original_phrase=entity_group.full_phrase,
                    slot_type="entity_filter"
                )
                extended_candidates[constraint_phrase] = constraint_candidates
                
                logger.info(f'Constraint "{constraint_phrase}" candidates: {len(constraint_candidates)}')
        
        return extended_candidates
    
    def _retrieve_unified_candidates(
        self, 
        phrase: str, 
        original_phrase: str, 
        slot_type: str
    ) -> List[SlotCandidate]:
        """
        Unified retrieval method using semantic/keywords/value approaches
        Returns top-20 candidates maximum per phrase
        """
        if not phrase.strip():
            logger.debug("Empty phrase provided for retrieval")
            return []
        
        logger.debug(f"Retrieving candidates for phrase '{phrase}' (type: {slot_type})")
        
        # Collect results from all three retrieval methods for RRF fusion
        all_retrieval_results = []  # List of (method_name, candidates_list)
        
        # 1. Keywords matching (SPLADE sparse retrieval)
        try:
            keyword_candidates = self._match_entity_keywords(phrase, slot_type, top_k=20)
            logger.debug(f"Keywords matching returned {len(keyword_candidates)} candidates")
            if keyword_candidates:
                all_retrieval_results.append(('keyword', keyword_candidates))
        except Exception as e:
            logger.debug(f"Keywords matching failed: {e}")
        
        # 2. Semantic matching (vector similarity)
        try:
            semantic_candidates = self._match_entity_semantic(phrase, slot_type, top_k=20)
            logger.debug(f"Semantic matching returned {len(semantic_candidates)} candidates")
            if semantic_candidates:
                all_retrieval_results.append(('semantic', semantic_candidates))
        except Exception as e:
            logger.debug(f"Semantic matching failed: {e}")
        
        # 3. Value matching with expansion (exact/fuzzy value matching in column data)
        try:
            # For entity_filter type, perform value expansion
            if slot_type == "entity_filter":
                expanded_values = self._extract_and_expand_values(phrase)
                logger.debug(f"Expanded values for filter '{phrase}': {expanded_values}")
            else:
                expanded_values = [phrase]  # No expansion for non-filter slots
            
            # Collect value candidates from all expanded values
            all_value_candidates = []
            for expanded_value in expanded_values:
                value_candidates = self._value_search(
                    phrase=expanded_value,
                    slot_type=slot_type,
                    original_phrase=original_phrase,
                    base_weight=1.0
                )
                all_value_candidates.extend(value_candidates)
            
                # print(f"******************Value Candidates for {expanded_value}*******************")
                # print(value_candidates)
                # print("***************************************************************************")
            
            # Remove duplicates based on candidate_metadata
            unique_value_candidates = self._deduplicate_candidates(all_value_candidates)
            
            if unique_value_candidates:
                all_retrieval_results.append(('value', unique_value_candidates))
        except Exception as e:
            logger.debug(f"Value matching failed: {e}")
        
        
        if not all_retrieval_results:
            logger.debug(f"No retrieval results for phrase '{phrase}', returning empty list")
            return []
        
        # Apply Reciprocal Rank Fusion (RRF) to combine results
        candidates = self._apply_rrf_fusion(
            all_retrieval_results, 
            original_phrase, 
            slot_type,
            max_candidates=20,  # Top-20 limit per phrase as requested
            min_rrf_score=0.001  # Lower threshold to allow more candidates
        )
        
        return candidates
    
    def _apply_rrf_fusion(self, retrieval_results: List[Tuple[str, List[SlotCandidate]]], 
                         original_phrase: str, slot_type: str, k: int = 60, 
                         max_candidates: int = 25, min_rrf_score: float = 0.001) -> List[SlotCandidate]:
        """
        Apply Reciprocal Rank Fusion (RRF) to combine results from multiple retrieval methods
        
        Args:
            retrieval_results: List of (method_name, candidates_list) tuples
            original_phrase: Original query phrase
            slot_type: Type of slot being processed
            k: RRF parameter (typically 60)
            max_candidates: Maximum number of candidates to return
            min_rrf_score: Minimum RRF score threshold for filtering
            
        Returns:
            List of SlotCandidate objects sorted by RRF score, filtered and limited
        """
        if not retrieval_results:
            logger.debug("No retrieval results provided for RRF fusion")
            return []
        
        # Dictionary to accumulate RRF scores: candidate_metadata -> (rrf_score, best_candidate, all_candidates)
        rrf_scores = {}
        
        # Process each retrieval method's results
        for method_name, candidates in retrieval_results:
            logger.debug(f"Processing method '{method_name}' with {len(candidates)} candidates")
            if not candidates:
                continue
            
            # Sort candidates by their original scores (highest first)
            sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
            
            # Calculate RRF contribution for each candidate
            for rank, candidate in enumerate(sorted_candidates, start=1):
                candidate_key = candidate.candidate_metadata
                
                # RRF formula: 1 / (k + rank)
                rrf_contribution = 1.0 / (k + rank)
                
                if candidate_key not in rrf_scores:
                    rrf_scores[candidate_key] = [0.0, candidate, [candidate]]
                else:
                    rrf_scores[candidate_key][2].append(candidate)
                
                # Accumulate RRF score
                rrf_scores[candidate_key][0] += rrf_contribution
                
                # Select best representative candidate with improved logic
                current_best = rrf_scores[candidate_key][1]
                new_best = self._select_best_candidate(current_best, candidate)
                rrf_scores[candidate_key][1] = new_best
        
        logger.debug(f"Total unique candidates after processing: {len(rrf_scores)}")
        
        # Create final candidates with RRF scores and filtering
        final_candidates = []
        filtered_count = 0
        
        for candidate_key, (rrf_score, best_candidate, all_candidates) in rrf_scores.items():
            # Apply minimum score filtering
            if rrf_score < min_rrf_score:
                filtered_count += 1
                continue
            
            # Find best matched_value from all candidates for this key
            best_matched_value = self._find_best_matched_value(all_candidates)
            
            # Find best encoding_mapping_info from all candidates for this key
            best_encoding_mapping_info = self._find_best_encoding_mapping_info(all_candidates)
            
            # Apply VALUE MATCH BOOST for candidates with concrete matched values
            final_rrf_score = rrf_score
            if best_matched_value:
                # Boost RRF score for value matches by 50% to increase their ranking
                # This ensures value matches have higher priority in final selection
                final_rrf_score = rrf_score * 1.5
                logger.debug(f"Value boost: {candidate_key} boosted from {rrf_score:.6f} to {final_rrf_score:.6f} (matched: {best_matched_value})")
            
            # Create new candidate with RRF score but preserve other metadata
            rrf_candidate = SlotCandidate(
                slot_type=best_candidate.slot_type,
                original_phrase=original_phrase,  # Use the entity phrase
                candidate_metadata=best_candidate.candidate_metadata,
                score=final_rrf_score,  # Use boosted RRF score for value matches
                matched_value=best_matched_value,  # Use best available matched_value
                match_type=f"rrf_fusion({best_candidate.match_type})",  # Indicate RRF fusion
                encoding_mapping_info=best_encoding_mapping_info  # Preserve encoding mapping info
            )
            final_candidates.append(rrf_candidate)
        
        logger.debug(f"{filtered_count} candidates filtered out due to low RRF score (< {min_rrf_score})")
        
        # Sort by RRF score (highest first)
        final_candidates.sort(key=lambda x: x.score, reverse=True)
        
        # Apply VALUE-AWARE candidate limit to preserve important value matches
        limited_candidates = self._apply_value_aware_candidate_limit(
            final_candidates, max_candidates, original_phrase
        )
        
        logger.debug(f"RRF fusion for '{original_phrase}': combined {len(retrieval_results)} methods "
                    f"into {len(final_candidates)} candidates, filtered to {len(limited_candidates)} "
                    f"(min_score={min_rrf_score}, max_count={max_candidates})")
        
        return limited_candidates
    
    def _apply_value_aware_candidate_limit(
        self, 
        candidates: List[SlotCandidate], 
        max_candidates: int, 
        original_phrase: str
    ) -> List[SlotCandidate]:
        """
        Apply intelligent candidate limiting that preserves value matches with high recall.
        
        Strategy:
        1. Always preserve candidates with matched_value (they have concrete value evidence)
        2. Fill remaining slots with highest RRF scores
        3. Ensure minimum value match representation even if RRF scores are lower
        
        Args:
            candidates: Sorted candidates by RRF score (highest first)
            max_candidates: Maximum number of candidates to return
            original_phrase: Original query phrase for context
            
        Returns:
            List[SlotCandidate]: Intelligently limited candidates preserving value matches
        """
        if len(candidates) <= max_candidates:
            return candidates
        
        # Separate candidates with and without matched values
        value_candidates = [c for c in candidates if c.matched_value]
        non_value_candidates = [c for c in candidates if not c.matched_value]
        
        # logger.debug(f"{len(value_candidates)} candidates with values, {len(non_value_candidates)} without values")
        
        # Strategy 1: Always preserve high-quality value matches (top 50% of value candidates)
        # but limit to reasonable number to avoid noise
        max_value_preserves = min(len(value_candidates), max(max_candidates // 3, 5))
        preserved_value_candidates = value_candidates[:max_value_preserves]
        
        # Strategy 2: Fill remaining slots with best overall candidates (including remaining value matches)
        remaining_slots = max_candidates - len(preserved_value_candidates)
        remaining_candidates = [c for c in candidates if c not in preserved_value_candidates]
        selected_remaining = remaining_candidates[:remaining_slots] if remaining_slots > 0 else []
        
        # Combine and maintain original RRF order
        final_selection = preserved_value_candidates + selected_remaining
        
        # Re-sort by original RRF score to maintain ranking logic
        final_selection.sort(key=lambda x: x.score, reverse=True)
        
        value_matches_in_final = sum(1 for c in final_selection if c.matched_value)
        # logger.debug(f"Final selection: {len(final_selection)} candidates, {value_matches_in_final} with matched values")
        
        return final_selection
    
    def _select_best_candidate(self, current_best: SlotCandidate, new_candidate: SlotCandidate) -> SlotCandidate:
        """
        Select the best representative candidate between two options.
        Priority: 1) Has matched_value, 2) Has encoding_mapping_info, 3) Higher original score
        """
        # If current_best has matched_value but new_candidate doesn't, keep current
        if current_best.matched_value and not new_candidate.matched_value:
            return current_best
        
        # If new_candidate has matched_value but current_best doesn't, use new
        if new_candidate.matched_value and not current_best.matched_value:
            return new_candidate
        
        # If both have or both lack matched_value, check encoding_mapping_info
        if current_best.encoding_mapping_info and not new_candidate.encoding_mapping_info:
            return current_best
        
        if new_candidate.encoding_mapping_info and not current_best.encoding_mapping_info:
            return new_candidate
        
        # If both have or both lack encoding_mapping_info, use the one with higher original score
        return new_candidate if new_candidate.score > current_best.score else current_best
    
    def _find_best_matched_value(self, candidates: List[SlotCandidate]) -> Optional[str]:
        """
        Find the best matched_value from a list of candidates.
        Returns the first non-empty matched_value found, or None if all are empty.
        """
        for candidate in candidates:
            if candidate.matched_value:
                return candidate.matched_value
        return None
    
    def _find_best_encoding_mapping_info(self, candidates: List[SlotCandidate]) -> Optional[Dict[str, Any]]:
        """
        Find the best encoding_mapping_info from a list of candidates.
        Returns the first non-empty encoding_mapping_info found, or None if all are empty.
        """
        for candidate in candidates:
            if candidate.encoding_mapping_info:
                return candidate.encoding_mapping_info
        return None
    
    def _find_metadata_join_relationships(self, table_names: List[str]) -> List[JoinCandidate]:
        """Find JOIN relationships using table metadata (primary_keys and foreign_keys)"""
        join_candidates = []
        
        # Get metadata for all tables
        table_metadata = {}
        for table_name in table_names:
            metadata = self._get_table_metadata(table_name)
            if metadata:
                table_metadata[table_name] = metadata

        # Find direct foreign key relationships
        for table1_name in table_names:
            if table1_name not in table_metadata:
                continue
            
            table1_metadata = table_metadata[table1_name]

            # print('###############')
            # print(table1_metadata)
            
            table1_fks = table1_metadata.get('foreign_keys', [])
            
            # Check each foreign key in table1
            for fk_info in table1_fks:
                if isinstance(fk_info, dict):
                    # Support both normalized keys and SQLite PRAGMA keys
                    ref_table = fk_info.get('referenced_table') or fk_info.get('table')
                    ref_column = fk_info.get('referenced_column') or fk_info.get('to')
                    fk_column = fk_info.get('column') or fk_info.get('from')
                    
                    if ref_table and ref_column and fk_column and ref_table in table_names:
                        join_candidates.append(JoinCandidate(
                            table1=table1_name,
                            column1=str(fk_column),
                            table2=str(ref_table),
                            column2=str(ref_column),
                            join_type='foreign_key',
                            confidence=1.0
                        ))
        
        # # Find reverse foreign key relationships (where other tables reference our tables)
        # for table1_name in table_names:
        #     if table1_name not in table_metadata:
        #         continue
                
        #     table1_metadata = table_metadata[table1_name]
        #     table1_pks = table1_metadata.get('primary_keys', [])
            
        #     # Check if any other table has foreign keys pointing to this table
        #     for table2_name in table_names:
        #         if table2_name == table1_name or table2_name not in table_metadata:
        #             continue
                    
        #         table2_metadata = table_metadata[table2_name]
        #         table2_fks = table2_metadata.get('foreign_keys', [])
                
        #         for fk_info in table2_fks:
        #             if isinstance(fk_info, dict):
        #                 ref_table = fk_info.get('referenced_table') or fk_info.get('table')
        #                 ref_column = fk_info.get('referenced_column') or fk_info.get('to')
        #                 fk_column = fk_info.get('column') or fk_info.get('from')
                        
        #                 if ref_table == table1_name and ref_column in table1_pks:
        #                     join_candidates.append(JoinCandidate(
        #                         table1=table2_name,
        #                         column1=str(fk_column),
        #                         table2=table1_name,
        #                         column2=str(ref_column),
        #                         join_type='foreign_key',
        #                         confidence=1.0
        #                     ))
        
        # Remove duplicates based on table pairs and columns -- Deleted Temporarily
        # unique_joins = self._deduplicate_join_candidates(join_candidates)
        
        return join_candidates
        
    def get_join_candidates_for_tables(self, table_names: List[str]) -> List[JoinCandidate]:
        """
        Get JOIN candidates for a given list of tables.
        This method should be called after schema merging to ensure we only detect
        joins for tables that are actually selected.
        
        Args:
            table_names: List of table names to find join relationships for
            
        Returns:
            List of JoinCandidate objects representing join relationships between the tables
        """
        if not table_names or len(table_names) < 2:
            logger.debug(f"Not enough tables ({len(table_names)}) to detect join relationships")
            return []
        
        # Remove duplicates and filter out None/empty values
        unique_tables = list(set(filter(None, table_names)))
        
        if len(unique_tables) < 2:
            logger.debug(f"Not enough unique tables ({len(unique_tables)}) to detect join relationships")
            return []
        
        logger.debug(f"Detecting JOIN relationships for tables: {unique_tables}")
        join_candidates = self._find_metadata_join_relationships(unique_tables)
        logger.debug(f"Found {len(join_candidates)} JOIN relationships for {len(unique_tables)} tables")
        
        return join_candidates
    
    def _match_entity_keywords(self, entity_phrase: str, slot_type: str, top_k: int = 10) -> List[SlotCandidate]:
        """Match entity phrase against table names and column names using SPLADE sparse retrieval"""
        logger.debug(f"SPLADE matcher searching for '{entity_phrase}' with top_k={top_k}")
        candidates = []
        
        # Search using SPLADE on table names and column names (if available)
        if not self.splade_available:
            logger.debug("SPLADE matcher not available, skipping keyword matching")
            return candidates
            
        try:
            splade_results = self.splade_matcher.search(entity_phrase, top_k=top_k)
            logger.debug(f"SPLADE search returned {len(splade_results)} results")
        
            for i, result in enumerate(splade_results):
                candidates.append(SlotCandidate(
                    slot_type=slot_type,
                    original_phrase=entity_phrase,
                    candidate_metadata=result.column_id,
                    score=result.score,
                    match_type='splade_sparse'
                ))
        except Exception as e:
            logger.debug(f"SPLADE matcher error: {e}")
        
        return candidates
    
    def _match_entity_semantic(self, entity_phrase: str, slot_type: str, top_k: int = 10) -> List[SlotCandidate]:
        """Match entity phrase using semantic similarity"""
        logger.debug(f"Vector matcher searching for '{entity_phrase}' with top_k={top_k}")
        candidates = []
        
        try:
            # Use vector matcher for semantic similarity
            vector_results = self.vector_matcher.search_facet(
                query=entity_phrase,
                facet='description',
                top_k=top_k
            )
            logger.debug(f"Vector search returned {len(vector_results)} results")
        
            for i, result in enumerate(vector_results):
                candidates.append(SlotCandidate(
                    slot_type=slot_type,
                    original_phrase=entity_phrase,
                    candidate_metadata=result.column_id,
                    score=result.score,
                    match_type='semantic'
                ))
        except Exception as e:
            logger.debug(f"Vector matcher error: {e}")
        
        return candidates
    
    def _extract_and_expand_values(self, filter_phrase: str) -> List[str]:
        """
        Extract core values and generate expansions using lightweight LLM call
        
        Args:
            filter_phrase: Filter phrase like "in California" or "greater than 50"
            
        Returns:
            List[str]: Expanded values including original phrase and extracted values/synonyms
        """
        if not self.llm_client:
            logger.warning("LLM client not available for value expansion")
            return [filter_phrase]
        
        prompt = f'''You are a value extraction expert. Given a filter phrase, extract any core values and provide common synonyms or abbreviations.

**Task**: Analyze the phrase and determine if it contains specific values that could be found in database columns.

**Rules**:
1. If the phrase contains a specific value (location, number, name, etc.), extract it
2. Generate common synonyms, abbreviations, or alternative representations
3. If it's purely a schema description (like "mailing address"), return null for value
4. Focus on values that would actually appear in database data

**Examples**:
- "in California" → {{"value": "California", "expansions": ["CA", "Calif"]}}
- "greater than 50" → {{"value": "50", "expansions": []}}
- "mailing state address" → {{"value": null, "expansions": []}}
- "in Orange County" → {{"value": "Orange County", "expansions": ["Orange", "OC"]}}

**Input**: "{filter_phrase}"

**Output** (JSON only):
```json
{{
    "value": "extracted_value_or_null",
    "expansions": ["synonym1", "synonym2"]
}}
```'''

        try:
            response = self.llm_client.call_with_messages(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if not json_match:
                # Try to find JSON without markdown
                json_match = re.search(r'(\{[^}]*"value"[^}]*\})', response, re.DOTALL)
            
            if json_match:
                import json
                parsed = json.loads(json_match.group(1))
                
                # Build expanded values list
                expanded_values = [filter_phrase]  # Always include original phrase
                
                core_value = parsed.get('value')
                if core_value and str(core_value).lower() != 'null':
                    expanded_values.append(str(core_value))
                
                expansions = parsed.get('expansions', [])
                if isinstance(expansions, list):
                    for exp in expansions:
                        if exp and str(exp).strip():
                            expanded_values.append(str(exp).strip())
                
                # Remove duplicates while preserving order
                seen = set()
                unique_values = []
                for val in expanded_values:
                    val_lower = val.lower()
                    if val_lower not in seen:
                        seen.add(val_lower)
                        unique_values.append(val)
                
                logger.debug(f"Value expansion for '{filter_phrase}' -> {unique_values}")
                return unique_values
                
        except Exception as e:
            logger.debug(f"Value expansion failed for '{filter_phrase}': {e}")
        
        # Fallback to original phrase
        return [filter_phrase]
    
    def _check_encoding_mapping(self, column_id: str, matched_value: str) -> Optional[Dict[str, Any]]:
        """
        Check if matched_value exists in the column's encoding_mapping and return key:value mapping info
        
        Args:
            column_id: Column identifier in format table.column
            matched_value: The matched value to check
            
        Returns:
            Dict containing encoding mapping info if found, None otherwise
        """
        if not matched_value or not column_id or '.' not in column_id:
            logger.debug(f"Invalid input for encoding mapping - matched_value: {matched_value}, column_id: {column_id}")
            return None
        
        try:
            # Parse column_id to get table and column names
            parts = column_id.split('.')
            if len(parts) >= 2:
                table_name = parts[-2]
                column_name = parts[-1]
                logger.debug(f"Checking {table_name}.{column_name} for matched_value: {matched_value}")
                
                # Get column metadata
                column_meta = self.column_provider.get_column_metadata(table_name, column_name)
                if not column_meta:
                    logger.debug(f"No column metadata found for {table_name}.{column_name}")
                    return None
                
                # Check encoding_mapping
                encoding_mapping_raw = column_meta.get('encoding_mapping', {})
                logger.debug(f"Raw encoding_mapping for {table_name}.{column_name}: {encoding_mapping_raw} (type: {type(encoding_mapping_raw)})")
                
                # Handle different formats of encoding_mapping
                encoding_mapping = {}
                if isinstance(encoding_mapping_raw, dict):
                    encoding_mapping = encoding_mapping_raw
                elif isinstance(encoding_mapping_raw, str):
                    try:
                        import json
                        encoding_mapping = json.loads(encoding_mapping_raw)
                        logger.debug(f"Parsed JSON string to dict: {encoding_mapping}")
                    except json.JSONDecodeError as e:
                        logger.debug(f"Failed to parse JSON string: {e}")
                        return None
                elif encoding_mapping_raw is not None:
                    logger.debug(f"Unexpected encoding_mapping type: {type(encoding_mapping_raw)}")
                    return None
                
                logger.debug(f"Processed encoding_mapping for {table_name}.{column_name}: {encoding_mapping}")
                
                if not encoding_mapping:
                    logger.debug(f"No encoding_mapping found for {table_name}.{column_name}")
                    return None
                
                # Look for matched_value in the values of encoding_mapping
                # Handle different data types for keys and values
                for key, value in encoding_mapping.items():
                    # Convert both to strings for comparison, handling None values
                    key_str = str(key) if key is not None else ""
                    value_str = str(value) if value is not None else ""
                    matched_value_str = str(matched_value) if matched_value is not None else ""
                    
                    # Try matching both value and key (in case the logic was reversed)
                    if (value_str.lower() == matched_value_str.lower() or 
                        key_str.lower() == matched_value_str.lower()):
                        encoding_info = {
                            'found_in_encoding': True,
                            'encoding_key': key_str,
                            'encoding_value': value_str,  
                            'original_matched_value': matched_value_str,
                            'match_type': 'value_match' if value_str.lower() == matched_value_str.lower() else 'key_match'
                        }
                        logger.debug(f"ENCODING MATCH FOUND! Returning: {encoding_info}")
                        return encoding_info
                
                logger.debug(f"No match found for '{matched_value}' in encoding_mapping")
                return None
                
        except Exception as e:
            logger.debug(f"Error checking encoding mapping for {column_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _deduplicate_candidates(self, candidates: List[SlotCandidate]) -> List[SlotCandidate]:
        """
        Remove duplicate candidates based on candidate_metadata, keeping the best score
        
        Args:
            candidates: List of SlotCandidate objects
            
        Returns:
            List[SlotCandidate]: Deduplicated candidates with best scores preserved
        """
        if not candidates:
            return []
        
        # Group by candidate_metadata and keep the best candidate for each
        best_candidates = {}  # candidate_metadata -> SlotCandidate
        
        for candidate in candidates:
            metadata = candidate.candidate_metadata
            if metadata not in best_candidates:
                best_candidates[metadata] = candidate
            else:
                # Use the improved selection logic that considers encoding_mapping_info
                best_candidates[metadata] = self._select_best_candidate(best_candidates[metadata], candidate)
        
        # Return as list, sorted by score descending
        result = list(best_candidates.values())
        result.sort(key=lambda x: x.score, reverse=True)
        
        logger.debug(f"Deduplicated {len(candidates)} candidates to {len(result)} unique candidates")
        return result
    
    def _value_search(
        self,
        phrase: str,
        slot_type: str,
        original_phrase: str,
        base_weight: float = 1.0,
    ) -> List[SlotCandidate]:
        """Run value matcher and map results to SlotCandidate with unified scoring."""
        logger.debug(f"Value matcher searching for '{phrase}'")
        
        try:
            results = self.value_matcher.search(phrase)
            logger.debug(f"Value search returned {len(results)} results")
            
            candidates: List[SlotCandidate] = []
            for i, result in enumerate(results):
                logger.debug(f"Value result {i}: {result.column_id}, score={result.match_score}")
                score = result.match_score
                matched_value = getattr(result, 'matched_value', None)
                match_type = getattr(result, 'match_type', None)
                
                # Check encoding_mapping for matched_value
                encoding_info = self._check_encoding_mapping(result.column_id, matched_value)
                
                if encoding_info:
                    logger.debug(f"Found encoding mapping for {result.column_id}: {encoding_info}")
                
                candidates.append(SlotCandidate(
                    slot_type=slot_type,
                    original_phrase=original_phrase,
                    candidate_metadata=result.column_id,
                    score=score,
                    matched_value=matched_value,
                    match_type=match_type,
                    encoding_mapping_info=encoding_info
                ))
            return candidates
        except Exception as e:
            logger.debug(f"Value matcher error: {e}")
            return []

    def _aggregate_value_matches(self, extended_candidates: Dict[str, List[SlotCandidate]]) -> Dict[str, Dict[str, Any]]:
        """Aggregate matched values/types from candidates keyed by column_id.
        Returns mapping: column_id -> { 'values': [...], 'types': [...], 'encoding_mappings': {...} }
        """
        column_to_matches: Dict[str, Dict[str, Any]] = {}
        
        candidates_with_values = 0
        for slot_key, cand_list in extended_candidates.items():
            for cand in cand_list:
                # 只有当 matched_value 是 None 时才跳过
                if cand.matched_value is None:
                    continue
                    
                candidates_with_values += 1
                key = cand.candidate_metadata
                if key not in column_to_matches:
                    column_to_matches[key] = {'values': [], 'types': [], 'encoding_mappings': {}}
                if cand.matched_value not in column_to_matches[key]['values']:
                    column_to_matches[key]['values'].append(cand.matched_value)
                if cand.match_type and cand.match_type not in column_to_matches[key]['types']:
                    column_to_matches[key]['types'].append(cand.match_type)
                
                # Handle encoding_mapping_info with data_type awareness
                if cand.encoding_mapping_info:
                    encoding_key = cand.encoding_mapping_info.get('encoding_key')
                    encoding_value = cand.encoding_mapping_info.get('encoding_value')
                    
                    if encoding_key is not None and encoding_value is not None:
                        # 根据data_type格式化value
                        formatted_value = self._format_value_with_data_type(encoding_value, key)
                        column_to_matches[key]['encoding_mappings'][str(encoding_key)] = formatted_value
        
        logger.debug(f"Found {candidates_with_values} candidates with matched values, aggregated into {len(column_to_matches)} unique columns")
        return column_to_matches

    def _format_value_with_data_type(self, value: Any, column_id: str) -> Any:
        """
        根据字段的data_type格式化单个值，保留原始数据类型
        
        Args:
            value: 要格式化的值
            column_id: 列标识符，用于获取data_type
            
        Returns:
            Any: 格式化后的值
        """
        try:
            # 获取字段的data_type
            data_type = self._get_column_data_type(column_id)
            
            # 根据data_type格式化
            if data_type and self._is_numeric_type(data_type):
                # 数字类型：尝试转换为数字
                return self._convert_to_numeric(value)
            else:
                # 字符串/文本类型：保持原值
                return value
                
        except Exception as e:
            logger.debug(f"Error formatting value for {column_id}: {e}")
            return value

    def _get_column_data_type(self, column_id: str) -> Optional[str]:
        """
        获取字段的data_type
        
        Args:
            column_id: 列标识符 (table.column格式)
            
        Returns:
            Optional[str]: 字段的data_type，如果获取失败则返回None
        """
        try:
            if '.' in column_id:
                parts = column_id.split('.')
                if len(parts) >= 2:
                    table_name = parts[-2]
                    column_name = parts[-1]
                    column_meta = self.column_provider.get_column_metadata(table_name, column_name)
                    if column_meta:
                        return column_meta.get('data_type')
        except Exception as e:
            logger.debug(f"Error getting data_type for {column_id}: {e}")
        return None
    
    def _is_numeric_type(self, data_type: str) -> bool:
        """
        判断data_type是否为数字类型
        
        Args:
            data_type: 数据类型字符串
            
        Returns:
            bool: 是否为数字类型
        """
        if not data_type:
            return False
        
        data_type_lower = str(data_type).lower()
        numeric_types = [
            'int', 'integer', 'bigint', 'smallint', 'tinyint',
            'float', 'double', 'decimal', 'numeric', 'real',
            'number', 'serial'
        ]
        
        return any(nt in data_type_lower for nt in numeric_types)
    
    def _convert_to_numeric(self, value: Any) -> Any:
        """
        尝试将值转换为数字类型
        
        Args:
            value: 要转换的值
            
        Returns:
            Any: 转换后的数字值，如果转换失败则返回原值
        """
        if value is None:
            return value
        
        # 如果已经是数字类型，直接返回
        if isinstance(value, (int, float)):
            return value
        
        # 尝试转换为数字
        try:
            # 先尝试转换为int
            if str(value).replace('.', '').replace('-', '').isdigit():
                if '.' in str(value):
                    return float(value)
                else:
                    return int(value)
            else:
                return value
        except (ValueError, TypeError):
            return value

    
    def _analyze_query(self, user_question: str) -> IntentAnalysis:
        if not self.query_analyzer:
            self.initialize_query_analyzer()

        return self.query_analyzer.analyze(user_question)

    def _select_schemas_per_query_term_with_llm(
        self,
        user_question: str,
        extended_candidates: Dict[str, List[SlotCandidate]],
        top_k: int,
        max_workers: int = 5
    ) -> PerQueryTermSelectionResult:
        """
        使用LLM为每个检索词选择top_k个schema（多线程并行处理）
        
        Args:
            user_question: 用户原始问题
            extended_candidates: 检索结果 (phrase -> List[SlotCandidate])
            top_k: 每个检索词需要选择的schema数量
            max_workers: 最大并发线程数
            
        Returns:
            PerQueryTermSelectionResult: 包含每个检索词的选择结果和合并结果
        """
        if not self.llm_client:
            raise ValueError("LLM client not initialized")
        
        logger.info(f"Starting LLM-based schema selection for {len(extended_candidates)} query terms (top_k={top_k})")
        
        query_term_selections = []
        
        def process_single_query_term(phrase: str, candidates: List[SlotCandidate]) -> QueryTermSchemaSelection:
            """处理单个检索词的LLM选择（用于多线程）"""
            try:

                prompt = self._build_llm_prompt_for_query_term(
                    user_question=user_question,
                    target_phrase=phrase,
                    candidates=candidates,
                    top_k=top_k
                )
                
                logger.debug(f"Calling LLM for phrase '{phrase}' with {len(candidates)} candidates")
                response = self.llm_client.call_with_messages(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                
                # 解析LLM响应
                selected_schemas = self._parse_llm_schema_selection_response(response)
                
                # 规范化schema格式
                normalized_schemas = []
                for schema in selected_schemas:
                    normalized = self._normalize_schema_format(schema)
                    if normalized:
                        normalized_schemas.append(normalized)
                
                logger.info(f"LLM selected {len(normalized_schemas)} schemas for phrase '{phrase}': {normalized_schemas}")
                
                # 收集每个selected schema的value match信息
                schemas_with_values = self._collect_value_matches_for_schemas(
                    selected_schemas=normalized_schemas,
                    candidates=candidates
                )
                
                return QueryTermSchemaSelection(
                    query_term=phrase,
                    selected_schemas=normalized_schemas,  
                    selected_schemas_with_values=schemas_with_values,
                    all_candidates_count=len(candidates)
                )
                
            except Exception as e:
                logger.error(f"Error processing query term '{phrase}': {e}", exc_info=True)
                return QueryTermSchemaSelection(
                    query_term=phrase,
                    selected_schemas=[],
                    selected_schemas_with_values=[],
                    all_candidates_count=len(candidates) if candidates else 0
                )
        
        # 使用多线程并行处理所有检索词
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_phrase = {
                executor.submit(process_single_query_term, phrase, candidates): phrase
                for phrase, candidates in extended_candidates.items()
                if candidates  # 只处理有候选的检索词
            }
            
            # 收集结果
            for future in as_completed(future_to_phrase):
                phrase = future_to_phrase[future]
                try:
                    selection_result = future.result()
                    query_term_selections.append(selection_result)
                except Exception as e:
                    logger.error(f"Error getting result for phrase '{phrase}': {e}", exc_info=True)
                    query_term_selections.append(QueryTermSchemaSelection(
                        query_term=phrase,
                        selected_schemas=[],
                        all_candidates_count=0
                    ))
        
        # 合并所有选择的schema并去重
        merged_schemas = []
        seen_schemas = set()
        for selection in query_term_selections:
            for schema in selection.selected_schemas:
                schema_lower = schema.lower()
                if schema_lower not in seen_schemas:
                    seen_schemas.add(schema_lower)
                    merged_schemas.append(schema)
        
        logger.info(f"LLM selection completed: {len(query_term_selections)} query terms processed, "
                   f"{len(merged_schemas)} unique schemas selected")
        
        return PerQueryTermSelectionResult(
            query_term_selections=query_term_selections,
            merged_schemas=merged_schemas
        )
    
    def _collect_value_matches_for_schemas(
        self,
        selected_schemas: List[str],
        candidates: List[SlotCandidate]
    ) -> List[SchemaWithValueMatches]:
        """
        收集每个selected schema对应的value match信息
        
        Args:
            selected_schemas: LLM选择的schema列表
            candidates: 该检索词的所有候选
            
        Returns:
            List[SchemaWithValueMatches]: 带value match信息的schema列表
        """
        schemas_with_values = []
        
        # 为每个selected schema构建value match信息
        for schema in selected_schemas:
            normalized_schema = self._normalize_schema_format(schema)
            if not normalized_schema:
                continue
            
            # 从candidates中找到所有匹配该schema的候选
            matched_values_set = set()
            match_types_set = set()
            encoding_mappings_dict = {}
            
            for candidate in candidates:
                candidate_schema = self._normalize_schema_format(candidate.candidate_metadata)
                if candidate_schema.lower() == normalized_schema.lower():
                    # 收集matched_value
                    if candidate.matched_value:
                        matched_values_set.add(candidate.matched_value)
                        # 只有当有matched_value时才收集match_type
                        if candidate.match_type:
                            match_types_set.add(candidate.match_type)
                    
                    # 收集encoding_mapping_info
                    if candidate.encoding_mapping_info:
                        encoding_key = candidate.encoding_mapping_info.get('encoding_key')
                        encoding_value = candidate.encoding_mapping_info.get('encoding_value')
                        if encoding_key is not None and encoding_value is not None:
                            # 根据data_type格式化value
                            formatted_value = self._format_value_with_data_type(encoding_value, normalized_schema)
                            encoding_mappings_dict[str(encoding_key)] = formatted_value
            
            # 创建SchemaWithValueMatches对象
            schema_with_values = SchemaWithValueMatches(
                schema=normalized_schema,
                matched_values=sorted(list(matched_values_set)),
                match_types=sorted(list(match_types_set)),
                encoding_mappings=encoding_mappings_dict
            )
            schemas_with_values.append(schema_with_values)
            
            # 打印调试信息
            if matched_values_set or encoding_mappings_dict:
                logger.debug(f"Schema '{normalized_schema}' has value matches: "
                           f"values={list(matched_values_set)}, "
                           f"types={list(match_types_set)}, "
                           f"encodings={encoding_mappings_dict}")
        
        return schemas_with_values
    
    def _parse_llm_schema_selection_response(self, response: str) -> Tuple[List[str], str]:
        """
        解析LLM返回的schema选择结果
        
        Args:
            response: LLM返回的文本响应
            
        Returns:
            Tuple[List[str], str]: (selected_schemas列表, reasoning字符串，reasoning可选)
        """
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if not json_match:
                # 尝试查找JSON对象（支持嵌套，使用更宽松的匹配）
                json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"selected_schemas"[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                parsed = json.loads(json_str)
                
                selected_schemas = parsed.get('selected_schemas', [])

                
                # 确保selected_schemas是列表
                if not isinstance(selected_schemas, list):
                    selected_schemas = []

                return selected_schemas
                
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response content: {response[:500]}")
            return []
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return []
    
    def _format_value_match_info_for_llm(self, candidate: SlotCandidate) -> str:
        """
        格式化value匹配信息用于LLM prompt展示
        
        Args:
            candidate: SlotCandidate对象
            
        Returns:
            str: 格式化的value匹配信息字符串，如果没有value匹配则返回空字符串
        """
        value_info_parts = []
        
        # 检查是否有matched_value
        if candidate.matched_value:
            match_type = candidate.match_type or "unknown"
            
            # 判断是否为exact match（高置信度信号）
            is_exact_match = (
                'exact' in match_type.lower() or 
                match_type.lower() == 'rrf_fusion(exact)'
            )
            
            # 格式化匹配信息
            if is_exact_match:
                value_info_parts.append(f"✅ EXACT VALUE MATCH: '{candidate.matched_value}'")
            else:
                value_info_parts.append(f"⚠️ Value Match: '{candidate.matched_value}' (type: {match_type})")
            
            # 检查encoding_mapping信息
            if candidate.encoding_mapping_info:
                encoding_info = candidate.encoding_mapping_info
                encoding_key = encoding_info.get('encoding_key', '')
                encoding_value = encoding_info.get('encoding_value', '')
                match_type_encoding = encoding_info.get('match_type', '')
                
                if encoding_key and encoding_value:
                    if match_type_encoding == 'value_match':
                        value_info_parts.append(f"   Encoding Mapping: '{encoding_key}' -> '{encoding_value}' (matched in encoding)")
                    else:
                        value_info_parts.append(f"   Encoding Mapping: '{encoding_key}' -> '{encoding_value}'")
        
        return "\n".join(value_info_parts) if value_info_parts else ""
    
    def _build_llm_prompt_for_query_term(
        self,
        user_question: str,
        target_phrase: str,
        candidates: List[SlotCandidate],
        top_k: int
    ) -> str:
        """
        为单个检索词构建LLM prompt
        
        Args:
            user_question: 用户原始问题
            target_phrase: 当前关注的短语（检索词）
            candidates: 候选Schema列表
            top_k: 需要选择的schema数量
            
        Returns:
            str: 格式化的LLM prompt
        """
        # 展示所有候选（不进行筛选）
        display_candidates = candidates
        
        # 构建候选schema信息
        candidate_schemas_text = []
        for idx, candidate in enumerate(display_candidates, 1):
            schema_info_parts = []
            
            # Schema标识
            schema_id = self._normalize_schema_format(candidate.candidate_metadata)
            schema_info_parts.append(f"Schema {idx}: {schema_id}")
            
            # 获取详细元数据信息
            metadata_info = self._get_candidate_metadata_info(candidate.candidate_metadata)
            if metadata_info:
                schema_info_parts.append(f"  Metadata: {metadata_info}")
            
            # Value匹配信息
            value_match_info = self._format_value_match_info_for_llm(candidate)
            if value_match_info:
                schema_info_parts.append(f"  {value_match_info}")
            
            candidate_schemas_text.append("\n".join(schema_info_parts))
        
        candidates_text = "\n\n".join(candidate_schemas_text)
        
        # 构建完整prompt
        prompt = f"""You are a schema matching expert for NL2SQL tasks. Your task is to identify all possible database schemas (columns) that could match entities mentioned in a natural language question.

**Task Context:**
In NL2SQL tasks, we need to convert a natural language question into a SQL query. To do this, we must identify all database schemas (table.column) that could potentially match entities mentioned in the question. Your goal is to find all possible matches between the question and database schemas.

**User Question:**
{user_question}

**Target Phrase (Containing Entities to be Matched):**
"{target_phrase}"

**Your Task:**
Evaluate the entity "{target_phrase}" extracted from the question and identify which schemas in the candidate list could potentially match this entity. Select the top {top_k} most likely matching schemas (table.column format).

**Candidate Schemas:**
{candidates_text}

**Evaluation Criteria:**
When evaluating whether a schema could match the entity "{target_phrase}", consider:

1. **Value Match (HIGH CONFIDENCE SIGNAL)**: 
   - ✅ EXACT VALUE MATCH indicates the column contains the exact value mentioned in the phrase
   - ⚠️ Value Match indicates partial or fuzzy match
   - This is a strong indicator that the schema could match the entity

2. **Semantic Match**:
   - Does the schema's description/name semantically align with the phrase?
   - Consider synonyms, abbreviations, and domain-specific terminology
   - Could this schema potentially represent or contain the entity mentioned in the phrase?

3. **Data Type Compatibility**:
   - Does the schema's data type make sense for the entity?
   - (e.g., numeric types for numbers, text types for names/locations)

4. **Potential Usage in SQL**:
   - Could this schema be used in the SQL query to represent or filter by the entity?
   - Consider how the entity might appear in WHERE clauses, SELECT clauses, or JOIN conditions

**Output Format:**
After carefully evaluating all candidates, return a JSON object with the following structure:
```json
{{
    "selected_schemas": ["table1.column1", "table2.column2", ...]
}}
```

**Important Notes:**
- Select up to {top_k} schemas that could potentially match the entity "{target_phrase}"
- **CRITICAL**: The schemas in "selected_schemas" must be ordered by relevance (most relevant first)
- Prioritize schemas with EXACT VALUE MATCH (✅) as they have high confidence - these should appear first
- Consider all possible matches - a schema could match even if the match is not perfect
- If a schema has both good semantic match AND value match, it should be prioritized and ranked higher
- The goal is to find all possible matches, not just the most obvious ones
- Order the schemas from highest to lowest relevance based on: value matches, semantic matches and data type compatibility.

**Your Selection (ordered by relevance):**"""

        return prompt
   
    def _get_candidate_metadata_info(self, candidate_metadata: str) -> str:
        """获取候选schema的详细元数据信息"""

        if '.' in candidate_metadata:
            table_name, column_name = candidate_metadata.split('.', 1)
            column_meta = self.column_provider.get_column_metadata(table_name, column_name)
            if column_meta:
                info_parts = []
                # 处理description的nan值
                description = column_meta.get('description', '')
                if description and str(description).lower() not in ['nan', 'none', 'null', '']:
                    info_parts.append(f"描述: {description}")
                
                # 处理data_type的nan值
                data_type = column_meta.get('data_type', '')
                if data_type and str(data_type).lower() not in ['nan', 'none', 'null', '']:
                    info_parts.append(f"类型: {data_type}")
                
                # 添加null值信息
                null_info = self._format_null_info(column_meta)
                if null_info:
                    info_parts.append(f"空值: {null_info}")
                
                # 处理semantic_tags
                if column_meta.get('semantic_tags'):
                    tags = [tag.get('content', '') for tag in column_meta['semantic_tags'][:2] 
                            if tag.get('content') and str(tag.get('content')).lower() not in ['nan', 'none', 'null', '']]
                    if tags:
                        info_parts.append(f"语义标签: {'; '.join(tags)}")
                return ' | '.join(info_parts)
        else:
            # 表级别的元数据
            table_meta = self.column_provider.get_table_metadata(candidate_metadata)
            if table_meta and table_meta.get('description'):
                description = table_meta.get('description', '')
                if description and str(description).lower() not in ['nan', 'none', 'null', '']:
                    return f"表描述: {description}"
        
    
    def _normalize_schema_format(self, schema: str) -> str:
        """Normalize to table.column format"""
        if not schema or not isinstance(schema, str): return ""
        schema = schema.strip()
        if not schema: return ""
        
        parts = schema.split('.')
        if len(parts) >= 3:
            return f"{parts[-2]}.{parts[-1]}"
        elif len(parts) == 2:
            return schema
        else:
            return schema
    
    def _extract_required_tables_from_columns(self, selected_columns: List[str]) -> List[str]:
        """Extract table names from selected columns"""
        required_tables = set()
        for column in selected_columns:
            if '.' in column:
                parts = column.split('.')
                if len(parts) >= 2:
                    table_name = '.'.join(parts[:-1])
                    required_tables.add(table_name)
        return sorted(list(required_tables))
    
    def _process_value_matches_for_selected_schemas(
        self,
        selected_columns: List[str],
        value_matches_all: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:

        """Filter value matches to only include selected schemas"""

        value_matches_selected = {}
        normalized_selected = {self._normalize_schema_format(col).lower(): col for col in selected_columns}
        
        for vm_key, vm_value in value_matches_all.items():
            normalized_vm_key = self._normalize_schema_format(vm_key)
            if normalized_vm_key.lower() in normalized_selected:
                final_key = normalized_selected[normalized_vm_key.lower()]
                processed_vm_value = vm_value.copy()
                if 'encoding_mappings' not in processed_vm_value:
                    processed_vm_value['encoding_mappings'] = {}
                value_matches_selected[final_key] = processed_vm_value
                
        return value_matches_selected

    def _format_null_info(self, column_meta: Dict[str, Any]) -> str:
        """
        格式化列的空值信息
        
        Args:
            column_meta: 列的元数据字典
            
        Returns:
            str: 格式化的空值信息字符串，如 "nullable", "no nulls", "5 nulls"
        """
        if not column_meta:
            return ""
        
        # 检查is_nullable标志
        is_nullable = column_meta.get('is_nullable')
        
        # 检查null_count统计
        null_count = column_meta.get('null_count')
        
        # 如果有具体的null计数信息
        if null_count is not None:
            try:
                null_count_int = int(null_count)
                if null_count_int == 0:
                    return "no nulls"
                else:
                    return f"{null_count_int} nulls"
            except (ValueError, TypeError):
                pass
        
        # 如果只有nullable标志
        if is_nullable is not None:
            if is_nullable:
                return "nullable"
            else:
                return "not null"
        
        return ""
    