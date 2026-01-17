# Multi-Layer Retriever implementing DAIL-SQL strategy

import logging
from typing import List, Optional, Dict, Any, Set
from dataclasses import replace

from caf.memory.types import MemoryQuery, EpisodicRecord
from .candidate import RetrievalCandidate

logger = logging.getLogger(__name__)

class MultiLayerRetriever:
    """
    Multi-layer retrieval implementing DAIL-SQL inspired strategy:
    
    - Same Database: NLQ similarity + SQL skeleton similarity (weighted fusion)
    - Cross Database: SQL skeleton similarity only
    - Ground truth exclusion: exact normalized question matching
    - Error SQL correction: add correct SQL for error records
    """
    
    def __init__(self, nlq_similarity, sql_similarity, config: Dict[str, Any]):
        self.nlq_similarity = nlq_similarity
        self.sql_similarity = sql_similarity
        self.config = config.get('multi_layer', {})
        
        # Retrieval parameters
        self.same_db_nlq_weight = self.config.get('same_db_nlq_weight', 0.6)
        self.same_db_sql_weight = self.config.get('same_db_sql_weight', 0.4)
        self.min_sql_skeleton_threshold = self.config.get('min_sql_skeleton_threshold', 0.1)
        
        logger.debug("MultiLayerRetriever initialized")
    
    def retrieve(self, 
                query: MemoryQuery,
                records: List[EpisodicRecord],
                generated_sql: Optional[str] = None,
                current_database_id: Optional[str] = None,
                exclude_target_question: bool = True) -> List[RetrievalCandidate]:
        """
        Enhanced episodic retrieval with multi-layer strategy
        
        Args:
            query: Memory query object
            records: List of available episodic records
            generated_sql: Generated SQL query for skeleton comparison  
            current_database_id: Current database ID for same/cross-db classification
            exclude_target_question: Whether to exclude records with same normalized question
            
        Returns:
            List of ranked RetrievalCandidate objects
        """
        logger.debug(f"Starting multi-layer retrieval with {len(records)} records")
        
        if not records:
            return []
        
        # Step 1: Exclude ground truth data (same normalized question)
        filtered_records = records
        if exclude_target_question:
            filtered_records = self._exclude_ground_truth(records, query.query_content)
        
        # Step 2: Apply multi-layer retrieval strategy
        candidates = self._apply_multilayer_retrieval(
            filtered_records,
            query.query_content, 
            generated_sql,
            current_database_id,
            query.limit
        )
        
        # Step 3: Add correct SQL records for error SQL records in top-k
        candidates = self._add_correct_sql_for_errors(candidates, filtered_records)
        
        logger.debug(f"Multi-layer retrieval returned {len(candidates)} candidates")
        
        return candidates
    
    def _exclude_ground_truth(self, records: List[EpisodicRecord], target_question: str) -> List[EpisodicRecord]:
        """Exclude records with same normalized question (ground truth exclusion)"""
        normalized_target = self._normalize_question_for_comparison(target_question)
        
        filtered_records = []
        excluded_count = 0
        
        for record in records:
            normalized_record_question = self._normalize_question_for_comparison(record.user_query)
            
            if normalized_record_question != normalized_target:
                filtered_records.append(record)
            else:
                excluded_count += 1
        
        logger.debug(f"Excluded {excluded_count} ground truth records, {len(filtered_records)} remaining")
        return filtered_records
    
    def _normalize_question_for_comparison(self, question: Optional[str]) -> Optional[str]:
        """
        Normalize user questions for duplicate comparison by lowercasing, trimming whitespace,
        and removing trailing punctuation.
        """
        if not question:
            return None
        
        # Normalize: trim whitespace, lowercase, and remove trailing punctuation
        normalized = ' '.join(question.strip().split()).lower()
        
        # Remove trailing punctuation marks
        import string
        while normalized and normalized[-1] in string.punctuation:
            normalized = normalized[:-1]
        
        return normalized
    
    def _apply_multilayer_retrieval(self,
                                   records: List[EpisodicRecord], 
                                   target_question: str,
                                   target_sql: Optional[str],
                                   current_database_id: Optional[str],
                                   limit: int) -> List[RetrievalCandidate]:
        """
        Apply multi-layer retrieval strategy based on database context.
        
        This method retrieves top-k results from same database and cross database separately,
        then merges them without re-sorting to preserve the distinction between the two paths.
        This allows easy identification of which results come from same_db vs cross_db.
        """
        
        # Separate records by database
        same_db_records = []
        cross_db_records = []
        
        for record in records:
            is_same_database = (record.database_id == current_database_id)
            if is_same_database:
                same_db_records.append(record)
            else:
                cross_db_records.append(record)
        
        logger.info(f"Same DB records: {len(same_db_records)}, Cross DB records: {len(cross_db_records)}")
        
        # Score and get top-k for same database records
        same_db_candidates = []
        for record in same_db_records:
            candidate = RetrievalCandidate(record=record, retrieval_type="same_db")
            candidate = self._score_same_database(candidate, target_question, target_sql)
            if candidate.final_score > 0:
                same_db_candidates.append(candidate)
        
        same_db_candidates.sort(key=lambda x: x.final_score, reverse=True)
        same_db_top_k = same_db_candidates[:limit]
        
        # Score and get top-k for cross-database records
        cross_db_candidates = []
        for record in cross_db_records:
            candidate = RetrievalCandidate(record=record, retrieval_type="cross_db")
            candidate = self._score_cross_database(candidate, target_sql)
            if candidate.final_score > 0:
                cross_db_candidates.append(candidate)
        
        cross_db_candidates.sort(key=lambda x: x.final_score, reverse=True)
        cross_db_top_k = cross_db_candidates[:limit]


        # Merge results: combine same_db and cross_db top-k results
        # Keep them separate (same_db first, then cross_db) without re-sorting
        # This preserves the distinction between the two retrieval paths
        merged_candidates = []
        seen_keys = set()
        
        # Add same_db candidates first (preserve their internal order)
        for candidate in same_db_top_k:
            key = f"{candidate.record.session_id}_{candidate.record.round_id}"
            if key not in seen_keys:
                merged_candidates.append(candidate)
                seen_keys.add(key)
        
        # Add cross_db candidates (preserve their internal order)
        for candidate in cross_db_top_k:
            key = f"{candidate.record.session_id}_{candidate.record.round_id}"
            if key not in seen_keys:
                merged_candidates.append(candidate)
                seen_keys.add(key)
        
        # Note: We do NOT re-sort by final_score here to preserve the separation
        # between same_db and cross_db results. Users can identify them via retrieval_type.
        
        logger.debug(f"Retrieved {len(same_db_top_k)} same_db and {len(cross_db_top_k)} cross_db candidates, "
                    f"merged to {len(merged_candidates)} total (preserving separate paths)")
        
        return merged_candidates
    
    def _score_same_database(self,
                            candidate: RetrievalCandidate,
                            target_question: str,
                            target_sql: Optional[str]) -> RetrievalCandidate:
        """Score candidate for same database using weighted fusion"""
        
        # Calculate NLQ similarity using embedding-based approach
        candidate.nlq_similarity = self.nlq_similarity.calculate_similarity(
            candidate.record.user_query, target_question
        )
        
        # Calculate SQL skeleton similarity
        if target_sql and candidate.record.generated_sql:
            candidate.sql_skeleton_similarity = self.sql_similarity.calculate_similarity(
                candidate.record.generated_sql, target_sql
            )
        
        # Weighted fusion
        candidate.final_score = (
            self.same_db_nlq_weight * candidate.nlq_similarity +
            self.same_db_sql_weight * candidate.sql_skeleton_similarity
        )
        
        candidate.explanation = (
            f"Same DB: NLQ={candidate.nlq_similarity:.3f} * {self.same_db_nlq_weight} + "
            f"SQL={candidate.sql_skeleton_similarity:.3f} * {self.same_db_sql_weight} = "
            f"{candidate.final_score:.3f}"
        )
        
        return candidate
    
    def _score_cross_database(self,
                             candidate: RetrievalCandidate,
                             target_sql: Optional[str]) -> RetrievalCandidate:
        """Score candidate for cross database using SQL skeleton similarity only"""
        
        if not target_sql or not candidate.record.generated_sql:
            candidate.final_score = 0.0
            candidate.explanation = "Cross DB: No SQL available for comparison"
            return candidate
        
        candidate.sql_skeleton_similarity = self.sql_similarity.calculate_similarity(
            candidate.record.generated_sql, target_sql
        )
        
        # For cross-database, only use SQL skeleton similarity
        # Apply minimum threshold to ensure quality
        if candidate.sql_skeleton_similarity >= self.min_sql_skeleton_threshold:
            candidate.final_score = candidate.sql_skeleton_similarity
        else:
            candidate.final_score = 0.0
        
        candidate.explanation = (
            f"Cross DB: SQL={candidate.sql_skeleton_similarity:.3f} "
            f"(threshold={self.min_sql_skeleton_threshold}) = {candidate.final_score:.3f}"
        )
        
        return candidate
    
    def _add_correct_sql_for_errors(self,
                                   candidates: List[RetrievalCandidate],
                                   all_records: List[EpisodicRecord]) -> List[RetrievalCandidate]:
        """
        For error SQL records in top-k, find and add corresponding correct SQL records
        if they are not already in the top-k results.
        """
        # Track existing candidate keys to avoid duplicates
        existing_keys = {f"{c.record.session_id}_{c.record.round_id}" for c in candidates}
        
        # Track error SQL records that need correct SQL
        error_records_to_match = []
        for candidate in candidates:
            if not candidate.record.label:
                error_records_to_match.append(candidate.record)
        
        if not error_records_to_match:
            return candidates
        
        # Find correct SQL records for each error SQL
        additional_candidates = []
        for error_record in error_records_to_match:

            normalized_question = self._normalize_question_for_comparison(error_record.user_query)
            if not normalized_question:
                continue
            
            # Find correct SQL records matching the error record
            for record in all_records:
                # if record is not a correct SQL, skip
                if not record.label:
                    continue
                
                record_normalized = self._normalize_question_for_comparison(record.user_query)
                
                # Match by normalized question and same database  
                if (record_normalized == normalized_question and 
                    record.database_id == error_record.database_id):
                    
                    record_key = f"{record.session_id}_{record.round_id}"
                    if record_key in existing_keys:
                        continue  # Already in candidates
                    
                    # Create a candidate for this correct SQL record
                    # Use a lower score to indicate it's added as a complement
                    candidate = RetrievalCandidate(
                        record=record,
                        retrieval_type="same_db" if record.database_id == error_record.database_id else "cross_db",
                        final_score=0.5,  # Lower score to indicate it's a complement
                        explanation="Added as correct SQL for error case"
                    )
                    additional_candidates.append(candidate)
                    existing_keys.add(record_key)
                    
                    logger.debug(f"Added correct SQL record for error SQL: "
                               f"question={normalized_question[:50]}..., "
                               f"error_session={error_record.session_id}, error_round={error_record.round_id}, "
                               f"correct_session={record.session_id}, correct_round={record.round_id}")
                    break  # Add only the first matching record for each error record
        
        # Merge additional candidates with original candidates
        if additional_candidates:
            all_candidates = candidates + additional_candidates
            # Re-sort by score
            all_candidates.sort(key=lambda x: x.final_score, reverse=True)
            logger.debug(f"Added {len(additional_candidates)} correct SQL records for error SQLs in top-k")
            return all_candidates
        
        return candidates
    
    
