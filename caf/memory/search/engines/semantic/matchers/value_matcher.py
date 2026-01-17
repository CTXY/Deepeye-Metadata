# Unified Value Matcher - Enhanced value matching with inverted index and LSH

import logging
import importlib.util
import re
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass

from ..indexing.column_indexer import ColumnFacetProvider
from ..indexing.value_indexer import ColumnValueIndex

logger = logging.getLogger(__name__)

# Optional RapidFuzz acceleration for string similarity without try/except
_RAPIDFUZZ_AVAILABLE = importlib.util.find_spec("rapidfuzz") is not None
if _RAPIDFUZZ_AVAILABLE:
    from rapidfuzz import fuzz as _rf_fuzz

@dataclass
class ValueMatchResult:
    """Result from value matching"""
    column_id: str
    match_score: float
    match_type: str  # 'exact', 'partial', 'fuzzy'
    matched_value: str

class ValueMatcher:
    """
    Value Matcher with Inverted Index and LSH
    
    For low-cardinality columns: Uses inverted index (value -> [column1, column2...])
    For high-cardinality columns: Uses MinHash LSH for approximate matching
    
    This integrates the logic from the existing ValueMatcher with enhanced
    support for the anchor column retrieval workflow.
    """
    
    def __init__(self, cardinality_threshold: int = 100000):
        self.cardinality_threshold = cardinality_threshold
        
        # Inverted index for low-cardinality columns: normalized_value -> List[(column_id, original_value)]
        self.exact_index: Dict[str, List[Tuple[str, str]]] = {}
        
        # LSH index for high-cardinality columns
        self.lsh_index: Optional[Dict[str, Any]] = None
        self.high_cardinality_columns: Dict[str, List[str]] = {}  # column_id -> values
        
        logger.debug(f"UnifiedValueMatcher initialized with cardinality threshold: {cardinality_threshold}")
    
    def build_indexes(self, provider: ColumnFacetProvider, value_index: Optional[ColumnValueIndex] = None) -> None:
        """
        Build inverted index and LSH index from column provider
        
        Args:
            provider: Column facet provider for accessing metadata
        """
        logger.info("Building unified value matching indexes (provider/value-index-backed)...")
        
        # Reset indexes
        self.exact_index = {}
        self.high_cardinality_columns = {}
        
        # Separate low and high cardinality columns
        low_cardinality_data = []
        high_cardinality_data = []

        # If a pre-built value index is provided, prefer it for raw values;
        # otherwise fall back to provider.get_distinct_values (mainly for backward compatibility).
        for col_id in provider.iter_column_ids():
            _, t, c = provider.split_column_id(col_id)

            # Get regular distinct values from value index (preferred) or provider
            if value_index is not None:
                vals = value_index.get_values(col_id)
            else:
                vals = provider.get_distinct_values(t, c)
            
            # Get encoded values from encoding_mapping
            encoding_mapping = provider.get_encoding_mapping(t, c)
            encoded_vals = []
            
            # Add both keys and values from encoding mapping for searching
            if encoding_mapping:
                for key, value in encoding_mapping.items():
                    if key and str(key).strip():
                        encoded_vals.append(str(key).strip())
                    if value and str(value).strip():
                        encoded_vals.append(str(value).strip())
            
            # Combine regular values with encoded values
            all_vals = vals + encoded_vals
            
            if not all_vals:
                continue
                
            distinct_count = len(set(all_vals))  # Deduplicate all values
            # print(f"Column {c}")
            # print(f"Distinct count: {distinct_count}")
            if distinct_count < self.cardinality_threshold:
                # Low cardinality - use exact index
                low_cardinality_data.append((col_id, all_vals))
            else:
                # High cardinality - use LSH
                high_cardinality_data.append((col_id, all_vals))
                self.high_cardinality_columns[col_id] = all_vals
        
        # Build exact inverted index
        self._build_exact_index(low_cardinality_data)
        
        # Build LSH index 
        if high_cardinality_data:
            self._build_lsh_index(high_cardinality_data)

        
        logger.info(f"Value indexes built: {len(self.exact_index)} exact values, "
                   f"{len(self.high_cardinality_columns)} high-cardinality columns")
    
    def search(self, query: str, top_k: int = 10) -> List[ValueMatchResult]:
        """
        Search for value matches in the query
        
        Args:
            query: User query string
            
        Returns:
            List of ValueMatchResult objects
        """
        results = []

        # Exact matching via inverted index
        exact_results = self._exact_match(query)
        results.extend(exact_results)
        
        # Fuzzy matching via LSH for high-cardinality columns
        if self.lsh_index:
            fuzzy_results = self._fuzzy_match(query)
            results.extend(fuzzy_results)
        
        # Deduplicate and return best matches
        final_results = self._deduplicate_results(results)
        
        if len(final_results) > top_k:
            return final_results[:top_k]
        else:
            return final_results
    
    def _build_exact_index(self, low_cardinality_data: List[Tuple[str, List[str]]]) -> None:
        """Build inverted index for low-cardinality columns"""
        for column_id, values in low_cardinality_data:
            for value in values:
                if not value or not str(value).strip():
                    continue
                    
                # Normalize value
                value_key = str(value).lower().strip()
                
                if len(value_key) < 2:  # Skip very short values
                    continue
                
                if value_key not in self.exact_index:
                    self.exact_index[value_key] = []
                
                pair = (column_id, str(value))
                if pair not in self.exact_index[value_key]:
                    self.exact_index[value_key].append(pair)
        
        logger.debug(f"Built exact index with {len(self.exact_index)} unique values")
    
    def _build_lsh_index(self, high_cardinality_data: List[Tuple[str, List[str]]]) -> None:
        """Build MinHash LSH index for high-cardinality columns"""
        try:
            from datasketch import MinHashLSH, MinHash
            
            # LSH parameters
            threshold = 0.7  # Similarity threshold
            num_perm = 128   # Number of permutations
            
            lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
            minhashes = {}
            
            for column_id, values in high_cardinality_data:
                # Create MinHash signature for this column's value set
                m = MinHash(num_perm=num_perm)
                
                for value in values:
                    if value and str(value).strip():
                        m.update(str(value).encode('utf-8'))
                
                minhashes[column_id] = m
                lsh.insert(column_id, m)
            
            self.lsh_index = {
                'lsh': lsh,
                'minhashes': minhashes,
                'threshold': threshold,
                'num_perm': num_perm
            }
            
            logger.debug(f"Built LSH index with {len(minhashes)} high-cardinality columns")
            
        except ImportError:
            logger.warning("datasketch not installed, skipping LSH index for high-cardinality columns")
            self.lsh_index = None
        except Exception as e:
            logger.error(f"Failed to build LSH index: {e}")
            self.lsh_index = None
    
    
    def _exact_match(self, value: str) -> List[ValueMatchResult]:
        """Find exact and partial matches using inverted index returning concrete values"""
        results = []
        value_lower = value.lower().strip()
        
        # Direct exact match
        if value_lower in self.exact_index:
            for column_id, original_value in self.exact_index[value_lower]:
                results.append(ValueMatchResult(
                    column_id=column_id,
                    match_score=1.0,
                    match_type='exact',
                    matched_value=str(original_value)
                ))
        
        # Partial matching: check if query value contains any indexed value or vice versa
        partial_matches = []
        for indexed_value_key, column_value_pairs in self.exact_index.items():
            # Skip if this is the exact match we already processed
            if indexed_value_key == value_lower:
                continue
            
            # Skip very short indexed values to avoid noisy matches
            if len(indexed_value_key) < 3:
                continue
            
            # Check both directions for partial matching
            # 1. Query contains indexed value (e.g., "Unified School District schools" contains "Unified School District")
            # 2. Indexed value contains query (e.g., "Unified School District" contains "School")
            contains_match = indexed_value_key in value_lower or value_lower in indexed_value_key
            
            if contains_match:
                # Calculate similarity score based on length ratio and match quality
                min_len = min(len(indexed_value_key), len(value_lower))
                max_len = max(len(indexed_value_key), len(value_lower))
                
                if min_len >= 3 and max_len > 0:  # Ensure minimum meaningful length
                    overlap_ratio = min_len / max_len
                    
                    # Only consider partial matches with reasonable overlap (at least 40%) 
                    # and minimum absolute length of 3 characters
                    if overlap_ratio >= 0.4 and min_len >= 3:
                        partial_score = 0.7 * overlap_ratio  # Partial matches get lower score than exact
                        
                        for column_id, original_value in column_value_pairs:
                            partial_matches.append(ValueMatchResult(
                                column_id=column_id,
                                match_score=partial_score,
                                match_type='partial',
                                matched_value=str(original_value)
                            ))
        
        # Sort partial matches by score and limit to top matches to avoid performance issues
        if partial_matches:
            partial_matches.sort(key=lambda x: x.match_score, reverse=True)
            results.extend(partial_matches[:20])  # Limit to top 20 partial matches
        
        return results
    
    def _fuzzy_match(self, value: str, min_threshold: float = 0.5) -> List[ValueMatchResult]:
        """Find fuzzy matches using LSH index and then select concrete values from candidate columns"""
        if not self.lsh_index:
            return []
        
        results = []
        
        try:
            from datasketch import MinHash
            
            # Create MinHash for the query value
            m = MinHash(num_perm=self.lsh_index['num_perm'])
            m.update(value.encode('utf-8'))
            
            # Query LSH index
            lsh = self.lsh_index['lsh']
            candidate_columns = lsh.query(m)
            
            # Calculate actual similarities with candidates
            minhashes = self.lsh_index['minhashes']
            
            for column_id in candidate_columns:
                if column_id in minhashes:
                    similarity = m.jaccard(minhashes[column_id])
                    
                    if similarity >= min_threshold:
                        # Within this column, find the best concrete value similar to the query
                        best_value = None
                        best_score = 0.0
                        for candidate_value in self.high_cardinality_columns.get(column_id, []):
                            score = self._string_similarity(value, str(candidate_value))
                            if score > best_score:
                                best_score = score
                                best_value = str(candidate_value)
                        if best_value is not None and best_score >= min_threshold:
                            results.append(ValueMatchResult(
                                column_id=column_id,
                                match_score=float(best_score),
                                match_type='fuzzy',
                                matched_value=str(best_value)
                            ))
            
        except Exception as e:
            logger.error(f"Fuzzy matching failed for value '{value}': {e}")
        
        return results

    def _string_similarity(self, a: str, b: str) -> float:
        """Similarity score in [0,1]. Uses RapidFuzz if available, else token Jaccard."""
        a_norm = str(a).strip().lower()
        b_norm = str(b).strip().lower()
        if not a_norm or not b_norm:
            return 0.0
        if a_norm == b_norm:
            return 1.0

        if _RAPIDFUZZ_AVAILABLE:
            # RapidFuzz returns 0-100; convert to 0-1
            ratio = _rf_fuzz.ratio(a_norm, b_norm)
            partial = _rf_fuzz.partial_ratio(a_norm, b_norm)
            token_set = _rf_fuzz.token_set_ratio(a_norm, b_norm)
            score = max(ratio, partial, token_set) / 100.0
            return float(score)

        # Fallback: Token Jaccard
        a_tokens = set(re.findall(r"[a-z0-9]+", a_norm))
        b_tokens = set(re.findall(r"[a-z0-9]+", b_norm))
        if not a_tokens or not b_tokens:
            return 0.0
        inter = len(a_tokens & b_tokens)
        union = len(a_tokens | b_tokens)

        return inter / union if union else 0.0
    
    def _deduplicate_results(self, results: List[ValueMatchResult]) -> List[ValueMatchResult]:
        """Deduplicate results, keeping the best score for each (column, value) pair"""
        if not results:
            return []
        
        # Group by (column_id, matched_value), keep best score
        best_results: Dict[Tuple[str, str], ValueMatchResult] = {}
        
        for result in results:
            key = (result.column_id, result.matched_value)
            if (key not in best_results or 
                result.match_score > best_results[key].match_score):
                best_results[key] = result
        
        # Sort by match score (descending) 
        final_results = list(best_results.values())
        final_results.sort(key=lambda r: r.match_score, reverse=True)
        
        return final_results
