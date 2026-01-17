"""
Join Path Discovery - Discover potential join paths between tables using statistical features and MinHash.

This module discovers join relationships that are not explicitly defined as foreign keys,
by analyzing:
1. Statistical features (data shape, patterns)
2. MinHash sketches for value overlap with CONTAINMENT scoring (NOT Jaccard!)
3. LLM semantic verification with sample values

"""

import logging
import sqlite3
import json
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from datasketch import MinHash, MinHashLSH

from caf.llm.client import BaseLLMClient, LLMConfig, create_llm_client
from caf.config.global_config import get_llm_config
from caf.config.paths import PathConfig
from caf.prompt import PromptFactory

logger = logging.getLogger(__name__)

# Configuration constants
DISTINCT_EXACT_THRESHOLD = 10000  # Use exact values if distinct_count <= this
MINHASH_NUM_PERM = 256  # Number of permutations for MinHash
VALUE_SIMILARITY_THRESHOLD = 0.6  # Minimum containment similarity (NOT Jaccard!)
LLM_BATCH_SIZE = 5  # Number of pairs to verify in each LLM batch
MAX_SAMPLE_VALUES_FOR_PROMPT = 5  # Max sample values to show in LLM prompt


@dataclass
class ColumnKey:
    """Column identifier"""
    table_name: str
    column_name: str

    def to_id(self) -> str:
        return f"{self.table_name}.{self.column_name}"


@dataclass
class ColumnSignature:
    """Signature for a column's values and statistical features"""
    table_name: str
    column_name: str
    distinct_count: int
    value_mode: str  # "exact" or "minhash"
    exact_values: Optional[Set[str]] = None
    minhash: Optional[MinHash] = None
    sample_values: Optional[List[str]] = None  # Sample values for LLM prompt
    
    # Statistical features
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None
    fixed_prefix: Optional[str] = None
    fixed_suffix: Optional[str] = None
    data_type: Optional[str] = None
    semantic_type: Optional[str] = None


@dataclass
class JoinCandidate:
    """Candidate join path between two columns"""
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    similarity_score: float  # MinHash Jaccard similarity
    statistical_features: Dict[str, Any]  # Shared statistical features
    llm_reasoning: Optional[str] = None  # LLM's reasoning for accepting this join


@dataclass
class SimilarColumnPair:
    """Similar columns that share values but are not join keys"""
    database_id: str
    table_a: str
    column_a: str
    table_b: str
    column_b: str
    similarity_score: float
    similarity_reason: str  # e.g., "boolean_flag", "low_cardinality_attribute"
    shared_features: Dict[str, Any]


class JoinPathDiscovery:
    """
    Discover potential join paths between tables using:
    1. Statistical feature analysis (data shape, patterns)
    2. MinHash sketches for value overlap detection
    3. LLM semantic verification
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Configuration
        similarity_cfg = self.config.get("similarity", {})
        self.distinct_exact_threshold = similarity_cfg.get(
            "distinct_exact_threshold", DISTINCT_EXACT_THRESHOLD
        )
        self.minhash_num_perm = similarity_cfg.get(
            "minhash_num_perm", MINHASH_NUM_PERM
        )
        self.value_similarity_threshold = similarity_cfg.get(
            "value_similarity_threshold", VALUE_SIMILARITY_THRESHOLD
        )
        self.llm_batch_size = similarity_cfg.get("llm_batch_size", LLM_BATCH_SIZE)
        
        # Database mapping cache
        self._dbid_to_path: Optional[Dict[str, str]] = None
        default_mapping = PathConfig.get_database_mapping_path()
        self.mapping_path: Path = Path(
            self.config.get("database_mapping_path", default_mapping)
        )
        
        # Initialize LLM client
        self.llm_client = self._init_llm_client()
        
        logger.debug("JoinPathDiscovery initialized")
    
    def _init_llm_client(self) -> BaseLLMClient:
        """Initialize LLM client for semantic verification"""
        try:
            global_llm_config = get_llm_config()
            llm_config = LLMConfig(
                provider=global_llm_config.provider,
                model_name='gpt-4o',
                api_key=global_llm_config.api_key,
                base_url=global_llm_config.base_url,
                temperature=global_llm_config.temperature,
                max_tokens=global_llm_config.max_tokens,
                timeout=global_llm_config.timeout,
            )
            logger.debug("Using LLM config from global config")
            return create_llm_client(llm_config)
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}. Join path discovery will skip LLM verification.")
            return None
    
    def discover_join_paths(
        self,
        database_path: str,
        database_id: str,
        column_df: pd.DataFrame,
        exclude_fk_relationships: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Discover potential join paths for a database.
        
        Args:
            database_path: Path to database file
            database_id: Database identifier
            column_df: DataFrame with column metadata (from semantic store)
            exclude_fk_relationships: If True, exclude relationships already defined as FK
            
        Returns:
            List of RelationshipMetadata dictionaries
        """
        logger.info("Starting join path discovery for database: %s", database_id)
        
        if column_df is None or column_df.empty:
            logger.warning("No column metadata available")
            return []
        
        # Get column keys
        column_keys = self._get_column_keys(column_df)
        if not column_keys:
            logger.warning("No columns eligible for join path discovery")
            return []
        
        # Load existing FK relationships if excluding them
        existing_fk_relationships: Set[Tuple[str, str, str, str]] = set()
        if exclude_fk_relationships:
            existing_fk_relationships = self._load_fk_relationships(database_path)
        
        # Step 1: Extract column signatures (statistical features + MinHash)
        logger.info("Step 1: Extracting column signatures...")
        signatures = self._extract_column_signatures(database_path, column_keys, column_df)
        
        if not signatures:
            logger.warning("No column signatures extracted")
            return []
        
        logger.info("Extracted signatures for %d columns", len(signatures))
        
        # Step 2: Find candidate join paths using MinHash similarity
        logger.info("Step 2: Finding candidate join paths...")
        candidates = self._find_candidate_joins(signatures, existing_fk_relationships)
        
        if not candidates:
            logger.info("No candidate join paths found")
            return []
        
        logger.info("Found %d candidate join paths", len(candidates))
        
        # DEBUG: Save Step 2 intermediate results
        self._save_step2_results(database_id, candidates)
        
        # Step 3: Verify with LLM (if available)
        if self.llm_client:
            logger.info("Step 3: Verifying candidates with LLM...")
            verified_candidates = self._verify_with_llm(candidates, column_df, signatures)
        else:
            logger.info("Step 3: Skipping LLM verification (LLM client not available)")
            verified_candidates = candidates
        
        # DEBUG: Save Step 3 intermediate results
        self._save_step3_results(database_id, verified_candidates, len(candidates))
        
        # Step 3.5: Extract and save similar column pairs (rejected join candidates)
        logger.info("Step 3.5: Extracting similar column pairs...")
        similar_pairs = self._extract_similar_columns(
            all_candidates=candidates,
            verified_join_candidates=verified_candidates,
            signatures=signatures,
            database_id=database_id
        )
        if similar_pairs:
            self._save_similar_columns(database_id, similar_pairs)
            logger.info(f"Found {len(similar_pairs)} similar column pairs (non-join relationships)")
        
        # Step 4: Convert to RelationshipMetadata format
        relationships = self._convert_to_relationship_metadata(verified_candidates, database_id, database_path)
        
        logger.info("Discovered %d join paths", len(relationships))
        return relationships
    
    def _get_column_keys(self, column_df: pd.DataFrame) -> List[ColumnKey]:
        """Extract column keys from column dataframe"""
        keys: List[ColumnKey] = []
        for _, row in column_df.iterrows():
            table_name = row.get("table_name")
            column_name = row.get("column_name")
            if not table_name or not column_name:
                continue
            keys.append(ColumnKey(table_name=table_name, column_name=column_name))
        return keys
    
    def _load_fk_relationships(self, database_path: str) -> Set[Tuple[str, str, str, str]]:
        """Load existing FK relationships from database"""
        fk_relationships = set()
        
        try:
            conn = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table_name in tables:
                try:
                    cursor = conn.execute(f"PRAGMA foreign_key_list(`{table_name}`)")
                    for fk_row in cursor:
                        source_table = table_name
                        source_column = fk_row[3]  # 'from'
                        target_table = fk_row[2]    # 'table'
                        target_column = fk_row[4]  # 'to'
                        
                        if source_table and source_column and target_table and target_column:
                            fk_relationships.add((source_table, source_column, target_table, target_column))
                except Exception as e:
                    logger.debug(f"Failed to load FK for table {table_name}: {e}")
                    continue
            
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to load FK relationships: {e}")
        
        return fk_relationships
    
    def _extract_column_signatures(
        self,
        database_path: str,
        column_keys: List[ColumnKey],
        column_df: pd.DataFrame
    ) -> Dict[str, ColumnSignature]:
        """Extract signatures (statistical features + MinHash) for all columns"""
        signatures: Dict[str, ColumnSignature] = {}
        
        # Build column metadata index
        col_metadata = {}
        for _, row in column_df.iterrows():
            col_id = f"{row['table_name']}.{row['column_name']}"
            col_metadata[col_id] = {
                'min_length': row.get('min_length'),
                'max_length': row.get('max_length'),
                'avg_length': row.get('avg_length'),
                'fixed_prefix': row.get('fixed_prefix'),
                'fixed_suffix': row.get('fixed_suffix'),
                'data_type': row.get('data_type'),
                'semantic_type': row.get('semantic_type'),
                'distinct_count': row.get('distinct_count'),
            }
        
        conn = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
        
        try:
            for col_key in column_keys:
                col_id = col_key.to_id()
                metadata = col_metadata.get(col_id, {})
                
                # Get distinct count
                distinct_count = metadata.get('distinct_count')
                if distinct_count is None or pd.isna(distinct_count):
                    # Calculate on the fly
                    try:
                        query = f"SELECT COUNT(DISTINCT `{col_key.column_name}`) FROM `{col_key.table_name}` WHERE `{col_key.column_name}` IS NOT NULL"
                        distinct_count = conn.execute(query).fetchone()[0]
                    except Exception as e:
                        logger.debug(f"Failed to get distinct count for {col_id}: {e}")
                        continue
                
                if distinct_count == 0:
                    continue
                
                # Extract values and build signature
                if distinct_count <= self.distinct_exact_threshold:
                    # Use exact values
                    try:
                        query = f"SELECT DISTINCT `{col_key.column_name}` FROM `{col_key.table_name}` WHERE `{col_key.column_name}` IS NOT NULL"
                        rows = conn.execute(query).fetchall()
                        exact_values = {str(row[0]) for row in rows if row[0] is not None}
                        
                        # Keep a few sample values for LLM prompt
                        sample_for_prompt = list(exact_values)[:MAX_SAMPLE_VALUES_FOR_PROMPT]
                        
                        signatures[col_id] = ColumnSignature(
                            table_name=col_key.table_name,
                            column_name=col_key.column_name,
                            distinct_count=int(distinct_count),
                            value_mode="exact",
                            exact_values=exact_values,
                            sample_values=sample_for_prompt,
                            min_length=metadata.get('min_length'),
                            max_length=metadata.get('max_length'),
                            avg_length=metadata.get('avg_length'),
                            fixed_prefix=metadata.get('fixed_prefix'),
                            fixed_suffix=metadata.get('fixed_suffix'),
                            data_type=metadata.get('data_type'),
                            semantic_type=metadata.get('semantic_type'),
                        )
                    except Exception as e:
                        logger.debug(f"Failed to extract exact values for {col_id}: {e}")
                        continue
                else:
                    # Use MinHash (sample values with RANDOM sampling to avoid bias)
                    try:
                        # CRITICAL FIX: Use ORDER BY RANDOM() to avoid sampling bias
                        # Without random sampling, we might miss joins if tables store data
                        # from different time periods or regions in different physical order
                        query = f"""
                            SELECT DISTINCT `{col_key.column_name}` 
                            FROM `{col_key.table_name}` 
                            WHERE `{col_key.column_name}` IS NOT NULL 
                            ORDER BY RANDOM() 
                            LIMIT 10000
                        """
                        rows = conn.execute(query).fetchall()
                        sample_values = [str(row[0]) for row in rows if row[0] is not None]
                        
                        mh = MinHash(num_perm=self.minhash_num_perm)
                        for v in sample_values:
                            mh.update(v.encode("utf-8"))
                        
                        # Keep a few sample values for LLM prompt
                        sample_for_prompt = sample_values[:MAX_SAMPLE_VALUES_FOR_PROMPT]
                        
                        signatures[col_id] = ColumnSignature(
                            table_name=col_key.table_name,
                            column_name=col_key.column_name,
                            distinct_count=int(distinct_count),
                            value_mode="minhash",
                            minhash=mh,
                            sample_values=sample_for_prompt,
                            min_length=metadata.get('min_length'),
                            max_length=metadata.get('max_length'),
                            avg_length=metadata.get('avg_length'),
                            fixed_prefix=metadata.get('fixed_prefix'),
                            fixed_suffix=metadata.get('fixed_suffix'),
                            data_type=metadata.get('data_type'),
                            semantic_type=metadata.get('semantic_type'),
                        )
                    except Exception as e:
                        logger.debug(f"Failed to extract MinHash for {col_id}: {e}")
                        continue
        
        finally:
            conn.close()
        
        return signatures
    
    def _should_skip_candidate(
        self, 
        sig_a: ColumnSignature, 
        sig_b: ColumnSignature,
        reason_tracker: Dict[str, int] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Early filtering: skip candidates that are unlikely to be valid join keys.
        
        Returns:
            (should_skip: bool, reason: str or None)
        """
        # Rule 1: Skip boolean-like columns (cardinality <= 2)
        # Boolean columns have only 0/1 or True/False, they are properties not keys
        if sig_a.distinct_count <= 2:
            reason = f"boolean_or_binary:{sig_a.table_name}.{sig_a.column_name}"
            if reason_tracker is not None:
                reason_tracker['boolean_columns'] = reason_tracker.get('boolean_columns', 0) + 1
            return True, reason
        if sig_b.distinct_count <= 2:
            reason = f"boolean_or_binary:{sig_b.table_name}.{sig_b.column_name}"
            if reason_tracker is not None:
                reason_tracker['boolean_columns'] = reason_tracker.get('boolean_columns', 0) + 1
            return True, reason
        
        # Rule 2: Skip very low-cardinality columns (cardinality <= 5)
        # These are likely categorical attributes (status, type, etc.), not keys
        if sig_a.distinct_count <= 5 or sig_b.distinct_count <= 5:
            reason = "low_cardinality_attribute"
            if reason_tracker is not None:
                reason_tracker['low_cardinality'] = reason_tracker.get('low_cardinality', 0) + 1
            return True, reason
        
        # Rule 3: Check naming patterns - skip obvious attribute columns
        attr_patterns = ['is', 'has', 'desc', 'description', 'note', 'comment', 'text', 'name']
        key_patterns = ['id', 'code', 'key', 'ref', 'uuid', 'num', 'number', 'index']
        # NEW: Add measurement/metric patterns
        metric_patterns = ['size', 'count', 'total', 'amount', 'quantity', 'price', 'cost', 
                          'rate', 'percent', 'ratio', 'weight', 'height', 'width', 'length',
                          'sum', 'avg', 'min', 'max', 'mean', 'median']
        # NEW: Add type/category attribute patterns - these are categorical attributes, NOT join keys
        type_patterns = ['type', 'category', 'class', 'kind', 'status', 'state', 'level', 
                        'grade', 'rank', 'tier', 'option', 'mode', 'style']
        
        def looks_like_attribute(col_name: str) -> bool:
            col_lower = col_name.lower()
            # Check prefix patterns (is*, has*, etc.)
            if any(col_lower.startswith(p) for p in ['is', 'has']):
                return True
            # Check if it contains descriptive words
            if any(p in col_lower for p in ['desc', 'description', 'note', 'comment', 'text']):
                return True
            # Check if it contains type/category patterns (e.g., "District Type", "School Type")
            # These are categorical attributes, NOT join keys
            if any(p in col_lower for p in type_patterns):
                return True
            return False
        
        def looks_like_type_attribute(col_name: str) -> bool:
            """Specifically check if column is a type/category attribute"""
            col_lower = col_name.lower()
            return any(p in col_lower for p in type_patterns)
        
        def looks_like_metric(col_name: str) -> bool:
            """Check if column name suggests it's a measurement/metric field"""
            col_lower = col_name.lower()
            return any(p in col_lower for p in metric_patterns)
        
        def looks_like_key(col_name: str) -> bool:
            col_lower = col_name.lower()
            return any(p in col_lower for p in key_patterns)
        
        a_is_attr = looks_like_attribute(sig_a.column_name)
        b_is_attr = looks_like_attribute(sig_b.column_name)
        a_is_key = looks_like_key(sig_a.column_name)
        b_is_key = looks_like_key(sig_b.column_name)
        a_is_metric = looks_like_metric(sig_a.column_name)
        b_is_metric = looks_like_metric(sig_b.column_name)
        a_is_type = looks_like_type_attribute(sig_a.column_name)
        b_is_type = looks_like_type_attribute(sig_b.column_name)
        
        # Rule 3c: STRICT - Skip if either column is a type/category attribute
        # Type attributes (e.g., "District Type", "School Type") are categorical properties, NOT join keys
        if a_is_type or b_is_type:
            reason = "type_attribute_not_join_key"
            if reason_tracker is not None:
                reason_tracker['type_attribute'] = reason_tracker.get('type_attribute', 0) + 1
            return True, reason
        
        # Rule 3a: Skip if one is a metric field joining with an ID field
        # Example: baseSetSize (metric) joining with id (key) is INVALID
        if (a_is_metric and b_is_key) or (b_is_metric and a_is_key):
            reason = "metric_to_id_mismatch"
            if reason_tracker is not None:
                reason_tracker['metric_to_id'] = reason_tracker.get('metric_to_id', 0) + 1
            return True, reason
        
        # Rule 3b: Skip if both are metric fields (unless they are the exact same metric)
        if a_is_metric and b_is_metric:
            if sig_a.column_name.lower() != sig_b.column_name.lower():
                reason = "both_are_metrics"
                if reason_tracker is not None:
                    reason_tracker['both_metrics'] = reason_tracker.get('both_metrics', 0) + 1
                return True, reason
        
        # Both look like attributes → skip
        if a_is_attr and b_is_attr:
            reason = "both_are_attributes"
            if reason_tracker is not None:
                reason_tracker['both_attributes'] = reason_tracker.get('both_attributes', 0) + 1
            return True, reason
        
        # STRICT RULE: One is clearly an attribute → skip (even if the other looks like a key)
        # Attributes (like "District Type", "School Type") should NEVER be used as join keys
        # They are categorical properties, not identifiers
        if a_is_attr:
            reason = "attribute_column_not_join_key"
            if reason_tracker is not None:
                reason_tracker['attribute_mismatch'] = reason_tracker.get('attribute_mismatch', 0) + 1
            return True, reason
        if b_is_attr:
            reason = "attribute_column_not_join_key"
            if reason_tracker is not None:
                reason_tracker['attribute_mismatch'] = reason_tracker.get('attribute_mismatch', 0) + 1
            return True, reason
        
        return False, None
    
    def _find_candidate_joins(
        self,
        signatures: Dict[str, ColumnSignature],
        existing_fk_relationships: Set[Tuple[str, str, str, str]]
    ) -> List[JoinCandidate]:
        """
        Find candidate join paths using MinHash similarity with LSH optimization.
        Complexity: O(N²) naive → near O(N) with LSH indexing.
        """
        candidates: List[JoinCandidate] = []
        
        # Track filtering statistics
        self._filter_stats = {
            'boolean_columns': 0,
            'low_cardinality': 0,
            'both_attributes': 0,
            'attribute_mismatch': 0,
            'metric_to_id': 0,
            'both_metrics': 0,
            'type_attribute': 0
        }
        
        # Separate signatures by mode
        exact_sigs = {k: v for k, v in signatures.items() if v.value_mode == "exact"}
        minhash_sigs = {k: v for k, v in signatures.items() if v.value_mode == "minhash"}
        
        # Process MinHash signatures with LSH indexing (optimization for large databases)
        if minhash_sigs:
            logger.debug(f"Building LSH index for {len(minhash_sigs)} MinHash signatures...")
            lsh_candidates = self._find_candidates_with_lsh(minhash_sigs, existing_fk_relationships)
            candidates.extend(lsh_candidates)
        
        if exact_sigs:
            logger.debug(f"Processing {len(exact_sigs)} exact value signatures...")
            exact_candidates = self._find_candidates_bruteforce(exact_sigs, existing_fk_relationships)
            candidates.extend(exact_candidates)
        
        # Cross-mode comparison (exact vs minhash)
        if exact_sigs and minhash_sigs:
            logger.debug(f"Processing cross-mode comparisons...")
            cross_candidates = self._find_candidates_cross_mode(
                exact_sigs, minhash_sigs, existing_fk_relationships
            )
            candidates.extend(cross_candidates)
        
        # Log filtering statistics
        logger.info(
            f"Early filtering removed: "
            f"boolean={self._filter_stats.get('boolean_columns', 0)}, "
            f"low_cardinality={self._filter_stats.get('low_cardinality', 0)}, "
            f"metric_to_id={self._filter_stats.get('metric_to_id', 0)}, "
            f"both_metrics={self._filter_stats.get('both_metrics', 0)}, "
            f"both_attributes={self._filter_stats.get('both_attributes', 0)}, "
            f"attribute_mismatch={self._filter_stats.get('attribute_mismatch', 0)}, "
            f"type_attribute={self._filter_stats.get('type_attribute', 0)}"
        )
        
        return candidates
    
    def _find_candidates_with_lsh(
        self,
        signatures: Dict[str, ColumnSignature],
        existing_fk_relationships: Set[Tuple[str, str, str, str]]
    ) -> List[JoinCandidate]:
        """Use LSH indexing to find similar MinHash signatures efficiently"""
        candidates: List[JoinCandidate] = []
        
        # Build LSH index
        # Use Jaccard threshold for LSH, but we'll recompute containment later
        lsh = MinHashLSH(threshold=0.3, num_perm=self.minhash_num_perm)
        
        for col_id, sig in signatures.items():
            if sig.minhash:
                lsh.insert(col_id, sig.minhash)
        
        # Query LSH for each signature
        checked_pairs = set()
        for col_id_a, sig_a in signatures.items():
            if not sig_a.minhash:
                continue
            
            # Query LSH for similar columns
            similar_col_ids = lsh.query(sig_a.minhash)
            
            for col_id_b in similar_col_ids:
                if col_id_a == col_id_b:
                    continue
                
                # Avoid duplicate comparisons
                pair_key = tuple(sorted([col_id_a, col_id_b]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)
                
                sig_b = signatures[col_id_b]
                table_a = sig_a.table_name
                table_b = sig_b.table_name
                
                # Skip same table pairs
                if table_a == table_b:
                    continue
                
                # Skip if already an FK relationship
                if (table_a, sig_a.column_name, table_b, sig_b.column_name) in existing_fk_relationships or \
                   (table_b, sig_b.column_name, table_a, sig_a.column_name) in existing_fk_relationships:
                    continue
                
                # EARLY FILTERING: Skip unlikely join key candidates
                should_skip, skip_reason = self._should_skip_candidate(sig_a, sig_b, self._filter_stats)
                if should_skip:
                    continue
                
                # Compute CONTAINMENT (not Jaccard returned by LSH!)
                similarity = self._compute_similarity(sig_a, sig_b)
                
                # Check statistical feature compatibility
                shared_features = self._extract_shared_features(sig_a, sig_b)
                
                # Accept candidate with flexible matching
                accept = False
                if similarity and similarity >= self.value_similarity_threshold:
                    accept = True
                elif similarity and similarity >= 0.3 and shared_features:
                    accept = True
                
                if accept:
                    candidates.append(JoinCandidate(
                        source_table=table_a,
                        source_column=sig_a.column_name,
                        target_table=table_b,
                        target_column=sig_b.column_name,
                        similarity_score=similarity,
                        statistical_features=shared_features
                    ))
        
        return candidates
    
    def _find_candidates_bruteforce(
        self,
        signatures: Dict[str, ColumnSignature],
        existing_fk_relationships: Set[Tuple[str, str, str, str]]
    ) -> List[JoinCandidate]:
        """Brute force comparison for exact value signatures (small sets)"""
        candidates: List[JoinCandidate] = []
        sig_list = list(signatures.items())
        
        for i in range(len(sig_list)):
            col_id_a, sig_a = sig_list[i]
            table_a = sig_a.table_name
            
            for j in range(i + 1, len(sig_list)):
                col_id_b, sig_b = sig_list[j]
                table_b = sig_b.table_name
                
                # Skip same table pairs
                if table_a == table_b:
                    continue
                
                # Skip if already an FK relationship
                if (table_a, sig_a.column_name, table_b, sig_b.column_name) in existing_fk_relationships or \
                   (table_b, sig_b.column_name, table_a, sig_a.column_name) in existing_fk_relationships:
                    continue
                
                # EARLY FILTERING: Skip unlikely join key candidates
                should_skip, skip_reason = self._should_skip_candidate(sig_a, sig_b, self._filter_stats)
                if should_skip:
                    continue
                
                # Compute containment similarity
                similarity = self._compute_similarity(sig_a, sig_b)
                
                # Check statistical feature compatibility
                shared_features = self._extract_shared_features(sig_a, sig_b)
                
                # Accept candidate with flexible matching
                accept = False
                if similarity and similarity >= self.value_similarity_threshold:
                    accept = True
                elif similarity and similarity >= 0.3 and shared_features:
                    accept = True
                
                if accept:
                    candidates.append(JoinCandidate(
                        source_table=table_a,
                        source_column=sig_a.column_name,
                        target_table=table_b,
                        target_column=sig_b.column_name,
                        similarity_score=similarity,
                        statistical_features=shared_features
                    ))
        
        return candidates
    
    def _find_candidates_cross_mode(
        self,
        exact_sigs: Dict[str, ColumnSignature],
        minhash_sigs: Dict[str, ColumnSignature],
        existing_fk_relationships: Set[Tuple[str, str, str, str]]
    ) -> List[JoinCandidate]:
        """Compare exact vs minhash signatures (cross-mode)"""
        candidates: List[JoinCandidate] = []
        
        for col_id_a, sig_a in exact_sigs.items():
            table_a = sig_a.table_name
            
            for col_id_b, sig_b in minhash_sigs.items():
                table_b = sig_b.table_name
                
                # Skip same table pairs
                if table_a == table_b:
                    continue
                
                # Skip if already an FK relationship
                if (table_a, sig_a.column_name, table_b, sig_b.column_name) in existing_fk_relationships or \
                   (table_b, sig_b.column_name, table_a, sig_a.column_name) in existing_fk_relationships:
                    continue
                
                # EARLY FILTERING: Skip unlikely join key candidates
                should_skip, skip_reason = self._should_skip_candidate(sig_a, sig_b, self._filter_stats)
                if should_skip:
                    continue
                
                # Compute containment similarity
                similarity = self._compute_similarity(sig_a, sig_b)
                
                # Check statistical feature compatibility
                shared_features = self._extract_shared_features(sig_a, sig_b)
                
                # Accept candidate with flexible matching
                accept = False
                if similarity and similarity >= self.value_similarity_threshold:
                    accept = True
                elif similarity and similarity >= 0.3 and shared_features:
                    accept = True
                
                if accept:
                    candidates.append(JoinCandidate(
                        source_table=table_a,
                        source_column=sig_a.column_name,
                        target_table=table_b,
                        target_column=sig_b.column_name,
                        similarity_score=similarity,
                        statistical_features=shared_features
                    ))
        
        return candidates
    
    def _compute_similarity(self, sig1: ColumnSignature, sig2: ColumnSignature) -> Optional[float]:
        """
        Compute CONTAINMENT score (NOT Jaccard!) between two signatures.
        
        For PK-FK detection, we need Containment = |A ∩ B| / min(|A|, |B|)
        This detects when one column's values are contained in another.
        
        Example: Order.customer_id (1M rows) → Customer.id (1K rows)
        - Jaccard = 1K / 1M ≈ 0.001 ❌ (misses the join!)
        - Containment = 1K / 1K = 1.0 ✓ (detects the join!)
        """
        # Case 1: Both exact
        if sig1.value_mode == "exact" and sig2.value_mode == "exact":
            if sig1.exact_values is None or sig2.exact_values is None:
                return None
            intersection = len(sig1.exact_values & sig2.exact_values)
            min_size = min(len(sig1.exact_values), len(sig2.exact_values))
            if min_size == 0:
                return 0.0
            # Containment: what fraction of the smaller set is in the intersection?
            return intersection / min_size
        
        # Case 2: Both MinHash
        if sig1.value_mode == "minhash" and sig2.value_mode == "minhash":
            if sig1.minhash is None or sig2.minhash is None:
                return None
            # Estimate containment using MinHash hash values
            hashes1 = set(sig1.minhash.hashvalues)
            hashes2 = set(sig2.minhash.hashvalues)
            intersection_size = len(hashes1 & hashes2)
            min_hash_size = min(len(hashes1), len(hashes2))
            if min_hash_size == 0:
                return 0.0
            # Approximate containment score
            return intersection_size / min_hash_size
        
        # Case 3: Mixed (exact + minhash)
        exact_sig = sig1 if sig1.value_mode == "exact" else sig2
        minhash_sig = sig2 if sig1.value_mode == "exact" else sig1
        
        if exact_sig.exact_values is None or minhash_sig.minhash is None:
            return None
        
        # Create temporary MinHash for exact set
        temp_mh = MinHash(num_perm=self.minhash_num_perm)
        for v in exact_sig.exact_values:
            temp_mh.update(v.encode("utf-8"))
        
        # Compute containment
        hashes1 = set(temp_mh.hashvalues)
        hashes2 = set(minhash_sig.minhash.hashvalues)
        intersection_size = len(hashes1 & hashes2)
        min_hash_size = min(len(hashes1), len(hashes2))
        if min_hash_size == 0:
            return 0.0
        return intersection_size / min_hash_size
    
    def _extract_shared_features(
        self,
        sig1: ColumnSignature,
        sig2: ColumnSignature
    ) -> Dict[str, Any]:
        """
        Extract shared statistical features between two signatures.
        FLEXIBLE matching to handle data transformations (e.g., IMEI with/without prefix).
        """
        features = {}
        
        # Length compatibility - FLEXIBLE: allow small differences (1-2 chars)
        # This handles cases like IMEI (13 vs 14 digits with '1' prefix)
        if sig1.min_length and sig2.min_length:
            if sig1.min_length == sig2.min_length:
                features['same_min_length'] = sig1.min_length
            elif abs(sig1.min_length - sig2.min_length) <= 2:
                features['similar_min_length'] = f"{sig1.min_length} vs {sig2.min_length}"
        
        if sig1.max_length and sig2.max_length:
            if sig1.max_length == sig2.max_length:
                features['same_max_length'] = sig1.max_length
            elif abs(sig1.max_length - sig2.max_length) <= 2:
                features['similar_max_length'] = f"{sig1.max_length} vs {sig2.max_length}"
        
        # Prefix/suffix compatibility
        if sig1.fixed_prefix and sig2.fixed_prefix:
            if sig1.fixed_prefix == sig2.fixed_prefix:
                features['shared_prefix'] = sig1.fixed_prefix
        if sig1.fixed_suffix and sig2.fixed_suffix:
            if sig1.fixed_suffix == sig2.fixed_suffix:
                features['shared_suffix'] = sig1.fixed_suffix
        
        # Data type compatibility
        if sig1.data_type and sig2.data_type:
            if sig1.data_type == sig2.data_type:
                features['same_data_type'] = sig1.data_type
        
        # Semantic type compatibility
        if sig1.semantic_type and sig2.semantic_type:
            if sig1.semantic_type == sig2.semantic_type:
                features['same_semantic_type'] = sig1.semantic_type
        
        return features
    
    def _verify_with_llm(
        self,
        candidates: List[JoinCandidate],
        column_df: pd.DataFrame,
        signatures: Dict[str, ColumnSignature]
    ) -> List[JoinCandidate]:
        """Verify candidate join paths with LLM"""
        if not self.llm_client or not candidates:
            return candidates
        
        # Build column metadata index
        col_metadata = {}
        for _, row in column_df.iterrows():
            col_id = f"{row['table_name']}.{row['column_name']}"
            col_metadata[col_id] = {
                'table_name': row.get('table_name'),
                'column_name': row.get('column_name'),
                'whole_column_name': row.get('whole_column_name', ''),
                'short_description': row.get('short_description', ''),
                'data_type': row.get('data_type', ''),
                'data_format': row.get('data_format', ''),
                'is_primary_key': row.get('is_primary_key', False),
            }
        
        verified: List[JoinCandidate] = []
        
        # Process in batches
        for i in range(0, len(candidates), self.llm_batch_size):
            batch = candidates[i:i + self.llm_batch_size]
            logger.debug("Verifying batch %d/%d", i // self.llm_batch_size + 1,
                        (len(candidates) + self.llm_batch_size - 1) // self.llm_batch_size)
            
            batch_verified = self._verify_batch_with_llm(batch, col_metadata, signatures)
            verified.extend(batch_verified)
        
        return verified
    
    def _verify_batch_with_llm(
        self,
        batch: List[JoinCandidate],
        col_metadata: Dict[str, Dict[str, Any]],
        signatures: Dict[str, ColumnSignature]
    ) -> List[JoinCandidate]:
        """Verify a batch of candidates with LLM"""
        prompt = self._build_verification_prompt(batch, col_metadata, signatures)
        
        print('-'*50)
        print(prompt)
        
        try:
            response = self.llm_client.call_with_messages(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            
            # Parse response
            verified = self._parse_llm_response(batch, response)
            return verified
        except Exception as e:
            logger.error("LLM verification failed: %s", e)
            # On error, return all candidates (conservative approach)
            return batch
    
    def _build_verification_prompt(
        self,
        batch: List[JoinCandidate],
        col_metadata: Dict[str, Dict[str, Any]],
        signatures: Dict[str, ColumnSignature] = None
    ) -> str:
        """Build LLM prompt for batch verification with sample values and strict semantic rules"""
        # Convert candidates to dict format for PromptFactory
        candidates = []
        for candidate in batch:
            col_a_id = f"{candidate.source_table}.{candidate.source_column}"
            col_b_id = f"{candidate.target_table}.{candidate.target_column}"
            
            meta_a = col_metadata.get(col_a_id, {})
            meta_b = col_metadata.get(col_b_id, {})
            
            # Prepare signature data
            sig_a = signatures.get(col_a_id) if signatures else None
            sig_b = signatures.get(col_b_id) if signatures else None
            
            # Build candidate dict
            candidate_dict = {
                'source_table': candidate.source_table,
                'source_column': candidate.source_column,
                'target_table': candidate.target_table,
                'target_column': candidate.target_column,
                'similarity_score': candidate.similarity_score,
                'statistical_features': candidate.statistical_features if hasattr(candidate, 'statistical_features') else None
            }
            
            candidates.append(candidate_dict)
        
        # Use PromptFactory to format the prompt
        # PromptFactory uses getattr, so it can handle ColumnSignature objects directly
        return PromptFactory.format_join_path_verification_prompt(
            candidates=candidates,
            col_metadata=col_metadata,
            signatures=signatures
        )
    
    def _format_column_description(self, metadata: Dict[str, Any]) -> str:
        """Format column description for LLM prompt"""
        parts = []
        
        table_name = metadata.get('table_name', '')
        column_name = metadata.get('column_name', '')
        whole_name = metadata.get('whole_column_name', '')
        name_to_use = whole_name if whole_name else column_name
        parts.append(f"{table_name}.{name_to_use}")
        
        # Add short_description first (contains global context)
        short_desc = metadata.get('short_description', '')
        if short_desc and str(short_desc).lower() not in ["nan", "none", "null", ""]:
            parts.append(f"Short Description: {short_desc}")
        
        
        data_type = metadata.get('data_type', '')
        if data_type and str(data_type).lower() not in ["nan", "none", "null", ""]:
            parts.append(f"Type: {data_type}")
        
        # Add PRIMARY KEY indicator
        is_pk = metadata.get('is_primary_key', False)
        if is_pk:
            parts.append("PRIMARY KEY")
        
        return " | ".join(parts)
    
    def _parse_llm_response(
        self,
        batch: List[JoinCandidate],
        response: str
    ) -> List[JoinCandidate]:
        """Parse LLM response and return verified candidates"""
        verified: List[JoinCandidate] = []
        
        try:
            import json
            import re
            
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
                response = response.strip()
            
            if response.startswith("["):
                results = json.loads(response)
            else:
                json_match = re.search(r"\[.*\]", response, re.DOTALL)
                if json_match:
                    results = json.loads(json_match.group())
                else:
                    logger.warning("Could not parse LLM response as JSON")
                    return batch  # Return all if parsing fails
            
            if not isinstance(results, list):
                logger.warning("LLM response is not a list")
                return batch
            
            if len(results) != len(batch):
                logger.warning("LLM response length does not match batch size")
                results = results[:len(batch)]
            
            for i, result in enumerate(results):
                if i >= len(batch):
                    break
                
                # Handle both old and new formats
                if isinstance(result, dict):
                    decision = result.get('decision', '').strip().lower()
                    reasoning = result.get('reasoning', '')
                    candidate_id = result.get('candidate_id', i + 1)
                    
                    # Accept "Yes" or "A" (for backward compatibility)
                    if decision in ["yes", "a"]:
                        candidate = batch[i]
                        candidate.llm_reasoning = reasoning
                        verified.append(candidate)
                        logger.debug(f"Candidate {candidate_id} ACCEPTED: {candidate.source_table}.{candidate.source_column} → {candidate.target_table}.{candidate.target_column}")
                    else:
                        logger.debug(f"Candidate {candidate_id} REJECTED: {reasoning}")
                else:
                    # Fallback to old format
                    result_str = str(result).strip().upper()
                    if result_str == "A":
                        verified.append(batch[i])
        
        except Exception as e:
            logger.error("Failed to parse LLM response: %s", e)
            return batch  # Return all if parsing fails
        
        return verified
    
    def _convert_to_relationship_metadata(
        self,
        candidates: List[JoinCandidate],
        database_id: str,
        database_path: str
    ) -> List[Dict[str, Any]]:
        """Convert verified candidates to RelationshipMetadata format"""
        relationships = []
        
        for candidate in candidates:
            relationship = {
                'database_id': database_id,
                'relationship_type': 'value_overlap_join',  # New relationship type
                'source_table': candidate.source_table,
                'target_table': candidate.target_table,
                'source_columns': [candidate.source_column],
                'target_columns': [candidate.target_column],
                'source': 'join_path_discovery',
            }
            
            # Compute cardinality by executing actual JOIN query
            cardinality = self._compute_relationship_cardinality(
                database_path,
                candidate.source_table,
                [candidate.source_column],
                candidate.target_table,
                [candidate.target_column]
            )
            if cardinality:
                relationship['cardinality'] = cardinality
            
            relationships.append(relationship)
        
        return relationships
    
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
            conn = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
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
    
    def _save_step2_results(self, database_id: str, candidates: List[JoinCandidate]) -> None:
        """DEBUG: Save Step 2 intermediate results (candidates before LLM verification)"""
        output_dir = Path("output/join_path_debug")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{database_id}_step2_candidates.json"
        
        # Convert JoinCandidate objects to dictionaries
        candidates_data = []
        for candidate in candidates:
            candidate_dict = {
                'source_table': candidate.source_table,
                'source_column': candidate.source_column,
                'target_table': candidate.target_table,
                'target_column': candidate.target_column,
                'similarity_score': candidate.similarity_score,
                'statistical_features': candidate.statistical_features
            }
            candidates_data.append(candidate_dict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'database_id': database_id,
                'total_candidates': len(candidates_data),
                'candidates': candidates_data
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"DEBUG: Saved Step 2 results to {output_file} ({len(candidates_data)} candidates)")
    
    def _save_step3_results(self, database_id: str, verified_candidates: List[JoinCandidate], total_before: int) -> None:
        """DEBUG: Save Step 3 intermediate results (candidates after LLM verification)"""
        output_dir = Path("output/join_path_debug")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{database_id}_step3_verified.json"
        
        # Convert JoinCandidate objects to dictionaries
        verified_data = []
        for candidate in verified_candidates:
            candidate_dict = {
                'source_table': candidate.source_table,
                'source_column': candidate.source_column,
                'target_table': candidate.target_table,
                'target_column': candidate.target_column,
                'similarity_score': candidate.similarity_score,
                'statistical_features': candidate.statistical_features
            }
            verified_data.append(candidate_dict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'database_id': database_id,
                'total_before_verification': total_before,
                'total_after_verification': len(verified_data),
                'filtered_out': total_before - len(verified_data),
                'verified_candidates': verified_data
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"DEBUG: Saved Step 3 results to {output_file} ({len(verified_data)}/{total_before} candidates verified)")
    
    def _classify_similarity_reason(
        self, 
        candidate: JoinCandidate,
        sig_a: Optional[ColumnSignature],
        sig_b: Optional[ColumnSignature]
    ) -> str:
        """Classify why two columns are similar (for documentation)"""
        # Check boolean/binary
        if sig_a and sig_a.distinct_count <= 2:
            return f"boolean_or_binary:{sig_a.table_name}.{sig_a.column_name}"
        if sig_b and sig_b.distinct_count <= 2:
            return f"boolean_or_binary:{sig_b.table_name}.{sig_b.column_name}"
        
        # Check low cardinality
        if sig_a and sig_a.distinct_count <= 5:
            return f"low_cardinality:{sig_a.distinct_count}_values"
        if sig_b and sig_b.distinct_count <= 5:
            return f"low_cardinality:{sig_b.distinct_count}_values"
        
        # Check data type from statistical features
        if candidate.statistical_features.get('same_data_type') == 'BOOLEAN':
            return "same_boolean_flag"
        
        # Check semantic type
        if 'same_semantic_type' in candidate.statistical_features:
            return f"same_semantic_type:{candidate.statistical_features['same_semantic_type']}"
        
        # Check naming patterns
        def is_attribute_name(name: str) -> bool:
            name_lower = name.lower()
            return any(name_lower.startswith(p) for p in ['is', 'has', 'desc'])
        
        def is_type_attribute(name: str) -> bool:
            """Check if column name contains type/category patterns"""
            name_lower = name.lower()
            type_patterns = ['type', 'category', 'class', 'kind', 'status', 'state', 'level', 
                            'grade', 'rank', 'tier', 'option', 'mode', 'style']
            return any(p in name_lower for p in type_patterns)
        
        # Check for type attributes first (more specific)
        if is_type_attribute(candidate.source_column) or is_type_attribute(candidate.target_column):
            return "type_attribute_not_join_key"
        
        if is_attribute_name(candidate.source_column) or is_attribute_name(candidate.target_column):
            return "attribute_naming_pattern"
        
        return "value_overlap"
    
    def _extract_similar_columns(
        self,
        all_candidates: List[JoinCandidate],
        verified_join_candidates: List[JoinCandidate],
        signatures: Dict[str, ColumnSignature],
        database_id: str
    ) -> List[SimilarColumnPair]:
        """Extract similar column pairs that were rejected as join keys"""
        verified_set = {
            (c.source_table, c.source_column, c.target_table, c.target_column)
            for c in verified_join_candidates
        }
        
        similar_pairs = []
        for candidate in all_candidates:
            pair_key = (candidate.source_table, candidate.source_column, 
                       candidate.target_table, candidate.target_column)
            
            if pair_key not in verified_set:
                # This is a similar column pair, not a join key
                col_a_id = f"{candidate.source_table}.{candidate.source_column}"
                col_b_id = f"{candidate.target_table}.{candidate.target_column}"
                sig_a = signatures.get(col_a_id)
                sig_b = signatures.get(col_b_id)
                
                reason = self._classify_similarity_reason(candidate, sig_a, sig_b)
                similar_pairs.append(SimilarColumnPair(
                    database_id=database_id,
                    table_a=candidate.source_table,
                    column_a=candidate.source_column,
                    table_b=candidate.target_table,
                    column_b=candidate.target_column,
                    similarity_score=candidate.similarity_score,
                    similarity_reason=reason,
                    shared_features=candidate.statistical_features
                ))
        
        return similar_pairs
    
    def _save_similar_columns(self, database_id: str, similar_pairs: List[SimilarColumnPair]) -> None:
        """Save similar column pairs to file for later analysis"""
        output_dir = Path("output/similar_columns")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{database_id}_similar_columns.json"
        
        # Convert to dictionaries
        similar_data = []
        for pair in similar_pairs:
            pair_dict = {
                'database_id': pair.database_id,
                'table_a': pair.table_a,
                'column_a': pair.column_a,
                'table_b': pair.table_b,
                'column_b': pair.column_b,
                'similarity_score': pair.similarity_score,
                'similarity_reason': pair.similarity_reason,
                'shared_features': pair.shared_features
            }
            similar_data.append(pair_dict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'database_id': database_id,
                'total_similar_pairs': len(similar_data),
                'similar_column_pairs': similar_data
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(similar_data)} similar column pairs to {output_file}")















