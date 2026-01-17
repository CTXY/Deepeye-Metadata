"""
Value Overlap Cluster Miner - Discover ambiguous columns based on value overlap.

This miner identifies columns with high value overlap (Jaccard similarity) 
without considering column names, focusing purely on the data values.
"""

import logging
import sqlite3
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union

import pandas as pd
from datasketch import MinHash

from ..stores.semantic import SemanticMemoryStore
from ..stores.ambiguous_pair import AmbiguousPairStore
from ..types import AmbiguousPair, DBElementRef
from ...config.paths import PathConfig

logger = logging.getLogger(__name__)

# Configuration constants
DISTINCT_EXACT_THRESHOLD = 10000  # Use exact values if distinct_count <= this
VALUE_JACCARD_THRESHOLD = 0.6  # Minimum Jaccard similarity to consider columns similar
MINHASH_NUM_PERM = 256  # Number of permutations for MinHash


@dataclass
class ColumnKey:
    """Column identifier"""
    table_name: str
    column_name: str

    def to_id(self) -> str:
        return f"{self.table_name}.{self.column_name}"


@dataclass
class ColumnValuesSignature:
    """Signature for a column's values (exact set or MinHash)"""
    table_name: str
    column_name: str
    distinct_count: int
    value_mode: str  # "exact" or "minhash"
    exact_values: Optional[Set[str]] = None  # Only when value_mode == "exact"
    minhash: Optional[MinHash] = None  # Only when value_mode == "minhash"


class ValueOverlapClusterMiner:
    """
    Miner for discovering value-overlap pairs within a single database.
    
    This miner identifies columns with high value overlap (Jaccard similarity >= 0.6)
    by directly querying the database and comparing value sets. It uses:
    - Exact value sets for low-cardinality columns (distinct_count <= 10000)
    - MinHash approximation for high-cardinality columns (distinct_count > 10000)
    
    Responsibilities:
    - Query database to extract distinct values for each column
    - Compute Jaccard similarity between column value sets
    - Build pairs of columns with high value overlap
    - Save pairs to AmbiguousPairStore
    """

    def __init__(
        self,
        semantic_store: SemanticMemoryStore,
        pair_store: AmbiguousPairStore,
        memory_config: Dict,
    ):
        self.semantic_store = semantic_store
        self.pair_store = pair_store
        self.memory_config = memory_config or {}
        
        # Configuration
        similarity_cfg = self.memory_config.get("similarity", {})
        self.distinct_exact_threshold = similarity_cfg.get(
            "distinct_exact_threshold", DISTINCT_EXACT_THRESHOLD
        )
        self.value_jaccard_threshold = similarity_cfg.get(
            "value_jaccard_threshold", VALUE_JACCARD_THRESHOLD
        )
        self.minhash_num_perm = similarity_cfg.get(
            "minhash_num_perm", MINHASH_NUM_PERM
        )

        # Database mapping cache
        self._dbid_to_path: Optional[Dict[str, str]] = None
        # Default mapping path is provided by PathConfig to avoid hardcoded paths
        default_mapping = PathConfig.get_database_mapping_path()
        self.mapping_path: Path = Path(
            self.memory_config.get("database_mapping_path", default_mapping)
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def mine_and_save_pairs(self, database_id: str) -> List[AmbiguousPair]:
        """
        High-level API:
        - Bind semantic & pair stores
        - Resolve database path from database_id
        - Extract column value signatures
        - Compute pairwise Jaccard similarities
        - Build pairs from high-overlap column pairs
        - Save results to pair store
        """
        logger.info(
            "Starting value overlap pair mining for database: %s", database_id
        )

        # Bind stores (check if already bound to avoid repeated loading)
        if self.semantic_store.current_database_id != database_id:
            self.semantic_store.bind_database(database_id)
        else:
            logger.debug("Semantic store already bound to %s, skipping bind", database_id)
            
        if self.pair_store.current_database_id != database_id:
            self.pair_store.bind_database(database_id)
        else:
            logger.debug("Pair store already bound to %s, skipping bind", database_id)

        # Get column list from semantic store
        column_df = self.semantic_store.dataframes.get("column")
        if column_df is None or column_df.empty:
            logger.warning(
                "No column metadata available for database: %s", database_id
            )
            pairs: List[AmbiguousPair] = []
            self.pair_store.save_pairs(database_id, pairs)
            return pairs

        # Resolve database path
        database_path = self._resolve_database_path(database_id)
        if not database_path:
            logger.error(
                "Failed to resolve database path for database_id: %s", database_id
            )
            pairs = []
            self.pair_store.save_pairs(database_id, pairs)
            return pairs

        # Extract column value signatures
        column_keys = self._get_column_keys(column_df)
        if not column_keys:
            logger.warning(
                "No columns eligible for value overlap mining in database: %s",
                database_id,
            )
            pairs = []
            self.pair_store.save_pairs(database_id, pairs)
            return pairs

        logger.info("Extracting value signatures for %d columns...", len(column_keys))
        signatures = self._extract_column_signatures(database_path, column_keys)

        if not signatures:
            logger.warning("No value signatures extracted")
            pairs = []
            self.pair_store.save_pairs(database_id, pairs)
            return pairs

        # Compute pairwise similarities
        logger.info("Computing pairwise Jaccard similarities...")
        high_overlap_pairs = self._find_high_overlap_pairs(signatures)

        if not high_overlap_pairs:
            logger.info(
                "No high-overlap column pairs found (threshold=%.2f) for database: %s",
                self.value_jaccard_threshold,
                database_id,
            )
            pairs = []
            self.pair_store.save_pairs(database_id, pairs)
            return pairs

        # Build AmbiguousPair objects from high-overlap pairs
        logger.info("Building pairs from %d high-overlap pairs...", len(high_overlap_pairs))
        pairs = self._build_pairs_from_overlaps(
            high_overlap_pairs, database_id
        )

        self.pair_store.save_pairs(database_id, pairs)

        logger.info(
            "Finished value overlap pair mining for database: %s (pairs=%d)",
            database_id,
            len(pairs),
        )
        return pairs

    # ------------------------------------------------------------------
    # Database path resolution
    # ------------------------------------------------------------------
    def _load_mapping(self) -> None:
        """Load database_id -> database_path mapping"""
        if self._dbid_to_path is not None:
            return

        self._dbid_to_path = {}
        if not self.mapping_path.exists():
            logger.warning("database_mapping.json not found at %s", self.mapping_path)
            return

        try:
            with self.mapping_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            # raw: {"/abs/path/to/db.sqlite": "database_id", ...}
            for abs_path, db_id in raw.items():
                if not db_id:
                    continue
                self._dbid_to_path[str(db_id)] = str(abs_path)
        except Exception as exc:
            logger.error(
                "Failed to load database mapping from %s: %s", self.mapping_path, exc
            )
            self._dbid_to_path = {}

    def _resolve_database_path(self, database_id: str) -> Optional[str]:
        """Resolve database_id to database_path using mapping"""
        self._load_mapping()
        if not self._dbid_to_path:
            return None
        return self._dbid_to_path.get(str(database_id))

    # ------------------------------------------------------------------
    # Column enumeration
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Value signature extraction
    # ------------------------------------------------------------------
    def _extract_column_signatures(
        self, database_path: str, column_keys: List[ColumnKey]
    ) -> Dict[str, ColumnValuesSignature]:
        """
        Extract value signatures for all columns.
        
        For each column:
        - Query distinct count and sample values
        - If distinct_count <= threshold: use exact set
        - If distinct_count > threshold: use MinHash
        """
        conn = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row

        signatures: Dict[str, ColumnValuesSignature] = {}

        try:
            for ck in column_keys:
                try:
                    sig = self._extract_single_column_signature(conn, ck)
                    if sig:
                        signatures[ck.to_id()] = sig
                except Exception as e:
                    logger.warning(
                        "Failed to extract signature for %s: %s", ck.to_id(), e
                    )
                    continue

        finally:
            conn.close()

        logger.info(
            "Extracted value signatures for %d/%d columns",
            len(signatures),
            len(column_keys),
        )
        return signatures

    def _extract_single_column_signature(
        self, conn: sqlite3.Connection, column_key: ColumnKey
    ) -> Optional[ColumnValuesSignature]:
        """Extract value signature for a single column"""
        table_name = column_key.table_name
        column_name = column_key.column_name

        # Get distinct count and sample values
        query = f"""
            WITH DistinctValues AS (
                SELECT DISTINCT `{column_name}` as value
                FROM `{table_name}`
                WHERE `{column_name}` IS NOT NULL
            )
            SELECT
                (SELECT COUNT(*) FROM DistinctValues) as total_distinct_count,
                value
            FROM DistinctValues
            LIMIT {self.distinct_exact_threshold + 1}
        """

        cursor = conn.execute(query)

        distinct_count = 0
        values: List[str] = []
        first_row = True

        for row in cursor:
            if first_row:
                distinct_count = row["total_distinct_count"]
                first_row = False

            if row["value"] is not None:
                # Normalize value: convert to string, strip, lowercase
                v = str(row["value"]).strip().lower()
                if v:  # Skip empty strings
                    values.append(v)

        if distinct_count == 0:
            logger.debug("Column %s has no distinct values", column_key.to_id())
            return None

        # Normalize values to set (deduplicate)
        values_set = set(values)

        # Decide: exact or MinHash
        if distinct_count <= self.distinct_exact_threshold:
            # Use exact set
            return ColumnValuesSignature(
                table_name=table_name,
                column_name=column_name,
                distinct_count=distinct_count,
                value_mode="exact",
                exact_values=values_set,
                minhash=None,
            )
        else:
            # Use MinHash (build from sample values)
            mh = MinHash(num_perm=self.minhash_num_perm)
            for v in values_set:
                mh.update(v.encode("utf-8"))

            return ColumnValuesSignature(
                table_name=table_name,
                column_name=column_name,
                distinct_count=distinct_count,
                value_mode="minhash",
                exact_values=None,
                minhash=mh,
            )

    # ------------------------------------------------------------------
    # Similarity computation
    # ------------------------------------------------------------------
    def _find_high_overlap_pairs(
        self, signatures: Dict[str, ColumnValuesSignature]
    ) -> List[Tuple[str, str, float]]:
        """
        Find all column pairs with Jaccard similarity >= threshold.
        
        Returns:
            List of (column_id1, column_id2, jaccard_score) tuples
        """
        pairs: List[Tuple[str, str, float]] = []
        sig_list = list(signatures.items())
        n = len(sig_list)

        logger.info("Computing Jaccard similarities for %d columns...", n)

        for i in range(n):
            col_id1, sig1 = sig_list[i]
            for j in range(i + 1, n):
                col_id2, sig2 = sig_list[j]

                jaccard = self._compute_jaccard(sig1, sig2)
                if jaccard is not None and jaccard >= self.value_jaccard_threshold:
                    pairs.append((col_id1, col_id2, jaccard))

        # Sort by score descending
        pairs.sort(key=lambda x: x[2], reverse=True)

        logger.info(
            "Found %d high-overlap pairs (threshold=%.2f)",
            len(pairs),
            self.value_jaccard_threshold,
        )

        # Log top pairs
        if pairs:
            logger.info("Top 10 highest overlap pairs:")
            for idx, (id1, id2, score) in enumerate(pairs[:10], 1):
                logger.info("  [%d] Jaccard=%.3f: %s <-> %s", idx, score, id1, id2)

        return pairs

    def _compute_jaccard(
        self, sig1: ColumnValuesSignature, sig2: ColumnValuesSignature
    ) -> Optional[float]:
        """
        Compute Jaccard similarity between two column signatures.
        
        Handles:
        - exact vs exact: direct set intersection/union
        - exact vs minhash: approximate using MinHash of exact set
        - minhash vs minhash: MinHash.jaccard()
        """
        # Case 1: Both exact
        if sig1.value_mode == "exact" and sig2.value_mode == "exact":
            if sig1.exact_values is None or sig2.exact_values is None:
                return None
            intersection = len(sig1.exact_values & sig2.exact_values)
            union = len(sig1.exact_values | sig2.exact_values)
            if union == 0:
                return 0.0
            return intersection / union

        # Case 2: Both MinHash
        if sig1.value_mode == "minhash" and sig2.value_mode == "minhash":
            if sig1.minhash is None or sig2.minhash is None:
                return None
            return sig1.minhash.jaccard(sig2.minhash)

        # Case 3: Mixed (exact + minhash)
        # Convert exact set to MinHash for comparison
        exact_sig = sig1 if sig1.value_mode == "exact" else sig2
        minhash_sig = sig2 if sig1.value_mode == "exact" else sig1

        if exact_sig.exact_values is None or minhash_sig.minhash is None:
            return None

        # Create temporary MinHash for exact set
        temp_mh = MinHash(num_perm=self.minhash_num_perm)
        for v in exact_sig.exact_values:
            temp_mh.update(v.encode("utf-8"))

        return temp_mh.jaccard(minhash_sig.minhash)

    # ------------------------------------------------------------------
    # Pair construction
    # ------------------------------------------------------------------
    def _build_pairs_from_overlaps(
        self,
        overlap_pairs: List[Tuple[str, str, float]],
        database_id: str,
    ) -> List[AmbiguousPair]:
        """
        Build AmbiguousPair objects from high-overlap column pairs.
        
        Each high-overlap pair becomes one AmbiguousPair directly.
        No clustering - we store the actual overlapping pairs.
        """
        if not overlap_pairs:
            return []

        pairs: List[AmbiguousPair] = []
        pair_idx = 0

        for col_id1, col_id2, jaccard_score in overlap_pairs:
            # Parse column IDs (format: table_name.column_name)
            try:
                table_a, col_a = col_id1.split(".", 1)
                table_b, col_b = col_id2.split(".", 1)
            except ValueError:
                logger.warning(
                    "Failed to parse column IDs: %s, %s", col_id1, col_id2
                )
                continue
            
            # Create pair
            pair_id = f"{database_id}_value_overlap_pair_{pair_idx:04d}"
            pair_idx += 1
            
            pairs.append(
                AmbiguousPair(
                    pair_id=pair_id,
                    database_id=database_id,
                    column_a=DBElementRef(table_name=table_a, column_name=col_a),
                    column_b=DBElementRef(table_name=table_b, column_name=col_b),
                    discovery_methods=["value_overlap"],
                    value_jaccard=jaccard_score,
                )
            )

        logger.info(
            "Built %d ambiguous pairs from value overlap",
            len(pairs),
        )

        return pairs

