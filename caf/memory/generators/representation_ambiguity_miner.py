"""
Representation Ambiguity Miner - Discover columns representing the same entity in different formats.

This miner identifies columns that represent the same real-world entity but in different
physical storage formats (e.g., ID vs Name vs Code).

The mining process has two steps:
1. Data-Driven Dependency Mining: Find 1-to-1 relationships between columns
2. LLM Semantic Verification: Verify if these pairs represent the same entity
"""

import logging
import sqlite3
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from ..stores.semantic import SemanticMemoryStore
from ..stores.similarity_cluster import SimilarityClusterStore
from ..types import SimilarityCluster, DBElementRef
from ...config.paths import PathConfig
from ...llm.client import BaseLLMClient, LLMConfig, create_llm_client
from ...config.global_config import get_llm_config
from caf.prompt import PromptFactory

logger = logging.getLogger(__name__)

# Configuration constants
ONE_TO_ONE_THRESHOLD = 0.05  # Allow 5% deviation for 1-to-1 relationship
LLM_BATCH_SIZE = 10  # Number of pairs to verify in each LLM batch
MIN_DISTINCT_COUNT = 2  # Minimum distinct count to consider a column


@dataclass
class ColumnKey:
    """Column identifier"""
    table_name: str
    column_name: str

    def to_id(self) -> str:
        return f"{self.table_name}.{self.column_name}"


@dataclass
class OneToOnePair:
    """A candidate 1-to-1 relationship pair"""
    col_a: ColumnKey
    col_b: ColumnKey
    n_a: int  # Distinct count of A (non-null)
    n_b: int  # Distinct count of B (non-null)
    n_ab: int  # Distinct count of (A, B) pairs (both non-null)
    ratio_a: float  # n_ab / n_a
    ratio_b: float  # n_ab / n_b


class RepresentationAmbiguityMiner:
    """
    Miner for discovering representation ambiguity (same entity, different formats).
    
    This miner:
    1. Scans database to find 1-to-1 relationships between columns (same table and cross-table)
    2. Uses LLM to verify if these pairs represent the same entity in different formats
    3. Builds clusters of columns representing the same entity
    """

    def __init__(
        self,
        semantic_store: SemanticMemoryStore,
        cluster_store: SimilarityClusterStore,
        memory_config: Dict,
        raw_config: Optional[Dict] = None,
    ):
        self.semantic_store = semantic_store
        self.cluster_store = cluster_store
        self.memory_config = memory_config or {}
        self._raw_config = raw_config

        # Configuration
        similarity_cfg = self.memory_config.get("similarity", {})
        self.one_to_one_threshold = similarity_cfg.get(
            "one_to_one_threshold", ONE_TO_ONE_THRESHOLD
        )
        self.llm_batch_size = similarity_cfg.get("llm_batch_size", LLM_BATCH_SIZE)
        self.min_distinct_count = similarity_cfg.get(
            "min_distinct_count", MIN_DISTINCT_COUNT
        )
        
        # Enable debug logging for threshold
        logger.info(
            "RepresentationAmbiguityMiner initialized with threshold=%.4f, min_distinct_count=%d",
            self.one_to_one_threshold,
            self.min_distinct_count
        )

        # Database mapping cache
        self._dbid_to_path: Optional[Dict[str, str]] = None
        default_mapping = PathConfig.get_database_mapping_path()
        self.mapping_path: Path = Path(
            self.memory_config.get("database_mapping_path", default_mapping)
        )

        # Initialize LLM client
        llm_config = None
        try:
            global_llm_config = get_llm_config()
            llm_config = LLMConfig(
                provider=global_llm_config.provider,
                model_name=global_llm_config.model_name,
                api_key=global_llm_config.api_key,
                base_url=global_llm_config.base_url,
                temperature=global_llm_config.temperature,
                max_tokens=global_llm_config.max_tokens,
                timeout=global_llm_config.timeout,
            )
            logger.debug("Using LLM config from global config")
        except (ValueError, Exception) as e:
            logger.debug(f"Failed to get LLM config from global config: {e}")
            llm_cfg = None

            # Try memory.semantic.search.llm_refinement first
            semantic_search = self.memory_config.get("semantic", {}).get("search", {})
            llm_refinement = semantic_search.get("llm_refinement", {})
            if llm_refinement:
                llm_cfg = llm_refinement
                logger.debug("Using LLM config from memory.semantic.search.llm_refinement")

            # If not found, try top-level llm in raw config
            if not llm_cfg and self._raw_config:
                llm_cfg = self._raw_config.get("llm", {})
                if llm_cfg:
                    logger.debug("Using LLM config from top-level llm section")

            # If still not found, try memory_config.llm
            if not llm_cfg:
                llm_cfg = self.memory_config.get("llm", {})
                if llm_cfg:
                    logger.debug("Using LLM config from memory_config.llm")

            # Create LLMConfig from found config
            if llm_cfg:
                llm_config = LLMConfig(
                    provider=llm_cfg.get("provider", "openai"),
                    model_name=llm_cfg.get("model_name", "gpt-4o-mini"),
                    api_key=llm_cfg.get("api_key"),
                    base_url=llm_cfg.get("base_url"),
                    temperature=llm_cfg.get("temperature", 0.1),
                    max_tokens=llm_cfg.get("max_tokens", 4000),
                    timeout=llm_cfg.get("timeout", 60),
                )
            else:
                # Check environment variable as last resort
                import os
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    provider = "openai" if os.getenv("OPENAI_API_KEY") else "anthropic"
                    logger.info(f"Using LLM API key from environment variable ({provider})")
                    llm_config = LLMConfig(
                        provider=provider,
                        model_name="gpt-4o-mini" if provider == "openai" else "claude-3-5-sonnet-20241022",
                        api_key=api_key,
                        base_url=None,
                    )
                else:
                    error_msg = (
                        "No LLM configuration found. Please either:\n"
                        "1. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable, or\n"
                        "2. Configure LLM settings in config/caf_config.yaml under 'llm' section, or\n"
                        "3. Configure LLM settings in config/caf_config.yaml under 'memory.semantic.search.llm_refinement' section"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

        try:
            self.llm_client = create_llm_client(llm_config)
        except Exception as e:
            error_msg = f"Failed to initialize LLM client: {e}. Please check your LLM configuration and API key."
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def mine_and_save_clusters(self, database_id: str) -> List[SimilarityCluster]:
        """
        High-level API:
        - Bind semantic & cluster stores
        - Resolve database path from database_id
        - Find 1-to-1 column pairs (same table and cross-table)
        - Verify with LLM if they represent the same entity
        - Build clusters from verified pairs
        - Save results to cluster store
        """
        logger.info(
            "Starting representation ambiguity mining for database: %s", database_id
        )

        self.semantic_store.bind_database(database_id)
        self.cluster_store.bind_database(database_id)

        # Get column list from semantic store
        column_df = self.semantic_store.dataframes.get("column")
        if column_df is None or column_df.empty:
            logger.warning(
                "No column metadata available for database: %s", database_id
            )
            clusters: List[SimilarityCluster] = []
            self.cluster_store.save_clusters(database_id, clusters)
            return clusters

        # Resolve database path
        database_path = self._resolve_database_path(database_id)
        if not database_path:
            logger.error(
                "Failed to resolve database path for database_id: %s", database_id
            )
            clusters = []
            self.cluster_store.save_clusters(database_id, clusters)
            return clusters

        # Get column keys
        column_keys = self._get_column_keys(column_df)
        if not column_keys:
            logger.warning(
                "No columns eligible for representation ambiguity mining in database: %s",
                database_id,
            )
            clusters = []
            self.cluster_store.save_clusters(database_id, clusters)
            return clusters

        # Step 1: Find 1-to-1 pairs
        logger.info("Step 1: Finding 1-to-1 column pairs...")
        one_to_one_pairs = self._find_one_to_one_pairs(database_path, column_keys)

        if not one_to_one_pairs:
            logger.info(
                "No 1-to-1 column pairs found for database: %s", database_id
            )
            clusters = []
            self.cluster_store.save_clusters(database_id, clusters)
            return clusters

        logger.info(
            "Found %d candidate 1-to-1 pairs", len(one_to_one_pairs)
        )

        # Step 2: Verify with LLM
        logger.info("Step 2: Verifying pairs with LLM...")
        verified_pairs = self._verify_with_llm(one_to_one_pairs, column_df, database_id)

        if not verified_pairs:
            logger.info(
                "No representation ambiguity pairs verified by LLM for database: %s",
                database_id,
            )
            clusters = []
            self.cluster_store.save_clusters(database_id, clusters)
            return clusters

        logger.info(
            "LLM verified %d representation ambiguity pairs", len(verified_pairs)
        )

        # Step 3: Build clusters from verified pairs
        logger.info("Building clusters from verified pairs...")
        clusters = self._build_clusters_from_pairs(verified_pairs, database_id)

        self.cluster_store.save_clusters(database_id, clusters)

        logger.info(
            "Finished representation ambiguity mining for database: %s (clusters=%d)",
            database_id,
            len(clusters),
        )
        return clusters

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
    # Step 1: Data-driven dependency mining
    # ------------------------------------------------------------------
    def _find_one_to_one_pairs(
        self, database_path: str, column_keys: List[ColumnKey]
    ) -> List[OneToOnePair]:
        """
        Find all 1-to-1 relationships between columns.
        
        Checks both:
        - Same table pairs: columns in the same table
        - Cross-table pairs: columns in different tables (via JOIN)
        """
        conn = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row

        pairs: List[OneToOnePair] = []

        try:
            # Group columns by table
            columns_by_table: Dict[str, List[ColumnKey]] = defaultdict(list)
            for ck in column_keys:
                columns_by_table[ck.table_name].append(ck)

            # Check same-table pairs
            logger.info("Checking same-table column pairs...")
            same_table_checked = 0
            for table_name, cols in columns_by_table.items():
                if len(cols) < 2:
                    continue
                
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        col_a, col_b = cols[i], cols[j]
                        same_table_checked += 1
                        pair = self._check_one_to_one_same_table(
                            conn, table_name, col_a, col_b
                        )
                        if pair:
                            pairs.append(pair)
            
            logger.info("Checked %d same-table pairs, found %d 1-to-1 pairs", 
                       same_table_checked, len(pairs))

            # Check cross-table pairs
            # NOTE: Cross-table 1-to-1 relationships should only be checked when there's
            # a foreign key relationship. JOIN must use the PK-FK relationship defined in the system.
            logger.info("Checking cross-table column pairs (with FK relationships only)...")
            cross_table_checked = 0
            
            # Load foreign key relationships
            fk_relationships = self._load_foreign_key_relationships(conn)
            
            table_names = list(columns_by_table.keys())
            for i in range(len(table_names)):
                for j in range(i + 1, len(table_names)):
                    table_a, table_b = table_names[i], table_names[j]
                    cols_a = columns_by_table[table_a]
                    cols_b = columns_by_table[table_b]
                    
                    # Find any FK relationship between these two tables (for JOIN)
                    fk_pairs = self._get_fk_pairs_for_tables(fk_relationships, table_a, table_b)
                    
                    if not fk_pairs:
                        # No FK relationship between these tables, skip
                        continue
                    
                    # Use the first FK relationship for JOIN (any FK relationship works)
                    fk_info = fk_pairs[0]
                    
                    for col_a in cols_a:
                        for col_b in cols_b:
                            # Check if col_a and col_b have 1-to-1 relationship after JOIN via FK
                            cross_table_checked += 1
                            pair = self._check_one_to_one_cross_table(
                                conn, table_a, col_a, table_b, col_b, fk_info
                            )
                            if pair:
                                pairs.append(pair)
            
            logger.info("Checked %d cross-table pairs, found %d 1-to-1 pairs total", 
                       cross_table_checked, len(pairs))

        finally:
            conn.close()

        logger.info(
            "Found %d 1-to-1 pairs (threshold=%.3f)",
            len(pairs),
            self.one_to_one_threshold,
        )

        return pairs

    def _check_one_to_one_same_table(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        col_a: ColumnKey,
        col_b: ColumnKey,
    ) -> Optional[OneToOnePair]:
        """Check if two columns in the same table have a 1-to-1 relationship"""
        try:
            # Calculate N_A, N_B, N_AB
            # Note: SQLite doesn't support COUNT(DISTINCT (col1, col2)) directly
            # We need to use a subquery or concatenation
            query = f"""
                SELECT
                    COUNT(DISTINCT `{col_a.column_name}`) as n_a,
                    COUNT(DISTINCT `{col_b.column_name}`) as n_b,
                    (SELECT COUNT(*) FROM (
                        SELECT DISTINCT `{col_a.column_name}`, `{col_b.column_name}`
                        FROM `{table_name}`
                        WHERE `{col_a.column_name}` IS NOT NULL AND `{col_b.column_name}` IS NOT NULL
                    )) as n_ab
                FROM `{table_name}`
                WHERE `{col_a.column_name}` IS NOT NULL AND `{col_b.column_name}` IS NOT NULL
                LIMIT 1
            """

            cursor = conn.execute(query)
            row = cursor.fetchone()
            if not row:
                return None

            n_a = row["n_a"]
            n_b = row["n_b"]
            n_ab = row["n_ab"]

            # Check if both columns have minimum distinct count
            if n_a < self.min_distinct_count or n_b < self.min_distinct_count:
                logger.info(
                    "[SAME-TABLE] Pair: %s.%s <-> %s.%s | "
                    "SKIPPED: distinct count too low (n_a=%d, n_b=%d, min=%d)",
                    table_name, col_a.column_name, table_name, col_b.column_name,
                    n_a, n_b, self.min_distinct_count
                )
                return None

            # Check 1-to-1 relationship: N_A ≈ N_AB and N_B ≈ N_AB
            if n_a == 0 or n_b == 0 or n_ab == 0:
                logger.info(
                    "[SAME-TABLE] Pair: %s.%s <-> %s.%s | "
                    "SKIPPED: zero counts (n_a=%d, n_b=%d, n_ab=%d)",
                    table_name, col_a.column_name, table_name, col_b.column_name,
                    n_a, n_b, n_ab
                )
                return None

            ratio_a = n_ab / n_a if n_a > 0 else 0.0
            ratio_b = n_ab / n_b if n_b > 0 else 0.0

            # Log all candidate pairs for debugging
            diff_a = abs(1.0 - ratio_a)
            diff_b = abs(1.0 - ratio_b)
            is_valid = diff_a <= self.one_to_one_threshold and diff_b <= self.one_to_one_threshold
            
            logger.info(
                "[SAME-TABLE] Pair: %s.%s <-> %s.%s | "
                "n_a=%d, n_b=%d, n_ab=%d | "
                "ratio_a=%.4f (diff=%.4f), ratio_b=%.4f (diff=%.4f) | "
                "threshold=%.4f | %s",
                table_name, col_a.column_name, table_name, col_b.column_name,
                n_a, n_b, n_ab,
                ratio_a, diff_a, ratio_b, diff_b,
                self.one_to_one_threshold,
                "✓ VALID" if is_valid else "✗ INVALID"
            )

            # Both ratios should be close to 1.0 (within threshold)
            # This ensures bidirectional 1-to-1: each value in A maps to at most one value in B,
            # and each value in B maps to at most one value in A
            if is_valid:
                return OneToOnePair(
                    col_a=col_a,
                    col_b=col_b,
                    n_a=n_a,
                    n_b=n_b,
                    n_ab=n_ab,
                    ratio_a=ratio_a,
                    ratio_b=ratio_b,
                )

        except Exception as e:
            logger.debug(
                "Failed to check 1-to-1 for %s.%s and %s.%s: %s",
                table_name,
                col_a.column_name,
                table_name,
                col_b.column_name,
                e,
            )

        return None

    def _check_one_to_one_cross_table(
        self,
        conn: sqlite3.Connection,
        table_a: str,
        col_a: ColumnKey,
        table_b: str,
        col_b: ColumnKey,
        fk_info: Tuple[str, str, str, str],
    ) -> Optional[OneToOnePair]:
        """
        Check if two columns in different tables have a 1-to-1 relationship via JOIN.
        
        The JOIN uses the FK relationship, but col_a and col_b can be any columns
        in their respective tables (not necessarily the FK columns).
        
        Args:
            fk_info: (source_table, source_column, target_table, target_column) 
                    representing the FK relationship to use for JOIN
        """
        try:
            # Extract FK relationship information
            fk_source_table, fk_source_column, fk_target_table, fk_target_column = fk_info
            
            # Build JOIN condition using FK relationship
            # The FK relationship connects fk_source_table.fk_source_column to fk_target_table.fk_target_column
            if fk_source_table == table_a and fk_target_table == table_b:
                # FK: table_a.fk_source_column -> table_b.fk_target_column
                join_condition = f"t1.`{fk_source_column}` = t2.`{fk_target_column}`"
            elif fk_source_table == table_b and fk_target_table == table_a:
                # FK: table_b.fk_source_column -> table_a.fk_target_column
                join_condition = f"t2.`{fk_source_column}` = t1.`{fk_target_column}`"
            else:
                logger.debug(
                    "FK relationship %s does not match tables %s and %s",
                    fk_info, table_a, table_b
                )
                return None
            
            # We want to check col_a (in table_a) and col_b (in table_b)
            # These columns are independent of the FK relationship used for JOIN
            select_col_a = f"t1.`{col_a.column_name}`"
            select_col_b = f"t2.`{col_b.column_name}`"
            where_col_a = f"t1.`{col_a.column_name}`"
            where_col_b = f"t2.`{col_b.column_name}`"
            
            # Calculate N_A, N_B, N_AB via JOIN using FK relationship
            # Note: SQLite doesn't support COUNT(DISTINCT (col1, col2)) directly
            query = f"""
                SELECT
                    COUNT(DISTINCT {select_col_a}) as n_a,
                    COUNT(DISTINCT {select_col_b}) as n_b,
                    (SELECT COUNT(*) FROM (
                        SELECT DISTINCT {select_col_a}, {select_col_b}
                        FROM `{table_a}` t1
                        INNER JOIN `{table_b}` t2
                        ON {join_condition}
                        WHERE {where_col_a} IS NOT NULL 
                          AND {where_col_b} IS NOT NULL
                    )) as n_ab
                FROM `{table_a}` t1
                INNER JOIN `{table_b}` t2
                ON {join_condition}
                WHERE {where_col_a} IS NOT NULL 
                  AND {where_col_b} IS NOT NULL
                LIMIT 1
            """

            cursor = conn.execute(query)
            row = cursor.fetchone()
            if not row:
                return None

            n_a = row["n_a"]
            n_b = row["n_b"]
            n_ab = row["n_ab"]

            # Check if both columns have minimum distinct count
            if n_a < self.min_distinct_count or n_b < self.min_distinct_count:
                fk_source_table, fk_source_column, fk_target_table, fk_target_column = fk_info
                join_info = f"JOIN: {fk_source_table}.{fk_source_column} = {fk_target_table}.{fk_target_column}"
                logger.info(
                    "[CROSS-TABLE] Pair: %s.%s <-> %s.%s | "
                    "%s | "
                    "SKIPPED: distinct count too low (n_a=%d, n_b=%d, min=%d)",
                    table_a, col_a.column_name, table_b, col_b.column_name,
                    join_info,
                    n_a, n_b, self.min_distinct_count
                )
                return None

            # Check 1-to-1 relationship: N_A ≈ N_AB and N_B ≈ N_AB
            if n_a == 0 or n_b == 0 or n_ab == 0:
                fk_source_table, fk_source_column, fk_target_table, fk_target_column = fk_info
                join_info = f"JOIN: {fk_source_table}.{fk_source_column} = {fk_target_table}.{fk_target_column}"
                logger.info(
                    "[CROSS-TABLE] Pair: %s.%s <-> %s.%s | "
                    "%s | "
                    "SKIPPED: zero counts (n_a=%d, n_b=%d, n_ab=%d)",
                    table_a, col_a.column_name, table_b, col_b.column_name,
                    join_info,
                    n_a, n_b, n_ab
                )
                return None

            ratio_a = n_ab / n_a if n_a > 0 else 0.0
            ratio_b = n_ab / n_b if n_b > 0 else 0.0

            # Log all candidate pairs for debugging
            diff_a = abs(1.0 - ratio_a)
            diff_b = abs(1.0 - ratio_b)
            is_valid = diff_a <= self.one_to_one_threshold and diff_b <= self.one_to_one_threshold
            
            # Extract FK info for logging
            fk_source_table, fk_source_column, fk_target_table, fk_target_column = fk_info
            join_info = f"JOIN: {fk_source_table}.{fk_source_column} = {fk_target_table}.{fk_target_column}"
            
            logger.info(
                "[CROSS-TABLE] Pair: %s.%s <-> %s.%s | "
                "%s | "
                "n_a=%d, n_b=%d, n_ab=%d | "
                "ratio_a=%.4f (diff=%.4f), ratio_b=%.4f (diff=%.4f) | "
                "threshold=%.4f | %s",
                table_a, col_a.column_name, table_b, col_b.column_name,
                join_info,
                n_a, n_b, n_ab,
                ratio_a, diff_a, ratio_b, diff_b,
                self.one_to_one_threshold,
                "✓ VALID" if is_valid else "✗ INVALID"
            )

            # Both ratios should be close to 1.0 (within threshold)
            if is_valid:
                return OneToOnePair(
                    col_a=col_a,
                    col_b=col_b,
                    n_a=n_a,
                    n_b=n_b,
                    n_ab=n_ab,
                    ratio_a=ratio_a,
                    ratio_b=ratio_b,
                )

        except Exception as e:
            logger.debug(
                "Failed to check cross-table 1-to-1 for %s.%s and %s.%s: %s",
                table_a,
                col_a.column_name,
                table_b,
                col_b.column_name,
                e,
            )

        return None

    def _load_foreign_key_relationships(self, conn: sqlite3.Connection) -> Set[Tuple[str, str, str, str]]:
        """Load foreign key relationships from database"""
        fk_relationships = set()
        
        try:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table_name in tables:
                try:
                    cursor = conn.execute(f"PRAGMA foreign_key_list(`{table_name}`)")
                    for fk_row in cursor:
                        # fk_row: (id, seq, table, from, to, on_update, on_delete, match)
                        source_table = table_name
                        source_column = fk_row[3]  # 'from'
                        target_table = fk_row[2]    # 'table'
                        target_column = fk_row[4]  # 'to'
                        
                        if source_table and source_column and target_table and target_column:
                            fk_relationships.add((source_table, source_column, target_table, target_column))
                            # Also add reverse direction
                            fk_relationships.add((target_table, target_column, source_table, source_column))
                except Exception as e:
                    logger.debug(f"Failed to load FK for table {table_name}: {e}")
                    continue
        except Exception as e:
            logger.debug(f"Failed to load FK relationships: {e}")
        
        return fk_relationships

    def _has_fk_relationship(
        self, 
        fk_relationships: Set[Tuple[str, str, str, str]],
        table_a: str, col_a: str,
        table_b: str, col_b: str
    ) -> bool:
        """Check if two columns have a foreign key relationship"""
        return (
            (table_a, col_a, table_b, col_b) in fk_relationships or
            (table_b, col_b, table_a, col_a) in fk_relationships
        )
    
    def _get_fk_relationship_for_columns(
        self,
        fk_relationships: Set[Tuple[str, str, str, str]],
        table_a: str, col_a: str,
        table_b: str, col_b: str
    ) -> Optional[Tuple[str, str, str, str]]:
        """
        Get the FK relationship tuple for two columns.
        Returns the FK relationship if it exists, None otherwise.
        """
        # Check if (table_a, col_a) -> (table_b, col_b) is a FK relationship
        if (table_a, col_a, table_b, col_b) in fk_relationships:
            return (table_a, col_a, table_b, col_b)
        
        # Check if (table_b, col_b) -> (table_a, col_a) is a FK relationship
        if (table_b, col_b, table_a, col_a) in fk_relationships:
            return (table_b, col_b, table_a, col_a)
        
        return None
    
    def _get_fk_pairs_for_tables(
        self,
        fk_relationships: Set[Tuple[str, str, str, str]],
        table_a: str,
        table_b: str
    ) -> List[Tuple[str, str, str, str]]:
        """
        Get all FK relationships between two tables.
        Returns list of (source_table, source_column, target_table, target_column) tuples.
        """
        pairs = []
        for fk in fk_relationships:
            source_table, source_column, target_table, target_column = fk
            if (source_table == table_a and target_table == table_b) or \
               (source_table == table_b and target_table == table_a):
                pairs.append(fk)
        return pairs

    def _are_columns_similar(self, col_a_name: str, col_b_name: str) -> bool:
        """Check if two column names are similar (both are IDs, codes, names, etc.)"""
        col_a_lower = col_a_name.lower()
        col_b_lower = col_b_name.lower()
        
        # Check if both contain similar semantic indicators
        id_indicators = ['id', 'identifier', 'key']
        code_indicators = ['code', 'cd']
        name_indicators = ['name', 'nm', 'title']
        
        def has_indicator(col_name, indicators):
            return any(ind in col_name for ind in indicators)
        
        # Both are IDs/identifiers
        if has_indicator(col_a_lower, id_indicators) and has_indicator(col_b_lower, id_indicators):
            return True
        
        # Both are codes
        if has_indicator(col_a_lower, code_indicators) and has_indicator(col_b_lower, code_indicators):
            return True
        
        # Both are names (but be careful - this might be too broad)
        # Only if they're clearly the same entity type (e.g., both end with same suffix)
        if has_indicator(col_a_lower, name_indicators) and has_indicator(col_b_lower, name_indicators):
            # Extract base entity (e.g., "school_name" -> "school")
            # This is a simple heuristic
            return False  # Be conservative - names are often not 1-to-1
        
        return False

    # ------------------------------------------------------------------
    # Step 2: LLM semantic verification
    # ------------------------------------------------------------------
    def _verify_with_llm(
        self,
        pairs: List[OneToOnePair],
        column_df: pd.DataFrame,
        database_id: str,
    ) -> List[Tuple[ColumnKey, ColumnKey]]:
        """
        Verify with LLM if pairs represent the same entity in different formats.
        
        Returns list of verified pairs (col_a, col_b).
        """
        if not pairs:
            return []

        # Build column metadata index
        col_metadata = self._build_column_metadata_index(column_df)

        # Process in batches
        verified_pairs: List[Tuple[ColumnKey, ColumnKey]] = []

        for i in range(0, len(pairs), self.llm_batch_size):
            batch = pairs[i : i + self.llm_batch_size]
            logger.debug(
                "Verifying batch %d/%d (%d pairs)",
                i // self.llm_batch_size + 1,
                (len(pairs) + self.llm_batch_size - 1) // self.llm_batch_size,
                len(batch),
            )

            batch_verified = self._verify_batch_with_llm(batch, col_metadata, database_id)
            verified_pairs.extend(batch_verified)

        return verified_pairs

    def _build_column_metadata_index(
        self, column_df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """Build index of column metadata for LLM prompts"""
        metadata: Dict[str, Dict[str, Any]] = {}

        for _, row in column_df.iterrows():
            table_name = row.get("table_name")
            column_name = row.get("column_name")
            if not table_name or not column_name:
                continue

            col_id = f"{table_name}.{column_name}"
            metadata[col_id] = {
                "table_name": table_name,
                "column_name": column_name,
                "whole_column_name": row.get("whole_column_name", ""),
                "description": row.get("description", ""),
                "data_type": row.get("data_type", ""),
                "data_format": row.get("data_format", ""),
                "semantic_type": row.get("semantic_type", ""),
            }

        return metadata

    def _verify_batch_with_llm(
        self,
        batch: List[OneToOnePair],
        col_metadata: Dict[str, Dict[str, Any]],
        database_id: str,
    ) -> List[Tuple[ColumnKey, ColumnKey]]:
        """Verify a batch of pairs with LLM"""
        # Build prompt for batch
        prompt = self._build_verification_prompt(batch, col_metadata)

        try:
            response = self.llm_client.call_with_messages(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )

            # Parse response
            verified_pairs = self._parse_llm_response(batch, response)
            return verified_pairs

        except Exception as e:
            logger.error("LLM verification failed: %s", e)
            # On error, return empty list (conservative approach)
            return []

    def _build_verification_prompt(
        self, batch: List[OneToOnePair], col_metadata: Dict[str, Dict[str, Any]]
    ) -> str:
        """Build LLM prompt for batch verification"""
        # Convert pairs to dict format for PromptFactory
        pairs = []
        for pair in batch:
            col_a_id = pair.col_a.to_id()
            col_b_id = pair.col_b.to_id()
            
            # Get metadata and enrich with column IDs
            meta_a = col_metadata.get(col_a_id, {})
            meta_b = col_metadata.get(col_b_id, {})
            
            # Ensure metadata has column IDs for PromptFactory
            if 'column_a_id' not in meta_a:
                meta_a = meta_a.copy()
                meta_a['column_a_id'] = col_a_id
            if 'column_b_id' not in meta_b:
                meta_b = meta_b.copy()
                meta_b['column_b_id'] = col_b_id
            
            pair_dict = {
                'column_a_id': col_a_id,
                'column_b_id': col_b_id,
                'statistics': {
                    'n_a': pair.n_a,
                    'n_b': pair.n_b,
                    'n_ab': pair.n_ab,
                    'ratio_a': pair.ratio_a,
                    'ratio_b': pair.ratio_b
                }
            }
            pairs.append(pair_dict)
        
        # Use PromptFactory to format the prompt
        return PromptFactory.format_representation_ambiguity_verification_prompt(
            pairs=pairs,
            col_metadata=col_metadata
        )

    def _format_column_description(
        self, metadata: Dict[str, Any], col_key: ColumnKey
    ) -> str:
        """Format column description for LLM prompt"""
        parts = []

        # Table and column name
        table_name = metadata.get("table_name", col_key.table_name)
        column_name = metadata.get("column_name", col_key.column_name)
        whole_name = metadata.get("whole_column_name", "")
        name_to_use = whole_name if whole_name else column_name
        parts.append(f"{table_name}.{name_to_use}")

        # Description
        description = metadata.get("description", "")
        if description and str(description).lower() not in ["nan", "none", "null", ""]:
            parts.append(f"Description: {description}")

        # Data type
        data_type = metadata.get("data_type", "")
        if data_type and str(data_type).lower() not in ["nan", "none", "null", ""]:
            parts.append(f"Type: {data_type}")

        # Data format
        data_format = metadata.get("data_format", "")
        if data_format and str(data_format).lower() not in ["nan", "none", "null", ""]:
            parts.append(f"Format: {data_format}")

        # Semantic type
        semantic_type = metadata.get("semantic_type", "")
        if semantic_type and str(semantic_type).lower() not in ["nan", "none", "null", ""]:
            parts.append(f"Semantic: {semantic_type}")

        return " | ".join(parts)

    def _parse_llm_response(
        self, batch: List[OneToOnePair], response: str
    ) -> List[Tuple[ColumnKey, ColumnKey]]:
        """Parse LLM response and return verified pairs"""
        verified_pairs: List[Tuple[ColumnKey, ColumnKey]] = []

        try:
            # Try to parse as JSON
            response = response.strip()
            # Remove markdown code blocks if present
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
                response = response.strip()

            # Parse JSON
            if response.startswith("["):
                results = json.loads(response)
            else:
                # Try to extract JSON from text
                import re
                json_match = re.search(r"\[.*\]", response, re.DOTALL)
                if json_match:
                    results = json.loads(json_match.group())
                else:
                    logger.warning("Could not parse LLM response as JSON: %s", response[:200])
                    return []

            # Validate results
            if not isinstance(results, list):
                logger.warning("LLM response is not a list: %s", type(results))
                return []

            if len(results) != len(batch):
                logger.warning(
                    "LLM response length (%d) does not match batch size (%d)",
                    len(results),
                    len(batch),
                )
                # Use what we have
                results = results[: len(batch)]

            # Extract verified pairs (where result is 'A')
            for i, result in enumerate(results):
                if i >= len(batch):
                    break

                # Normalize result
                result_str = str(result).strip().upper()
                if result_str == "A":
                    verified_pairs.append((batch[i].col_a, batch[i].col_b))

        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON: %s", e)
            logger.debug("Response was: %s", response[:500])

        return verified_pairs

    # ------------------------------------------------------------------
    # Step 3: Cluster construction
    # ------------------------------------------------------------------
    def _build_clusters_from_pairs(
        self,
        verified_pairs: List[Tuple[ColumnKey, ColumnKey]],
        database_id: str,
    ) -> List[SimilarityCluster]:
        """Build connected components from verified pairs"""
        if not verified_pairs:
            return []

        # Union-Find structure
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            if parent.get(x, x) != x:
                parent[x] = find(parent[x])
            return parent.get(x, x)

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        # Initialize parent for all columns in pairs
        for col_a, col_b in verified_pairs:
            col_a_id = col_a.to_id()
            col_b_id = col_b.to_id()
            parent.setdefault(col_a_id, col_a_id)
            parent.setdefault(col_b_id, col_b_id)

        # Union all pairs
        for col_a, col_b in verified_pairs:
            col_a_id = col_a.to_id()
            col_b_id = col_b.to_id()
            union(col_a_id, col_b_id)

        # Group columns by root
        clusters_map: Dict[str, List[str]] = defaultdict(list)
        for col_id in parent.keys():
            root = find(col_id)
            clusters_map[root].append(col_id)

        clusters: List[SimilarityCluster] = []
        cluster_idx = 0

        for root, members in clusters_map.items():
            if len(members) < 2:
                continue

            # Build DBElementRef list
            elements: List[DBElementRef] = []
            for col_id in sorted(members):
                try:
                    table_name, column_name = col_id.split(".", 1)
                except ValueError:
                    continue
                elements.append(
                    DBElementRef(table_name=table_name, column_name=column_name)
                )

            if len(elements) < 2:
                continue

            cluster_id = f"{database_id}_representation_ambiguity_cluster_{cluster_idx:04d}"
            cluster_idx += 1

            clusters.append(
                SimilarityCluster(
                    cluster_id=cluster_id,
                    database_id=database_id,
                    elements=elements,
                    methods=["representation_ambiguity"],
                    semantic_score_min=None,
                    semantic_score_max=None,
                    semantic_score_avg=None,
                    value_jaccard_min=None,
                    value_jaccard_max=None,
                    value_overlap_min=None,
                    value_overlap_max=None,
                )
            )

        return clusters

