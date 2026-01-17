"""
Pseudo-Query Collision Miner - Discover ambiguous columns using reverse query generation.

This miner identifies columns that are likely to be confused during question-schema matching
by generating natural language queries for each column and checking if those queries
incorrectly retrieve other columns.
"""

import logging
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from ..stores.semantic import SemanticMemoryStore
from ..stores.ambiguous_pair import AmbiguousPairStore
from ..types import AmbiguousPair, DBElementRef, CollisionInfo
from ..search.engines.semantic.indexing.column_indexer import ColumnFacetProvider
from ..search.engines.semantic.matchers.splade_matcher import SPLADEMatcher, SPLADEMatchResult
from ...llm.client import BaseLLMClient, LLMConfig, create_llm_client
from ...config.paths import PathConfig
from ...config.global_config import get_llm_config
from caf.prompt import PromptFactory

logger = logging.getLogger(__name__)

# Configuration constants
# For SPLADE scores (sparse vector inner products), we use relative thresholds
COLLISION_SCORE_THRESHOLD = None  # Absolute threshold disabled for SPLADE (scores are not normalized)
TOP_K_RETRIEVAL = 5  # Number of top results to check
SCORE_DIFF_THRESHOLD = None  # Absolute difference threshold disabled (use relative instead)
RELATIVE_SCORE_THRESHOLD = 0.7  # Distractor score must be >= target_score * this ratio
RELATIVE_SCORE_DIFF_THRESHOLD = 0.3  # Relative difference: |distractor - target| / target <= this
QUERIES_PER_COLUMN = 3  # Number of queries to generate per column (2 exact_match, 1 concept_based)
BATCH_SIZE = 10  # Number of columns to process in parallel


@dataclass
class ColumnKey:
    """Column identifier"""
    table_name: str
    column_name: str

    def to_id(self, database_id: Optional[str] = None) -> str:
        """Generate column ID, optionally with database_id prefix"""
        if database_id:
            return f"{database_id}.{self.table_name}.{self.column_name}"
        return f"{self.table_name}.{self.column_name}"


@dataclass
class GeneratedQuery:
    """A generated query with its type"""
    query_text: str
    query_type: str  # "exact_match" or "concept_based"


@dataclass
class CollisionResult:
    """Result of a collision detection"""
    source_column_id: str
    distractor_column_id: str
    trigger_query: str
    query_type: str
    target_score: float
    distractor_score: float
    collision_type: str  # "semantic_ambiguity"


class CustomColumnFacetProvider:
    """
    Custom facet provider for pseudo-query collision mining.
    
    Only provides two facets:
    1. "column_name": Just the column name
    2. "column_name_with_desc": Column name + column description (no table info)
    """
    
    def __init__(self, database_id: str, column_df: pd.DataFrame):
        self.database_id = database_id
        self.column_df = column_df
        # Build index for fast lookup
        self._col_idx: Dict[Tuple[str, str], int] = {}
        for i, row in column_df.iterrows():
            t = row.get('table_name')
            c = row.get('column_name')
            if t and c:
                self._col_idx[(str(t), str(c))] = i
    
    def iter_column_ids(self):
        """Iterate over all column IDs"""
        for (table_name, column_name) in self._col_idx.keys():
            yield f"{self.database_id}.{table_name}.{column_name}"
    
    def get_faceted_texts(self, column_id: str) -> Dict[str, str]:
        """
        Get custom facet texts for a column.
        
        Returns:
            Dict with keys:
            - "column_name": Just the column name
            - "column_name_with_desc": Column name + description (if available)
        """
        # Parse column_id: database_id.table.column
        parts = column_id.split(".")
        if len(parts) != 3:
            return {"column_name": "", "column_name_with_desc": ""}
        
        _, table_name, column_name = parts
        
        # Get column row
        if (table_name, column_name) not in self._col_idx:
            return {"column_name": "", "column_name_with_desc": ""}
        
        row = self.column_df.loc[self._col_idx[(table_name, column_name)]]
        
        # Extract column name
        col_name = str(row.get("column_name", "")).strip()
        whole_col_name = str(row.get("whole_column_name", "")).strip()
        
        # Use whole_column_name if available and different, otherwise use column_name
        name_to_use = whole_col_name if whole_col_name and whole_col_name != col_name else col_name
        
        # Extract description
        description = str(row.get("description", "")).strip()
        
        # Build facets
        facet_texts = {
            "column_name": name_to_use,
            "column_name_with_desc": name_to_use
        }
        
        # Add description if available
        if description:
            facet_texts["column_name_with_desc"] = f"{name_to_use} {description}"
        
        return facet_texts


class PseudoQueryCollisionMiner:
    """
    Miner for discovering ambiguous columns using pseudo-query collision.
    
    This miner:
    1. Generates natural language queries for each column (2 types: exact match, concept-based)
    2. Uses SPLADE to retrieve columns for each generated query
    3. Detects collisions: when a query for column A retrieves column B with high score
    4. Filters out PK/FK relationships
    5. Outputs ambiguous pairs (not clusters)
    """

    def __init__(
        self,
        semantic_store: SemanticMemoryStore,
        pair_store: AmbiguousPairStore,
        memory_config: Dict,
        raw_config: Optional[Dict] = None,
    ):
        self.semantic_store = semantic_store
        self.pair_store = pair_store
        self.memory_config = memory_config or {}
        self._raw_config = raw_config  # Store raw config for accessing top-level llm section
        
        # Configuration
        similarity_cfg = self.memory_config.get("similarity", {})
        # Absolute thresholds (for backward compatibility, but not recommended for SPLADE)
        self.collision_score_threshold = similarity_cfg.get(
            "collision_score_threshold", COLLISION_SCORE_THRESHOLD
        )
        self.score_diff_threshold = similarity_cfg.get(
            "score_diff_threshold", SCORE_DIFF_THRESHOLD
        )
        # Relative thresholds (recommended for SPLADE scores)
        self.relative_score_threshold = similarity_cfg.get(
            "relative_score_threshold", RELATIVE_SCORE_THRESHOLD
        )
        self.relative_score_diff_threshold = similarity_cfg.get(
            "relative_score_diff_threshold", RELATIVE_SCORE_DIFF_THRESHOLD
        )
        self.top_k_retrieval = similarity_cfg.get("top_k_retrieval", TOP_K_RETRIEVAL)
        self.queries_per_column = similarity_cfg.get(
            "queries_per_column", QUERIES_PER_COLUMN
        )
        self.batch_size = similarity_cfg.get("batch_size", BATCH_SIZE)
        
        # Initialize LLM client
        # Try multiple sources for LLM configuration
        llm_config = None
        
        # First, try to get from global config (may throw exception if invalid)
        try:
            global_llm_config = get_llm_config()
            # Convert GlobalLLMConfig to LLMConfig
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
            # Fallback: try to get from memory_config or other sources
            llm_cfg = None
            
            # Try memory.semantic.search.llm_refinement first (most specific)
            semantic_search = self.memory_config.get("semantic", {}).get("search", {})
            llm_refinement = semantic_search.get("llm_refinement", {})
            if llm_refinement:
                llm_cfg = llm_refinement
                logger.debug("Using LLM config from memory.semantic.search.llm_refinement")
            
            # If not found, try top-level llm in raw config (if available)
            if not llm_cfg and self._raw_config:
                llm_cfg = self._raw_config.get("llm", {})
                if llm_cfg:
                    logger.debug("Using LLM config from top-level llm section")
            
            # If still not found, try memory_config.llm (unlikely but possible)
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
                # Last resort: use defaults (will likely fail without API key)
                logger.warning("No LLM configuration found, using defaults (may fail without API key)")
                llm_config = LLMConfig(
                    provider="openai",
                    model_name="gpt-4o-mini",
                    api_key=None,
                    base_url=None,
                )
        
        self.llm_client = create_llm_client(llm_config)
        
        # Initialize SPLADE matcher
        # Read SPLADE config from semantic.search.splade (not from embedding config)
        splade_cfg = (
            self.memory_config.get("semantic", {})
            .get("search", {})
            .get("splade", {})
        )
        # Fallback to default if splade config not found
        splade_model = splade_cfg.get(
            "model_name", "/home/yangchenyu/pre-trained-models/splade-v3"
        ) if splade_cfg else "/home/yangchenyu/pre-trained-models/splade-v3"
        splade_cache_path = PathConfig.get_splade_cache_path()
        
        # Use custom facet weights for column-only facets
        self.splade_matcher = SPLADEMatcher(
            facet_weights={"column_name": 3.0, "column_name_with_desc": 1.0},
            model_name=splade_model,
            cache_path=splade_cache_path,
        )
        
        # Column facet provider (will be initialized when building indexes)
        self.column_provider: Optional[ColumnFacetProvider] = None
        
        # Current database ID (for column_id generation)
        self.current_database_id: Optional[str] = None
        
        # Cache for PK/FK relationships
        self._pk_fk_relationships: Optional[Set[Tuple[str, str]]] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def mine_and_save_pairs(self, database_id: str) -> List[AmbiguousPair]:
        """
        High-level API:
        - Bind semantic & pair stores
        - Get column list from semantic store
        - Build SPLADE indexes
        - Generate queries for each column
        - Detect collisions
        - Filter PK/FK relationships
        - Build pairs from collisions (not clusters)
        - Save results to pair store
        """
        logger.info(
            "Starting pseudo-query collision mining for database: %s", database_id
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

        # Get column keys
        column_keys = self._get_column_keys(column_df)
        if not column_keys:
            logger.warning(
                "No columns eligible for pseudo-query collision mining in database: %s",
                database_id,
            )
            pairs = []
            self.pair_store.save_pairs(database_id, pairs)
            return pairs

        # Store database_id for column_id generation
        self.current_database_id = database_id
        
        # Build SPLADE indexes
        logger.info("Building SPLADE indexes for %d columns...", len(column_keys))
        self._build_indexes(database_id, column_df)

        # Load PK/FK relationships for filtering
        self._load_pk_fk_relationships(column_df)

        # Generate queries and detect collisions
        logger.info("Generating queries and detecting collisions...")
        collisions = self._detect_collisions(column_keys, column_df, database_id)

        if not collisions:
            logger.info(
                "No collisions detected for database: %s",
                database_id,
            )
            pairs = []
            self.pair_store.save_pairs(database_id, pairs)
            return pairs

        # Build pairs from collisions (directly, no clustering)
        logger.info("Building pairs from %d collisions...", len(collisions))
        pairs = self._build_pairs_from_collisions(collisions, database_id)

        self.pair_store.save_pairs(database_id, pairs)

        logger.info(
            "Finished pseudo-query collision mining for database: %s (pairs=%d)",
            database_id,
            len(pairs),
        )
        return pairs

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
    # Index building
    # ------------------------------------------------------------------
    def _build_indexes(self, database_id: str, column_df: pd.DataFrame) -> None:
        """Build SPLADE indexes for all columns using custom facets"""
        # Initialize custom column provider (only column name and column name + description)
        custom_provider = CustomColumnFacetProvider(database_id, column_df)
        
        # Store reference for metadata access later
        self.column_provider = ColumnFacetProvider(
            self.memory_config.get("semantic", {}).get("search", {}).get(
                "anchor_column", {}
            )
        )
        self.column_provider.attach(database_id, self.semantic_store.dataframes)

        # Build SPLADE indexes using custom facets
        # Load model if not already loaded
        self.splade_matcher._load_model()
        self.splade_matcher.splade_indexes = {}
        
        # Build index for each custom facet
        for facet in ["column_name", "column_name_with_desc"]:
            self._build_custom_facet_index(facet, custom_provider)
    
    def _build_custom_facet_index(self, facet: str, provider: CustomColumnFacetProvider) -> None:
        """Build SPLADE index for a custom facet"""
        from pathlib import Path
        import pickle
        
        # Check if cached index exists (use custom cache file name)
        cache_file = self.splade_matcher.cache_path / f"splade_index_pseudo_query_{facet}.pkl"
        if cache_file.exists():
            try:
                logger.debug(f"Loading cached SPLADE index for facet '{facet}'")
                with open(cache_file, 'rb') as f:
                    self.splade_matcher.splade_indexes[facet] = pickle.load(f)
                return
            except Exception as e:
                logger.warning(f"Failed to load cached index: {e}, rebuilding...")
        
        texts = []
        column_ids = []
        
        # Collect texts for this facet
        for col_id in provider.iter_column_ids():
            txt = provider.get_faceted_texts(col_id).get(facet, "")
            if txt and txt.strip():
                texts.append(txt)
                column_ids.append(col_id)
        
        if not texts:
            logger.warning(f"No texts for facet '{facet}'")
            return
        
        try:
            # Generate SPLADE representations for all documents
            logger.debug(f"Generating SPLADE representations for {len(texts)} documents (facet: {facet})...")
            doc_representations = []
            
            # Process documents in batches to avoid memory issues
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_reps = self.splade_matcher._encode_documents(batch_texts)
                doc_representations.extend(batch_reps)
            
            # Build inverted index from sparse representations
            inverted_index = self.splade_matcher._build_inverted_index(doc_representations, column_ids)
            
            # Store index data
            self.splade_matcher.splade_indexes[facet] = {
                'inverted_index': inverted_index,
                'doc_representations': doc_representations,
                'column_ids': column_ids,
                'texts': texts
            }
            
            # Cache the index
            with open(cache_file, 'wb') as f:
                pickle.dump(self.splade_matcher.splade_indexes[facet], f)
            
            logger.debug(f"Built SPLADE index for facet '{facet}' with {len(texts)} documents")
            
        except Exception as e:
            logger.error(f"Failed to build SPLADE index for facet '{facet}': {e}")

    # ------------------------------------------------------------------
    # PK/FK relationship loading
    # ------------------------------------------------------------------
    def _load_pk_fk_relationships(self, column_df: pd.DataFrame) -> None:
        """Load primary key and foreign key relationships for filtering"""
        self._pk_fk_relationships = set()

        # Get table metadata
        table_df = self.semantic_store.dataframes.get("table")
        if table_df is None or table_df.empty:
            logger.warning("No table metadata available for PK/FK filtering")
            return

        # Get database_id for column_id generation
        database_id = self.current_database_id or ""
        
        # Build column_id -> table_name mapping
        col_to_table: Dict[str, str] = {}
        for _, row in column_df.iterrows():
            table_name = row.get("table_name")
            column_name = row.get("column_name")
            if table_name and column_name:
                # Use full format: database_id.table.column
                col_id = f"{database_id}.{table_name}.{column_name}" if database_id else f"{table_name}.{column_name}"
                col_to_table[col_id] = table_name

        # Extract PK relationships (same table)
        for _, row in column_df.iterrows():
            table_name = row.get("table_name")
            column_name = row.get("column_name")
            is_pk = row.get("is_primary_key", False)
            
            if not table_name or not column_name:
                continue

            if is_pk:
                # Mark all columns in the same table as PK-related
                for _, other_row in column_df.iterrows():
                    other_table = other_row.get("table_name")
                    other_col = other_row.get("column_name")
                    if (
                        other_table == table_name
                        and other_col
                        and other_col != column_name
                    ):
                        col_id1 = f"{database_id}.{table_name}.{column_name}" if database_id else f"{table_name}.{column_name}"
                        col_id2 = f"{database_id}.{table_name}.{other_col}" if database_id else f"{table_name}.{other_col}"
                        # Add both directions
                        self._pk_fk_relationships.add((col_id1, col_id2))
                        self._pk_fk_relationships.add((col_id2, col_id1))

        # Extract FK relationships from table metadata
        for _, table_row in table_df.iterrows():
            table_name = table_row.get("table_name")
            foreign_keys = table_row.get("foreign_keys", [])
            
            if not table_name or not foreign_keys:
                continue

            for fk_info in foreign_keys:
                if isinstance(fk_info, dict):
                    # Support both normalized keys and SQLite PRAGMA keys
                    ref_table = fk_info.get("referenced_table") or fk_info.get("table")
                    ref_column = fk_info.get("referenced_column") or fk_info.get("to")
                    fk_column = fk_info.get("column") or fk_info.get("from")
                    
                    if ref_table and ref_column and fk_column:
                        col_id1 = f"{database_id}.{table_name}.{fk_column}" if database_id else f"{table_name}.{fk_column}"
                        col_id2 = f"{database_id}.{ref_table}.{ref_column}" if database_id else f"{ref_table}.{ref_column}"
                        # Add both directions
                        self._pk_fk_relationships.add((col_id1, col_id2))
                        self._pk_fk_relationships.add((col_id2, col_id1))

        logger.info(
            "Loaded %d PK/FK relationships for filtering", len(self._pk_fk_relationships)
        )

    def _is_pk_fk_relationship(self, col_id1: str, col_id2: str) -> bool:
        """Check if two columns have a PK/FK relationship"""
        if self._pk_fk_relationships is None:
            return False
        return (col_id1, col_id2) in self._pk_fk_relationships

    def _get_column_metadata_for_logging(self, column_id: str) -> Dict[str, Any]:
        """Get column metadata for logging purposes"""
        if not self.column_provider:
            return {}
        
        try:
            # Parse column_id: database_id.table.column or table.column
            parts = column_id.split(".")
            if len(parts) == 3:
                _, table_name, column_name = parts
            elif len(parts) == 2:
                table_name, column_name = parts
            else:
                return {}
            
            meta = self.column_provider.get_column_metadata(table_name, column_name)
            meta["table_name"] = table_name
            meta["column_name"] = column_name
            
            # Clean up None/NaN values
            for key, value in meta.items():
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    meta[key] = "N/A"
                elif isinstance(value, str) and value.lower() in ["nan", "none", "null", ""]:
                    meta[key] = "N/A"
            
            return meta
        except Exception as e:
            logger.debug(f"Failed to get metadata for {column_id}: {e}")
            return {}

    # ------------------------------------------------------------------
    # Query generation
    # ------------------------------------------------------------------
    def _generate_queries_for_column(
        self, column_key: ColumnKey, column_df: pd.DataFrame
    ) -> List[GeneratedQuery]:
        """
        Generate queries for a single column using LLM.
        
        Generates 3 search phrases:
        1. Exact Match #1: Direct phrase about the column
        2. Exact Match #2: Another direct phrase about the column
        3. Concept-Based: Abstract/conceptual phrase that might trigger semantic confusion
        """
        # Get column metadata
        col_row = column_df[
            (column_df["table_name"] == column_key.table_name)
            & (column_df["column_name"] == column_key.column_name)
        ]
        
        if col_row.empty:
            return []

        row = col_row.iloc[0]
        description = row.get("description") or ""
        column_name = row.get("column_name") or ""
        
        # Build prompt using PromptFactory
        prompt = PromptFactory.format_query_generation_prompt(
            column_name=column_name,
            description=description
        )

        try:
            # Call LLM
            response = self.llm_client.call_with_messages(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # Higher temperature for diversity
            )

            # Parse response
            queries = self._parse_query_response(response)
            
            logger.debug(
                "Generated %d queries for column %s",
                len(queries),
                column_key.to_id(),
            )
            
            return queries

        except Exception as e:
            logger.warning(
                "Failed to generate queries for column %s: %s", column_key.to_id(), e
            )
            return []

    def _build_query_generation_prompt(
        self, table_name: str, column_name: str, description: str
    ) -> str:
        """
        Build the prompt for query generation.
        
        DEPRECATED: Use PromptFactory.format_query_generation_prompt instead.
        This method is kept for backward compatibility.
        """
        return PromptFactory.format_query_generation_prompt(
            column_name=column_name,
            description=description
        )

    def _parse_query_response(self, response: str) -> List[GeneratedQuery]:
        """Parse LLM response to extract queries"""
        queries: List[GeneratedQuery] = []
        
        try:
            # Try to extract JSON array
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
            
            # Parse JSON
            query_list = json.loads(response)
            
            if not isinstance(query_list, list):
                logger.warning("LLM response is not a list: %s", response[:100])
                return []
            
            # Map to query types: 2 exact_match, 1 concept_based
            query_types = ["exact_match", "exact_match", "concept_based"]
            
            for i, query_text in enumerate(query_list):
                if isinstance(query_text, str) and query_text.strip():
                    query_type = query_types[i] if i < len(query_types) else "unknown"
                    queries.append(
                        GeneratedQuery(
                            query_text=query_text.strip(), query_type=query_type
                        )
                    )
            
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM response as JSON: %s", e)
            logger.debug("Response was: %s", response[:200])
            
            # Fallback: try to extract queries from text
            lines = response.split("\n")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#") and len(line) < 100:
                    # Assume it's a query
                    queries.append(
                        GeneratedQuery(query_text=line, query_type="unknown")
                    )
        
        return queries

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------
    def _detect_collisions(
        self,
        column_keys: List[ColumnKey],
        column_df: pd.DataFrame,
        database_id: str,
    ) -> List[CollisionResult]:
        """
        Generate queries for all columns and detect collisions.
        
        Uses parallel processing for efficiency.
        """
        all_collisions: List[CollisionResult] = []
        
        # Process columns in batches (temporarily set to 1 for review)
        with ThreadPoolExecutor(max_workers=1) as executor: # max_workers=self.batch_size
            # Submit all tasks
            future_to_column = {
                executor.submit(
                    self._process_single_column, column_key, column_df
                ): column_key
                for column_key in column_keys
            }
            
            # Collect results
            for future in as_completed(future_to_column):
                column_key = future_to_column[future]
                try:
                    collisions = future.result()
                    all_collisions.extend(collisions)
                except Exception as e:
                    logger.warning(
                        "Failed to process column %s: %s", column_key.to_id(), e
                    )
        
        logger.info("Detected %d total collisions", len(all_collisions))
        return all_collisions

    def _process_single_column(
        self, column_key: ColumnKey, column_df: pd.DataFrame
    ) -> List[CollisionResult]:
        """Process a single column: generate queries and detect collisions"""
        collisions: List[CollisionResult] = []
        
        # Generate queries
        queries = self._generate_queries_for_column(column_key, column_df)
        if not queries:
            return collisions
        
        # Use full column_id format (database_id.table.column) to match SPLADE results
        target_column_id = column_key.to_id(self.current_database_id)
        
        # Get target column metadata for logging
        target_meta = self._get_column_metadata_for_logging(target_column_id)
        
        logger.info(
            "\n" + "="*80 + "\n"
            f"Processing Column: {target_column_id}\n"
            f"  Table: {column_key.table_name}\n"
            f"  Column: {column_key.column_name}\n"
            f"  Description: {target_meta.get('description', 'N/A')}\n"
            + "="*80
        )
        
        # For each generated query, retrieve and check for collisions
        for query_idx, query in enumerate(queries, 1):
            try:
                logger.info(
                    f"\n--- Query {query_idx}/{len(queries)} for column {column_key.column_name}: '{query.query_text}' "
                    f"(type: {query.query_type}) ---"
                )
                
                # Retrieve top-K results using SPLADE
                results = self.splade_matcher.search(
                    query.query_text, top_k=self.top_k_retrieval
                )
                
                if not results:
                    logger.info("  No results retrieved")
                    continue
                
                logger.info(f"  Retrieved {len(results)} results:")
                
                # Check if target column is in results
                target_found = False
                target_score = 0.0
                target_rank = -1
                
                # First pass: find target
                for rank, result in enumerate(results, 1):
                    if result.column_id == target_column_id:
                        target_found = True
                        target_score = result.score
                        target_rank = rank
                        break
                
                # Second pass: log all results with relative score info (if target found)
                for rank, result in enumerate(results, 1):
                    result_meta = self._get_column_metadata_for_logging(result.column_id)
                    is_target = result.column_id == target_column_id
                    
                    if is_target:
                        marker = ">>> TARGET <<<"
                        relative_info = ""
                    else:
                        marker = ""
                        if target_found and target_score > 0:
                            relative_ratio = result.score / target_score
                            relative_diff_pct = abs(result.score - target_score) / target_score * 100
                            relative_info = f" (ratio: {relative_ratio:.3f}, diff: {relative_diff_pct:.1f}%)"
                        else:
                            relative_info = ""
                    
                    logger.info(
                        f"    [{rank}] {result.column_id} (score: {result.score:.3f}, "
                        f"facet: {result.facet}){relative_info} {marker}\n"
                        f"        Table: {result_meta.get('table_name', 'N/A')}\n"
                        f"        Column: {result_meta.get('column_name', 'N/A')}\n"
                        f"        Description: {result_meta.get('description', 'N/A')}\n"
                    )
                
                # If target not found in Top-K, discard this query's results
                if not target_found:
                    logger.info(
                        f"  ⚠️  Target column {target_column_id} not found in Top-{self.top_k_retrieval}, "
                        f"discarding this query's results"
                    )
                    continue
                
                logger.info(
                    f"  ✓ Target found at rank {target_rank} with score {target_score:.3f}"
                )
                
                # Check for distractor hits
                distractor_count = 0
                for rank, result in enumerate(results):
                    distractor_id = result.column_id
                    distractor_score = result.score
                    
                    # Skip target itself
                    if distractor_id == target_column_id:
                        continue
                    
                    # Check collision conditions using relative thresholds (suitable for SPLADE)
                    # SPLADE scores are sparse vector inner products, not normalized to [0,1]
                    # So we use relative comparisons instead of absolute thresholds
                    
                    score_diff = abs(distractor_score - target_score)
                    relative_diff = score_diff / target_score if target_score > 0 else float('inf')
                    relative_ratio = distractor_score / target_score if target_score > 0 else 0.0
                    
                    # Collision conditions (any of these):
                    # 1. Distractor score >= target score (distractor is as good or better)
                    # 2. Distractor score is within relative threshold of target (e.g., >= 70% of target)
                    # 3. Relative score difference is small (e.g., <= 30% of target)
                    condition1 = distractor_score >= target_score
                    condition2 = relative_ratio >= self.relative_score_threshold
                    condition3 = relative_diff <= self.relative_score_diff_threshold
                    is_collision = condition1 or condition2 or condition3
                    
                    # Optional: Check absolute threshold if configured (for backward compatibility)
                    # But this is generally not recommended for SPLADE scores
                    if self.collision_score_threshold is not None:
                        if distractor_score < self.collision_score_threshold:
                            continue
                    
                    if is_collision:
                        # Determine which condition triggered the collision
                        trigger_reasons = []
                        if condition1:
                            trigger_reasons.append("distractor >= target")
                        if condition2:
                            trigger_reasons.append(f"ratio >= {self.relative_score_threshold:.2f}")
                        if condition3:
                            trigger_reasons.append(f"rel_diff <= {self.relative_score_diff_threshold:.2f}")
                        trigger_reason = " | ".join(trigger_reasons)
                        distractor_count += 1
                        # All query types are semantic ambiguity (exact_match or concept_based)
                        collision_type = "semantic_ambiguity"
                        
                        distractor_meta = self._get_column_metadata_for_logging(distractor_id)
                        relative_diff_pct = (score_diff / target_score * 100) if target_score > 0 else 0.0
                        logger.info(
                            f"  🚨 COLLISION #{distractor_count} detected:\n"
                            f"      Distractor: {distractor_id}\n"
                            f"      Distractor Score: {distractor_score:.3f} "
                            f"(Target: {target_score:.3f}, Diff: {score_diff:.3f})\n"
                            f"      Relative Ratio: {relative_ratio:.3f} "
                            f"(Relative Diff: {relative_diff_pct:.1f}%)\n"
                            f"      Trigger: {trigger_reason}\n"
                            f"      Type: {collision_type}\n"
                            f"      Distractor Description: {distractor_meta.get('description', 'N/A')}\n"
                        )
                        
                        collisions.append(
                            CollisionResult(
                                source_column_id=target_column_id,
                                distractor_column_id=distractor_id,
                                trigger_query=query.query_text,
                                query_type=query.query_type,
                                target_score=target_score,
                                distractor_score=distractor_score,
                                collision_type=collision_type,
                            )
                        )
                
                if distractor_count == 0:
                    logger.info("  No collisions detected for this query")
            
            except Exception as e:
                logger.warning(
                    "Failed to process query '%s' for column %s: %s",
                    query.query_text,
                    target_column_id,
                    e,
                )
                continue
        
        logger.info(f"\nTotal collisions for {target_column_id}: {len(collisions)}\n")
        return collisions

    # ------------------------------------------------------------------
    # Pair construction
    # ------------------------------------------------------------------
    def _build_pairs_from_collisions(
        self,
        collisions: List[CollisionResult],
        database_id: str,
    ) -> List[AmbiguousPair]:
        """
        Build ambiguous pairs directly from collisions.
        
        Each collision represents one pair of ambiguous columns.
        No clustering - we store the actual conflicting pairs.
        """
        if not collisions:
            return []

        # Filter out PK/FK relationships
        filtered_collisions = [
            c
            for c in collisions
            if not self._is_pk_fk_relationship(
                c.source_column_id, c.distractor_column_id
            )
        ]
        
        logger.info(
            "Filtered %d collisions (removed %d PK/FK relationships)",
            len(filtered_collisions),
            len(collisions) - len(filtered_collisions),
        )

        if not filtered_collisions:
            return []

        # Group collisions by pair (deduplicate bidirectional collisions)
        pair_collisions: Dict[Tuple[Tuple[str, str], Tuple[str, str]], List[CollisionResult]] = defaultdict(list)
        
        for collision in filtered_collisions:
            # Parse column IDs
            source_parts = collision.source_column_id.split(".")
            distractor_parts = collision.distractor_column_id.split(".")
            
            # Handle both formats: database_id.table.column or table.column
            if len(source_parts) == 3:
                _, source_table, source_col = source_parts
            elif len(source_parts) == 2:
                source_table, source_col = source_parts
            else:
                continue
                
            if len(distractor_parts) == 3:
                _, distractor_table, distractor_col = distractor_parts
            elif len(distractor_parts) == 2:
                distractor_table, distractor_col = distractor_parts
            else:
                continue
            
            # Create canonical pair key (sorted)
            key_a = (source_table, source_col)
            key_b = (distractor_table, distractor_col)
            pair_key = tuple(sorted([key_a, key_b]))
            
            pair_collisions[pair_key].append(collision)

        # Build AmbiguousPair objects
        pairs: List[AmbiguousPair] = []
        pair_idx = 0

        for pair_key, collision_list in pair_collisions.items():
            (table_a, col_a), (table_b, col_b) = pair_key
            
            # Build collision details
            collision_details: List[CollisionInfo] = []
            collision_scores = []
            
            for collision in collision_list:
                collision_details.append(
                    CollisionInfo(
                        trigger_query=collision.trigger_query,
                        source_column_id=collision.source_column_id,
                        distractor_column_id=collision.distractor_column_id,
                        collision_score=collision.distractor_score,
                        target_score=collision.target_score,
                        query_type=collision.query_type,
                        collision_type=collision.collision_type,
                    )
                )
                collision_scores.append(collision.distractor_score)
            
            # Compute average collision score
            avg_collision_score = sum(collision_scores) / len(collision_scores) if collision_scores else None
            
            # Create pair
            pair_id = f"{database_id}_pseudo_query_pair_{pair_idx:04d}"
            pair_idx += 1
            
            pairs.append(
                AmbiguousPair(
                    pair_id=pair_id,
                    database_id=database_id,
                    column_a=DBElementRef(table_name=table_a, column_name=col_a),
                    column_b=DBElementRef(table_name=table_b, column_name=col_b),
                    discovery_methods=["pseudo_query_collision"],
                    semantic_collision_score=avg_collision_score,
                    collision_details=collision_details,
                )
            )

        logger.info(
            "Built %d unique ambiguous pairs from %d collisions",
            len(pairs),
            len(filtered_collisions),
        )

        return pairs


