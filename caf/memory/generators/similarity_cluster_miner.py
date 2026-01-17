import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Set

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from ..stores.semantic import SemanticMemoryStore
from ..stores.similarity_cluster import SimilarityClusterStore
from ..types import SimilarityCluster, DBElementRef

logger = logging.getLogger(__name__)


SEMANTIC_THRESHOLD = 0.9  # Similarity threshold for identifying similar columns


@dataclass
class ColumnKey:
    table_name: str
    column_name: str

    def to_id(self) -> str:
        return f"{self.table_name}.{self.column_name}"


class SimilarityClusterMiner:
    """
    Miner for discovering similarity clusters within a single database.

    Responsibilities:
    - READ column metadata from SemanticMemoryStore (no writes).
    - Use SPLADE (Sparse Lexical and Dense) to identify semantically similar columns.
    - Optionally use MinHash/value overlap (if available) to refine clusters.
    - Materialize clusters into SimilarityClusterStore.
    """

    def __init__(
        self,
        semantic_store: SemanticMemoryStore,
        cluster_store: SimilarityClusterStore,
        memory_config: Dict,
    ):
        self.semantic_store = semantic_store
        self.cluster_store = cluster_store
        self.memory_config = memory_config or {}

        # Similarity threshold configuration
        similarity_cfg = self.memory_config.get("similarity", {})
        self.semantic_threshold = similarity_cfg.get(
            "semantic_threshold", SEMANTIC_THRESHOLD
        )

        # SPLADE model configuration
        # Read SPLADE config from semantic.search.splade (not from embedding config)
        splade_cfg = (
            self.memory_config.get("semantic", {})
            .get("search", {})
            .get("splade", {})
        )
        
        # Fallback to defaults if splade config not found
        if not splade_cfg:
            model_name = "/home/yangchenyu/pre-trained-models/splade-v3"
            model_path = model_name
            device_str = "cpu"
            self.batch_size = 32
        else:
            model_name = splade_cfg.get("model_name", "/home/yangchenyu/pre-trained-models/splade-v3")
            model_path = splade_cfg.get("model_path", model_name)
            device_str = splade_cfg.get("device", "cpu")
            self.batch_size = splade_cfg.get("batch_size", 32)
        
        # Device setup
        if device_str == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_str)
        
        # Initialize SPLADE model
        self.splade_model = None
        self.splade_tokenizer = None
        self._load_splade_model(model_path)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def mine_and_save_clusters(
        self, database_id: str, use_value_overlap: bool = True
    ) -> List[SimilarityCluster]:
        """
        High-level API:
        - Bind semantic & cluster stores.
        - Mine clusters for the given database.
        - Save results to cluster store.
        """
        logger.info(
            "Starting similarity cluster mining for database: %s", database_id
        )

        self.semantic_store.bind_database(database_id)
        self.cluster_store.bind_database(database_id)

        column_df = self.semantic_store.dataframes.get("column")
        if column_df is None or column_df.empty:
            logger.warning(
                "No column metadata available for database: %s", database_id
            )
            clusters: List[SimilarityCluster] = []
            self.cluster_store.save_clusters(database_id, clusters)
            return clusters

        column_keys, texts = self._build_column_texts(column_df)
        if not column_keys:
            logger.warning(
                "No columns eligible for similarity mining in database: %s",
                database_id,
            )
            clusters = []
            self.cluster_store.save_clusters(database_id, clusters)
            return clusters

        # Debug: log sample texts
        logger.info("Sample column texts (first 5):")
        for i, (ck, text) in enumerate(zip(column_keys[:5], texts[:5])):
            logger.info("  [%d] %s: %s", i, ck.to_id(), text[:100] + "..." if len(text) > 100 else text)
        
        # Encode texts using SPLADE
        splade_representations = self._encode_splade_batch(texts)

        semantic_pairs = self._find_semantic_pairs_splade(column_keys, splade_representations, column_df)
        if not semantic_pairs:
            logger.info(
                "No semantic-similar column pairs found for database: %s",
                database_id,
            )
            clusters = []
            self.cluster_store.save_clusters(database_id, clusters)
            return clusters

        # Placeholder for future value-based refinement
        # If use_value_overlap is False, we only rely on semantic pairs.
        pair_scores = {
            (ck1.to_id(), ck2.to_id()): score for ck1, ck2, score in semantic_pairs
        }

        clusters = self._build_clusters_from_pairs(column_keys, semantic_pairs, pair_scores, database_id)
        self.cluster_store.save_clusters(database_id, clusters)

        logger.info(
            "Finished similarity cluster mining for database: %s (clusters=%d)",
            database_id,
            len(clusters),
        )
        return clusters

    # ------------------------------------------------------------------
    # Column text & embedding
    # ------------------------------------------------------------------
    def _build_column_texts(
        self, column_df: pd.DataFrame
    ) -> Tuple[List[ColumnKey], List[str]]:
        """
        Build natural-language texts for each column using table/column
        name and description, plus additional metadata if available.
        """
        keys: List[ColumnKey] = []
        texts: List[str] = []

        for _, row in column_df.iterrows():
            table_name = row.get("table_name")
            column_name = row.get("column_name")
            if not table_name or not column_name:
                continue

            description = row.get("description") or ""

            text = f"{column_name}: {description}"

            keys.append(ColumnKey(table_name=table_name, column_name=column_name))
            texts.append(text)

        return keys, texts

    # ------------------------------------------------------------------
    # SPLADE model loading
    # ------------------------------------------------------------------
    def _load_splade_model(self, model_path: str):
        """Load SPLADE model and tokenizer"""
        try:
            logger.info(f"Loading SPLADE model: {model_path}")
            
            # Check if model_path is a local path
            model_path_obj = Path(model_path)
            if model_path_obj.exists() and model_path_obj.is_dir():
                logger.info(f"Loading SPLADE model from local path: {model_path_obj}")
                self.splade_tokenizer = AutoTokenizer.from_pretrained(str(model_path_obj))
                self.splade_model = AutoModelForMaskedLM.from_pretrained(str(model_path_obj))
            else:
                logger.info(f"Loading SPLADE model from HuggingFace Hub: {model_path}")
                self.splade_tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.splade_model = AutoModelForMaskedLM.from_pretrained(model_path)
            
            self.splade_model.to(self.device)
            self.splade_model.eval()
            
            logger.info("SPLADE model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SPLADE model: {e}")
            raise

    # ------------------------------------------------------------------
    # SPLADE encoding
    # ------------------------------------------------------------------
    def _encode_splade_batch(self, texts: List[str]) -> List[Dict[int, float]]:
        """
        Encode texts using SPLADE model to get sparse representations.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of sparse representations (dict of token_id: weight)
        """
        representations = []
        
        with torch.no_grad():
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Tokenize batch
                inputs = self.splade_tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get model output
                outputs = self.splade_model(**inputs)
                logits = outputs.logits  # [batch_size, seq_len, vocab_size]
                
                # Apply ReLU and aggregation (max pooling over sequence)
                relu_logits = torch.relu(logits)
                sparse_reps = torch.max(relu_logits, dim=1)[0]  # [batch_size, vocab_size]
                
                # Convert to sparse representation (only non-zero values)
                sparse_reps = sparse_reps.cpu().numpy()
                
                for sparse_rep in sparse_reps:
                    non_zero_indices = np.nonzero(sparse_rep)[0]
                    
                    # Create sparse dict representation
                    sparse_dict = {}
                    for idx in non_zero_indices:
                        weight = float(sparse_rep[idx])
                        if weight > 0.01:  # Filter very small weights
                            sparse_dict[int(idx)] = weight
                    
                    representations.append(sparse_dict)
        
        return representations

    def _compute_splade_similarity(
        self, rep1: Dict[int, float], rep2: Dict[int, float]
    ) -> float:
        """
        Compute similarity between two SPLADE sparse representations.
        
        Similarity is computed as the dot product of the sparse vectors.
        This is equivalent to cosine similarity when vectors are normalized,
        but SPLADE weights are already learned to be meaningful.
        
        Args:
            rep1: First sparse representation (dict of token_id: weight)
            rep2: Second sparse representation (dict of token_id: weight)
            
        Returns:
            Similarity score (float)
        """
        # Compute dot product of sparse vectors
        similarity = 0.0
        common_tokens = set(rep1.keys()) & set(rep2.keys())
        
        for token_id in common_tokens:
            similarity += rep1[token_id] * rep2[token_id]
        
        # Normalize by the magnitude of both vectors for cosine similarity
        # This helps normalize the score to [0, 1] range
        norm1 = sum(w * w for w in rep1.values()) ** 0.5
        norm2 = sum(w * w for w in rep2.values()) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Normalized dot product (cosine similarity)
        normalized_similarity = similarity / (norm1 * norm2)
        
        # Clamp to [0, 1] and return
        return float(max(0.0, min(1.0, normalized_similarity)))

    # ------------------------------------------------------------------
    # Semantic similarity with SPLADE
    # ------------------------------------------------------------------
    def _find_semantic_pairs_splade(
        self, 
        column_keys: List[ColumnKey], 
        splade_representations: List[Dict[int, float]], 
        column_df: Optional[pd.DataFrame] = None
    ) -> List[Tuple[ColumnKey, ColumnKey, float]]:
        """
        Use SPLADE sparse representations to find similar column pairs.
        Computes pairwise similarities and filters by threshold.
        """
        if len(column_keys) != len(splade_representations):
            raise ValueError("Number of column keys and SPLADE representations must match")

        n = len(column_keys)
        if n == 0:
            return []

        pair_results: List[Tuple[ColumnKey, ColumnKey, float]] = []
        all_scores = []  # Debug: collect all scores

        # Compute pairwise similarities
        logger.info(f"Computing SPLADE similarities for {n} columns...")
        for i in range(n):
            for j in range(i + 1, n):
                score = self._compute_splade_similarity(
                    splade_representations[i],
                    splade_representations[j]
                )
                all_scores.append(score)

                if score >= self.semantic_threshold:
                    pair_results.append((column_keys[i], column_keys[j], score))

        # Sort pairs by score for logging
        pair_results.sort(key=lambda x: x[2], reverse=True)

        # Debug: log score statistics and examples
        if all_scores:
            scores_array = np.array(all_scores)
            logger.info(
                "SPLADE similarity score statistics: min=%.3f, max=%.3f, mean=%.3f, median=%.3f, "
                "p75=%.3f, p90=%.3f, count_>=0.5=%d, count_>=0.6=%d, count_>=0.7=%d, count_>=0.75=%d",
                scores_array.min(), scores_array.max(), scores_array.mean(), np.median(scores_array),
                np.percentile(scores_array, 75), np.percentile(scores_array, 90),
                np.sum(scores_array >= 0.5), np.sum(scores_array >= 0.6),
                np.sum(scores_array >= 0.7), np.sum(scores_array >= 0.75)
            )
            
            # Show top similarity pairs
            if pair_results and column_df is not None:
                logger.info("Top 10 highest similarity pairs:")
                for i, (ck1, ck2, score) in enumerate(pair_results[:10], 1):
                    # Get descriptions for these columns
                    desc1 = self._get_column_description(column_df, ck1)
                    desc2 = self._get_column_description(column_df, ck2)
                    logger.info(
                        "  [%d] Score=%.3f: %s <-> %s",
                        i, score, ck1.to_id(), ck2.to_id()
                    )
                    if desc1 and isinstance(desc1, str):
                        logger.info("      %s desc: %s", ck1.to_id(), desc1[:100] + "..." if len(desc1) > 100 else desc1)
                    if desc2 and isinstance(desc2, str):
                        logger.info("      %s desc: %s", ck2.to_id(), desc2[:100] + "..." if len(desc2) > 100 else desc2)
            
            # Show some low similarity examples
            if len(all_scores) > 0:
                logger.info("Sample of low similarity scores (bottom 5%):")
                low_threshold = np.percentile(scores_array, 5)
                low_scores = [s for s in all_scores if s < low_threshold]
                if low_scores:
                    for score_val in sorted(low_scores)[:5]:
                        logger.info("  Low score example: %.3f", score_val)
        
        logger.info(
            "Found %d semantic-similar column pairs (>= %.2f) using SPLADE",
            len(pair_results),
            self.semantic_threshold,
        )
        
        # Print all similarity pairs above threshold
        if pair_results:
            logger.info("=" * 80)
            logger.info("All similarity pairs (score >= %.2f): %d pairs", 
                        self.semantic_threshold, len(pair_results))
            logger.info("=" * 80)
            for i, (ck1, ck2, score) in enumerate(pair_results, 1):
                desc1 = self._get_column_description(column_df, ck1) if column_df is not None else None
                desc2 = self._get_column_description(column_df, ck2) if column_df is not None else None
                logger.info(
                    "  [%d] Score=%.3f: %s <-> %s",
                    i, score, ck1.to_id(), ck2.to_id()
                )
                if desc1 and isinstance(desc1, str):
                    logger.info("      %s desc: %s", ck1.to_id(), desc1[:100] + "..." if len(desc1) > 100 else desc1)
                if desc2 and isinstance(desc2, str):
                    logger.info("      %s desc: %s", ck2.to_id(), desc2[:100] + "..." if len(desc2) > 100 else desc2)
        return pair_results

    # ------------------------------------------------------------------
    # Cluster construction
    # ------------------------------------------------------------------
    def _build_clusters_from_pairs(
        self,
        all_columns: List[ColumnKey],
        pairs: List[Tuple[ColumnKey, ColumnKey, float]],
        pair_scores: Dict[Tuple[str, str], float],
        database_id: str,
    ) -> List[SimilarityCluster]:
        """
        Build connected components on the column graph defined by
        semantic pairs. For now we only use semantic similarity;
        value-overlap refinement can be added later.
        """
        if not pairs:
            return []

        # Union-Find structure on column IDs
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            if parent.get(x, x) != x:
                parent[x] = find(parent[x])
            return parent.get(x, x)

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        # Initialize parent for all columns that appear in pairs
        for ck1, ck2, _ in pairs:
            parent.setdefault(ck1.to_id(), ck1.to_id())
            parent.setdefault(ck2.to_id(), ck2.to_id())

        # Union all pairs that pass semantic threshold
        # Note: pairs are already filtered by threshold, but we check again for safety
        for ck1, ck2, score in pairs:
            if score >= self.semantic_threshold:
                union(ck1.to_id(), ck2.to_id())

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

            # Compute semantic statistics within cluster
            scores: List[float] = []
            member_set: Set[str] = set(members)
            for (id1, id2), score in pair_scores.items():
                if id1 in member_set and id2 in member_set:
                    scores.append(score)

            if not scores:
                semantic_min = semantic_max = semantic_avg = None
            else:
                semantic_min = float(min(scores))
                semantic_max = float(max(scores))
                semantic_avg = float(sum(scores) / len(scores))

            # Build DBElementRef list
            elements: List[DBElementRef] = []
            for col_id in sorted(member_set):
                try:
                    table_name, column_name = col_id.split(".", 1)
                except ValueError:
                    continue
                elements.append(
                    DBElementRef(table_name=table_name, column_name=column_name)
                )

            if len(elements) < 2:
                continue

            cluster_id = f"{database_id}_cluster_{cluster_idx:04d}"
            cluster_idx += 1

            clusters.append(
                SimilarityCluster(
                    cluster_id=cluster_id,
                    database_id=database_id,
                    elements=elements,
                    methods=["semantic_name_desc"],
                    semantic_score_min=semantic_min,
                    semantic_score_max=semantic_max,
                    semantic_score_avg=semantic_avg,
                )
            )

        return clusters

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _find_column_key_by_id(
        column_keys: List[ColumnKey], col_id: str
    ) -> Optional[ColumnKey]:
        for ck in column_keys:
            if ck.to_id() == col_id:
                return ck
        return None
    
    @staticmethod
    def _get_column_description(column_df: pd.DataFrame, ck: ColumnKey) -> Optional[str]:
        """Get description for a column from the dataframe"""
        if column_df is None or column_df.empty:
            return None
        matches = column_df[
            (column_df['table_name'] == ck.table_name) & 
            (column_df['column_name'] == ck.column_name)
        ]
        if not matches.empty:
            desc = matches.iloc[0].get('description')
            # Handle NaN values (pandas returns float nan for missing values)
            if desc is None or (isinstance(desc, float) and pd.isna(desc)):
                return None
            # Convert to string and check if it's not empty
            desc_str = str(desc).strip()
            return desc_str if desc_str else None
        return None


