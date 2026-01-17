import logging
from pathlib import Path
from typing import Dict, List, Optional, Iterable

import pandas as pd

from ..types import SimilarityCluster, DBElementRef, CollisionInfo

logger = logging.getLogger(__name__)


class SimilarityClusterStore:
    """
    Store for similarity clusters within a database.

    Design goals:
    - Lightweight, not part of MemoryBase routing (not a MemoryStore).
    - Per-database storage using a single Parquet/CSV file.
    - Only stores lightweight column references (table_name, column_name)
      plus coarse statistics for debugging/analysis.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.current_database_id: Optional[str] = None
        self._clusters_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def bind_database(self, database_id: str) -> None:
        """
        Bind to a specific database and load its clusters from disk.

        This does NOT interact with MemoryBase; callers are responsible
        for keeping database_id consistent with semantic memory.
        """
        if self.current_database_id == database_id and self._clusters_df is not None:
            return

        self.current_database_id = database_id
        self._clusters_df = self._load_clusters_for_db(database_id)

        logger.info(
            "SimilarityClusterStore bound to database: %s (clusters=%d)",
            database_id,
            0 if self._clusters_df is None else len(self._clusters_df),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def save_clusters(self, database_id: str, clusters: Iterable[SimilarityCluster]) -> None:
        """
        Replace all clusters for a database with the provided list.
        """
        df = self._clusters_to_dataframe(database_id, list(clusters))
        self._save_clusters_for_db(database_id, df)

        # If this DB is currently bound, update in-memory cache as well.
        if self.current_database_id == database_id:
            self._clusters_df = df

        logger.info(
            "Saved %d similarity clusters for database: %s",
            len(df),
            database_id,
        )

    def append_clusters(self, database_id: str, clusters: Iterable[SimilarityCluster]) -> None:
        """
        Append clusters to existing ones for a database.

        NOTE: This method does not perform deduplication; the caller is
        expected to manage cluster_id uniqueness.
        """
        new_df = self._clusters_to_dataframe(database_id, list(clusters))
        existing_df = self._load_clusters_for_db(database_id)

        if existing_df is not None and not existing_df.empty:
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df

        self._save_clusters_for_db(database_id, combined)

        if self.current_database_id == database_id:
            self._clusters_df = combined

        logger.info(
            "Appended %d similarity clusters for database: %s (total=%d)",
            len(new_df),
            database_id,
            len(combined),
        )

    def list_clusters(self, min_size: int = 2) -> List[SimilarityCluster]:
        """
        List all clusters for the currently bound database.
        """
        if not self._ensure_bound():
            return []

        df = self._clusters_df
        if df is None or df.empty:
            return []

        # Filter by cluster size if we have elements stored as lists
        if "elements" in df.columns and min_size > 1:
            mask = df["elements"].apply(
                lambda elems: isinstance(elems, list) and len(elems) >= min_size
            )
            df = df[mask]

        return [self._row_to_cluster(row) for _, row in df.iterrows()]

    def get_clusters_for_column(self, table_name: str, column_name: str) -> List[SimilarityCluster]:
        """
        Get all clusters that contain the specified column.

        This implementation is intentionally simple and scans all clusters.
        If this becomes a bottleneck, we can add an explicit reverse index.
        """
        if not self._ensure_bound():
            return []

        key = (table_name, column_name)
        matches: List[SimilarityCluster] = []

        df = self._clusters_df
        if df is None or df.empty:
            return []

        for _, row in df.iterrows():
            elements = row.get("elements") or []
            for elem in elements:
                try:
                    if (
                        isinstance(elem, dict)
                        and elem.get("table_name") == key[0]
                        and elem.get("column_name") == key[1]
                    ):
                        matches.append(self._row_to_cluster(row))
                        break
                    elif isinstance(elem, DBElementRef):
                        if elem.table_name == key[0] and elem.column_name == key[1]:
                            matches.append(self._row_to_cluster(row))
                            break
                except Exception:
                    continue

        return matches

    def get_similar_columns(
        self, table_name: str, column_name: str
    ) -> List[Dict[str, str]]:
        """
        Get all columns that appear in the same clusters as the given column.

        Returns a list of dicts with keys: table_name, column_name, cluster_id.
        """
        clusters = self.get_clusters_for_column(table_name, column_name)
        results: List[Dict[str, str]] = []
        target = (table_name, column_name)

        for cluster in clusters:
            for elem in cluster.elements:
                if elem.table_name == target[0] and elem.column_name == target[1]:
                    continue
                results.append(
                    {
                        "table_name": elem.table_name,
                        "column_name": elem.column_name,
                        "cluster_id": cluster.cluster_id,
                    }
                )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_bound(self) -> bool:
        if not self.current_database_id:
            logger.warning("SimilarityClusterStore used before bind_database()")
            return False
        if self._clusters_df is None:
            self._clusters_df = self._load_clusters_for_db(self.current_database_id)
        return True

    def _clusters_file_for_db(self, database_id: str) -> Path:
        return self.storage_path / f"similarity_clusters_{database_id}.parquet"

    def _load_clusters_for_db(self, database_id: str) -> Optional[pd.DataFrame]:
        file_path = self._clusters_file_for_db(database_id)
        if not file_path.exists():
            return pd.DataFrame()

        try:
            df = pd.read_parquet(file_path)
            logger.debug(
                "Loaded %d similarity clusters for database: %s",
                len(df),
                database_id,
            )
            return df
        except Exception as e:
            logger.error("Failed to load similarity clusters for %s: %s", database_id, e)
            return pd.DataFrame()

    def _save_clusters_for_db(self, database_id: str, df: pd.DataFrame) -> None:
        file_path = self._clusters_file_for_db(database_id)
        try:
            df.to_parquet(file_path, index=False)
        except Exception as e:
            logger.error("Failed to save similarity clusters for %s: %s", database_id, e)
            raise

    def _clusters_to_dataframe(
        self, database_id: str, clusters: List[SimilarityCluster]
    ) -> pd.DataFrame:
        if not clusters:
            return pd.DataFrame(
                columns=[
                    "cluster_id",
                    "database_id",
                    "elements",
                    "methods",
                    "semantic_score_min",
                    "semantic_score_max",
                    "semantic_score_avg",
                    "value_jaccard_min",
                    "value_jaccard_max",
                    "value_overlap_min",
                    "value_overlap_max",
                    "collision_info",
                ]
            )

        rows = []
        for c in clusters:
            # Ensure database_id consistency
            db_id = c.database_id or database_id
            rows.append(
                {
                    "cluster_id": c.cluster_id,
                    "database_id": db_id,
                    "elements": [
                        {"table_name": e.table_name, "column_name": e.column_name}
                        for e in c.elements
                    ],
                    "methods": list(c.methods or []),
                    "semantic_score_min": c.semantic_score_min,
                    "semantic_score_max": c.semantic_score_max,
                    "semantic_score_avg": c.semantic_score_avg,
                    "value_jaccard_min": c.value_jaccard_min,
                    "value_jaccard_max": c.value_jaccard_max,
                    "value_overlap_min": c.value_overlap_min,
                    "value_overlap_max": c.value_overlap_max,
                    "collision_info": (
                        [ci.dict() for ci in c.collision_info] 
                        if c.collision_info else None
                    ),
                }
            )

        df = pd.DataFrame(rows)
        return df

    def _row_to_cluster(self, row: pd.Series) -> SimilarityCluster:
        elements_raw = row.get("elements") or []
        elements: List[DBElementRef] = []
        for elem in elements_raw:
            if isinstance(elem, DBElementRef):
                elements.append(elem)
            elif isinstance(elem, dict):
                table = elem.get("table_name")
                column = elem.get("column_name")
                if table and column:
                    elements.append(DBElementRef(table_name=table, column_name=column))

        # Parse collision_info
        collision_info_raw = row.get("collision_info")
        collision_info: Optional[List[CollisionInfo]] = None
        if collision_info_raw:
            if isinstance(collision_info_raw, list):
                collision_info = [
                    CollisionInfo(**ci) if isinstance(ci, dict) else ci
                    for ci in collision_info_raw
                ]
            elif isinstance(collision_info_raw, str):
                # Handle JSON string case
                import json
                try:
                    collision_info_list = json.loads(collision_info_raw)
                    collision_info = [CollisionInfo(**ci) for ci in collision_info_list]
                except (json.JSONDecodeError, TypeError):
                    collision_info = None

        return SimilarityCluster(
            cluster_id=row.get("cluster_id"),
            database_id=row.get("database_id"),
            elements=elements,
            methods=list(row.get("methods") or []),
            semantic_score_min=row.get("semantic_score_min"),
            semantic_score_max=row.get("semantic_score_max"),
            semantic_score_avg=row.get("semantic_score_avg"),
            value_jaccard_min=row.get("value_jaccard_min"),
            value_jaccard_max=row.get("value_jaccard_max"),
            value_overlap_min=row.get("value_overlap_min"),
            value_overlap_max=row.get("value_overlap_max"),
            collision_info=collision_info,
        )









