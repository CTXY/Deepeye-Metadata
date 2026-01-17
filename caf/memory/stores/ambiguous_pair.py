import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Iterable, Set, Tuple

import pandas as pd

from ..types import AmbiguousPair, DBElementRef, CollisionInfo, DiffProfile, DataContentProfile, SemanticIntentProfile

logger = logging.getLogger(__name__)


class AmbiguousPairStore:
    """
    Store for ambiguous column pairs within a database.

    Design goals:
    - Pairwise storage (not group/cluster) for precise disambiguation
    - Per-database storage using Parquet file
    - Supports deduplication: same pair from different miners
    - Lightweight column references with rich analysis results (DiffProfile)
    """

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.current_database_id: Optional[str] = None
        self._pairs_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def bind_database(self, database_id: str) -> None:
        """
        Bind to a specific database and load its pairs from disk.
        """
        if self.current_database_id == database_id and self._pairs_df is not None:
            return

        self.current_database_id = database_id
        self._pairs_df = self._load_pairs_for_db(database_id)

        logger.info(
            "AmbiguousPairStore bound to database: %s (pairs=%d)",
            database_id,
            0 if self._pairs_df is None else len(self._pairs_df),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def save_pairs(self, database_id: str, pairs: Iterable[AmbiguousPair]) -> None:
        """
        Replace all pairs for a database with the provided list.
        """
        df = self._pairs_to_dataframe(database_id, list(pairs))
        self._save_pairs_for_db(database_id, df)

        # If this DB is currently bound, update in-memory cache as well.
        if self.current_database_id == database_id:
            self._pairs_df = df

        logger.info(
            "Saved %d ambiguous pairs for database: %s",
            len(df),
            database_id,
        )

    def append_pairs(
        self, 
        database_id: str, 
        pairs: Iterable[AmbiguousPair],
        deduplicate: bool = True
    ) -> None:
        """
        Append pairs to existing ones for a database.

        Args:
            database_id: Database identifier
            pairs: Pairs to append
            deduplicate: If True, merge duplicate pairs (same column pair from different miners)
        """
        new_df = self._pairs_to_dataframe(database_id, list(pairs))
        existing_df = self._load_pairs_for_db(database_id)

        if existing_df is not None and not existing_df.empty:
            if deduplicate:
                combined = self._merge_duplicate_pairs(existing_df, new_df)
            else:
                combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df

        self._save_pairs_for_db(database_id, combined)

        if self.current_database_id == database_id:
            self._pairs_df = combined

        logger.info(
            "Appended %d ambiguous pairs for database: %s (total=%d, deduplicated=%s)",
            len(new_df),
            database_id,
            len(combined),
            deduplicate,
        )

    def list_pairs(self) -> List[AmbiguousPair]:
        """
        List all pairs for the currently bound database.
        """
        if not self._ensure_bound():
            return []

        df = self._pairs_df
        if df is None or df.empty:
            return []

        return [self._row_to_pair(row) for _, row in df.iterrows()]

    def get_pair(self, table_a: str, column_a: str, table_b: str, column_b: str) -> Optional[AmbiguousPair]:
        """
        Get a specific pair by column identifiers.
        
        Order-agnostic: (A, B) and (B, A) are considered the same pair.
        """
        if not self._ensure_bound():
            return None

        df = self._pairs_df
        if df is None or df.empty:
            return None

        # Try both orderings
        for _, row in df.iterrows():
            col_a = row.get("column_a")
            col_b = row.get("column_b")
            
            if isinstance(col_a, dict) and isinstance(col_b, dict):
                # Check both orderings
                match_forward = (
                    col_a.get("table_name") == table_a and
                    col_a.get("column_name") == column_a and
                    col_b.get("table_name") == table_b and
                    col_b.get("column_name") == column_b
                )
                match_backward = (
                    col_a.get("table_name") == table_b and
                    col_a.get("column_name") == column_b and
                    col_b.get("table_name") == table_a and
                    col_b.get("column_name") == column_a
                )
                
                if match_forward or match_backward:
                    return self._row_to_pair(row)

        return None

    def get_pairs_for_column(self, table_name: str, column_name: str) -> List[AmbiguousPair]:
        """
        Get all pairs that contain the specified column.
        """
        if not self._ensure_bound():
            return []

        pairs: List[AmbiguousPair] = []
        df = self._pairs_df
        if df is None or df.empty:
            return []

        for _, row in df.iterrows():
            col_a = row.get("column_a")
            col_b = row.get("column_b")
            
            if isinstance(col_a, dict) and isinstance(col_b, dict):
                contains_column = (
                    (col_a.get("table_name") == table_name and col_a.get("column_name") == column_name) or
                    (col_b.get("table_name") == table_name and col_b.get("column_name") == column_name)
                )
                
                if contains_column:
                    pairs.append(self._row_to_pair(row))

        return pairs

    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about stored pairs.
        """
        if not self._ensure_bound():
            return {}

        df = self._pairs_df
        if df is None or df.empty:
            return {"total_pairs": 0}

        # Count by discovery method
        method_counts = {}
        for _, row in df.iterrows():
            methods = row.get("discovery_methods", [])
            if isinstance(methods, list):
                for method in methods:
                    method_counts[method] = method_counts.get(method, 0) + 1

        # Count analyzed vs unanalyzed
        analyzed_count = 0
        for _, row in df.iterrows():
            if row.get("diff_profile") is not None:
                analyzed_count += 1

        return {
            "total_pairs": len(df),
            "analyzed_pairs": analyzed_count,
            "unanalyzed_pairs": len(df) - analyzed_count,
            "discovery_method_counts": method_counts,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_bound(self) -> bool:
        if not self.current_database_id:
            logger.warning("AmbiguousPairStore used before bind_database()")
            return False
        if self._pairs_df is None:
            self._pairs_df = self._load_pairs_for_db(self.current_database_id)
        return True

    def _pairs_file_for_db(self, database_id: str) -> Path:
        return self.storage_path / f"ambiguous_pairs_{database_id}.parquet"

    def _load_pairs_for_db(self, database_id: str) -> Optional[pd.DataFrame]:
        file_path = self._pairs_file_for_db(database_id)
        if not file_path.exists():
            return pd.DataFrame()

        try:
            df = pd.read_parquet(file_path)
            logger.debug(
                "Loaded %d ambiguous pairs for database: %s",
                len(df),
                database_id,
            )
            return df
        except Exception as e:
            logger.error("Failed to load ambiguous pairs for %s: %s", database_id, e)
            return pd.DataFrame()

    def _save_pairs_for_db(self, database_id: str, df: pd.DataFrame) -> None:
        file_path = self._pairs_file_for_db(database_id)
        try:
            df.to_parquet(file_path, index=False)
        except Exception as e:
            logger.error("Failed to save ambiguous pairs for %s: %s", database_id, e)
            raise

    def _pairs_to_dataframe(
        self, database_id: str, pairs: List[AmbiguousPair]
    ) -> pd.DataFrame:
        if not pairs:
            return pd.DataFrame(
                columns=[
                    "pair_id",
                    "database_id",
                    "column_a",
                    "column_b",
                    "discovery_methods",
                    "semantic_collision_score",
                    "value_jaccard",
                    "collision_details",
                    "diff_profile",
                    "discovered_at",
                    "last_analyzed_at",
                ]
            )

        rows = []
        for p in pairs:
            # Ensure database_id consistency
            db_id = p.database_id or database_id
            
            # Serialize diff_profile to JSON string for Parquet compatibility
            diff_profile_json = None
            if p.diff_profile:
                diff_profile_json = json.dumps(p.diff_profile.dict())
            
            # Serialize collision_details to JSON string for Parquet compatibility
            collision_details_json = None
            if p.collision_details:
                collision_details_json = json.dumps([ci.dict() for ci in p.collision_details])
            
            rows.append(
                {
                    "pair_id": p.pair_id,
                    "database_id": db_id,
                    "column_a": {
                        "table_name": p.column_a.table_name,
                        "column_name": p.column_a.column_name,
                    },
                    "column_b": {
                        "table_name": p.column_b.table_name,
                        "column_name": p.column_b.column_name,
                    },
                    "discovery_methods": list(p.discovery_methods or []),
                    "semantic_collision_score": p.semantic_collision_score,
                    "value_jaccard": p.value_jaccard,
                    "collision_details": collision_details_json,
                    "diff_profile": diff_profile_json,
                    "discovered_at": p.discovered_at,
                    "last_analyzed_at": p.last_analyzed_at,
                }
            )

        df = pd.DataFrame(rows)
        return df

    def _row_to_pair(self, row: pd.Series) -> AmbiguousPair:
        # Parse column_a
        col_a_dict = row.get("column_a")
        if isinstance(col_a_dict, dict):
            column_a = DBElementRef(
                table_name=col_a_dict.get("table_name"),
                column_name=col_a_dict.get("column_name"),
            )
        else:
            column_a = col_a_dict

        # Parse column_b
        col_b_dict = row.get("column_b")
        if isinstance(col_b_dict, dict):
            column_b = DBElementRef(
                table_name=col_b_dict.get("table_name"),
                column_name=col_b_dict.get("column_name"),
            )
        else:
            column_b = col_b_dict

        # Parse collision_details from JSON string
        collision_details_raw = row.get("collision_details")
        collision_details: Optional[List[CollisionInfo]] = None
        if collision_details_raw:
            if isinstance(collision_details_raw, str):
                # Deserialize from JSON string
                try:
                    collision_details_list = json.loads(collision_details_raw)
                    collision_details = [CollisionInfo(**ci) for ci in collision_details_list]
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning("Failed to parse collision_details JSON: %s", e)
            elif isinstance(collision_details_raw, list):
                # Legacy format: direct list of dicts
                collision_details = [
                    CollisionInfo(**ci) if isinstance(ci, dict) else ci
                    for ci in collision_details_raw
                ]

        # Parse diff_profile from JSON string
        diff_profile_raw = row.get("diff_profile")
        diff_profile: Optional[DiffProfile] = None
        if diff_profile_raw:
            if isinstance(diff_profile_raw, str):
                # Deserialize from JSON string
                try:
                    diff_profile_dict = json.loads(diff_profile_raw)
                    diff_profile = DiffProfile(**diff_profile_dict)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning("Failed to parse diff_profile JSON: %s", e)
            elif isinstance(diff_profile_raw, dict):
                # Legacy format: direct dict
                diff_profile = DiffProfile(**diff_profile_raw)

        return AmbiguousPair(
            pair_id=row.get("pair_id"),
            database_id=row.get("database_id"),
            column_a=column_a,
            column_b=column_b,
            discovery_methods=list(row.get("discovery_methods") or []),
            semantic_collision_score=row.get("semantic_collision_score"),
            value_jaccard=row.get("value_jaccard"),
            collision_details=collision_details,
            diff_profile=diff_profile,
            discovered_at=row.get("discovered_at"),
            last_analyzed_at=row.get("last_analyzed_at"),
        )

    def _merge_duplicate_pairs(self, existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge duplicate pairs from different miners.
        
        For pairs with the same (column_a, column_b):
        - Merge discovery_methods lists
        - Take max of semantic_collision_score
        - Take max of value_jaccard
        - Merge collision_details lists
        - Keep existing diff_profile if present
        """
        if existing_df.empty:
            return new_df
        if new_df.empty:
            return existing_df

        # Build index of existing pairs
        existing_pairs: Dict[Tuple[Tuple[str, str], Tuple[str, str]], int] = {}
        for idx, row in existing_df.iterrows():
            col_a = row.get("column_a")
            col_b = row.get("column_b")
            if isinstance(col_a, dict) and isinstance(col_b, dict):
                key_a = (col_a.get("table_name"), col_a.get("column_name"))
                key_b = (col_b.get("table_name"), col_b.get("column_name"))
                pair_key = tuple(sorted([key_a, key_b]))
                existing_pairs[pair_key] = idx

        # Merge new pairs
        rows_to_append = []
        for _, new_row in new_df.iterrows():
            col_a = new_row.get("column_a")
            col_b = new_row.get("column_b")
            if not isinstance(col_a, dict) or not isinstance(col_b, dict):
                continue

            key_a = (col_a.get("table_name"), col_a.get("column_name"))
            key_b = (col_b.get("table_name"), col_b.get("column_name"))
            pair_key = tuple(sorted([key_a, key_b]))

            if pair_key in existing_pairs:
                # Merge with existing pair
                existing_idx = existing_pairs[pair_key]
                existing_row = existing_df.loc[existing_idx]

                # Merge discovery_methods
                existing_methods = set(existing_row.get("discovery_methods") or [])
                new_methods = set(new_row.get("discovery_methods") or [])
                merged_methods = list(existing_methods | new_methods)

                # Take max scores
                semantic_score = max(
                    existing_row.get("semantic_collision_score") or 0.0,
                    new_row.get("semantic_collision_score") or 0.0,
                ) or None
                
                value_jaccard = max(
                    existing_row.get("value_jaccard") or 0.0,
                    new_row.get("value_jaccard") or 0.0,
                ) or None

                # Merge collision_details (handle JSON string format)
                existing_collisions_raw = existing_row.get("collision_details")
                new_collisions_raw = new_row.get("collision_details")
                
                # Parse existing collisions
                existing_collisions = []
                if existing_collisions_raw:
                    if isinstance(existing_collisions_raw, str):
                        try:
                            existing_collisions = json.loads(existing_collisions_raw)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    elif isinstance(existing_collisions_raw, list):
                        existing_collisions = existing_collisions_raw
                
                # Parse new collisions
                new_collisions = []
                if new_collisions_raw:
                    if isinstance(new_collisions_raw, str):
                        try:
                            new_collisions = json.loads(new_collisions_raw)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    elif isinstance(new_collisions_raw, list):
                        new_collisions = new_collisions_raw
                
                # Merge and serialize back to JSON string
                merged_collisions = None
                if existing_collisions or new_collisions:
                    merged_collisions = json.dumps(existing_collisions + new_collisions)

                # Update existing row
                existing_df.at[existing_idx, "discovery_methods"] = merged_methods
                existing_df.at[existing_idx, "semantic_collision_score"] = semantic_score
                existing_df.at[existing_idx, "value_jaccard"] = value_jaccard
                existing_df.at[existing_idx, "collision_details"] = merged_collisions
            else:
                # Add new pair
                rows_to_append.append(new_row)

        if rows_to_append:
            new_pairs_df = pd.DataFrame(rows_to_append)
            return pd.concat([existing_df, new_pairs_df], ignore_index=True)
        else:
            return existing_df

