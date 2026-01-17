import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from .column_indexer import ColumnFacetProvider
from caf.config.paths import PathConfig

logger = logging.getLogger(__name__)


class ColumnValueIndex:
    """
    Lightweight per-database column value index.

    It stores, for each column_id ("{database_id}.{table}.{column}"),
    a list of distinct values as strings.

    This index is intended to be:
    - Built from the real database once
    - Serialized to disk for later reuse
    - Consumed by ValueMatcher (inverted index + optional LSH)
    """

    def __init__(self, values_by_column: Dict[str, List[str]]):
        self.values_by_column = values_by_column or {}

    def get_values(self, column_id: str) -> List[str]:
        return self.values_by_column.get(column_id, [])

    def iter_columns(self) -> List[str]:
        return list(self.values_by_column.keys())


class ValueIndexBuilder:
    """
    Helper for building and loading ColumnValueIndex.

    Responsibilities:
    - Read database_mapping.json and resolve database_id -> sqlite path
    - For each (table, column) coming from ColumnFacetProvider, run a
      DISTINCT scan on the underlying database column (with a configurable limit)
    - Persist the collected values to disk for later reuse
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Where to store per-database value index files
        default_root = Path("./memory/value_indexes")
        self.index_root: Path = Path(self.config.get("index_root", default_root))
        self.index_root.mkdir(parents=True, exist_ok=True)

        # Path to database_mapping.json
        # Default to centralized PathConfig location to avoid hardcoded paths.
        default_mapping = PathConfig.get_database_mapping_path()
        self.mapping_path: Path = Path(
            self.config.get("database_mapping_path", default_mapping)
        )

        # Max distinct values fetched per column when building index
        self.max_distinct_per_column: int = int(self.config.get("max_distinct_per_column", 50000))

        self._dbid_to_path: Optional[Dict[str, str]] = None

        logger.debug(
            "ValueIndexBuilder initialized: index_root=%s, mapping_path=%s, max_distinct_per_column=%d",
            self.index_root,
            self.mapping_path,
            self.max_distinct_per_column,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_or_load(self, database_id: str, provider: ColumnFacetProvider) -> ColumnValueIndex:
        """
        Load an existing value index for the given database_id if present,
        otherwise build it from the underlying sqlite database and persist it.
        """
        index_path = self._get_index_file_path(database_id)
        if index_path.exists():
            try:
                logger.info("Loading existing value index from %s", index_path)
                with index_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                # Keys are column_ids, values are lists of strings
                if isinstance(data, dict):
                    return ColumnValueIndex(
                        {str(k): [str(vv) for vv in (v or [])] for k, v in data.items()}
                    )
            except Exception as exc:
                logger.warning("Failed to load value index from %s: %s. Rebuilding.", index_path, exc)

        # Need to build from scratch
        db_path = self._resolve_database_path(database_id)
        if not db_path:
            logger.warning("No database path found for database_id=%s; returning empty value index", database_id)
            return ColumnValueIndex({})

        logger.info("Building value index for database_id=%s from %s", database_id, db_path)
        values_by_column: Dict[str, List[str]] = {}

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
        except Exception as exc:
            logger.error("Failed to open database %s for value index build: %s", db_path, exc)
            return ColumnValueIndex({})

        try:
            for col_id in provider.iter_column_ids():
                _, table_name, column_name = provider.split_column_id(col_id)
                column_values = self._fetch_distinct_values_for_column(conn, table_name, column_name)
                if column_values:
                    values_by_column[col_id] = column_values
        finally:
            conn.close()

        # Persist to disk for later reuse
        try:
            serializable = {k: list(v) for k, v in values_by_column.items()}
            with index_path.open("w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False)
            logger.info(
                "Value index built for database_id=%s with %d columns and saved to %s",
                database_id,
                len(values_by_column),
                index_path,
            )
        except Exception as exc:
            logger.warning("Failed to persist value index to %s: %s", index_path, exc)

        return ColumnValueIndex(values_by_column)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_index_file_path(self, database_id: str) -> Path:
        safe_id = str(database_id).replace("/", "_")
        return self.index_root / f"{safe_id}_values.json"

    def _load_mapping(self) -> None:
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
            logger.error("Failed to load database mapping from %s: %s", self.mapping_path, exc)
            self._dbid_to_path = {}

    def _resolve_database_path(self, database_id: str) -> Optional[str]:
        self._load_mapping()
        if not self._dbid_to_path:
            return None
        return self._dbid_to_path.get(str(database_id))

    def _fetch_distinct_values_for_column(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
    ) -> List[str]:
        """
        Fetch a (possibly truncated) list of distinct non-null values for a single column.

        This is only used to build the value index and is independent of the profiling
        statistics stored in semantic metadata.
        """
        try:
            limit = self.max_distinct_per_column
            query = f"""
                SELECT DISTINCT `{column_name}` as value
                FROM `{table_name}`
                WHERE `{column_name}` IS NOT NULL
                LIMIT {limit}
            """
            cursor = conn.execute(query)
            values: List[str] = []
            for row in cursor:
                v = row["value"]
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    values.append(s)
            return values
        except Exception as exc:
            logger.debug(
                "Failed to fetch distinct values for %s.%s during value index build: %s",
                table_name,
                column_name,
                exc,
            )
            return []


