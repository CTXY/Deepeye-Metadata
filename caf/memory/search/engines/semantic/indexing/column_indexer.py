# Lightweight Column Facet Provider - Zero-copy facet provider over metadata DataFrames

import logging
from typing import Dict, Any, List, Optional, Iterable, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

class ColumnFacetProvider:
    """
    A lightweight, zero-copy facet provider over your metadata DataFrames.
    It does NOT duplicate/serialize column metadata. All facet texts/values
    are computed on-the-fly from `dataframes['table']` and `dataframes['column']`.

    column_id format stays: "{database_id}.{table_name}.{column_name}"
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.database_id: Optional[str] = None
        self.dataframes: Optional[Dict[str, pd.DataFrame]] = None
        self.max_values_per_facet = self.config.get('max_values_per_facet', 5)
        self.max_description_length = self.config.get('max_description_length', 500)

        # internal indices for fast lookup (no copies of payload)
        self._col_df: Optional[pd.DataFrame] = None
        self._tbl_df: Optional[pd.DataFrame] = None
        # maps (table_name, column_name) -> row_index
        self._col_idx: Dict[Tuple[str, str], int] = {}
        # maps table_name -> row_index
        self._tbl_idx: Dict[str, int] = {}
        logger.debug("ColumnFacetProvider initialized")

    def attach(self, database_id: str, dataframes: Dict[str, pd.DataFrame]) -> None:
        self.database_id = database_id
        self.dataframes = dataframes
        self._col_df = dataframes.get('column')
        self._tbl_df = dataframes.get('table')

        if self._col_df is None or self._col_df.empty:
            logger.warning("No column metadata found in dataframes['column']")
            return

        # build lightweight indices
        self._col_idx.clear()
        for i, row in self._col_df.iterrows():
            t = row.get('table_name'); c = row.get('column_name')
            if t and c:
                self._col_idx[(str(t), str(c))] = i

        self._tbl_idx.clear()
        if self._tbl_df is not None and not self._tbl_df.empty:
            for i, row in self._tbl_df.iterrows():
                t = row.get('table_name')
                if t:
                    self._tbl_idx[str(t)] = i

        logger.info(f"Facet provider attached: {len(self._col_idx)} columns indexed (lazy)")

    # ---------- ID / parts ----------
    def make_column_id(self, table_name: str, column_name: str) -> str:
        return f"{self.database_id}.{table_name}.{column_name}"

    def split_column_id(self, column_id: str) -> Tuple[str, str, str]:
        # returns (database_id, table_name, column_name)
        parts = column_id.split('.', 2)
        if len(parts) != 3:
            raise ValueError(f"Bad column_id: {column_id}")
        return parts[0], parts[1], parts[2]

    # ---------- iteration ----------
    def iter_column_ids(self) -> Iterable[str]:
        if not self._col_df is None:
            for (t, c) in self._col_idx.keys():
                yield self.make_column_id(t, c)

    # ---------- raw row access ----------
    def _get_col_row(self, table_name: str, column_name: str) -> Optional[pd.Series]:
        if (table_name, column_name) not in self._col_idx: return None
        return self._col_df.loc[self._col_idx[(table_name, column_name)]]

    def _get_tbl_row(self, table_name: str) -> Optional[pd.Series]:
        if table_name not in self._tbl_idx: return None
        return self._tbl_df.loc[self._tbl_idx[table_name]]

    # ---------- public metadata helpers ----------
    def get_column_metadata(self, table_name: str, column_name: str) -> Dict[str, Any]:
        row = self._get_col_row(table_name, column_name)
        if row is None: return {}
        return {
            "description": row.get("description", "") or "",
            "data_type": row.get("data_type", "") or "",
            "whole_column_name": row.get("whole_column_name", "") or "",
            "encoding_mapping": row.get("encoding_mapping", {}) or {},
            "data_format": row.get("data_format", "") or "",
            "has_nulls": row.get("is_nullable") or row.get("null_count") > 0,
        }

    def get_table_metadata(self, table_name: str) -> Dict[str, Any]:
        row = self._get_tbl_row(table_name)
        if row is None: return {}
        return {
            "table_name": row.get("table_name"),
            "primary_keys": row.get("primary_keys", []) or [],
            "foreign_keys": row.get("foreign_keys", []) or [],
            "description": row.get("description", "") or ""
        }

    def get_distinct_values(self, table_name: str, column_name: str) -> List[str]:
        """
        Get value examples for a column from top_k_values.
        
        We use top_k_values (most frequent values) which are sufficient for:
        - LLM prompts
        - BM25 values facet
        
        Note: distinct_values field has been removed from the schema.
        """
        row = self._get_col_row(table_name, column_name)
        if row is None:
            return []

        # Use top_k_values as the only source
        top_k = row.get("top_k_values")
        if isinstance(top_k, dict) and top_k:
            values = list(top_k.keys())
        else:
            # No top_k_values available
            return []

        out: List[str] = []
        for v in values:
            s = str(v).strip()
            if s:
                out.append(s)
        return out

    def get_encoding_mapping(self, table_name: str, column_name: str) -> Dict[str, str]:
        """Get encoding mapping for a column if it exists"""
        row = self._get_col_row(table_name, column_name)
        if row is None: return {}
        mapping = row.get("encoding_mapping", {})
        if mapping is None or not isinstance(mapping, dict):
            return {}
        return mapping

    # ---------- facet texts (lazy, zero-copy) ----------
    def get_faceted_texts(self, column_id: str) -> Dict[str, str]:
        _, t, c = self.split_column_id(column_id)
        col_meta = self.get_column_metadata(t, c)
        tbl_meta = self.get_table_metadata(t)

        # names facet
        names = []
        if c: names.append(str(c))
        wc = col_meta.get("whole_column_name")
        if wc and wc != c: names.append(str(wc))
        names_str = " ".join(dict.fromkeys([s for s in names if s]))

        # description facet
        parts = []
        wc_disp = wc or c or ""
        if wc_disp: parts.append(f"Column {wc_disp}")
        parts.append(f"Belong to Table {t}")
        if col_meta.get("description"):
            parts.append(f"Column Description: {col_meta['description']}")
        if tbl_meta.get("description"):
            parts.append(f"Table Description: {tbl_meta['description']}")
        desc = ". ".join(parts) + "."

        # values facet (sampled)
        vals = self.get_distinct_values(t, c)[: self.max_values_per_facet]
        vals_str = " ".join(vals) if vals else ""

        return {
            "names": names_str,
            "description": desc[: self.max_description_length] if desc else "",
            "values": vals_str
        }
