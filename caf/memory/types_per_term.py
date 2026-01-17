# Per-Query-Term Grouped Response Models for Semantic Search
# Use these models when return_per_term=True

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class ColumnItem(BaseModel):
    """Single column item with metadata and value matches"""
    schema_ref: str                                    # table.column format (renamed from 'schema' to avoid BaseModel conflict)
    table_name: str
    column_name: str
    column_metadata: Dict[str, Any]                   # Full column metadata
    value_matches: Optional[Dict[str, Any]] = None    # matched_values, match_types, encoding_mappings
    

class TermItem(BaseModel):
    """Single term item with definition and related info"""
    term_id: str
    term_name: str
    definition: Optional[str] = None
    formula: Optional[str] = None
    example_usage: Optional[str] = None
    related_tables: Optional[List[str]] = None
    related_columns: Optional[List[str]] = None
    schemas: Optional[List[str]] = None
    signal_scores: Optional[Dict[str, float]] = None
    explanation: Optional[str] = None
    score: float = 1.0


class JoinRelationship(BaseModel):
    """JOIN relationship between tables"""
    table1: str
    column1: str
    table2: str
    column2: str
    join_type: str              # 'foreign_key', 'semantic_similarity', 'value_match'
    confidence: float = 1.0


class QueryTermResult(BaseModel):
    """Results for a single query term"""
    query_term: str
    column_items: List[ColumnItem] = []
    term_items: List[TermItem] = []


class PerTermMemoryResponse(BaseModel):
    """Memory query response grouped by query term (for return_per_term=True)"""
    query_term_results: List[QueryTermResult]       # Results grouped by query term
    all_tables: List[str]                            # All tables involved
    all_schemas: List[str]                           # All schemas (table.column) involved


class FlatMemoryResponse(BaseModel):
    """
    Memory query response as a flat list (for return_per_term=False)
    
    This is a simplified version of PerTermMemoryResponse without query term grouping.
    All columns and terms are returned as flat lists for easy consumption.
    """
    column_items: List[ColumnItem] = []              # All column items (flat, not grouped by term)
    term_items: List[TermItem] = []                  # All term items (flat, not grouped by term)
    all_tables: List[str] = []                       # All tables involved
    all_schemas: List[str] = []                      # All schemas (table.column) involved
    join_relationships: List[JoinRelationship] = []  # Join relationships between tables
    query_time_ms: int = 0                           # Query time in milliseconds
