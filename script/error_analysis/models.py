"""
Pydantic models for structured data
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ComparativeStrategy(BaseModel):
    """Strategy comparison for incorrect vs correct SQL"""
    pattern: str = Field(..., description="The abstract SQL pattern used")
    implication: str = Field(..., description="The side effect or logical consequence")


class GuidanceStructure(BaseModel):
    """Structured guidance for error correction"""
    intent: str = Field(..., description="The user's logical goal")
    strategy_incorrect: ComparativeStrategy
    strategy_correct: ComparativeStrategy
    actionable_advice: str = Field(..., description="The rule for future SQL generation")


class RetrievalKey(BaseModel):
    """Key information for retrieving similar error patterns"""
    nl_triggers: List[str] = Field(..., description="Keywords from NLQ triggering this logic")
    sql_risk_atoms: List[str] = Field(..., description="Keywords from Incorrect SQL representing the risk")


class MinerOutput(BaseModel):
    """Complete output from Miner Agent"""
    retrieval_key: RetrievalKey
    guidance: GuidanceStructure
    is_generalizable: bool = Field(default=True, description="Whether this error pattern is generalizable across databases")
    generalizability_reason: str = Field(default="", description="Brief reason for generalizability judgment")


class ProcessedSample(BaseModel):
    """Processed sample data structure"""
    question_id: int
    nlq: str
    evidence: Optional[str] = ""
    
    # Original SQLs
    incorrect_sql: str
    correct_sql: str
    db_path: str
    
    # Qualified SQLs (after alias resolution)
    qualified_incorrect_sql: Optional[str] = None
    qualified_correct_sql: Optional[str] = None
    
    # Masked SQLs
    masked_incorrect_sql: Optional[str] = None
    masked_correct_sql: Optional[str] = None
    mapping_dict: Optional[Dict[str, str]] = None  # Original -> Masked mapping
    
    # Schema information
    incorrect_schema: Optional[Dict[str, Any]] = None
    correct_schema: Optional[Dict[str, Any]] = None
    is_pure_schema_error: Optional[bool] = None
    schema_overlap_score: Optional[float] = None
    
    # Miner output
    miner_output: Optional[MinerOutput] = None
    
    # Verification result
    verification_passed: Optional[bool] = None
    verification_details: Optional[Dict[str, Any]] = None
    
    # Processing status
    processing_status: str = "pending"  # pending, schema_error, processed, failed
    error_message: Optional[str] = None


class Insight(BaseModel):
    """Final insight structure for output"""
    insight_id: str
    retrieval_key: RetrievalKey
    guidance: GuidanceStructure
    
    # Example SQLs for understanding (qualified versions with table names)
    qualified_incorrect_sql: Optional[str] = Field(None, description="Example of incorrect SQL (qualified)")
    qualified_correct_sql: Optional[str] = Field(None, description="Example of correct SQL (qualified)")
    
    # Supporting data
    source_question_ids: List[int] = Field(default_factory=list)
    verification_success_count: int = 0
    verification_total_count: int = 0
    verification_success_rate: float = 0.0
    
    # Metadata
    created_at: str
    
    class Config:
        json_encoders = {
            'datetime': lambda v: v.isoformat()
        }


