# Data types and models for User Feedback subsystem

from enum import Enum
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class FeedbackType(Enum):
    """Feedback type enumeration"""
    SQL_VALIDATION = "sql_validation"      # SQL correctness validation
    RESULT_QUALITY = "result_quality"      # Result quality assessment
    CLARIFICATION = "clarification"        # Clarification questions
    PREFERENCE = "preference"              # User preferences

class FeedbackContext(BaseModel):
    """Feedback context for LLM evaluation"""
    feedback_type: FeedbackType
    user_query: str                        # Original query
    generated_sql: str                     # Generated SQL
    execution_result: Optional[Dict] = None # Execution result
    options: Optional[List[str]] = None    # Choice options (for multi-choice feedback)
    
    # LLM feedback mode support
    ground_truth_sql: Optional[str] = None  # Ground truth SQL (for LLM mode)
    db_schema: Optional[str] = None         # Database schema info (for LLM mode)
    timeout_seconds: Optional[int] = 300    # Feedback timeout
    
    def dict(self, **kwargs):
        """Override dict method to handle FeedbackType serialization"""
        data = super().dict(**kwargs)
        # Convert FeedbackType enum to string
        if 'feedback_type' in data and isinstance(data['feedback_type'], FeedbackType):
            data['feedback_type'] = data['feedback_type'].value
        return data

class UserFeedback(BaseModel):
    """User feedback response"""
    is_correct: bool                       # Whether SQL is correct
    error_category: Optional[str] = None   # Error category
    error_subcategory: Optional[str] = None # Error subcategory  
    analysis: Optional[str] = None         # Error analysis
    suggestion: Optional[str] = None       # Correction suggestion
    
    # Metadata
    feedback_id: Optional[str] = None      # Unique feedback ID
    response_time_ms: Optional[int] = None # Response time
    timestamp: Optional[str] = None        # Timestamp
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.utcnow().isoformat()
        super().__init__(**data)
