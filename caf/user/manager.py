# User Feedback Manager - core feedback collection and management

import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from .types import FeedbackContext, UserFeedback
from .interfaces.base import BaseFeedbackInterface
from .interfaces.llm import LLMFeedbackInterface
from .interfaces.cli import CLIFeedbackInterface
from .exceptions import UserFeedbackError, InvalidFeedbackContextError
from ..config.loader import CAFConfig

logger = logging.getLogger(__name__)

class UserFeedbackManager:
    """
    User Feedback Manager - CAF system's internal component
    
    Handles feedback collection through different interfaces and integrates
    tightly with memory system.
    """
    
    def __init__(self, config: Dict[str, Any], memory_base=None):
        self.config = config
        self.memory_base = memory_base  # Tight integration with memory
        self._interface = self._create_interface()
        
        logger.info("UserFeedbackManager initialized")
    
    def _create_interface(self) -> BaseFeedbackInterface:
        """Create specific feedback interface based on configuration"""
        interface_type = self.config.get('interface_type', 'cli')
        
        if interface_type == 'cli':
            return CLIFeedbackInterface(self.config)
        elif interface_type == 'llm':
            return LLMFeedbackInterface(self.config)
        else:
            raise UserFeedbackError("UNSUPPORTED_INTERFACE", f"Unsupported interface type: {interface_type}")
    
    def collect_feedback(self, context: FeedbackContext) -> UserFeedback:
        """
        Collect user feedback - simplified version without complex events
        """
        # Validate context
        self._validate_context(context)
        
        # Collect feedback through interface
        start_time = time.time()
        
        try:
            feedback = self._interface.collect_feedback(context)
            
            # Supplement metadata
            response_time = int((time.time() - start_time) * 1000)
            feedback.response_time_ms = response_time
            feedback.feedback_id = str(uuid.uuid4())
            
            if not feedback.timestamp:
                feedback.timestamp = datetime.utcnow().isoformat()
            
            logger.info(f"Feedback collected successfully: {feedback.feedback_id}")
            return feedback
            
        except Exception as e:
            logger.error(f"Failed to collect feedback: {e}")
            raise UserFeedbackError("COLLECTION_FAILED", f"Feedback collection failed: {e}")
    
    def _validate_context(self, context: FeedbackContext) -> None:
        """Validate feedback context"""
        if not context.user_query:
            raise InvalidFeedbackContextError("user_query is required")
        
        if not context.generated_sql:
            raise InvalidFeedbackContextError("generated_sql is required")
        
        # For LLM mode, additional validation
        if self.config.get('interface_type') == 'llm':
            if context.feedback_type.value == 'sql_validation':
                if not context.ground_truth_sql:
                    logger.warning("ground_truth_sql not provided for LLM SQL validation")
                
                if not context.db_schema:
                    logger.warning("db_schema not provided for LLM SQL validation")
    
    def get_interface_type(self) -> str:
        """Get current interface type"""
        return self.config.get('interface_type', 'cli')
    
    def cleanup(self) -> None:
        """Cleanup feedback manager resources"""
        if hasattr(self._interface, 'cleanup'):
            self._interface.cleanup()
        
        logger.info("UserFeedbackManager cleanup completed")
