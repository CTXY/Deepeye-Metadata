# Base feedback interface class

from abc import ABC, abstractmethod
from ..types import FeedbackContext, UserFeedback

class BaseFeedbackInterface(ABC):
    """Base class for feedback interfaces"""
    
    def __init__(self, config: dict):
        self.config = config
    
    @abstractmethod
    def collect_feedback(self, context: FeedbackContext) -> UserFeedback:
        """Collect user feedback"""
        pass
    
    def cleanup(self) -> None:
        """Cleanup interface resources (optional implementation)"""
        pass
