# Cognitive Augmentation Framework (CAF) - Main Package
# 
# This package provides cognitive augmentation capabilities for NL2SQL systems,
# including memory management, user feedback, and session management.

from .system import CAFSystem
from .config import CAFConfig
from .config.global_config import initialize_global_config
from .memory.types import MemoryQuery, MemoryResponse, MemoryType, SQLExecutionResult, UserFeedback as MemoryUserFeedback
from .user.types import FeedbackContext, UserFeedback, FeedbackType
from typing import Optional

def initialize(config_path: Optional[str] = None, config: Optional[CAFConfig] = None) -> CAFSystem:
    """
    Initialize the CAF cognitive augmentation system.
    
    Args:
        config_path: Path to configuration file
        config: Configuration object (alternative to config_path)
    
    Returns:
        CAFSystem: Initialized CAF system instance
    
    Note:
        Global configuration (GlobalConfigManager) is automatically initialized
        by CAFSystem during initialization. No need to call initialize_global_config manually.
    """
    if config is None:
        if config_path is None:
            config = CAFConfig.default()
        else:
            config = CAFConfig.from_file(config_path)
    
    # CAFSystem will automatically initialize global config internally
    return CAFSystem(config)

# Main exports
__all__ = [
    'initialize',
    'CAFSystem',
    'CAFConfig',
    'MemoryQuery', 
    'MemoryResponse', 
    'MemoryType', 
    'SQLExecutionResult',
    'MemoryUserFeedback',
    'FeedbackContext', 
    'UserFeedback', 
    'FeedbackType'
]

__version__ = "0.1.0"
__author__ = "Chenyu Yang"
__description__ = "Cognitive Augmentation Framework for NL2SQL Systems"
