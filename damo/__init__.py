# Simplified NL2SQL Reasoning Module with Direct CAF Usage
# 
# This is a basic example showing how to use NL2SQL with CAF cognitive framework.
# For usage, see main.py which contains the core reasoning logic using CAF's native interfaces.

from .config import NL2SQLConfig  
from .data_handler import BirdDataHandler, DatabaseSchema, BirdDataItem
from .main import nl2sql_reasoning, NL2SQLQuery, NL2SQLResult, NL2SQLResponse, NL2SQLMode

# Simplified exports - just the essentials for users to understand the system
__all__ = [
    'nl2sql_reasoning',    # Main reasoning function with direct CAF usage
    'NL2SQLConfig',       # Configuration
    'NL2SQLQuery',        # Input query model
    'NL2SQLResult',       # Output result model  
    'NL2SQLResponse',     # Complete response model
    'NL2SQLMode',         # Reasoning mode enum
    'BirdDataHandler',    # Data handling utilities
    'DatabaseSchema',     # Database schema model
    'BirdDataItem',       # BIRD dataset item model
]

__version__ = "0.1.0"
__author__ = "Chenyu Yang"
__description__ = "Simplified NL2SQL Reasoning Module with CAF Integration"
