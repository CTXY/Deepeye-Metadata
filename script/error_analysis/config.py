"""
Configuration for Error Analysis System
"""

import os
from pathlib import Path

class Config:
    """Configuration class for error analysis system"""
    
    # Project paths
    PROJECT_ROOT = Path("/home/yangchenyu/DeepEye-SQL-Metadata")
    
    # Data paths
    BIRD_DEV_JSON = PROJECT_ROOT / "data/bird/dev/dev.json"
    INCORRECT_RESULTS_JSON = PROJECT_ROOT / "results/bird-dev/qwen3-coder-30b-a3b_incorrect.json"
    
    # Output paths
    OUTPUT_DIR = PROJECT_ROOT / "output/error_analysis"
    INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
    INSIGHTS_FILE = OUTPUT_DIR / "insights.jsonl"
    
    # Database mapping
    DB_MAPPING_PATH = PROJECT_ROOT / "dataset/bird/database_mapping.json"
    
    # LLM Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = "gpt-4o-mini"
    OPENAI_TEMPERATURE = 0.0
    OPENAI_MAX_TOKENS = 2048
    
    # Processing configuration
    BATCH_SIZE = 10
    DELAY_BETWEEN_REQUESTS = 0.5  # seconds
    MAX_RETRIES = 3
    
    # Schema error threshold
    # If schema overlap < threshold, consider it as pure schema error
    SCHEMA_OVERLAP_THRESHOLD = 1.0
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure all output directories exist"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)















