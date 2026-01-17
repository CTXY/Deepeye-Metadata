# Configuration for NL2SQL reasoning module

import os
import sys
import yaml
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path

# Add project root to path to import PathConfig
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from caf.config.paths import PathConfig

class NL2SQLMode(Enum):
    """NL2SQL reasoning mode"""
    LLM_ONLY = "llm_only"

class LLMConfig(BaseModel):
    """Configuration for LLM-based reasoning"""
    provider: str = "openai"           # openai, anthropic, etc.
    model_name: str = "gpt-4"          # Model identifier
    api_key: Optional[str] = None      # API key (can be from env)
    base_url: Optional[str] = None     # Custom base URL for API
    max_tokens: int = 2048             # Maximum tokens for generation
    temperature: float = 0.0           # Temperature for generation
    timeout: int = 30                  # Request timeout in seconds
    
    def __init__(self, **data):
        """Initialize and get API key from environment if not provided"""
        super().__init__(**data)
        if self.api_key is None:
            if self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")

# Removed finetuned config as only LLM approach is supported

class DataConfig(BaseModel):
    """Configuration for data handling"""
    bird_data_path: str = str(PathConfig.BIRD_DATASET_PATH)  # Path to BIRD dataset
    cache_dir: str = str(PathConfig.CACHE_ROOT)              # Cache directory
    use_cache: bool = True                                   # Whether to use cache
    use_database_description: bool = False                   # Whether to include database description files
    
class EvaluationConfig(BaseModel):
    """Configuration for SQL execution and evaluation"""
    timeout: float = 30.0              # SQL execution timeout
    num_cpus: int = 1                  # Number of CPUs for parallel execution
    enable_execution: bool = True      # Whether to execute generated SQL

class CAFConfig(BaseModel):
    """Configuration for CAF integration"""
    enabled: bool = True               # Whether to use CAF
    config_path: Optional[str] = None  # Path to CAF config file
    memory_types: List[str] = ["semantic", "episodic", "procedural"]  # Memory types to use
    feedback_enabled: bool = True      # Whether to collect feedback
    
    # Episodic memory retrieval settings (limits and thresholds only, enable_episodic is in NL2SQLConfig)
    episodic_success_limit: int = 3            # Number of successful cases to retrieve
    episodic_error_limit: int = 3              # Number of error cases to retrieve
    episodic_similarity_threshold: float = 0.3 # Similarity threshold for episodic retrieval
    
    # Semantic memory retrieval settings (limit only, enable_semantic is in NL2SQLConfig)
    semantic_limit: Optional[int] = None       # Limit for semantic memory (None = system default)

class NL2SQLConfig(BaseModel):
    """Main configuration for NL2SQL reasoning module"""
    
    # Core settings
    mode: NL2SQLMode = NL2SQLMode.LLM_ONLY
    
    # Memory retrieval settings (DAMO-level, independent of CAF)
    enable_episodic: bool = True      # Whether to enable episodic memory retrieval
    enable_semantic: bool = True       # Whether to enable semantic memory retrieval
    
    # Sub-configurations
    llm: LLMConfig = LLMConfig()
    data: DataConfig = DataConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    caf: CAFConfig = CAFConfig()
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    @classmethod
    def default(cls) -> 'NL2SQLConfig':
        """Create default configuration"""
        return cls()
    
    @classmethod
    def from_file(cls, config_path: str) -> 'NL2SQLConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # Create instance first
        instance = cls(**config_data)
        
        # Resolve CAF config path relative to the main config file
        if instance.caf.config_path and not os.path.isabs(instance.caf.config_path):
            config_dir = os.path.dirname(os.path.abspath(config_path))
            instance.caf.config_path = os.path.normpath(os.path.join(config_dir, instance.caf.config_path))
        
        return instance
    
    def to_file(self, config_path: str):
        """Save configuration to YAML file"""
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, allow_unicode=True)
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Check mode-specific requirements
        if self.mode == NL2SQLMode.LLM_ONLY:
            if not self.llm.api_key:
                errors.append("LLM API key is required for LLM_ONLY mode")
        
        # finetuned/hybrid not supported
        
        # Check data paths
        if not os.path.exists(self.data.bird_data_path):
            errors.append(f"BIRD data path does not exist: {self.data.bird_data_path}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True
