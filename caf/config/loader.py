# Configuration loader and management for CAF system

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
from .paths import PathConfig

class CAFConfig:
    """CAF Configuration Management"""
    
    def __init__(self, config_data: Dict[str, Any], config_path: Optional[str] = None):
        """Initialize with configuration data
        
        Args:
            config_data: Configuration dictionary
            config_path: Optional path to the config file (for global config initialization)
        """
        self.memory = config_data.get('memory', {})
        self.feedback = config_data.get('feedback', {})
        self.database = config_data.get('database', {})
        self._raw_data = config_data
        self._config_path = config_path  # Store config path for global config initialization
        # Alias for backward compatibility and easier access
        self.raw_config = config_data
    
    @classmethod
    def from_file(cls, config_path: str) -> 'CAFConfig':
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        
        # Merge with default configuration
        default_config = cls._get_default_config()
        merged_config = cls._deep_merge(default_config, raw_config)
        
        # Validate configuration
        cls._validate_config(merged_config)
        
        # Store config path for global config initialization
        return cls(merged_config, config_path=str(config_path))
    
    @classmethod
    def default(cls) -> 'CAFConfig':
        """Create configuration with default values"""
        return cls(cls._get_default_config())
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration values using unified PathConfig"""
        return {
            'memory': {
                'storage_backend': 'sqlite',
                'database_path': str(PathConfig.MEMORY_DB_PATH),
                'cache_size': 1000,
                
                # Semantic Memory configuration
                'semantic': {
                    'storage_path': str(PathConfig.SEMANTIC_MEMORY_PATH),
                    'embedding_model': 'all-MiniLM-L6-v2',
                    'embedding_batch_size': 100
                },
                
                # Episodic Memory configuration
                'episodic': {
                    'storage_path': str(PathConfig.EPISODIC_MEMORY_PATH),
                    'cache_size': 100
                },
                
            },
            'feedback': {
                'interface_type': 'cli',  # CLI mode by default
                'llm_provider': 'openai',  # or 'claude'
                'model_name': 'gpt-4',
                'base_url': None,  # Custom base URL for API
                'timeout_seconds': 60
            },
            'database': {
                'type': 'sqlite',
                'path': str(PathConfig.CAF_DB_PATH)
            }
        }
    
    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = CAFConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """Validate configuration structure"""
        required_sections = ['memory', 'feedback', 'database']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate feedback interface type
        interface_type = config['feedback'].get('interface_type')
        if interface_type not in ['cli', 'llm']:
            raise ValueError(f"Unsupported feedback interface type: {interface_type}")
        
        # If LLM interface, validate provider
        if interface_type == 'llm':
            provider = config['feedback'].get('llm_provider')
            if provider not in ['openai', 'claude']:
                raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key"""
        keys = key.split('.')
        value = self._raw_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self._raw_data.copy()
