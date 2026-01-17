# Global Configuration Manager for CAF System
# Provides unified configuration management for OpenAI and other LLM providers

import os
import logging
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class GlobalLLMConfig:
    """Global LLM Configuration"""
    provider: str = "openai"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 60
    
    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        if self.provider == "openai":
            return self.api_key is not None
        return True  # Other providers might not need API key
    
    def get_client_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for client initialization"""
        kwargs = {}
        if self.api_key:
            kwargs['api_key'] = self.api_key
        if self.base_url:
            kwargs['base_url'] = self.base_url
        if self.timeout:
            kwargs['timeout'] = self.timeout
        return kwargs

class GlobalConfigManager:
    """
    Global Configuration Manager - Singleton
    
    Manages unified configuration for the entire CAF system, with priority:
    1. Environment variables (highest priority)
    2. Configuration files
    3. Default values (lowest priority)
    """
    
    _instance: Optional['GlobalConfigManager'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'GlobalConfigManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._llm_config: Optional[GlobalLLMConfig] = None
            self._config_sources: Dict[str, Any] = {}
            self._initialized = True
            logger.debug("GlobalConfigManager initialized")
    
    def load_from_env(self) -> None:
        """Load configuration from environment variables"""
        env_config = {}
        
        # OpenAI configuration from environment
        if os.getenv('OPENAI_API_KEY'):
            env_config['openai_api_key'] = os.getenv('OPENAI_API_KEY')
        if os.getenv('OPENAI_BASE_URL'):
            env_config['openai_base_url'] = os.getenv('OPENAI_BASE_URL')
        if os.getenv('OPENAI_MODEL'):
            env_config['openai_model'] = os.getenv('OPENAI_MODEL')
        
        if env_config:
            self._config_sources['environment'] = env_config
            logger.info(f"Loaded configuration from environment variables: {list(env_config.keys())}")
    
    def load_from_file(self, config_path: Union[str, Path]) -> None:
        """Load configuration from YAML file"""
        try:
            import yaml
            config_path = Path(config_path)
            
            if not config_path.exists():
                logger.warning(f"Configuration file not found: {config_path}")
                return
            
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
            
            self._config_sources['file'] = file_config
            logger.info(f"Loaded configuration from file: {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    def get_llm_config(self, provider: str = "openai") -> GlobalLLMConfig:
        """
        Get LLM configuration with priority resolution
        
        Priority: Environment Variables > Config File > Default Values
        """
        if self._llm_config is not None:
            return self._llm_config
        
        # Start with default configuration
        config = GlobalLLMConfig(provider=provider)
        
        # Apply file configuration
        file_config = self._config_sources.get('file', {})
        if file_config:
            config = self._apply_file_config(config, file_config)
        
        # Apply environment configuration (highest priority)
        env_config = self._config_sources.get('environment', {})
        if env_config:
            config = self._apply_env_config(config, env_config)
        
        # Validate configuration
        if not config.is_valid():
            error_msg = f"Invalid LLM configuration for provider '{provider}': missing API key"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        self._llm_config = config
        logger.info(f"LLM configuration resolved: provider={config.provider}, model={config.model_name}, has_api_key={config.api_key is not None}, base_url={config.base_url}")
        
        return config
    
    def _apply_file_config(self, config: GlobalLLMConfig, file_config: Dict[str, Any]) -> GlobalLLMConfig:
        """Apply configuration from file"""
        # Try different configuration sections in order of preference
        sources_to_try = [
            file_config.get('llm', {}),  # Top-level LLM section
            file_config.get('feedback', {}),  # Feedback section
            file_config.get('memory', {}).get('semantic', {}).get('search', {}).get('llm_refinement', {})  # Semantic search section
        ]
        
        for source in sources_to_try:
            if not source:
                continue
                
            # Update config from this source
            if 'provider' in source:
                config.provider = source['provider']
            if 'api_key' in source:
                config.api_key = source['api_key']
            if 'openai_api_key' in source:
                config.api_key = source['openai_api_key']
            if 'base_url' in source:
                config.base_url = source['base_url']
            if 'model_name' in source:
                config.model_name = source['model_name']
            if 'temperature' in source:
                config.temperature = source['temperature']
            if 'max_tokens' in source:
                config.max_tokens = source['max_tokens']
            if 'timeout' in source or 'timeout_seconds' in source:
                config.timeout = source.get('timeout', source.get('timeout_seconds', config.timeout))
        
        return config
    
    def _apply_env_config(self, config: GlobalLLMConfig, env_config: Dict[str, Any]) -> GlobalLLMConfig:
        """Apply environment configuration (highest priority)"""
        if 'openai_api_key' in env_config:
            config.api_key = env_config['openai_api_key']
        if 'openai_base_url' in env_config:
            config.base_url = env_config['openai_base_url']
        if 'openai_model' in env_config:
            config.model_name = env_config['openai_model']
        
        return config
    
    def get_openai_client(self):
        """Get configured OpenAI client"""
        config = self.get_llm_config("openai")
        
        try:
            # Try modern OpenAI client first
            from openai import OpenAI
            kwargs = config.get_client_kwargs()
            client = OpenAI(**kwargs)
            logger.debug("Created modern OpenAI client")
            return client, True  # Returns (client, is_modern)
            
        except ImportError:
            # Fallback to legacy OpenAI client
            import openai
            if config.api_key:
                openai.api_key = config.api_key
            if config.base_url:
                openai.api_base = config.base_url
            logger.debug("Using legacy OpenAI client")
            return openai, False  # Returns (client, is_modern)
    
    def reset(self) -> None:
        """Reset configuration (useful for testing)"""
        self._llm_config = None
        self._config_sources.clear()
        logger.debug("Configuration reset")
    
    @classmethod
    def initialize(cls, config_path: Optional[Union[str, Path]] = None) -> 'GlobalConfigManager':
        """
        Initialize global configuration manager
        
        Args:
            config_path: Optional path to configuration file
            
        Returns:
            GlobalConfigManager instance
        """
        manager = cls()
        
        # Load from environment first
        manager.load_from_env()
        
        # Load from file if provided
        if config_path:
            manager.load_from_file(config_path)
        
        return manager

# Global instance - singleton pattern
_global_config: Optional[GlobalConfigManager] = None

def get_global_config() -> GlobalConfigManager:
    """Get the global configuration manager instance"""
    global _global_config
    if _global_config is None:
        _global_config = GlobalConfigManager()
    return _global_config

def initialize_global_config(config_path: Optional[Union[str, Path]] = None) -> GlobalConfigManager:
    """Initialize global configuration with optional config file"""
    global _global_config
    _global_config = GlobalConfigManager.initialize(config_path)
    return _global_config

def get_llm_config(provider: str = "openai") -> GlobalLLMConfig:
    """Convenience function to get LLM configuration"""
    return get_global_config().get_llm_config(provider)

def get_openai_client():
    """Convenience function to get OpenAI client"""
    return get_global_config().get_openai_client()
