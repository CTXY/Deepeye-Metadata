# CAF LLM Client - Unified LLM client for CAF system

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
from enum import Enum

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    CLAUDE = "claude"

@dataclass
class LLMConfig:
    """Configuration for LLM client"""
    provider: Union[str, LLMProvider]
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 60
    
    def __post_init__(self):
        if isinstance(self.provider, str):
            self.provider = LLMProvider(self.provider)

class BaseLLMClient(ABC):
    """Base class for LLM clients"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider = config.provider
        self.model_name = config.model_name
        self._client = None
        self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the LLM provider client"""
        pass
    
    @abstractmethod
    def call(self, prompt: str, **kwargs) -> str:
        """Make a call to the LLM"""
        pass
    
    @abstractmethod
    def call_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Make a call with structured messages"""
        pass
    
    def call_json(self, prompt: str, **kwargs) -> str:
        """Call LLM with JSON response format if supported"""
        return self.call(prompt, **kwargs)
    
    def validate_connection(self) -> Dict[str, Any]:
        """Test the connection to the LLM provider"""
        try:
            test_response = self.call("Respond with: Connection successful")
            return {
                'valid': True,
                'provider': self.provider.value,
                'model': self.model_name,
                'response': test_response.strip()[:100]
            }
        except Exception as e:
            return {
                'valid': False,
                'provider': self.provider.value,
                'model': self.model_name,
                'error': str(e)
            }

def create_llm_client(config: LLMConfig) -> BaseLLMClient:
    """Factory function to create appropriate LLM client"""
    if config.provider == LLMProvider.OPENAI:
        from .providers import OpenAIProvider
        return OpenAIProvider(config)
    elif config.provider == LLMProvider.CLAUDE:
        from .providers import ClaudeProvider
        return ClaudeProvider(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")

