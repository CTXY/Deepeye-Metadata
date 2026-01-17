# CAF LLM Providers - Specific implementations for different providers

import logging
from typing import Dict, Any, List, Optional

from .client import BaseLLMClient, LLMConfig
from ..config.global_config import get_global_config

logger = logging.getLogger(__name__)

class OpenAIProvider(BaseLLMClient):
    """OpenAI LLM provider implementation"""
    
    def _initialize_client(self):
        """Initialize OpenAI client using global configuration"""
        try:
            # Get configuration from global config manager with fallback to local config
            global_config_manager = get_global_config()
            
            try:
                # Try to get from global configuration first
                client, is_modern = global_config_manager.get_openai_client()
                self._client = client
                self._use_modern_client = is_modern
                logger.debug(f"Initialized OpenAI client from global config for model: {self.model_name}")
                return
                
            except (ValueError, Exception) as global_error:
                # Fallback to local configuration if global config fails
                logger.warning(f"Global config failed: {global_error}, falling back to local config")
                
                # Try modern OpenAI client first with local config
                try:
                    from openai import OpenAI
                    kwargs = {}
                    if self.config.api_key:
                        kwargs['api_key'] = self.config.api_key
                    if self.config.base_url:
                        kwargs['base_url'] = self.config.base_url
                    if self.config.timeout:
                        kwargs['timeout'] = self.config.timeout
                    
                    self._client = OpenAI(**kwargs)
                    self._use_modern_client = True
                    logger.debug(f"Initialized modern OpenAI client with local config for model: {self.model_name}")
                    
                except ImportError:
                    # Fallback to legacy OpenAI client with local config
                    import openai
                    if self.config.api_key:
                        openai.api_key = self.config.api_key
                    if self.config.base_url:
                        openai.api_base = self.config.base_url
                        
                    self._client = openai
                    self._use_modern_client = False
                    logger.debug(f"Initialized legacy OpenAI client with local config for model: {self.model_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def call(self, prompt: str, **kwargs) -> str:
        """Make a call to OpenAI"""
        try:
            # Merge config with kwargs
            call_kwargs = {
                'temperature': kwargs.get('temperature', self.config.temperature),
                'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
            }
            
            # Support model override via kwargs
            model = kwargs.get('model', self.model_name)
            
            if self._use_modern_client:
                messages = [{"role": "user", "content": prompt}]
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **call_kwargs
                )
                return response.choices[0].message.content
            else:
                # Legacy client
                response = self._client.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    **call_kwargs
                )
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def call_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Make a call with structured messages"""
        try:
            call_kwargs = {
                'temperature': kwargs.get('temperature', self.config.temperature),
                'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
            }
            
            # Support model override via kwargs
            model = kwargs.get('model', self.model_name)
            
            if self._use_modern_client:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **call_kwargs
                )
                return response.choices[0].message.content
            else:
                response = self._client.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    **call_kwargs
                )
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"OpenAI API call with messages failed: {e}")
            raise
    
    def call_json(self, prompt: str, **kwargs) -> str:
        """Call with JSON response format"""
        try:
            call_kwargs = {
                'temperature': kwargs.get('temperature', self.config.temperature),
                'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
            }
            
            # Support model override via kwargs
            model = kwargs.get('model', self.model_name)
            
            if self._use_modern_client:
                # Modern client supports response_format
                call_kwargs['response_format'] = {"type": "json_object"}
                
            messages = [{"role": "user", "content": prompt}]
            
            if self._use_modern_client:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **call_kwargs
                )
                return response.choices[0].message.content
            else:
                # Legacy client doesn't support response_format, just call normally
                response = self._client.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    **call_kwargs
                )
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"OpenAI JSON API call failed: {e}")
            raise


class ClaudeProvider(BaseLLMClient):
    """Anthropic Claude LLM provider implementation"""
    
    def _initialize_client(self):
        """Initialize Anthropic client"""
        try:
            from anthropic import Anthropic
            
            kwargs = {}
            if self.config.api_key:
                kwargs['api_key'] = self.config.api_key
            if self.config.timeout:
                kwargs['timeout'] = self.config.timeout
                
            self._client = Anthropic(**kwargs)
            logger.debug(f"Initialized Anthropic client for model: {self.model_name}")
            
        except ImportError:
            raise ValueError("Anthropic package not installed. Install with: pip install anthropic")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            raise
    
    def call(self, prompt: str, **kwargs) -> str:
        """Make a call to Claude"""
        try:
            call_kwargs = {
                'temperature': kwargs.get('temperature', self.config.temperature),
                'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
            }
            
            # Support model override via kwargs
            model = kwargs.get('model', self.model_name)
            
            response = self._client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **call_kwargs
            )
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            raise
    
    def call_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Make a call with structured messages"""
        try:
            call_kwargs = {
                'temperature': kwargs.get('temperature', self.config.temperature),
                'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
            }
            
            # Support model override via kwargs
            model = kwargs.get('model', self.model_name)
            
            response = self._client.messages.create(
                model=model,
                messages=messages,
                **call_kwargs
            )
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Claude API call with messages failed: {e}")
            raise

