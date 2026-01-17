# LLM Feedback Interface - LLM-driven SQL evaluation

import json
import re
import logging
from typing import Dict, Any, Optional

from .base import BaseFeedbackInterface
from ..types import FeedbackContext, UserFeedback
from ..exceptions import LLMProviderError, FeedbackTimeoutError
from ...config.global_config import get_global_config
from ...prompts.sql_evaluation import get_sql_evaluator_prompt

logger = logging.getLogger(__name__)

class LLMFeedbackInterface(BaseFeedbackInterface):
    """
    LLM Feedback Interface - minimal implementation
    
    Uses LLM to automatically evaluate SQL correctness and provide structured feedback.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm_client = self._initialize_llm_client()
        self.sql_evaluator_prompt = get_sql_evaluator_prompt()
        
        logger.info(f"LLMFeedbackInterface initialized with provider: {config.get('llm_provider', 'openai')}")
    
    def _initialize_llm_client(self):
        """Initialize LLM client using global configuration with fallback to local config"""
        try:
            # Try global configuration first
            global_config_manager = get_global_config()
            llm_provider = self.config.get('llm_provider', 'openai')
            
            if llm_provider == 'openai':
                try:
                    # Try to get from global configuration
                    client, is_modern = global_config_manager.get_openai_client()
                    logger.debug("Using OpenAI client from global configuration")
                    return client
                except (ValueError, Exception) as global_error:
                    logger.warning(f"Global config failed: {global_error}, falling back to local config")
                    
                    # Fallback to local configuration
                    import openai
                    api_key = self.config.get('openai_api_key')
                    base_url = self.config.get('base_url')
                    
                    # Handle different versions of openai package
                    if hasattr(openai, 'OpenAI'):
                        # v1.0+ API
                        kwargs = {}
                        if api_key:
                            kwargs['api_key'] = api_key
                        if base_url:
                            kwargs['base_url'] = base_url
                        
                        if kwargs:
                            return openai.OpenAI(**kwargs)
                        else:
                            return openai.OpenAI()
                    else:
                        # Legacy API
                        if api_key:
                            openai.api_key = api_key
                        if base_url:
                            openai.api_base = base_url
                        return openai
                        
            elif llm_provider == 'claude':
                # Claude doesn't use global config yet, use local config
                import anthropic
                api_key = self.config.get('claude_api_key')
                base_url = self.config.get('base_url')
                
                kwargs = {}
                if api_key:
                    kwargs['api_key'] = api_key
                if base_url:
                    kwargs['base_url'] = base_url
                
                if kwargs:
                    return anthropic.Anthropic(**kwargs)
                else:
                    return anthropic.Anthropic()  # Use environment variable
            else:
                raise ValueError(f"Unsupported LLM provider: {llm_provider}")
                
        except ImportError as e:
            raise LLMProviderError(llm_provider, f"Required package not installed: {e}")
        except Exception as e:
            raise LLMProviderError(llm_provider, f"Failed to initialize LLM client: {e}")
    
    
    def collect_feedback(self, context: FeedbackContext) -> UserFeedback:
        """Use LLM to evaluate SQL and provide structured feedback"""
        try:
            # Prepare LLM input
            prompt = self.sql_evaluator_prompt.format(
                question=context.user_query,
                db_schema=context.db_schema or "Schema not provided",
                generated_sql=context.generated_sql,
                ground_truth_sql=context.ground_truth_sql or "Ground truth not provided"
            )
            
            # Call LLM for evaluation
            llm_response = self._call_llm(prompt)

            print('--------------------------------')
            print(prompt)
            print('--------------------------------')
            print(llm_response)
            print('--------------------------------')
            evaluation_result = self._parse_llm_response(llm_response)
            
            # Convert to standard UserFeedback format
            feedback = UserFeedback(
                is_correct=evaluation_result.get('is_correct', False),
                error_category=evaluation_result.get('error_category'),
                error_subcategory=evaluation_result.get('error_subcategory'),
                analysis=evaluation_result.get('analysis'),
                suggestion=evaluation_result.get('suggestion')
            )
            
            logger.info(f"LLM evaluation completed: correct={feedback.is_correct}")
            return feedback
            
        except Exception as e:
            logger.error(f"LLM feedback collection failed: {e}")
            # Return a safe fallback feedback
            return UserFeedback(
                is_correct=False,
                error_category="Other Errors",
                error_subcategory="Other", 
                analysis="LLM evaluation failed - manual review required",
                suggestion="Please manually verify the SQL correctness"
            )
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM to generate evaluation result"""
        llm_provider = self.config.get('llm_provider', 'openai')
        model_name = self.config.get('model_name', 'gpt-4')
        
        try:
            if llm_provider == 'openai':
                # Handle different versions of openai package
                if hasattr(self.llm_client, 'chat'):
                    # v1.0+ API
                    response = self.llm_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=1000
                    )
                    return response.choices[0].message.content
                else:
                    # Legacy API
                    response = self.llm_client.ChatCompletion.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=1000
                    )
                    return response.choices[0].message.content
            
            elif llm_provider == 'claude':
                response = self.llm_client.messages.create(
                    model=model_name,
                    max_tokens=1000,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
        except Exception as e:
            raise LLMProviderError(llm_provider, f"LLM API call failed: {e}")
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response"""
        try:
            # Extract JSON part (prevent LLM from adding extra text before/after JSON)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                return result
            else:
                raise ValueError("LLM response does not contain valid JSON")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw LLM response: {response}")
            
            # Return safe fallback
            return {
                "is_correct": False,
                "error_category": "Other Errors",
                "error_subcategory": "Other",
                "analysis": "Failed to parse LLM evaluation response",
                "suggestion": "Manual review required"
            }
    
    def cleanup(self) -> None:
        """Cleanup LLM interface resources"""
        super().cleanup()
        # Close any persistent connections if needed
        logger.debug("LLMFeedbackInterface cleanup completed")
