"""
OpenAI Service - Integration with OpenAI's GPT models.
Handles authentication, model selection, and response processing.
"""

import openai
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from llms.types import LLMResponse


class OpenAIService:
    """
    OpenAI integration service for GPT models.
    Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Set up OpenAI client
        openai.api_key = config.get('api_key')
        if not openai.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Configuration
        self.model = config.get('model', 'gpt-4')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2000)
        self.timeout = config.get('timeout', 30)
        
        # Rate limiting
        self.requests_per_minute = config.get('requests_per_minute', 60)
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_window = 60  # seconds
        
        # Cost tracking
        self.total_tokens_used = 0
        self.estimated_cost = 0.0
        self.token_costs = {
            'gpt-4': {'input': 0.03, 'output': 0.06},  # per 1K tokens
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002}
        }
        
        self.logger.info(f"OpenAI service initialized with model: {self.model}")
    
    def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Reset counter if window has passed
        if current_time - self.last_request_time > self.rate_limit_window:
            self.request_count = 0
            self.last_request_time = current_time
        
        # Check if we're within limits
        if self.request_count >= self.requests_per_minute:
            wait_time = self.rate_limit_window - (current_time - self.last_request_time)
            if wait_time > 0:
                self.logger.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                self.request_count = 0
        
        self.request_count += 1
    
    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate estimated cost for the request."""
        if self.model not in self.token_costs:
            return 0.0
        
        costs = self.token_costs[self.model]
        input_cost = (prompt_tokens / 1000) * costs['input']
        output_cost = (completion_tokens / 1000) * costs['output']
        
        return input_cost + output_cost
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """
        Generate response using OpenAI's API.
        
        Args:
            messages: List of message objects with role and content
            **kwargs: Additional parameters for the API call
            
        Returns:
            LLMResponse object with standardized format
        """
        self._check_rate_limit()
        
        try:
            # Prepare request parameters
            request_params = {
                'model': kwargs.get('model', self.model),
                'messages': messages,
                'temperature': kwargs.get('temperature', self.temperature),
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'timeout': self.timeout
            }
            
            # Make API call
            self.logger.debug(f"Making OpenAI API call with model: {request_params['model']}")
            response = openai.ChatCompletion.create(**request_params)
            
            # Extract response data
            content = response.choices[0].message.content
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            # Update tracking
            self.total_tokens_used += total_tokens
            cost = self._calculate_cost(prompt_tokens, completion_tokens)
            self.estimated_cost += cost
            
            # Create standardized response
            llm_response = LLMResponse(
                content=content,
                provider="openai",
                model=request_params['model'],
                tokens_used=total_tokens,
                cost=cost,
                metadata={
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'finish_reason': response.choices[0].finish_reason,
                    'response_id': response.id
                }
            )
            
            self.logger.info(f"OpenAI response generated. Tokens: {total_tokens}, Cost: ${cost:.4f}")
            return llm_response
            
        except openai.error.RateLimitError as e:
            self.logger.error(f"OpenAI rate limit exceeded: {e}")
            # Wait and retry once
            time.sleep(60)
            return self.generate_response(messages, **kwargs)
            
        except openai.error.AuthenticationError as e:
            self.logger.error(f"OpenAI authentication error: {e}")
            raise ValueError("Invalid OpenAI API key")
            
        except openai.error.APIError as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI API error: {e}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error in OpenAI service: {e}")
            raise RuntimeError(f"OpenAI service error: {e}")
    
    def generate_embedding(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        """
        Generate embeddings for text using OpenAI's embedding models.
        
        Args:
            text: Text to embed
            model: Embedding model to use
            
        Returns:
            List of embedding values
        """
        self._check_rate_limit()
        
        try:
            response = openai.Embedding.create(
                model=model,
                input=text
            )
            
            embedding = response.data[0].embedding
            self.logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models."""
        try:
            models = openai.Model.list()
            model_ids = [model.id for model in models.data if 'gpt' in model.id]
            return sorted(model_ids)
        except Exception as e:
            self.logger.error(f"Error fetching models: {e}")
            return ['gpt-4', 'gpt-3.5-turbo']  # fallback
    
    def set_model(self, model: str):
        """Set the model to use for requests."""
        available_models = self.get_available_models()
        if model not in available_models:
            self.logger.warning(f"Model {model} may not be available")
        
        self.model = model
        self.logger.info(f"Model changed to: {model}")
    
    def get_current_model(self) -> str:
        """Get currently selected model."""
        return self.model
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status and usage statistics."""
        return {
            'service': 'openai',
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'total_tokens_used': self.total_tokens_used,
            'estimated_cost': round(self.estimated_cost, 4),
            'requests_this_window': self.request_count,
            'rate_limit': self.requests_per_minute,
            'api_key_configured': bool(openai.api_key)
        }
    
    def reset_usage_stats(self):
        """Reset usage tracking statistics."""
        self.total_tokens_used = 0
        self.estimated_cost = 0.0
        self.request_count = 0
        self.logger.info("Usage statistics reset")
    
    def validate_configuration(self) -> bool:
        """Validate that the service is properly configured."""
        try:
            # Test with a simple request
            test_messages = [{"role": "user", "content": "Hello"}]
            self.generate_response(test_messages, max_tokens=5)
            return True
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False