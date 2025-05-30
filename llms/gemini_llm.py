"""
Gemini Service - Integration with Google's Gemini AI models.
Handles authentication, model selection, and response processing.
"""

import google.generativeai as genai
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from utils.logger import get_logger

from llms.types import LLMResponse

# Initialize logger
logger = get_logger(
    name=__name__,
    level=logging.INFO,
    log_file="logs/streamlit_ui.log",
    format_string='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class GeminiService:
    """
    Google Gemini integration service.
    Supports Gemini Pro, Gemini Pro Vision, and other Gemini models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Set up Gemini client
        api_key = config.get('api_key')
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        genai.configure(api_key=api_key)
        
        # Configuration
        self.model_name = config.get('model', 'gemini-2.5-flash-preview-05-20')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_output_tokens', 5000)
        self.timeout = config.get('timeout', 30)
        
        # Initialize model
        self.model = genai.GenerativeModel(self.model_name)
        
        # Generation config
        self.generation_config = genai.types.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            top_p=config.get('top_p', 0.8),
            top_k=config.get('top_k', 40)
        )
        
        # Safety settings
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        # Rate limiting
        self.requests_per_minute = config.get('requests_per_minute', 60)
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_window = 60  # seconds
        
        # Usage tracking
        self.total_requests = 0
        self.total_tokens_used = 0
        
        logger.info(f"Gemini service initialized with model: {self.model_name}")
    
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
                logger.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                self.request_count = 0
        
        self.request_count += 1
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert OpenAI-style messages to Gemini prompt format.
        
        Args:
            messages: List of message objects with role and content
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System Instructions: {content}\n")
            elif role == 'user':
                prompt_parts.append(f"User: {content}\n")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}\n")
        
        return '\n'.join(prompt_parts)
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """
        Generate response using Gemini's API.
        
        Args:
            messages: List of message objects with role and content
            **kwargs: Additional parameters for the API call
            
        Returns:
            LLMResponse object with standardized format
        """
        self._check_rate_limit()
        
        try:
            # Convert messages to prompt
            prompt = self._convert_messages_to_prompt(messages)
            
            # Update generation config if parameters provided
            gen_config = self.generation_config
            if 'temperature' in kwargs or 'max_tokens' in kwargs:
                gen_config = genai.types.GenerationConfig(
                    temperature=kwargs.get('temperature', self.temperature),
                    max_output_tokens=kwargs.get('max_tokens', self.max_tokens),
                    top_p=self.generation_config.top_p,
                    top_k=self.generation_config.top_k
                )
            
            # Make API call
            logger.debug(f"Making Gemini API call with model: {self.model_name}")
            response = self.model.generate_content(
                prompt,
                generation_config=gen_config,
                safety_settings=self.safety_settings
            )
            
            # Check response status
            if not response.candidates:
                logger.error("No candidates returned from Gemini")
                return LLMResponse(
                    content="Error: No response generated",
                    provider="gemini",
                    model=self.model_name,
                    tokens_used=0,
                    cost=0.0,
                    metadata={'error': 'No candidates returned'}
                )
            
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason.name if candidate.finish_reason else 'UNKNOWN'
            
            # Handle different finish reasons
            if finish_reason == 'SAFETY':
                logger.warning("Response was blocked by safety filters")
                safety_ratings = [
                    {
                        'category': rating.category.name,
                        'probability': rating.probability.name
                    }
                    for rating in candidate.safety_ratings
                ]
                return LLMResponse(
                    content="Response blocked by safety filters. Please rephrase your request.",
                    provider="gemini",
                    model=self.model_name,
                    tokens_used=0,
                    cost=0.0,
                    metadata={
                        'finish_reason': finish_reason,
                        'safety_ratings': safety_ratings,
                        'error': 'Safety filter triggered'
                    }
                )
            elif finish_reason == 'STOP':
                content = response.text
            else:
                logger.warning(f"Unexpected finish reason: {finish_reason}")
                return LLMResponse(
                    content=f"Error: Unexpected finish reason ({finish_reason})",
                    provider="gemini",
                    model=self.model_name,
                    tokens_used=0,
                    cost=0.0,
                    metadata={'error': f'Unexpected finish reason: {finish_reason}'}
                )
            
            # Update tracking
            self.total_requests += 1
            
            # Estimate token usage
            estimated_tokens = len(prompt.split()) + len(content.split())
            self.total_tokens_used += estimated_tokens
            
            # Create standardized response
            return LLMResponse(
                content=content,
                provider="gemini",
                model=self.model_name,
                tokens_used=estimated_tokens,
                cost=0.0,
                metadata={
                    'finish_reason': finish_reason,
                    'safety_ratings': [
                        {
                            'category': rating.category.name,
                            'probability': rating.probability.name
                        }
                        for rating in candidate.safety_ratings
                    ],
                    'estimated_tokens': estimated_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"Error in Gemini service: {e}")
            
            # Handle common errors
            if "API_KEY_INVALID" in str(e):
                raise ValueError("Invalid Gemini API key")
            elif "QUOTA_EXCEEDED" in str(e):
                raise RuntimeError("Gemini API quota exceeded")
            elif "MODEL_NOT_FOUND" in str(e):
                raise ValueError(f"Gemini model '{self.model_name}' not found")
            else:
                # Return a structured error response instead of raising
                return LLMResponse(
                    content=f"Error: {str(e)}",
                    provider="gemini",
                    model=self.model_name,
                    tokens_used=0,
                    cost=0.0,
                    metadata={'error': str(e)}
                )
    
    def generate_with_image(self, prompt: str, image_data: bytes, mime_type: str = "image/jpeg") -> LLMResponse:
        """
        Generate response with image input (Gemini Pro Vision).
        
        Args:
            prompt: Text prompt
            image_data: Binary image data
            mime_type: Image MIME type
            
        Returns:
            LLMResponse object
        """
        if 'vision' not in self.model_name.lower():
            # Switch to vision model temporarily
            vision_model = genai.GenerativeModel('gemini-pro-vision')
        else:
            vision_model = self.model
        
        self._check_rate_limit()
        
        try:
            # Create image part
            image_part = {
                "mime_type": mime_type,
                "data": image_data
            }
            
            # Generate content with image
            response = vision_model.generate_content(
                [prompt, image_part],
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            content = response.text if response.candidates[0].finish_reason.name != 'SAFETY' else "Response blocked by safety filters"
            
            # Update tracking
            self.total_requests += 1
            estimated_tokens = len(prompt.split()) + len(content.split())
            self.total_tokens_used += estimated_tokens
            
            return LLMResponse(
                content=content,
                provider="gemini",
                model=f"{self.model_name}-vision",
                tokens_used=estimated_tokens,
                cost=0.0,
                metadata={
                    'finish_reason': response.candidates[0].finish_reason.name,
                    'has_image': True,
                    'image_mime_type': mime_type
                }
            )
            
        except Exception as e:
            logger.error(f"Error in Gemini vision service: {e}")
            raise RuntimeError(f"Gemini vision error: {e}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available Gemini models."""
        try:
            models = genai.list_models()
            model_names = [model.name.split('/')[-1] for model in models if 'generateContent' in model.supported_generation_methods]
            return sorted(model_names)
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return ['gemini-pro', 'gemini-pro-vision']  # fallback
    
    def set_model(self, model: str):
        """Set the model to use for requests."""
        try:
            self.model = genai.GenerativeModel(model)
            self.model_name = model
            logger.info(f"Model changed to: {model}")
        except Exception as e:
            logger.error(f"Error setting model {model}: {e}")
            raise ValueError(f"Invalid model: {model}")
    
    def get_current_model(self) -> str:
        """Get currently selected model."""
        return self.model_name
    
    def set_safety_settings(self, settings: List[Dict[str, str]]):
        """Update safety settings for content generation."""
        self.safety_settings = settings
        logger.info("Safety settings updated")
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status and usage statistics."""
        return {
            'service': 'gemini',
            'model': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'total_requests': self.total_requests,
            'total_tokens_used': self.total_tokens_used,
            'requests_this_window': self.request_count,
            'rate_limit': self.requests_per_minute,
            'api_key_configured': bool(self.config.get('api_key')),
            'safety_settings_count': len(self.safety_settings)
        }
    
    def reset_usage_stats(self):
        """Reset usage tracking statistics."""
        self.total_requests = 0
        self.total_tokens_used = 0
        self.request_count = 0
        logger.info("Usage statistics reset")
    
    def validate_configuration(self) -> bool:
        """Validate that the service is properly configured."""
        try:
            # Test with a simple request
            test_messages = [{"role": "user", "content": "Hello"}]
            response = self.generate_response(test_messages)
            return len(response.content) > 0
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text (approximate for Gemini).
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token on average
        return len(text) // 4
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            models = genai.list_models()
            current_model = None
            
            for model in models:
                if model.name.endswith(self.model_name):
                    current_model = model
                    break
            
            if current_model:
                return {
                    'name': current_model.name,
                    'display_name': current_model.display_name,
                    'description': current_model.description,
                    'input_token_limit': current_model.input_token_limit,
                    'output_token_limit': current_model.output_token_limit,
                    'supported_generation_methods': current_model.supported_generation_methods,
                    'temperature_range': (0.0, 1.0),
                    'top_p_range': (0.0, 1.0),
                    'top_k_range': (1, 40)
                }
            else:
                return {'name': self.model_name, 'status': 'Model info not available'}
                
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {'error': str(e)}