"""
AI Engine Service - Multi-LLM Provider Integration
Handles communication with OpenAI, Gemini, and other LLM providers
with intelligent fallback, caching, and response processing.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import asyncio
import hashlib

# External imports (to be installed via requirements.txt)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"


class MessageRole(Enum):
    """Message roles for conversation history."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class LLMMessage:
    """Structured message for LLM conversations."""
    role: MessageRole
    content: str
    timestamp: float = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LLMResponse:
    """Structured response from LLM providers."""
    content: str
    provider: LLMProvider
    model: str
    tokens_used: int
    response_time: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    async def generate_response(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Generate response from the LLM provider."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and configured."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if OPENAI_AVAILABLE and self.is_available():
            openai.api_key = self.config.get('api_key')
            if self.config.get('base_url'):
                openai.api_base = self.config.get('base_url')
    
    def is_available(self) -> bool:
        """Check if OpenAI is available and configured."""
        if not OPENAI_AVAILABLE:
            self.logger.warning("OpenAI library not installed")
            return False
        
        api_key = self.config.get('api_key')
        if not api_key or api_key == 'your-openai-api-key-here':
            self.logger.warning("OpenAI API key not configured")
            return False
        
        return True
    
    async def generate_response(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Generate response using OpenAI API."""
        start_time = time.time()
        
        try:
            # Convert messages to OpenAI format
            openai_messages = []
            for msg in messages:
                openai_messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
            
            # Prepare request parameters
            request_params = {
                "model": self.config.get('model', 'gpt-4'),
                "messages": openai_messages,
                "max_tokens": self.config.get('max_tokens', 2000),
                "temperature": self.config.get('temperature', 0.7),
                "timeout": self.config.get('timeout', 30)
            }
            
            # Override with any additional kwargs
            request_params.update(kwargs)
            
            # Make the API call
            response = await openai.ChatCompletion.acreate(**request_params)
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.choices[0].message.content,
                provider=LLMProvider.OPENAI,
                model=request_params['model'],
                tokens_used=response.usage.total_tokens,
                response_time=response_time,
                success=True,
                metadata={
                    'finish_reason': response.choices[0].finish_reason,
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"OpenAI API error: {e}")
            
            return LLMResponse(
                content="",
                provider=LLMProvider.OPENAI,
                model=self.config.get('model', 'gpt-4'),
                tokens_used=0,
                response_time=response_time,
                success=False,
                error=str(e)
            )


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if GEMINI_AVAILABLE and self.is_available():
            genai.configure(api_key=self.config.get('api_key'))
            self.model = genai.GenerativeModel(self.config.get('model', 'gemini-pro'))
    
    def is_available(self) -> bool:
        """Check if Gemini is available and configured."""
        if not GEMINI_AVAILABLE:
            self.logger.warning("Google Gemini library not installed")
            return False
        
        api_key = self.config.get('api_key')
        if not api_key or api_key == 'your-gemini-api-key-here':
            self.logger.warning("Gemini API key not configured")
            return False
        
        return True
    
    async def generate_response(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        """Generate response using Gemini API."""
        start_time = time.time()
        
        try:
            # Convert messages to Gemini format
            # Gemini uses a different conversation format
            conversation_text = ""
            for msg in messages:
                if msg.role == MessageRole.SYSTEM:
                    conversation_text += f"System: {msg.content}\n\n"
                elif msg.role == MessageRole.USER:
                    conversation_text += f"User: {msg.content}\n\n"
                elif msg.role == MessageRole.ASSISTANT:
                    conversation_text += f"Assistant: {msg.content}\n\n"
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=self.config.get('temperature', 0.7),
                max_output_tokens=self.config.get('max_tokens', 2000)
            )
            
            # Generate response
            response = await self.model.generate_content_async(
                conversation_text,
                generation_config=generation_config
            )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.text,
                provider=LLMProvider.GEMINI,
                model=self.config.get('model', 'gemini-pro'),
                tokens_used=response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else 0,
                response_time=response_time,
                success=True,
                metadata={
                    'safety_ratings': response.safety_ratings if hasattr(response, 'safety_ratings') else [],
                    'finish_reason': response.finish_reason if hasattr(response, 'finish_reason') else None
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"Gemini API error: {e}")
            
            return LLMResponse(
                content="",
                provider=LLMProvider.GEMINI,
                model=self.config.get('model', 'gemini-pro'),
                tokens_used=0,
                response_time=response_time,
                success=False,
                error=str(e)
            )


class AIEngine:
    """Main AI Engine service for managing multiple LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize providers
        self.providers = {}
        self._initialize_providers()
        
        # Response caching
        self.cache = {}
        self.cache_enabled = True
        self.cache_ttl = 3600  # 1 hour
        
        # Usage tracking
        self.usage_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'provider_usage': {}
        }
    
    def _initialize_providers(self):
        """Initialize all available LLM providers."""
        llm_config = self.config.get('llm', {})
        
        # Initialize OpenAI
        if 'openai' in llm_config:
            try:
                provider = OpenAIProvider(llm_config['openai'])
                if provider.is_available():
                    self.providers[LLMProvider.OPENAI] = provider
                    self.logger.info("OpenAI provider initialized")
                else:
                    self.logger.warning("OpenAI provider not available")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI provider: {e}")
        
        # Initialize Gemini
        if 'gemini' in llm_config:
            try:
                provider = GeminiProvider(llm_config['gemini'])
                if provider.is_available():
                    self.providers[LLMProvider.GEMINI] = provider
                    self.logger.info("Gemini provider initialized")
                else:
                    self.logger.warning("Gemini provider not available")
            except Exception as e:
                self.logger.error(f"Failed to initialize Gemini provider: {e}")
        
        if not self.providers:
            self.logger.warning("No LLM providers available!")
    
    def _get_cache_key(self, messages: List[LLMMessage], provider: LLMProvider, **kwargs) -> str:
        """Generate cache key for request."""
        content = json.dumps([asdict(msg) for msg in messages], sort_keys=True)
        params = json.dumps(kwargs, sort_keys=True)
        cache_string = f"{provider.value}:{content}:{params}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cached response is still valid."""
        return time.time() - timestamp < self.cache_ttl
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of available providers."""
        return list(self.providers.keys())
    
    def get_default_provider(self) -> Optional[LLMProvider]:
        """Get the default provider based on config."""
        default_name = self.config.get('llm', {}).get('default_provider', 'openai')
        
        try:
            default_provider = LLMProvider(default_name)
            if default_provider in self.providers:
                return default_provider
        except ValueError:
            pass
        
        # Fallback to first available provider
        if self.providers:
            return next(iter(self.providers.keys()))
        
        return None
    
    async def generate_response(
        self,
        messages: Union[str, List[LLMMessage]],
        provider: Optional[LLMProvider] = None,
        use_cache: bool = True,
        fallback_on_error: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response from LLM with intelligent provider selection and fallback.
        
        Args:
            messages: Either a string prompt or list of LLMMessage objects
            provider: Specific provider to use (optional)
            use_cache: Whether to use response caching
            fallback_on_error: Whether to try other providers on failure
            **kwargs: Additional parameters for the LLM
        """
        
        # Convert string to message list if needed
        if isinstance(messages, str):
            messages = [LLMMessage(role=MessageRole.USER, content=messages)]
        
        # Determine provider to use
        if provider is None:
            provider = self.get_default_provider()
        
        if provider not in self.providers:
            return LLMResponse(
                content="",
                provider=provider or LLMProvider.OPENAI,
                model="unknown",
                tokens_used=0,
                response_time=0,
                success=False,
                error="Provider not available"
            )
        
        # Check cache
        if use_cache and self.cache_enabled:
            cache_key = self._get_cache_key(messages, provider, **kwargs)
            if cache_key in self.cache:
                cached_response, timestamp = self.cache[cache_key]
                if self._is_cache_valid(timestamp):
                    self.logger.debug(f"Cache hit for {provider.value}")
                    cached_response.metadata['cached'] = True
                    return cached_response
        
        # Try primary provider
        self.usage_stats['total_requests'] += 1
        
        try:
            response = await self.providers[provider].generate_response(messages, **kwargs)
            
            if response.success:
                self.usage_stats['successful_requests'] += 1
                self.usage_stats['total_tokens'] += response.tokens_used
                
                # Update provider usage stats
                if provider.value not in self.usage_stats['provider_usage']:
                    self.usage_stats['provider_usage'][provider.value] = {
                        'requests': 0, 'tokens': 0, 'errors': 0
                    }
                self.usage_stats['provider_usage'][provider.value]['requests'] += 1
                self.usage_stats['provider_usage'][provider.value]['tokens'] += response.tokens_used
                
                # Cache successful response
                if use_cache and self.cache_enabled:
                    self.cache[cache_key] = (response, time.time())
                
                return response
            else:
                self.usage_stats['failed_requests'] += 1
                if provider.value in self.usage_stats['provider_usage']:
                    self.usage_stats['provider_usage'][provider.value]['errors'] += 1
                
                # Try fallback if enabled
                if fallback_on_error and len(self.providers) > 1:
                    self.logger.info(f"Primary provider {provider.value} failed, trying fallback")
                    return await self._try_fallback_providers(messages, provider, **kwargs)
                
                return response
                
        except Exception as e:
            self.usage_stats['failed_requests'] += 1
            self.logger.error(f"Error with provider {provider.value}: {e}")
            
            if fallback_on_error and len(self.providers) > 1:
                return await self._try_fallback_providers(messages, provider, **kwargs)
            
            return LLMResponse(
                content="",
                provider=provider,
                model="unknown",
                tokens_used=0,
                response_time=0,
                success=False,
                error=str(e)
            )
    
    async def _try_fallback_providers(
        self,
        messages: List[LLMMessage],
        failed_provider: LLMProvider,
        **kwargs
    ) -> LLMResponse:
        """Try other providers when primary fails."""
        
        for provider in self.providers:
            if provider != failed_provider:
                try:
                    self.logger.info(f"Trying fallback provider: {provider.value}")
                    response = await self.providers[provider].generate_response(messages, **kwargs)
                    
                    if response.success:
                        response.metadata['fallback_used'] = True
                        response.metadata['failed_provider'] = failed_provider.value
                        return response
                        
                except Exception as e:
                    self.logger.error(f"Fallback provider {provider.value} also failed: {e}")
                    continue
        
        # All providers failed
        return LLMResponse(
            content="",
            provider=failed_provider,
            model="unknown",
            tokens_used=0,
            response_time=0,
            success=False,
            error="All providers failed"
        )
    
    async def analyze_task_intent(self, user_input: str) -> Dict[str, Any]:
        """Analyze user input to determine task intent and parameters."""
        
        system_prompt = """You are an AI assistant that analyzes user requests for desktop automation tasks.
        
        Analyze the user's request and return a JSON response with the following structure:
        {
            "intent": "task_category",
            "confidence": 0.95,
            "task_type": "specific_task",
            "parameters": {
                "application": "app_name",
                "action": "action_type",
                "target": "target_element",
                "data": "additional_data"
            },
            "complexity": "low|medium|high",
            "estimated_steps": 5,
            "requires_confirmation": true|false
        }
        
        Common task categories:
        - web_automation: Browser-based tasks
        - file_management: File operations
        - application_control: Desktop app control
        - system_control: System-level operations
        - communication: Messaging/email tasks
        - data_processing: Data manipulation tasks
        """
        
        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content=system_prompt),
            LLMMessage(role=MessageRole.USER, content=f"Analyze this request: {user_input}")
        ]
        
        response = await self.generate_response(messages, temperature=0.3)
        
        if response.success:
            try:
                # Try to parse JSON response
                intent_data = json.loads(response.content)
                return intent_data
            except json.JSONDecodeError:
                # Fallback to basic parsing
                return {
                    "intent": "unknown",
                    "confidence": 0.5,
                    "task_type": "manual_analysis_required",
                    "parameters": {"raw_input": user_input},
                    "complexity": "medium",
                    "estimated_steps": 3,
                    "requires_confirmation": True,
                    "raw_response": response.content
                }
        else:
            return {
                "intent": "error",
                "confidence": 0.0,
                "task_type": "llm_error",
                "parameters": {"error": response.error},
                "complexity": "high",
                "estimated_steps": 0,
                "requires_confirmation": True
            }
    
    async def generate_task_steps(self, intent_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed execution steps for a task."""
        
        system_prompt = """You are an AI assistant that creates detailed step-by-step instructions for desktop automation tasks.
        
        Given a task intent analysis, create a list of specific, actionable steps that can be executed by automation scripts.
        
        Return a JSON array of steps, where each step has this structure:
        {
            "step_number": 1,
            "action": "click|type|wait|screenshot|open_app|close_app|navigate|scroll",
            "target": "description_of_target_element",
            "data": "text_to_type_or_additional_data",
            "wait_time": 2.0,
            "screenshot_before": true,
            "screenshot_after": false,
            "verification": "how_to_verify_step_completed"
        }
        """
        
        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content=system_prompt),
            LLMMessage(role=MessageRole.USER, content=f"Generate steps for this task: {json.dumps(intent_data, indent=2)}")
        ]
        
        response = await self.generate_response(messages, temperature=0.2)
        
        if response.success:
            try:
                steps = json.loads(response.content)
                if isinstance(steps, list):
                    return steps
            except json.JSONDecodeError:
                pass
        
        # Fallback to basic steps
        return [{
            "step_number": 1,
            "action": "manual_execution",
            "target": "user_intervention_required",
            "data": f"Unable to generate automated steps. Task: {intent_data.get('task_type', 'unknown')}",
            "wait_time": 0,
            "screenshot_before": True,
            "screenshot_after": False,
            "verification": "manual_verification"
        }]
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all providers."""
        return self.usage_stats.copy()
    
    def clear_cache(self):
        """Clear the response cache."""
        self.cache.clear()
        self.logger.info("Response cache cleared")
    
    def set_cache_ttl(self, ttl_seconds: int):
        """Set cache time-to-live."""
        self.cache_ttl = ttl_seconds
        self.logger.info(f"Cache TTL set to {ttl_seconds} seconds")