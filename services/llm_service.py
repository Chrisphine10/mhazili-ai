"""
LLM Service - Main interface for AI language model interactions.
Supports OpenAI and Gemini providers with unified interface.
"""

import yaml
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from llms.gemini_llm import GeminiService
from llms.openai_llm import OpenAIService
from llms.types import LLMResponse
import os



class LLMProvider(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass
class LLMResponse:
    """Standardized response format from LLM providers."""
    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMService:
    """
    Unified interface for LLM providers (OpenAI, Gemini).
    Handles provider selection, configuration, and response formatting.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Initialize provider instances
        self.providers = {}
        self._initialize_providers()
        
        # Set active provider to Gemini by default
        self.active_provider = LLMProvider.GEMINI
        if LLMProvider.GEMINI not in self.providers:
            raise ValueError("Gemini provider is not available. Please check your configuration.")
        
        self.logger.info(f"LLM Service initialized with provider: {self.active_provider.value}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.error(f"Config file not found: {config_path}")
            return {}
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing config file: {e}")
            return {}
    
    def _initialize_providers(self):
        """Initialize available LLM providers based on configuration."""
        llm_config = self.config.get('llm', {})
        
        # Initialize Gemini first
        gemini_config = llm_config.get('gemini', {})
        if not gemini_config.get('api_key'):
            self.logger.warning("No Gemini API key found in configuration")
            gemini_config = {
                'api_key': os.getenv('GEMINI_API_KEY'),
                'model': 'gemini-2.5-flash-preview-05-20'
            }
        
        try:
            self.providers[LLMProvider.GEMINI] = GeminiService(gemini_config)
            self.logger.info("Gemini provider initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini provider: {e}")
        
        # Initialize OpenAI if configured
        openai_config = llm_config.get('openai', {})
        if openai_config.get('api_key'):
            try:
                self.providers[LLMProvider.OPENAI] = OpenAIService(openai_config)
                self.logger.info("OpenAI provider initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize OpenAI provider: {e}")
        
        if not self.providers:
            raise ValueError("No LLM providers were successfully initialized. Check your configuration.")
    
    def set_provider(self, provider: str):
        """Set the active LLM provider."""
        try:
            provider_enum = LLMProvider(provider.lower())
            if provider_enum not in self.providers:
                raise ValueError(f"Provider {provider} is not available")
            
            self.active_provider = provider_enum
            self.logger.info(f"Switched to provider: {provider}")
        except ValueError as e:
            self.logger.error(f"Invalid provider selection: {e}")
            raise
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return [provider.value for provider in self.providers.keys()]
    
    def analyze_intent(self, user_input: str, context: Optional[str] = None) -> LLMResponse:
        """
        Analyze user intent and return structured response.
        
        Args:
            user_input: The user's request
            context: Optional context from previous interactions
            
        Returns:
            LLMResponse with intent analysis
        """
        if self.active_provider not in self.providers:
            raise ValueError(f"Active provider {self.active_provider.value} is not available")
        
        system_prompt = """
        You are an AI assistant that analyzes user intents for desktop automation tasks.
        
        Classify the user's request into one of these categories:
        - web_search: User wants to search for information online
        - app_control: User wants to open, close, or interact with applications
        - file_management: User wants to create, move, delete, or organize files
        - system_control: User wants to adjust system settings or controls
        - communication: User wants to send messages, emails, or make calls
        - custom_automation: Complex multi-step tasks requiring automation
        - unclear: Intent is not clear and needs clarification
        
        Respond with JSON format:
        {
            "intent": "category_name",
            "confidence": 0.95,
            "parameters": {
                "action": "specific_action",
                "target": "target_object",
                "details": "additional_details"
            },
            "clarification_needed": false,
            "suggested_steps": ["step1", "step2"]
        }
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        if context:
            messages.insert(-1, {"role": "assistant", "content": f"Previous context: {context}"})
        
        provider = self.providers[self.active_provider]
        return provider.generate_response(messages)
    
    def generate_task_steps(self, intent_analysis: str, user_request: str) -> LLMResponse:
        """
        Generate detailed steps for task execution based on intent analysis.
        
        Args:
            intent_analysis: JSON string from intent analysis
            user_request: Original user request
            
        Returns:
            LLMResponse with detailed execution steps
        """
        system_prompt = """
        You are an AI assistant that creates detailed automation steps for desktop tasks.
        
        Based on the intent analysis and user request, create a step-by-step execution plan.
        
        Respond with JSON format:
        {
            "task_type": "automation_category",
            "steps": [
                {
                    "action": "action_type",
                    "target": "target_element",
                    "parameters": {"key": "value"},
                    "wait_condition": "condition_to_wait_for",
                    "error_handling": "fallback_action"
                }
            ],
            "success_criteria": "how_to_verify_completion",
            "estimated_duration": "time_estimate"
        }
        
        Available action types:
        - open_application: Open an app
        - click_element: Click on UI element
        - type_text: Enter text
        - key_combination: Press key combo
        - wait: Wait for specified time
        - screenshot: Take screenshot
        - detect_element: Find UI element
        - extract_text: Extract text from screen
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Intent Analysis: {intent_analysis}\n\nUser Request: {user_request}"}
        ]
        
        provider = self.providers[self.active_provider]
        return provider.generate_response(messages)
    
    def summarize_content(self, content: str, max_length: int = 200) -> LLMResponse:
        """
        Summarize content to specified length.
        
        Args:
            content: Content to summarize
            max_length: Maximum length of summary
            
        Returns:
            LLMResponse with summarized content
        """
        system_prompt = f"""
        Summarize the following content in {max_length} characters or less.
        Focus on the most important information and key points.
        Make the summary clear and concise.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        
        provider = self.providers[self.active_provider]
        return provider.generate_response(messages)
    
    def extract_information(self, content: str, extraction_type: str) -> LLMResponse:
        """
        Extract specific information from content.
        
        Args:
            content: Content to analyze
            extraction_type: Type of information to extract (emails, dates, names, etc.)
            
        Returns:
            LLMResponse with extracted information
        """
        if extraction_type == "task_steps":
            system_prompt = """
            You are a task automation planner for desktop systems. Given a task description,
            generate a list of structured steps to be executed programmatically.
            
            Each step should be a dictionary with:
            - 'action': name of the action (e.g., 'open_app', 'type', 'click', 'search_web', etc.)
            - 'params': dictionary with any required parameters (e.g., 'path', 'text', 'url')
            - Optional 'delay' between steps
            
            Return ONLY a JSON array of steps, nothing else. For example:
            [
                {
                    "action": "open_app",
                    "params": {"app_name": "chrome"},
                    "delay": 1
                },
                {
                    "action": "type",
                    "params": {"text": "search query"},
                    "delay": 0.5
                }
            ]
            """
        else:
            system_prompt = f"""
            Extract {extraction_type} from the provided content.
            
            Return the results in JSON format:
            {{
                "extracted_items": ["item1", "item2"],
                "count": 2,
                "confidence": 0.95
            }}
            
            If no items are found, return an empty list.
            """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        
        provider = self.providers[self.active_provider]
        return provider.generate_response(messages)
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status information about current provider."""
        if self.active_provider not in self.providers:
            return {"status": "error", "message": "No active provider"}
        
        provider = self.providers[self.active_provider]
        return {
            "active_provider": self.active_provider.value,
            "available_providers": self.get_available_providers(),
            "provider_status": provider.get_status(),
            "model": provider.get_current_model()
        }