"""
Shared types for LLM services
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class LLMResponse:
    """Standardized response format from LLM providers."""
    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None 