"""
Embedding provider abstraction layer.

This module provides an abstraction for different embedding providers,
allowing easy switching between OpenAI, HuggingFace, and other providers.
"""

from .base import EmbeddingProvider
from .factory import EmbeddingProviderFactory

__all__ = [
    "EmbeddingProvider",
    "EmbeddingProviderFactory",
]
