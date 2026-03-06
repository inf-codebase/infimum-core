"""
core.ai.embeddings — unified embedding provider package.

Public surface
--------------
EmbeddingProvider          : abstract base (extends BaseProvider)
OpenAIEmbeddingProvider    : concrete OpenAI implementation
EmbeddingProviderFactory   : legacy factory (kept for backward compatibility)
"""

from .base import EmbeddingProvider
from .providers.openai import OpenAIEmbeddingProvider
# Re-export legacy factory so existing imports still work
from ..preprocessing.embeddings.factory import EmbeddingProviderFactory

__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "EmbeddingProviderFactory",
]
