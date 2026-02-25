"""
DEPRECATED: Use `core.ai.embeddings` instead.

This module is kept for backward compatibility only.
"""

import warnings

warnings.warn(
    "core.ai.preprocessing.embeddings.openai_provider is deprecated. "
    "Import from 'core.ai.embeddings.providers.openai' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from core.ai.embeddings.providers.openai import OpenAIEmbeddingProvider  # noqa: E402, F401

__all__ = ["OpenAIEmbeddingProvider"]
