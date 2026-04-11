"""
DEPRECATED: Use `core.ai.embeddings` instead.

This module is kept for backward compatibility only. All symbols are
re-exported from the new, unified embedding package
``core.ai.embeddings``.

Migration
---------
Old import::

    from infimum.ai.preprocessing.embeddings.base import EmbeddingProvider

New import::

    from infimum.ai.embeddings import EmbeddingProvider
"""

import warnings

warnings.warn(
    "core.ai.preprocessing.embeddings.base is deprecated. "
    "Import from 'core.ai.embeddings' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from infimum.ai.embeddings.base import EmbeddingProvider  # noqa: E402, F401

__all__ = ["EmbeddingProvider"]
