"""
Core abstractions for the AI library.

This module contains abstract base classes and interfaces that define
the contracts for providers, data loaders, preprocessing transforms, and observers.
"""

from .providers.base import BaseProvider, ModelHandle
from .providers.config import ModelConfig, ModelConfigBuilder
from .providers.registry import ProviderRegistry

from .data.base import BaseLoader, DataItem

from .preprocessing.base import BaseTransform
from .preprocessing.pipeline import TransformPipeline


__all__ = [
    # Providers
    "BaseProvider",
    "ModelConfig",
    "ModelHandle",
    "ProviderRegistry",
    "ModelConfigBuilder",
    # Data
    "BaseLoader",
    "DataItem",
    # Preprocessing
    "BaseTransform",
    "TransformPipeline",
]
