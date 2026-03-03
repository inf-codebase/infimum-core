"""
Core abstractions for the AI library.

This module contains abstract base classes and interfaces that define
the contracts for providers, data loaders, preprocessing transforms, and observers.
"""

from .providers.base import BaseProvider, ModelHandle
from .providers.config import ModelConfig, ModelConfigBuilder
from .providers.registry import ProviderRegistry
from .providers.factory import ProviderFactory
from .providers.registration import register_provider, unregister_provider

from .data.base import BaseLoader, DataItem
from .data.registry import LoaderRegistry
from .data.factory import LoaderFactory
from .data.registration import register_loader, unregister_loader

from .preprocessing.base import BaseTransform
from .preprocessing.pipeline import TransformPipeline
from .preprocessing.registry import TransformRegistry
from .preprocessing.factory import TransformFactory
from .preprocessing.registration import register_transform, unregister_transform


__all__ = [
    # Providers
    "BaseProvider",
    "ModelConfig",
    "ModelHandle",
    "ProviderRegistry",
    "ProviderFactory",
    "ModelConfigBuilder",
    "register_provider",
    "unregister_provider",
    # Data
    "BaseLoader",
    "DataItem",
    "LoaderRegistry",
    "LoaderFactory",
    "register_loader",
    "unregister_loader",
    # Preprocessing
    "BaseTransform",
    "TransformPipeline",
    "TransformRegistry",
    "TransformFactory",
    "register_transform",
    "unregister_transform",
]
