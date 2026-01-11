"""
Core abstractions for the AI library.

This module contains abstract base classes and interfaces that define
the contracts for providers, data loaders, preprocessing transforms, and observers.
"""

from .providers.base import BaseProvider, ModelHandle
from .providers.config import ModelConfig, ModelConfigBuilder
from .providers.registry import ProviderRegistry
from .providers.factory import ProviderFactory

from .data.base import BaseLoader, DataItem
from .data.registry import LoaderRegistry
from .data.factory import LoaderFactory

from .preprocessing.base import BaseTransform
from .preprocessing.pipeline import TransformPipeline
from .preprocessing.registry import TransformRegistry
from .preprocessing.factory import TransformFactory

from .observers.base import Observer, Observable
from .observers.events import Event

__all__ = [
    # Providers
    "BaseProvider",
    "ModelConfig",
    "ModelHandle",
    "ProviderRegistry",
    "ProviderFactory",
    "ModelConfigBuilder",
    # Data
    "BaseLoader",
    "DataItem",
    "LoaderRegistry",
    "LoaderFactory",
    # Preprocessing
    "BaseTransform",
    "TransformPipeline",
    "TransformRegistry",
    "TransformFactory",
    # Observers
    "Observer",
    "Observable",
    "Event",
]
