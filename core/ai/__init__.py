"""
AI Library - General and reusable library for AI downstream applications.

This library provides:
- LLM (Language Models): Text generation, RAG, agents
- VLM (Vision-Language Models): Multimodal understanding
- Speech: Speech-to-text and text-to-speech

All modules use unified provider, data loading, and preprocessing systems.
"""

# Core abstractions
from .base import (
    # Providers
    BaseProvider,
    ModelConfig,
    ModelHandle,
    ProviderRegistry,
    ProviderFactory,
    ModelConfigBuilder,
    # Data
    BaseLoader,
    DataItem,
    LoaderRegistry,
    LoaderFactory,
    # Preprocessing
    BaseTransform,
    TransformPipeline,
    TransformRegistry,
    TransformFactory,
    # Observers
    Observer,
    Observable,
    Event,
)

# LLM module imports
from .llm import (
    Agent,
    ToolRegistry,
)

# Speech module
from .speech.models.speech2text import Speech2Text
from .speech.models.text2speech import Text2Speech

__all__ = [
    # Core
    "BaseProvider",
    "ModelConfig",
    "ModelHandle",
    "ProviderRegistry",
    "ProviderFactory",
    "ModelConfigBuilder",
    "BaseLoader",
    "DataItem",
    "LoaderRegistry",
    "LoaderFactory",
    "BaseTransform",
    "TransformPipeline",
    "TransformRegistry",
    "TransformFactory",
    "Observer",
    "Observable",
    "Event",
    # LLM / Agent
    "Agent",
    "ToolRegistry",
    # Speech
    "Speech2Text",
    "Text2Speech",
]
