"""
Unified registration for model providers.

Provides a single entry point that registers a provider in both the
ProviderFactory (for creation by name) and the ProviderRegistry (for
metadata search/discovery), keeping them in sync.

Recommended usage::

    from core.ai.base.providers.registration import register_provider
    from core.ai.base.providers.registry import ProviderMetadata

    register_provider(
        model_type="vlm",
        provider_name="llava",
        provider_class=LLaVAProviderAdapter,
        metadata=ProviderMetadata(
            model_type="vlm",
            provider_name="llava",
            capabilities={"inference", "multimodal"},
            description="LLaVA vision-language model",
        ),
    )
"""

from typing import Type

from .base import BaseProvider
from .factory import ProviderFactory
from .registry import ProviderRegistry, ProviderMetadata


def register_provider(
    model_type: str,
    provider_name: str,
    provider_class: Type[BaseProvider],
    metadata: ProviderMetadata,
) -> None:
    """Register a provider for both creation and discovery.

    This is the **recommended** way to register a new model provider.
    It atomically updates both:

    * ``ProviderFactory`` – so the provider can be created by name via
      ``ProviderFactory.create(model_type, provider_name, config)``.
    * ``ProviderRegistry`` – so the provider can be discovered/searched via
      ``ProviderRegistry.search(model_type=..., capabilities=...)``.

    The registry key is derived as ``"{model_type}-{provider_name}"``.

    Args:
        model_type: Model type (e.g. ``"llm"``, ``"vlm"``, ``"speech"``).
        provider_name: Provider name (e.g. ``"whisper"``, ``"llava"``).
        provider_class: The concrete ``BaseProvider`` subclass.
        metadata: A ``ProviderMetadata`` instance describing the provider.
    """
    key = (model_type, provider_name)
    ProviderFactory._registry[key] = provider_class
    provider_id = f"{model_type}-{provider_name}"
    ProviderRegistry._providers[provider_id] = metadata


def unregister_provider(model_type: str, provider_name: str) -> None:
    """Remove a provider from both Factory and Registry.

    Args:
        model_type: Model type.
        provider_name: Provider name.
    """
    ProviderFactory.unregister(model_type, provider_name)
    provider_id = f"{model_type}-{provider_name}"
    if provider_id in ProviderRegistry._providers:
        del ProviderRegistry._providers[provider_id]
