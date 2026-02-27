"""
Registry pattern for provider discovery and metadata management.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass
class ProviderMetadata:
    """Metadata for a provider."""

    model_type: str
    provider_name: str
    capabilities: Set[str]
    description: str = ""
    version: str = ""
    requirements: List[str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.requirements is None:
            self.requirements = []


class ProviderRegistry:
    """Registry for managing provider metadata."""

    _providers: Dict[str, ProviderMetadata] = {}

    @classmethod
    def register(cls, provider_id: str, metadata: ProviderMetadata) -> None:
        """
        Register provider with metadata.

        .. deprecated::
            Use :func:`register_provider` from ``core.ai.base.providers.registration``
            instead, which registers in both Factory and Registry.

        Args:
            provider_id: Unique provider identifier
            metadata: Provider metadata
        """
        warnings.warn(
            "ProviderRegistry.register() is deprecated. "
            "Use register_provider() from core.ai.base.providers.registration instead, "
            "which registers in both Factory and Registry.",
            DeprecationWarning,
            stacklevel=2,
        )
        cls._providers[provider_id] = metadata

    @classmethod
    def get(cls, provider_id: str) -> Optional[ProviderMetadata]:
        """
        Get provider metadata.

        Args:
            provider_id: Provider identifier

        Returns:
            ProviderMetadata or None if not found
        """
        return cls._providers.get(provider_id)

    @classmethod
    def list_all(cls) -> List[str]:
        """
        List all registered providers.

        Returns:
            List of provider IDs
        """
        return list(cls._providers.keys())

    @classmethod
    def search(
        cls, model_type: Optional[str] = None, capabilities: Optional[List[str]] = None
    ) -> List[str]:
        """
        Search providers by type and capabilities.

        Args:
            model_type: Model type to filter by
            capabilities: Required capabilities

        Returns:
            List of matching provider IDs
        """
        results = []
        for pid, meta in cls._providers.items():
            if model_type and meta.model_type != model_type:
                continue
            if capabilities and not all(c in meta.capabilities for c in capabilities):
                continue
            results.append(pid)
        return results

    @classmethod
    def get_by_type(cls, model_type: str) -> List[ProviderMetadata]:
        """
        Get all providers for a model type.

        Args:
            model_type: Model type

        Returns:
            List of provider metadata
        """
        return [
            meta for meta in cls._providers.values() if meta.model_type == model_type
        ]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers."""
        cls._providers.clear()
