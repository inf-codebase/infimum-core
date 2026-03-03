"""
Unified provider registry and factory.

Single source of truth for provider implementations and their metadata.
Replaces the older Factory, Registry, and registration modules.

Example:

```
class MyLLMProvider(BaseProvider):
    def load_model(self, config: ModelConfig) -> ModelHandle:
        # Your real loading logic here
        model = {"name": "my-llm", "path": config.model_path}
        return ModelHandle(model=model, config=config)

    def unload_model(self, handle: ModelHandle) -> None:
        # Cleanup if needed
        pass

# Typically done at import time
ProviderRegistry.register(
    model_type="llm",
    provider_name="my_llm",
    provider_class=MyLLMProvider,
    metadata=ProviderMetadata(
        model_type="llm",
        provider_name="my_llm",
        capabilities={"chat", "completion"},
        description="Custom LLM provider",
        version="1.0.0",
    ),
)

config = ModelConfig(
    model_type="llm",
    provider="my_llm",
    model_path="/models/my-llm.bin",
)

provider = ProviderRegistry.create("llm", "my_llm", config)  # calls load_model under the hood
handle = provider.get_model(config)                          # use handle.model ...


# List all provider IDs (e.g. ["llm-my_llm", "vlm-mobileclip", ...])
all_ids = ProviderRegistry.list_all()

# Get metadata by ID
meta = ProviderRegistry.get("llm-my_llm")

# Search by type and capabilities
llm_chat_ids = ProviderRegistry.search(model_type="llm", capabilities=["chat"])

# Get all metadata objects for a given type
llm_metas = ProviderRegistry.get_by_type("llm")

```

"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Type

from .base import BaseProvider
from .config import ModelConfig


@dataclass
class ProviderMetadata:
    """Metadata for a provider."""

    model_type: str
    provider_name: str
    capabilities: Set[str]
    description: str = ""
    version: str = ""
    requirements: List[str] = None

    def __post_init__(self) -> None:
        if self.requirements is None:
            self.requirements = []


class ProviderRegistry:
    """Single registry/factory for model providers."""

    # (model_type, provider_name) -> (provider_class, metadata)
    _store: Dict[Tuple[str, str], Tuple[Type[BaseProvider], ProviderMetadata]] = {}

    @staticmethod
    def _provider_id(model_type: str, provider_name: str) -> str:
        return f"{model_type}-{provider_name}"

    @classmethod
    def register(
        cls,
        model_type: str,
        provider_name: str,
        provider_class: Type[BaseProvider],
        metadata: ProviderMetadata,
    ) -> None:
        """Register a provider implementation and its metadata."""
        key = (model_type, provider_name)
        cls._store[key] = (provider_class, metadata)

    @classmethod
    def unregister(cls, model_type: str, provider_name: str) -> None:
        """Unregister a provider."""
        key = (model_type, provider_name)
        if key in cls._store:
            del cls._store[key]

    @classmethod
    def create(
        cls, model_type: str, provider_name: str, config: ModelConfig
    ) -> BaseProvider:
        """Create a provider instance by type and name."""
        key = (model_type, provider_name)
        entry = cls._store.get(key)
        if entry is None:
            available = cls.list_providers(model_type)
            raise ValueError(
                f"Provider '{provider_name}' for '{model_type}' not registered. "
                f"Available providers: {available}"
            )
        provider_class, _ = entry
        return provider_class(config)

    @classmethod
    def list_providers(cls, model_type: Optional[str] = None) -> List[str]:
        """List available provider names, optionally filtered by model_type."""
        keys = cls._store.keys()
        if model_type:
            return [name for (mt, name) in keys if mt == model_type]
        return [name for (_, name) in keys]

    @classmethod
    def is_registered(cls, model_type: str, provider_name: str) -> bool:
        """Check if a provider is registered."""
        return (model_type, provider_name) in cls._store

    @classmethod
    def get(cls, provider_id: str) -> Optional[ProviderMetadata]:
        """Get provider metadata by provider ID 'model_type-provider_name'."""
        for (mt, name), (_, meta) in cls._store.items():
            if cls._provider_id(mt, name) == provider_id:
                return meta
        return None

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered provider IDs."""
        return [cls._provider_id(mt, name) for (mt, name) in cls._store.keys()]

    @classmethod
    def search(
        cls, model_type: Optional[str] = None, capabilities: Optional[List[str]] = None
    ) -> List[str]:
        """Search providers by type and required capabilities."""
        results: List[str] = []
        for (mt, name), (_, meta) in cls._store.items():
            if model_type and meta.model_type != model_type:
                continue
            if capabilities and not all(c in meta.capabilities for c in capabilities):
                continue
            results.append(cls._provider_id(mt, name))
        return results

    @classmethod
    def get_by_type(cls, model_type: str) -> List[ProviderMetadata]:
        """Get all provider metadata for a given model type."""
        metas: List[ProviderMetadata] = []
        for (_, _name), (_, meta) in cls._store.items():
            if meta.model_type == model_type:
                metas.append(meta)
        return metas

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers (used mainly in tests)."""
        cls._store.clear()
