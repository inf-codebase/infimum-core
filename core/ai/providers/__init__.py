"""Provider implementations for different model types."""

# Register providers with factory
from .llm.langchain import LangChainProviderAdapter
from .vlm.llava import LLaVAProviderAdapter
from .vlm.model_manager import ModelManagerProviderAdapter

from ...core.providers.factory import ProviderFactory

# Register LLM providers
ProviderFactory.register("llm", "langchain", LangChainProviderAdapter)

# Register VLM providers
ProviderFactory.register("vlm", "llava", LLaVAProviderAdapter)
ProviderFactory.register("vlm", "model_manager", ModelManagerProviderAdapter)

__all__ = [
    "LangChainProviderAdapter",
    "LLaVAProviderAdapter",
    "ModelManagerProviderAdapter",
]
