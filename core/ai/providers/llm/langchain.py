"""
Adapter pattern for existing LangChain-based RAG providers.

Adapts RAGQueryEngine to BaseProvider interface.
"""

from typing import Optional
from ...core.providers.base import BaseProvider, ModelConfig, ModelHandle
from ...core.providers.registry import ProviderRegistry, ProviderMetadata


class LangChainProviderAdapter(BaseProvider):
    """
    Adapter for existing LangChain-based RAG providers.
    
    Adapts RAGQueryEngine to BaseProvider interface.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize adapter.
        
        Args:
            config: Model configuration
        """
        super().__init__(config)
        self._rag_engine = None
    
    def load_model(self, config: ModelConfig) -> ModelHandle:
        """
        Adapt RAGQueryEngine to BaseProvider interface.
        
        Args:
            config: Model configuration
            
        Returns:
            ModelHandle: Handle containing RAG engine
        """
        # Import here to avoid circular dependencies
        from ...llm.models.rag import RAGQueryEngine
        
        # Extract provider from config or use default
        provider = config.extra_params.get("rag_provider", "openai")
        
        # Create RAG engine (existing code)
        rag_engine = RAGQueryEngine(provider=provider)
        
        # Wrap in ModelHandle
        return ModelHandle(
            model=rag_engine,
            config=config,
            metadata={
                "type": "rag",
                "provider": provider,
                "capabilities": ["generate_response", "retrieve_context"]
            }
        )
    
    def unload_model(self, handle: ModelHandle) -> None:
        """
        Cleanup RAG engine.
        
        Args:
            handle: Model handle to unload
        """
        if hasattr(handle.model, 'cleanup'):
            handle.model.cleanup()
        self._rag_engine = None


# Register adapter
ProviderRegistry.register(
    "llm-langchain",
    ProviderMetadata(
        model_type="llm",
        provider_name="langchain",
        capabilities={"rag", "chat", "completion"},
        description="Adapter for existing LangChain-based RAG providers",
        version="1.0.0"
    )
)
