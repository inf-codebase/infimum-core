"""
Embedding utilities using the provider abstraction layer.

This module provides a backward-compatible interface for generating embeddings
while using the new provider system under the hood.
"""

from typing import List, Union, Optional
from core.utils import auto_config
from core.ai.embeddings.factory import EmbeddingProviderFactory
from core.ai.embeddings.openai_provider import OpenAIEmbeddingProvider
from core.exceptions import EmbeddingError, ConfigurationError
from loguru import logger

# Register default OpenAI provider
EmbeddingProviderFactory.register("openai", OpenAIEmbeddingProvider)


def text_to_embedding(
    texts: Union[str, List[str]], 
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **provider_kwargs
) -> List[List[float]]:
    """
    Transform text or a list of texts into embedding vectors using the configured provider.

    This function uses the embedding provider system, which allows switching between
    different providers (OpenAI, HuggingFace, etc.) without changing code.

    Args:
        texts: The input text or list of texts to be transformed into embeddings.
        provider: Optional provider name (defaults to 'openai' or EMBEDDING_PROVIDER from config).
        model: Optional model name (uses provider's default if not specified).
        **provider_kwargs: Additional arguments to pass to provider constructor
                          (e.g., api_key for OpenAI if not using auto_config).

    Returns:
        List[List[float]]: A list of embedding vectors, each as a list of floats.

    Raises:
        ValueError: If the provider is not configured or if the API request fails.

    Example:
        ```python
        # Using default provider (OpenAI from config)
        embeddings = text_to_embedding(["Hello world", "How are you?"])
        
        # Using specific provider
        embeddings = text_to_embedding("Hello", provider="openai")
        
        # With custom API key
        embeddings = text_to_embedding("Hello", provider="openai", api_key="sk-...")
        ```
    """
    # Determine provider name
    if provider is None:
        provider = getattr(auto_config, 'EMBEDDING_PROVIDER', 'openai')
    
    # Get provider-specific kwargs from auto_config if not provided
    if provider == "openai" and not provider_kwargs.get("api_key"):
        api_key = getattr(auto_config, 'OPENAI_API_KEY', None)
        if api_key:
            provider_kwargs["api_key"] = api_key
        else:
            raise ConfigurationError(
                "OpenAI API key is not set. Please set the OPENAI_API_KEY in the "
                "configuration or pass it as api_key parameter."
            )
    
    # Get or create provider instance (cached by factory)
    try:
        embedding_provider = EmbeddingProviderFactory.create(
            provider, 
            **provider_kwargs
        )
        
        # Generate embeddings
        embeddings = embedding_provider.embed(texts, model=model)
        
        return embeddings
    except (ValueError, ConfigurationError) as e:
        # Re-raise configuration/validation errors as-is
        raise
    except Exception as e:
        logger.error(f"Error while generating embeddings with provider {provider}: {str(e)}")
        raise EmbeddingError(f"Failed to generate embeddings: {str(e)}") from e
