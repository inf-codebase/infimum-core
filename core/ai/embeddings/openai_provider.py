"""
OpenAI embedding provider implementation.

This module provides an implementation of EmbeddingProvider for OpenAI's embedding API.
"""

from typing import List, Union, Optional
from openai import OpenAI
from loguru import logger

from .base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider implementation.
    
    This provider uses OpenAI's embedding API to generate embeddings.
    Supports models like text-embedding-3-small, text-embedding-3-large, etc.
    
    Example:
        ```python
        provider = OpenAIEmbeddingProvider(api_key="sk-...")
        embeddings = provider.embed(["Hello world", "How are you?"])
        dimension = provider.get_embedding_dimension()
        ```
    """
    
    # Model dimension mapping
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        "text-embedding-002": 1536,
    }
    
    def __init__(self, api_key: str, default_model: str = "text-embedding-3-small"):
        """Initialize OpenAI embedding provider.
        
        Args:
            api_key: OpenAI API key
            default_model: Default model to use for embeddings
        """
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=api_key)
        self.default_model = default_model
        logger.debug(f"Initialized OpenAIEmbeddingProvider with model: {default_model}")
    
    def embed(
        self, 
        texts: Union[str, List[str]], 
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings using OpenAI API.
        
        Args:
            texts: Single text string or list of text strings
            model: Optional model name (uses default if None)
        
        Returns:
            List of embedding vectors
        
        Raises:
            ValueError: If API request fails
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        model = model or self.default_model
        
        try:
            response = self.client.embeddings.create(
                model=model,
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            logger.info(f"Successfully generated embeddings for {len(texts)} texts using model {model}")
            return embeddings
        except Exception as e:
            logger.error(f"Error while generating embeddings: {str(e)}")
            raise ValueError(f"Failed to generate embeddings: {str(e)}") from e
    
    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Get the dimension of embeddings for a model.
        
        Args:
            model: Optional model name (uses default if None)
        
        Returns:
            Dimension of the embedding vectors
        """
        model = model or self.default_model
        dimension = self.MODEL_DIMENSIONS.get(model, 1536)  # Default to 1536 if unknown
        logger.debug(f"Embedding dimension for model {model}: {dimension}")
        return dimension
