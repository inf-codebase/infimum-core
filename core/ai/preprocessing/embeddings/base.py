"""
Base interface for embedding providers.

This module defines the abstract base class that all embedding providers must implement.
"""

"""
Base interface for embedding providers.

This module defines the abstract base class that all embedding providers must implement.
Providers can be OpenAI, HuggingFace, Cohere, or any other embedding service.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.
    
    This class defines the interface that all embedding providers must implement.
    Providers can be OpenAI, HuggingFace, Cohere, or any other embedding service.
    
    Example:
        ```python
        class MyEmbeddingProvider(EmbeddingProvider):
            def embed(self, texts, model=None):
                # Implementation
                pass
            
            def get_embedding_dimension(self, model=None):
                # Implementation
                pass
        ```
    """
    
    @abstractmethod
    def embed(
        self, 
        texts: Union[str, List[str]], 
        model: Optional[str] = None
    ) -> List[List[float]]:
        """Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of text strings
            model: Optional model name (uses default if None)
        
        Returns:
            List of embedding vectors, each as a list of floats.
            If a single text was provided, returns a list with one vector.
        
        Raises:
            ValueError: If the API key is not set or if the API request fails.
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Get the dimension of embeddings for a model.
        
        Args:
            model: Optional model name (uses default if None)
        
        Returns:
            Dimension of the embedding vectors for the specified model.
        """
        pass
