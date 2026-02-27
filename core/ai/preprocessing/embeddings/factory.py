"""
Embedding provider factory with caching support.

This module provides a factory for creating and managing embedding provider instances.
It includes caching to avoid creating multiple instances with the same configuration.
"""

from typing import Dict, Type, Optional, List
from loguru import logger

from .base import EmbeddingProvider


class EmbeddingProviderFactory:
    """Factory for creating embedding provider instances with caching.
    
    This factory manages provider registration and instance creation.
    It caches provider instances to avoid repeated instantiation with the same config.
    
    Example:
        ```python
        # Register a provider
        EmbeddingProviderFactory.register("openai", OpenAIEmbeddingProvider)
        
        # Create or get cached instance
        provider = EmbeddingProviderFactory.create("openai", api_key="sk-...")
        
        # Clear cache if needed
        EmbeddingProviderFactory.clear_cache()
        ```
    """
    
    _providers: Dict[str, Type[EmbeddingProvider]] = {}
    _instances: Dict[str, EmbeddingProvider] = {}  # Cache instances
    
    @classmethod
    def register(cls, name: str, provider_class: Type[EmbeddingProvider]) -> None:
        """Register a provider class.
        
        Args:
            name: Provider name (e.g., "openai", "huggingface")
            provider_class: Provider class that implements EmbeddingProvider
        
        Raises:
            ValueError: If provider_class is not a subclass of EmbeddingProvider
        """
        if not issubclass(provider_class, EmbeddingProvider):
            raise ValueError(
                f"Provider class must be a subclass of EmbeddingProvider, "
                f"got {provider_class}"
            )
        
        cls._providers[name] = provider_class
        logger.info(f"Registered embedding provider: {name}")
    
    @classmethod
    def create(
        cls, 
        name: str, 
        cache: bool = True, 
        **kwargs
    ) -> EmbeddingProvider:
        """Create or get cached provider instance.
        
        Args:
            name: Provider name (must be registered)
            cache: Whether to cache the instance (default: True)
            **kwargs: Arguments to pass to provider constructor
        
        Returns:
            EmbeddingProvider instance
        
        Raises:
            ValueError: If provider name is not registered
        """
        if name not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(
                f"Unknown provider: {name}. Available providers: {available}"
            )
        
        # Create cache key from provider name and kwargs
        # Sort kwargs items for consistent hashing
        cache_key = f"{name}:{hash(frozenset(sorted(kwargs.items())))}"
        
        # Return cached instance if available and caching enabled
        if cache and cache_key in cls._instances:
            logger.debug(f"Returning cached provider instance: {name}")
            return cls._instances[cache_key]
        
        # Create new instance
        provider_class = cls._providers[name]
        try:
            instance = provider_class(**kwargs)
            
            if cache:
                cls._instances[cache_key] = instance
                logger.debug(f"Cached provider instance: {name}")
            
            return instance
        except Exception as e:
            logger.error(f"Failed to create provider {name}: {e}")
            raise
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear cached provider instances."""
        count = len(cls._instances)
        cls._instances.clear()
        logger.info(f"Cleared {count} cached provider instances")
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered provider names.
        
        Returns:
            List of registered provider names
        """
        return list(cls._providers.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a provider is registered.
        
        Args:
            name: Provider name to check
        
        Returns:
            True if registered, False otherwise
        """
        return name in cls._providers
