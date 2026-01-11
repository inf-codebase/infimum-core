"""
Factory pattern for creating model providers.

Allows creation of providers without knowing concrete classes.
"""

from typing import Dict, Tuple, Type, List, Optional
from .base import BaseProvider
from .config import ModelConfig


class ProviderFactory:
    """Factory for creating model providers."""
    
    _registry: Dict[Tuple[str, str], Type[BaseProvider]] = {}
    
    @classmethod
    def register(cls, model_type: str, provider_name: str, provider_class: Type[BaseProvider]) -> None:
        """
        Register a provider implementation.
        
        Args:
            model_type: Model type (llm, vlm, speech)
            provider_name: Provider name
            provider_class: Provider class
        """
        key = (model_type, provider_name)
        cls._registry[key] = provider_class
    
    @classmethod
    def create(cls, model_type: str, provider_name: str, config: ModelConfig) -> BaseProvider:
        """
        Create a provider instance.
        
        Args:
            model_type: Model type
            provider_name: Provider name
            config: Model configuration
            
        Returns:
            BaseProvider instance
            
        Raises:
            ValueError: If provider not registered
        """
        key = (model_type, provider_name)
        if key not in cls._registry:
            available = cls.list_providers(model_type)
            raise ValueError(
                f"Provider '{provider_name}' for '{model_type}' not registered. "
                f"Available providers: {available}"
            )
        provider_class = cls._registry[key]
        return provider_class(config)
    
    @classmethod
    def list_providers(cls, model_type: Optional[str] = None) -> List[str]:
        """
        List available providers.
        
        Args:
            model_type: Optional model type to filter by
            
        Returns:
            List of provider names
        """
        if model_type:
            return [name for (mt, name) in cls._registry.keys() if mt == model_type]
        return [name for (_, name) in cls._registry.keys()]
    
    @classmethod
    def is_registered(cls, model_type: str, provider_name: str) -> bool:
        """
        Check if provider is registered.
        
        Args:
            model_type: Model type
            provider_name: Provider name
            
        Returns:
            True if registered
        """
        return (model_type, provider_name) in cls._registry
    
    @classmethod
    def unregister(cls, model_type: str, provider_name: str) -> None:
        """
        Unregister a provider.
        
        Args:
            model_type: Model type
            provider_name: Provider name
        """
        key = (model_type, provider_name)
        if key in cls._registry:
            del cls._registry[key]
