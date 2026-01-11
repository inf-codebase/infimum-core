"""
Factory pattern for creating data loaders.

Allows creation of loaders without knowing concrete classes.
"""

from typing import Dict, Type, List, Optional
from .base import BaseLoader


class LoaderFactory:
    """Factory for creating data loaders."""
    
    _registry: Dict[str, Type[BaseLoader]] = {}
    
    @classmethod
    def register(cls, loader_name: str, loader_class: Type[BaseLoader]) -> None:
        """
        Register a loader implementation.
        
        Args:
            loader_name: Loader name
            loader_class: Loader class
        """
        cls._registry[loader_name] = loader_class
    
    @classmethod
    def create(cls, loader_name: str) -> BaseLoader:
        """
        Create a loader instance.
        
        Args:
            loader_name: Loader name
            
        Returns:
            BaseLoader instance
            
        Raises:
            ValueError: If loader not registered
        """
        if loader_name not in cls._registry:
            available = cls.list_loaders()
            raise ValueError(
                f"Loader '{loader_name}' not registered. "
                f"Available loaders: {available}"
            )
        loader_class = cls._registry[loader_name]
        return loader_class()
    
    @classmethod
    def list_loaders(cls) -> List[str]:
        """
        List available loaders.
        
        Returns:
            List of loader names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, loader_name: str) -> bool:
        """
        Check if loader is registered.
        
        Args:
            loader_name: Loader name
            
        Returns:
            True if registered
        """
        return loader_name in cls._registry
    
    @classmethod
    def unregister(cls, loader_name: str) -> None:
        """
        Unregister a loader.
        
        Args:
            loader_name: Loader name
        """
        if loader_name in cls._registry:
            del cls._registry[loader_name]
