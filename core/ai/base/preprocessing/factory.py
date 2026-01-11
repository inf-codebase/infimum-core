"""
Factory pattern for creating transforms and pipelines.

Allows creation of transforms without knowing concrete classes.
"""

from typing import Dict, Type, List, Optional
from .base import BaseTransform
from .pipeline import TransformPipeline


class TransformFactory:
    """Factory for creating transforms and pipelines."""
    
    _registry: Dict[str, Type[BaseTransform]] = {}
    
    @classmethod
    def register(cls, transform_name: str, transform_class: Type[BaseTransform]) -> None:
        """
        Register a transform implementation.
        
        Args:
            transform_name: Transform name
            transform_class: Transform class
        """
        cls._registry[transform_name] = transform_class
    
    @classmethod
    def create(cls, transform_name: str, **kwargs) -> BaseTransform:
        """
        Create a transform instance.
        
        Args:
            transform_name: Transform name
            **kwargs: Transform initialization arguments
            
        Returns:
            BaseTransform instance
            
        Raises:
            ValueError: If transform not registered
        """
        if transform_name not in cls._registry:
            available = cls.list_transforms()
            raise ValueError(
                f"Transform '{transform_name}' not registered. "
                f"Available transforms: {available}"
            )
        transform_class = cls._registry[transform_name]
        return transform_class(**kwargs)
    
    @classmethod
    def create_pipeline(cls, transform_names: List[str], **kwargs) -> TransformPipeline:
        """
        Create a pipeline from transform names.
        
        Args:
            transform_names: List of transform names
            **kwargs: Transform initialization arguments (applied to all)
            
        Returns:
            TransformPipeline instance
        """
        transforms = [cls.create(name, **kwargs) for name in transform_names]
        return TransformPipeline(transforms)
    
    @classmethod
    def list_transforms(cls) -> List[str]:
        """
        List available transforms.
        
        Returns:
            List of transform names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, transform_name: str) -> bool:
        """
        Check if transform is registered.
        
        Args:
            transform_name: Transform name
            
        Returns:
            True if registered
        """
        return transform_name in cls._registry
    
    @classmethod
    def unregister(cls, transform_name: str) -> None:
        """
        Unregister a transform.
        
        Args:
            transform_name: Transform name
        """
        if transform_name in cls._registry:
            del cls._registry[transform_name]
