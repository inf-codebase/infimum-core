"""
Registry pattern for transform discovery and metadata management.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass
class TransformMetadata:
    """Metadata for a transform."""
    transform_name: str
    data_type: str
    description: str = ""
    version: str = ""
    requirements: List[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.requirements is None:
            self.requirements = []


class TransformRegistry:
    """Registry for managing transform metadata."""
    
    _transforms: Dict[str, TransformMetadata] = {}
    
    @classmethod
    def register(cls, transform_id: str, metadata: TransformMetadata) -> None:
        """
        Register transform with metadata.
        
        Args:
            transform_id: Unique transform identifier
            metadata: Transform metadata
        """
        cls._transforms[transform_id] = metadata
    
    @classmethod
    def get(cls, transform_id: str) -> Optional[TransformMetadata]:
        """
        Get transform metadata.
        
        Args:
            transform_id: Transform identifier
            
        Returns:
            TransformMetadata or None if not found
        """
        return cls._transforms.get(transform_id)
    
    @classmethod
    def list_all(cls) -> List[str]:
        """
        List all registered transforms.
        
        Returns:
            List of transform IDs
        """
        return list(cls._transforms.keys())
    
    @classmethod
    def search(cls, data_type: Optional[str] = None) -> List[str]:
        """
        Search transforms by data type.
        
        Args:
            data_type: Data type to filter by
            
        Returns:
            List of matching transform IDs
        """
        if data_type is None:
            return cls.list_all()
        return [
            tid for tid, meta in cls._transforms.items()
            if meta.data_type == data_type
        ]
    
    @classmethod
    def get_by_type(cls, data_type: str) -> List[TransformMetadata]:
        """
        Get all transforms for a data type.
        
        Args:
            data_type: Data type
            
        Returns:
            List of transform metadata
        """
        return [meta for meta in cls._transforms.values() if meta.data_type == data_type]
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered transforms."""
        cls._transforms.clear()
