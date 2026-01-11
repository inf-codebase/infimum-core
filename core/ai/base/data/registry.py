"""
Registry pattern for data loader discovery and metadata management.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass
class LoaderMetadata:
    """Metadata for a data loader."""
    data_type: str
    loader_name: str
    supported_formats: Set[str]
    description: str = ""
    version: str = ""
    requirements: List[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.requirements is None:
            self.requirements = []


class LoaderRegistry:
    """Registry for managing data loader metadata."""
    
    _loaders: Dict[str, LoaderMetadata] = {}
    
    @classmethod
    def register(cls, loader_id: str, metadata: LoaderMetadata) -> None:
        """
        Register loader with metadata.
        
        Args:
            loader_id: Unique loader identifier
            metadata: Loader metadata
        """
        cls._loaders[loader_id] = metadata
    
    @classmethod
    def get(cls, loader_id: str) -> Optional[LoaderMetadata]:
        """
        Get loader metadata.
        
        Args:
            loader_id: Loader identifier
            
        Returns:
            LoaderMetadata or None if not found
        """
        return cls._loaders.get(loader_id)
    
    @classmethod
    def list_all(cls) -> List[str]:
        """
        List all registered loaders.
        
        Returns:
            List of loader IDs
        """
        return list(cls._loaders.keys())
    
    @classmethod
    def search(cls, data_type: Optional[str] = None, format: Optional[str] = None) -> List[str]:
        """
        Search loaders by type and format.
        
        Args:
            data_type: Data type to filter by
            format: Required format support
            
        Returns:
            List of matching loader IDs
        """
        results = []
        for lid, meta in cls._loaders.items():
            if data_type and meta.data_type != data_type:
                continue
            if format and format not in meta.supported_formats:
                continue
            results.append(lid)
        return results
    
    @classmethod
    def get_by_type(cls, data_type: str) -> List[LoaderMetadata]:
        """
        Get all loaders for a data type.
        
        Args:
            data_type: Data type
            
        Returns:
            List of loader metadata
        """
        return [meta for meta in cls._loaders.values() if meta.data_type == data_type]
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered loaders."""
        cls._loaders.clear()
