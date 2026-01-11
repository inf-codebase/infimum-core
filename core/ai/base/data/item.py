"""
Data item abstraction.

Represents a generic data item with metadata and type information.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
from pathlib import Path


@dataclass
class DataItem:
    """
    Generic data item.
    
    Represents any type of data (image, text, audio, video, etc.)
    with associated metadata and type information.
    """
    data: Any
    data_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[Union[str, Path]] = None
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get metadata value.
        
        Args:
            key: Metadata key
            default: Default value if key not found
            
        Returns:
            Metadata value or default
        """
        return self.metadata.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set metadata value.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def update_metadata(self, **kwargs) -> None:
        """
        Update multiple metadata values.
        
        Args:
            **kwargs: Metadata key-value pairs
        """
        self.metadata.update(kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert data item to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "data_type": self.data_type,
            "metadata": self.metadata,
            "source": str(self.source) if self.source else None,
        }
