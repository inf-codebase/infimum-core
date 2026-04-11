"""
Base transform interface using Strategy pattern.

Strategy Pattern: Different transforms implement different transformation strategies.
"""

from abc import ABC, abstractmethod
from ..data.base import DataItem


class BaseTransform(ABC):
    """
    Abstract base class for data transforms.
    
    Uses Strategy Pattern: Each transform implements different transformation strategies.
    """
    
    @abstractmethod
    def transform(self, data: DataItem) -> DataItem:
        """
        Transform data - strategy-specific implementation.
        
        Args:
            data: Input data item
            
        Returns:
            Transformed data item
        """
        pass
    
    def __call__(self, data: DataItem) -> DataItem:
        """
        Make transform callable.
        
        Args:
            data: Input data item
            
        Returns:
            Transformed data item
        """
        return self.transform(data)
