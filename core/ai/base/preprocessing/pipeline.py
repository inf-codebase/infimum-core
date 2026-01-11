"""
Chain of Responsibility pattern for preprocessing pipelines.

Each transform is a link in the chain, processing data and passing to next.
"""

from typing import List, Optional
from .base import BaseTransform
from ..data.base import DataItem


class TransformLink:
    """
    Link in the transformation chain.
    
    Chain of Responsibility Pattern: Each link processes data and passes to next.
    """
    
    def __init__(self, transform: BaseTransform):
        """
        Initialize link with transform.
        
        Args:
            transform: Transform to apply
        """
        self.transform = transform
        self._next: Optional['TransformLink'] = None
    
    def set_next(self, link: 'TransformLink') -> 'TransformLink':
        """
        Set next link in chain.
        
        Args:
            link: Next link
            
        Returns:
            Next link for chaining
        """
        self._next = link
        return link
    
    def process(self, data: DataItem) -> DataItem:
        """
        Process data and pass to next link.
        
        Args:
            data: Input data item
            
        Returns:
            Processed data item
        """
        # Process this link
        data = self.transform.transform(data)
        
        # Pass to next if exists
        if self._next:
            return self._next.process(data)
        return data


class TransformPipeline:
    """
    Chain of responsibility for transforms.
    
    Builds a chain of transforms and applies them sequentially.
    """
    
    def __init__(self, transforms: List[BaseTransform]):
        """
        Initialize pipeline with transforms.
        
        Args:
            transforms: List of transforms to apply
            
        Raises:
            ValueError: If no transforms provided
        """
        if not transforms:
            raise ValueError("At least one transform required")
        
        # Build chain
        links = [TransformLink(t) for t in transforms]
        self._head = links[0]
        current = self._head
        for link in links[1:]:
            current = current.set_next(link)
    
    def apply(self, data: DataItem) -> DataItem:
        """
        Apply pipeline to data.
        
        Args:
            data: Input data item
            
        Returns:
            Processed data item
        """
        return self._head.process(data)
    
    def __call__(self, data: DataItem) -> DataItem:
        """
        Make pipeline callable.
        
        Args:
            data: Input data item
            
        Returns:
            Processed data item
        """
        return self.apply(data)
