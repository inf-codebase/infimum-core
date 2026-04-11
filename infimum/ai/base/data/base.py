"""
Base data loader interface using Strategy and Template Method patterns.

Strategy Pattern: Different loaders implement different loading strategies.
Template Method Pattern: Common workflow defined in load() with hooks.
"""

from abc import ABC, abstractmethod
from typing import Callable, Union, Any, Optional, List
from pathlib import Path
from infimum.engine.design_pattern import Observable, Event, EventType
from .item import DataItem


class BaseLoader(ABC, Observable):
    """
    Abstract base class for data loaders.
    
    Uses Strategy Pattern: Each loader implements different loading strategies.
    Uses Template Method Pattern: load() defines common workflow with hooks.
    """
    
    def __init__(self):
        """Initialize loader."""
        super().__init__()
    
    @abstractmethod
    def _load(self, source: Union[str, Path, Any], data_collator: Optional[Callable] = None, frame_indices: List[int] = None) -> DataItem:
        """
        Load data - strategy-specific implementation.
        
        Args:
            source: Data source (path, URL, stream, etc.)
            data_collator: Optional data collator function
            frame_indices: Optional list of frame indices to load (for video loading)
        Returns:
            DataItem: Loaded data item
        """
        pass
    
    def load(self, source: Union[str, Path, Any], data_collator: Optional[Callable] = None, frame_indices: List[int] = None) -> DataItem:
        """
        Template method - defines common workflow with hooks.
        
        Steps:
        1. Validate source (common)
        2. Pre-process source (hook)
        3. Load data (abstract - must implement)
        4. Post-process data (hook)
        5. Validate data (hook)
        
        Args:
            source: Data source
            data_collator: Optional data collator function
            frame_indices: Optional list of frame indices to load (for video loading)
        Returns:
            DataItem: Loaded data item
        """
        # Step 1: Validate (common)
        source = self._validate_source(source)
        
        # Step 2: Pre-process (hook)
        source = self._pre_process_source(source)
        
        # Step 3: Load data (abstract - must implement)
        self.notify(Event(
            type=EventType.DATA_LOADING_STARTED,
            data={"source": str(source)},
            source=self.__class__.__name__
        ))
        
        try:
            item = self._load(source, data_collator, frame_indices)
            
            # Step 4: Post-process (hook)
            item = self._post_process_data(item)
            
            # Step 5: Validate (hook)
            self._validate_data(item)
            
            self.notify(Event(
                type=EventType.DATA_LOADING_COMPLETED,
                data={"source": str(source), "data_type": item.data_type},
                source=self.__class__.__name__
            ))
            
            return item
            
        except Exception as e:
            self.notify(Event(
                type=EventType.DATA_LOADING_FAILED,
                data={"source": str(source), "error": str(e)},
                source=self.__class__.__name__
            ))
            raise
    
    def _validate_source(self, source: Union[str, Path, Any]) -> Union[str, Path, Any]:
        """
        Validate source - common step.
        
        Args:
            source: Data source
            
        Returns:
            Validated source
            
        Raises:
            ValueError: If source is invalid
        """
        if source is None:
            raise ValueError("Source cannot be None")
        return source
    
    def _pre_process_source(self, source: Union[str, Path, Any]) -> Union[str, Path, Any]:
        """
        Pre-process source - hook method, can be overridden.
        
        Args:
            source: Data source
            
        Returns:
            Pre-processed source
        """
        return source
    
    def _post_process_data(self, item: DataItem) -> DataItem:
        """
        Post-process loaded data - hook method, can be overridden.
        
        Args:
            item: Data item
            
        Returns:
            Post-processed data item
        """
        return item
    
    def _validate_data(self, item: DataItem) -> None:
        """
        Validate loaded data - hook method, can be overridden.
        
        Args:
            item: Data item to validate
            
        Raises:
            ValueError: If data is invalid
        """
        if item.data is None:
            raise ValueError("Loaded data cannot be None")
        if not item.data_type:
            raise ValueError("Data type must be specified")
