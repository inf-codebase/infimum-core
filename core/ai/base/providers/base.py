"""
Base provider interface using Strategy and Template Method patterns.

Strategy Pattern: Different providers implement different loading strategies.
Template Method Pattern: Common workflow defined in get_model() with hooks.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from .config import ModelConfig
from ..observers.base import Observable
from ..observers.events import Event, EventType


@dataclass
class ModelHandle:
    """Handle to a loaded model."""
    model: Any
    config: ModelConfig
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a component from the model handle."""
        if isinstance(self.model, dict):
            return self.model.get(key, default)
        return getattr(self.model, key, default)


class BaseProvider(ABC, Observable):
    """
    Abstract base class for model providers.
    
    Uses Strategy Pattern: Each provider implements different loading strategies.
    Uses Template Method Pattern: get_model() defines common workflow with hooks.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize provider.
        
        Args:
            config: Optional model configuration
        """
        super().__init__()
        self.config = config
    
    @abstractmethod
    def load_model(self, config: ModelConfig) -> ModelHandle:
        """
        Load a model - strategy-specific implementation.
        
        Args:
            config: Model configuration
            
        Returns:
            ModelHandle: Handle to the loaded model
        """
        pass
    
    @abstractmethod
    def unload_model(self, handle: ModelHandle) -> None:
        """
        Unload a model - strategy-specific implementation.
        
        Args:
            handle: Model handle to unload
        """
        pass
    
    def get_model(self, config: ModelConfig) -> ModelHandle:
        """
        Template method - defines common workflow with hooks.
        
        Steps:
        1. Validate configuration (common)
        2. Check cache (hook)
        3. Load model (abstract - must implement)
        4. Post-process (hook)
        5. Cache (hook)
        
        Args:
            config: Model configuration
            
        Returns:
            ModelHandle: Handle to the loaded model
        """
        # Step 1: Validate (common)
        self._validate_config(config)
        
        # Step 2: Check cache (hook)
        if self._is_cached(config):
            cached_handle = self._get_cached(config)
            self.notify(Event(
                type=EventType.MODEL_LOADING_COMPLETED,
                data={"config": config, "cached": True},
                source=self.__class__.__name__
            ))
            return cached_handle
        
        # Step 3: Load model (abstract - must implement)
        self.notify(Event(
            type=EventType.MODEL_LOADING_STARTED,
            data={"config": config},
            source=self.__class__.__name__
        ))
        
        try:
            handle = self.load_model(config)
            
            # Step 4: Post-process (hook)
            handle = self._post_process(handle)
            
            # Step 5: Cache (hook)
            self._cache(config, handle)
            
            self.notify(Event(
                type=EventType.MODEL_LOADING_COMPLETED,
                data={"config": config, "handle": handle},
                source=self.__class__.__name__
            ))
            
            return handle
            
        except Exception as e:
            self.notify(Event(
                type=EventType.MODEL_LOADING_FAILED,
                data={"config": config, "error": str(e)},
                source=self.__class__.__name__
            ))
            raise
    
    def _validate_config(self, config: ModelConfig) -> None:
        """
        Validate configuration - common step, can't be overridden easily.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not config.model_path:
            raise ValueError("model_path is required")
    
    def _is_cached(self, config: ModelConfig) -> bool:
        """
        Check if model is cached - hook method, can be overridden.
        
        Args:
            config: Model configuration
            
        Returns:
            bool: True if cached
        """
        return False
    
    def _get_cached(self, config: ModelConfig) -> ModelHandle:
        """
        Get cached model - hook method, can be overridden.
        
        Args:
            config: Model configuration
            
        Returns:
            ModelHandle: Cached model handle
        """
        raise NotImplementedError("Caching not implemented")
    
    def _post_process(self, handle: ModelHandle) -> ModelHandle:
        """
        Post-process loaded model - hook method, can be overridden.
        
        Args:
            handle: Model handle
            
        Returns:
            ModelHandle: Post-processed handle
        """
        return handle
    
    def _cache(self, config: ModelConfig, handle: ModelHandle) -> None:
        """
        Cache model - hook method, can be overridden.
        
        Args:
            config: Model configuration
            handle: Model handle to cache
        """
        pass
    
    def get_model_info(self, handle: ModelHandle) -> Dict[str, Any]:
        """
        Get information about a loaded model.
        
        Args:
            handle: Model handle
            
        Returns:
            Dict with model information
        """
        return {
            "config": handle.config,
            "metadata": handle.metadata,
        }
