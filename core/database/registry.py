"""
Database backend registry for plugin-based extensibility.

This module provides a registry system that allows database backends to be
registered and created without modifying core code. This enables a plugin
architecture where new database backends can be added easily.
"""

from typing import Dict, Type, Callable, Optional, List
from loguru import logger

from core.database.base import DatabaseManager
from core.database.config import DatabaseConnectionConfig


class DatabaseBackendRegistry:
    """Registry for database backend implementations.
    
    This registry allows database backends to be registered and created
    dynamically. It supports both class-based registration and factory functions.
    
    Example:
        ```python
        # Register a backend class
        DatabaseBackendRegistry.register("my_backend", MyBackendManager)
        
        # Create an instance
        config = DatabaseConnectionConfig(host="localhost", port=5432)
        manager = DatabaseBackendRegistry.create("my_backend", config)
        ```
    """
    
    _backends: Dict[str, Type[DatabaseManager]] = {}
    _factories: Dict[str, Callable] = {}
    
    @classmethod
    def register(
        cls, 
        name: str, 
        backend_class: Type[DatabaseManager],
        factory: Optional[Callable] = None
    ) -> None:
        """Register a database backend.
        
        Args:
            name: Backend name (e.g., "postgres", "milvus", "my_custom_backend")
            backend_class: Backend class that implements DatabaseManager
            factory: Optional factory function for custom instantiation logic
        
        Raises:
            ValueError: If backend_class is not a subclass of DatabaseManager
        """
        if not issubclass(backend_class, DatabaseManager):
            raise ValueError(
                f"Backend class must be a subclass of DatabaseManager, "
                f"got {backend_class}"
            )
        
        cls._backends[name] = backend_class
        if factory:
            cls._factories[name] = factory
        
        logger.info(f"Registered database backend: {name}")
    
    @classmethod
    def create(
        cls, 
        name: str, 
        config: DatabaseConnectionConfig,
        **kwargs
    ) -> DatabaseManager:
        """Create a database manager instance.
        
        Args:
            name: Backend name (must be registered)
            config: DatabaseConnectionConfig instance
            **kwargs: Additional arguments to pass to manager constructor
        
        Returns:
            DatabaseManager instance
        
        Raises:
            ValueError: If backend name is not registered
        """
        # Try factory first if available
        if name in cls._factories:
            logger.debug(f"Using factory for backend: {name}")
            return cls._factories[name](config, **kwargs)
        
        # Try class-based creation
        if name in cls._backends:
            backend_class = cls._backends[name]
            logger.debug(f"Creating backend instance: {name}")
            
            # Try from_config first (new API)
            if hasattr(backend_class, 'from_config'):
                return backend_class.from_config(config, **kwargs)
            else:
                # Fallback: extract config attributes for backward compatibility
                config_dict = config.dict(exclude_none=True)
                return backend_class(**config_dict, **kwargs)
        
        # Backend not found
        available = list(cls._backends.keys())
        raise ValueError(
            f"Unknown backend: {name}. Available backends: {available}"
        )
    
    @classmethod
    def list_backends(cls) -> List[str]:
        """List all registered backend names.
        
        Returns:
            List of registered backend names
        """
        return list(cls._backends.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a backend is registered.
        
        Args:
            name: Backend name to check
        
        Returns:
            True if registered, False otherwise
        """
        return name in cls._backends
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a backend.
        
        Args:
            name: Backend name to unregister
        
        Raises:
            KeyError: If backend is not registered
        """
        if name not in cls._backends:
            raise KeyError(f"Backend '{name}' is not registered")
        
        del cls._backends[name]
        if name in cls._factories:
            del cls._factories[name]
        
        logger.info(f"Unregistered database backend: {name}")
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered backends."""
        count = len(cls._backends)
        cls._backends.clear()
        cls._factories.clear()
        logger.info(f"Cleared {count} registered backends")
