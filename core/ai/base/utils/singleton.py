"""
Enhanced Singleton pattern implementation.

Provides thread-safe singleton with better error handling and extensibility.
"""

import threading
from typing import Optional, TypeVar, Type

T = TypeVar('T')


class SingletonMeta(type):
    """
    Metaclass for singleton pattern.
    
    Thread-safe implementation with double-checked locking.
    """
    
    _instances: dict = {}
    _locks: dict = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        """Create or return existing instance."""
        if cls not in cls._instances:
            # Get or create lock for this class
            if cls not in cls._locks:
                with cls._lock:
                    if cls not in cls._locks:
                        cls._locks[cls] = threading.Lock()
            
            # Double-checked locking
            with cls._locks[cls]:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        
        return cls._instances[cls]
    
    @classmethod
    def get_instance(cls, instance_cls: Type[T]) -> Optional[T]:
        """
        Get singleton instance of a class.
        
        Args:
            instance_cls: Class to get instance of
            
        Returns:
            Instance or None if not created
        """
        return cls._instances.get(instance_cls)
    
    @classmethod
    def reset(cls, instance_cls: Type) -> None:
        """
        Reset singleton instance (useful for testing).
        
        Args:
            instance_cls: Class to reset
        """
        if instance_cls in cls._instances:
            del cls._instances[instance_cls]


class Singleton:
    """
    Base class for singleton pattern.
    
    Usage:
        class MyClass(Singleton):
            def __init__(self):
                # Initialization code
                pass
    """
    
    def __new__(cls, *args, **kwargs):
        """Ensure only one instance exists."""
        if not hasattr(cls, '_instance'):
            cls._instance = None
        if cls._instance is None:
            cls._lock = threading.Lock()
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> 'Singleton':
        """
        Get singleton instance.
        
        Returns:
            Singleton instance
        """
        if not hasattr(cls, '_instance') or cls._instance is None:
            cls()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset singleton (useful for testing)."""
        if hasattr(cls, '_instance'):
            cls._instance = None
