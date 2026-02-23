"""
Error handling utilities for consistent error management.

This module provides decorators and utilities for consistent error handling
across the codebase, ensuring that errors are properly logged and transformed
into appropriate custom exceptions.
"""

from functools import wraps
from typing import Callable, TypeVar, Any
from loguru import logger

from core.utils.exceptions import (
    DatabaseException,
    DatabaseConnectionError,
    DatabaseQueryError,
    DatabaseConfigurationError,
    EmbeddingError,
    ProviderError,
)

T = TypeVar('T')


def handle_database_errors(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for consistent database error handling.
    
    This decorator catches database-related exceptions and ensures they
    are properly logged and wrapped in appropriate custom exceptions.
    
    Example:
        ```python
        @handle_database_errors
        def insert_data(self, data):
            # Database operations
            pass
        ```
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DatabaseException:
            # Re-raise our custom exceptions as-is
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise DatabaseException(f"Database operation failed: {e}") from e
    return wrapper


def handle_embedding_errors(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for consistent embedding error handling.
    
    This decorator catches embedding-related exceptions and ensures they
    are properly logged and wrapped in EmbeddingError.
    
    Example:
        ```python
        @handle_embedding_errors
        def generate_embeddings(self, texts):
            # Embedding operations
            pass
        ```
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (EmbeddingError, ProviderError):
            # Re-raise provider exceptions as-is
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise EmbeddingError(f"Embedding operation failed: {e}") from e
    return wrapper


def handle_connection_errors(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for consistent connection error handling.
    
    This decorator specifically handles connection-related errors and
    wraps them in DatabaseConnectionError.
    
    Example:
        ```python
        @handle_connection_errors
        def connect(self):
            # Connection logic
            pass
        ```
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DatabaseConnectionError:
            # Re-raise connection errors as-is
            raise
        except (ConnectionError, TimeoutError, OSError) as e:
            # Wrap system connection errors
            logger.error(f"Connection error in {func.__name__}: {e}", exc_info=True)
            raise DatabaseConnectionError(f"Failed to establish connection: {e}") from e
        except Exception as e:
            # Wrap other unexpected exceptions
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise DatabaseConnectionError(f"Connection operation failed: {e}") from e
    return wrapper
