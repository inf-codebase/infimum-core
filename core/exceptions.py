"""
Custom exception hierarchy for the core module.

This module provides a structured exception hierarchy that allows for
better error handling and more specific error messages throughout the codebase.
"""


class CoreException(Exception):
    """Base exception for core module.
    
    All custom exceptions in the core module should inherit from this class.
    This allows catching all core-related exceptions with a single except clause.
    """
    pass


class DatabaseException(CoreException):
    """Base exception for database operations.
    
    This exception is raised for general database-related errors.
    More specific exceptions should inherit from this class.
    """
    pass


class DatabaseConnectionError(DatabaseException):
    """Database connection errors.
    
    Raised when there are issues connecting to a database, such as:
    - Connection timeout
    - Invalid connection parameters
    - Network issues
    - Authentication failures
    """
    pass


class DatabaseQueryError(DatabaseException):
    """Database query execution errors.
    
    Raised when there are issues executing queries, such as:
    - SQL syntax errors
    - Constraint violations
    - Invalid query parameters
    """
    pass


class DatabaseConfigurationError(DatabaseException):
    """Database configuration errors.
    
    Raised when there are issues with database configuration, such as:
    - Missing required configuration parameters
    - Invalid configuration values
    - Configuration conflicts
    """
    pass


class ConfigurationError(CoreException):
    """Configuration-related errors.
    
    Raised for general configuration issues that are not database-specific,
    such as:
    - Missing environment variables
    - Invalid configuration file format
    - Configuration validation failures
    """
    pass


class ProviderError(CoreException):
    """Provider-related errors.
    
    Base exception for errors related to external providers (embedding,
    LLM, etc.). More specific provider errors should inherit from this.
    """
    pass


class EmbeddingError(ProviderError):
    """Embedding generation errors.
    
    Raised when there are issues generating embeddings, such as:
    - API key missing or invalid
    - API request failures
    - Rate limiting
    - Invalid input text
    """
    pass


class RegistryError(CoreException):
    """Registry-related errors.
    
    Raised when there are issues with the plugin registry system, such as:
    - Attempting to register an invalid backend
    - Backend not found in registry
    - Registration conflicts
    """
    pass
