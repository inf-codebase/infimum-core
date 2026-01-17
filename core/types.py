"""
Type aliases and common type definitions for the core module.

This module provides type aliases for common patterns used throughout
the codebase, making type hints more readable and maintainable.
"""

from typing import Dict, List, Any, Union, Optional, Protocol
from typing_extensions import Protocol as ProtocolType  # For Python < 3.8 compatibility

# Vector database types
Point = Dict[str, Any]
"""Type alias for a single point in a vector database.
A point typically contains 'id', 'vector', and 'payload'/'metadata' fields.
"""

Points = List[Point]
"""Type alias for a list of points."""

Vector = List[float]
"""Type alias for a single embedding vector (list of floats)."""

Vectors = List[Vector]
"""Type alias for a list of embedding vectors."""

# Database configuration protocol
class DatabaseConfig(Protocol):
    """Protocol for database configuration objects.
    
    This protocol defines the structure that database configuration objects
    should follow, allowing for flexible configuration types.
    """
    host: Optional[str]
    port: Optional[int]
    database: Optional[str]
    user: Optional[str]
    password: Optional[str]
    connection_string: Optional[str]


# Collection/Table identifiers
CollectionIdentifier = Union[str, type, Any]
"""Type alias for collection/table identifiers.
Can be a string name, a class (Document/Entity), or an instance.
"""

# Query results
QueryResult = Dict[str, Any]
"""Type alias for a single query result."""

QueryResults = List[QueryResult]
"""Type alias for query result lists."""

# Model types
ModelInstance = Any
"""Type alias for a model instance (BaseEntity, Document, etc.)."""

ModelClass = type
"""Type alias for a model class."""

# Filter types
FilterExpression = Union[str, Dict[str, Any], Any]
"""Type alias for filter expressions.
Can be a string (SQL-like), dict (MongoDB-like), or backend-specific format.
"""
