"""
Configuration models for database connections and operations.

This module provides Pydantic models for type-safe configuration of database
connections, vector collections, and indexes.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class VectorIndexConfig(BaseModel):
    """Configuration for vector database indexes.
    
    This model defines the configuration for creating indexes on vector fields
    in vector databases. Different backends may support different index types
    and parameters.
    
    Attributes:
        metric_type: Distance metric to use (e.g., "L2", "COSINE", "IP")
        index_type: Index algorithm (e.g., "IVF_FLAT", "HNSW")
        params: Index-specific parameters as a dictionary
    """
    
    metric_type: str = Field(
        default="L2",
        description="Distance metric (L2, COSINE, IP, etc.)"
    )
    index_type: str = Field(
        default="IVF_FLAT",
        description="Index algorithm (IVF_FLAT, HNSW, etc.)"
    )
    params: Dict[str, Any] = Field(
        default_factory=lambda: {"nlist": 1024},
        description="Index-specific parameters"
    )
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow provider-specific params


class VectorCollectionConfig(BaseModel):
    """Configuration for vector collections.
    
    This model defines the configuration for creating collections in vector
    databases, including vector dimensions and index settings.
    
    Attributes:
        vector_size: Dimension of the vector field
        has_vector: Whether the collection contains vector field
        index_config: Optional index configuration (auto-created if None)
    """
    
    vector_size: int = Field(
        default=1536,
        description="Vector dimension",
        gt=0
    )
    has_vector: bool = Field(
        default=True,
        description="Whether the collection contains vector field"
    )
    index_config: Optional[VectorIndexConfig] = Field(
        default=None,
        description="Index configuration (auto-created if None)"
    )


class DatabaseConnectionConfig(BaseModel):
    """Base configuration for database connections.
    
    This model provides a common interface for database connection parameters.
    Different database types may use different subsets of these fields.
    
    Attributes:
        host: Database host address
        port: Database port number
        database: Database name
        user: Database username
        password: Database password
        connection_string: Full connection string (alternative to individual fields)
    """
    
    host: Optional[str] = Field(
        default=None,
        description="Database host address"
    )
    port: Optional[int] = Field(
        default=None,
        description="Database port number",
        gt=0,
        lt=65536
    )
    database: Optional[str] = Field(
        default=None,
        description="Database name"
    )
    user: Optional[str] = Field(
        default=None,
        description="Database username"
    )
    password: Optional[str] = Field(
        default=None,
        description="Database password"
    )
    connection_string: Optional[str] = Field(
        default=None,
        description="Full connection string (alternative to individual fields)"
    )
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow database-specific fields
