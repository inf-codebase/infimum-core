from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional


class DatabaseManager(ABC):
    """Base class for database managers"""
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def insert_or_update(self, model, auto_commit=True, update_if_true_conditions=None):
        """
        Insert a new record or update an existing one based on conditions.

        Args:
            model: The model instance to insert or update
            auto_commit: Whether to commit the transaction immediately
            update_if_true_conditions: Dictionary of field-value pairs to use for finding existing records

        Returns:
            The inserted or updated model instance
        """
        pass

    @abstractmethod
    def query_or_create_new(self, model_class, query_conditions=None):
        """
        Query for an existing record or create a new instance if not found.

        Args:
            model_class: The model class to query or instantiate
            query_conditions: Dictionary of field-value pairs to use for finding existing records

        Returns:
            An existing record instance or a new (unsaved) instance
        """
        pass
    
    @abstractmethod
    def get_session(self):
        pass

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
    
    model_config = ConfigDict(extra="allow")  # Allow provider-specific params


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
    
    model_config = ConfigDict(extra="allow")  # Allow database-specific fields
