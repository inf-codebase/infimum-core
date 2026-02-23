"""
Database interface definitions for different database types.

This module provides abstract base classes for different database manager types:
- VectorDatabaseManager: For vector databases (Milvus, Qdrant)
- RelationalDatabaseManager: For relational databases (PostgreSQL, MySQL, SQLite)
- DocumentDatabaseManager: For document databases (MongoDB)

It also provides Protocol-based interfaces for async operations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Type, TYPE_CHECKING, Protocol, runtime_checkable

from core.database.base import DatabaseManager
from core.base.entity import Document

if TYPE_CHECKING:
    from core.database.base import VectorIndexConfig


class VectorDatabaseManager(DatabaseManager):
    """Interface for vector database operations.
    
    This interface defines the contract for vector database managers that support
    operations like inserting vectors, querying by similarity, and managing collections.
    """
    
    @abstractmethod
    def ensure_collection(
        self, 
        collection_name: str, 
        has_vector: bool, 
        vector_size: int,
        index_config: Optional['VectorIndexConfig'] = None
    ) -> Any:
        """Ensure collection exists with proper configuration.
        
        Args:
            collection_name: Name of the collection
            has_vector: Whether the collection contains vector field
            vector_size: Dimension of the vector field
            index_config: Optional index configuration (forward reference)
        
        Returns:
            Collection object or equivalent
        """
        pass
    
    @abstractmethod
    def insert_points(
        self, 
        collection_name: str, 
        points: List[Dict[str, Any]], 
        has_vector: bool = True
    ) -> None:
        """Insert vector points into collection.
        
        Args:
            collection_name: Name of the collection
            points: List of point dictionaries with 'vector' and optional 'payload'/'metadata'
            has_vector: Whether points contain vector data
        """
        pass
    
    @abstractmethod
    def query_points(
        self,
        collection_name: str,
        query_vector: List[float],
        top: int = 10,
        filter: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Query similar points by vector.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector for similarity search
            top: Number of results to return
            filter: Optional filter expression (format depends on backend)
        
        Returns:
            List of result dictionaries with 'id', 'distance', and 'payload'/'metadata'
        """
        pass
    
    @abstractmethod
    def scroll_points(
        self,
        collection_name: str,
        offset: Optional[int] = None,
        limit: int = 100,
        filter: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Scroll through points in the collection.
        
        Args:
            collection_name: Name of the collection
            offset: Optional offset for pagination
            limit: Maximum number of results
            filter: Optional filter expression
        
        Returns:
            List of point dictionaries
        """
        pass
    
    @abstractmethod
    def delete_points(self, collection_name: str, point_ids: List[str]) -> None:
        """Delete points from a collection by their IDs.
        
        Args:
            collection_name: Name of the collection
            point_ids: List of point IDs to delete
        """
        pass
    
    @abstractmethod
    def update_point_payload(
        self, 
        collection_name: str, 
        point_id: str, 
        payload: dict
    ) -> None:
        """Update the payload/metadata of a specific point.
        
        Args:
            collection_name: Name of the collection
            point_id: ID of the point to update
            payload: Dictionary of metadata to update
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connection is established.
        
        Returns:
            True if connected, False otherwise
        """
        pass


# Async Protocol Interfaces (for type checking and structural typing)
@runtime_checkable
class AsyncVectorDatabaseManager(Protocol):
    """Protocol for async vector database operations.
    
    This protocol defines the async interface for vector database managers.
    Classes that implement async versions of vector operations should match this protocol.
    
    Note: This is a Protocol for structural typing. Classes don't need to explicitly
    inherit from it, they just need to implement the methods.
    
    Example:
        ```python
        class MyAsyncManager:
            async def connect(self) -> None: ...
            async def insert_points(...) -> None: ...
            # ... other async methods
        ```
    """
    
    async def connect(self) -> None:
        """Establish async connection to database."""
        ...
    
    async def close(self) -> None:
        """Close async connection."""
        ...
    
    async def ensure_collection(
        self,
        collection_name: str,
        has_vector: bool,
        vector_size: int,
        index_config: Optional['VectorIndexConfig'] = None
    ) -> Any:
        """Ensure collection exists (async)."""
        ...
    
    async def insert_points(
        self,
        collection_name: str,
        points: List[Dict[str, Any]],
        has_vector: bool = True
    ) -> None:
        """Insert vector points (async)."""
        ...
    
    async def query_points(
        self,
        collection_name: str,
        query_vector: List[float],
        top: int = 10,
        filter: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Query similar points (async)."""
        ...
    
    async def scroll_points(
        self,
        collection_name: str,
        offset: Optional[int] = None,
        limit: int = 100,
        filter: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Scroll through points (async)."""
        ...
    
    async def delete_points(self, collection_name: str, point_ids: List[str]) -> None:
        """Delete points (async)."""
        ...
    
    async def update_point_payload(
        self,
        collection_name: str,
        point_id: str,
        payload: dict
    ) -> None:
        """Update point payload (async)."""
        ...
    
    def is_connected(self) -> bool:
        """Check if connection is established."""
        ...


class RelationalDatabaseManager(DatabaseManager):
    """Interface for relational database operations.
    
    This interface defines the contract for relational database managers that support
    operations like insert/update, query, and table creation.
    """
    
    @abstractmethod
    def insert_or_update(
        self, 
        model, 
        auto_commit: bool = True, 
        update_if_true_conditions: Optional[Dict] = None
    ):
        """Insert a new record or update an existing one based on conditions.
        
        Args:
            model: The model instance to insert or update
            auto_commit: Whether to commit the transaction immediately
            update_if_true_conditions: Dictionary of field-value pairs to use for finding existing records
        
        Returns:
            The inserted or updated model instance
        """
        pass
    
    @abstractmethod
    def query_or_create_new(
        self, 
        model_class, 
        query_conditions: Optional[Dict] = None
    ):
        """Query for an existing record or create a new instance if not found.
        
        Args:
            model_class: The model class to query or instantiate
            query_conditions: Dictionary of field-value pairs to use for finding existing records
        
        Returns:
            An existing record instance or a new (unsaved) instance
        """
        pass
    
    @abstractmethod
    def create_tables(
        self, 
        entities: Optional[List] = None, 
        drop_first: bool = False
    ) -> bool:
        """Create database tables for entity classes.
        
        Args:
            entities: List of entity classes. If None, creates all tables bound to metadata.
            drop_first: If True, drops all tables before creating them.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connection is established.
        
        Returns:
            True if connected, False otherwise
        """
        pass


# Async Protocol Interfaces (for type checking and structural typing)
@runtime_checkable
class AsyncVectorDatabaseManager(Protocol):
    """Protocol for async vector database operations.
    
    This protocol defines the async interface for vector database managers.
    Classes that implement async versions of vector operations should match this protocol.
    
    Note: This is a Protocol for structural typing. Classes don't need to explicitly
    inherit from it, they just need to implement the methods.
    
    Example:
        ```python
        class MyAsyncManager:
            async def connect(self) -> None: ...
            async def insert_points(...) -> None: ...
            # ... other async methods
        ```
    """
    
    async def connect(self) -> None:
        """Establish async connection to database."""
        ...
    
    async def close(self) -> None:
        """Close async connection."""
        ...
    
    async def ensure_collection(
        self,
        collection_name: str,
        has_vector: bool,
        vector_size: int,
        index_config: Optional['VectorIndexConfig'] = None
    ) -> Any:
        """Ensure collection exists (async)."""
        ...
    
    async def insert_points(
        self,
        collection_name: str,
        points: List[Dict[str, Any]],
        has_vector: bool = True
    ) -> None:
        """Insert vector points (async)."""
        ...
    
    async def query_points(
        self,
        collection_name: str,
        query_vector: List[float],
        top: int = 10,
        filter: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Query similar points (async)."""
        ...
    
    async def scroll_points(
        self,
        collection_name: str,
        offset: Optional[int] = None,
        limit: int = 100,
        filter: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Scroll through points (async)."""
        ...
    
    async def delete_points(self, collection_name: str, point_ids: List[str]) -> None:
        """Delete points (async)."""
        ...
    
    async def update_point_payload(
        self,
        collection_name: str,
        point_id: str,
        payload: dict
    ) -> None:
        """Update point payload (async)."""
        ...
    
    def is_connected(self) -> bool:
        """Check if connection is established."""
        ...


class DocumentDatabaseManager(DatabaseManager):
    """Interface for document database operations (MongoDB, etc.).
    
    This interface defines the contract for document database managers that support
    operations like insert, find, update, and delete on document collections.
    """
    
    @abstractmethod
    def get_collection(
        self, 
        collection: Union[str, Type[Document], Document]
    ) -> Any:
        """Get collection by name, class, or instance.
        
        Args:
            collection: Can be a string name, Document class, or Document instance
        
        Returns:
            Collection object or equivalent
        """
        pass
    
    @abstractmethod
    def insert_one(
        self, 
        collection: Union[str, Type[Document], Document], 
        document: Dict[str, Any]
    ) -> str:
        """Insert a single document.
        
        Args:
            collection: Collection identifier (string, class, or instance)
            document: Document dictionary to insert
        
        Returns:
            ID of the inserted document as string
        """
        pass
    
    @abstractmethod
    def find_one(
        self, 
        collection: Union[str, Type[Document], Document], 
        query: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Find a single document.
        
        Args:
            collection: Collection identifier (string, class, or instance)
            query: Query dictionary
        
        Returns:
            Document dictionary if found, None otherwise
        """
        pass
    
    @abstractmethod
    def update_one(
        self,
        collection: Union[str, Type[Document], Document],
        query: Dict[str, Any],
        update: Dict[str, Any]
    ) -> bool:
        """Update a single document.
        
        Args:
            collection: Collection identifier (string, class, or instance)
            query: Query dictionary to find the document
            update: Update dictionary
        
        Returns:
            True if document was updated, False otherwise
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connection is established.
        
        Returns:
            True if connected, False otherwise
        """
        pass


# Async Protocol Interfaces (for type checking and structural typing)
@runtime_checkable
class AsyncVectorDatabaseManager(Protocol):
    """Protocol for async vector database operations.
    
    This protocol defines the async interface for vector database managers.
    Classes that implement async versions of vector operations should match this protocol.
    
    Note: This is a Protocol for structural typing. Classes don't need to explicitly
    inherit from it, they just need to implement the methods.
    
    Example:
        ```python
        class MyAsyncManager:
            async def connect(self) -> None: ...
            async def insert_points(...) -> None: ...
            # ... other async methods
        ```
    """
    
    async def connect(self) -> None:
        """Establish async connection to database."""
        ...
    
    async def close(self) -> None:
        """Close async connection."""
        ...
    
    async def ensure_collection(
        self,
        collection_name: str,
        has_vector: bool,
        vector_size: int,
        index_config: Optional['VectorIndexConfig'] = None
    ) -> Any:
        """Ensure collection exists (async)."""
        ...
    
    async def insert_points(
        self,
        collection_name: str,
        points: List[Dict[str, Any]],
        has_vector: bool = True
    ) -> None:
        """Insert vector points (async)."""
        ...
    
    async def query_points(
        self,
        collection_name: str,
        query_vector: List[float],
        top: int = 10,
        filter: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Query similar points (async)."""
        ...
    
    async def scroll_points(
        self,
        collection_name: str,
        offset: Optional[int] = None,
        limit: int = 100,
        filter: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Scroll through points (async)."""
        ...
    
    async def delete_points(self, collection_name: str, point_ids: List[str]) -> None:
        """Delete points (async)."""
        ...
    
    async def update_point_payload(
        self,
        collection_name: str,
        point_id: str,
        payload: dict
    ) -> None:
        """Update point payload (async)."""
        ...
    
    def is_connected(self) -> bool:
        """Check if connection is established."""
        ...
