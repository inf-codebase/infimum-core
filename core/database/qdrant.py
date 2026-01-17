import logging
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient, models, AsyncQdrantClient
from qdrant_client.http.models import PointStruct, Filter

from core.database.base import DatabaseManager
from core.database.interfaces import VectorDatabaseManager
from core.database.base import DatabaseConnectionConfig, VectorIndexConfig

logger = logging.getLogger(__name__)


class QdrantManager(VectorDatabaseManager):
    def __init__(self, qdrant_url=None, qdrant_api_key=None, use_memory=False, **kwargs):
        """Initialize QdrantManager with connection parameters.
        
        Args:
            qdrant_url: Qdrant server URL (or DatabaseConnectionConfig for new API)
            qdrant_api_key: Optional API key
            use_memory: Whether to use in-memory Qdrant
            **kwargs: Additional arguments
        """
        # Support both old API and new API (DatabaseConnectionConfig)
        if isinstance(qdrant_url, DatabaseConnectionConfig):
            config = qdrant_url
            self.qdrant_url = config.connection_string or config.host
            self.qdrant_api_key = config.password or qdrant_api_key  # API key might be in password field
            self.use_memory = getattr(config, 'use_memory', False)
        else:
            self.qdrant_url = qdrant_url
            self.qdrant_api_key = qdrant_api_key
            self.use_memory = use_memory
        self.client = None
    
    @classmethod
    def from_config(cls, config: DatabaseConnectionConfig, **kwargs):
        """Create manager from configuration object (new API).
        
        Args:
            config: DatabaseConnectionConfig instance
            **kwargs: Additional arguments (e.g., use_memory, qdrant_api_key)
        
        Returns:
            QdrantManager instance
        """
        return cls(
            qdrant_url=config.connection_string or config.host,
            qdrant_api_key=kwargs.get('qdrant_api_key') or config.password,
            use_memory=kwargs.get('use_memory', False),
            **{k: v for k, v in kwargs.items() if k not in ['qdrant_api_key', 'use_memory']}
        )

    def connect(self):
        """Establish connection to Qdrant (memory or server)."""
        if not self.client:
            if self.use_memory:
                self.client = QdrantClient(":memory:")
            else:
                self.client = QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    timeout=5,
                )

    def close(self):
        """Close the Qdrant connection (dummy for interface consistency)"""
        if self.client:
            self.client.close()
            self.client = None
    
    def is_connected(self) -> bool:
        """Check if connection is established.
        
        Returns:
            True if connected, False otherwise
        """
        return self.client is not None

    def ensure_collection(
        self, 
        collection_name: str, 
        has_vector: bool, 
        vector_size: int = 1536,
        index_config: Optional[VectorIndexConfig] = None
    ):
        try:
            self.client.get_collection(collection_name)
        except Exception:
            if has_vector:
                # Use index_config if provided, otherwise use defaults
                if index_config:
                    # Map metric_type to Qdrant Distance enum
                    distance_map = {
                        "L2": models.Distance.EUCLID,
                        "COSINE": models.Distance.COSINE,
                        "IP": models.Distance.DOT,
                    }
                    distance = distance_map.get(index_config.metric_type.upper(), models.Distance.COSINE)
                else:
                    distance = models.Distance.COSINE
                
                vector_config = models.VectorParams(
                    size=vector_size,
                    distance=distance,
                )
            else:
                vector_config = None
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vector_config,
            )
            logger.info(f"Created collection {collection_name}")

    def insert_points(
        self, collection_name: str, points: List[Dict[str, Any]], has_vector=True
    ) -> None:
        # Ensure the collection exists
        self.ensure_collection(collection_name, has_vector=has_vector)

        """Insert points (vectors) into Qdrant collection."""
        qdrant_points = [
            PointStruct(
                id=point["id"], vector=point["vector"], payload=point.get("payload", {})
            )
            for point in points
        ]
        self.client.upsert(collection_name=collection_name, points=qdrant_points)

    def query_points(
        self,
        collection_name: str,
        query_vector: List[float],
        top: int = 10,
        filter: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Query similar points by vector."""
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top,
            query_filter=filter,
        )
        return [r.dict() for r in results]

    def scroll_points(
        self,
        collection_name: str,
        offset: Optional[int] = None,
        limit: int = 100,
        filter: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """Scroll through points in the collection."""
        scroll_result = self.client.scroll(
            collection_name=collection_name,
            offset=offset,
            limit=limit,
            scroll_filter=filter,
        )
        return [point.dict() for point in scroll_result[0]]

    def delete_points(self, collection_name: str, point_ids: List[str]):
        """Delete points from a collection by their IDs."""
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=point_ids),
            )
            logger.info(f"Deleted {len(point_ids)} points from {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete points: {e}", exc_info=True)
            raise

    def update_point_payload(self, collection_name: str, point_id: str, payload: dict):
        """Update the payload of a specific point."""
        try:
            self.client.set_payload(
                collection_name=collection_name, payload=payload, points=[point_id]
            )
            logger.info(f"Updated payload for point {point_id} in {collection_name}")
        except Exception as e:
            logger.error(f"Failed to update payload: {e}", exc_info=True)
            raise

    def insert_or_update(self, model, auto_commit=True, update_if_true_conditions=None):
        """
        Insert or update a model in Qdrant.
        Note: This is a stub implementation for DatabaseManager interface.
        Qdrant uses a different data model (vectors/points) than SQL databases.
        """
        raise NotImplementedError(
            "QdrantManager does not support insert_or_update. "
            "Use insert_points() or update_point_payload() instead."
        )

    def query_or_create_new(self, model_class, query_conditions=None):
        """
        Query or create a new model instance.
        Note: This is a stub implementation for DatabaseManager interface.
        Qdrant uses a different data model (vectors/points) than SQL databases.
        """
        raise NotImplementedError(
            "QdrantManager does not support query_or_create_new. "
            "Use query_points() or scroll_points() instead."
        )


class AsyncQdrantManager(QdrantManager):
    """Asynchronous version of QdrantManager.
    
    This class provides async versions of all Qdrant operations.
    It inherits from QdrantManager but overrides methods to be async.
    
    Example:
        ```python
        async_manager = AsyncQdrantManager(qdrant_url="http://localhost:6333")
        await async_manager.connect()
        await async_manager.insert_points("my_collection", points_data)
        results = await async_manager.query_points("my_collection", query_vector)
        ```
    """
    
    def connect(self):
        """Establish an asynchronous connection to Qdrant."""
        if not self.client:
            if self.use_memory:
                self.client = AsyncQdrantClient(":memory:")
            else:
                self.client = AsyncQdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    timeout=5,
                )

    """Async version of QdrantManager for future use."""

    # Placeholder for async methods if needed in the future
    async def ensure_collection(self, collection_name, has_vector, vector_size=1536):
        """Ensure the collection exists in async mode."""
        try:
            await self.client.get_collection(collection_name)
        except Exception:
            vector_config = (
                models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                )
                if has_vector
                else None
            )
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vector_config,
            )
            logger.info(f"Created collection {collection_name}")

    async def insert_points(self, collection_name, points, has_vector=True):
        """Insert points (vectors) into Qdrant collection in async mode."""
        await self.ensure_collection(collection_name, has_vector=has_vector)

        qdrant_points = [
            PointStruct(
                id=point["id"], vector=point["vector"], payload=point.get("payload", {})
            )
            for point in points
        ]
        await self.client.upsert(collection_name=collection_name, points=qdrant_points)

    async def update_point_payload(self, collection_name, point_id, payload):
        """Update the payload of a specific point in async mode."""
        try:
            await self.client.set_payload(
                collection_name=collection_name, payload=payload, points=[point_id]
            )
            logger.info(f"Updated payload for point {point_id} in {collection_name}")
        except Exception as e:
            logger.error(f"Failed to update payload: {e}", exc_info=True)
            raise

    async def delete_points(self, collection_name, point_ids):
        """Delete points from a collection by their IDs in async mode."""
        try:
            await self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=point_ids),
            )
            logger.info(f"Deleted {len(point_ids)} points from {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete points: {e}", exc_info=True)
            raise

    async def scroll_points(self, collection_name, offset=None, limit=100, filter=None):
        """Scroll through points in the collection in async mode."""
        scroll_result = await self.client.scroll(
            collection_name=collection_name,
            offset=offset,
            limit=limit,
            scroll_filter=filter,
        )
        return [point.dict() for point in scroll_result[0]]

    async def query_points(self, collection_name, query_vector, top=10, filter=None):
        """Query similar points by vector in async mode."""
        results = await self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top,
            query_filter=filter,
        )
        return [r.dict() for r in results]

    async def close(self):
        """Close the Qdrant client connection."""
        await self.client.close()

    async def insert_or_update(self, model, auto_commit=True, update_if_true_conditions=None):
        """
        Insert or update a model in Qdrant.
        Note: This is a stub implementation for DatabaseManager interface.
        Qdrant uses a different data model (vectors/points) than SQL databases.
        """
        raise NotImplementedError(
            "AsyncQdrantManager does not support insert_or_update. "
            "Use insert_points() or update_point_payload() instead."
        )

    async def query_or_create_new(self, model_class, query_conditions=None):
        """
        Query or create a new model instance.
        Note: This is a stub implementation for DatabaseManager interface.
        Qdrant uses a different data model (vectors/points) than SQL databases.
        """
        raise NotImplementedError(
            "AsyncQdrantManager does not support query_or_create_new. "
            "Use query_points() or scroll_points() instead."
        )
