import logging
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient, models, AsyncQdrantClient
from qdrant_client.http.models import PointStruct, Filter

from src.core.database.postgres import DatabaseManager

logger = logging.getLogger(__name__)


class QdrantManager(DatabaseManager):
    def __init__(self, qdrant_url=None, qdrant_api_key=None, use_memory=False):
        """Initialize QdrantManager with connection parameters."""
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.use_memory = use_memory
        self.client = None

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
        self.client.close()
        self.client = None

    def ensure_collection(
        self, collection_name: str, has_vector: bool, vector_size: int = 1536
    ):
        try:
            self.client.get_collection(collection_name)
        except Exception:
            vector_config = (
                models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                )
                if has_vector
                else None
            )
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


class AsyncQdrantManager(QdrantManager):
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
