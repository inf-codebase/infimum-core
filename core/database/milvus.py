"""
Milvus Repository Module

This module provides MilvusManager and AsyncMilvusManager classes for interacting with a Milvus vector database.
These classes implement the DatabaseManager interface and provide methods for connecting to Milvus,
managing collections, and performing vector operations such as insertion, querying, and deletion.

Usage:
    # Synchronous usage
    manager = MilvusManager(milvus_host="localhost", milvus_port=19530)
    manager.connect()
    manager.insert_points("my_collection", points_data)
    results = manager.query_points("my_collection", query_vector)

    # Asynchronous usage
    async_manager = AsyncMilvusManager(milvus_host="localhost", milvus_port=19530)
    await async_manager.connect()
    await async_manager.insert_points("my_collection", points_data)
    results = await async_manager.query_points("my_collection", query_vector)

Note: Make sure to close the connection when you're done using the manager.
"""

from typing import List, Dict, Any, Optional

from pymilvus import (
    connections,
    utility,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
)

from core.database.postgres import DatabaseManager
from loguru import logger


class MilvusManager(DatabaseManager):
    def __init__(self, milvus_host="localhost", milvus_port=19530):
        """Initialize MilvusManager with connection parameters."""
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.connection = None

    def connect(self):
        """Establish connection to Milvus."""
        if not self.connection:
            try:
                self.connection = connections.connect(
                    alias="default",
                    host=self.milvus_host,
                    port=self.milvus_port,
                )
                logger.info(f"Connected to Milvus at {self.milvus_host}:{self.milvus_port}")
            except Exception as e:
                logger.error(f"Failed to connect to Milvus: {e}", exc_info=True)
                raise

    def close(self):
        """Close the Milvus connection."""
        if self.connection:
            connections.disconnect("default")
            self.connection = None
            logger.info("Disconnected from Milvus")

    def is_connected(self):
        """Check if the connection to Milvus is established."""
        return self.connection is not None

    def ensure_collection(
        self, collection_name: str, has_vector: bool, vector_size: int = 1536
    ):
        """
        Ensure the collection exists in Milvus and has proper index.

        Args:
            collection_name: Name of the collection
            has_vector: Whether the collection contains vector field
            vector_size: Dimension of the vector field

        Returns:
            Collection object
        """
        if not utility.has_collection(collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]
            if has_vector:
                fields.append(FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_size))

            schema = CollectionSchema(fields)
            collection = Collection(name=collection_name, schema=schema)
            logger.info(f"Created collection {collection_name}")

            # Create index if collection has vector field
            if has_vector:
                self.ensure_index(collection)
        else:
            collection = Collection(name=collection_name)
            logger.info(f"Collection {collection_name} already exists")

            # Ensure index exists for existing collection with vector field
            if has_vector and not collection.has_index():
                self.ensure_index(collection)

        return collection

    def ensure_index(self, collection: Collection):
        """
        Ensure the collection has a proper vector index.

        Args:
            collection: Milvus Collection object
        """
        try:
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }

            collection.create_index(
                field_name="vector",
                index_params=index_params
            )
            logger.info(f"Created index for collection {collection.name}")
        except Exception as e:
            if "index already exists" not in str(e).lower():
                logger.error(f"Failed to create index: {e}")
                raise
            logger.info(f"Index already exists for collection {collection.name}")

    def insert_points(
        self, collection_name: str, points: List[Dict[str, Any]], has_vector=True
    ) -> None:
        """Insert points (vectors) into Milvus collection."""
        collection = self.ensure_collection(collection_name, has_vector)

        try:
            insert_data = []
            for point in points:
                data = {
                    "metadata": point.get("metadata", {})
                }
                if has_vector:
                    data["vector"] = point["vector"]
                insert_data.append(data)

            collection.insert(insert_data)
            collection.flush()
            logger.info(f"Inserted {len(points)} points into {collection_name}")
        except Exception as e:
            logger.error(f"Failed to insert points: {e}", exc_info=True)
            raise

    def query_points(
        self,
        collection_name: str,
        query_vector: List[float],
        top: int = 10,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query similar points by vector."""
        collection = Collection(name=collection_name)
        collection.load()

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top,
            expr=filter,
            output_fields=["id", "payload"],
        )

        return [
            {"id": hit.id, "distance": hit.distance, "payload": hit.entity.get("payload")}
            for hit in results[0]
        ]

    def scroll_points(
        self,
        collection_name: str,
        offset: Optional[int] = None,
        limit: int = 100,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Scroll through points in the collection."""
        collection = Collection(name=collection_name)
        results = collection.query(
            expr=filter,
            offset=offset,
            limit=limit,
            output_fields=["id", "payload", "vector"],
        )

        return [
            {"id": item["id"], "payload": item["payload"], "vector": item.get("vector")}
            for item in results
        ]

    def delete_points(self, collection_name: str, point_ids: List[str]):
        """Delete points from a collection by their IDs."""
        collection = Collection(name=collection_name)
        try:
            collection.delete(f"id in {point_ids}")
            logger.info(f"Deleted {len(point_ids)} points from {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete points: {e}", exc_info=True)
            raise

    def update_point_payload(self, collection_name: str, point_id: str, payload: dict):
        """Update the payload of a specific point."""
        collection = Collection(name=collection_name)
        try:
            collection.upsert([{"id": point_id, "payload": payload}])
            logger.info(f"Updated payload for point {point_id} in {collection_name}")
        except Exception as e:
            logger.error(f"Failed to update payload: {e}", exc_info=True)
            raise


import asyncio

class AsyncMilvusManager(MilvusManager):
    async def connect(self):
        """Establish an asynchronous connection to Milvus."""
        if not self.connection:
            try:
                loop = asyncio.get_event_loop()
                self.connection = await loop.run_in_executor(
                    None,
                    lambda: connections.connect(
                        alias="default",
                        host=self.milvus_host,
                        port=self.milvus_port,
                    )
                )
                logger.info(f"Connected to Milvus at {self.milvus_host}:{self.milvus_port}")
            except Exception as e:
                logger.error(f"Failed to connect to Milvus: {e}", exc_info=True)
                raise

    async def close(self):
        """Close the Milvus connection asynchronously."""
        if self.connection:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, connections.disconnect, "default")
            self.connection = None
            logger.info("Disconnected from Milvus")

    async def ensure_collection(
        self, collection_name: str, has_vector: bool, vector_size: int = 1536
    ):
        """
        Ensure the collection exists in Milvus and has proper index asynchronously.

        Args:
            collection_name: Name of the collection
            has_vector: Whether the collection contains vector field
            vector_size: Dimension of the vector field

        Returns:
            Collection object
        """
        loop = asyncio.get_event_loop()
        if not await loop.run_in_executor(None, utility.has_collection, collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="payload", dtype=DataType.JSON),
            ]
            if has_vector:
                fields.append(FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_size))

            schema = CollectionSchema(fields)
            collection = await loop.run_in_executor(
                None, Collection, collection_name, schema
            )
            logger.info(f"Created collection {collection_name}")

            # Create index if collection has vector field
            if has_vector:
                await self.ensure_index(collection)
        else:
            collection = await loop.run_in_executor(None, Collection, collection_name)
            logger.info(f"Collection {collection_name} already exists")

            # Ensure index exists for existing collection with vector field
            if has_vector:
                has_index = await loop.run_in_executor(None, collection.has_index)
                if not has_index:
                    await self.ensure_index(collection)

        return collection

    async def ensure_index(self, collection: Collection):
        """
        Ensure the collection has a proper vector index asynchronously.

        Args:
            collection: Milvus Collection object
        """
        try:
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                collection.create_index,
                "vector",
                index_params
            )
            logger.info(f"Created index for collection {collection.name}")
        except Exception as e:
            if "index already exists" not in str(e).lower():
                logger.error(f"Failed to create index: {e}")
                raise
            logger.info(f"Index already exists for collection {collection.name}")

    async def insert_points(
        self, collection_name: str, points: List[Dict[str, Any]], has_vector=True
    ) -> None:
        """Insert points (vectors) into Milvus collection asynchronously."""
        collection = await self.ensure_collection(collection_name, has_vector)

        try:
            loop = asyncio.get_event_loop()
            for point in points:
                insert_data = {
                    "id": str(point["id"]),  # Ensure ID is a string
                    "payload": point.get("payload", {})
                }
                if has_vector:
                    insert_data["vector"] = point["vector"]

                await loop.run_in_executor(None, collection.insert, [insert_data])  # Insert as a list with a single item

            await loop.run_in_executor(None, collection.flush)
            logger.info(f"Inserted {len(points)} points into {collection_name}")
        except Exception as e:
            logger.error(f"Failed to insert points: {e}", exc_info=True)
            raise

    async def query_points(
        self,
        collection_name: str,
        query_vector: List[float],
        top: int = 10,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query similar points by vector asynchronously."""
        loop = asyncio.get_event_loop()
        collection = await loop.run_in_executor(None, Collection, collection_name)
        await loop.run_in_executor(None, collection.load)

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = await loop.run_in_executor(
            None,
            collection.search,
            [query_vector],
            "vector",
            search_params,
            top,
            filter,
            ["id", "payload"],
        )

        return [
            {"id": hit.id, "distance": hit.distance, "payload": hit.entity.get("payload")}
            for hit in results[0]
        ]

    async def scroll_points(
        self,
        collection_name: str,
        offset: Optional[int] = None,
        limit: int = 100,
        filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Scroll through points in the collection asynchronously."""
        loop = asyncio.get_event_loop()
        collection = await loop.run_in_executor(None, Collection, collection_name)
        results = await loop.run_in_executor(
            None,
            collection.query,
            filter,
            offset,
            limit,
            ["id", "payload", "vector"],
        )

        return [
            {"id": item["id"], "payload": item["payload"], "vector": item.get("vector")}
            for item in results
        ]

    async def delete_points(self, collection_name: str, point_ids: List[str]):
        """Delete points from a collection by their IDs asynchronously."""
        loop = asyncio.get_event_loop()
        collection = await loop.run_in_executor(None, Collection, collection_name)
        try:
            await loop.run_in_executor(None, collection.delete, f"id in {point_ids}")
            logger.info(f"Deleted {len(point_ids)} points from {collection_name}")
        except Exception as e:
            logger.error(f"Failed to delete points: {e}", exc_info=True)
            raise

    async def update_point_payload(self, collection_name: str, point_id: str, payload: dict):
        """Update the payload of a specific point asynchronously."""
        loop = asyncio.get_event_loop()
        collection = await loop.run_in_executor(None, Collection, collection_name)
        try:
            await loop.run_in_executor(
                None, collection.upsert, [{"id": point_id, "payload": payload}]
            )
            logger.info(f"Updated payload for point {point_id} in {collection_name}")
        except Exception as e:
            logger.error(f"Failed to update payload: {e}", exc_info=True)
            raise
