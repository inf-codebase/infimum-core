"""
MongoDB Repository Module

This module provides SyncMongoManager and AsyncMongoManager classes for interacting with MongoDB.
These classes implement the DocumentDatabaseManager interface and provide methods for connecting to MongoDB,
managing collections, and performing document operations such as insertion, querying, and deletion.

Usage:
    # Synchronous usage
    manager = SyncMongoManager(
        connection_string="mongodb://localhost:27017",
        database="test_db"
    )
    manager.insert_one("my_collection", document_data)
    result = manager.find_one("my_collection", query)

    # Asynchronous usage
    async_manager = AsyncMongoManager(
        connection_string="mongodb://localhost:27017",
        database="test_db"
    )
    await async_manager.insert_one("my_collection", document_data)
    result = await async_manager.find_one("my_collection", query)

Note: Make sure to close the connection when you're done using the manager.
"""

from typing import Any, Collection, Dict, List, Optional, Union, Type
from datetime import datetime
from loguru import logger

from infimum.base.entity import Document
from infimum.database.base import DatabaseManager
from infimum.database.interfaces import DocumentDatabaseManager

# Optional MongoDB import - only needed for MongoDB managers
try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None


class MongoManagerBase:
    """Base class for MongoDB managers with common configuration handling"""
    def __init__(self, connection_string: str, database: str, **kwargs):
        self.connection_string = connection_string
        self.database = database

    def _get_collection_name(self, collection: Union[str, Type[Document], Document]) -> str:
        """Get collection name from string, Document class or instance"""
        if isinstance(collection, str):
            return collection
        elif isinstance(collection, type) and issubclass(collection, Document):
            return collection.get_collection_name()
        elif issubclass(type(collection), Document):  # Check if instance is derived from Document
            return collection.get_collection_name()
        raise TypeError(f"Invalid collection type: {type(collection)}")


class SyncMongoManager(MongoManagerBase, DocumentDatabaseManager):
    """Synchronous MongoDB Manager"""

    def __init__(self, connection_string: str, database: str, **kwargs):
        super().__init__(connection_string, database)
        self.connect()
    
    def is_connected(self) -> bool:
        """Check if connection is established."""
        return self.client is not None and hasattr(self, 'db')

    def get_collection(self, collection: Union[str, Type[Document], Document]) -> Collection:
        """Get MongoDB collection

        Args:
            collection: Can be:
                - str: Collection name
                - Document class: Will use class's collection name
                - Document instance: Will use instance's collection name

        Returns:
            Collection: MongoDB collection

        Examples:
            # Using string
            collection = manager.get_collection("companies")

            # Using Document class
            collection = manager.get_collection(Company)

            # Using Document instance
            company = Company(name="Test")
            collection = manager.get_collection(company)
        """
        collection_name = self._get_collection_name(collection)
        return self.client[self.database][collection_name]

    def connect(self):
        if MongoClient is None:
            raise ImportError(
                "SyncMongoManager requires 'pymongo' to be installed. "
                "Install it with: pip install -e '.[mongo]' or pip install pymongo>=4.16.0"
            )
        if not hasattr(self, 'client') or not self.client:
            self.client = MongoClient(self.connection_string)
            self.db = self.client[self.database]

    def insert_one(self, collection: Union[str, Type[Document], Document], document: Dict[str, Any]) -> str:
        """Insert a single document"""
        document['created_at'] = datetime.utcnow()
        document['updated_at'] = document['created_at']
        result = self.get_collection(collection).insert_one(document)
        return str(result.inserted_id)

    def insert_many(self, collection: Union[str, Type[Document], Document], documents: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple documents"""
        now = datetime.utcnow()
        for doc in documents:
            doc['created_at'] = now
            doc['updated_at'] = now
        result = self.get_collection(collection).insert_many(documents)
        return [str(id) for id in result.inserted_ids]

    def find_one(self, collection: Union[str, Type[Document], Document], query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document"""
        return self.get_collection(collection).find_one(query)

    def find_many(self,
                 collection: Union[str, Type[Document], Document],
                 query: Dict[str, Any],
                 skip: int = 0,
                 limit: int = 100) -> List[Dict[str, Any]]:
        """Find multiple documents"""
        return list(self.get_collection(collection).find(query).skip(skip).limit(limit))

    def update_one(self,
                  collection: Union[str, Type[Document], Document],
                  query: Dict[str, Any],
                  update: Dict[str, Any]) -> bool:
        """Update a single document"""
        update = {"$set": update}
        update['$set']['updated_at'] = datetime.utcnow()
        result = self.get_collection(collection).update_one(query, update)
        return result.modified_count > 0

    def update_many(self,
                   collection: Union[str, Type[Document], Document],
                   query: Dict[str, Any],
                   update: Dict[str, Any]) -> int:
        """Update multiple documents"""
        update = {"$set": update}
        update['$set']['updated_at'] = datetime.utcnow()
        result = self.get_collection(collection).update_many(query, update)
        return result.modified_count

    def delete_one(self, collection: Union[str, Type[Document], Document], query: Dict[str, Any]) -> bool:
        """Delete a single document"""
        result = self.get_collection(collection).delete_one(query)
        return result.deleted_count > 0

    def delete_many(self, collection: Union[str, Type[Document], Document], query: Dict[str, Any]) -> int:
        """Delete multiple documents"""
        result = self.get_collection(collection).delete_many(query)
        return result.deleted_count

    def close(self):
        """Close the database connection"""
        if hasattr(self, 'client') and self.client:
            self.client.close()

    def insert_or_update(self, model, auto_commit=True, update_if_true_conditions=None):
        """
        Insert a model or update if it exists based on update_if_true_conditions.
        For MongoDB, auto_commit is ignored as operations are immediately committed.
        """
        collection_name = self._get_collection_name(model.__class__)
        collection = self.client[self.database][collection_name]

        # Convert model to dict for MongoDB
        model_dict = model.dict(exclude_unset=True, exclude_none=True)

        if update_if_true_conditions:
            # Try to update
            result = collection.update_one(
                update_if_true_conditions,
                {"$set": model_dict}
            )

            if result.matched_count > 0:
                return model

        # Insert if not updated
        result = collection.insert_one(model_dict)
        model.id = str(result.inserted_id)
        return model

    def query_or_create_new(self, model_class, query_conditions=None):
        """
        Query for an existing record or create a new instance if not found.
        """
        if not query_conditions:
            return model_class()

        collection_name = self._get_collection_name(model_class)
        collection = self.client[self.database][collection_name]

        # Try to find existing record
        existing = collection.find_one(query_conditions)

        if existing:
            # Convert MongoDB document to model instance
            if '_id' in existing:
                existing['id'] = str(existing.pop('_id'))
            return model_class(**existing)
        else:
            # Create new instance with the query conditions
            return model_class(**query_conditions)


class AsyncMongoManager(MongoManagerBase, DocumentDatabaseManager):
    """Asynchronous MongoDB Manager"""

    def __init__(self, connection_string: str, database: str, **kwargs):
        super().__init__(connection_string, database)
        self.connect()
    
    def is_connected(self) -> bool:
        """Check if connection is established."""
        return self.client is not None and hasattr(self, 'db')

    def get_collection(self, collection: Union[str, Type[Document], Document]):
        """Get MongoDB collection asynchronously"""
        collection_name = self._get_collection_name(collection)
        return self.client[self.database][collection_name]

    def connect(self):
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError:
            raise ImportError(
                "AsyncMongoManager requires 'motor' to be installed. "
                "Install it with: pip install -e '.[mongo]' or pip install motor>=3.0.0"
            )
        if not hasattr(self, 'client') or not self.client:
            self.client = AsyncIOMotorClient(self.connection_string)
            self.db = self.client[self.database]

    async def insert_one(self, collection: Union[str, Type[Document], Document], document: Dict[str, Any]) -> str:
        """Insert a single document asynchronously"""
        document['created_at'] = datetime.utcnow()
        document['updated_at'] = document['created_at']
        result = await self.get_collection(collection).insert_one(document)
        return str(result.inserted_id)

    async def insert_many(self, collection: Union[str, Type[Document], Document], documents: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple documents asynchronously"""
        now = datetime.utcnow()
        for doc in documents:
            doc['created_at'] = now
            doc['updated_at'] = now
        result = self.get_collection(collection).insert_many(documents)
        return [str(id) for id in result.inserted_ids]

    async def find_one(self, collection: Union[str, Type[Document], Document], query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document asynchronously"""
        return await self.get_collection(collection).find_one(query)

    async def find_many(self,
                       collection: Union[str, Type[Document], Document],
                       query: Dict[str, Any],
                       skip: int = 0,
                       limit: int = -1) -> List[Dict[str, Any]]:
        """Find multiple documents asynchronously"""
        if limit == -1:
            cursor = self.get_collection(collection).find(query).skip(skip)
        else:
            cursor = self.get_collection(collection).find(query).skip(skip).limit(limit)
        return [doc async for doc in cursor]

    async def update_one(self,
                        collection: Union[str, Type[Document], Document],
                        query: Dict[str, Any],
                        update: Dict[str, Any]) -> bool:
        """Update a single document asynchronously"""
        update = {
            "$set": update
        }
        update['$set']['updated_at'] = datetime.utcnow()
        result = await self.get_collection(collection).update_one(
            query, update
        )
        return result.modified_count > 0

    async def update_many(self,
                         collection: Union[str, Type[Document], Document],
                         query: Dict[str, Any],
                         update: Dict[str, Any]) -> int:
        """Update multiple documents asynchronously"""
        update = {
            "$set": update
        }
        update['$set']['updated_at'] = datetime.utcnow()
        result = await self.get_collection(collection).update_many(query, update)
        return result.modified_count

    async def delete_one(self, collection: Union[str, Type[Document], Document], query: Dict[str, Any]) -> bool:
        """Delete a single document asynchronously"""
        result = await self.get_collection(collection).delete_one(query)
        return result.deleted_count > 0

    async def delete_many(self, collection: Union[str, Type[Document], Document], query: Dict[str, Any]) -> int:
        """Delete multiple documents asynchronously"""
        result = await self.get_collection(collection).delete_many(query)
        return result.deleted_count

    async def close(self):
        """Close the database connection asynchronously"""
        if hasattr(self, 'client') and self.client:
            await self.client.close()

    async def insert_or_update(self, model, auto_commit=True, update_if_true_conditions=None):
        """
        Insert a model or update if it exists based on update_if_true_conditions.
        For MongoDB, auto_commit is ignored as operations are immediately committed.
        """
        collection_name = self._get_collection_name(model.__class__)
        collection = self.client[self.database][collection_name]

        # Convert model to dict for MongoDB
        model_dict = model.dict(exclude_unset=True, exclude_none=True)

        if update_if_true_conditions:
            # Try to update
            result = await collection.update_one(
                update_if_true_conditions,
                {"$set": model_dict}
            )

            if result.matched_count > 0:
                return model

        # Insert if not updated
        result = await collection.insert_one(model_dict)
        model.id = str(result.inserted_id)
        return model

    async def query_or_create_new(self, model_class, query_conditions=None):
        """
        Query for an existing record or create a new instance if not found.
        """
        if not query_conditions:
            return model_class()

        collection_name = self._get_collection_name(model_class)
        collection = self.client[self.database][collection_name]

        # Try to find existing record
        existing = await collection.find_one(query_conditions)

        if existing:
            # Convert MongoDB document to model instance
            if '_id' in existing:
                existing['id'] = str(existing.pop('_id'))
            return model_class(**existing)
        else:
            # Create new instance with the query conditions
            return model_class(**query_conditions)
