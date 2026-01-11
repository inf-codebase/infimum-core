from sqlalchemy import create_engine, text
import pandas as pd
from typing import Any, Collection, Dict, List, Optional, Union, Type
from src.core.base.entity import Document
from src.core.utils import auto_config
from sqlalchemy.orm import sessionmaker
from abc import abstractmethod
import numpy as np
from pymongo import MongoClient

from contextlib import contextmanager
from datetime import datetime
from typing import Optional, List, Dict, Any
from loguru import logger
from typing import List
from urllib.parse import quote
from sqlalchemy.exc import OperationalError
from sqlalchemy.engine import make_url


class DatabaseManager:
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

class SQLManager(DatabaseManager):
    """Base SQL manager implementation"""

    def __init__(self, engine_info: str = None, **kwargs):
        """Initialize with engine info"""
        self.engine_info = engine_info
        self.engine = None
        self.session = None

    def connect(self, show_sql_query_flag=False):
        """Connect to the database"""
        self.engine = create_engine(self.engine_info, echo=show_sql_query_flag)
        session = sessionmaker(bind=self.engine)
        self.session = session()

    @contextmanager
    def get_session(self):
        """Get a database session as a context manager.

        This method provides a session that automatically handles:
        - Session creation (if not already connected)
        - Commit on successful completion
        - Rollback on exceptions
        - Proper cleanup

        Usage:
            with self.db.get_session() as session:
                # Use session for database operations
                user = session.query(User).filter_by(email=email).first()
                session.add(new_user)
                # Session is automatically committed here
        """
        # Ensure we're connected
        if not self.is_connected():
            self.connect()

        # Create a new session for each request to avoid state issues
        # expire_on_commit=False allows entities to be used after session closes
        Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        session = Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise e
        finally:
            session.close()

    def is_connected(self):
        """Check if the database is connected"""
        return self.session is not None and self.engine is not None

    def close(self):
        """Close the database connection"""
        if self.session:
            self.session.close()
        if self.engine:
            self.engine.dispose()

    def execute(self, sql):
        """Execute raw SQL query"""
        try:
            result = self.session.execute(sql)
            self.session.commit()
            return result
        except Exception as e:
            self.session.rollback()
            raise e

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
        try:
            if update_if_true_conditions:
                q = self.session.query(model.__class__).filter_by(**update_if_true_conditions)
                if q.count() > 0:
                    db_entity = q.first()

                    # Get model data using appropriate method
                    if hasattr(model, 'dict'):
                        # Pydantic model
                        data = model.dict(exclude_unset=True, exclude_none=True, exclude={"id"})
                    elif hasattr(model, 'to_dict'):
                        # Model with to_dict method
                        data = model.to_dict(exclude=["id"])
                    else:
                        # Fallback to using vars() but filter out private attributes and id
                        data = {k: v for k, v in vars(model).items()
                               if not k.startswith('_') and k != 'id'}

                    # Update fields from the dictionary
                    for key, value in data.items():
                        setattr(db_entity, key, value)
                    self.session.add(db_entity)
                    if auto_commit:
                        self.session.commit()
                        self.session.refresh(db_entity)
                    return db_entity
                else:
                    self.session.add(model)
                    if auto_commit:
                        self.session.commit()
                        self.session.refresh(model)
                    return model
            else:
                self.session.add(model)
                if auto_commit:
                    self.session.commit()
                    self.session.refresh(model)
                return model
        except Exception as e:
            # Make sure to roll back the session on any error
            self.session.rollback()
            logger.error(f"Error in insert_or_update: {str(e)}")
            raise e

    def query_or_create_new(self, model_class, query_conditions=None):
        """
        Query for an existing record or create a new instance if not found.

        Args:
            model_class: The model class to query or instantiate
            query_conditions: Dictionary of field-value pairs to use for finding existing records

        Returns:
            An existing record instance or a new (unsaved) instance
        """
        try:
            q = self.session.query(model_class).filter_by(**(query_conditions or {}))
            if query_conditions and q.count() > 0:
                return q.first()
            else:
                return model_class(**(query_conditions or {}))
        except Exception as e:
            # Make sure to roll back the session on any error
            self.session.rollback()
            logger.error(f"Error in query_or_create_new: {str(e)}")
            raise e

    def create_tables(self, entities=None, drop_first=False):
        """Create database tables for all provided entity classes

        Args:
            entities (list, optional): List of entity classes. If None, creates all tables bound to the metadata.
            drop_first (bool, optional): If True, drops all tables before creating them.
        """
        from src.core.base.entity import BaseEntity

        try:
            if drop_first:
                # Only drop tables for the provided entities if specified
                if entities:
                    for entity in entities:
                        entity.__table__.drop(self.engine, checkfirst=True)
                else:
                    BaseEntity.metadata.drop_all(self.engine)

            # Create tables
            if entities:
                # Create only specified tables
                for entity in entities:
                    entity.__table__.create(self.engine, checkfirst=True)
            else:
                # Create all tables
                BaseEntity.metadata.create_all(self.engine)

            logger.info(f"Tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            return False

class PostgresDatabaseManagerImpl(SQLManager):
    def __init__(self, engine_info=None, **kwargs):
        super().__init__(engine_info, **kwargs)

    def get_collection(self, name: str):
        # For PostgreSQL, this could return a table object
        return name  # or implement table handling

    def to_dataframe(self, sql, chunksize=None):
        data = pd.read_sql_query(sql, con=self.engine, chunksize=chunksize)
        return data

    def add_limit_if_debug(self, sql_command):
        new_query = f"{sql_command} limit {auto_config.DB_LIMIT_QUERY_RECORDS}" if auto_config.DEBUG else sql_command
        return new_query

    def get_column_names(self, table_name, schema="public"):
        sql = f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = '{schema}' AND table_name = '{table_name}';
        """
        return self.execute(sql=sql).all()

    def get_column_index(self, column_name, table_name, schema="public"):
        columns = self.get_column_names(table_name=table_name, schema=schema)
        index_array = np.argwhere(np.array(columns) == column_name)
        return int(index_array[0][0])

    def insert_or_update(self, model, auto_commit=True, update_if_true_conditions=None, log = False):
        try:
            if log: logger.info(f"Inserting or updating model: {model.__class__.__name__} with conditions: {update_if_true_conditions}")
            if update_if_true_conditions:
                q = self.session.query(model.__class__).filter_by(**update_if_true_conditions)
                if q.count() > 0:
                    if log: logger.info(f"Updating model: {model.__class__.__name__}")
                    db_entity = q.first()

                    # Get model data using appropriate method
                    if hasattr(model, 'dict'):
                        # Pydantic model
                        data = model.dict(exclude_unset=True, exclude_none=True, exclude={"id"})
                    elif hasattr(model, 'to_dict'):
                        # Model with to_dict method
                        data = model.to_dict(exclude=["id"])
                    else:
                        # Fallback to using vars() but filter out private attributes and id
                        data = {k: v for k, v in vars(model).items()
                               if not k.startswith('_') and k != 'id'}

                    # Update fields from the dictionary
                    for key, value in data.items():
                        setattr(db_entity, key, value)
                    self.session.add(db_entity)
                    self.session.commit()
                    self.session.refresh(db_entity)
                    self.session.flush()
                else:
                    if log: logger.info(f"Inserting model: {model.__class__.__name__} with conditions: {update_if_true_conditions}")
                    self.session.add(model)
                    if auto_commit:
                        self.session.commit()
            else:
                if log: logger.info(f"Inserting model: {model.__class__.__name__}")
                self.session.add(model)
                if auto_commit:
                    self.session.commit()
            return model
        except Exception as e:
            # Roll back the session on any error
            self.session.rollback()
            logger.error(f"Error in insert_or_update: {str(e)}")
            raise e

    def query_or_create_new(self, model_class, query_conditions=None):
        try:
            q = self.session.query(model_class).filter_by(**query_conditions)
            if query_conditions and q.count() > 0:
                return q.first()
            else:
                return model_class(**query_conditions)
        except Exception as e:
            # Roll back the session on any error
            self.session.rollback()
            logger.error(f"Error in query_or_create_new: {str(e)}")
            raise e

    def query(self, model_class, query_conditions=None):
        try:
            if query_conditions is None:
                # When no conditions are provided, return all records
                q = self.session.query(model_class)
            else:
                # Apply the filter conditions
                q = self.session.query(model_class).filter_by(**query_conditions)

            return q
        except Exception as e:
            # Roll back the session on any error
            self.session.rollback()
            logger.error(f"Error in query: {str(e)}")
            raise e

class SyncMongoManager(MongoManagerBase, DatabaseManager):
    """Synchronous MongoDB Manager"""

    def __init__(self, connection_string: str, database: str, **kwargs):
        super().__init__(connection_string, database)
        self.connect()

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
        if not self.client:
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
        if self.client:
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

class AsyncMongoManager(MongoManagerBase, DatabaseManager):
    """Asynchronous MongoDB Manager"""

    def __init__(self, connection_string: str, database: str, **kwargs):
        super().__init__(connection_string, database)
        self.connect()

    def get_collection(self, collection: Union[str, Type[Document], Document]):
        """Get MongoDB collection asynchronously"""
        collection_name = self._get_collection_name(collection)
        return self.client[self.database][collection_name]

    def connect(self):
        from motor.motor_asyncio import AsyncIOMotorClient
        if not self.client:
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
        if self.client:
            self.client.close()

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


class DatabaseFactory:
    """Factory class for creating database managers."""

    class DatabaseType:
        """Enum of supported database types."""
        POSTGRES = "postgres"
        MONGO = "mongo"
        PDF = "pdf"
        SQLITE = "sqlite"
        MYSQL = "mysql"
        REDIS = "redis"
        ARANGO = "arango"
        MILVUS = "milvus"

    def __init__(self):
        # Store (config, manager) tuples where manager may be None if not yet created
        self._db_store: Dict[str, tuple[Dict, Optional[DatabaseManager]]] = {}

    def register_config(self, name: str, config: Dict):
        """Register a database configuration."""
        self._db_store[name] = (config.copy(), None)

    @staticmethod
    def build_connection_string(db_type: str, host: str, port: str, database: str,
                               user: str, password: str) -> str:
        """
        Build connection string for common database types.

        Args:
            db_type: Type of database (postgres, mysql, sqlite)
            host: Database host
            port: Database port
            database: Database name
            user: Database username
            password: Database password

        Returns:
            str: Formatted connection string
        """
        if db_type == DatabaseFactory.DatabaseType.POSTGRES:
            prefix = "postgresql"
            return f"{prefix}://{user}:{quote(password)}@{host}:{port}/{database}"
        elif db_type == DatabaseFactory.DatabaseType.MYSQL:
            prefix = "mysql+pymysql"
            return f"{prefix}://{user}:{quote(password)}@{host}:{port}/{database}"
        elif db_type == DatabaseFactory.DatabaseType.SQLITE:
            return f"sqlite:///{database}"
        else:
            raise ValueError(f"Connection string generation not supported for {db_type}")

    def create_manager(self, name: str, async_mode: bool = False) -> DatabaseManager:
        """Create a database manager instance."""
        if name not in self._db_store:
            raise KeyError(f"No configuration registered for '{name}'")

        config, manager = self._db_store[name]
        if manager is None:
            config = config.copy()
            db_type = config.pop('type', 'mongo')  # Default to mongo if type not specified

            if db_type == self.DatabaseType.POSTGRES:
                if async_mode:
                    raise ValueError("Async mode not supported for PostgreSQL")
                manager = PostgresDatabaseManagerImpl()
                manager.engine_info = config['engine_info']
                manager.database_type = 'postgres'
                try:
                    manager.connect(config.get('show_sql_query', False))
                except OperationalError as oe:
                    msg = str(oe).lower()
                    if auto_config.AUTO_CREATE_DATABASE and ('does not exist' in msg or ('database' in msg and 'does not exist' in msg)):
                        try:
                            # Parse URL and connect to default 'postgres' database to create the target DB
                            url = make_url(manager.engine_info)
                            target_db = url.database
                            admin_url = url.set(database='postgres')
                            logger.info(f"Database '{target_db}' does not exist. AUTO_CREATE_DATABASE enabled: attempting to create it on host {admin_url.host}.")
                            admin_engine = create_engine(str(admin_url))
                            with admin_engine.connect() as conn:
                                conn = conn.execution_options(isolation_level='AUTOCOMMIT')
                                conn.execute(text(f'CREATE DATABASE "{target_db}"'))
                            admin_engine.dispose()
                            # Retry manager.connect
                            manager.connect(config.get('show_sql_query', False))
                        except Exception as ex:
                            logger.error(f"Auto-create database failed: {ex}")
                            raise
                    else:
                        # Propagate the original error if auto-create is disabled or the error isn't 'db missing'
                        raise
            elif db_type == self.DatabaseType.SQLITE:
                if async_mode:
                    raise ValueError("Async mode not supported for SQLite")
                manager = PostgresDatabaseManagerImpl()  # Reusing PostgresManager for SQLite
                manager.engine_info = config['engine_info']
                manager.database_type = 'sqlite'
                manager.connect()
            elif db_type == self.DatabaseType.MYSQL:
                if async_mode:
                    raise ValueError("Async mode not supported for MySQL")
                manager = PostgresDatabaseManagerImpl()  # Reusing PostgresManager for MySQL
                manager.engine_info = config['engine_info']
                manager.database_type = 'mysql'
                manager.connect()
            elif db_type == self.DatabaseType.PDF:
                raise ValueError(" not supported for PDF manager")
            elif db_type == self.DatabaseType.MILVUS:
                from src.core.base.repository.milvus_repository import MilvusManager
                manager = MilvusManager(milvus_host=config.get('host', 'localhost'), milvus_port=config.get('port', 19530))
                manager.connect()
            else:  # mongo
                manager = AsyncMongoManager(**config) if async_mode else SyncMongoManager(**config)

            self._db_store[name] = (config, manager)

        return self._db_store[name][1]

    def close(self, name: str):
        """Close specific database manager."""
        if name in self._db_store:
            config, manager = self._db_store[name]
            if manager is not None:
                manager.close()
                self._db_store[name] = (config, None)
                del self._db_store[name]

    def close_all(self):
        """Close all database connections."""
        for name, (config, manager) in self._db_store.items():
            if manager is not None:
                manager.close()
                self._db_store[name] = (config, None)
        self._db_store = {}

