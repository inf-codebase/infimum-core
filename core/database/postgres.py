from sqlalchemy import create_engine, text
import pandas as pd
from typing import Any, Collection, Dict, List, Optional, Union, Type
from core.base.entity import Document
from core.utils import auto_config
from sqlalchemy.orm import sessionmaker
from abc import abstractmethod
import numpy as np

from contextlib import contextmanager
from datetime import datetime
from typing import Optional, List, Dict, Any
from loguru import logger
from typing import List
from urllib.parse import quote
from sqlalchemy.exc import OperationalError
from sqlalchemy.engine import make_url
from core.database.base import DatabaseManager
from core.database.interfaces import RelationalDatabaseManager, DocumentDatabaseManager


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
        from core.base.entity import BaseEntity

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

class PostgresDatabaseManagerImpl(SQLManager, RelationalDatabaseManager):
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


class DatabaseFactory:
    """Factory class for creating database managers.
    
    This factory uses the DatabaseBackendRegistry to create database managers,
    enabling a plugin-based architecture. It maintains backward compatibility
    with the old configuration-based approach.
    """

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
        
        # Initialize registry and register built-in backends
        from core.database.registry import DatabaseBackendRegistry
        from core.database.plugins import register_builtin_backends, discover_database_plugins
        
        self._registry = DatabaseBackendRegistry
        
        # Register built-in backends
        register_builtin_backends()
        
        # Try to discover plugins (non-blocking if package doesn't exist)
        try:
            discover_database_plugins()
        except Exception as e:
            logger.debug(f"Plugin discovery failed (non-critical): {e}")

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
        """Create a database manager instance using the registry.
        
        This method uses the DatabaseBackendRegistry to create managers,
        with fallback to legacy creation logic for backward compatibility.
        
        Args:
            name: Name of the registered database configuration
            async_mode: Whether to use async manager (if available)
        
        Returns:
            DatabaseManager instance
        
        Raises:
            KeyError: If no configuration is registered for the name
            ValueError: If backend creation fails
        """
        if name not in self._db_store:
            raise KeyError(f"No configuration registered for '{name}'")

        config_dict, manager = self._db_store[name]
        if manager is None:
            config_dict = config_dict.copy()
            db_type = config_dict.get('type', 'mongo')  # Default to mongo if type not specified
            
            # Handle async mode
            if async_mode and db_type in ['milvus', 'qdrant', 'mongo']:
                db_type = f"{db_type}_async"
            
            # Try to create using registry (new approach)
            try:
                from core.database.base import DatabaseConnectionConfig
                
                # Convert config dict to DatabaseConnectionConfig
                # Preserve all fields for backward compatibility
                config = DatabaseConnectionConfig(**config_dict)
                
                # Try registry-based creation
                manager = self._registry.create(db_type, config, async_mode=async_mode)
                
                # For special cases that need additional setup, handle them
                if db_type == self.DatabaseType.POSTGRES or db_type == self.DatabaseType.SQLITE or db_type == self.DatabaseType.MYSQL:
                    # SQL databases need engine_info and connection setup
                    if isinstance(manager, PostgresDatabaseManagerImpl):
                        manager.engine_info = config_dict.get('engine_info')
                        manager.database_type = db_type
                        try:
                            manager.connect(config_dict.get('show_sql_query', False))
                        except OperationalError as oe:
                            # Handle auto-create database for PostgreSQL
                            msg = str(oe).lower()
                            if auto_config.AUTO_CREATE_DATABASE and ('does not exist' in msg or ('database' in msg and 'does not exist' in msg)):
                                try:
                                    url = make_url(manager.engine_info)
                                    target_db = url.database
                                    admin_url = url.set(database='postgres')
                                    logger.info(f"Database '{target_db}' does not exist. AUTO_CREATE_DATABASE enabled: attempting to create it on host {admin_url.host}.")
                                    admin_engine = create_engine(str(admin_url))
                                    with admin_engine.connect() as conn:
                                        conn = conn.execution_options(isolation_level='AUTOCOMMIT')
                                        conn.execute(text(f'CREATE DATABASE "{target_db}"'))
                                    admin_engine.dispose()
                                    manager.connect(config_dict.get('show_sql_query', False))
                                except Exception as ex:
                                    logger.error(f"Auto-create database failed: {ex}")
                                    raise
                            else:
                                raise
                
                self._db_store[name] = (config_dict, manager)
                return manager
                
            except (TypeError, ValueError) as e:
                # Fallback to legacy creation if registry approach fails
                logger.debug(f"Registry creation failed for {db_type}, falling back to legacy: {e}")
                manager = self._create_manager_legacy(db_type, config_dict, async_mode)
                self._db_store[name] = (config_dict, manager)
                return manager

        return self._db_store[name][1]
    
    def _create_manager_legacy(self, db_type: str, config_dict: Dict, async_mode: bool) -> DatabaseManager:
        """Legacy manager creation (for backward compatibility).
        
        This method contains the original creation logic as a fallback
        when registry-based creation fails.
        
        Args:
            db_type: Database type
            config_dict: Configuration dictionary
            async_mode: Whether to use async manager
        
        Returns:
            DatabaseManager instance
        """
        config = config_dict.copy()
        
        config = config_dict.copy()
        
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
            raise ValueError("PDF manager not supported")
        elif db_type == self.DatabaseType.MILVUS:
            from core.database.milvus import MilvusManager
            manager = MilvusManager(milvus_host=config_dict.get('host', 'localhost'), milvus_port=config_dict.get('port', 19530))
            manager.connect()
        else:  # mongo
            from core.database.mongo import SyncMongoManager, AsyncMongoManager
            manager = AsyncMongoManager(**config_dict) if async_mode else SyncMongoManager(**config_dict)

        return manager

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

