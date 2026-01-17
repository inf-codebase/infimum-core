from core.database import DatabaseFactory
from core.engine.context import context
from fastapi import FastAPI
from core.engine.decorators import func_decorator
from loguru import logger
from core.utils.auto_config import get_config_by_prefixes
from core.base.registry import EntityRegistry  # Import the registry
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select

class Startup:
    @staticmethod
    def include_service(app: FastAPI, service_type_or_router):
        router = service_type_or_router
        app.include_router(router)
        return func_decorator

    @staticmethod
    def initialize(create_tables=True, discover_entities=True, entities_package='entities'):
        try:
            # Discover entities if requested
            if discover_entities:
                EntityRegistry.discover_entities(entities_package)

            database_configs = get_config_by_prefixes(["POSTGRES_DATABASE", "MONGO_DATABASE", "SQLITE_DATABASE", "MYSQL_DATABASE", "PDF_DATABASE"])
            logger.info(f"Database configs: {database_configs}")
            for config_key, database_name in database_configs.items():
                # Register the database based on its type
                if config_key.startswith("POSTGRES_DATABASE"):
                    context.register_postgres(database_name)
                    logger.info(f"Registered database: {database_name}")

                    # Create tables if requested and it's a SQL database
                    if create_tables:
                        db_manager = context.get_database_manager(database_name)
                        if hasattr(db_manager, 'create_tables'):
                            success = EntityRegistry.create_tables(db_manager)
                            if success:
                                logger.info(f"Created tables for database: {database_name}")
                            else:
                                logger.error(f"Failed to create tables for database: {database_name}")

                elif config_key.startswith("SQLITE_DATABASE"):
                    context.register_sqlite(database_name)
                    logger.info(f"Registered database: {database_name}")

                    # Create tables if requested and it's a SQL database
                    if create_tables:
                        db_manager = context.get_database_manager(database_name)
                        if hasattr(db_manager, 'create_tables'):
                            success = EntityRegistry.create_tables(db_manager)
                            if success:
                                logger.info(f"Created tables for database: {database_name}")
                            else:
                                logger.error(f"Failed to create tables for database: {database_name}")

                elif config_key.startswith("MYSQL_DATABASE"):
                    context.register_mysql(database_name)
                    logger.info(f"Registered database: {database_name}")

                    # Create tables if requested and it's a SQL database
                    if create_tables:
                        db_manager = context.get_database_manager(database_name)
                        if hasattr(db_manager, 'create_tables'):
                            success = EntityRegistry.create_tables(db_manager)
                            if success:
                                logger.info(f"Created tables for database: {database_name}")
                            else:
                                logger.error(f"Failed to create tables for database: {database_name}")

                elif config_key.startswith("MONGO_DATABASE"):
                    context.register_mongo(database_name)
                    logger.info(f"Registered database: {database_name}")

                elif config_key.startswith("PDF_DATABASE"):
                    context.register_pdf(database_name)
                    logger.info(f"Registered database: {database_name}")

                else:
                    logger.warning(f"Unknown database type for config: {config_key}")
                    continue

        except Exception as e:
            logger.error(f"Error initializing database tables: {str(e)}")
        return func_decorator

    @staticmethod
    def create_tables_for_db(database_name, drop_first=False):
        """Create tables for a specific database"""
        try:
            db_manager = context.get_database_manager(database_name)
            if hasattr(db_manager, 'create_tables'):
                success = EntityRegistry.create_tables(db_manager, drop_first)
                if success:
                    logger.info(f"Created tables for database: {database_name}")
                else:
                    logger.error(f"Failed to create tables for database: {database_name}")
            else:
                logger.warning(f"Database {database_name} doesn't support table creation")
        except Exception as e:
            logger.error(f"Error creating tables for {database_name}: {str(e)}")
