"""
Plugin discovery system for database backends.

This module provides functionality to auto-discover and register database
backend plugins from a package. This enables a plugin architecture where
third-party backends can be automatically discovered.
"""

import importlib
import pkgutil
from typing import Optional
from loguru import logger

from .registry import DatabaseBackendRegistry


def discover_database_plugins(package_name: str = "core.database.plugins") -> int:
    """Auto-discover database backend plugins from a package.
    
    This function searches for modules in the specified package that have
    a `register_backend` function and calls it to register the backend.
    
    Args:
        package_name: Name of the package to search for plugins (default: "core.database.plugins")
    
    Returns:
        Number of plugins discovered and registered
    
    Example:
        A plugin module should look like:
        ```python
        # core/database/plugins/my_backend.py
        from core.database.registry import DatabaseBackendRegistry
        from core.database.interfaces import VectorDatabaseManager
        
        class MyBackendManager(VectorDatabaseManager):
            # Implementation
            pass
        
        def register_backend():
            DatabaseBackendRegistry.register("my_backend", MyBackendManager)
        ```
    """
    discovered_count = 0
    
    try:
        package = importlib.import_module(package_name)
        
        # Check if package has a __path__ attribute (it's a package)
        if not hasattr(package, '__path__'):
            logger.warning(f"{package_name} is not a package, skipping plugin discovery")
            return 0
        
        # Walk through all modules in the package
        for _, name, is_pkg in pkgutil.iter_modules(package.__path__):
            if is_pkg:
                # Skip sub-packages for now (could be extended to recurse)
                continue
            
            try:
                # Import the module
                full_module_name = f"{package_name}.{name}"
                module = importlib.import_module(full_module_name)
                
                # Check if module has register_backend function
                if hasattr(module, 'register_backend'):
                    register_func = getattr(module, 'register_backend')
                    if callable(register_func):
                        # Call the registration function
                        register_func()
                        discovered_count += 1
                        logger.info(f"Discovered and registered plugin: {name}")
                    else:
                        logger.warning(f"register_backend in {full_module_name} is not callable")
                else:
                    logger.debug(f"Module {full_module_name} does not have register_backend function")
            
            except ImportError as e:
                logger.warning(f"Failed to import plugin module {name}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error registering plugin {name}: {e}", exc_info=True)
                continue
        
        if discovered_count > 0:
            logger.info(f"Discovered {discovered_count} database plugin(s) from {package_name}")
        else:
            logger.debug(f"No plugins found in {package_name}")
    
    except ImportError:
        # Package doesn't exist - this is fine, just log it
        logger.debug(f"Plugin package {package_name} does not exist, skipping discovery")
    except Exception as e:
        logger.error(f"Error discovering plugins from {package_name}: {e}", exc_info=True)
    
    return discovered_count


def register_builtin_backends() -> None:
    """Register all built-in database backends.
    
    This function registers the standard database backends that come with
    the core module. It's called automatically when DatabaseFactory is initialized.
    """
    from core.database.postgres import PostgresDatabaseManagerImpl
    from core.database.milvus import MilvusManager, AsyncMilvusManager
    from core.database.qdrant import QdrantManager, AsyncQdrantManager
    from core.database.postgres import SyncMongoManager, AsyncMongoManager
    
    # Register sync backends
    DatabaseBackendRegistry.register("postgres", PostgresDatabaseManagerImpl)
    DatabaseBackendRegistry.register("milvus", MilvusManager)
    DatabaseBackendRegistry.register("qdrant", QdrantManager)
    DatabaseBackendRegistry.register("mongo", SyncMongoManager)
    
    # Register async backends with suffix
    DatabaseBackendRegistry.register("milvus_async", AsyncMilvusManager)
    DatabaseBackendRegistry.register("qdrant_async", AsyncQdrantManager)
    DatabaseBackendRegistry.register("mongo_async", AsyncMongoManager)
    
    logger.info("Registered built-in database backends")
