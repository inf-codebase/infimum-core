import asyncio
from contextlib import contextmanager
import functools
import inspect
from typing import Any, Callable, Dict, TypeVar, Optional, Union, List
from ..database import (
    DatabaseFactory,
    DatabaseManager,
    PostgresDatabaseManagerImpl,
    MilvusManager
)
from ..utils import auto_config
from loguru import logger

T = TypeVar('T')

class InjectionContainer:
    """
    Unified container for managing dependency injection, including database connections
    and general application dependencies.
    
    This container supports multiple named instances, allowing different scopes
    or contexts to have their own dependency registrations.
    
    Example:
        ```python
        # Create named container
        container = InjectionContainer.get_instance("my_app")
        
        # Use global container (default)
        container = InjectionContainer.get_instance()  # or use 'context'
        ```
    """
    
    _instances: Dict[str, 'InjectionContainer'] = {}
    
    def __init__(self, name: str = "default"):
        """Initialize injection container.
        
        Args:
            name: Name of the container instance (default: "default")
        """
        self.name = name
        self._dependencies: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._database_factory = DatabaseFactory()
        self._database_configs = {}
    
    @classmethod
    def get_instance(cls, name: str = "default") -> 'InjectionContainer':
        """Get or create a named container instance.
        
        This method implements a singleton pattern per name, ensuring that
        the same name always returns the same instance.
        
        Args:
            name: Container name (default: "default")
        
        Returns:
            InjectionContainer instance
        """
        if not hasattr(cls, '_instances'):
            cls._instances = {}
        
        if name not in cls._instances:
            cls._instances[name] = cls(name)
            logger.debug(f"Created new InjectionContainer instance: {name}")
        
        return cls._instances[name]
    
    @classmethod
    def clear_instance(cls, name: str) -> None:
        """Clear a named container instance.
        
        Args:
            name: Container name to clear
        """
        if hasattr(cls, '_instances') and name in cls._instances:
            del cls._instances[name]
            logger.debug(f"Cleared InjectionContainer instance: {name}")
    
    @classmethod
    def clear_all_instances(cls) -> None:
        """Clear all container instances."""
        if hasattr(cls, '_instances'):
            count = len(cls._instances)
            cls._instances.clear()
            logger.info(f"Cleared {count} InjectionContainer instance(s)")

    #
    # General dependency injection methods
    #
    def register(self, name: str, dependency: Any) -> None:
        """Register a dependency"""
        self._dependencies[name] = dependency

    def register_factory(self, name: str, factory: Callable) -> None:
        """Register a factory function for creating dependencies"""
        self._factories[name] = factory

    def get(self, name: str) -> Any:
        """Get a dependency by name"""
        if name in self._dependencies:
            return self._dependencies[name]
        if name in self._factories:
            return self._factories[name]()

        # Check if this is a registered database
        if name in self._database_configs:
            return self.get_database_manager(name)

        raise KeyError(f"Dependency '{name}' not found")

    def inject(self, *dependencies: str) -> Callable:
        """Decorator for injecting dependencies into a function"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Get dependencies from container
                deps = {dep: self.get(dep) for dep in dependencies}
                # Create partial function with dependencies
                partial_func = functools.partial(func, **deps)
                # Call the function with remaining args and kwargs
                return partial_func(*args, **kwargs)
            return wrapper
        return decorator

    def inject_class(self, *dependencies: str) -> Callable:
        """Decorator for injecting dependencies into a class"""
        def decorator(cls: type) -> type:
            # Get dependencies from container
            deps = {dep: self.get(dep) for dep in dependencies}
            # Create a new class with injected dependencies
            class InjectedClass(cls):
                def __init__(self, *args, **kwargs):
                    # Merge dependencies with kwargs
                    kwargs.update(deps)
                    super().__init__(*args, **kwargs)

            # Preserve original class name and module
            InjectedClass.__name__ = cls.__name__
            InjectedClass.__module__ = cls.__module__
            return InjectedClass
        return decorator

    #
    # Database context management methods
    #
    def _inject_from_config(self, func: Callable, config_prefix: str = None):
        """
        Helper for injecting database configuration values from auto_config when not provided.

        Args:
            func: The function to wrap
            config_prefix: The prefix used in auto_config for this database type (e.g., 'DB_', 'MONGO_')
        """
        def wrapper(name: str, **kwargs):
            # Determine the config prefix if not specified
            prefix = config_prefix
            if prefix is None:
                # Extract database type from function name
                db_type = func.__name__
                prefix = db_type
                logger.warning(f"Using function name:{prefix} as prefix for searching in auto_config")

            # Check each potential parameter that could be injected
            # Extract parameter names from the function signature
            param_mappings = {}
            signature = inspect.signature(func)
            for param_name in signature.parameters:
                # Skip 'self' and 'name' parameters which are handled separately
                if param_name not in ['self', 'name']:
                    # Convert parameter name to uppercase for config key
                    config_key = f'{prefix}{param_name.upper()}'
                    param_mappings[param_name] = config_key

            # Inject parameters from config when not explicitly provided
            injected_kwargs = kwargs.copy()
            for param, config_key in param_mappings.items():
                if param not in injected_kwargs or injected_kwargs[param] is None:
                    if hasattr(auto_config, config_key):
                        injected_kwargs[param] = getattr(auto_config, config_key)

            return func(name, **injected_kwargs)
        return wrapper

    def register_mongo(self, name: str = None, uri: str = None, db_name: str = None) -> None:
        """
        Register a MongoDB configuration

        Args:
            name: Name to register the database as
            uri: MongoDB connection URI (if None, will try to get from MONGO_URI in auto_config)
            db_name: MongoDB database name (if None, will try to get from MONGO_DATABASE in auto_config)
        """
        register_func = self._inject_from_config(self._register_mongo, 'MONGO_')
        register_func(name, uri=uri, db_name=db_name)

    def _register_mongo(self, name: str = None, uri: str = None, db_name: str = None) -> None:
        """Internal implementation of MongoDB registration"""
        if uri is None:
            raise ValueError("MongoDB URI must be provided or available in auto_config.MONGO_URI")
        if db_name is None:
            raise ValueError("MongoDB database name must be provided or available in auto_config.MONGO_DATABASE")

        config = {
            "connection_string": uri,
            "database": db_name,
            "type": DatabaseFactory.DatabaseType.MONGO
        }
        self._database_configs[name] = config
        self._database_factory.register_config(name, config)

    def register_postgres(self, name: str = None, host: str = None, port: str = None, database: str = None,
                         user: str = None, password: str = None) -> None:
        """
        Register a PostgreSQL configuration

        Args:
            name: Name to register the database as
            host: Database host (if None, will try to get from POSTGRES_HOST in auto_config)
            port: Database port (if None, will try to get from POSTGRES_PORT in auto_config)
            database: Database name (if None, will try to get from POSTGRES_DATABASE in auto_config)
            user: Database username (if None, will try to get from POSTGRES_USER in auto_config)
            password: Database password (if None, will try to get from POSTGRES_PASSWORD in auto_config)
        """
        register_func = self._inject_from_config(self._register_postgres, 'POSTGRES_')
        register_func(name, host=host, port=port, database=database, user=user, password=password)

    def _register_postgres(self, name: str = None, host: str = None, port: str = None, database: str = None,
                          user: str = None, password: str = None) -> None:
        """Internal implementation of PostgreSQL registration"""
        connection_string = DatabaseFactory.build_connection_string(
                DatabaseFactory.DatabaseType.POSTGRES, host, port, database, user, password
        )
        config = {
            "engine_info": connection_string,
            "database_type": "postgres",
            "type": DatabaseFactory.DatabaseType.POSTGRES
        }
        self._database_configs[name] = config
        self._database_factory.register_config(name, config)

    def register_sqlite(self, name: str = None, database_path: str = None) -> None:
        """
        Register a SQLite database configuration

        Args:
            name: Name to register the database as
            database_path: Path to SQLite database file (if None, will try to get from SQLITE_PATH in auto_config)
        """
        register_func = self._inject_from_config(self._register_sqlite, 'SQLITE_')
        register_func(name, database_path=database_path)

    def _register_sqlite(self, name: str = None, database_path: str = None) -> None:
        """Internal implementation of SQLite registration"""
        if database_path is None:
            raise ValueError("SQLite database path must be provided or available in auto_config.SQLITE_PATH")

        connection_string = f"sqlite:///{database_path}"

        config = {
            "engine_info": connection_string,
            "database_type": "sqlite",
            "type": DatabaseFactory.DatabaseType.SQLITE
        }
        self._database_configs[name] = config
        self._database_factory.register_config(name, config)

    def register_mysql(self, name: str = None, connection_string: str = None,
                      host: str = None, port: str = None, database: str = None,
                      user: str = None, password: str = None) -> None:
        """
        Register a MySQL configuration

        Args:
            name: Name to register the database as
            connection_string: Full MySQL connection string (optional)
            host: Database host (if None, will try to get from MYSQL_HOST in auto_config)
            port: Database port (if None, will try to get from MYSQL_PORT in auto_config)
            database: Database name (if None, will try to get from MYSQL_DATABASE in auto_config)
            user: Database username (if None, will try to get from MYSQL_USER in auto_config)
            password: Database password (if None, will try to get from MYSQL_PASSWORD in auto_config)
        """
        register_func = self._inject_from_config(self._register_mysql, 'MYSQL_')
        register_func(name, connection_string=connection_string, host=host, port=port,
                     database=database, user=user, password=password)

    def _register_mysql(self, name: str = None, connection_string: str = None,
                       host: str = None, port: str = None, database: str = None,
                       user: str = None, password: str = None) -> None:
        """Internal implementation of MySQL registration"""
        if connection_string is None and all([host, port, database, user, password]):
            connection_string = DatabaseFactory.build_connection_string(
                DatabaseFactory.DatabaseType.MYSQL, host, port, database, user, password
            )
        elif connection_string is None:
            raise ValueError("Either connection_string or all connection parameters must be provided in args or auto_config")

        config = {
            "engine_info": connection_string,
            "database_type": "mysql",
            "type": DatabaseFactory.DatabaseType.MYSQL
        }
        self._database_configs[name] = config
        self._database_factory.register_config(name, config)

    def register_pdf(self, name: str = None, **kwargs) -> None:
        """
        Register a PDF Vector Store configuration

        Args:
            name: Name to register the database as
            chunk_size: Size of text chunks (if None, will try to get from PDF_CHUNK_SIZE in auto_config)
            chunk_overlap: Overlap between chunks (if None, will try to get from PDF_CHUNK_OVERLAP in auto_config)
            max_workers: Maximum worker threads (if None, will try to get from PDF_MAX_WORKERS in auto_config)
            batch_size: Batch processing size (if None, will try to get from PDF_BATCH_SIZE in auto_config)
        """
        register_func = self._inject_from_config(self._register_pdf, 'PDF_')
        register_func(name, **kwargs)

    def _register_pdf(self, name: str = None, **kwargs) -> None:
        """Internal implementation of PDF Vector Store registration"""
        config = {
            "type": DatabaseFactory.DatabaseType.PDF,
            "chunk_size": kwargs.get("chunk_size", 1000),
            "chunk_overlap": kwargs.get("chunk_overlap", 200),
            "max_workers": kwargs.get("max_workers", 4),
            "batch_size": kwargs.get("batch_size", 10)
        }
        self._database_configs[name] = config
        self._database_factory.register_config(name, config)

    def register_milvus(self, name: str = None, host: str = None, port: str = None) -> None:
        """
        Register a Milvus configuration

        Args:
            name: Name to register the database as
            host: Milvus host (if None, will try to get from MILVUS_HOST in auto_config)
            port: Milvus port (if None, will try to get from MILVUS_PORT in auto_config)
        """
        register_func = self._inject_from_config(self._register_milvus, 'MILVUS_')
        register_func(name, host=host, port=port)

    def _register_milvus(self, name: str = None, host: str = None, port: str = None) -> None:
        """Internal implementation of Milvus registration"""
        config = {
            "type": DatabaseFactory.DatabaseType.MILVUS,
            "host": host,
            "port": port
        }
        self._database_configs[name] = config
        self._database_factory.register_config(name, config)

    def get_database_manager(self, name: str = None, async_mode: bool = False) -> DatabaseManager:
        """Get a database manager by name"""
        if name not in self._database_configs:
            logger.warning(f"Database '{name}' not registered yet. Attempting auto-registration.")
            # Try auto-registering the database
            success = self._auto_register_database(name)
            if not success:
                raise KeyError(f"No database registered with name '{name}'")

        return self._database_factory.create_manager(name, async_mode)

    def get_database_config(self, name: str) -> Dict[str, Any]:
        """Get database configuration by name"""
        if name not in self._database_configs:
            raise KeyError(f"No database config found with name '{name}'")
        return self._database_configs[name]

    def close_database(self, name: str) -> None:
        """Close specific database manager"""
        self._database_factory.close(name)

    def close_all_databases(self) -> None:
        """Close all database connections"""
        self._database_factory.close_all()

    @classmethod
    def _auto_register_database(cls, db_name_or_prefix: str) -> bool:
        """
        Automatically register a database based on a name or prefix.

        Supports multiple patterns:
        1. Simple prefix: POSTGRES_ -> POSTGRES_HOST, POSTGRES_PORT, etc.
        2. Prefix with DB: PRODUCTION_POSTGRES_DB_ -> PRODUCTION_POSTGRES_DB_HOST, etc.
        3. MongoDB patterns: MONGO_BE_DB -> MONGO_BE_DB_URI, MONGO_BE_DB_NAME
        4. Milvus patterns: MILVUS_ -> MILVUS_HOST, MILVUS_PORT

        Args:
            db_name_or_prefix: Database name or prefix to search for in config

        Returns:
            bool: True if registration was successful, False otherwise
        """

        logger.info(f"Attempting to auto-register database for '{db_name_or_prefix}'")

        # Get all available config keys
        config_keys = set()
        if hasattr(auto_config.config.config.repository, 'data'):
            config_keys = set(auto_config.config.config.repository.data.keys())

        # Helper function to safely get config value
        def get_config_value(key: str, default=None):
            return auto_config.config.config.repository.data.get(key, default) if config_keys else default

        # Try to find database type and extract prefix patterns
        prefix_patterns = cls._extract_prefix_patterns(db_name_or_prefix, config_keys)

        for pattern_info in prefix_patterns:
            prefix = pattern_info['prefix']
            db_type = pattern_info['type']
            param_pattern = pattern_info['param_pattern']

            logger.info(f"Trying to register {db_type} database using prefix '{prefix}' with pattern '{param_pattern}'")

            try:
                if db_type == 'postgres':
                    host = get_config_value(f"{prefix}{param_pattern}HOST", "localhost")
                    port = get_config_value(f"{prefix}{param_pattern}PORT", "5432")
                    database = get_config_value(f"{prefix}{param_pattern}DATABASE", "postgres")
                    user = get_config_value(f"{prefix}{param_pattern}USER", "postgres")
                    password = get_config_value(f"{prefix}{param_pattern}PASSWORD", "")

                    context.register_postgres(db_name_or_prefix, host=host, port=port,
                                        database=database, user=user, password=password)
                    logger.info(f"Auto-registered PostgreSQL database '{db_name_or_prefix}'")
                    return True

                elif db_type == 'mysql':
                    host = get_config_value(f"{prefix}{param_pattern}HOST", "localhost")
                    port = get_config_value(f"{prefix}{param_pattern}PORT", "3306")
                    database = get_config_value(f"{prefix}{param_pattern}DATABASE", prefix.lower())
                    user = get_config_value(f"{prefix}{param_pattern}USER", "root")
                    password = get_config_value(f"{prefix}{param_pattern}PASSWORD", "")

                    context.register_mysql(db_name_or_prefix, host=host, port=port,
                                    database=database, user=user, password=password)
                    logger.info(f"Auto-registered MySQL database '{db_name_or_prefix}'")
                    return True

                elif db_type == 'sqlite':
                    path = get_config_value(f"{prefix}{param_pattern}PATH", f"{prefix.lower()}.db")
                    context.register_sqlite(db_name_or_prefix, database_path=path)
                    logger.info(f"Auto-registered SQLite database '{db_name_or_prefix}'")
                    return True

                elif db_type == 'mongo':
                    uri = get_config_value(f"{prefix}{param_pattern}URI")
                    db_name = get_config_value(f"{prefix}{param_pattern}NAME") or get_config_value(f"{prefix}{param_pattern}DATABASE")

                    if uri and db_name:
                        context.register_mongo(db_name_or_prefix, uri=uri, db_name=db_name)
                        logger.info(f"Auto-registered MongoDB database '{db_name_or_prefix}'")
                        return True

                elif db_type == 'milvus':
                    host = get_config_value(f"{prefix}{param_pattern}HOST", "localhost")
                    port = get_config_value(f"{prefix}{param_pattern}PORT", "19530")

                    context.register_milvus(db_name_or_prefix, host=host, port=port)
                    logger.info(f"Auto-registered Milvus database '{db_name_or_prefix}'")
                    return True

            except Exception as e:
                logger.error(f"Error registering {db_type} database '{db_name_or_prefix}': {str(e)}")
                continue

        logger.error(f"No database found matching pattern '{db_name_or_prefix}'. Please check your configuration.")
        return False

    @classmethod
    def _extract_prefix_patterns(cls, db_name_or_prefix: str, config_keys: set) -> list:
        """
        Extract possible prefix patterns from the db_name_or_prefix and available config keys.

        Returns a list of pattern info dicts with 'prefix', 'type', and 'param_pattern'.
        """
        patterns = []

        # Normalize input
        normalized_input = db_name_or_prefix.upper().replace('-', '_').rstrip('_')

        # Pattern 1: Direct prefix match (e.g., POSTGRES_ -> POSTGRES_HOST)
        for key in config_keys:
            if key.startswith(normalized_input) and key.endswith('_TYPE'):
                db_type = auto_config.config.config.repository.data.get(key, '').lower()
                if db_type in ['postgres', 'mysql', 'sqlite', 'mongo', 'milvus']:
                    patterns.append({
                        'prefix': normalized_input,
                        'type': db_type,
                        'param_pattern': '_'
                    })

        # Pattern 2: Find keys that contain our input as prefix (e.g., PRODUCTION_POSTGRES_DB_TYPE)
        for key in config_keys:
            if key.endswith('_TYPE') and normalized_input in key:
                db_type = auto_config.config.config.repository.data.get(key, '').lower()
                if db_type in ['postgres', 'mysql', 'sqlite', 'mongo', 'milvus']:
                    # Extract the full prefix (everything before _TYPE)
                    full_prefix = key[:-5]  # Remove '_TYPE'
                    patterns.append({
                        'prefix': full_prefix,
                        'type': db_type,
                        'param_pattern': '_'
                    })

        # Pattern 3: MongoDB special pattern (MONGO_*_DB_* or MONGO_*_*)
        if 'mongo' in normalized_input.lower():
            for key in config_keys:
                if key.startswith('MONGO_') and ('_URI' in key or '_NAME' in key):
                    # Extract prefix pattern (everything before _URI or _NAME)
                    if key.endswith('_URI'):
                        prefix = key[:-4]
                    elif key.endswith('_NAME'):
                        prefix = key[:-5]
                    else:
                        continue

                    # Check if we have both URI and NAME with this prefix
                    uri_key = f"{prefix}_URI"
                    name_key = f"{prefix}_NAME"

                    if uri_key in config_keys and name_key in config_keys:
                        # Check if this prefix matches our input
                        if (normalized_input in prefix or prefix in normalized_input or
                            normalized_input.replace('_', '') in prefix.replace('_', '')):
                            patterns.append({
                                'prefix': prefix,
                                'type': 'mongo',
                                'param_pattern': '_'
                            })

        # Pattern 4: Standard patterns without explicit TYPE (fallback for common names)
        common_db_patterns = {
            'POSTGRES': 'postgres',
            'MYSQL': 'mysql',
            'SQLITE': 'sqlite',
            'MILVUS': 'milvus'
        }

        for pattern_name, db_type in common_db_patterns.items():
            if pattern_name in normalized_input:
                # Check if we have config keys for this pattern
                test_keys = [f"{pattern_name}_HOST", f"{normalized_input}_HOST", f"{normalized_input}HOST"]
                for test_key in test_keys:
                    if test_key in config_keys:
                        prefix_to_use = test_key.replace('_HOST', '').replace('HOST', '')
                        patterns.append({
                            'prefix': prefix_to_use,
                            'type': db_type,
                            'param_pattern': '_' if test_key.endswith('_HOST') else ''
                        })
                        break

        # Remove duplicates and sort by specificity (longer prefixes first)
        unique_patterns = []
        seen = set()
        for pattern in sorted(patterns, key=lambda x: len(x['prefix']), reverse=True):
            key = (pattern['prefix'], pattern['type'], pattern['param_pattern'])
            if key not in seen:
                seen.add(key)
                unique_patterns.append(pattern)

        logger.debug(f"Extracted patterns for '{db_name_or_prefix}': {unique_patterns}")
        return unique_patterns

# Create global instance of InjectionContainer for easy access (backward compatibility)
# This uses the "default" instance
context = InjectionContainer.get_instance("default")

# Track current context for scoped operations
context._current = context

# Helper functions that use the current context (defaults to global context)
def inject(*dependencies: str) -> Callable:
    """Global inject decorator using the current context"""
    current = getattr(context, '_current', context)
    return current.inject(*dependencies)

def inject_class(*dependencies: str) -> Callable:
    """Global inject_class decorator using the current context"""
    current = getattr(context, '_current', context)
    return current.inject_class(*dependencies)

def register(name: str, dependency: Any) -> None:
    """Global register function using the current context"""
    current = getattr(context, '_current', context)
    current.register(name, dependency)

def register_factory(name: str, factory: Callable) -> None:
    """Global register_factory function using the current context"""
    current = getattr(context, '_current', context)
    current.register_factory(name, factory)

def get(name: str) -> Any:
    """Global get function using the current context"""
    current = getattr(context, '_current', context)
    return current.get(name)

def inject_database(db_name: str = None, async_mode: bool = False):
    """Global inject_database decorator using the container"""
    return context.inject_database(db_name, async_mode)

def register_mongo(name: str, **kwargs) -> None:
    """Global register_mongo function using the container"""
    context.register_mongo(name, **kwargs)

def register_postgres(name: str, **kwargs) -> None:
    """Global register_postgres function using the container"""
    context.register_postgres(name, **kwargs)

def register_sqlite(name: str, **kwargs) -> None:
    """Global register_sqlite function using the container"""
    context.register_sqlite(name, **kwargs)

def register_mysql(name: str, **kwargs) -> None:
    """Global register_mysql function using the container"""
    context.register_mysql(name, **kwargs)

def register_pdf(name: str, **kwargs) -> None:
    """Global register_pdf function using the container"""
    context.register_pdf(name, **kwargs)

def register_milvus(name: str, **kwargs) -> None:
    """Global register_milvus function using the container"""
    context.register_milvus(name, **kwargs)

def get_database_manager(name: str, async_mode: bool = False) -> DatabaseManager:
    """Global get_database_manager function using the container"""
    return context.get_database_manager(name, async_mode)

@contextmanager
def database_session(db_name: str, async_mode: bool = False):
    """Global database_session context manager using the current context"""
    current = getattr(context, '_current', context)
    if hasattr(current, 'database_session'):
        with current.database_session(db_name, async_mode) as manager:
            yield manager
    else:
        # Fallback: use get_database_manager
        manager = current.get_database_manager(db_name, async_mode)
        try:
            yield manager
        finally:
            # Cleanup if needed
            pass

def with_database(db_name_or_prefix: str, async_mode: bool = False) -> DatabaseManager:
        """
        Helper function to provide database connection in dependency injection.

        Args:
            db_name_or_prefix: Name or prefix of the registered database to connect to
            async_mode: Whether to use async database connection

        Returns:
            A database manager instance for the specified database

        Example:
            def __init__(self, db = with_database("my_database")):
                self.db = db.session  # For SQL databases

            def __init__(self, milvus_db = with_database("MILVUS_")):
                self.milvus_db = milvus_db  # For Milvus database
        """
        try:
            db_manager = context.get_database_manager(db_name_or_prefix, async_mode)

            # For SQL databases, we want to ensure connection and return the session
            if isinstance(db_manager, PostgresDatabaseManagerImpl) and not db_manager.is_connected():
                db_manager.connect()

            # For Milvus, we want to ensure connection
            if isinstance(db_manager, MilvusManager) and not db_manager.is_connected():
                db_manager.connect()

            return db_manager
        except KeyError:
            logger.warning(f"Database '{db_name_or_prefix}' not registered yet. Attempting auto-registration.")
            # Try auto-registering databases
            context._auto_register_database(db_name_or_prefix)

            # Try again after auto-registration
            try:
                db_manager = context.get_database_manager(db_name_or_prefix, async_mode)

                # For SQL databases, we want to ensure connection and return the session
                if isinstance(db_manager, PostgresDatabaseManagerImpl) and not db_manager.is_connected():
                    db_manager.connect()

                # For Milvus, we want to ensure connection
                if isinstance(db_manager, MilvusManager) and not db_manager.is_connected():
                    db_manager.connect()

                return db_manager
            except KeyError:
                logger.error(f"Failed to auto-register database '{db_name_or_prefix}'. Please check your configuration.")
                return None


@contextmanager
def container_scope(container_name: str = "default"):
    """Context manager for scoped dependency injection.
    
    This allows temporarily switching to a different container instance
    within a context, useful for testing or isolated execution contexts.
    
    Args:
        container_name: Name of the container to use in this scope
    
    Example:
        ```python
        with container_scope("test"):
            # All DI operations in this block use the "test" container
            register("test_db", test_database)
            db = get("test_db")
        # Back to default container
        ```
    """
    old_context = getattr(context, '_current', context)
    new_context = InjectionContainer.get_instance(container_name)
    context._current = new_context
    try:
        yield new_context
    finally:
        context._current = old_context or context


# Example usage for clarity
if __name__ == "__main__":
    # Register a database
    register_postgres("users", host="localhost", database="users")

    # Register dependencies
    register('logger', print)
    register('config', {'debug': True})

    # Example 1: Function with database injection
    @inject_database("users")
    def get_user(db_manager, user_id):
        return db_manager.execute_query(f"SELECT * FROM users WHERE id = {user_id}")

    # Example 2: Function with dependency injection
    @inject('logger', 'config')
    def process_data(data, logger=None, config=None):
        logger(f"Processing data with config: {config}")
        return data * 2

    # Example 3: Class with database injection
    @inject_database("users")
    class UserService:
        def __init__(self, data_manager, name="default"):
            self.db = data_manager
            self.name = name

        def get_user(self, user_id):
            return self.db.execute_query(f"SELECT * FROM users WHERE id = {user_id}")

    # Example 4: Class with dependency injection
    @inject_class('logger', 'config')
    class DataProcessor:
        def __init__(self, logger=None, config=None):
            self.logger = logger
            self.config = config

        def process(self, data):
            self.logger(f"Processing with {self.config}")
            return data
