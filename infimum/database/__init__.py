from .base import DatabaseManager
from .postgres import (
    DatabaseFactory,
    PostgresDatabaseManagerImpl,
    SQLManager,
)
try:
    from .mongo import SyncMongoManager, AsyncMongoManager, MongoManagerBase
except ImportError as e:
    # Create placeholder classes that provide helpful error messages
    class SyncMongoManager:
        """Placeholder for SyncMongoManager when pymongo is not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "SyncMongoManager requires 'pymongo' to be installed. "
                "Install it with: pip install -e '.[mongo]' or pip install pymongo>=4.16.0"
            ) from e
    
    class AsyncMongoManager:
        """Placeholder for AsyncMongoManager when motor is not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "AsyncMongoManager requires 'motor' to be installed. "
                "Install it with: pip install -e '.[mongo]' or pip install motor>=3.0.0"
            ) from e
    
    class MongoManagerBase:
        """Placeholder for MongoManagerBase when pymongo is not installed."""
        pass
from .interfaces import (
    VectorDatabaseManager,
    RelationalDatabaseManager,
    DocumentDatabaseManager,
    AsyncVectorDatabaseManager,
)
from .base import (
    VectorIndexConfig,
    VectorCollectionConfig,
    DatabaseConnectionConfig,
)

# Optional database backends - only imported if available
try:
    from .milvus import MilvusManager
except ImportError as e:
    # Create a placeholder class that provides helpful error message
    class MilvusManager:
        """Placeholder for MilvusManager when pymilvus is not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "MilvusManager requires 'pymilvus' to be installed. "
                "Install it with: pip install -e '.[milvus]' or pip install pymilvus>=2.6.6"
            ) from e

try:
    from .qdrant import QdrantManager
except ImportError as e:
    # Create a placeholder class that provides helpful error message
    class QdrantManager:
        """Placeholder for QdrantManager when qdrant-client is not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "QdrantManager requires 'qdrant-client' to be installed. "
                "Install it with: pip install -e '.[qdrant]' or pip install qdrant-client>=1.7.0"
            ) from e

__all__ = [
    "DatabaseManager",
    "DatabaseFactory",
    "PostgresDatabaseManagerImpl",
    "SQLManager",
    "MongoManagerBase",
    "SyncMongoManager",
    "AsyncMongoManager",
    "MilvusManager",
    "QdrantManager",
    # New interfaces
    "VectorDatabaseManager",
    "RelationalDatabaseManager",
    "DocumentDatabaseManager",
    "AsyncVectorDatabaseManager",
    # Configuration models
    "VectorIndexConfig",
    "VectorCollectionConfig",
    "DatabaseConnectionConfig",
]

