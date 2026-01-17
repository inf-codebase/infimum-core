from .base import DatabaseManager
from .postgres import (
    DatabaseFactory,
    PostgresDatabaseManagerImpl,
    SQLManager,
    MongoManagerBase,
)
from .interfaces import (
    VectorDatabaseManager,
    RelationalDatabaseManager,
    DocumentDatabaseManager,
    AsyncVectorDatabaseManager,
)
from .config import (
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

