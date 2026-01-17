"""
Contract tests for database interfaces.

This module tests that all database manager implementations properly
implement their respective interfaces, ensuring interface compliance.
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any

from core.database.interfaces import (
    VectorDatabaseManager,
    RelationalDatabaseManager,
    DocumentDatabaseManager,
)
from core.database.milvus import MilvusManager
from core.database.qdrant import QdrantManager
from core.database.postgres import PostgresDatabaseManagerImpl, SyncMongoManager
from core.database.config import VectorIndexConfig, DatabaseConnectionConfig


class TestVectorDatabaseManagerInterface:
    """Test that vector database managers implement VectorDatabaseManager interface."""
    
    def test_milvus_manager_implements_interface(self):
        """Test that MilvusManager implements VectorDatabaseManager."""
        assert issubclass(MilvusManager, VectorDatabaseManager)
        
        manager = MilvusManager(milvus_host="localhost", milvus_port=19530)
        assert isinstance(manager, VectorDatabaseManager)
    
    def test_qdrant_manager_implements_interface(self):
        """Test that QdrantManager implements VectorDatabaseManager."""
        assert issubclass(QdrantManager, VectorDatabaseManager)
        
        manager = QdrantManager(qdrant_url="http://localhost:6333")
        assert isinstance(manager, VectorDatabaseManager)
    
    def test_vector_managers_have_required_methods(self):
        """Test that vector managers have all required interface methods."""
        manager = MilvusManager(milvus_host="localhost", milvus_port=19530)
        
        # Check all abstract methods exist
        assert hasattr(manager, 'ensure_collection')
        assert hasattr(manager, 'insert_points')
        assert hasattr(manager, 'query_points')
        assert hasattr(manager, 'scroll_points')
        assert hasattr(manager, 'delete_points')
        assert hasattr(manager, 'update_point_payload')
        assert hasattr(manager, 'is_connected')
        assert hasattr(manager, 'connect')
        assert hasattr(manager, 'close')
    
    def test_vector_manager_from_config(self):
        """Test that vector managers support from_config class method."""
        config = DatabaseConnectionConfig(host="localhost", port=19530)
        
        # MilvusManager should support from_config
        manager = MilvusManager.from_config(config)
        assert isinstance(manager, MilvusManager)
        assert manager.milvus_host == "localhost"
        assert manager.milvus_port == 19530
        
        # QdrantManager should support from_config
        config2 = DatabaseConnectionConfig(host="http://localhost", port=6333)
        manager2 = QdrantManager.from_config(config2, use_memory=True)
        assert isinstance(manager2, QdrantManager)


class TestRelationalDatabaseManagerInterface:
    """Test that relational database managers implement RelationalDatabaseManager interface."""
    
    def test_postgres_manager_implements_interface(self):
        """Test that PostgresDatabaseManagerImpl implements RelationalDatabaseManager."""
        assert issubclass(PostgresDatabaseManagerImpl, RelationalDatabaseManager)
    
    def test_relational_managers_have_required_methods(self):
        """Test that relational managers have all required interface methods."""
        manager = PostgresDatabaseManagerImpl(engine_info="sqlite:///:memory:")
        manager.connect()
        
        # Check all abstract methods exist
        assert hasattr(manager, 'insert_or_update')
        assert hasattr(manager, 'query_or_create_new')
        assert hasattr(manager, 'create_tables')
        assert hasattr(manager, 'is_connected')
        assert hasattr(manager, 'connect')
        assert hasattr(manager, 'close')
        
        manager.close()


class TestDocumentDatabaseManagerInterface:
    """Test that document database managers implement DocumentDatabaseManager interface."""
    
    def test_mongo_manager_implements_interface(self):
        """Test that SyncMongoManager implements DocumentDatabaseManager."""
        assert issubclass(SyncMongoManager, DocumentDatabaseManager)
    
    def test_document_managers_have_required_methods(self):
        """Test that document managers have all required interface methods."""
        from unittest.mock import patch, MagicMock
        
        # Mock MongoClient to avoid actual connection and ImportError
        mock_client_class = MagicMock()
        mock_client_instance = MagicMock()
        mock_db = MagicMock()
        # Set up the client to return the db when indexed
        mock_client_instance.__getitem__.return_value = mock_db
        mock_client_class.return_value = mock_client_instance
        
        # Patch MongoClient in the postgres module
        # This replaces the None value with a mock class
        with patch('core.database.postgres.MongoClient', mock_client_class):
            # Use a mock connection string for testing
            manager = SyncMongoManager(
                connection_string="mongodb://localhost:27017",
                database="test_db"
            )
            
            # Verify that client and db were set by connect()
            assert hasattr(manager, 'client')
            assert manager.client is not None
            assert hasattr(manager, 'db')
            assert manager.db is not None
            
            # Check all abstract methods exist
            assert hasattr(manager, 'get_collection')
            assert hasattr(manager, 'insert_one')
            assert hasattr(manager, 'find_one')
            assert hasattr(manager, 'update_one')
            assert hasattr(manager, 'is_connected')
            assert hasattr(manager, 'connect')
            assert hasattr(manager, 'close')
            
            manager.close()


class TestInterfaceCompliance:
    """Test interface compliance and method signatures."""
    
    def test_vector_manager_method_signatures(self):
        """Test that vector manager methods have correct signatures."""
        import inspect
        
        manager = MilvusManager(milvus_host="localhost", milvus_port=19530)
        
        # Check ensure_collection signature
        sig = inspect.signature(manager.ensure_collection)
        params = list(sig.parameters.keys())
        assert 'collection_name' in params
        assert 'has_vector' in params
        assert 'vector_size' in params
        assert 'index_config' in params
        
        # Check insert_points signature
        sig = inspect.signature(manager.insert_points)
        params = list(sig.parameters.keys())
        assert 'collection_name' in params
        assert 'points' in params
        assert 'has_vector' in params
        
        # Check query_points signature
        sig = inspect.signature(manager.query_points)
        params = list(sig.parameters.keys())
        assert 'collection_name' in params
        assert 'query_vector' in params
        assert 'top' in params
        assert 'filter' in params
    
    def test_configuration_model_usage(self):
        """Test that managers can use configuration models."""
        from core.database.config import VectorIndexConfig, DatabaseConnectionConfig
        
        # Test VectorIndexConfig
        index_config = VectorIndexConfig(
            metric_type="COSINE",
            index_type="HNSW",
            params={"M": 16, "ef_construction": 200}
        )
        assert index_config.metric_type == "COSINE"
        assert index_config.index_type == "HNSW"
        assert index_config.params["M"] == 16
        
        # Test DatabaseConnectionConfig
        db_config = DatabaseConnectionConfig(
            host="localhost",
            port=19530,
            database="test_db"
        )
        assert db_config.host == "localhost"
        assert db_config.port == 19530
