"""
Integration tests for database factory with registry system.

This module tests the integration between DatabaseFactory and
DatabaseBackendRegistry, ensuring they work together correctly.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from core.database.postgres import DatabaseFactory
from core.database.registry import DatabaseBackendRegistry
from core.database.config import DatabaseConnectionConfig
from core.database.milvus import MilvusManager


class TestDatabaseFactoryIntegration:
    """Integration tests for DatabaseFactory with registry."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factory = DatabaseFactory()
    
    def test_factory_uses_registry(self):
        """Test that factory uses registry for backend creation."""
        # Register a test backend
        class TestBackend(MilvusManager):
            pass
        
        DatabaseBackendRegistry.register("test_milvus", TestBackend)
        
        # Register config
        config = {
            'type': 'test_milvus',
            'host': 'localhost',
            'port': 19530
        }
        self.factory.register_config("test_db", config)
        
        # Create manager
        manager = self.factory.create_manager("test_db")
        
        assert isinstance(manager, TestBackend)
    
    def test_factory_fallback_to_legacy(self):
        """Test that factory falls back to legacy creation if registry fails."""
        # Register config with unknown type (will try registry first, then legacy)
        config = {
            'type': 'postgres',
            'engine_info': 'sqlite:///:memory:'
        }
        self.factory.register_config("test_db", config)
        
        # Should work via legacy path
        manager = self.factory.create_manager("test_db")
        
        assert manager is not None
    
    def test_factory_handles_async_mode(self):
        """Test that factory handles async mode correctly."""
        config = {
            'type': 'milvus',
            'host': 'localhost',
            'port': 19530
        }
        self.factory.register_config("test_db", config)
        
        # Request async manager
        manager = self.factory.create_manager("test_db", async_mode=True)
        
        # Should get async version
        from core.database.milvus import AsyncMilvusManager
        assert isinstance(manager, AsyncMilvusManager)
    
    def test_factory_manager_caching(self):
        """Test that factory caches manager instances."""
        config = {
            'type': 'milvus',
            'host': 'localhost',
            'port': 19530
        }
        self.factory.register_config("test_db", config)
        
        manager1 = self.factory.create_manager("test_db")
        manager2 = self.factory.create_manager("test_db")
        
        # Should return same instance
        assert manager1 is manager2
    
    def test_factory_close_manager(self):
        """Test closing a manager through factory."""
        config = {
            'type': 'milvus',
            'host': 'localhost',
            'port': 19530
        }
        self.factory.register_config("test_db", config)
        
        manager = self.factory.create_manager("test_db")
        manager.close = Mock()
        
        self.factory.close("test_db")
        
        manager.close.assert_called_once()
    
    def test_factory_close_all(self):
        """Test closing all managers through factory."""
        config1 = {'type': 'milvus', 'host': 'localhost', 'port': 19530}
        config2 = {'type': 'milvus', 'host': 'localhost', 'port': 19531}
        
        self.factory.register_config("db1", config1)
        self.factory.register_config("db2", config2)
        
        manager1 = self.factory.create_manager("db1")
        manager2 = self.factory.create_manager("db2")
        
        manager1.close = Mock()
        manager2.close = Mock()
        
        self.factory.close_all()
        
        manager1.close.assert_called_once()
        manager2.close.assert_called_once()
