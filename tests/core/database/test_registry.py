"""
Tests for database backend registry system.

This module tests the plugin registry functionality, including registration,
creation, and discovery of database backends.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from core.database.registry import DatabaseBackendRegistry
from core.database.config import DatabaseConnectionConfig
from core.database.base import DatabaseManager
from core.database.milvus import MilvusManager
from core.database.qdrant import QdrantManager


class TestDatabaseBackendRegistry:
    """Test cases for DatabaseBackendRegistry."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Clear registry before each test
        DatabaseBackendRegistry.clear()
    
    def teardown_method(self):
        """Clean up after each test."""
        DatabaseBackendRegistry.clear()
    
    def test_register_backend(self):
        """Test registering a backend."""
        class TestBackend(DatabaseManager):
            def connect(self): pass
            def close(self): pass
            def insert_or_update(self, *args, **kwargs): pass
            def query_or_create_new(self, *args, **kwargs): pass
        
        DatabaseBackendRegistry.register("test_backend", TestBackend)
        
        assert DatabaseBackendRegistry.is_registered("test_backend")
        assert "test_backend" in DatabaseBackendRegistry.list_backends()
    
    def test_register_invalid_backend(self):
        """Test that registering non-DatabaseManager raises error."""
        class NotADatabaseManager:
            pass
        
        with pytest.raises(ValueError, match="must be a subclass of DatabaseManager"):
            DatabaseBackendRegistry.register("invalid", NotADatabaseManager)
    
    def test_create_backend_with_from_config(self):
        """Test creating backend using from_config method."""
        DatabaseBackendRegistry.register("milvus", MilvusManager)
        
        config = DatabaseConnectionConfig(host="localhost", port=19530)
        manager = DatabaseBackendRegistry.create("milvus", config)
        
        assert isinstance(manager, MilvusManager)
        assert manager.milvus_host == "localhost"
        assert manager.milvus_port == 19530
    
    def test_create_backend_without_from_config(self):
        """Test creating backend without from_config (fallback)."""
        class TestBackend(DatabaseManager):
            def __init__(self, host=None, port=None):
                self.host = host
                self.port = port
            def connect(self): pass
            def close(self): pass
            def insert_or_update(self, *args, **kwargs): pass
            def query_or_create_new(self, *args, **kwargs): pass
        
        DatabaseBackendRegistry.register("test", TestBackend)
        
        config = DatabaseConnectionConfig(host="localhost", port=5432)
        manager = DatabaseBackendRegistry.create("test", config)
        
        assert isinstance(manager, TestBackend)
        assert manager.host == "localhost"
        assert manager.port == 5432
    
    def test_create_unknown_backend(self):
        """Test that creating unknown backend raises error."""
        config = DatabaseConnectionConfig(host="localhost", port=5432)
        
        with pytest.raises(ValueError, match="Unknown backend"):
            DatabaseBackendRegistry.create("unknown_backend", config)
    
    def test_list_backends(self):
        """Test listing registered backends."""
        DatabaseBackendRegistry.register("backend1", MilvusManager)
        DatabaseBackendRegistry.register("backend2", QdrantManager)
        
        backends = DatabaseBackendRegistry.list_backends()
        
        assert "backend1" in backends
        assert "backend2" in backends
        assert len(backends) >= 2
    
    def test_unregister_backend(self):
        """Test unregistering a backend."""
        DatabaseBackendRegistry.register("test_backend", MilvusManager)
        assert DatabaseBackendRegistry.is_registered("test_backend")
        
        DatabaseBackendRegistry.unregister("test_backend")
        
        assert not DatabaseBackendRegistry.is_registered("test_backend")
        assert "test_backend" not in DatabaseBackendRegistry.list_backends()
    
    def test_unregister_nonexistent_backend(self):
        """Test that unregistering nonexistent backend raises error."""
        with pytest.raises(KeyError):
            DatabaseBackendRegistry.unregister("nonexistent")
    
    def test_clear_registry(self):
        """Test clearing all registered backends."""
        DatabaseBackendRegistry.register("backend1", MilvusManager)
        DatabaseBackendRegistry.register("backend2", QdrantManager)
        
        assert len(DatabaseBackendRegistry.list_backends()) >= 2
        
        DatabaseBackendRegistry.clear()
        
        assert len(DatabaseBackendRegistry.list_backends()) == 0
    
    def test_register_with_factory(self):
        """Test registering backend with custom factory function."""
        class TestBackend(DatabaseManager):
            def __init__(self, custom_param):
                self.custom_param = custom_param
            def connect(self): pass
            def close(self): pass
            def insert_or_update(self, *args, **kwargs): pass
            def query_or_create_new(self, *args, **kwargs): pass
        
        def factory(config, **kwargs):
            return TestBackend(custom_param="custom_value")
        
        DatabaseBackendRegistry.register("test", TestBackend, factory=factory)
        
        config = DatabaseConnectionConfig()
        manager = DatabaseBackendRegistry.create("test", config)
        
        assert isinstance(manager, TestBackend)
        assert manager.custom_param == "custom_value"


class TestPluginDiscovery:
    """Test plugin discovery functionality."""
    
    @patch('core.database.plugins.importlib.import_module')
    @patch('core.database.plugins.pkgutil.iter_modules')
    def test_discover_plugins(self, mock_iter_modules, mock_import_module):
        """Test discovering plugins from a package."""
        from core.database.plugins import discover_database_plugins
        
        # Mock package structure
        mock_package = Mock()
        mock_package.__path__ = ['/fake/path']
        mock_import_module.return_value = mock_package
        
        # Mock module iterator
        mock_iter_modules.return_value = [
            (None, 'plugin1', False),
            (None, 'plugin2', False),
        ]
        
        # Mock plugin modules
        mock_module1 = Mock()
        mock_module1.register_backend = Mock()
        mock_module2 = Mock()
        mock_module2.register_backend = Mock()
        
        def import_side_effect(name):
            if name == "core.database.plugins":
                return mock_package
            elif name == "core.database.plugins.plugin1":
                return mock_module1
            elif name == "core.database.plugins.plugin2":
                return mock_module2
            raise ImportError()
        
        mock_import_module.side_effect = import_side_effect
        
        count = discover_database_plugins("core.database.plugins")
        
        assert count == 2
        mock_module1.register_backend.assert_called_once()
        mock_module2.register_backend.assert_called_once()
    
    @patch('core.database.plugins.importlib.import_module')
    def test_discover_plugins_package_not_exists(self, mock_import_module):
        """Test that missing plugin package doesn't raise error."""
        from core.database.plugins import discover_database_plugins
        
        mock_import_module.side_effect = ImportError("No module named 'core.database.plugins'")
        
        # Should not raise, just return 0
        count = discover_database_plugins("core.database.plugins")
        assert count == 0
    
    def test_register_builtin_backends(self):
        """Test that built-in backends are registered."""
        from core.database.plugins import register_builtin_backends
        from core.database.registry import DatabaseBackendRegistry
        
        # Clear registry first
        DatabaseBackendRegistry.clear()
        
        # Register built-ins
        register_builtin_backends()
        
        # Check that built-ins are registered
        assert DatabaseBackendRegistry.is_registered("postgres")
        assert DatabaseBackendRegistry.is_registered("milvus")
        assert DatabaseBackendRegistry.is_registered("qdrant")
        assert DatabaseBackendRegistry.is_registered("mongo")
        assert DatabaseBackendRegistry.is_registered("milvus_async")
        assert DatabaseBackendRegistry.is_registered("qdrant_async")
        assert DatabaseBackendRegistry.is_registered("mongo_async")
