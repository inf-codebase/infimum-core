"""Unit tests for core.engine.context module."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from core.engine.context import (
    InjectionContainer,
    context,
    inject,
    register,
    get,
    register_postgres,
    register_mongo,
    get_database_manager,
    with_database
)


class TestInjectionContainer:
    """Test cases for InjectionContainer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.container = InjectionContainer()

    def test_register_dependency(self):
        """Test registering a dependency."""
        mock_dep = Mock()
        self.container.register("test_dep", mock_dep)
        
        assert self.container.get("test_dep") == mock_dep

    def test_register_factory(self):
        """Test registering a factory function."""
        def factory():
            return "created"
        
        self.container.register_factory("test_factory", factory)
        
        assert self.container.get("test_factory") == "created"

    def test_get_nonexistent_dependency(self):
        """Test getting non-existent dependency raises error."""
        with pytest.raises(KeyError):
            self.container.get("nonexistent")

    def test_inject_decorator(self):
        """Test inject decorator."""
        self.container.register("dep1", "value1")
        self.container.register("dep2", "value2")
        
        @self.container.inject("dep1", "dep2")
        def test_func(dep1, dep2, arg1):
            return f"{dep1}-{dep2}-{arg1}"
        
        result = test_func("test")
        assert result == "value1-value2-test"

    def test_register_postgres(self):
        """Test registering PostgreSQL database."""
        with patch('core.engine.context.auto_config') as mock_config:
            mock_config.POSTGRES_HOST = "localhost"
            mock_config.POSTGRES_PORT = "5432"
            mock_config.POSTGRES_DATABASE = "testdb"
            mock_config.POSTGRES_USER = "user"
            mock_config.POSTGRES_PASSWORD = "pass"
            
            self.container.register_postgres("test_db")
            
            assert "test_db" in self.container._database_configs

    def test_register_mongo(self):
        """Test registering MongoDB."""
        with patch('core.engine.context.auto_config') as mock_config:
            mock_config.MONGO_URI = "mongodb://localhost"
            mock_config.MONGO_DATABASE = "testdb"
            
            self.container.register_mongo("test_db")
            
            assert "test_db" in self.container._database_configs

    def test_register_sqlite(self):
        """Test registering SQLite database."""
        with patch('core.engine.context.auto_config') as mock_config:
            mock_config.SQLITE_PATH = "/path/to/db.sqlite"
            
            self.container.register_sqlite("test_db")
            
            assert "test_db" in self.container._database_configs

    def test_get_database_manager(self):
        """Test getting database manager."""
        mock_manager = Mock()
        self.container._database_factory.create_manager = Mock(return_value=mock_manager)
        self.container._database_configs["test_db"] = {"type": "postgres"}
        
        result = self.container.get_database_manager("test_db")
        
        assert result == mock_manager

    def test_close_database(self):
        """Test closing a database."""
        self.container._database_factory.close = Mock()
        
        self.container.close_database("test_db")
        
        self.container._database_factory.close.assert_called_once_with("test_db")

    def test_close_all_databases(self):
        """Test closing all databases."""
        self.container._database_factory.close_all = Mock()
        
        self.container.close_all_databases()
        
        self.container._database_factory.close_all.assert_called_once()


class TestGlobalFunctions:
    """Test cases for global helper functions."""

    def test_register_global(self):
        """Test global register function."""
        register("test_global", "value")
        
        assert get("test_global") == "value"

    @patch('core.engine.context.context')
    def test_register_postgres_global(self, mock_context):
        """Test global register_postgres function."""
        register_postgres("test_db", host="localhost")
        
        mock_context.register_postgres.assert_called_once()

    @patch('core.engine.context.context')
    def test_get_database_manager_global(self, mock_context):
        """Test global get_database_manager function."""
        mock_manager = Mock()
        mock_context.get_database_manager.return_value = mock_manager
        
        result = get_database_manager("test_db")
        
        assert result == mock_manager

    @patch('core.engine.context.context')
    def test_with_database(self, mock_context):
        """Test with_database helper function."""
        mock_manager = Mock()
        mock_manager.is_connected.return_value = True
        mock_context.get_database_manager.return_value = mock_manager
        
        result = with_database("test_db")
        
        assert result == mock_manager
