"""Unit tests for core.engine.startup module."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from core.engine.startup import Startup
from fastapi import FastAPI


class TestStartup:
    """Test cases for Startup class."""

    def test_include_service(self):
        """Test including a service/router in FastAPI app."""
        app = FastAPI()
        from fastapi import APIRouter
        mock_router = APIRouter()
        
        result = Startup.include_service(app, mock_router)
        
        # Verify router was included (check that routers list contains it)
        assert len(app.routes) > 0
        assert result is not None

    @patch('core.engine.startup.EntityRegistry')
    @patch('core.engine.startup.context')
    @patch('core.engine.startup.get_config_by_prefixes')
    def test_initialize_with_postgres(self, mock_get_config, mock_context, mock_registry):
        """Test initialization with PostgreSQL database."""
        mock_get_config.return_value = {
            "POSTGRES_DATABASE_MAIN": "main_db"
        }
        mock_db_manager = Mock()
        mock_db_manager.create_tables = Mock(return_value=True)
        mock_context.get_database_manager.return_value = mock_db_manager
        mock_registry.create_tables.return_value = True
        
        Startup.initialize(create_tables=True, discover_entities=False)
        
        mock_context.register_postgres.assert_called()
        mock_registry.create_tables.assert_called()

    @patch('core.engine.startup.EntityRegistry')
    @patch('core.engine.startup.context')
    @patch('core.engine.startup.get_config_by_prefixes')
    def test_initialize_with_mongo(self, mock_get_config, mock_context, mock_registry):
        """Test initialization with MongoDB."""
        mock_get_config.return_value = {
            "MONGO_DATABASE_MAIN": "main_db"
        }
        
        Startup.initialize(create_tables=False, discover_entities=False)
        
        mock_context.register_mongo.assert_called()

    @patch('core.engine.startup.EntityRegistry')
    @patch('core.engine.startup.context')
    @patch('core.engine.startup.get_config_by_prefixes')
    def test_initialize_discover_entities(self, mock_get_config, mock_context, mock_registry):
        """Test initialization with entity discovery."""
        mock_get_config.return_value = {}
        
        Startup.initialize(discover_entities=True, entities_package="test_entities")
        
        mock_registry.discover_entities.assert_called_once_with("test_entities")

    @patch('core.engine.startup.EntityRegistry')
    @patch('core.engine.startup.context')
    def test_create_tables_for_db(self, mock_context, mock_registry):
        """Test creating tables for specific database."""
        mock_db_manager = Mock()
        mock_db_manager.create_tables = Mock(return_value=True)
        mock_context.get_database_manager.return_value = mock_db_manager
        mock_registry.create_tables.return_value = True
        
        Startup.create_tables_for_db("test_db", drop_first=False)
        
        mock_registry.create_tables.assert_called_once_with(mock_db_manager, False)

    @patch('core.engine.startup.EntityRegistry')
    @patch('core.engine.startup.context')
    def test_create_tables_for_db_with_drop(self, mock_context, mock_registry):
        """Test creating tables with drop_first option."""
        mock_db_manager = Mock()
        mock_db_manager.create_tables = Mock(return_value=True)
        mock_context.get_database_manager.return_value = mock_db_manager
        mock_registry.create_tables.return_value = True
        
        Startup.create_tables_for_db("test_db", drop_first=True)
        
        mock_registry.create_tables.assert_called_once_with(mock_db_manager, True)
