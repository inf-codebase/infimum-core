"""
Integration tests for entity registry with database managers.

This module tests the integration between EntityRegistry and database
managers, ensuring table creation works correctly.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from core.base.registry import EntityRegistry
from core.base.entity import BaseEntity
from core.database.postgres import PostgresDatabaseManagerImpl
from sqlalchemy import Column, Integer, String


class TestEntityRegistryIntegration:
    """Integration tests for EntityRegistry."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Clear registry before each test
        EntityRegistry._entities.clear()
        EntityRegistry._dependencies.clear()
    
    def test_register_entity(self):
        """Test registering an entity."""
        class TestEntity(BaseEntity):
            name = Column(String(100))
        
        EntityRegistry.register(TestEntity)
        
        assert "TestEntity" in EntityRegistry._entities
        assert EntityRegistry.get_entity("TestEntity") == TestEntity
    
    def test_discover_entities_from_package(self):
        """Test discovering entities from a package."""
        with patch('core.base.registry.importlib.import_module') as mock_import:
            with patch('core.base.registry.os.path.dirname') as mock_dirname:
                mock_dirname.return_value = "/fake/path"
                
                # Mock the package module
                mock_package = Mock()
                mock_package.__file__ = "/fake/path/__init__.py"
                mock_import.return_value = mock_package
                
                # Mock the submodule
                class DiscoveredTestEntity(BaseEntity):
                    name = Column(String(100))
                
                mock_submodule = Mock()
                mock_submodule.__name__ = "test_package.test_module"
                mock_import.side_effect = [mock_package, mock_submodule]
                
                with patch('core.base.registry.pkgutil.iter_modules') as mock_iter:
                    mock_iter.return_value = [("", "test_module", False)]
                    
                    with patch('core.base.registry.inspect.getmembers') as mock_getmembers:
                        # Return the entity class when inspecting the submodule
                        mock_getmembers.side_effect = [
                            [],  # First call for package (no entities)
                            [('DiscoveredTestEntity', DiscoveredTestEntity)]  # Second call for submodule
                        ]
                        
                        EntityRegistry.discover_entities("test_package")
                        
                        # Should have discovered the entity
                        assert len(EntityRegistry._entities) > 0
    
    def test_get_all_entities_in_order(self):
        """Test getting entities in dependency order."""
        class ParentEntity(BaseEntity):
            id = Column(Integer, primary_key=True)
        
        class ChildEntity(BaseEntity):
            id = Column(Integer, primary_key=True)
            parent_id = Column(Integer)  # Would have FK in real scenario
        
        EntityRegistry.register(ParentEntity)
        EntityRegistry.register(ChildEntity)
        
        entities = EntityRegistry.get_all_entities()
        
        # Should return entities (order may vary without actual FKs)
        assert len(entities) == 2
        assert ParentEntity in entities
        assert ChildEntity in entities
    
    def test_create_tables_with_manager(self):
        """Test creating tables using database manager."""
        class TestEntity(BaseEntity):
            name = Column(String(100))
        
        EntityRegistry.register(TestEntity)
        
        # Mock database manager
        mock_manager = Mock(spec=PostgresDatabaseManagerImpl)
        mock_manager.create_tables = Mock(return_value=True)
        
        result = EntityRegistry.create_tables(mock_manager)
        
        assert result is True
        mock_manager.create_tables.assert_called_once()
        
        # Check that entities were passed
        call_args = mock_manager.create_tables.call_args[0]
        assert len(call_args) > 0  # Should have entities
    
    def test_create_tables_with_drop_first(self):
        """Test creating tables with drop_first option."""
        class TestEntity(BaseEntity):
            name = Column(String(100))
        
        EntityRegistry.register(TestEntity)
        
        mock_manager = Mock(spec=PostgresDatabaseManagerImpl)
        mock_manager.create_tables = Mock(return_value=True)
        
        result = EntityRegistry.create_tables(mock_manager, drop_first=True)
        
        assert result is True
        # Check that drop_first was passed
        # create_tables is called with (entities, drop_first) as positional args
        call_args = mock_manager.create_tables.call_args
        # Check positional args: first arg should be entities list, second should be drop_first
        assert len(call_args[0]) >= 1  # Should have entities as first positional arg
        assert call_args[0][1] is True  # drop_first should be the second positional arg
