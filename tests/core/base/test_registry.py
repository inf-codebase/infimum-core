"""Unit tests for core.base.registry module."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from core.base.registry import EntityRegistry
from core.base.entity import BaseEntity
from sqlalchemy import Column, String, Integer, ForeignKey


class TestEntityRegistry:
    """Test cases for EntityRegistry class."""

    def setup_method(self):
        """Reset registry before each test."""
        EntityRegistry._entities.clear()
        EntityRegistry._dependencies.clear()

    def test_register_entity(self):
        """Test registering an entity class."""
        class TestEntity(BaseEntity):
            name = Column(String(100))
        
        EntityRegistry.register(TestEntity)
        assert TestEntity.__name__ in EntityRegistry._entities
        assert EntityRegistry._entities[TestEntity.__name__] == TestEntity

    def test_register_base_entity_ignored(self):
        """Test that BaseEntity itself is not registered."""
        initial_count = len(EntityRegistry._entities)
        EntityRegistry.register(BaseEntity)
        assert len(EntityRegistry._entities) == initial_count

    def test_register_non_entity_ignored(self):
        """Test that non-BaseEntity classes are ignored."""
        class RegularClass:
            pass
        
        initial_count = len(EntityRegistry._entities)
        EntityRegistry.register(RegularClass)
        assert len(EntityRegistry._entities) == initial_count

    def test_get_entity(self):
        """Test retrieving a registered entity."""
        class TestEntity(BaseEntity):
            pass
        
        EntityRegistry.register(TestEntity)
        retrieved = EntityRegistry.get_entity("TestEntity")
        assert retrieved == TestEntity

    def test_get_entity_not_found(self):
        """Test retrieving a non-existent entity returns None."""
        result = EntityRegistry.get_entity("NonExistentEntity")
        assert result is None

    def test_get_all_entities(self):
        """Test getting all registered entities."""
        class Entity1(BaseEntity):
            pass
        
        class Entity2(BaseEntity):
            pass
        
        EntityRegistry.register(Entity1)
        EntityRegistry.register(Entity2)
        
        entities = EntityRegistry.get_all_entities()
        assert len(entities) == 2
        assert Entity1 in entities
        assert Entity2 in entities

    @patch('core.base.registry.importlib.import_module')
    @patch('core.base.registry.pkgutil.iter_modules')
    @patch('core.base.registry.inspect.getmembers')
    def test_discover_entities(self, mock_getmembers, mock_iter_modules, mock_import_module):
        """Test entity discovery from a package."""
        # Mock package structure
        mock_module = Mock()
        mock_module.__file__ = "/path/to/package/__init__.py"
        
        class DiscoveredEntity(BaseEntity):
            pass
        
        mock_getmembers.return_value = [('DiscoveredEntity', DiscoveredEntity)]
        mock_iter_modules.return_value = [('module1', False)]
        mock_import_module.return_value = mock_module
        
        EntityRegistry.discover_entities('test_package')
        
        assert 'DiscoveredEntity' in EntityRegistry._entities

    @patch('core.base.registry.logger')
    def test_discover_entities_invalid_package(self, mock_logger):
        """Test entity discovery with invalid package."""
        EntityRegistry.discover_entities('non_existent_package')
        # Should log warning but not raise exception
        assert True

    def test_topological_sort_simple(self):
        """Test topological sort with no dependencies."""
        class Entity1(BaseEntity):
            pass
        
        class Entity2(BaseEntity):
            pass
        
        EntityRegistry.register(Entity1)
        EntityRegistry.register(Entity2)
        
        ordered = EntityRegistry._topological_sort()
        assert len(ordered) == 2

    def test_topological_sort_with_dependencies(self):
        """Test topological sort with dependencies."""
        class ParentEntity(BaseEntity):
            id = Column(Integer, primary_key=True)
        
        class ChildEntity(BaseEntity):
            id = Column(Integer, primary_key=True)
            parent_id = Column(Integer, ForeignKey('parententity.id'))
        
        EntityRegistry.register(ParentEntity)
        EntityRegistry.register(ChildEntity)
        
        # Re-analyze dependencies
        EntityRegistry._analyze_all_dependencies()
        
        ordered = EntityRegistry._topological_sort()
        # Parent should come before child
        parent_idx = next(i for i, e in enumerate(ordered) if e.__name__ == 'ParentEntity')
        child_idx = next(i for i, e in enumerate(ordered) if e.__name__ == 'ChildEntity')
        assert parent_idx < child_idx

    @patch('core.base.registry.logger')
    def test_create_tables_with_db_manager(self, mock_logger):
        """Test table creation with database manager."""
        class TestEntity(BaseEntity):
            pass
        
        EntityRegistry.register(TestEntity)
        
        mock_db_manager = Mock()
        mock_db_manager.create_tables = Mock(return_value=True)
        
        result = EntityRegistry.create_tables(mock_db_manager)
        assert result is True
        mock_db_manager.create_tables.assert_called_once()

    @patch('core.base.registry.logger')
    def test_create_tables_with_engine(self, mock_logger):
        """Test table creation with engine."""
        class TestEntity(BaseEntity):
            pass
        
        EntityRegistry.register(TestEntity)
        
        mock_engine = Mock()
        mock_db_manager = Mock()
        mock_db_manager.engine = mock_engine
        mock_db_manager.create_tables = None
        
        with patch('core.base.entity.BaseEntity.metadata') as mock_metadata:
            result = EntityRegistry.create_tables(mock_db_manager)
            assert result is True

    @patch('core.base.registry.logger')
    def test_create_tables_with_drop_first(self, mock_logger):
        """Test table creation with drop_first option."""
        class TestEntity(BaseEntity):
            pass
        
        EntityRegistry.register(TestEntity)
        
        mock_db_manager = Mock()
        mock_db_manager.create_tables = Mock(return_value=True)
        
        result = EntityRegistry.create_tables(mock_db_manager, drop_first=True)
        assert result is True
        mock_db_manager.create_tables.assert_called_once_with(
            EntityRegistry.get_all_entities(), True
        )

    def test_analyze_dependencies(self):
        """Test dependency analysis."""
        class ParentEntity(BaseEntity):
            id = Column(Integer, primary_key=True)
        
        class ChildEntity(BaseEntity):
            id = Column(Integer, primary_key=True)
            parent_id = Column(Integer, ForeignKey('parententity.id'))
        
        EntityRegistry.register(ParentEntity)
        EntityRegistry.register(ChildEntity)
        
        EntityRegistry._analyze_dependencies(ChildEntity)
        
        assert 'ParentEntity' in EntityRegistry._dependencies.get('ChildEntity', set())
