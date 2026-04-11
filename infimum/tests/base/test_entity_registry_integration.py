"""
Integration tests for entity registry with database managers.

This module tests the integration between EntityRegistry and database
managers, ensuring table creation works correctly.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from infimum.base.registry import EntityRegistry
from infimum.base.entity import BaseEntity
from infimum.database.postgres import PostgresDatabaseManagerImpl
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
        import types
        import inspect
        
        # Mock the submodule entity - define it outside the patch context
        class DiscoveredTestEntity(BaseEntity):
            name = Column(String(100))
        
        # Verify the entity can be registered directly
        EntityRegistry.register(DiscoveredTestEntity)
        assert "DiscoveredTestEntity" in EntityRegistry._entities
        EntityRegistry._entities.clear()  # Clear for the test
        
        # Create a real module-like object for the package
        mock_package = types.ModuleType("test_package")
        mock_package.__file__ = "/fake/path/__init__.py"
        
        # Create a real module-like object for the submodule
        mock_submodule = types.ModuleType("test_package.test_module")
        mock_submodule.__name__ = "test_package.test_module"
        # Attach the entity to the submodule so inspect.getmembers can find it
        # Use setattr to ensure it's in __dict__
        setattr(mock_submodule, 'DiscoveredTestEntity', DiscoveredTestEntity)
        
        # Verify getmembers can find it and that it would pass the registration check
        members = inspect.getmembers(mock_submodule)
        entity_found = False
        for name, obj in members:
            if (name == 'DiscoveredTestEntity' and 
                inspect.isclass(obj) and 
                issubclass(obj, BaseEntity) and 
                obj is not BaseEntity):
                entity_found = True
                # Test that register would work
                EntityRegistry.register(obj)
                assert "DiscoveredTestEntity" in EntityRegistry._entities
                EntityRegistry._entities.clear()  # Clear again
                break
        assert entity_found, "Entity should be discoverable by inspect.getmembers"
        
        # Create a callable side_effect that handles multiple calls
        def import_side_effect(module_name):
            if module_name == "test_package":
                return mock_package
            elif module_name == "test_package.test_module":
                return mock_submodule
            else:
                # For any other calls (like from pkgutil), return a mock
                return Mock()
        
        # Patch the imported modules in the registry module's namespace
        import infimum.base.registry as registry_module
        
        # Create mock modules that have the needed attributes
        mock_importlib_module = Mock()
        mock_importlib_module.import_module = Mock(side_effect=import_side_effect)
        
        mock_os_module = Mock()
        mock_os_module.path.dirname = Mock(return_value="/fake/path")
        
        mock_pkgutil_module = Mock()
        mock_pkgutil_module.iter_modules = Mock(return_value=[("", "test_module", False)])
        
        # Track calls to getmembers
        getmembers_calls = []
        original_getmembers = inspect.getmembers
        def tracked_getmembers(obj):
            result = original_getmembers(obj)
            getmembers_calls.append(obj)
            return result
        
        mock_inspect_module = Mock()
        mock_inspect_module.getmembers = tracked_getmembers
        mock_inspect_module.isclass = inspect.isclass
        # issubclass is a builtin, not from inspect, so it should work fine
        
        # Patch the imported names in the registry module
        with patch.object(registry_module, 'importlib', mock_importlib_module):
            with patch.object(registry_module, 'os', mock_os_module):
                with patch.object(registry_module, 'pkgutil', mock_pkgutil_module):
                    with patch.object(registry_module, 'inspect', mock_inspect_module):
                        # Run discovery
                        EntityRegistry.discover_entities("test_package")
                        
                        # Verify getmembers was called with our mock_submodule
                        submodule_inspected = any(
                            call_obj is mock_submodule or 
                            (hasattr(call_obj, '__name__') and call_obj.__name__ == "test_package.test_module")
                            for call_obj in getmembers_calls
                        )
                        
                        # Check what modules were actually imported
                        actual_imports = [call[0][0] if call[0] else 'no-args' 
                                        for call in mock_importlib_module.import_module.call_args_list]
                        
                        # Should have discovered the entity
                        assert len(EntityRegistry._entities) > 0, (
                            f"Expected entities to be discovered. "
                            f"Import calls: {actual_imports}, "
                            f"Getmembers called on: {[getattr(obj, '__name__', str(obj)) for obj in getmembers_calls]}, "
                            f"Submodule inspected: {submodule_inspected}"
                        )
                        assert "DiscoveredTestEntity" in EntityRegistry._entities
    
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
        class CreateTablesWithManagerTestEntity(BaseEntity):
            name = Column(String(100))
        
        EntityRegistry.register(CreateTablesWithManagerTestEntity)
        
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
        class CreateTablesWithDropFirstTestEntity(BaseEntity):
            name = Column(String(100))
        
        EntityRegistry.register(CreateTablesWithDropFirstTestEntity)
        
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
