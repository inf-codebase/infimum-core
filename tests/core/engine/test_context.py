"""
Tests for dependency injection container and context management.

This module tests the InjectionContainer, multiple instance support,
and scoped container functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock
from core.engine.context import (
    InjectionContainer,
    container_scope,
    register,
    get,
    register_factory,
)
from core.database import DatabaseManager


class TestInjectionContainer:
    """Test cases for InjectionContainer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Clear all instances before each test
        if hasattr(InjectionContainer, '_instances'):
            InjectionContainer._instances.clear()
    
    def teardown_method(self):
        """Clean up after each test."""
        InjectionContainer.clear_all_instances()
    
    def test_register_and_get_dependency(self):
        """Test registering and retrieving a dependency."""
        container = InjectionContainer.get_instance("test")
        
        container.register("test_dep", "test_value")
        
        assert container.get("test_dep") == "test_value"
    
    def test_register_factory(self):
        """Test registering a factory function."""
        container = InjectionContainer.get_instance("test")
        
        def factory():
            return {"created": True}
        
        container.register_factory("factory_dep", factory)
        
        result = container.get("factory_dep")
        assert result["created"] is True
    
    def test_get_nonexistent_dependency(self):
        """Test that getting nonexistent dependency raises error."""
        container = InjectionContainer.get_instance("test")
        
        with pytest.raises(KeyError, match="Dependency 'nonexistent' not found"):
            container.get("nonexistent")
    
    def test_multiple_instances(self):
        """Test that multiple named instances work independently."""
        container1 = InjectionContainer.get_instance("container1")
        container2 = InjectionContainer.get_instance("container2")
        
        container1.register("dep", "value1")
        container2.register("dep", "value2")
        
        assert container1.get("dep") == "value1"
        assert container2.get("dep") == "value2"
    
    def test_singleton_per_name(self):
        """Test that same name returns same instance."""
        container1 = InjectionContainer.get_instance("test")
        container2 = InjectionContainer.get_instance("test")
        
        assert container1 is container2
        
        container1.register("test", "value")
        assert container2.get("test") == "value"
    
    def test_clear_instance(self):
        """Test clearing a specific instance."""
        container = InjectionContainer.get_instance("test")
        container.register("dep", "value")
        
        InjectionContainer.clear_instance("test")
        
        # Getting a new instance should be fresh
        new_container = InjectionContainer.get_instance("test")
        with pytest.raises(KeyError):
            new_container.get("dep")
    
    def test_clear_all_instances(self):
        """Test clearing all instances."""
        container1 = InjectionContainer.get_instance("test1")
        container2 = InjectionContainer.get_instance("test2")
        
        container1.register("dep1", "value1")
        container2.register("dep2", "value2")
        
        InjectionContainer.clear_all_instances()
        
        # New instances should be fresh
        new_container1 = InjectionContainer.get_instance("test1")
        new_container2 = InjectionContainer.get_instance("test2")
        
        with pytest.raises(KeyError):
            new_container1.get("dep1")
        with pytest.raises(KeyError):
            new_container2.get("dep2")


class TestContainerScope:
    """Test cases for container_scope context manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from core.engine.context import context
        # Reset current context
        context._current = context
    
    def test_container_scope_switches_context(self):
        """Test that container_scope switches to different container."""
        from core.engine.context import context
        
        # Register in default context
        register("default_dep", "default_value")
        
        # Use scoped container
        with container_scope("scoped"):
            register("scoped_dep", "scoped_value")
            assert get("scoped_dep") == "scoped_value"
            # Default dep should not be available in scoped context
            with pytest.raises(KeyError):
                get("default_dep")
        
        # Back to default context
        assert get("default_dep") == "default_value"
        with pytest.raises(KeyError):
            get("scoped_dep")
    
    def test_container_scope_nested(self):
        """Test nested container scopes."""
        from core.engine.context import context
        
        with container_scope("outer"):
            register("outer_dep", "outer_value")
            
            with container_scope("inner"):
                register("inner_dep", "inner_value")
                assert get("inner_dep") == "inner_value"
                # Outer dep should not be available in inner scope
                with pytest.raises(KeyError):
                    get("outer_dep")
            
            # Back to outer scope
            assert get("outer_dep") == "outer_value"
            with pytest.raises(KeyError):
                get("inner_dep")
    
    def test_container_scope_restores_previous(self):
        """Test that container_scope restores previous context."""
        from core.engine.context import context
        
        original_current = context._current
        
        with container_scope("test"):
            assert context._current.name == "test"
        
        # Should restore original
        assert context._current is original_current


class TestGlobalFunctions:
    """Test cases for global DI functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from core.engine.context import context
        context._current = context
        # Clear default container
        if hasattr(context, '_dependencies'):
            context._dependencies.clear()
        if hasattr(context, '_factories'):
            context._factories.clear()
    
    def test_global_register_and_get(self):
        """Test global register and get functions."""
        register("global_dep", "global_value")
        
        assert get("global_dep") == "global_value"
    
    def test_global_register_factory(self):
        """Test global register_factory function."""
        def factory():
            return "factory_value"
        
        register_factory("factory_dep", factory)
        
        assert get("factory_dep") == "factory_value"
    
    def test_global_functions_use_current_context(self):
        """Test that global functions use current context."""
        from core.engine.context import context
        
        # Register in default
        register("default_dep", "default")
        
        # Switch context and register
        with container_scope("other"):
            register("other_dep", "other")
            assert get("other_dep") == "other"
            # Default should not be available
            with pytest.raises(KeyError):
                get("default_dep")
        
        # Back to default
        assert get("default_dep") == "default"
