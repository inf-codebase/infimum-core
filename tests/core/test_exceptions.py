"""
Tests for custom exception hierarchy.

This module tests that exceptions are properly structured and
can be caught at appropriate levels.
"""

import pytest
from core.exceptions import (
    CoreException,
    DatabaseException,
    DatabaseConnectionError,
    DatabaseQueryError,
    DatabaseConfigurationError,
    ConfigurationError,
    ProviderError,
    EmbeddingError,
    RegistryError,
)


class TestExceptionHierarchy:
    """Test that exception hierarchy is correct."""
    
    def test_database_exceptions_inherit_from_core(self):
        """Test that database exceptions inherit from CoreException."""
        assert issubclass(DatabaseException, CoreException)
        assert issubclass(DatabaseConnectionError, DatabaseException)
        assert issubclass(DatabaseQueryError, DatabaseException)
        assert issubclass(DatabaseConfigurationError, DatabaseException)
    
    def test_provider_exceptions_inherit_from_core(self):
        """Test that provider exceptions inherit from CoreException."""
        assert issubclass(ProviderError, CoreException)
        assert issubclass(EmbeddingError, ProviderError)
    
    def test_configuration_exception_inherits_from_core(self):
        """Test that ConfigurationError inherits from CoreException."""
        assert issubclass(ConfigurationError, CoreException)
    
    def test_registry_exception_inherits_from_core(self):
        """Test that RegistryError inherits from CoreException."""
        assert issubclass(RegistryError, CoreException)
    
    def test_catch_database_exceptions(self):
        """Test that catching DatabaseException catches all database errors."""
        try:
            raise DatabaseConnectionError("Connection failed")
        except DatabaseException:
            # Should catch all database exceptions
            pass
        else:
            pytest.fail("DatabaseException should catch DatabaseConnectionError")
    
    def test_catch_core_exceptions(self):
        """Test that catching CoreException catches all core errors."""
        exceptions = [
            DatabaseException("DB error"),
            ConfigurationError("Config error"),
            ProviderError("Provider error"),
            RegistryError("Registry error"),
        ]
        
        for exc in exceptions:
            try:
                raise exc
            except CoreException:
                # Should catch all core exceptions
                pass
            else:
                pytest.fail(f"CoreException should catch {type(exc).__name__}")
    
    def test_exception_messages(self):
        """Test that exceptions have proper messages."""
        msg = "Test error message"
        
        exc = DatabaseConnectionError(msg)
        assert str(exc) == msg
        
        exc = EmbeddingError(msg)
        assert str(exc) == msg
        
        exc = ConfigurationError(msg)
        assert str(exc) == msg
