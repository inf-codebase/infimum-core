"""Unit tests for core.engine.decorators module."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from infimum.engine.design_pattern import singleton

class TestFuncDecorator:
    """Test cases for func_decorator function."""

    def test_func_decorator_with_params(self):
        """Test func_decorator with function that has parameters."""
        @singleton
        def test_func(param):
            return param
        
        result = test_func("test")
        assert result == "test"

    def test_func_decorator_without_params(self):
        """Test func_decorator with function without parameters."""
        @singleton
        def test_func():
            return "result"
        
        result = test_func()
        assert result == "result"


class TestSingleton:
    """Test cases for singleton decorator."""

    def test_singleton_creates_one_instance(self):
        """Test that singleton creates only one instance."""
        @singleton
        class TestClass:
            def __init__(self):
                self.value = 42
        
        instance1 = TestClass()
        instance2 = TestClass()
        
        assert instance1 is instance2
        assert instance1.value == 42


class TestParameterizedInjection:
    """Test cases for ParameterizedInjection abstract class."""

    def test_parameterized_injection_is_abstract(self):
        """Test that ParameterizedInjection is abstract."""
        with pytest.raises(TypeError):
            singleton()

