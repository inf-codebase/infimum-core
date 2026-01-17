"""Unit tests for core.engine.decorators module."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from core.engine.decorators import (
    func_decorator,
    singleton,
    ParameterizedInjection,
    _safe_to_dict,
    _pick,
)
try:
    from core.ai.vlm.decorators import vlm_persist_after
except ImportError:
    # Fallback to old location for backward compatibility
    from core.engine.decorators import vlm_persist_after


class TestFuncDecorator:
    """Test cases for func_decorator function."""

    def test_func_decorator_with_params(self):
        """Test func_decorator with function that has parameters."""
        @func_decorator
        def test_func(param):
            return param
        
        result = test_func("test")
        assert result == "test"

    def test_func_decorator_without_params(self):
        """Test func_decorator with function without parameters."""
        @func_decorator
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
            ParameterizedInjection()


class TestSafeToDict:
    """Test cases for _safe_to_dict function."""

    def test_safe_to_dict_with_none(self):
        """Test _safe_to_dict with None."""
        result = _safe_to_dict(None)
        assert result == {}

    def test_safe_to_dict_with_dict(self):
        """Test _safe_to_dict with dict."""
        data = {"key": "value"}
        result = _safe_to_dict(data)
        assert result == data

    def test_safe_to_dict_with_pydantic_v1(self):
        """Test _safe_to_dict with Pydantic v1 model."""
        class MockPydantic:
            def dict(self):
                return {"key": "value"}
        
        result = _safe_to_dict(MockPydantic())
        assert result == {"key": "value"}

    def test_safe_to_dict_with_pydantic_v2(self):
        """Test _safe_to_dict with Pydantic v2 model."""
        class MockPydantic:
            def model_dump(self):
                return {"key": "value"}
        
        result = _safe_to_dict(MockPydantic())
        assert result == {"key": "value"}

    def test_safe_to_dict_with_object(self):
        """Test _safe_to_dict with regular object."""
        class TestObj:
            def __init__(self):
                self.public_attr = "value"
                self._private_attr = "hidden"
        
        result = _safe_to_dict(TestObj())
        assert result == {"public_attr": "value"}


class TestPick:
    """Test cases for _pick function."""

    def test_pick_finds_first_key(self):
        """Test _pick finds first available key."""
        data = {"key1": "value1", "key2": "value2"}
        result = _pick(data, "key1", "key2")
        assert result == "value1"

    def test_pick_finds_second_key(self):
        """Test _pick finds second key when first is missing."""
        data = {"key2": "value2"}
        result = _pick(data, "key1", "key2")
        assert result == "value2"

    def test_pick_returns_default(self):
        """Test _pick returns default when no keys found."""
        data = {}
        result = _pick(data, "key1", "key2", default="default")
        assert result == "default"

    def test_pick_ignores_none(self):
        """Test _pick ignores None values."""
        data = {"key1": None, "key2": "value2"}
        result = _pick(data, "key1", "key2")
        assert result == "value2"


class TestVlmPersistAfter:
    """Test cases for vlm_persist_after decorator."""

    @pytest.mark.asyncio
    async def test_vlm_persist_after_async(self):
        """Test vlm_persist_after with async function."""
        mock_task_instance = Mock()
        mock_apply_result = Mock()
        mock_apply_result.id = "task_123"
        mock_task_instance.apply_async.return_value = mock_apply_result
        
        @vlm_persist_after(mock_task_instance, queue="test")
        async def test_func():
            return {
                "transcript": "test transcript",
                "video_segment_id": "seg_123"
            }
        
        result = await test_func()
        
        assert result["transcript"] == "test transcript"
        mock_task_instance.apply_async.assert_called_once()

    def test_vlm_persist_after_sync(self):
        """Test vlm_persist_after with sync function."""
        mock_task_instance = Mock()
        mock_apply_result = Mock()
        mock_apply_result.id = "task_123"
        mock_task_instance.apply_async.return_value = mock_apply_result
        
        @vlm_persist_after(mock_task_instance, queue="test")
        def test_func():
            return {
                "transcript": "test transcript",
                "video_segment_id": "seg_123"
            }
        
        result = test_func()
        
        assert result["transcript"] == "test transcript"
        mock_task_instance.apply_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_vlm_persist_after_missing_transcript(self):
        """Test vlm_persist_after skips when transcript is missing."""
        mock_task_instance = Mock()
        
        @vlm_persist_after(mock_task_instance, queue="test")
        async def test_func():
            return {"video_segment_id": "seg_123"}
        
        result = await test_func()
        
        assert "transcript" not in result
        mock_task_instance.apply_async.assert_not_called()
