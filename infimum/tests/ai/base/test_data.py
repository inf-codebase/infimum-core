"""
Unit tests for data loading system.

Tests Factory, Registry, Strategy, and Template Method patterns.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from infimum.ai.base.data.base import BaseLoader
from infimum.ai.base.data.item import DataItem
from infimum.engine.design_pattern import Event, EventType


class TestDataItem(unittest.TestCase):
    """Test DataItem."""
    
    def test_data_item_creation(self):
        """Test creating a DataItem."""
        item = DataItem(
            data="test_data",
            data_type="text",
            source="/path/to/file"
        )
        
        self.assertEqual(item.data, "test_data")
        self.assertEqual(item.data_type, "text")
        self.assertEqual(item.source, "/path/to/file")
        self.assertIsInstance(item.metadata, dict)
    
    def test_data_item_get_set(self):
        """Test DataItem get/set methods."""
        item = DataItem(data="test", data_type="text")
        
        item.set("key1", "value1")
        self.assertEqual(item.get("key1"), "value1")
        self.assertEqual(item.get("nonexistent", "default"), "default")
    
    def test_data_item_update_metadata(self):
        """Test updating metadata."""
        item = DataItem(data="test", data_type="text")
        item.update_metadata(key1="value1", key2="value2")
        
        self.assertEqual(item.metadata["key1"], "value1")
        self.assertEqual(item.metadata["key2"], "value2")
    
    def test_data_item_to_dict(self):
        """Test converting DataItem to dictionary."""
        item = DataItem(
            data="test",
            data_type="text",
            source="/path",
            metadata={"key": "value"}
        )
        
        result = item.to_dict()
        self.assertEqual(result["data_type"], "text")
        self.assertEqual(result["metadata"]["key"], "value")
        self.assertEqual(result["source"], "/path")


class TestBaseLoader(unittest.TestCase):
    """Test BaseLoader (Strategy + Template Method pattern)."""
    
    def setUp(self):
        """Set up test loader implementation."""
        class TestLoader(BaseLoader):
            def _load(self, source):
                return DataItem(
                    data=f"loaded_{source}",
                    data_type="test",
                    source=str(source)
                )
        
        self.TestLoader = TestLoader
    
    def test_loader_abstract(self):
        """Test that BaseLoader cannot be instantiated."""
        with self.assertRaises(TypeError):
            BaseLoader()
    
    def test_template_method_workflow(self):
        """Test template method workflow."""
        loader = self.TestLoader()
        item = loader.load("test_source")
        
        self.assertEqual(item.data, "loaded_test_source")
        self.assertEqual(item.data_type, "test")
    
    def test_validation_in_template_method(self):
        """Test that validation is called."""
        loader = self.TestLoader()
        
        with self.assertRaises(ValueError):
            loader.load(None)  # Invalid source
    
    def test_pre_process_hook(self):
        """Test pre-process hook."""
        class PreProcessLoader(self.TestLoader):
            def _pre_process_source(self, source):
                return f"preprocessed_{source}"
        
        loader = PreProcessLoader()
        item = loader.load("test")
        
        self.assertEqual(item.data, "loaded_preprocessed_test")
    
    def test_post_process_hook(self):
        """Test post-process hook."""
        class PostProcessLoader(self.TestLoader):
            def _post_process_data(self, item):
                item.metadata["post_processed"] = True
                return item
        
        loader = PostProcessLoader()
        item = loader.load("test")
        
        self.assertTrue(item.metadata.get("post_processed", False))
    
    def test_validate_data_hook(self):
        """Test validate data hook."""
        class ValidateLoader(self.TestLoader):
            def _validate_data(self, item):
                if "invalid" in item.data:
                    raise ValueError("Invalid data")
        
        loader = ValidateLoader()
        
        # Valid data
        item = loader.load("test")
        self.assertIsNotNone(item)
        
        # Invalid data would raise in _load, but we can test the hook
        class InvalidLoader(self.TestLoader):
            def _load(self, source):
                return DataItem(data=None, data_type="")  # Invalid
        
        loader = InvalidLoader()
        with self.assertRaises(ValueError):
            loader.load("test")
    
    def test_observer_notifications(self):
        """Test that observers are notified."""
        class TestObserver:
            def __init__(self):
                self.events = []
            
            def on_event(self, event):
                self.events.append(event)
        
        loader = self.TestLoader()
        observer = TestObserver()
        loader.attach(observer)
        
        loader.load("test")
        
        self.assertGreater(len(observer.events), 0)
        event_types = [e.type for e in observer.events]
        self.assertIn(EventType.DATA_LOADING_STARTED, event_types)
        self.assertIn(EventType.DATA_LOADING_COMPLETED, event_types)

