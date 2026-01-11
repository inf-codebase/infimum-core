"""
Unit tests for preprocessing system.

Tests Chain of Responsibility, Factory, and Registry patterns.
"""

import unittest
from unittest.mock import Mock

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from core.ai.base.preprocessing.base import BaseTransform
from core.ai.base.preprocessing.pipeline import TransformPipeline, TransformLink
from core.ai.base.preprocessing.factory import TransformFactory
from core.ai.base.preprocessing.registry import TransformRegistry, TransformMetadata
from core.ai.base.data.item import DataItem


class TestBaseTransform(unittest.TestCase):
    """Test BaseTransform (Strategy pattern)."""
    
    def setUp(self):
        """Set up test transform."""
        class TestTransform(BaseTransform):
            def transform(self, data):
                data.metadata["transformed"] = True
                return data
        
        self.TestTransform = TestTransform
    
    def test_transform_abstract(self):
        """Test that BaseTransform cannot be instantiated."""
        with self.assertRaises(TypeError):
            BaseTransform()
    
    def test_transform_callable(self):
        """Test that transform is callable."""
        transform = self.TestTransform()
        item = DataItem(data="test", data_type="text")
        
        result = transform(item)
        self.assertTrue(result.metadata.get("transformed", False))


class TestTransformLink(unittest.TestCase):
    """Test TransformLink (Chain of Responsibility)."""
    
    def setUp(self):
        """Set up test transforms."""
        class AddOneTransform(BaseTransform):
            def transform(self, data):
                if isinstance(data.data, int):
                    data.data += 1
                return data
        
        class MultiplyTwoTransform(BaseTransform):
            def transform(self, data):
                if isinstance(data.data, int):
                    data.data *= 2
                return data
        
        self.AddOne = AddOneTransform
        self.MultiplyTwo = MultiplyTwoTransform
    
    def test_single_link(self):
        """Test processing with single link."""
        transform = self.AddOne()
        link = TransformLink(transform)
        
        item = DataItem(data=5, data_type="number")
        result = link.process(item)
        
        self.assertEqual(result.data, 6)
    
    def test_chain_processing(self):
        """Test processing through a chain."""
        link1 = TransformLink(self.AddOne())
        link2 = TransformLink(self.MultiplyTwo())
        link1.set_next(link2)
        
        item = DataItem(data=5, data_type="number")
        result = link1.process(item)
        
        # (5 + 1) * 2 = 12
        self.assertEqual(result.data, 12)
    
    def test_chain_order(self):
        """Test that chain order matters."""
        # Chain: AddOne -> MultiplyTwo
        link1 = TransformLink(self.AddOne())
        link2 = TransformLink(self.MultiplyTwo())
        link1.set_next(link2)
        
        item = DataItem(data=5, data_type="number")
        result1 = link1.process(item)
        
        # Chain: MultiplyTwo -> AddOne
        link3 = TransformLink(self.MultiplyTwo())
        link4 = TransformLink(self.AddOne())
        link3.set_next(link4)
        
        item2 = DataItem(data=5, data_type="number")
        result2 = link3.process(item2)
        
        # Results should be different
        self.assertEqual(result1.data, 12)  # (5+1)*2
        self.assertEqual(result2.data, 11)  # (5*2)+1


class TestTransformPipeline(unittest.TestCase):
    """Test TransformPipeline (Chain of Responsibility)."""
    
    def setUp(self):
        """Set up test transforms."""
        class AddOneTransform(BaseTransform):
            def transform(self, data):
                if isinstance(data.data, int):
                    data.data += 1
                return data
        
        class MultiplyTwoTransform(BaseTransform):
            def transform(self, data):
                if isinstance(data.data, int):
                    data.data *= 2
                return data
        
        self.AddOne = AddOneTransform
        self.MultiplyTwo = MultiplyTwoTransform
    
    def test_pipeline_creation(self):
        """Test creating a pipeline."""
        transforms = [self.AddOne(), self.MultiplyTwo()]
        pipeline = TransformPipeline(transforms)
        
        self.assertIsNotNone(pipeline._head)
    
    def test_pipeline_empty_transforms(self):
        """Test that empty pipeline raises error."""
        with self.assertRaises(ValueError):
            TransformPipeline([])
    
    def test_pipeline_apply(self):
        """Test applying pipeline."""
        transforms = [self.AddOne(), self.MultiplyTwo()]
        pipeline = TransformPipeline(transforms)
        
        item = DataItem(data=5, data_type="number")
        result = pipeline.apply(item)
        
        # (5 + 1) * 2 = 12
        self.assertEqual(result.data, 12)
    
    def test_pipeline_callable(self):
        """Test that pipeline is callable."""
        transforms = [self.AddOne()]
        pipeline = TransformPipeline(transforms)
        
        item = DataItem(data=5, data_type="number")
        result = pipeline(item)
        
        self.assertEqual(result.data, 6)
    
    def test_pipeline_multiple_transforms(self):
        """Test pipeline with multiple transforms."""
        class AddTransform(BaseTransform):
            def __init__(self, value):
                self.value = value
            
            def transform(self, data):
                if isinstance(data.data, int):
                    data.data += self.value
                return data
        
        transforms = [
            AddTransform(1),
            AddTransform(2),
            AddTransform(3)
        ]
        pipeline = TransformPipeline(transforms)
        
        item = DataItem(data=0, data_type="number")
        result = pipeline.apply(item)
        
        self.assertEqual(result.data, 6)  # 0 + 1 + 2 + 3


class TestTransformFactory(unittest.TestCase):
    """Test TransformFactory (Factory pattern)."""
    
    def setUp(self):
        """Set up test transform."""
        class TestTransform(BaseTransform):
            def __init__(self, value=1):
                self.value = value
            
            def transform(self, data):
                if isinstance(data.data, int):
                    data.data += self.value
                return data
        
        self.TestTransform = TestTransform
        TransformFactory._registry.clear()
    
    def test_register_and_create(self):
        """Test registering and creating transforms."""
        TransformFactory.register("test", self.TestTransform)
        transform = TransformFactory.create("test", value=5)
        
        self.assertIsInstance(transform, self.TestTransform)
        self.assertEqual(transform.value, 5)
    
    def test_create_unregistered_transform(self):
        """Test creating unregistered transform raises error."""
        with self.assertRaises(ValueError) as cm:
            TransformFactory.create("nonexistent")
        
        self.assertIn("not registered", str(cm.exception))
    
    def test_create_pipeline(self):
        """Test creating a pipeline from transform names."""
        TransformFactory.register("add1", self.TestTransform)
        TransformFactory.register("add2", self.TestTransform)
        
        pipeline = TransformFactory.create_pipeline(["add1", "add2"], value=1)
        
        item = DataItem(data=0, data_type="number")
        result = pipeline.apply(item)
        
        self.assertEqual(result.data, 2)  # 0 + 1 + 1
    
    def test_list_transforms(self):
        """Test listing transforms."""
        TransformFactory.register("t1", self.TestTransform)
        TransformFactory.register("t2", self.TestTransform)
        
        transforms = TransformFactory.list_transforms()
        self.assertEqual(len(transforms), 2)
        self.assertIn("t1", transforms)
        self.assertIn("t2", transforms)


class TestTransformRegistry(unittest.TestCase):
    """Test TransformRegistry (Registry pattern)."""
    
    def setUp(self):
        """Clear registry before each test."""
        TransformRegistry.clear()
    
    def test_register_and_get(self):
        """Test registering and retrieving transforms."""
        metadata = TransformMetadata(
            transform_name="test_transform",
            data_type="text",
            description="Test transform"
        )
        TransformRegistry.register("test", metadata)
        
        retrieved = TransformRegistry.get("test")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.transform_name, "test_transform")
        self.assertEqual(retrieved.data_type, "text")
    
    def test_search_by_type(self):
        """Test searching transforms by data type."""
        TransformRegistry.register("t1", TransformMetadata("t1", "text"))
        TransformRegistry.register("t2", TransformMetadata("t2", "text"))
        TransformRegistry.register("t3", TransformMetadata("t3", "image"))
        
        text_transforms = TransformRegistry.search(data_type="text")
        self.assertEqual(len(text_transforms), 2)
        self.assertIn("t1", text_transforms)
        self.assertIn("t2", text_transforms)
    
    def test_get_by_type(self):
        """Test getting transforms by type."""
        TransformRegistry.register("t1", TransformMetadata("t1", "text"))
        TransformRegistry.register("t2", TransformMetadata("t2", "image"))
        
        text_metadata = TransformRegistry.get_by_type("text")
        self.assertEqual(len(text_metadata), 1)
        self.assertEqual(text_metadata[0].transform_name, "t1")


if __name__ == '__main__':
    unittest.main()
