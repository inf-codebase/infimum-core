"""
Integration tests for base module.

Tests how different patterns work together.
"""

import unittest
from unittest.mock import Mock, MagicMock

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from core.ai.base.providers.base import BaseProvider, ModelHandle
from core.ai.base.providers.config import ModelConfig, ModelConfigBuilder
from core.ai.base.providers.factory import ProviderFactory
from core.ai.base.providers.registry import ProviderRegistry, ProviderMetadata

from core.ai.base.data.base import BaseLoader
from core.ai.base.data.item import DataItem
from core.ai.base.data.factory import LoaderFactory

from core.ai.base.preprocessing.base import BaseTransform
from core.ai.base.preprocessing.pipeline import TransformPipeline
from core.ai.base.preprocessing.factory import TransformFactory

from core.ai.base.observers.base import Observer, Observable
from core.ai.base.observers.events import Event, EventType


class TestProviderDataIntegration(unittest.TestCase):
    """Test integration between providers and data loaders."""
    
    def setUp(self):
        """Set up test components."""
        # Mock provider
        class MockProvider(BaseProvider):
            def load_model(self, config):
                return ModelHandle(
                    model={"model": "mock"},
                    config=config
                )
            
            def unload_model(self, handle):
                pass
        
        # Mock loader
        class MockLoader(BaseLoader):
            def _load(self, source):
                return DataItem(
                    data="loaded_data",
                    data_type="test",
                    source=str(source)
                )
        
        self.MockProvider = MockProvider
        self.MockLoader = MockLoader
        
        # Clear registries
        ProviderFactory._registry.clear()
        LoaderFactory._registry.clear()
    
    def test_provider_with_loader(self):
        """Test using provider with data loader."""
        # Register components
        ProviderFactory.register("llm", "mock", self.MockProvider)
        LoaderFactory.register("test", self.MockLoader)
        
        # Create components using factories
        config = (ModelConfigBuilder()
            .with_model_type("llm")
            .with_provider("mock")
            .with_model_path("/path")
            .build())
        
        provider = ProviderFactory.create("llm", "mock", config)
        loader = LoaderFactory.create("test")
        
        # Use together
        data_item = loader.load("source")
        handle = provider.get_model(config)
        
        self.assertIsNotNone(data_item)
        self.assertIsNotNone(handle)


class TestDataPreprocessingIntegration(unittest.TestCase):
    """Test integration between data loaders and preprocessing."""
    
    def setUp(self):
        """Set up test components."""
        class MockLoader(BaseLoader):
            def _load(self, source):
                return DataItem(
                    data=5,
                    data_type="number",
                    source=str(source)
                )
        
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
        
        self.MockLoader = MockLoader
        self.AddOne = AddOneTransform
        self.MultiplyTwo = MultiplyTwoTransform
        
        LoaderFactory._registry.clear()
        TransformFactory._registry.clear()
    
    def test_loader_with_pipeline(self):
        """Test using loader with preprocessing pipeline."""
        # Register loader
        LoaderFactory.register("test", self.MockLoader)
        
        # Create pipeline
        transforms = [self.AddOne(), self.MultiplyTwo()]
        pipeline = TransformPipeline(transforms)
        
        # Use together
        loader = LoaderFactory.create("test")
        data_item = loader.load("source")
        processed = pipeline.apply(data_item)
        
        # (5 + 1) * 2 = 12
        self.assertEqual(processed.data, 12)
    
    def test_loader_with_factory_pipeline(self):
        """Test using loader with factory-created pipeline."""
        LoaderFactory.register("test", self.MockLoader)
        TransformFactory.register("add1", self.AddOne)
        TransformFactory.register("mul2", self.MultiplyTwo)
        
        loader = LoaderFactory.create("test")
        pipeline = TransformFactory.create_pipeline(["add1", "mul2"])
        
        data_item = loader.load("source")
        processed = pipeline.apply(data_item)
        
        self.assertEqual(processed.data, 12)


class TestObserverIntegration(unittest.TestCase):
    """Test observer integration with providers and loaders."""
    
    def setUp(self):
        """Set up test components."""
        class MockProvider(BaseProvider):
            def load_model(self, config):
                return ModelHandle(model="mock", config=config)
            
            def unload_model(self, handle):
                pass
        
        class MockLoader(BaseLoader):
            def _load(self, source):
                return DataItem(data="data", data_type="test")
        
        self.MockProvider = MockProvider
        self.MockLoader = MockLoader
    
    def test_provider_observer(self):
        """Test observers with providers."""
        class EventCollector(Observer):
            def __init__(self):
                self.events = []
            
            def on_event(self, event):
                self.events.append(event)
        
        config = ModelConfig(
            model_type="llm",
            provider="test",
            model_path="/path"
        )
        
        provider = self.MockProvider()
        collector = EventCollector()
        provider.attach(collector)
        
        provider.get_model(config)
        
        # Should have received events
        self.assertGreater(len(collector.events), 0)
        event_types = [e.type for e in collector.events]
        self.assertIn(EventType.MODEL_LOADING_STARTED, event_types)
    
    def test_loader_observer(self):
        """Test observers with loaders."""
        class EventCollector(Observer):
            def __init__(self):
                self.events = []
            
            def on_event(self, event):
                self.events.append(event)
        
        loader = self.MockLoader()
        collector = EventCollector()
        loader.attach(collector)
        
        loader.load("source")
        
        # Should have received events
        self.assertGreater(len(collector.events), 0)
        event_types = [e.type for e in collector.events]
        self.assertIn(EventType.DATA_LOADING_STARTED, event_types)


class TestFullWorkflow(unittest.TestCase):
    """Test complete workflow using all patterns."""
    
    def setUp(self):
        """Set up complete test workflow."""
        # Mock provider
        class MockProvider(BaseProvider):
            def load_model(self, config):
                return ModelHandle(
                    model={"processor": Mock()},
                    config=config
                )
            
            def unload_model(self, handle):
                pass
        
        # Mock loader
        class MockLoader(BaseLoader):
            def _load(self, source):
                return DataItem(
                    data="raw_data",
                    data_type="text",
                    source=str(source)
                )
        
        # Mock transform
        class ProcessTransform(BaseTransform):
            def transform(self, data):
                data.metadata["processed"] = True
                data.data = f"processed_{data.data}"
                return data
        
        self.MockProvider = MockProvider
        self.MockLoader = MockLoader
        self.ProcessTransform = ProcessTransform
        
        # Clear registries
        ProviderFactory._registry.clear()
        LoaderFactory._registry.clear()
        TransformFactory._registry.clear()
    
    def test_complete_workflow(self):
        """Test complete workflow: config -> provider -> loader -> preprocessing."""
        # Register components
        ProviderFactory.register("llm", "mock", self.MockProvider)
        LoaderFactory.register("text", self.MockLoader)
        TransformFactory.register("process", self.ProcessTransform)
        
        # Build configuration
        config = (ModelConfigBuilder()
            .with_model_type("llm")
            .with_provider("mock")
            .with_model_path("/path/to/model")
            .with_device("cuda")
            .build())
        
        # Create components
        provider = ProviderFactory.create("llm", "mock", config)
        loader = LoaderFactory.create("text")
        pipeline = TransformFactory.create_pipeline(["process"])
        
        # Execute workflow
        # 1. Load data
        data_item = loader.load("input.txt")
        self.assertEqual(data_item.data, "raw_data")
        
        # 2. Preprocess
        processed_item = pipeline.apply(data_item)
        self.assertEqual(processed_item.data, "processed_raw_data")
        self.assertTrue(processed_item.metadata.get("processed", False))
        
        # 3. Load model
        handle = provider.get_model(config)
        self.assertIsNotNone(handle)
        self.assertEqual(handle.config, config)
        
        # All components work together
        self.assertIsNotNone(processed_item)
        self.assertIsNotNone(handle)


if __name__ == '__main__':
    unittest.main()
