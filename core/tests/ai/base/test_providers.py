"""
Unit tests for provider system.

Tests Factory, Registry, Strategy, Template Method, and Builder patterns.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from core.ai.base.providers.base import BaseProvider, ModelHandle
from core.ai.base.providers.config import ModelConfig, ModelConfigBuilder, ModelType
from core.ai.base.providers.factory import ProviderFactory
from core.ai.base.providers.registry import ProviderRegistry, ProviderMetadata
from core.ai.base.observers.base import Observer
from core.ai.base.observers.events import Event, EventType


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig dataclass."""
    
    def test_model_config_creation(self):
        """Test creating a ModelConfig."""
        config = ModelConfig(
            model_type="llm",
            provider="huggingface",
            model_path="/path/to/model"
        )
        self.assertEqual(config.model_type, "llm")
        self.assertEqual(config.provider, "huggingface")
        self.assertEqual(config.model_path, "/path/to/model")
    
    def test_model_config_validation(self):
        """Test ModelConfig validation."""
        # Valid config
        config = ModelConfig(
            model_type="llm",
            provider="huggingface",
            model_path="/path/to/model"
        )
        config.validate()  # Should not raise
        
        # Invalid configs
        with self.assertRaises(ValueError):
            config = ModelConfig(model_type="", provider="hf", model_path="/path")
            config.validate()
        
        with self.assertRaises(ValueError):
            config = ModelConfig(model_type="llm", provider="", model_path="/path")
            config.validate()
        
        with self.assertRaises(ValueError):
            config = ModelConfig(model_type="llm", provider="hf", model_path="")
            config.validate()
    
    def test_model_config_hashable(self):
        """Test that ModelConfig is hashable."""
        config1 = ModelConfig(
            model_type="llm",
            provider="hf",
            model_path="/path"
        )
        config2 = ModelConfig(
            model_type="llm",
            provider="hf",
            model_path="/path"
        )
        self.assertEqual(hash(config1), hash(config2))
        
        # Different configs should have different hashes
        config3 = ModelConfig(
            model_type="vlm",
            provider="hf",
            model_path="/path"
        )
        self.assertNotEqual(hash(config1), hash(config3))


class TestModelConfigBuilder(unittest.TestCase):
    """Test ModelConfigBuilder (Builder pattern)."""
    
    def test_builder_fluent_interface(self):
        """Test builder's fluent interface."""
        config = (ModelConfigBuilder()
            .with_model_type("llm")
            .with_provider("huggingface")
            .with_model_path("/path/to/model")
            .with_device("cuda")
            .with_quantization(4)
            .with_temperature(0.7)
            .build())
        
        self.assertEqual(config.model_type, "llm")
        self.assertEqual(config.provider, "huggingface")
        self.assertEqual(config.model_path, "/path/to/model")
        self.assertEqual(config.device, "cuda")
        self.assertTrue(config.load_4bit)
        self.assertFalse(config.load_8bit)
        self.assertEqual(config.temperature, 0.7)
    
    def test_builder_quantization(self):
        """Test quantization settings."""
        # 8-bit quantization
        config = (ModelConfigBuilder()
            .with_model_type("llm")
            .with_provider("hf")
            .with_model_path("/path")
            .with_quantization(8)
            .build())
        self.assertTrue(config.load_8bit)
        self.assertFalse(config.load_4bit)
        
        # 4-bit quantization
        config = (ModelConfigBuilder()
            .with_model_type("llm")
            .with_provider("hf")
            .with_model_path("/path")
            .with_quantization(4)
            .build())
        self.assertTrue(config.load_4bit)
        self.assertFalse(config.load_8bit)
        
        # Invalid quantization
        with self.assertRaises(ValueError):
            ModelConfigBuilder().with_quantization(16)
    
    def test_builder_extra_params(self):
        """Test adding extra parameters."""
        config = (ModelConfigBuilder()
            .with_model_type("llm")
            .with_provider("hf")
            .with_model_path("/path")
            .with_extra_param("custom_param", "value")
            .with_extra_param("another_param", 123)
            .build())
        
        self.assertEqual(config.extra_params["custom_param"], "value")
        self.assertEqual(config.extra_params["another_param"], 123)
    
    def test_builder_validation(self):
        """Test that builder validates on build."""
        with self.assertRaises(ValueError):
            ModelConfigBuilder().build()  # Missing required fields


class TestProviderRegistry(unittest.TestCase):
    """Test ProviderRegistry (Registry pattern)."""
    
    def setUp(self):
        """Clear registry before each test."""
        ProviderRegistry.clear()
    
    def test_register_and_get(self):
        """Test registering and retrieving providers."""
        metadata = ProviderMetadata(
            model_type="llm",
            provider_name="test_provider",
            capabilities={"chat", "completion"}
        )
        ProviderRegistry.register("test-llm", metadata)
        
        retrieved = ProviderRegistry.get("test-llm")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.model_type, "llm")
        self.assertEqual(retrieved.provider_name, "test_provider")
    
    def test_list_all(self):
        """Test listing all providers."""
        ProviderRegistry.register("p1", ProviderMetadata("llm", "p1", {"chat"}))
        ProviderRegistry.register("p2", ProviderMetadata("vlm", "p2", {"inference"}))
        
        all_providers = ProviderRegistry.list_all()
        self.assertEqual(len(all_providers), 2)
        self.assertIn("p1", all_providers)
        self.assertIn("p2", all_providers)
    
    def test_search_by_type(self):
        """Test searching providers by type."""
        ProviderRegistry.register("llm1", ProviderMetadata("llm", "llm1", {"chat"}))
        ProviderRegistry.register("llm2", ProviderMetadata("llm", "llm2", {"completion"}))
        ProviderRegistry.register("vlm1", ProviderMetadata("vlm", "vlm1", {"inference"}))
        
        llm_providers = ProviderRegistry.search(model_type="llm")
        self.assertEqual(len(llm_providers), 2)
        self.assertIn("llm1", llm_providers)
        self.assertIn("llm2", llm_providers)
    
    def test_search_by_capabilities(self):
        """Test searching providers by capabilities."""
        ProviderRegistry.register("p1", ProviderMetadata("llm", "p1", {"chat", "completion"}))
        ProviderRegistry.register("p2", ProviderMetadata("llm", "p2", {"chat"}))
        ProviderRegistry.register("p3", ProviderMetadata("llm", "p3", {"completion"}))
        
        # Search for providers with both capabilities
        results = ProviderRegistry.search(model_type="llm", capabilities=["chat", "completion"])
        self.assertEqual(len(results), 1)
        self.assertIn("p1", results)
        
        # Search for providers with one capability
        results = ProviderRegistry.search(model_type="llm", capabilities=["chat"])
        self.assertEqual(len(results), 2)
        self.assertIn("p1", results)
        self.assertIn("p2", results)
    
    def test_get_by_type(self):
        """Test getting providers by type."""
        ProviderRegistry.register("llm1", ProviderMetadata("llm", "llm1", {"chat"}))
        ProviderRegistry.register("vlm1", ProviderMetadata("vlm", "vlm1", {"inference"}))
        
        llm_metadata = ProviderRegistry.get_by_type("llm")
        self.assertEqual(len(llm_metadata), 1)
        self.assertEqual(llm_metadata[0].provider_name, "llm1")


class TestProviderFactory(unittest.TestCase):
    """Test ProviderFactory (Factory pattern)."""
    
    def setUp(self):
        """Set up test provider."""
        # Create a mock provider class
        class MockProvider(BaseProvider):
            def load_model(self, config):
                return ModelHandle(model="mock_model", config=config)
            
            def unload_model(self, handle):
                pass
        
        self.MockProvider = MockProvider
        ProviderFactory._registry.clear()
    
    def test_register_and_create(self):
        """Test registering and creating providers."""
        ProviderFactory.register("llm", "mock", self.MockProvider)
        
        config = ModelConfig(
            model_type="llm",
            provider="mock",
            model_path="/path"
        )
        provider = ProviderFactory.create("llm", "mock", config)
        
        self.assertIsInstance(provider, self.MockProvider)
        self.assertEqual(provider.config, config)
    
    def test_create_unregistered_provider(self):
        """Test creating an unregistered provider raises error."""
        config = ModelConfig(
            model_type="llm",
            provider="nonexistent",
            model_path="/path"
        )
        
        with self.assertRaises(ValueError) as cm:
            ProviderFactory.create("llm", "nonexistent", config)
        
        self.assertIn("not registered", str(cm.exception))
    
    def test_list_providers(self):
        """Test listing providers."""
        ProviderFactory.register("llm", "p1", self.MockProvider)
        ProviderFactory.register("llm", "p2", self.MockProvider)
        ProviderFactory.register("vlm", "p3", self.MockProvider)
        
        all_providers = ProviderFactory.list_providers()
        self.assertEqual(len(all_providers), 3)
        
        llm_providers = ProviderFactory.list_providers("llm")
        self.assertEqual(len(llm_providers), 2)
        self.assertIn("p1", llm_providers)
        self.assertIn("p2", llm_providers)
    
    def test_is_registered(self):
        """Test checking if provider is registered."""
        ProviderFactory.register("llm", "test", self.MockProvider)
        
        self.assertTrue(ProviderFactory.is_registered("llm", "test"))
        self.assertFalse(ProviderFactory.is_registered("llm", "nonexistent"))
    
    def test_unregister(self):
        """Test unregistering a provider."""
        ProviderFactory.register("llm", "test", self.MockProvider)
        self.assertTrue(ProviderFactory.is_registered("llm", "test"))
        
        ProviderFactory.unregister("llm", "test")
        self.assertFalse(ProviderFactory.is_registered("llm", "test"))


class TestBaseProvider(unittest.TestCase):
    """Test BaseProvider (Strategy + Template Method pattern)."""
    
    def setUp(self):
        """Set up test provider implementation."""
        class TestProvider(BaseProvider):
            def __init__(self, config=None):
                super().__init__(config)
                self._cache_dict = {}
            
            def load_model(self, config):
                return ModelHandle(
                    model="test_model",
                    config=config,
                    metadata={"loaded": True}
                )
            
            def unload_model(self, handle):
                pass
            
            def _is_cached(self, config):
                return config in self._cache_dict
            
            def _get_cached(self, config):
                return self._cache_dict[config]
            
            def _cache(self, config, handle):
                self._cache_dict[config] = handle
        
        self.TestProvider = TestProvider
    
    def test_load_model_abstract(self):
        """Test that BaseProvider cannot be instantiated."""
        with self.assertRaises(TypeError):
            BaseProvider()
    
    def test_template_method_workflow(self):
        """Test template method workflow."""
        config = ModelConfig(
            model_type="llm",
            provider="test",
            model_path="/path"
        )
        provider = self.TestProvider()
        
        # First load - should load and cache
        handle1 = provider.get_model(config)
        self.assertEqual(handle1.model, "test_model")
        self.assertTrue(handle1.metadata["loaded"])
        
        # Second load - should use cache
        handle2 = provider.get_model(config)
        self.assertEqual(handle1, handle2)  # Same handle from cache
    
    def test_validation_in_template_method(self):
        """Test that validation is called in template method."""
        config = ModelConfig(
            model_type="llm",
            provider="test",
            model_path=""  # Invalid - empty path
        )
        provider = self.TestProvider()
        
        with self.assertRaises(ValueError):
            provider.get_model(config)
    
    def test_observer_notifications(self):
        """Test that observers are notified during model loading."""
        class TestObserver(Observer):
            def __init__(self):
                self.events = []
            
            def on_event(self, event):
                self.events.append(event)
        
        config = ModelConfig(
            model_type="llm",
            provider="test",
            model_path="/path"
        )
        provider = self.TestProvider()
        observer = TestObserver()
        provider.attach(observer)
        
        provider.get_model(config)
        
        # Should have received events
        self.assertGreater(len(observer.events), 0)
        event_types = [e.type for e in observer.events]
        self.assertIn(EventType.MODEL_LOADING_STARTED, event_types)
        self.assertIn(EventType.MODEL_LOADING_COMPLETED, event_types)
    
    def test_post_process_hook(self):
        """Test post-process hook."""
        class PostProcessProvider(self.TestProvider):
            def _post_process(self, handle):
                handle.metadata["post_processed"] = True
                return handle
        
        config = ModelConfig(
            model_type="llm",
            provider="test",
            model_path="/path"
        )
        provider = PostProcessProvider()
        handle = provider.get_model(config)
        
        self.assertTrue(handle.metadata.get("post_processed", False))
    
    def test_get_model_info(self):
        """Test getting model info."""
        config = ModelConfig(
            model_type="llm",
            provider="test",
            model_path="/path"
        )
        provider = self.TestProvider()
        handle = provider.get_model(config)
        
        info = provider.get_model_info(handle)
        self.assertEqual(info["config"], config)
        self.assertIn("metadata", info)


class TestModelHandle(unittest.TestCase):
    """Test ModelHandle."""
    
    def test_model_handle_creation(self):
        """Test creating a ModelHandle."""
        config = ModelConfig(
            model_type="llm",
            provider="test",
            model_path="/path"
        )
        handle = ModelHandle(
            model={"tokenizer": "tok", "model": "mod"},
            config=config,
            metadata={"test": "value"}
        )
        
        self.assertEqual(handle.get("tokenizer"), "tok")
        self.assertEqual(handle.get("model"), "mod")
        self.assertEqual(handle.metadata["test"], "value")
    
    def test_model_handle_get_with_default(self):
        """Test ModelHandle.get() with default value."""
        config = ModelConfig(
            model_type="llm",
            provider="test",
            model_path="/path"
        )
        handle = ModelHandle(
            model={"tokenizer": "tok"},
            config=config
        )
        
        self.assertEqual(handle.get("nonexistent", "default"), "default")


if __name__ == '__main__':
    unittest.main()
