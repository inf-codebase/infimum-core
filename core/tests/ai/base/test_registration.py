"""
Unit tests for unified registration system.

Tests that register_loader, register_provider, and register_transform
update both Factory and Registry in a single call.
"""

import unittest
import warnings

import sys
import types
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# The top-level ai/__init__.py eagerly imports from ai.llm, ai.speech, etc.
# which depend on modules that may not be available. We stub them all out so
# that importing ai.base.* (which is self-contained) works correctly.
_stub_modules = [
    # LLM tree
    "ai.llm",
    "ai.llm.agent",
    "ai.llm.tools",
    "ai.llm.tools.web_search",
    "ai.llm.tools.calculator",
    "ai.llm.tools.weather",
    "ai.llm.tools.time",
    "ai.llm.planner",
    "ai.llm.memory",
    # Speech tree
    "ai.speech",
    "ai.speech.models",
    "ai.speech.models.speech2text",
    "ai.speech.models.text2speech",
    # Concrete providers / data / preprocessing (outside base)
    "ai.providers",
    "ai.providers.speech",
    "ai.providers.speech.whisper_provider",
    "ai.providers.speech.medasr_provider",
    "ai.data",
    "ai.data.loaders",
    "ai.preprocessing",
    "ai.preprocessing.transforms",
]
for _name in _stub_modules:
    if _name not in sys.modules:
        _mod = types.ModuleType(_name)
        _mod.__path__ = []  # type: ignore[attr-defined]
        sys.modules[_name] = _mod

# Set attributes that ai/__init__.py tries to import from ai.llm
for _attr in [
    "Agent",
    "ToolManager",
    "Tool",
    "WebSearchTool",
    "CalculatorTool",
    "WeatherTool",
    "TimeTool",
    "Planner",
    "Memory",
    "AGENT_PROMPTS",
    "PLANNER_PROMPTS",
    "MEMORY_PROMPTS",
]:
    setattr(sys.modules["ai.llm"], _attr, None)

# Set attributes that ai/__init__.py tries to import from ai.speech.*
for _attr in ["Speech2Text", "Text2Speech"]:
    setattr(sys.modules["ai.speech.models.speech2text"], _attr, type(_attr, (), {}))
    setattr(sys.modules["ai.speech.models.text2speech"], _attr, type(_attr, (), {}))

# Now the actual imports through the proper package hierarchy
from ai.base.data.base import BaseLoader  # noqa: E402
from ai.base.data.item import DataItem  # noqa: E402
from ai.base.data.factory import LoaderFactory  # noqa: E402
from ai.base.data.registry import LoaderRegistry, LoaderMetadata  # noqa: E402
from ai.base.data.registration import register_loader, unregister_loader  # noqa: E402

from ai.base.providers.base import BaseProvider, ModelHandle  # noqa: E402
from ai.base.providers.config import ModelConfig  # noqa: E402
from ai.base.providers.factory import ProviderFactory  # noqa: E402
from ai.base.providers.registry import ProviderRegistry, ProviderMetadata  # noqa: E402
from ai.base.providers.registration import register_provider, unregister_provider  # noqa: E402

from ai.base.preprocessing.base import BaseTransform  # noqa: E402
from ai.base.preprocessing.factory import TransformFactory  # noqa: E402
from ai.base.preprocessing.registry import TransformRegistry, TransformMetadata  # noqa: E402
from ai.base.preprocessing.registration import register_transform, unregister_transform  # noqa: E402


# --- Test helpers ---


class _StubLoader(BaseLoader):
    def _load(self, source):
        return DataItem(data=source, data_type="test")


class _StubProvider(BaseProvider):
    def load_model(self, config):
        return ModelHandle(model="mock", config=config)

    def unload_model(self, handle):
        pass


class _StubTransform(BaseTransform):
    def transform(self, data):
        data.metadata["transformed"] = True
        return data


# --- Loader tests ---


class TestUnifiedLoaderRegistration(unittest.TestCase):
    """Test that register_loader writes to both Factory and Registry."""

    def setUp(self):
        LoaderFactory._registry.clear()
        LoaderRegistry._loaders.clear()

    def test_register_populates_both(self):
        """register_loader should populate Factory AND Registry."""
        meta = LoaderMetadata(
            data_type="image",
            loader_name="test_img",
            supported_formats={"jpg", "png"},
        )
        register_loader("test_img", _StubLoader, meta)

        # Factory side
        self.assertTrue(LoaderFactory.is_registered("test_img"))
        loader = LoaderFactory.create("test_img")
        self.assertIsInstance(loader, _StubLoader)

        # Registry side
        retrieved = LoaderRegistry.get("test_img")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.data_type, "image")
        self.assertIn("jpg", retrieved.supported_formats)

    def test_unregister_removes_both(self):
        """unregister_loader should remove from Factory AND Registry."""
        meta = LoaderMetadata(
            data_type="text",
            loader_name="txt",
            supported_formats={"txt"},
        )
        register_loader("txt", _StubLoader, meta)

        unregister_loader("txt")
        self.assertFalse(LoaderFactory.is_registered("txt"))
        self.assertIsNone(LoaderRegistry.get("txt"))

    def test_search_after_register(self):
        """Registry search should find loaders added via register_loader."""
        register_loader(
            "img1",
            _StubLoader,
            LoaderMetadata("image", "img1", {"jpg"}),
        )
        register_loader(
            "txt1",
            _StubLoader,
            LoaderMetadata("text", "txt1", {"txt"}),
        )
        results = LoaderRegistry.search(data_type="image")
        self.assertIn("img1", results)
        self.assertNotIn("txt1", results)

    def test_no_deprecation_warning(self):
        """register_loader should NOT trigger a deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            register_loader(
                "no_warn",
                _StubLoader,
                LoaderMetadata("test", "no_warn", {"any"}),
            )
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        self.assertEqual(len(dep_warnings), 0)


# --- Provider tests ---


class TestUnifiedProviderRegistration(unittest.TestCase):
    """Test that register_provider writes to both Factory and Registry."""

    def setUp(self):
        ProviderFactory._registry.clear()
        ProviderRegistry._providers.clear()

    def test_register_populates_both(self):
        """register_provider should populate Factory AND Registry."""
        meta = ProviderMetadata(
            model_type="llm",
            provider_name="mock",
            capabilities={"chat", "completion"},
        )
        register_provider("llm", "mock", _StubProvider, meta)

        # Factory side
        self.assertTrue(ProviderFactory.is_registered("llm", "mock"))
        config = ModelConfig(model_type="llm", provider="mock", model_path="/p")
        provider = ProviderFactory.create("llm", "mock", config)
        self.assertIsInstance(provider, _StubProvider)

        # Registry side
        retrieved = ProviderRegistry.get("llm-mock")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.model_type, "llm")
        self.assertIn("chat", retrieved.capabilities)

    def test_unregister_removes_both(self):
        """unregister_provider should remove from Factory AND Registry."""
        meta = ProviderMetadata(
            model_type="vlm",
            provider_name="test",
            capabilities={"inference"},
        )
        register_provider("vlm", "test", _StubProvider, meta)

        unregister_provider("vlm", "test")
        self.assertFalse(ProviderFactory.is_registered("vlm", "test"))
        self.assertIsNone(ProviderRegistry.get("vlm-test"))

    def test_search_after_register(self):
        """Registry search should find providers added via register_provider."""
        register_provider(
            "llm",
            "p1",
            _StubProvider,
            ProviderMetadata("llm", "p1", {"chat"}),
        )
        register_provider(
            "vlm",
            "p2",
            _StubProvider,
            ProviderMetadata("vlm", "p2", {"inference"}),
        )
        results = ProviderRegistry.search(model_type="llm")
        self.assertIn("llm-p1", results)
        self.assertNotIn("vlm-p2", results)

    def test_no_deprecation_warning(self):
        """register_provider should NOT trigger a deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            register_provider(
                "llm",
                "no_warn",
                _StubProvider,
                ProviderMetadata("llm", "no_warn", {"chat"}),
            )
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        self.assertEqual(len(dep_warnings), 0)


# --- Transform tests ---


class TestUnifiedTransformRegistration(unittest.TestCase):
    """Test that register_transform writes to both Factory and Registry."""

    def setUp(self):
        TransformFactory._registry.clear()
        TransformRegistry._transforms.clear()

    def test_register_populates_both(self):
        """register_transform should populate Factory AND Registry."""
        meta = TransformMetadata(
            transform_name="test_t",
            data_type="image",
            description="Stub transform",
        )
        register_transform("test_t", _StubTransform, meta)

        # Factory side
        self.assertTrue(TransformFactory.is_registered("test_t"))
        t = TransformFactory.create("test_t")
        self.assertIsInstance(t, _StubTransform)

        # Registry side
        retrieved = TransformRegistry.get("test_t")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.data_type, "image")

    def test_unregister_removes_both(self):
        """unregister_transform should remove from Factory AND Registry."""
        meta = TransformMetadata(
            transform_name="rm_t",
            data_type="text",
        )
        register_transform("rm_t", _StubTransform, meta)

        unregister_transform("rm_t")
        self.assertFalse(TransformFactory.is_registered("rm_t"))
        self.assertIsNone(TransformRegistry.get("rm_t"))

    def test_search_after_register(self):
        """Registry search should find transforms added via register_transform."""
        register_transform(
            "img_t",
            _StubTransform,
            TransformMetadata("img_t", "image"),
        )
        register_transform(
            "txt_t",
            _StubTransform,
            TransformMetadata("txt_t", "text"),
        )
        results = TransformRegistry.search(data_type="image")
        self.assertIn("img_t", results)
        self.assertNotIn("txt_t", results)

    def test_no_deprecation_warning(self):
        """register_transform should NOT trigger a deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            register_transform(
                "no_warn",
                _StubTransform,
                TransformMetadata("no_warn", "test"),
            )
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        self.assertEqual(len(dep_warnings), 0)


# --- Deprecation warning tests ---


class TestDeprecationWarnings(unittest.TestCase):
    """Test that old standalone register methods emit deprecation warnings."""

    def setUp(self):
        LoaderFactory._registry.clear()
        LoaderRegistry._loaders.clear()
        ProviderFactory._registry.clear()
        ProviderRegistry._providers.clear()
        TransformFactory._registry.clear()
        TransformRegistry._transforms.clear()

    def test_loader_factory_warns(self):
        """LoaderFactory.register() should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LoaderFactory.register("x", _StubLoader)
        dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
        self.assertEqual(len(dep), 1)
        self.assertIn("register_loader", str(dep[0].message))

    def test_loader_registry_warns(self):
        """LoaderRegistry.register() should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LoaderRegistry.register("x", LoaderMetadata("t", "x", {"a"}))
        dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
        self.assertEqual(len(dep), 1)
        self.assertIn("register_loader", str(dep[0].message))

    def test_provider_factory_warns(self):
        """ProviderFactory.register() should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ProviderFactory.register("llm", "x", _StubProvider)
        dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
        self.assertEqual(len(dep), 1)
        self.assertIn("register_provider", str(dep[0].message))

    def test_provider_registry_warns(self):
        """ProviderRegistry.register() should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ProviderRegistry.register("x", ProviderMetadata("llm", "x", {"c"}))
        dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
        self.assertEqual(len(dep), 1)
        self.assertIn("register_provider", str(dep[0].message))

    def test_transform_factory_warns(self):
        """TransformFactory.register() should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TransformFactory.register("x", _StubTransform)
        dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
        self.assertEqual(len(dep), 1)
        self.assertIn("register_transform", str(dep[0].message))

    def test_transform_registry_warns(self):
        """TransformRegistry.register() should emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            TransformRegistry.register("x", TransformMetadata("x", "t"))
        dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
        self.assertEqual(len(dep), 1)
        self.assertIn("register_transform", str(dep[0].message))


if __name__ == "__main__":
    unittest.main()
