"""
Integration tests for the re-factored embedding provider chain.

Run with:
    pytest core/tests/ai/embeddings/test_integration.py -v

Coverage
--------
1. BaseProvider inheritance chain
2. ProviderFactory / ProviderRegistry integration
3. Observer / lifecycle events
4. embed() + get_embedding_dimension() API
5. Backward-compatibility (old import path still works, with DeprecationWarning)
"""

import warnings
from unittest.mock import MagicMock, patch

import pytest


def _make_config(model_name: str = "text-embedding-3-small", api_key: str = "sk-test"):
    from infimum.ai.base.providers.config import ModelConfig

    return ModelConfig(
        model_type="embedding",
        provider="openai",
        model_name=model_name,
        model_path="",
        extra_params={"api_key": api_key},
    )


# ------------------------------------------------------------------ #
# 1. Class hierarchy                                                    #
# ------------------------------------------------------------------ #


class TestClassHierarchy:
    """EmbeddingProvider must be in the BaseProvider lineage."""

    def test_embedding_provider_extends_base_provider(self):
        from infimum.ai.base.providers.base import BaseProvider
        from infimum.ai.embeddings.base import EmbeddingProvider

        assert issubclass(EmbeddingProvider, BaseProvider)

    def test_openai_provider_extends_embedding_provider(self):
        from infimum.ai.embeddings.base import EmbeddingProvider
        from infimum.ai.embeddings.providers.openai import OpenAIEmbeddingProvider

        assert issubclass(OpenAIEmbeddingProvider, EmbeddingProvider)

    def test_embedding_provider_is_abstract(self):
        """Cannot instantiate EmbeddingProvider directly."""
        from infimum.ai.embeddings.base import EmbeddingProvider

        with pytest.raises(TypeError):
            EmbeddingProvider()  # type: ignore[abstract]

    def test_model_type_enum_has_embedding(self):
        from infimum.ai.base.providers.config import ModelType

        assert ModelType.EMBEDDING == "embedding"


# ------------------------------------------------------------------ #
# 2. Direct instantiation (backward-compatible path)                  #
# ------------------------------------------------------------------ #


class TestDirectInstantiation:
    """Provider created without the factory — original usage pattern."""

    def test_init_success(self):
        from infimum.ai.embeddings.providers.openai import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(api_key="sk-test")
        assert provider.default_model == "text-embedding-3-small"
        assert provider.client is not None

    def test_init_requires_api_key(self):
        from infimum.ai.embeddings.providers.openai import OpenAIEmbeddingProvider

        with pytest.raises(ValueError, match="API key is required"):
            OpenAIEmbeddingProvider(api_key="")

    def test_custom_default_model(self):
        from infimum.ai.embeddings.providers.openai import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", default_model="text-embedding-3-large"
        )
        assert provider.default_model == "text-embedding-3-large"

    @patch("core.ai.embeddings.providers.openai.OpenAI")
    def test_embed_single_text(self, mock_openai_cls):
        from infimum.ai.embeddings.providers.openai import OpenAIEmbeddingProvider

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
        )

        provider = OpenAIEmbeddingProvider(api_key="sk-test")
        result = provider.embed("hello world")

        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()

    @patch("core.ai.embeddings.providers.openai.OpenAI")
    def test_embed_list_of_texts(self, mock_openai_cls):
        from infimum.ai.embeddings.providers.openai import OpenAIEmbeddingProvider

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = MagicMock(
            data=[
                MagicMock(embedding=[0.1, 0.2]),
                MagicMock(embedding=[0.3, 0.4]),
            ]
        )

        provider = OpenAIEmbeddingProvider(api_key="sk-test")
        result = provider.embed(["text A", "text B"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]

    @patch("core.ai.embeddings.providers.openai.OpenAI")
    def test_embed_raises_on_api_error(self, mock_openai_cls):
        from infimum.ai.embeddings.providers.openai import OpenAIEmbeddingProvider

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.side_effect = RuntimeError("API 500")

        provider = OpenAIEmbeddingProvider(api_key="sk-test")
        with pytest.raises(ValueError, match="Failed to generate embeddings"):
            provider.embed("boom")

    def test_get_embedding_dimension_known_models(self):
        from infimum.ai.embeddings.providers.openai import OpenAIEmbeddingProvider

        p = OpenAIEmbeddingProvider(api_key="sk-test")
        assert p.get_embedding_dimension("text-embedding-3-small") == 1536
        assert p.get_embedding_dimension("text-embedding-3-large") == 3072
        assert p.get_embedding_dimension("text-embedding-ada-002") == 1536

    def test_get_embedding_dimension_unknown_falls_back(self):
        from infimum.ai.embeddings.providers.openai import OpenAIEmbeddingProvider

        p = OpenAIEmbeddingProvider(api_key="sk-test")
        assert p.get_embedding_dimension("totally-unknown-model") == 1536


# ------------------------------------------------------------------ #
# 3. ProviderFactory integration                                       #
# ------------------------------------------------------------------ #


class TestProviderFactory:
    """Embedding providers must be creatable via ProviderFactory."""

    def test_openai_is_registered(self):
        from infimum.ai.base.providers.factory import ProviderFactory

        # Force import so the auto-registration code at module-level runs
        import infimum.ai.embeddings.providers.openai  # noqa: F401

        assert ProviderFactory.is_registered("embedding", "openai")

    def test_factory_create_returns_openai_provider(self):
        from infimum.ai.base.providers.factory import ProviderFactory
        from infimum.ai.embeddings.providers.openai import OpenAIEmbeddingProvider

        import infimum.ai.embeddings.providers.openai  # noqa: F401

        config = _make_config()
        provider = ProviderFactory.create("embedding", "openai", config)
        assert isinstance(provider, OpenAIEmbeddingProvider)

    def test_factory_unknown_type_raises(self):
        from infimum.ai.base.providers.factory import ProviderFactory

        config = _make_config()
        with pytest.raises(ValueError, match="not registered"):
            ProviderFactory.create("embedding", "nonexistent", config)

    def test_list_embedding_providers(self):
        from infimum.ai.base.providers.factory import ProviderFactory

        import infimum.ai.embeddings.providers.openai  # noqa: F401

        providers = ProviderFactory.list_providers("embedding")
        assert "openai" in providers


# ------------------------------------------------------------------ #
# 4. ProviderRegistry integration                                      #
# ------------------------------------------------------------------ #


class TestProviderRegistry:
    def test_openai_registered_in_registry(self):
        from infimum.ai.base.providers.registry import ProviderRegistry

        import infimum.ai.embeddings.providers.openai  # noqa: F401

        meta = ProviderRegistry.get("embedding-openai")
        assert meta is not None
        assert meta.model_type == "embedding"
        assert meta.provider_name == "openai"
        assert "embed" in meta.capabilities

    def test_search_by_model_type(self):
        from infimum.ai.base.providers.registry import ProviderRegistry

        import infimum.ai.embeddings.providers.openai  # noqa: F401

        results = ProviderRegistry.search(model_type="embedding")
        assert "embedding-openai" in results

    def test_get_by_type(self):
        from infimum.ai.base.providers.registry import ProviderRegistry

        import infimum.ai.embeddings.providers.openai  # noqa: F401

        providers = ProviderRegistry.get_by_type("embedding")
        assert any(m.provider_name == "openai" for m in providers)


# ------------------------------------------------------------------ #
# 5. Observer / lifecycle events                                       #
# ------------------------------------------------------------------ #


class TestObserverEvents:
    """The shared observer chain must fire for embedding operations."""

    @patch("core.ai.embeddings.providers.openai.OpenAI")
    def test_embed_fires_inference_events(self, mock_openai_cls):
        from infimum.ai.base.observers.base import Observer
        from infimum.ai.base.observers.events import EventType
        from infimum.ai.embeddings.providers.openai import OpenAIEmbeddingProvider

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1, 0.2])]
        )

        received: list = []

        class SpyObserver(Observer):
            def on_event(self, event):
                received.append(event.type)

        provider = OpenAIEmbeddingProvider(api_key="sk-test")
        provider.attach(SpyObserver())
        provider.embed("test text")

        assert EventType.INFERENCE_STARTED in received
        assert EventType.INFERENCE_COMPLETED in received

    @patch("core.ai.embeddings.providers.openai.OpenAI")
    def test_embed_fires_failed_event_on_error(self, mock_openai_cls):
        from infimum.ai.base.observers.base import Observer
        from infimum.ai.base.observers.events import EventType
        from infimum.ai.embeddings.providers.openai import OpenAIEmbeddingProvider

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.embeddings.create.side_effect = RuntimeError("timeout")

        received: list = []

        class SpyObserver(Observer):
            def on_event(self, event):
                received.append(event.type)

        provider = OpenAIEmbeddingProvider(api_key="sk-test")
        provider.attach(SpyObserver())

        with pytest.raises(ValueError):
            provider.embed("boom")

        assert EventType.INFERENCE_FAILED in received

    @patch("core.ai.embeddings.providers.openai.OpenAI")
    def test_load_model_fires_loading_events(self, mock_openai_cls):
        """BaseProvider.get_model() must emit MODEL_LOADING_STARTED/COMPLETED."""
        from infimum.ai.base.observers.base import Observer
        from infimum.ai.base.observers.events import EventType
        from infimum.ai.embeddings.providers.openai import OpenAIEmbeddingProvider

        mock_openai_cls.return_value = MagicMock()

        received: list = []

        class SpyObserver(Observer):
            def on_event(self, event):
                received.append(event.type)

        config = _make_config()
        provider = OpenAIEmbeddingProvider(api_key="sk-test", config=config)
        provider.attach(SpyObserver())
        provider.get_model(config)

        assert EventType.MODEL_LOADING_STARTED in received
        assert EventType.MODEL_LOADING_COMPLETED in received


# ------------------------------------------------------------------ #
# 6. Backward-compatibility (old import path)                          #
# ------------------------------------------------------------------ #


class TestBackwardCompatibility:
    """Existing code importing from preprocessing.embeddings must still work."""

    def test_old_base_import_still_works_with_warning(self):
        import importlib
        import sys

        # Remove from cache so the module-level warning fires again
        sys.modules.pop("core.ai.preprocessing.embeddings.base", None)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import infimum.ai.preprocessing.embeddings.base  # noqa: F401
            importlib.reload(core.ai.preprocessing.embeddings.base)

            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    def test_old_openai_import_still_works_with_warning(self):
        import importlib
        import sys

        sys.modules.pop("core.ai.preprocessing.embeddings.openai_provider", None)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import infimum.ai.preprocessing.embeddings.openai_provider  # noqa: F401
            importlib.reload(core.ai.preprocessing.embeddings.openai_provider)

            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    def test_old_factory_create_still_works(self):
        """EmbeddingProviderFactory.create() must remain functional."""
        from infimum.ai.preprocessing.embeddings.factory import EmbeddingProviderFactory
        from infimum.ai.embeddings.providers.openai import OpenAIEmbeddingProvider

        EmbeddingProviderFactory.register("openai", OpenAIEmbeddingProvider)
        provider = EmbeddingProviderFactory.create("openai", api_key="sk-test", cache=False)
        assert isinstance(provider, OpenAIEmbeddingProvider)
