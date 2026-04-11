"""
Tests for embedding provider system.

This module tests the embedding provider abstraction, factory,
and provider implementations.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from infimum.ai.preprocessing.embeddings.base import EmbeddingProvider
from infimum.ai.preprocessing.embeddings.openai_provider import OpenAIEmbeddingProvider
from infimum.ai.preprocessing.embeddings.factory import EmbeddingProviderFactory
from infimum.utils.exceptions import EmbeddingError, ConfigurationError


class TestEmbeddingProvider:
    """Test cases for EmbeddingProvider base class."""
    
    def test_embedding_provider_is_abstract(self):
        """Test that EmbeddingProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EmbeddingProvider()
    
    def test_concrete_provider_implements_interface(self):
        """Test that concrete providers implement required methods."""
        provider = OpenAIEmbeddingProvider(api_key="test_key")
        
        assert hasattr(provider, 'embed')
        assert hasattr(provider, 'get_embedding_dimension')
        assert callable(provider.embed)
        assert callable(provider.get_embedding_dimension)


class TestOpenAIEmbeddingProvider:
    """Test cases for OpenAIEmbeddingProvider."""
    
    def test_initialization(self):
        """Test OpenAIEmbeddingProvider initialization."""
        provider = OpenAIEmbeddingProvider(api_key="test_key")
        
        assert provider.default_model == "text-embedding-3-small"
        assert provider.client is not None
    
    def test_initialization_without_api_key(self):
        """Test that initialization without API key raises error."""
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            OpenAIEmbeddingProvider(api_key="")
    
    def test_initialization_with_custom_model(self):
        """Test initialization with custom default model."""
        provider = OpenAIEmbeddingProvider(
            api_key="test_key",
            default_model="text-embedding-3-large"
        )
        
        assert provider.default_model == "text-embedding-3-large"
    
    @patch('core.ai.preprocessing.embeddings.openai_provider.OpenAI')
    def test_embed_single_text(self, mock_openai_class):
        """Test embedding a single text."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3, 0.4])]
        mock_client.embeddings.create.return_value = mock_response
        
        provider = OpenAIEmbeddingProvider(api_key="test_key")
        result = provider.embed("test text")
        
        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3, 0.4]
        mock_client.embeddings.create.assert_called_once()
    
    @patch('core.ai.preprocessing.embeddings.openai_provider.OpenAI')
    def test_embed_list_of_texts(self, mock_openai_class):
        """Test embedding a list of texts."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2]),
            Mock(embedding=[0.3, 0.4])
        ]
        mock_client.embeddings.create.return_value = mock_response
        
        provider = OpenAIEmbeddingProvider(api_key="test_key")
        result = provider.embed(["text1", "text2"])
        
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]
    
    @patch('core.ai.preprocessing.embeddings.openai_provider.OpenAI')
    def test_embed_with_custom_model(self, mock_openai_class):
        """Test embedding with custom model."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2])]
        mock_client.embeddings.create.return_value = mock_response
        
        provider = OpenAIEmbeddingProvider(api_key="test_key")
        provider.embed("text", model="text-embedding-3-large")
        
        # Check that custom model was used
        call_args = mock_client.embeddings.create.call_args
        assert call_args[1]['model'] == "text-embedding-3-large"
    
    @patch('core.ai.preprocessing.embeddings.openai_provider.OpenAI')
    def test_embed_api_error(self, mock_openai_class):
        """Test that API errors are properly handled."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.embeddings.create.side_effect = Exception("API Error")
        
        provider = OpenAIEmbeddingProvider(api_key="test_key")
        
        with pytest.raises(ValueError, match="Failed to generate embeddings"):
            provider.embed("text")
    
    def test_get_embedding_dimension(self):
        """Test getting embedding dimension for different models."""
        provider = OpenAIEmbeddingProvider(api_key="test_key")
        
        assert provider.get_embedding_dimension("text-embedding-3-small") == 1536
        assert provider.get_embedding_dimension("text-embedding-3-large") == 3072
        assert provider.get_embedding_dimension("text-embedding-ada-002") == 1536
    
    def test_get_embedding_dimension_default_model(self):
        """Test getting dimension for default model."""
        provider = OpenAIEmbeddingProvider(api_key="test_key")
        
        dimension = provider.get_embedding_dimension()
        assert dimension == 1536  # Default model dimension
    
    def test_get_embedding_dimension_unknown_model(self):
        """Test getting dimension for unknown model (should default to 1536)."""
        provider = OpenAIEmbeddingProvider(api_key="test_key")
        
        dimension = provider.get_embedding_dimension("unknown-model")
        assert dimension == 1536  # Default fallback


class TestEmbeddingProviderFactory:
    """Test cases for EmbeddingProviderFactory."""
    
    def setup_method(self):
        """Set up test fixtures."""
        EmbeddingProviderFactory.clear_cache()
        # Clear registered providers (keep OpenAI if already registered)
        if hasattr(EmbeddingProviderFactory, '_providers'):
            # Don't clear, just work with what's there
            pass
    
    def test_register_provider(self):
        """Test registering a provider."""
        class TestProvider(EmbeddingProvider):
            def embed(self, texts, model=None):
                return [[0.1, 0.2]]
            def get_embedding_dimension(self, model=None):
                return 2
        
        EmbeddingProviderFactory.register("test", TestProvider)
        
        assert EmbeddingProviderFactory.is_registered("test")
        assert "test" in EmbeddingProviderFactory.list_providers()
    
    def test_register_invalid_provider(self):
        """Test that registering non-EmbeddingProvider raises error."""
        class NotAProvider:
            pass
        
        with pytest.raises(ValueError, match="must be a subclass of EmbeddingProvider"):
            EmbeddingProviderFactory.register("invalid", NotAProvider)
    
    def test_create_provider(self):
        """Test creating a provider instance."""
        EmbeddingProviderFactory.register("openai", OpenAIEmbeddingProvider)
        
        provider = EmbeddingProviderFactory.create(
            "openai",
            api_key="test_key",
            cache=False
        )
        
        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.default_model == "text-embedding-3-small"
    
    def test_create_unknown_provider(self):
        """Test that creating unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            EmbeddingProviderFactory.create("unknown")
    
    def test_provider_caching(self):
        """Test that provider instances are cached."""
        EmbeddingProviderFactory.register("openai", OpenAIEmbeddingProvider)
        
        provider1 = EmbeddingProviderFactory.create(
            "openai",
            api_key="test_key",
            cache=True
        )
        
        provider2 = EmbeddingProviderFactory.create(
            "openai",
            api_key="test_key",
            cache=True
        )
        
        # Should be the same instance when cached
        assert provider1 is provider2
    
    def test_provider_no_caching(self):
        """Test that caching can be disabled."""
        EmbeddingProviderFactory.register("openai", OpenAIEmbeddingProvider)
        
        provider1 = EmbeddingProviderFactory.create(
            "openai",
            api_key="test_key",
            cache=False
        )
        
        provider2 = EmbeddingProviderFactory.create(
            "openai",
            api_key="test_key",
            cache=False
        )
        
        # Should be different instances when not cached
        assert provider1 is not provider2
    
    def test_clear_cache(self):
        """Test clearing provider cache."""
        EmbeddingProviderFactory.register("openai", OpenAIEmbeddingProvider)
        
        # Create and cache a provider
        provider1 = EmbeddingProviderFactory.create(
            "openai",
            api_key="test_key",
            cache=True
        )
        
        # Clear cache
        EmbeddingProviderFactory.clear_cache()
        
        # New instance should be different
        provider2 = EmbeddingProviderFactory.create(
            "openai",
            api_key="test_key",
            cache=True
        )
        
        assert provider1 is not provider2
    
    def test_list_providers(self):
        """Test listing registered providers."""
        class TestProvider(EmbeddingProvider):
            def embed(self, texts, model=None):
                return [[0.1]]
            def get_embedding_dimension(self, model=None):
                return 1
        
        EmbeddingProviderFactory.register("test", TestProvider)
        
        providers = EmbeddingProviderFactory.list_providers()
        
        assert "test" in providers
        assert "openai" in providers  # Should be registered by default
    
    def test_is_registered(self):
        """Test checking if provider is registered."""
        assert EmbeddingProviderFactory.is_registered("openai")
        assert not EmbeddingProviderFactory.is_registered("nonexistent")
