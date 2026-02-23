"""Unit tests for core.utils.embedding_utils module."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from core.utils.embedding_utils import text_to_embedding
from core.utils.exceptions import EmbeddingError, ConfigurationError
from core.ai.embeddings.factory import EmbeddingProviderFactory


class TestTextToEmbedding:
    """Test cases for text_to_embedding function (using new provider system)."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear provider cache
        EmbeddingProviderFactory.clear_cache()

    @patch('core.utils.embedding_utils.EmbeddingProviderFactory')
    @patch('core.utils.embedding_utils.auto_config')
    def test_text_to_embedding_single_text(self, mock_config, mock_factory):
        """Test text_to_embedding with single text string."""
        mock_config.OPENAI_API_KEY = "test_key"
        mock_config.EMBEDDING_PROVIDER = "openai"
        
        # Mock provider instance
        mock_provider = Mock()
        mock_provider.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_factory.create.return_value = mock_provider
        
        result = text_to_embedding("test text")
        
        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]
        mock_provider.embed.assert_called_once_with("test text", model=None)

    @patch('core.utils.embedding_utils.EmbeddingProviderFactory')
    @patch('core.utils.embedding_utils.auto_config')
    def test_text_to_embedding_list(self, mock_config, mock_factory):
        """Test text_to_embedding with list of texts."""
        mock_config.OPENAI_API_KEY = "test_key"
        mock_config.EMBEDDING_PROVIDER = "openai"
        
        mock_provider = Mock()
        mock_provider.embed.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        mock_factory.create.return_value = mock_provider
        
        result = text_to_embedding(["text1", "text2"])
        
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    @patch('core.utils.embedding_utils.auto_config')
    def test_text_to_embedding_no_api_key(self, mock_config):
        """Test text_to_embedding raises error when API key is missing."""
        mock_config.OPENAI_API_KEY = None
        mock_config.EMBEDDING_PROVIDER = "openai"
        
        with pytest.raises(ConfigurationError, match="OpenAI API key is not set"):
            text_to_embedding("test text")

    @patch('core.utils.embedding_utils.EmbeddingProviderFactory')
    @patch('core.utils.embedding_utils.auto_config')
    def test_text_to_embedding_api_error(self, mock_config, mock_factory):
        """Test text_to_embedding handles API errors."""
        mock_config.OPENAI_API_KEY = "test_key"
        mock_config.EMBEDDING_PROVIDER = "openai"
        
        mock_provider = Mock()
        mock_provider.embed.side_effect = Exception("API Error")
        mock_factory.create.return_value = mock_provider
        
        with pytest.raises(EmbeddingError, match="Failed to generate embeddings"):
            text_to_embedding("test text")

    @patch('core.utils.embedding_utils.EmbeddingProviderFactory')
    @patch('core.utils.embedding_utils.auto_config')
    def test_text_to_embedding_with_custom_provider(self, mock_config, mock_factory):
        """Test text_to_embedding with custom provider."""
        mock_config.EMBEDDING_PROVIDER = "openai"
        
        mock_provider = Mock()
        mock_provider.embed.return_value = [[0.1, 0.2]]
        mock_factory.create.return_value = mock_provider
        
        result = text_to_embedding("text", provider="openai", api_key="custom_key")
        
        assert len(result) == 1
        # Check that factory was called with custom api_key
        mock_factory.create.assert_called_once()
        call_kwargs = mock_factory.create.call_args[1]
        assert call_kwargs.get('api_key') == "custom_key"

    @patch('core.utils.embedding_utils.EmbeddingProviderFactory')
    @patch('core.utils.embedding_utils.auto_config')
    def test_text_to_embedding_with_model(self, mock_config, mock_factory):
        """Test text_to_embedding with custom model."""
        mock_config.OPENAI_API_KEY = "test_key"
        mock_config.EMBEDDING_PROVIDER = "openai"
        
        mock_provider = Mock()
        mock_provider.embed.return_value = [[0.1, 0.2]]
        mock_factory.create.return_value = mock_provider
        
        text_to_embedding("text", model="text-embedding-3-large")
        
        # Check that embed was called with model
        mock_provider.embed.assert_called_once_with("text", model="text-embedding-3-large")
