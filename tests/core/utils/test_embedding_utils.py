"""Unit tests for core.utils.embedding_utils module."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from core.utils.embedding_utils import text_to_embedding


class TestTextToEmbedding:
    """Test cases for text_to_embedding function."""

    @patch('core.utils.embedding_utils.openai_client')
    @patch('core.utils.embedding_utils.auto_config')
    def test_text_to_embedding_single_text(self, mock_config, mock_client):
        """Test text_to_embedding with single text string."""
        mock_config.OPENAI_API_KEY = "test_key"
        mock_config.OPENAI_TEXT_EMBEDDING_MODEL = "text-embedding-ada-002"
        
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        
        result = text_to_embedding("test text")
        
        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]
        mock_client.embeddings.create.assert_called_once()

    @patch('core.utils.embedding_utils.openai_client')
    @patch('core.utils.embedding_utils.auto_config')
    def test_text_to_embedding_list(self, mock_config, mock_client):
        """Test text_to_embedding with list of texts."""
        mock_config.OPENAI_API_KEY = "test_key"
        mock_config.OPENAI_TEXT_EMBEDDING_MODEL = "text-embedding-ada-002"
        
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_client.embeddings.create.return_value = mock_response
        
        result = text_to_embedding(["text1", "text2"])
        
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    @patch('core.utils.embedding_utils.auto_config')
    def test_text_to_embedding_no_api_key(self, mock_config):
        """Test text_to_embedding raises error when API key is missing."""
        mock_config.OPENAI_API_KEY = None
        
        with pytest.raises(ValueError) as exc_info:
            text_to_embedding("test text")
        
        assert "OpenAI API key is not set" in str(exc_info.value)

    @patch('core.utils.embedding_utils.openai_client')
    @patch('core.utils.embedding_utils.auto_config')
    def test_text_to_embedding_api_error(self, mock_config, mock_client):
        """Test text_to_embedding handles API errors."""
        mock_config.OPENAI_API_KEY = "test_key"
        mock_config.OPENAI_TEXT_EMBEDDING_MODEL = "text-embedding-ada-002"
        mock_client.embeddings.create.side_effect = Exception("API Error")
        
        with pytest.raises(ValueError) as exc_info:
            text_to_embedding("test text")
        
        assert "Failed to generate embeddings" in str(exc_info.value)
