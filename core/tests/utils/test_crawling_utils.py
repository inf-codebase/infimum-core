"""Unit tests for core.utils.crawling_utils module."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from core.utils.crawling_utils import (
    remove_trailing_slash,
    normalize_url_data,
    parse_proxy_url,
    jina_read_content
)
from urllib.parse import urlparse


class TestRemoveTrailingSlash:
    """Test cases for remove_trailing_slash function."""

    def test_remove_trailing_slash_with_slash(self):
        """Test remove_trailing_slash removes trailing slash."""
        result = remove_trailing_slash("https://example.com/")
        assert result == "https://example.com"

    def test_remove_trailing_slash_without_slash(self):
        """Test remove_trailing_slash doesn't change URL without slash."""
        result = remove_trailing_slash("https://example.com")
        assert result == "https://example.com"

    def test_remove_trailing_slash_multiple_slashes(self):
        """Test remove_trailing_slash removes only trailing slash."""
        result = remove_trailing_slash("https://example.com/path/")
        assert result == "https://example.com/path"


class TestNormalizeUrlData:
    """Test cases for normalize_url_data function."""

    def test_normalize_url_data(self):
        """Test normalize_url_data normalizes URL."""
        url_data = {
            "url": "https://example.com/ ",
            "other_field": "value"
        }
        
        result = normalize_url_data(url_data)
        
        assert result["url"] == "https://example.com"
        assert result["other_field"] == "value"

    def test_normalize_url_data_with_trailing_slash(self):
        """Test normalize_url_data removes trailing slash."""
        url_data = {
            "url": "https://example.com/path/"
        }
        
        result = normalize_url_data(url_data)
        
        assert result["url"] == "https://example.com/path"


class TestParseProxyUrl:
    """Test cases for parse_proxy_url function."""

    def test_parse_proxy_url_simple(self):
        """Test parse_proxy_url with simple proxy URL."""
        result = parse_proxy_url("http://proxy.example.com:8080")
        
        assert result["server"] == "http://proxy.example.com:8080"
        assert "username" not in result

    def test_parse_proxy_url_with_auth(self):
        """Test parse_proxy_url with authentication."""
        result = parse_proxy_url("http://user:pass@proxy.example.com:8080")
        
        assert result["server"] == "http://proxy.example.com:8080"
        assert result["username"] == "user"
        assert result["password"] == "pass"

    def test_parse_proxy_url_https(self):
        """Test parse_proxy_url with HTTPS proxy."""
        result = parse_proxy_url("https://proxy.example.com:8080")
        
        assert result["server"] == "https://proxy.example.com:8080"


class TestJinaReadContent:
    """Test cases for jina_read_content function."""

    @patch('core.utils.crawling_utils.requests.get')
    def test_jina_read_content_success(self, mock_get):
        """Test jina_read_content successfully reads content."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'data: {"content": "Test content"}'
        ]
        mock_get.return_value = mock_response
        
        result = jina_read_content("https://example.com")
        
        assert result == "Test content"
        mock_get.assert_called_once()

    @patch('core.utils.crawling_utils.requests.get')
    def test_jina_read_content_retry(self, mock_get):
        """Test jina_read_content retries on failure."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("Server error")
        mock_get.return_value = mock_response
        
        with pytest.raises(Exception):
            jina_read_content("https://example.com", retry_limit=2)
        
        assert mock_get.call_count == 2

    @patch('core.utils.crawling_utils.requests.get')
    def test_jina_read_content_no_content(self, mock_get):
        """Test jina_read_content returns None when no content."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'data: {"content": ""}'
        ]
        mock_get.return_value = mock_response
        
        result = jina_read_content("https://example.com")
        
        assert result is None
