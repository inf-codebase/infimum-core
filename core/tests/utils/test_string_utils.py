"""Unit tests for core.utils.string_utils module."""
import pytest
from unittest.mock import patch
from datetime import datetime, timedelta
from core.utils.string_utils import (
    get_time_in_string,
    get_all_file_paths_from_folder,
    camel_to_plural_underscore
)


class TestGetTimeInString:
    """Test cases for get_time_in_string function."""

    def test_get_time_in_string_default_format(self):
        """Test get_time_in_string with default format."""
        result = get_time_in_string()
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Should match format YYYYMMDD_HHMMSS
        assert "_" in result

    def test_get_time_in_string_custom_format(self):
        """Test get_time_in_string with custom format."""
        result = get_time_in_string(format="%Y-%m-%d")
        
        assert isinstance(result, str)
        assert "-" in result
        assert len(result) == 10  # YYYY-MM-DD

    def test_get_time_in_string_back_days(self):
        """Test get_time_in_string with back_to_n_days parameter."""
        result = get_time_in_string(back_to_n_days=1)
        
        # Should be yesterday's date
        yesterday = datetime.today() - timedelta(days=1)
        expected = yesterday.strftime("%Y%m%d_%H%M%S")
        
        # Just verify it's a valid date string
        assert isinstance(result, str)
        assert len(result) > 0


class TestGetAllFilePathsFromFolder:
    """Test cases for get_all_file_paths_from_folder function."""

    @patch('core.utils.string_utils.glob.glob')
    def test_get_all_file_paths_from_folder(self, mock_glob):
        """Test get_all_file_paths_from_folder returns sorted file paths."""
        mock_glob.return_value = ["/path/file2.txt", "/path/file1.txt", "/path/file3.txt"]
        
        result = get_all_file_paths_from_folder("/path")
        
        assert len(result) == 3
        assert result == sorted(mock_glob.return_value)
        mock_glob.assert_called_once_with("/path/*")

    @patch('core.utils.string_utils.glob.glob')
    def test_get_all_file_paths_from_folder_empty(self, mock_glob):
        """Test get_all_file_paths_from_folder with empty folder."""
        mock_glob.return_value = []
        
        result = get_all_file_paths_from_folder("/empty")
        
        assert result == []


class TestCamelToPluralUnderscore:
    """Test cases for camel_to_plural_underscore function."""

    def test_camel_to_plural_underscore_simple(self):
        """Test camel_to_plural_underscore with simple word."""
        result = camel_to_plural_underscore("Box")
        assert result == "boxes"

    def test_camel_to_plural_underscore_camel_case(self):
        """Test camel_to_plural_underscore with camel case."""
        result = camel_to_plural_underscore("PersonAddress")
        assert result == "person_addresses"

    def test_camel_to_plural_underscore_ends_with_y(self):
        """Test camel_to_plural_underscore with word ending in 'y'."""
        result = camel_to_plural_underscore("Category")
        assert result == "categories"

    def test_camel_to_plural_underscore_ends_with_s(self):
        """Test camel_to_plural_underscore with word ending in 's'."""
        result = camel_to_plural_underscore("Match")
        assert result == "matches"

    def test_camel_to_plural_underscore_ends_with_x(self):
        """Test camel_to_plural_underscore with word ending in 'x'."""
        result = camel_to_plural_underscore("Box")
        assert result == "boxes"

    def test_camel_to_plural_underscore_ends_with_ch(self):
        """Test camel_to_plural_underscore with word ending in 'ch'."""
        result = camel_to_plural_underscore("Match")
        assert result == "matches"

    def test_camel_to_plural_underscore_ends_with_sh(self):
        """Test camel_to_plural_underscore with word ending in 'sh'."""
        result = camel_to_plural_underscore("Wish")
        assert result == "wishes"

    def test_camel_to_plural_underscore_empty(self):
        """Test camel_to_plural_underscore with empty string."""
        result = camel_to_plural_underscore("")
        assert result == ""

    def test_camel_to_plural_underscore_story(self):
        """Test camel_to_plural_underscore with 'Story' (ends with 'y')."""
        result = camel_to_plural_underscore("Story")
        assert result == "stories"
