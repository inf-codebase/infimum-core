"""Unit tests for core.utils.validation_utils module."""
import pytest
from datetime import date
from core.utils.validation_utils import (
    validate_email,
    validate_student_id,
    validate_name,
    parse_iso_date,
    normalize_row_number,
    validate_file_size,
    is_large_file,
    validate_import_row
)


class TestValidateEmail:
    """Test cases for validate_email function."""

    def test_validate_email_valid(self):
        """Test validate_email with valid email."""
        is_valid, error = validate_email("test@example.com")
        assert is_valid is True
        assert error is None

    def test_validate_email_invalid_format(self):
        """Test validate_email with invalid format."""
        is_valid, error = validate_email("invalid_email")
        assert is_valid is False
        assert "Invalid email format" in error

    def test_validate_email_none(self):
        """Test validate_email with None (optional field)."""
        is_valid, error = validate_email(None)
        assert is_valid is True
        assert error is None

    def test_validate_email_empty_string(self):
        """Test validate_email with empty string."""
        is_valid, error = validate_email("")
        assert is_valid is True
        assert error is None

    def test_validate_email_too_long(self):
        """Test validate_email with email exceeding max length."""
        long_email = "a" * 250 + "@example.com"
        is_valid, error = validate_email(long_email)
        assert is_valid is False
        assert "exceeds maximum length" in error


class TestValidateStudentId:
    """Test cases for validate_student_id function."""

    def test_validate_student_id_valid(self):
        """Test validate_student_id with valid ID."""
        is_valid, error = validate_student_id("STU123")
        assert is_valid is True
        assert error is None

    def test_validate_student_id_none(self):
        """Test validate_student_id with None."""
        is_valid, error = validate_student_id(None)
        assert is_valid is False
        assert "required" in error

    def test_validate_student_id_empty(self):
        """Test validate_student_id with empty string."""
        is_valid, error = validate_student_id("")
        assert is_valid is False
        assert "required" in error

    def test_validate_student_id_too_long(self):
        """Test validate_student_id exceeding max length."""
        long_id = "A" * 65
        is_valid, error = validate_student_id(long_id)
        assert is_valid is False
        assert "exceeds maximum length" in error


class TestValidateName:
    """Test cases for validate_name function."""

    def test_validate_name_valid(self):
        """Test validate_name with valid name."""
        is_valid, error = validate_name("John Doe")
        assert is_valid is True
        assert error is None

    def test_validate_name_none(self):
        """Test validate_name with None."""
        is_valid, error = validate_name(None)
        assert is_valid is False
        assert "required" in error

    def test_validate_name_empty(self):
        """Test validate_name with empty string."""
        is_valid, error = validate_name("")
        assert is_valid is False
        assert "required" in error

    def test_validate_name_too_long(self):
        """Test validate_name exceeding max length."""
        long_name = "A" * 256
        is_valid, error = validate_name(long_name)
        assert is_valid is False
        assert "exceeds maximum length" in error


class TestParseIsoDate:
    """Test cases for parse_iso_date function."""

    def test_parse_iso_date_iso_format(self):
        """Test parse_iso_date with ISO format."""
        parsed, error = parse_iso_date("2023-12-25")
        assert parsed == date(2023, 12, 25)
        assert error is None

    def test_parse_iso_date_us_format(self):
        """Test parse_iso_date with US format."""
        parsed, error = parse_iso_date("12/25/2023")
        assert parsed == date(2023, 12, 25)
        assert error is None

    def test_parse_iso_date_european_format(self):
        """Test parse_iso_date with European format."""
        parsed, error = parse_iso_date("25/12/2023")
        assert parsed == date(2023, 12, 25)
        assert error is None

    def test_parse_iso_date_none(self):
        """Test parse_iso_date with None."""
        parsed, error = parse_iso_date(None)
        assert parsed is None
        assert error is None

    def test_parse_iso_date_invalid(self):
        """Test parse_iso_date with invalid format."""
        parsed, error = parse_iso_date("invalid_date")
        assert parsed is None
        assert "Invalid date format" in error


class TestNormalizeRowNumber:
    """Test cases for normalize_row_number function."""

    def test_normalize_row_number_with_header(self):
        """Test normalize_row_number with header row."""
        result = normalize_row_number(0, has_header=True)
        assert result == 2  # Row 0 is header, data row 0 is row 2

    def test_normalize_row_number_without_header(self):
        """Test normalize_row_number without header row."""
        result = normalize_row_number(0, has_header=False)
        assert result == 1  # No header, data row 0 is row 1


class TestValidateFileSize:
    """Test cases for validate_file_size function."""

    def test_validate_file_size_valid(self):
        """Test validate_file_size with valid size."""
        is_valid, error = validate_file_size(1024 * 1024)  # 1MB
        assert is_valid is True
        assert error is None

    def test_validate_file_size_too_large(self):
        """Test validate_file_size with file exceeding limit."""
        large_size = 11 * 1024 * 1024  # 11MB
        is_valid, error = validate_file_size(large_size)
        assert is_valid is False
        assert "exceeds maximum limit" in error


class TestIsLargeFile:
    """Test cases for is_large_file function."""

    def test_is_large_file_true(self):
        """Test is_large_file returns True for large file."""
        large_size = 101 * 1024 * 1024  # 101MB
        assert is_large_file(large_size) is True

    def test_is_large_file_false(self):
        """Test is_large_file returns False for small file."""
        small_size = 50 * 1024 * 1024  # 50MB
        assert is_large_file(small_size) is False


class TestValidateImportRow:
    """Test cases for validate_import_row function."""

    def test_validate_import_row_valid(self):
        """Test validate_import_row with valid row."""
        row = {
            "student_id": "STU123",
            "name": "John Doe",
            "email": "john@example.com",
            "date_of_birth": "2000-01-01"
        }
        
        result = validate_import_row(row, 0, has_header=True)
        
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
        assert result["row_number"] == 2

    def test_validate_import_row_invalid(self):
        """Test validate_import_row with invalid row."""
        row = {
            "student_id": "",  # Missing required field
            "name": "",  # Missing required field
            "email": "invalid_email"  # Invalid format
        }
        
        result = validate_import_row(row, 0, has_header=True)
        
        assert result["is_valid"] is False
        assert len(result["errors"]) > 0

    def test_validate_import_row_with_optional_fields(self):
        """Test validate_import_row with optional fields."""
        row = {
            "student_id": "STU123",
            "name": "John Doe",
            "email": None,  # Optional
            "date_of_birth": None  # Optional
        }
        
        result = validate_import_row(row, 0, has_header=True)
        
        assert result["is_valid"] is True
        assert result["validated_data"]["email"] is None
