# -*- coding: utf-8 -*-
"""Validation utilities for batch import operations."""

import re
from datetime import date, datetime
from typing import Optional, Tuple

# Email validation regex (RFC 5322 simplified)
EMAIL_REGEX = re.compile(
    r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
)

# Limits
MAX_STUDENT_ID_LENGTH = 64
MAX_NAME_LENGTH = 255
MAX_EMAIL_LENGTH = 255
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
LARGE_FILE_THRESHOLD_MB = 100
LARGE_FILE_THRESHOLD_BYTES = LARGE_FILE_THRESHOLD_MB * 1024 * 1024


def validate_email(email: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate email format.
    
    Args:
        email: Email address to validate (None is valid - optional field)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if email is None or email.strip() == '':
        return True, None
    
    email = email.strip()
    
    if len(email) > MAX_EMAIL_LENGTH:
        return False, f"Email exceeds maximum length of {MAX_EMAIL_LENGTH} characters"
    
    if not EMAIL_REGEX.match(email):
        return False, "Invalid email format"
    
    return True, None


def validate_student_id(student_id: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate student ID format and length.
    
    Args:
        student_id: Student ID to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if student_id is None or student_id.strip() == '':
        return False, "Student ID is required"
    
    student_id = student_id.strip()
    
    if len(student_id) > MAX_STUDENT_ID_LENGTH:
        return False, f"Student ID exceeds maximum length of {MAX_STUDENT_ID_LENGTH} characters"
    
    return True, None


def validate_name(name: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Validate student name.
    
    Args:
        name: Student name to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if name is None or name.strip() == '':
        return False, "Name is required"
    
    name = name.strip()
    
    if len(name) > MAX_NAME_LENGTH:
        return False, f"Name exceeds maximum length of {MAX_NAME_LENGTH} characters"
    
    return True, None


def parse_iso_date(date_str: Optional[str]) -> Tuple[Optional[date], Optional[str]]:
    """
    Parse date string from multiple common formats.
    
    Supported formats:
    - YYYY-MM-DD (ISO format, recommended)
    - M/D/YYYY or MM/DD/YYYY (US format, common in Excel)
    - DD/MM/YYYY (European format)
    - YYYY/MM/DD
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        Tuple of (parsed_date, error_message)
    """
    if date_str is None or date_str.strip() == '':
        return None, None
    
    date_str = date_str.strip()
    
    # List of date formats to try, in order of preference
    date_formats = [
        '%Y-%m-%d',      # 2008-02-21 (ISO, preferred)
        '%m/%d/%Y',      # 2/21/2008 or 02/21/2008 (US Excel)
        '%d/%m/%Y',      # 21/02/2008 (European)
        '%Y/%m/%d',      # 2008/02/21
        '%m-%d-%Y',      # 02-21-2008 (US with dashes)
        '%d-%m-%Y',      # 21-02-2008 (European with dashes)
    ]
    
    for fmt in date_formats:
        try:
            parsed = datetime.strptime(date_str, fmt).date()
            return parsed, None
        except ValueError:
            continue
    
    # If none of the formats worked, return error with helpful message
    return None, f"Invalid date format: '{date_str}'. Supported: YYYY-MM-DD, M/D/YYYY, DD/MM/YYYY"


def normalize_row_number(index: int, has_header: bool = True) -> int:
    """
    Convert 0-based index to 1-based row number accounting for header.
    
    Args:
        index: 0-based index of the data row
        has_header: Whether file has a header row
        
    Returns:
        1-based row number as seen in the file
    """
    if has_header:
        # Row 0 is header, so data row 0 is actually row 2 in file
        return index + 2
    else:
        # No header, so data row 0 is row 1 in file
        return index + 1


def validate_file_size(file_size_bytes: int) -> Tuple[bool, Optional[str]]:
    """
    Validate file size against maximum limit.
    
    Args:
        file_size_bytes: File size in bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if file_size_bytes > MAX_FILE_SIZE_BYTES:
        return False, f"File size exceeds maximum limit of {MAX_FILE_SIZE_MB}MB"
    return True, None


def is_large_file(file_size_bytes: int) -> bool:
    """
    Check if file is considered large (requires background processing).
    
    Args:
        file_size_bytes: File size in bytes
        
    Returns:
        True if file should use background job processing
    """
    return file_size_bytes > LARGE_FILE_THRESHOLD_BYTES


def validate_import_row(row: dict, row_index: int, has_header: bool = True) -> dict:
    """
    Validate a single import row and return validation result.
    
    Args:
        row: Dictionary containing row data
        row_index: 0-based index of the row
        has_header: Whether file has header row
        
    Returns:
        Dictionary with validation results:
        {
            'is_valid': bool,
            'row_number': int,
            'errors': list[str],
            'validated_data': dict
        }
    """
    errors = []
    row_number = normalize_row_number(row_index, has_header)
    
    # Validate student_id
    student_id = row.get('student_id', '').strip() if row.get('student_id') else ''
    is_valid, error = validate_student_id(student_id)
    if not is_valid:
        errors.append(error)
    
    # Validate name
    name = row.get('name', '').strip() if row.get('name') else ''
    is_valid, error = validate_name(name)
    if not is_valid:
        errors.append(error)
    
    # Validate email (optional)
    email = row.get('email', '').strip() if row.get('email') else None
    is_valid, error = validate_email(email)
    if not is_valid:
        errors.append(error)
    
    # Validate date_of_birth (optional)
    dob_str = row.get('date_of_birth', '').strip() if row.get('date_of_birth') else None
    parsed_date, error = parse_iso_date(dob_str)
    if error:
        errors.append(error)
    
    return {
        'is_valid': len(errors) == 0,
        'row_number': row_number,
        'errors': errors,
        'validated_data': {
            'student_id': student_id,
            'name': name,
            'email': email if email else None,
            'date_of_birth': parsed_date,
            'grade': row.get('grade', '').strip() if row.get('grade') else None,
            'classroom_code': row.get('classroom_code', '').strip() if row.get('classroom_code') else None,
            'photo_path': row.get('photo_path', '').strip() if row.get('photo_path') else None,
            'expected_graduation_year': row.get('expected_graduation_year'),
            'enrollment_date': row.get('enrollment_date'),
        }
    }
