"""File utility functions for handling uploads and temporary files."""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from fastapi import UploadFile


def save_uploaded_file(
    file: UploadFile,
    temp_dir: Optional[str] = None,
    preserve_name: bool = True
) -> Tuple[str, bool]:
    """
    Save an uploaded file to a temporary location with content-based deduplication.
    
    Uses SHA-256 hashing to detect duplicate files based on content. If a file with
    identical content already exists, returns the existing file path instead of creating
    a duplicate.
    
    Args:
        file: FastAPI UploadFile object to save
        temp_dir: Optional directory to save the file in. If None, uses system temp directory
        preserve_name: If True, uses hash-based naming with original extension. 
                      If False, generates a unique temporary filename
        
    Returns:
        Tuple[str, bool]: (file_path, is_new) where:
            - file_path: Path to the saved (or existing) temporary file
            - is_new: True if file was newly saved, False if existing file was reused
        
    Raises:
        ValueError: If file or filename is invalid
        IOError: If file cannot be written
    """
    if not file:
        raise ValueError("No file provided")
    
    if not file.filename:
        raise ValueError("File has no filename")
    
    # Determine the temporary directory
    if temp_dir:
        temp_directory = Path(temp_dir)
        temp_directory.mkdir(parents=True, exist_ok=True)
    else:
        temp_directory = Path(tempfile.gettempdir())
    
    # Determine the filename
    if preserve_name:
        # Hash the file content to detect duplicates
        hasher = hashlib.sha256()
        chunk_size = 1024 * 1024  # 1MB chunks
        
        file.file.seek(0)  # Ensure we're at the start
        while True:
            chunk = file.file.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
        
        file_hash = hasher.hexdigest()
        file.file.seek(0)  # Reset for writing
        
        # Create filename with hash and original extension
        suffix = Path(file.filename).suffix
        hash_filename = f"{file_hash}{suffix}"
        temp_path = temp_directory / hash_filename
        
        # If file with this hash already exists, return existing path
        if temp_path.exists():
            file.file.seek(0)  # Reset file pointer
            return str(temp_path), False
    else:
        # Generate a unique temporary file
        suffix = Path(file.filename).suffix
        fd, temp_path_str = tempfile.mkstemp(suffix=suffix, dir=str(temp_directory))
        os.close(fd)  # Close the file descriptor, we'll write via open()
        temp_path = Path(temp_path_str)
    
    # Write the uploaded file to disk
    try:
        with open(temp_path, "wb") as buffer:
            # Read file in chunks to handle large files
            chunk_size = 1024 * 1024  # 1MB chunks
            while True:
                chunk = file.file.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)
        
        # Reset file pointer in case it needs to be read again
        file.file.seek(0)
        
        return str(temp_path), True
    
    except Exception as e:
        # Clean up the temporary file if write failed
        if temp_path.exists():
            temp_path.unlink()
        raise IOError(f"Failed to save uploaded file: {str(e)}") from e


def cleanup_temp_file(file_path: str) -> None:
    """
    Safely delete a temporary file.
    
    Args:
        file_path: Path to the file to delete
    """
    try:
        path = Path(file_path)
        if path.exists() and path.is_file():
            path.unlink()
    except Exception:
        # Silently fail - temp files will be cleaned up by OS eventually
        pass
