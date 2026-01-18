"""
Package installation utilities.

Provides common functions for automatically installing Python packages
when they are missing, with support for uv and pip.
"""

import subprocess
import sys
from typing import Optional, List
from loguru import logger


def ensure_package_installed(
    package_name: str,
    import_name: Optional[str] = None,
    install_name: Optional[str] = None,
    prefer_uv: bool = True
) -> None:
    """
    Ensure a Python package is installed, auto-install if missing.
    
    This function attempts to import the package, and if it fails,
    automatically installs it using uv or pip.
    
    Args:
        package_name: Name of the package to import (e.g., "whisper")
        import_name: Optional alternative name for import (defaults to package_name)
        install_name: Optional package name for installation (defaults to package_name)
        prefer_uv: Whether to prefer uv over pip (default: True)
    
    Raises:
        ImportError: If installation fails or import still fails after installation
    
    Example:
        ```python
        from core.engine.package_utils import ensure_package_installed
        
        # Auto-install if missing
        ensure_package_installed("whisper", install_name="openai-whisper")
        import whisper
        ```
    """
    import_name = import_name or package_name
    install_name = install_name or package_name
    
    # Try to import first
    try:
        __import__(import_name)
        return
    except ImportError:
        pass
    
    # Try to auto-install
    logger.info(f"{package_name} library not found. Attempting to install...")
    
    # Build install commands
    install_commands: List[List[str]] = []
    
    if prefer_uv:
        # Try uv first
        install_commands.append(["uv", "pip", "install", install_name])
    
    # Fall back to pip
    install_commands.append([sys.executable, "-m", "pip", "install", install_name])
    
    # Try each command
    last_error = None
    for cmd in install_commands:
        try:
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            logger.success(f"✅ {package_name} installed successfully!")
            break
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            last_error = e
            if cmd == install_commands[-1]:  # Last attempt failed
                error_msg = (
                    f"Failed to install {package_name}. "
                    f"Tried: {' and '.join([' '.join(c) for c in install_commands])}\n"
                    f"Error: {str(e)}\n"
                    f"Please install manually with: pip install {install_name}"
                )
                logger.error(error_msg)
                raise ImportError(error_msg) from e
            continue
    
    # Retry import after installation
    try:
        __import__(import_name)
        logger.info(f"Successfully imported {import_name} after installation")
    except ImportError as e:
        error_msg = (
            f"{package_name} installation completed but import still fails. "
            f"Please restart the Python kernel and try again.\n"
            f"Original error: {str(e)}"
        )
        logger.error(error_msg)
        raise ImportError(error_msg) from e


def install_package(
    package_name: str,
    prefer_uv: bool = True
) -> bool:
    """
    Install a Python package using uv or pip.
    
    Args:
        package_name: Name of the package to install (e.g., "openai-whisper")
        prefer_uv: Whether to prefer uv over pip (default: True)
    
    Returns:
        bool: True if installation succeeded, False otherwise
    
    Example:
        ```python
        from core.engine.package_utils import install_package
        
        if install_package("openai-whisper"):
            print("Installation successful!")
        ```
    """
    install_commands: List[List[str]] = []
    
    if prefer_uv:
        install_commands.append(["uv", "pip", "install", package_name])
    
    install_commands.append([sys.executable, "-m", "pip", "install", package_name])
    
    for cmd in install_commands:
        try:
            logger.info(f"Installing {package_name} with: {' '.join(cmd)}")
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            logger.success(f"✅ {package_name} installed successfully!")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            if cmd == install_commands[-1]:  # Last attempt failed
                logger.error(f"Failed to install {package_name}")
                return False
            continue
    
    return True
