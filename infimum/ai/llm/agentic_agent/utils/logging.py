"""
Logging configuration for the AI Agent v2.
"""

import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

from loguru import logger

from ..config import get_settings


def setup_logging(
    level: Optional[str] = None,
    format_type: Optional[str] = None,
    file_path: Optional[str] = None,
) -> None:
    """
    Setup logging configuration using loguru.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type (json or text)
        file_path: Path to log file (optional)
    """
    settings = get_settings()
    
    # Use provided values or fall back to settings
    log_level = level or settings.log_level
    log_format = format_type or settings.log_format
    log_file = file_path or settings.log_file
    
    # Remove default handler
    logger.remove()
    
    # Define formats
    if log_format.lower() == "json":
        # JSON format for structured logging
        json_format = (
            "{"
            '"timestamp": "{time:YYYY-MM-DD HH:mm:ss.SSS}", '
            '"level": "{level}", '
            '"logger": "{name}", '
            '"function": "{function}", '
            '"line": {line}, '
            '"message": "{message}", '
            '"extra": {extra}'
            "}"
        )
        console_format = json_format
        file_format = json_format
    else:
        # Text format for human readability
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        )
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=console_format,
        level=log_level,
        colorize=True,
        serialize=log_format.lower() == "json",
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=file_format,
            level=log_level,
            rotation="10 MB",
            retention="30 days",
            compression="gz",
            serialize=log_format.lower() == "json",
        )


def get_logger(name: str) -> Any:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


class LoggerAdapter:
    """
    Adapter to add structured logging capabilities.
    """
    
    def __init__(self, logger_instance: Any, context: Optional[Dict[str, Any]] = None):
        self.logger = logger_instance
        self.context = context or {}
    
    def _log_with_context(self, level: str, message: str, **kwargs) -> None:
        """Log with additional context."""
        extra_context = {**self.context, **kwargs}
        getattr(self.logger, level)(message, **extra_context)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log_with_context("debug", message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log_with_context("info", message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log_with_context("warning", message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log_with_context("error", message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._log_with_context("critical", message, **kwargs)
    
    def bind(self, **kwargs) -> "LoggerAdapter":
        """Bind additional context to the logger."""
        new_context = {**self.context, **kwargs}
        return LoggerAdapter(self.logger, new_context)


def get_structured_logger(
    name: str,
    context: Optional[Dict[str, Any]] = None,
) -> LoggerAdapter:
    """
    Get a structured logger with additional context.
    
    Args:
        name: Logger name
        context: Additional context to include in all logs
    
    Returns:
        Structured logger adapter
    """
    base_logger = get_logger(name)
    return LoggerAdapter(base_logger, context)


def log_execution_time(func):
    """
    Decorator to log function execution time.
    """
    from functools import wraps
    import time
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_logger = get_logger(func.__module__)
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            func_logger.info(
                f"Function {func.__name__} completed successfully",
                execution_time=execution_time,
                function=func.__name__,
                module=func.__module__,
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            func_logger.error(
                f"Function {func.__name__} failed",
                error=str(e),
                execution_time=execution_time,
                function=func.__name__,
                module=func.__module__,
            )
            raise
    
    return wrapper


def log_agent_event(
    event_type: str,
    agent_id: str,
    session_id: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Log agent-specific events with structured data.
    
    Args:
        event_type: Type of event (execution_start, tool_call, error, etc.)
        agent_id: Agent identifier
        session_id: Session identifier
        **kwargs: Additional event data
    """
    event_logger = get_logger("agent.events")
    
    event_data = {
        "event_type": event_type,
        "agent_id": agent_id,
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        **kwargs,
    }
    
    event_logger.info(f"Agent event: {event_type}", **event_data)