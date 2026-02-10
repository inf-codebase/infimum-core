"""
Custom exceptions for the AI Agent v2.
"""

from typing import Optional, Any, Dict


class AgentError(Exception):
    """Base exception for agent-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.original_error = original_error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
            "original_error": str(self.original_error) if self.original_error else None,
        }


class ConfigurationError(AgentError):
    """Exception raised for configuration-related errors."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if config_key:
            context["config_key"] = config_key
        
        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            context=context,
            **kwargs,
        )


class ToolError(AgentError):
    """Exception raised for tool execution errors."""
    
    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        tool_input: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if tool_name:
            context["tool_name"] = tool_name
        if tool_input:
            context["tool_input"] = tool_input
        
        super().__init__(
            message=message,
            error_code="TOOL_ERROR",
            context=context,
            **kwargs,
        )


class MemoryError(AgentError):
    """Exception raised for memory system errors."""
    
    def __init__(
        self,
        message: str,
        memory_type: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if memory_type:
            context["memory_type"] = memory_type
        if operation:
            context["operation"] = operation
        
        super().__init__(
            message=message,
            error_code="MEMORY_ERROR",
            context=context,
            **kwargs,
        )


class WorkflowError(AgentError):
    """Exception raised for workflow execution errors."""
    
    def __init__(
        self,
        message: str,
        workflow_step: Optional[str] = None,
        state_info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if workflow_step:
            context["workflow_step"] = workflow_step
        if state_info:
            context["state_info"] = state_info
        
        super().__init__(
            message=message,
            error_code="WORKFLOW_ERROR",
            context=context,
            **kwargs,
        )


class RateLimitError(AgentError):
    """Exception raised when rate limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if service:
            context["service"] = service
        if retry_after:
            context["retry_after"] = retry_after
        
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            context=context,
            **kwargs,
        )


class ValidationError(AgentError):
    """Exception raised for input validation errors."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = str(value)
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            context=context,
            **kwargs,
        )


class TimeoutError(AgentError):
    """Exception raised when operations timeout."""
    
    def __init__(
        self,
        message: str,
        timeout_duration: Optional[float] = None,
        operation: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if timeout_duration:
            context["timeout_duration"] = timeout_duration
        if operation:
            context["operation"] = operation
        
        super().__init__(
            message=message,
            error_code="TIMEOUT_ERROR",
            context=context,
            **kwargs,
        )


def handle_agent_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    reraise: bool = True,
) -> AgentError:
    """
    Convert any exception to an AgentError with context.
    
    Args:
        error: Original exception
        context: Additional context
        reraise: Whether to reraise the exception
    
    Returns:
        AgentError instance
        
    Raises:
        AgentError: If reraise is True
    """
    if isinstance(error, AgentError):
        agent_error = error
    else:
        agent_error = AgentError(
            message=str(error),
            error_code="GENERAL_ERROR",
            context=context,
            original_error=error,
        )
    
    if reraise:
        raise agent_error
    
    return agent_error