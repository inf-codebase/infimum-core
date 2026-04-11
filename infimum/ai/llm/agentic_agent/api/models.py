"""
Pydantic models for the API layer.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from ..agent.types import AgentResponse, Task, ToolResult, Message


class QueryRequest(BaseModel):
    """Request model for agent queries."""
    query: str = Field(..., min_length=1, max_length=10000, description="User query")
    session_id: Optional[str] = Field(default=None, description="Session identifier for conversation continuity")
    stream: bool = Field(default=False, description="Enable streaming response")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Agent configuration overrides")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context metadata")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What is the weather like in New York today?",
                "session_id": "session_123",
                "stream": False,
                "config": {
                    "temperature": 0.1,
                    "max_iterations": 5
                },
                "context": {
                    "user_id": "user_456",
                    "source": "web_interface"
                }
            }
        }
    }


class QueryResponse(BaseModel):
    """Response model for agent queries."""
    response: str = Field(..., description="Agent response")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    request_id: str = Field(..., description="Unique request identifier")
    agent_id: str = Field(..., description="Agent identifier")
    
    # Execution details
    execution_time: float = Field(..., ge=0, description="Execution time in seconds")
    token_usage: Dict[str, int] = Field(default_factory=dict, description="Token usage statistics")
    
    # Task and tool information
    tasks_completed: int = Field(default=0, ge=0, description="Number of tasks completed")
    tools_used: List[str] = Field(default_factory=list, description="List of tools used")
    tool_results: List[ToolResult] = Field(default_factory=list, description="Tool execution results")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    success: bool = Field(default=True, description="Whether the request was successful")
    error_message: Optional[str] = Field(default=None, description="Error message if request failed")
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history."""
    session_id: str = Field(..., description="Session identifier")
    messages: List[Message] = Field(..., description="Conversation messages")
    total_messages: int = Field(..., ge=0, description="Total number of messages")
    session_start: datetime = Field(..., description="Session start time")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }


class AgentStatusResponse(BaseModel):
    """Response model for agent status."""
    agent_id: str = Field(..., description="Agent identifier")
    status: str = Field(..., description="Agent status")
    model: str = Field(..., description="Current model")
    tools_available: int = Field(..., ge=0, description="Number of available tools")
    memory_enabled: bool = Field(..., description="Whether memory is enabled")
    uptime: float = Field(..., ge=0, description="Agent uptime in seconds")
    
    # Statistics
    total_queries: int = Field(default=0, ge=0, description="Total queries processed")
    successful_queries: int = Field(default=0, ge=0, description="Successful queries")
    failed_queries: int = Field(default=0, ge=0, description="Failed queries")
    average_response_time: float = Field(default=0.0, ge=0, description="Average response time")
    
    # Configuration
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")
    
    timestamp: datetime = Field(default_factory=datetime.now, description="Status timestamp")
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    version: str = Field(..., description="Application version")
    
    # Service checks
    database_healthy: bool = Field(default=True, description="Database health status")
    memory_healthy: bool = Field(default=True, description="Memory system health")
    tools_healthy: bool = Field(default=True, description="Tools system health")
    
    # System information
    uptime: float = Field(..., ge=0, description="System uptime in seconds")
    memory_usage: Dict[str, float] = Field(default_factory=dict, description="Memory usage statistics")
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    request_id: Optional[str] = Field(default=None, description="Request identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }


class ToolListResponse(BaseModel):
    """Response model for available tools."""
    tools: List[Dict[str, Any]] = Field(..., description="Available tools")
    total_tools: int = Field(..., ge=0, description="Total number of tools")
    categories: List[str] = Field(..., description="Tool categories")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "tools": [
                    {
                        "name": "web_search",
                        "description": "Search the web for information",
                        "category": "search",
                        "parameters": {
                            "query": "string",
                            "num_results": "integer"
                        }
                    }
                ],
                "total_tools": 5,
                "categories": ["search", "calculation", "weather", "time"]
            }
        }
    }


class SessionRequest(BaseModel):
    """Request model for session operations."""
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    
    
class SessionResponse(BaseModel):
    """Response model for session operations."""
    session_id: str = Field(..., description="Session identifier") 
    action: str = Field(..., description="Action performed")
    success: bool = Field(..., description="Whether action was successful")
    message: Optional[str] = Field(default=None, description="Additional message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Action timestamp")
    
    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }