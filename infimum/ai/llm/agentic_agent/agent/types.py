"""
Type definitions for the AI Agent v2.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Message types in the conversation."""
    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"
    TOOL = "tool"


class ToolCallType(str, Enum):
    """Types of tool calls."""
    SEARCH = "search"
    CALCULATOR = "calculator"
    WEATHER = "weather"
    TIME = "time"
    WEB_SCRAPE = "web_scrape"


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Message(BaseModel):
    """A message in the conversation."""
    type: MessageType
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ToolCall(BaseModel):
    """A tool call request."""
    id: str
    tool_name: str
    tool_input: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class ToolResult(BaseModel):
    """Result from a tool execution."""
    tool_call_id: str
    tool_name: str
    result: Any
    success: bool = True
    error_message: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Task(BaseModel):
    """A task in the agent's execution plan."""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    results: List[ToolResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentResponse(BaseModel):
    """Response from the agent."""
    response: Optional[str] = None
    tasks: List[Task] = Field(default_factory=list)
    messages: List[Any] = Field(default_factory=list)  # Allow any message type
    tool_results: List[ToolResult] = Field(default_factory=list)
    execution_time: float = 0.0
    token_usage: Dict[str, int] = Field(default_factory=dict)
    agent_id: str
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentConfig(BaseModel):
    """Configuration for agent execution."""
    max_iterations: int = 10
    max_execution_time: int = 300
    temperature: float = 0.1
    model: str = "gpt-4o-mini"
    enable_memory: bool = True
    enable_tools: bool = True
    tools: List[str] = Field(default_factory=list)
    verbose: bool = False


class ExecutionContext(BaseModel):
    """Context for agent execution."""
    session_id: str
    user_id: Optional[str] = None
    request_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }