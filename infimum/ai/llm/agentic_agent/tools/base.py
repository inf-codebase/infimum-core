"""
Base tool implementation for the AI Agent v2.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional, Type, Union
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun

# Define ToolResult locally to avoid circular imports
class ToolResult(BaseModel):
    """Result from a tool execution."""
    tool_call_id: str
    tool_name: str
    result: Any
    success: bool = True
    error_message: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


class BaseToolInput(BaseModel):
    """Base input schema for tools."""
    pass


class BaseAgentTool(BaseTool, ABC):
    """
    Base class for all agent tools with enhanced error handling and monitoring.
    """
    
    # Tool metadata
    tool_id: str = Field(..., description="Unique tool identifier")
    timeout: int = Field(default=30, description="Tool execution timeout in seconds")
    max_retries: int = Field(default=2, description="Maximum retry attempts")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    @property
    @abstractmethod
    def args_schema(self) -> Type[BaseModel]:
        """Return the input schema for this tool."""
        pass
    
    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Synchronous tool execution with error handling."""
        start_time = datetime.now()
        tool_call_id = f"{self.tool_id}_{start_time.isoformat()}"
        
        try:
            # Validate input
            validated_input = self.args_schema(**kwargs)
            
            # Execute the tool
            result = self._execute(validated_input)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=self.name,
                result=result,
                success=True,
                execution_time=execution_time,
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=self.name,
                result=None,
                success=False,
                error_message=str(e),
                execution_time=execution_time,
            )
    
    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Asynchronous tool execution with error handling."""
        start_time = datetime.now()
        tool_call_id = f"{self.tool_id}_{start_time.isoformat()}"
        
        try:
            # Validate input
            validated_input = self.args_schema(**kwargs)
            
            # Execute the tool with timeout
            result = await asyncio.wait_for(
                self._aexecute(validated_input),
                timeout=self.timeout
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=self.name,
                result=result,
                success=True,
                execution_time=execution_time,
            )
            
        except asyncio.TimeoutError:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=self.name,
                result=None,
                success=False,
                error_message=f"Tool execution timed out after {self.timeout} seconds",
                execution_time=execution_time,
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=self.name,
                result=None,
                success=False,
                error_message=str(e),
                execution_time=execution_time,
            )
    
    @abstractmethod
    def _execute(self, input_data: BaseModel) -> Any:
        """Synchronous execution implementation."""
        pass
    
    async def _aexecute(self, input_data: BaseModel) -> Any:
        """Asynchronous execution implementation. Default to sync."""
        # Run sync version in thread pool by default
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._execute, input_data)
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get tool information for debugging and monitoring."""
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "input_schema": self.args_schema.schema(),
        }