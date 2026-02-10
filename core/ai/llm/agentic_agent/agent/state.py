"""
State management for the LangGraph-based AI Agent.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Annotated
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from .types import Task, ToolResult, Message, ExecutionContext


class AgentState(TypedDict):
    """
    The state of the agent during execution.
    
    This represents the shared state that gets passed between nodes
    in the LangGraph workflow.
    """
    
    # Core conversation state
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Execution context
    session_id: str
    request_id: str
    user_query: str
    
    # Planning and task management
    tasks: List[Task]
    current_task_index: int
    
    # Tool execution results
    tool_results: List[ToolResult]
    
    # Memory and context
    memory_context: str
    relevant_memories: List[Dict[str, Any]]
    
    # Execution tracking
    iteration_count: int
    max_iterations: int
    start_time: datetime
    execution_time: float
    
    # Response generation
    final_response: Optional[str]
    intermediate_responses: List[str]
    
    # Error handling
    errors: List[str]
    retry_count: int
    max_retries: int
    
    # Metadata
    metadata: Dict[str, Any]
    token_usage: Dict[str, int]
    
    # Control flow
    should_continue: bool
    next_action: Optional[str]
    termination_reason: Optional[str]


def create_initial_state(
    user_query: str,
    session_id: str,
    request_id: str,
    max_iterations: int = 10,
    max_retries: int = 3,
    metadata: Optional[Dict[str, Any]] = None,
) -> AgentState:
    """Create initial agent state for a new conversation."""
    
    return AgentState(
        # Core conversation
        messages=[],
        
        # Execution context
        session_id=session_id,
        request_id=request_id,
        user_query=user_query,
        
        # Planning
        tasks=[],
        current_task_index=0,
        
        # Tools
        tool_results=[],
        
        # Memory
        memory_context="",
        relevant_memories=[],
        
        # Execution tracking
        iteration_count=0,
        max_iterations=max_iterations,
        start_time=datetime.now(),
        execution_time=0.0,
        
        # Responses
        final_response=None,
        intermediate_responses=[],
        
        # Error handling
        errors=[],
        retry_count=0,
        max_retries=max_retries,
        
        # Metadata
        metadata=metadata or {},
        token_usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        
        # Control flow
        should_continue=True,
        next_action=None,
        termination_reason=None,
    )


def update_state(
    state: AgentState,
    **kwargs: Any
) -> AgentState:
    """Update agent state with new values."""
    
    # Calculate execution time
    if "start_time" in state:
        execution_time = (datetime.now() - state["start_time"]).total_seconds()
        kwargs["execution_time"] = execution_time
    
    # Update the state
    updated_state = state.copy()
    updated_state.update(kwargs)
    
    return updated_state


def should_terminate(state: AgentState) -> bool:
    """Check if the agent should terminate execution."""
    
    # Check iteration limit
    if state["iteration_count"] >= state["max_iterations"]:
        return True
    
    # Check retry limit
    if state["retry_count"] >= state["max_retries"]:
        return True
    
    # Check if final response is generated
    if state["final_response"] is not None:
        return True
    
    # Check explicit continuation flag
    if not state["should_continue"]:
        return True
    
    # Check for critical errors
    if len(state["errors"]) > 0:
        # Consider terminating on critical errors
        critical_errors = [
            error for error in state["errors"]
            if "critical" in error.lower() or "fatal" in error.lower()
        ]
        if critical_errors:
            return True
    
    return False


def get_state_summary(state: AgentState) -> Dict[str, Any]:
    """Get a summary of the current state for logging/debugging."""
    
    return {
        "session_id": state["session_id"],
        "request_id": state["request_id"],
        "iteration_count": state["iteration_count"],
        "task_count": len(state["tasks"]),
        "current_task": state["current_task_index"],
        "tool_results": len(state["tool_results"]),
        "messages": len(state["messages"]),
        "execution_time": state["execution_time"],
        "errors": len(state["errors"]),
        "should_continue": state["should_continue"],
        "final_response_ready": state["final_response"] is not None,
        "token_usage": state["token_usage"],
    }