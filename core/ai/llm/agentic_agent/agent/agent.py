"""
Production AI Agent implementation using LangGraph and LangChain.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

# from langchain.agents import AgentExecutor
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState, create_initial_state, should_terminate, get_state_summary
from .types import AgentResponse, AgentConfig, ExecutionContext
from .workflow import create_agent_workflow
from ..memory import MemoryManager
from ..tools import ToolRegistry, get_default_tools
from ..config import get_settings
from ..utils.logging import get_logger


logger = get_logger(__name__)


class Agent:
    """
    Production-ready AI Agent using LangGraph for workflow orchestration
    and LangChain for tool integration and memory management.
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
        memory_manager: Optional[MemoryManager] = None,
        custom_workflows: Optional[Dict[str, Callable]] = None,
    ):
        self.settings = get_settings()
        self.agent_id = str(uuid.uuid4())
        self.config = config or AgentConfig()

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.config.model,
            temperature=self.config.temperature,
            openai_api_key=self.settings.openai_api_key,
            timeout=self.settings.openai_timeout,
        )

        # Initialize tools
        self.tool_registry = tool_registry or get_default_tools(
            serp_api_key=self.settings.serp_api_key,
        )
        self.tools = self.tool_registry.get_tools() if self.config.enable_tools else []

        # Initialize memory
        self.memory_manager = memory_manager
        if self.config.enable_memory and not self.memory_manager:
            self.memory_manager = MemoryManager(
                memory_type=self.settings.memory_type,
                max_tokens=self.settings.memory_max_tokens,
            )

        # Create workflow
        self.workflow = self._create_workflow(custom_workflows)

        # Initialize checkpointer for state persistence
        self.checkpointer = MemorySaver()

        # Compile the graph
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

        logger.info(f"Agent {self.agent_id} initialized with {len(self.tools)} tools")

    def _create_workflow(
        self, custom_workflows: Optional[Dict[str, Callable]] = None
    ) -> StateGraph:
        """Create the LangGraph workflow."""
        return create_agent_workflow(
            llm=self.llm,
            tools=self.tools,
            memory_manager=self.memory_manager,
            config=self.config,
            custom_workflows=custom_workflows,
        )

    async def arun(
        self,
        query: str,
        context: Optional[ExecutionContext] = None,
        stream: bool = False,
    ) -> AgentResponse:
        """Asynchronously run the agent with a query."""
        return await self._execute_async(query, context, stream)

    def run(
        self,
        query: str,
        context: Optional[ExecutionContext] = None,
        stream: bool = False,
    ) -> AgentResponse:
        """Synchronously run the agent with a query."""
        import asyncio

        return asyncio.run(self._execute_async(query, context, stream))

    async def _execute_async(
        self,
        query: str,
        context: Optional[ExecutionContext] = None,
        stream: bool = False,
    ) -> AgentResponse:
        """Execute the agent workflow."""
        start_time = datetime.now()

        # Create execution context
        if not context:
            context = ExecutionContext(
                session_id=str(uuid.uuid4()),
                request_id=str(uuid.uuid4()),
            )

        # Create initial state
        initial_state = create_initial_state(
            user_query=query,
            session_id=context.session_id,
            request_id=context.request_id,
            max_iterations=self.config.max_iterations,
            metadata=context.metadata,
        )

        # Add user message to memory
        if self.memory_manager:
            self.memory_manager.add_user_message(
                query, {"request_id": context.request_id}
            )

        try:
            logger.info(f"Starting agent execution: {context.request_id}")

            # Configure execution
            config = {"configurable": {"thread_id": context.session_id}}

            if stream:
                # Stream execution results
                response = await self._stream_execution(initial_state, config)
            else:
                # Regular execution
                response = await self._regular_execution(initial_state, config)

            # Add AI response to memory
            if self.memory_manager and response.response:
                self.memory_manager.add_ai_message(
                    response.response,
                    {
                        "request_id": context.request_id,
                        "execution_time": response.execution_time,
                        "tool_calls": len(response.tool_results),
                    },
                )

            logger.info(f"Agent execution completed: {context.request_id}")
            return response

        except Exception as e:
            logger.error(f"Agent execution failed: {context.request_id}: {str(e)}")

            # Create error response
            execution_time = (datetime.now() - start_time).total_seconds()
            return AgentResponse(
                response=f"I encountered an error while processing your request: {str(e)}",
                execution_time=execution_time,
                agent_id=self.agent_id,
                session_id=context.session_id,
            )

    async def _regular_execution(
        self,
        initial_state: AgentState,
        config: Dict[str, Any],
    ) -> AgentResponse:
        """Execute the agent workflow regularly."""
        # Execute the workflow
        final_state = await self.app.ainvoke(initial_state, config)

        # Create response from final state
        return self._create_response_from_state(final_state)

    async def _stream_execution(
        self,
        initial_state: AgentState,
        config: Dict[str, Any],
    ) -> AgentResponse:
        """Execute the agent workflow with streaming."""
        final_state = None

        # Stream the execution
        async for state in self.app.astream(initial_state, config):
            final_state = state

            # Log intermediate states
            if self.config.verbose:
                summary = get_state_summary(final_state)
                logger.debug(f"Intermediate state: {summary}")

        # Create response from final state
        if final_state:
            return self._create_response_from_state(final_state)
        else:
            raise RuntimeError("No final state received from workflow")

    def _create_response_from_state(self, state: AgentState) -> AgentResponse:
        """Create AgentResponse from final agent state."""
        # Ensure we have a valid response string
        final_response = state.get("final_response")
        if final_response is None:
            final_response = "No response generated"

        return AgentResponse(
            response=final_response,
            tasks=state.get("tasks", []),
            messages=state.get("messages", []),
            tool_results=state.get("tool_results", []),
            execution_time=state.get("execution_time", 0.0),
            token_usage=state.get("token_usage", {}),
            agent_id=self.agent_id,
            session_id=state.get("session_id"),
        )

    def get_conversation_history(
        self, session_id: str, limit: int = 10
    ) -> List[BaseMessage]:
        """Get conversation history for a session."""
        if not self.memory_manager:
            return []

        return self.memory_manager.get_conversation_history(limit=limit)

    def clear_session(self, session_id: str) -> None:
        """Clear session data."""
        if self.memory_manager:
            self.memory_manager.clear_session_memory()

        # Clear checkpointer state
        try:
            config = {"configurable": {"thread_id": session_id}}
            # Note: MemorySaver doesn't have a clear method, but we can reset
            # by starting a new conversation thread
        except Exception as e:
            logger.warning(f"Failed to clear session state: {e}")

    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent configuration and status information."""
        return {
            "agent_id": self.agent_id,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_iterations": self.config.max_iterations,
            "max_execution_time": self.config.max_execution_time,
            "tools_enabled": self.config.enable_tools,
            "memory_enabled": self.config.enable_memory,
            "available_tools": self.tool_registry.get_tool_names()
            if self.config.enable_tools
            else [],
            "tool_count": len(self.tools),
            "memory_stats": self.memory_manager.get_memory_stats()
            if self.memory_manager
            else None,
        }

    def add_tool(self, tool) -> None:
        """Add a new tool to the agent."""
        self.tool_registry.register_tool(tool)
        if self.config.enable_tools:
            self.tools = self.tool_registry.get_tools()
            # Note: In production, you'd want to recreate the workflow
            # This is a simplified approach

    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool from the agent."""
        # This would require more complex implementation to properly
        # remove from registry and recreate workflow
        logger.warning("Tool removal not implemented in this version")
        return False

    def update_config(self, new_config: AgentConfig) -> None:
        """Update agent configuration."""
        self.config = new_config

        # Update LLM if model changed
        if new_config.model != self.llm.model_name:
            self.llm = ChatOpenAI(
                model=new_config.model,
                temperature=new_config.temperature,
                openai_api_key=self.settings.openai_api_key,
                timeout=self.settings.openai_timeout,
            )

            # Recreate workflow with new LLM
            self.workflow = self._create_workflow()
            self.app = self.workflow.compile(checkpointer=self.checkpointer)

        logger.info(f"Agent configuration updated: {self.agent_id}")

    def get_workflow_graph(self) -> Dict[str, Any]:
        """Get the workflow graph structure for visualization."""
        try:
            # Get the compiled graph structure
            return {
                "nodes": list(self.workflow.nodes.keys()),
                "edges": [(edge.source, edge.target) for edge in self.workflow.edges],
                "entry_point": getattr(self.workflow, "entry_point", "start"),
                "end_points": [
                    node for node, edges in self.workflow.edges if not edges
                ],
            }
        except Exception as e:
            logger.warning(f"Failed to get workflow graph: {e}")
            return {"error": str(e)}
