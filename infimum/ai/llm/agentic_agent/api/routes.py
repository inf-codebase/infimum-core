"""
API routes for the AI Agent v2.
"""

import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from starlette.status import HTTP_200_OK, HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR

from .models import (
    QueryRequest,
    QueryResponse,
    ConversationHistoryResponse,
    AgentStatusResponse,
    HealthCheckResponse,
    ErrorResponse,
    ToolListResponse,
    SessionRequest,
    SessionResponse,
)
from ..agent import Agent
from ..agent.types import AgentConfig, ExecutionContext
from ..tools import get_default_tools
from ..memory import MemoryManager
from infimum.base.api_router_registry import APIRouterRegistry
from ..config import get_settings
from ..utils.logging import get_logger, log_agent_event
from ..utils.exceptions import handle_agent_error


logger = get_logger(__name__)
settings = get_settings()


# Global agent instance (in production, you might want a more sophisticated agent management)
_agent_instance: Optional[Agent] = None


async def get_agent() -> Agent:
    """Get or create the agent instance."""
    global _agent_instance
    
    if _agent_instance is None:
        logger.info("Initializing agent instance")
        
        # Create tools
        tool_registry = get_default_tools(
            serp_api_key=settings.serp_api_key,
        )
        
        # Create memory manager
        memory_manager = MemoryManager(
            memory_type=settings.memory_type,
            max_tokens=settings.memory_max_tokens,
        )
        
        # Create agent config
        agent_config = AgentConfig(
            max_iterations=settings.max_iterations,
            max_execution_time=settings.max_execution_time,
            temperature=settings.openai_temperature,
            model=settings.openai_model,
            enable_memory=True,
            enable_tools=True,
        )
        
        # Create agent
        _agent_instance = Agent(
            config=agent_config,
            tool_registry=tool_registry,
            memory_manager=memory_manager,
        )
        
        logger.info(f"Agent instance created: {_agent_instance.agent_id}")
    
    return _agent_instance


def setup_routes(app) -> None:
    """Setup API routes."""
    
    # Main API router
    api_router = APIRouter(prefix="/api/v1", tags=["agent"])
    
    @api_router.post("/query", response_model=QueryResponse)
    async def query_agent(
        request: QueryRequest,
        background_tasks: BackgroundTasks,
        http_request: Request,
        agent: Agent = Depends(get_agent),
    ):
        """Process a query with the AI agent."""
        
        request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))
        start_time = datetime.now()
        
        try:
            # Create execution context
            context = ExecutionContext(
                session_id=request.session_id or str(uuid.uuid4()),
                request_id=request_id,
                metadata={
                    "user_agent": http_request.headers.get("user-agent"),
                    "client_ip": http_request.client.host if http_request.client else None,
                    **(request.context or {}),
                },
            )
            
            # Log event
            log_agent_event(
                event_type="query_start",
                agent_id=agent.agent_id,
                session_id=context.session_id,
                query=request.query[:100],  # Truncate for logging
            )
            
            # Execute query
            if request.stream:
                # TODO: Implement streaming response
                response = await agent.arun(
                    query=request.query,
                    context=context,
                    stream=False,  # Disable for now
                )
            else:
                response = await agent.arun(
                    query=request.query,
                    context=context,
                    stream=False,
                )
            
            # Log success event
            log_agent_event(
                event_type="query_success",
                agent_id=agent.agent_id,
                session_id=context.session_id,
                execution_time=response.execution_time,
                tools_used=len(response.tool_results),
            )
            
            # Convert to API response format
            api_response = QueryResponse(
                response=response.response,
                session_id=context.session_id,
                request_id=request_id,
                agent_id=response.agent_id,
                execution_time=response.execution_time,
                token_usage=response.token_usage,
                tasks_completed=len([t for t in response.tasks if t.status.value == "completed"]),
                tools_used=[tr.tool_name for tr in response.tool_results],
                tool_results=response.tool_results,
                success=True,
            )
            
            return api_response
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Log error event
            log_agent_event(
                event_type="query_error",
                agent_id=agent.agent_id,
                session_id=request.session_id or "unknown",
                error=str(e),
                execution_time=execution_time,
            )
            
            # Handle error
            agent_error = handle_agent_error(e, reraise=False)
            
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail=agent_error.to_dict(),
            )
    
    @api_router.get("/status", response_model=AgentStatusResponse)
    async def get_agent_status(agent: Agent = Depends(get_agent)):
        """Get agent status and information."""
        
        try:
            agent_info = agent.get_agent_info()
            
            # TODO: Add actual statistics tracking
            status_response = AgentStatusResponse(
                agent_id=agent_info["agent_id"],
                status="healthy",
                model=agent_info["model"],
                tools_available=agent_info["tool_count"],
                memory_enabled=agent_info["memory_enabled"],
                uptime=0.0,  # TODO: Track actual uptime
                config=agent_info,
            )
            
            return status_response
            
        except Exception as e:
            logger.error(f"Failed to get agent status: {e}")
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get agent status: {str(e)}",
            )
    
    @api_router.get("/tools", response_model=ToolListResponse)
    async def get_available_tools(agent: Agent = Depends(get_agent)):
        """Get list of available tools."""
        
        try:
            tools_info = []
            for tool_name in agent.tool_registry.get_tool_names():
                tool_info = agent.tool_registry.get_tool_info(tool_name)
                if tool_info:
                    tools_info.append(tool_info)
            
            categories = agent.tool_registry.list_categories()
            
            return ToolListResponse(
                tools=tools_info,
                total_tools=len(tools_info),
                categories=categories,
            )
            
        except Exception as e:
            logger.error(f"Failed to get tools: {e}")
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get tools: {str(e)}",
            )
    
    @api_router.get("/history/{session_id}", response_model=ConversationHistoryResponse)
    async def get_conversation_history(
        session_id: str,
        limit: int = 50,
        agent: Agent = Depends(get_agent),
    ):
        """Get conversation history for a session."""
        
        try:
            messages = agent.get_conversation_history(session_id, limit=limit)
            
            # Convert to API format
            api_messages = []
            for msg in messages:
                api_messages.append({
                    "type": msg.__class__.__name__,
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat(),  # TODO: Get actual timestamp
                })
            
            return ConversationHistoryResponse(
                session_id=session_id,
                messages=api_messages,
                total_messages=len(api_messages),
                session_start=datetime.now(),  # TODO: Track actual session start
                last_activity=datetime.now(),
            )
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get conversation history: {str(e)}",
            )
    
    @api_router.post("/session/clear", response_model=SessionResponse)
    async def clear_session(
        request: SessionRequest,
        agent: Agent = Depends(get_agent),
    ):
        """Clear a conversation session."""
        
        try:
            session_id = request.session_id or str(uuid.uuid4())
            
            agent.clear_session(session_id)
            
            return SessionResponse(
                session_id=session_id,
                action="clear",
                success=True,
                message="Session cleared successfully",
            )
            
        except Exception as e:
            logger.error(f"Failed to clear session: {e}")
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to clear session: {str(e)}",
            )
    
    # Health check router
    health_router = APIRouter(tags=["health"])
    
    @health_router.get("/health", response_model=HealthCheckResponse)
    async def health_check():
        """Health check endpoint."""
        
        try:
            # TODO: Add actual health checks for dependencies
            return HealthCheckResponse(
                status="healthy",
                version=settings.app_version,
                uptime=0.0,  # TODO: Track actual uptime
                database_healthy=True,
                memory_healthy=True,
                tools_healthy=True,
                memory_usage={
                    "used": 0.0,
                    "available": 0.0,
                    "percent": 0.0,
                },
            )
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthCheckResponse(
                status="unhealthy",
                version=settings.app_version,
                uptime=0.0,
                database_healthy=False,
                memory_healthy=False,
                tools_healthy=False,
                memory_usage={},
            )
    
    @health_router.get("/ready")
    async def readiness_check():
        """Readiness check for Kubernetes."""
        try:
            # Check if agent is ready
            agent = await get_agent()
            return {"status": "ready", "agent_id": agent.agent_id}
        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Service not ready",
            )
    
    @health_router.get("/live")
    async def liveness_check():
        """Liveness check for Kubernetes."""
        return {"status": "alive", "timestamp": datetime.now().isoformat()}
    
    # Register and mount via APIRouterRegistry
    APIRouterRegistry.register_router(api_router)
    APIRouterRegistry.register_router(health_router)
    APIRouterRegistry.include_all(app)
    
    logger.info("API routes setup completed")