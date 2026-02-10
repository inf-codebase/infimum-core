"""
Middleware for the FastAPI application.
"""

import time
import uuid
from typing import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from ..config import get_settings
from ..utils.logging import get_logger


logger = get_logger(__name__)
settings = get_settings()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Add request context and timing information."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Add request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Add timestamp
        start_time = time.time()
        request.state.start_time = start_time
        
        # Log request
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host if request.client else None,
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path}",
                request_id=request_id,
                status_code=response.status_code,
                duration=duration,
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            logger.error(
                f"Request failed: {request.method} {request.url.path}",
                request_id=request_id,
                error=str(e),
                duration=duration,
            )
            
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Clean old entries
        current_time = time.time()
        self.clients = {
            ip: timestamps for ip, timestamps in self.clients.items()
            if any(ts > current_time - self.period for ts in timestamps)
        }
        
        # Check rate limit
        if client_ip in self.clients:
            # Filter recent requests
            recent_requests = [
                ts for ts in self.clients[client_ip]
                if ts > current_time - self.period
            ]
            
            if len(recent_requests) >= self.calls:
                logger.warning(
                    f"Rate limit exceeded for client {client_ip}",
                    client_ip=client_ip,
                    requests=len(recent_requests),
                    limit=self.calls,
                )
                
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: {self.calls} requests per {self.period} seconds",
                    headers={"Retry-After": str(self.period)},
                )
            
            # Add current request
            self.clients[client_ip] = recent_requests + [current_time]
        else:
            # First request from this client
            self.clients[client_ip] = [current_time]
        
        return await call_next(request)


def setup_middleware(app: FastAPI) -> None:
    """Setup all middleware for the FastAPI application."""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware (security)
    if settings.is_production():
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.allowed_hosts,
        )
    
    # Rate limiting
    if settings.rate_limit_requests > 0:
        app.add_middleware(
            RateLimitMiddleware,
            calls=settings.rate_limit_requests,
            period=60,  # 1 minute
        )
    
    # Request context (should be last)
    app.add_middleware(RequestContextMiddleware)
    
    logger.info("Middleware setup completed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    
    # Startup
    logger.info("AI Agent API starting up...")
    
    # Initialize any global resources here
    # For example, database connections, ML model loading, etc.
    
    logger.info("AI Agent API startup completed")
    
    yield
    
    # Shutdown
    logger.info("AI Agent API shutting down...")
    
    # Cleanup resources here
    # For example, close database connections, save state, etc.
    
    logger.info("AI Agent API shutdown completed")