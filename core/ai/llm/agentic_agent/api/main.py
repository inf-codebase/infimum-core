"""
FastAPI application factory for the AI Agent v2.
"""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from .middleware import setup_middleware, lifespan
from .routes import setup_routes
from ..config import get_settings
from ..utils.logging import setup_logging, get_logger


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    
    settings = get_settings()
    
    # Setup logging
    setup_logging(
        level=settings.log_level,
        format_type=settings.log_format,
        file_path=settings.log_file,
    )
    
    logger = get_logger(__name__)
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        description="Production-ready AI Agent built with LangChain and LangGraph",
        version=settings.app_version,
        docs_url="/docs" if not settings.is_production() else None,
        redoc_url="/redoc" if not settings.is_production() else None,
        openapi_url="/openapi.json" if not settings.is_production() else None,
        lifespan=lifespan,
    )
    
    # Setup middleware
    setup_middleware(app)
    
    # Setup routes
    setup_routes(app)
    
    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint - redirect to docs."""
        if not settings.is_production():
            return RedirectResponse(url="/docs")
        return {
            "message": f"Welcome to {settings.app_name} v{settings.app_version}",
            "status": "running",
            "docs": "Documentation not available in production mode",
        }
    
    logger.info(f"FastAPI application created: {settings.app_name} v{settings.app_version}")
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
        reload=settings.is_development(),
    )