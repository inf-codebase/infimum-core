"""
Production configuration settings for the AI Agent v2.
Uses pydantic-settings for type-safe configuration management.
"""

import os
from functools import lru_cache
from typing import List, Optional, Dict, Any

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Production configuration settings."""
    
    # Application settings
    app_name: str = Field(default="AI Agent v2", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment (development/staging/production)")
    debug: bool = Field(default=False, description="Debug mode")
    
    # OpenAI Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4", description="Default OpenAI model")
    openai_temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Model temperature")
    openai_max_tokens: int = Field(default=2048, ge=1, le=4096, description="Max tokens per response")
    openai_timeout: int = Field(default=60, ge=1, description="OpenAI API timeout in seconds")
    
    # Agent Configuration
    max_iterations: int = Field(default=10, ge=1, le=50, description="Max agent iterations")
    max_execution_time: int = Field(default=300, ge=30, description="Max execution time in seconds")
    max_retry_attempts: int = Field(default=3, ge=1, description="Max retry attempts")
    
    # Memory Configuration
    memory_type: str = Field(default="buffer", description="Memory type (buffer/summary/vector)")
    memory_max_tokens: int = Field(default=2000, ge=100, description="Max tokens in memory")
    vector_store_type: str = Field(default="chroma", description="Vector store type")
    embedding_model: str = Field(default="text-embedding-ada-002", description="Embedding model")
    
    # Search Configuration
    serp_api_key: Optional[str] = Field(default=None, description="SerpAPI key for Google search")
    search_results_limit: int = Field(default=5, ge=1, le=20, description="Max search results")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, ge=1, le=65535, description="API port")
    api_workers: int = Field(default=1, ge=1, description="Number of API workers")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json/text)")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    # Monitoring Configuration
    enable_tracing: bool = Field(default=False, description="Enable LangSmith tracing")
    langsmith_project: Optional[str] = Field(default=None, description="LangSmith project name")
    langsmith_api_key: Optional[str] = Field(default=None, description="LangSmith API key")
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100, ge=1, description="Requests per minute")
    rate_limit_burst: int = Field(default=20, ge=1, description="Burst limit")
    
    # Security
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    allowed_hosts: List[str] = Field(default=["*"], description="Allowed hosts")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"
        
    @validator("environment")
    def validate_environment(cls, v: str) -> str:
        allowed = ["development", "staging", "production"]
        if v.lower() not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v.lower()
    
    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of: {allowed}")
        return v.upper()
    
    @validator("openai_api_key")
    def validate_openai_api_key(cls, v: str) -> str:
        # Allow empty string for development, but warn about it
        if not v:
            import warnings
            warnings.warn("OpenAI API key is not set. Some features may not work.", UserWarning)
            return v
        if not v.startswith("sk-"):
            raise ValueError("Valid OpenAI API key is required (starts with 'sk-')")
        return v
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration."""
        return {
            "api_key": self.openai_api_key,
            "model": self.openai_model,
            "temperature": self.openai_temperature,
            "max_tokens": self.openai_max_tokens,
            "timeout": self.openai_timeout,
        }
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory configuration."""
        return {
            "type": self.memory_type,
            "max_tokens": self.memory_max_tokens,
            "vector_store_type": self.vector_store_type,
            "embedding_model": self.embedding_model,
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()