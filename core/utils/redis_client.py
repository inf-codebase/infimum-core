#backend\src\core\utils\redis_client.py
"""Reusable Redis client utilities for the FastVLM backend.

This module exposes a lazily-initialised asyncio Redis client that shares a
connection pool across the application.  It also provides basic health checks
and retry helpers so other services can rely on a single entry point for Redis
connectivity.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from redis.asyncio import Redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import RedisError

from core.utils import auto_config

logger = logging.getLogger(__name__)


def _to_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _to_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


class RedisSettings:
    """Runtime configuration for Redis connectivity."""

    @property
    def host(self) -> str:
        return getattr(auto_config, "REDIS_HOST", "localhost")

    @property
    def port(self) -> int:
        return _to_int(getattr(auto_config, "REDIS_PORT", 6379), 6379)

    @property
    def db(self) -> int:
        return _to_int(getattr(auto_config, "REDIS_DB", 0), 0)

    @property
    def password(self) -> Optional[str]:
        pwd = getattr(auto_config, "REDIS_PASSWORD", "")
        return pwd if pwd else None

    @property
    def url(self) -> str:
        return getattr(auto_config, "REDIS_URL", f"redis://{self.host}:{self.port}/{self.db}")

    @property
    def max_connections(self) -> int:
        return _to_int(getattr(auto_config, "REDIS_POOL_MAX_CONNECTIONS", 20), 20)

    @property
    def retry_max_attempts(self) -> int:
        return _to_int(getattr(auto_config, "REDIS_RETRY_MAX_ATTEMPTS", 5), 5)

    @property
    def retry_base_delay(self) -> float:
        return _to_float(getattr(auto_config, "REDIS_RETRY_BASE_DELAY", 0.5), 0.5)


_settings = RedisSettings()
_pool: Optional[ConnectionPool] = None
_client: Optional[Redis] = None
_pool_lock = asyncio.Lock()


async def _get_or_create_client() -> Redis:
    global _pool, _client

    if _client is not None:
        return _client

    async with _pool_lock:
        if _client is not None:
            return _client

        logger.debug(
            "Initialising Redis connection pool", extra={"host": _settings.host, "port": _settings.port}
        )
        _pool = ConnectionPool.from_url(
            _settings.url,
            max_connections=_settings.max_connections,
            # ``decode_responses`` is disabled so callers receive ``bytes``.
            # Redis' asyncio encoder, however, still expects a real encoding
            # name for command payloads.  Passing ``None`` (which disables
            # encoding) leads to ``TypeError: encode() argument 'encoding' must
            # be str, not None`` when strings are written to the connection.
            # Using the default ``utf-8`` keeps the payload handling simple
            # without changing the response semantics.
            encoding="utf-8",
            decode_responses=False,
        )
        _client = Redis(connection_pool=_pool)

    return _client


async def get_client() -> Redis:
    """Return a shared asyncio Redis client instance."""

    return await _get_or_create_client()


async def ping() -> bool:
    """Check Redis availability using the PING command."""

    client = await get_client()
    try:
        response = await client.ping()
        return bool(response)
    except RedisError as exc:  # pragma: no cover - logging path
        logger.warning("Redis ping failed", exc_info=exc)
        return False


async def wait_for_connection() -> bool:
    """Attempt to connect to Redis with exponential backoff."""

    for attempt in range(_settings.retry_max_attempts):
        if await ping():
            return True

        delay = _settings.retry_base_delay * (2 ** attempt)
        logger.debug(
            "Redis connection attempt failed; retrying", extra={"attempt": attempt + 1, "delay": delay}
        )
        await asyncio.sleep(delay)

    return False


async def check_health() -> dict:
    """Return a health status payload suitable for API responses."""

    healthy = await ping()
    status = "ok" if healthy else "unhealthy"
    return {
        "status": status,
        "host": _settings.host,
        "port": _settings.port,
        "database": _settings.db,
    }


async def close_connections() -> None:
    """Close the shared Redis connection pool."""

    global _pool, _client

    async with _pool_lock:
        if _client is not None:
            await _client.close()
            _client = None

        if _pool is not None:
            await _pool.disconnect()
            _pool = None


def get_settings() -> RedisSettings:
    """Expose the currently loaded Redis settings."""

    return _settings


__all__ = [
    "RedisSettings",
    "get_client",
    "get_settings",
    "check_health",
    "wait_for_connection",
    "close_connections",
]
