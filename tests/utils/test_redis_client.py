"""Unit tests for core.utils.redis_client module."""
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from core.utils.redis_client import (
    RedisSettings,
    get_client,
    ping,
    wait_for_connection,
    check_health,
    close_connections,
    get_settings
)


class TestRedisSettings:
    """Test cases for RedisSettings class."""

    @patch('core.utils.redis_client.auto_config')
    def test_host_property(self, mock_config):
        """Test host property."""
        mock_config.REDIS_HOST = "test_host"
        settings = RedisSettings()
        assert settings.host == "test_host"

    @patch('core.utils.redis_client.auto_config')
    def test_port_property(self, mock_config):
        """Test port property."""
        mock_config.REDIS_PORT = "6379"
        settings = RedisSettings()
        assert settings.port == 6379

    @patch('core.utils.redis_client.auto_config')
    def test_db_property(self, mock_config):
        """Test db property."""
        mock_config.REDIS_DB = "1"
        settings = RedisSettings()
        assert settings.db == 1

    @patch('core.utils.redis_client.auto_config')
    def test_password_property(self, mock_config):
        """Test password property."""
        mock_config.REDIS_PASSWORD = "test_pass"
        settings = RedisSettings()
        assert settings.password == "test_pass"

    @patch('core.utils.redis_client.auto_config')
    def test_password_empty(self, mock_config):
        """Test password property returns None for empty string."""
        mock_config.REDIS_PASSWORD = ""
        settings = RedisSettings()
        assert settings.password is None

    @patch('core.utils.redis_client.auto_config')
    def test_url_property(self, mock_config):
        """Test url property."""
        mock_config.REDIS_URL = "redis://localhost:6379/0"
        settings = RedisSettings()
        assert settings.url == "redis://localhost:6379/0"

    @patch('core.utils.redis_client.auto_config')
    def test_max_connections_property(self, mock_config):
        """Test max_connections property."""
        mock_config.REDIS_POOL_MAX_CONNECTIONS = "20"
        settings = RedisSettings()
        assert settings.max_connections == 20


class TestGetClient:
    """Test cases for get_client function."""

    @pytest.mark.asyncio
    @patch('core.utils.redis_client._get_or_create_client')
    async def test_get_client(self, mock_get_client):
        """Test get_client returns client."""
        mock_client = AsyncMock()
        mock_get_client.return_value = mock_client
        
        result = await get_client()
        
        assert result == mock_client


class TestPing:
    """Test cases for ping function."""

    @pytest.mark.asyncio
    @patch('core.utils.redis_client.get_client')
    async def test_ping_success(self, mock_get_client):
        """Test ping returns True on success."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_get_client.return_value = mock_client
        
        result = await ping()
        
        assert result is True
        mock_client.ping.assert_called_once()

    @pytest.mark.asyncio
    @patch('core.utils.redis_client.get_client')
    async def test_ping_failure(self, mock_get_client):
        """Test ping returns False on failure."""
        from redis.exceptions import RedisError
        mock_client = AsyncMock()
        mock_client.ping.side_effect = RedisError("Connection error")
        mock_get_client.return_value = mock_client
        
        result = await ping()
        
        assert result is False


class TestWaitForConnection:
    """Test cases for wait_for_connection function."""

    @pytest.mark.asyncio
    @patch('core.utils.redis_client.ping')
    async def test_wait_for_connection_success(self, mock_ping):
        """Test wait_for_connection succeeds immediately."""
        mock_ping.return_value = True
        
        result = await wait_for_connection()
        
        assert result is True

    @pytest.mark.asyncio
    @patch('core.utils.redis_client.ping')
    @patch('core.utils.redis_client.asyncio.sleep')
    async def test_wait_for_connection_retry(self, mock_sleep, mock_ping):
        """Test wait_for_connection retries on failure."""
        mock_ping.side_effect = [False, False, True]
        mock_sleep.return_value = None
        
        result = await wait_for_connection()
        
        assert result is True
        assert mock_ping.call_count == 3


class TestCheckHealth:
    """Test cases for check_health function."""

    @pytest.mark.asyncio
    @patch('core.utils.redis_client.ping')
    @patch('core.utils.redis_client._settings')
    async def test_check_health_ok(self, mock_settings, mock_ping):
        """Test check_health returns ok status."""
        mock_ping.return_value = True
        mock_settings.host = "localhost"
        mock_settings.port = 6379
        mock_settings.db = 0
        
        result = await check_health()
        
        assert result["status"] == "ok"
        assert result["host"] == "localhost"
        assert result["port"] == 6379

    @pytest.mark.asyncio
    @patch('core.utils.redis_client.ping')
    @patch('core.utils.redis_client._settings')
    async def test_check_health_unhealthy(self, mock_settings, mock_ping):
        """Test check_health returns unhealthy status."""
        mock_ping.return_value = False
        mock_settings.host = "localhost"
        mock_settings.port = 6379
        mock_settings.db = 0
        
        result = await check_health()
        
        assert result["status"] == "unhealthy"


class TestCloseConnections:
    """Test cases for close_connections function."""

    @pytest.mark.asyncio
    async def test_close_connections(self):
        """Test close_connections closes client and pool."""
        mock_client_instance = AsyncMock()
        mock_pool_instance = AsyncMock()
        mock_lock_instance = AsyncMock()
        mock_lock_instance.__aenter__ = AsyncMock(return_value=None)
        mock_lock_instance.__aexit__ = AsyncMock(return_value=None)
        
        # Set up mocks
        import core.utils.redis_client as redis_module
        original_client = redis_module._client
        original_pool = redis_module._pool
        original_lock = redis_module._pool_lock
        
        try:
            redis_module._client = mock_client_instance
            redis_module._pool = mock_pool_instance
            redis_module._pool_lock = mock_lock_instance
            
            await close_connections()
            
            mock_client_instance.close.assert_called_once()
            mock_pool_instance.disconnect.assert_called_once()
        finally:
            # Restore original values
            redis_module._client = original_client
            redis_module._pool = original_pool
            redis_module._pool_lock = original_lock


class TestGetSettings:
    """Test cases for get_settings function."""

    def test_get_settings(self):
        """Test get_settings returns settings instance."""
        result = get_settings()
        assert isinstance(result, RedisSettings)
