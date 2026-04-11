"""Unit tests for core.engine.security.token_blacklist module."""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

# Check if jose is available
try:
    from jose import jwt as jose_jwt
    HAS_JOSE = True
except ImportError:
    HAS_JOSE = False

pytestmark = pytest.mark.skipif(not HAS_JOSE, reason="python-jose not installed")


class TestTokenBlacklistService:
    """Test cases for TokenBlacklistService class."""

    @pytest.fixture
    def blacklist_service(self):
        """Create a fresh TokenBlacklistService instance for each test."""
        from infimum.engine.security.token_blacklist import TokenBlacklistService
        
        # Create service without starting cleanup thread
        service = TokenBlacklistService(
            secret_key="test-secret-key",
            algorithm="HS256",
            start_cleanup_thread=False
        )
        yield service
        service.clear()

    @pytest.fixture
    def jwt_token(self):
        """Create a valid JWT token for testing."""
        from jose import jwt
        from datetime import datetime, timezone, timedelta
        
        payload = {
            "sub": "123",
            "username": "testuser",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "iat": datetime.now(timezone.utc),
            "jti": "test-jti-12345",
            "type": "access"
        }
        return jwt.encode(payload, "test-secret-key", algorithm="HS256")

    @pytest.fixture
    def expired_jwt_token(self):
        """Create an expired JWT token for testing."""
        from jose import jwt
        
        payload = {
            "sub": "123",
            "username": "testuser",
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),  # Expired
            "iat": datetime.now(timezone.utc) - timedelta(hours=2),
            "jti": "expired-jti-12345",
            "type": "access"
        }
        return jwt.encode(payload, "test-secret-key", algorithm="HS256")

    def test_blacklist_token_success(self, blacklist_service, jwt_token):
        """Test that blacklist_token successfully blacklists a valid token."""
        result = blacklist_service.blacklist_token(jwt_token)
        
        assert result is True
        assert blacklist_service.is_blacklisted(jwt_token) is True

    def test_blacklist_token_already_blacklisted(self, blacklist_service, jwt_token):
        """Test that blacklisting an already blacklisted token returns True."""
        blacklist_service.blacklist_token(jwt_token)
        result = blacklist_service.blacklist_token(jwt_token)
        
        assert result is True

    def test_blacklist_token_invalid_token(self, blacklist_service):
        """Test that blacklist_token returns False for invalid token."""
        result = blacklist_service.blacklist_token("invalid-token")
        
        assert result is False

    def test_blacklist_token_missing_jti(self, blacklist_service):
        """Test that blacklist_token returns False for token without JTI."""
        from jose import jwt
        
        payload = {
            "sub": "123",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            # No jti
        }
        token = jwt.encode(payload, "test-secret-key", algorithm="HS256")
        
        result = blacklist_service.blacklist_token(token)
        assert result is False

    def test_is_blacklisted_not_blacklisted(self, blacklist_service, jwt_token):
        """Test that is_blacklisted returns False for non-blacklisted token."""
        result = blacklist_service.is_blacklisted(jwt_token)
        
        assert result is False

    def test_is_blacklisted_after_blacklisting(self, blacklist_service, jwt_token):
        """Test that is_blacklisted returns True after blacklisting."""
        blacklist_service.blacklist_token(jwt_token)
        result = blacklist_service.is_blacklisted(jwt_token)
        
        assert result is True

    def test_is_blacklisted_invalid_token(self, blacklist_service):
        """Test that is_blacklisted returns False for invalid token."""
        # Invalid tokens should not be treated as blacklisted
        # (let JWT verification handle the invalid token)
        result = blacklist_service.is_blacklisted("invalid-token")
        
        assert result is False

    def test_is_jti_blacklisted(self, blacklist_service, jwt_token):
        """Test is_jti_blacklisted method."""
        assert blacklist_service.is_jti_blacklisted("test-jti-12345") is False
        
        blacklist_service.blacklist_token(jwt_token)
        
        assert blacklist_service.is_jti_blacklisted("test-jti-12345") is True

    def test_cleanup_expired_tokens(self, blacklist_service):
        """Test that cleanup_expired_tokens removes expired tokens."""
        from jose import jwt
        
        # Add an expired token to blacklist manually
        expired_jti = "expired-test-jti"
        expired_exp = datetime.now(timezone.utc) - timedelta(hours=1)
        blacklist_service._blacklisted_tokens[expired_jti] = expired_exp.timestamp()
        
        # Add a valid token
        valid_jti = "valid-test-jti"
        valid_exp = datetime.now(timezone.utc) + timedelta(hours=1)
        blacklist_service._blacklisted_tokens[valid_jti] = valid_exp.timestamp()
        
        # Cleanup
        cleaned = blacklist_service.cleanup_expired_tokens()
        
        assert cleaned == 1
        assert expired_jti not in blacklist_service._blacklisted_tokens
        assert valid_jti in blacklist_service._blacklisted_tokens

    def test_force_cleanup(self, blacklist_service):
        """Test force_cleanup method."""
        # Add an expired token
        expired_jti = "force-cleanup-jti"
        expired_exp = datetime.now(timezone.utc) - timedelta(hours=1)
        blacklist_service._blacklisted_tokens[expired_jti] = expired_exp.timestamp()
        
        cleaned = blacklist_service.force_cleanup()
        
        assert cleaned == 1

    def test_get_stats_empty(self, blacklist_service):
        """Test get_stats with empty blacklist."""
        stats = blacklist_service.get_stats()
        
        assert stats["total_blacklisted_tokens"] == 0
        assert stats["active_blacklisted_tokens"] == 0
        assert stats["expired_blacklisted_tokens"] == 0

    def test_get_stats_with_tokens(self, blacklist_service, jwt_token):
        """Test get_stats with blacklisted tokens."""
        blacklist_service.blacklist_token(jwt_token)
        
        # Add an expired token manually
        expired_jti = "stats-expired-jti"
        expired_exp = datetime.now(timezone.utc) - timedelta(hours=1)
        blacklist_service._blacklisted_tokens[expired_jti] = expired_exp.timestamp()
        
        stats = blacklist_service.get_stats()
        
        assert stats["total_blacklisted_tokens"] == 2
        assert stats["active_blacklisted_tokens"] == 1
        assert stats["expired_blacklisted_tokens"] == 1

    def test_clear(self, blacklist_service, jwt_token):
        """Test clear method removes all tokens."""
        blacklist_service.blacklist_token(jwt_token)
        assert len(blacklist_service._blacklisted_tokens) == 1
        
        blacklist_service.clear()
        
        assert len(blacklist_service._blacklisted_tokens) == 0

    def test_secret_key_property(self, blacklist_service):
        """Test secret_key property returns configured value."""
        assert blacklist_service.secret_key == "test-secret-key"

    def test_algorithm_property(self, blacklist_service):
        """Test algorithm property returns configured value."""
        assert blacklist_service.algorithm == "HS256"


class TestTokenBlacklistSingleton:
    """Test cases for singleton instance management."""

    def test_get_token_blacklist_service_singleton(self):
        """Test that get_token_blacklist_service returns singleton."""
        from infimum.engine.security.token_blacklist import (
            get_token_blacklist_service,
            reset_blacklist_service
        )
        
        # Reset first to ensure clean state
        reset_blacklist_service()
        
        service1 = get_token_blacklist_service()
        service2 = get_token_blacklist_service()
        
        assert service1 is service2
        
        # Cleanup
        reset_blacklist_service()

    def test_reset_blacklist_service(self):
        """Test that reset_blacklist_service clears the singleton."""
        from infimum.engine.security.token_blacklist import (
            get_token_blacklist_service,
            reset_blacklist_service
        )
        
        service1 = get_token_blacklist_service()
        reset_blacklist_service()
        service2 = get_token_blacklist_service()
        
        assert service1 is not service2
        
        # Cleanup
        reset_blacklist_service()


class TestTokenBlacklistThreadSafety:
    """Test cases for thread safety."""

    def test_concurrent_blacklist(self):
        """Test that blacklisting tokens is thread-safe."""
        from infimum.engine.security.token_blacklist import TokenBlacklistService
        from jose import jwt
        import threading
        
        service = TokenBlacklistService(
            secret_key="test-secret",
            algorithm="HS256",
            start_cleanup_thread=False
        )
        
        tokens = []
        for i in range(100):
            payload = {
                "sub": str(i),
                "exp": datetime.now(timezone.utc) + timedelta(hours=1),
                "iat": datetime.now(timezone.utc),
                "jti": f"concurrent-jti-{i}",
            }
            tokens.append(jwt.encode(payload, "test-secret", algorithm="HS256"))
        
        errors = []
        
        def blacklist_token(token):
            try:
                service.blacklist_token(token)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=blacklist_token, args=(t,)) for t in tokens]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(service._blacklisted_tokens) == 100
        
        service.clear()
