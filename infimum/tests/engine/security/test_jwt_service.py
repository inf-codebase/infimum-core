"""Unit tests for core.engine.security.jwt_service module."""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

# Check if dependencies are available
try:
    from jose import jwt as jose_jwt
    import bcrypt
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="python-jose or bcrypt not installed")


class TestJWTService:
    """Test cases for JWTService class."""

    @pytest.fixture
    def jwt_service(self):
        """Create a fresh JWTService instance for each test."""
        from infimum.engine.security.jwt_service import JWTService
        from infimum.engine.security.token_blacklist import TokenBlacklistService
        
        # Create blacklist service without cleanup thread
        blacklist = TokenBlacklistService(
            secret_key="test-secret-key",
            algorithm="HS256",
            start_cleanup_thread=False
        )
        
        service = JWTService(
            secret_key="test-secret-key",
            algorithm="HS256",
            access_token_expire_minutes=30,
            refresh_token_expire_days=7,
            blacklist_service=blacklist
        )
        yield service
        blacklist.clear()

    @pytest.fixture
    def user_data(self):
        """Sample user data for testing."""
        return {
            "id": 123,
            "username": "testuser",
            "email": "test@example.com",
            "role": "admin",
            "roles": ["admin", "user"]
        }

    def test_create_access_token_returns_string(self, jwt_service, user_data):
        """Test that create_access_token returns a string."""
        token = jwt_service.create_access_token(user_data)
        
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_access_token_valid_jwt(self, jwt_service, user_data):
        """Test that create_access_token produces valid JWT."""
        from jose import jwt
        
        token = jwt_service.create_access_token(user_data)
        
        # Should be decodable
        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        
        assert payload["sub"] == "123"
        assert payload["username"] == "testuser"
        assert payload["email"] == "test@example.com"
        assert payload["role"] == "admin"
        assert payload["type"] == "access"
        assert "jti" in payload
        assert "exp" in payload
        assert "iat" in payload

    def test_create_access_token_with_custom_expiry(self, jwt_service, user_data):
        """Test create_access_token with custom expiration."""
        from jose import jwt
        
        expires_delta = timedelta(hours=2)
        token = jwt_service.create_access_token(user_data, expires_delta=expires_delta)
        
        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        
        # Token should expire in approximately 2 hours
        exp_time = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        expected_exp = datetime.now(timezone.utc) + expires_delta
        
        # Allow 5 second tolerance
        assert abs((exp_time - expected_exp).total_seconds()) < 5

    def test_create_refresh_token_returns_string(self, jwt_service, user_data):
        """Test that create_refresh_token returns a string."""
        token = jwt_service.create_refresh_token(user_data)
        
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_refresh_token_valid_jwt(self, jwt_service, user_data):
        """Test that create_refresh_token produces valid JWT with correct type."""
        from jose import jwt
        
        token = jwt_service.create_refresh_token(user_data)
        
        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        
        assert payload["sub"] == "123"
        assert payload["username"] == "testuser"
        assert payload["type"] == "refresh"
        assert "jti" in payload
        assert payload["jti"].endswith("-refresh")

    def test_create_token_pair(self, jwt_service, user_data):
        """Test create_token_pair returns both tokens."""
        tokens = jwt_service.create_token_pair(user_data)
        
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert tokens["token_type"] == "bearer"
        assert tokens["expires_in"] == 30 * 60  # 30 minutes in seconds

    def test_verify_token_valid(self, jwt_service, user_data):
        """Test verify_token with valid token."""
        token = jwt_service.create_access_token(user_data)
        
        payload = jwt_service.verify_token(token)
        
        assert payload is not None
        assert payload["sub"] == "123"
        assert payload["username"] == "testuser"

    def test_verify_token_invalid(self, jwt_service):
        """Test verify_token with invalid token."""
        payload = jwt_service.verify_token("invalid-token")
        
        assert payload is None

    def test_verify_token_wrong_secret(self, jwt_service, user_data):
        """Test verify_token with token signed by different secret."""
        from jose import jwt
        
        payload = {
            "sub": "123",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        }
        token = jwt.encode(payload, "different-secret", algorithm="HS256")
        
        result = jwt_service.verify_token(token)
        
        assert result is None

    def test_is_token_expired_not_expired(self, jwt_service):
        """Test is_token_expired with non-expired token."""
        payload = {
            "exp": (datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()
        }
        
        assert jwt_service.is_token_expired(payload) is False

    def test_is_token_expired_expired(self, jwt_service):
        """Test is_token_expired with expired token."""
        payload = {
            "exp": (datetime.now(timezone.utc) - timedelta(hours=1)).timestamp()
        }
        
        assert jwt_service.is_token_expired(payload) is True

    def test_is_token_expired_no_exp(self, jwt_service):
        """Test is_token_expired with no exp claim."""
        payload = {}
        
        assert jwt_service.is_token_expired(payload) is True

    def test_get_user_from_token_valid(self, jwt_service, user_data):
        """Test get_user_from_token with valid token."""
        token = jwt_service.create_access_token(user_data)
        
        result = jwt_service.get_user_from_token(token)
        
        assert result is not None
        assert result["success"] is True
        assert result["user_id"] == "123"
        assert result["username"] == "testuser"
        assert result["email"] == "test@example.com"
        assert result["role"] == "admin"
        assert result["token_type"] == "access"

    def test_get_user_from_token_blacklisted(self, jwt_service, user_data):
        """Test get_user_from_token with blacklisted token."""
        token = jwt_service.create_access_token(user_data)
        jwt_service.blacklist_token(token)
        
        result = jwt_service.get_user_from_token(token)
        
        assert result is None

    def test_get_user_from_token_invalid(self, jwt_service):
        """Test get_user_from_token with invalid token."""
        result = jwt_service.get_user_from_token("invalid-token")
        
        assert result is None

    def test_blacklist_token(self, jwt_service, user_data):
        """Test blacklist_token method."""
        token = jwt_service.create_access_token(user_data)
        
        result = jwt_service.blacklist_token(token)
        
        assert result is True
        assert jwt_service.get_user_from_token(token) is None

    def test_refresh_access_token_valid(self, jwt_service, user_data):
        """Test refresh_access_token with valid refresh token."""
        refresh_token = jwt_service.create_refresh_token(user_data)
        
        new_access_token = jwt_service.refresh_access_token(refresh_token)
        
        assert new_access_token is not None
        # Verify the new access token works
        result = jwt_service.get_user_from_token(new_access_token)
        assert result is not None
        assert result["user_id"] == "123"

    def test_refresh_access_token_with_access_token(self, jwt_service, user_data):
        """Test refresh_access_token fails with access token."""
        access_token = jwt_service.create_access_token(user_data)
        
        result = jwt_service.refresh_access_token(access_token)
        
        assert result is None

    def test_refresh_access_token_blacklisted(self, jwt_service, user_data):
        """Test refresh_access_token fails with blacklisted refresh token."""
        refresh_token = jwt_service.create_refresh_token(user_data)
        jwt_service.blacklist_token(refresh_token)
        
        result = jwt_service.refresh_access_token(refresh_token)
        
        assert result is None

    def test_refresh_access_token_invalid(self, jwt_service):
        """Test refresh_access_token with invalid token."""
        result = jwt_service.refresh_access_token("invalid-token")
        
        assert result is None

    def test_hash_password(self, jwt_service):
        """Test hash_password method (delegated to PasswordService)."""
        hashed = jwt_service.hash_password("test_password")
        
        assert isinstance(hashed, str)
        assert hashed.startswith("$2b$")

    def test_verify_password(self, jwt_service):
        """Test verify_password method (delegated to PasswordService)."""
        hashed = jwt_service.hash_password("test_password")
        
        assert jwt_service.verify_password("test_password", hashed) is True
        assert jwt_service.verify_password("wrong_password", hashed) is False


class TestJWTServiceConfiguration:
    """Test cases for JWTService configuration."""

    def test_default_access_token_expire_minutes(self):
        """Test default access token expiration."""
        from infimum.engine.security.jwt_service import JWTService
        
        service = JWTService(secret_key="test")
        
        # Default should be 30 minutes
        assert service.access_token_expire_minutes == 30

    def test_default_refresh_token_expire_days(self):
        """Test default refresh token expiration."""
        from infimum.engine.security.jwt_service import JWTService
        
        service = JWTService(secret_key="test")
        
        # Default should be 7 days
        assert service.refresh_token_expire_days == 7

    def test_custom_expire_times(self):
        """Test custom expiration times."""
        from infimum.engine.security.jwt_service import JWTService
        
        service = JWTService(
            secret_key="test",
            access_token_expire_minutes=60,
            refresh_token_expire_days=14
        )
        
        assert service.access_token_expire_minutes == 60
        assert service.refresh_token_expire_days == 14


class TestJWTServiceSingleton:
    """Test cases for singleton instance management."""

    def test_get_jwt_service_singleton(self):
        """Test that get_jwt_service returns singleton."""
        from infimum.engine.security.jwt_service import get_jwt_service, reset_jwt_service
        
        reset_jwt_service()
        
        service1 = get_jwt_service()
        service2 = get_jwt_service()
        
        assert service1 is service2
        
        reset_jwt_service()

    def test_reset_jwt_service(self):
        """Test that reset_jwt_service clears the singleton."""
        from infimum.engine.security.jwt_service import get_jwt_service, reset_jwt_service
        
        service1 = get_jwt_service()
        reset_jwt_service()
        service2 = get_jwt_service()
        
        assert service1 is not service2
        
        reset_jwt_service()


class TestJWTServiceEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.fixture
    def jwt_service(self):
        """Create a fresh JWTService instance."""
        from infimum.engine.security.jwt_service import JWTService
        from infimum.engine.security.token_blacklist import TokenBlacklistService
        
        blacklist = TokenBlacklistService(
            secret_key="test-secret-key",
            algorithm="HS256",
            start_cleanup_thread=False
        )
        
        return JWTService(
            secret_key="test-secret-key",
            algorithm="HS256",
            blacklist_service=blacklist
        )

    def test_user_data_with_minimal_fields(self, jwt_service):
        """Test token creation with minimal user data."""
        user_data = {"id": 1}
        
        token = jwt_service.create_access_token(user_data)
        result = jwt_service.get_user_from_token(token)
        
        assert result is not None
        assert result["user_id"] == "1"
        assert result["username"] == ""

    def test_user_data_with_string_id(self, jwt_service):
        """Test token creation with string user ID."""
        user_data = {"id": "uuid-string-123", "username": "testuser"}
        
        token = jwt_service.create_access_token(user_data)
        result = jwt_service.get_user_from_token(token)
        
        assert result["user_id"] == "uuid-string-123"

    def test_user_data_with_multiple_roles(self, jwt_service):
        """Test token creation with multiple roles."""
        user_data = {
            "id": 1,
            "username": "admin",
            "roles": ["admin", "moderator", "user"]
        }
        
        token = jwt_service.create_access_token(user_data)
        result = jwt_service.get_user_from_token(token)
        
        assert result["roles"] == ["admin", "moderator", "user"]

    def test_unique_jti_per_token(self, jwt_service):
        """Test that each token gets a unique JTI."""
        from jose import jwt
        import time
        
        user_data = {"id": 1, "username": "test"}
        
        jtis = set()
        for _ in range(10):
            token = jwt_service.create_access_token(user_data)
            payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
            jtis.add(payload["jti"])
            time.sleep(0.001)  # Small delay to ensure timestamp difference
        
        # All JTIs should be unique
        assert len(jtis) == 10
