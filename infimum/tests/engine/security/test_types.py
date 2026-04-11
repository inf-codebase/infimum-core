"""Unit tests for core.engine.security.types module."""
import pytest
from typing import get_type_hints


class TestUserDataTypedDict:
    """Test cases for UserData TypedDict."""

    def test_user_data_can_be_created(self):
        """Test that UserData dict can be created with expected fields."""
        from infimum.engine.security.types import UserData
        
        user: UserData = {
            "id": 123,
            "username": "testuser",
            "email": "test@example.com",
            "role": "admin",
            "roles": ["admin", "user"],
            "first_name": "Test",
            "last_name": "User",
            "disabled": False,
            "email_verified": True
        }
        
        assert user["id"] == 123
        assert user["username"] == "testuser"
        assert user["email"] == "test@example.com"

    def test_user_data_partial(self):
        """Test that UserData can be partial (total=False)."""
        from infimum.engine.security.types import UserData
        
        # Should work with just some fields
        user: UserData = {
            "id": 1,
            "username": "test"
        }
        
        assert user["id"] == 1

    def test_user_data_with_different_id_types(self):
        """Test UserData accepts different ID types."""
        from infimum.engine.security.types import UserData
        
        # Integer ID
        user1: UserData = {"id": 123}
        assert user1["id"] == 123
        
        # String ID
        user2: UserData = {"id": "uuid-string"}
        assert user2["id"] == "uuid-string"


class TestTokenPayloadTypedDict:
    """Test cases for TokenPayload TypedDict."""

    def test_token_payload_can_be_created(self):
        """Test that TokenPayload dict can be created."""
        from infimum.engine.security.types import TokenPayload
        from datetime import datetime, timezone
        
        payload: TokenPayload = {
            "sub": "123",
            "username": "testuser",
            "email": "test@example.com",
            "role": "admin",
            "exp": datetime.now(timezone.utc),
            "iat": datetime.now(timezone.utc),
            "jti": "unique-token-id",
            "type": "access"
        }
        
        assert payload["sub"] == "123"
        assert payload["type"] == "access"

    def test_token_payload_partial(self):
        """Test that TokenPayload can be partial."""
        from infimum.engine.security.types import TokenPayload
        
        payload: TokenPayload = {
            "sub": "123",
            "type": "refresh"
        }
        
        assert payload["type"] == "refresh"


class TestTokenResponseTypedDict:
    """Test cases for TokenResponse TypedDict."""

    def test_token_response_can_be_created(self):
        """Test that TokenResponse dict can be created."""
        from infimum.engine.security.types import TokenResponse
        
        response: TokenResponse = {
            "access_token": "eyJ...",
            "refresh_token": "eyJ...",
            "token_type": "bearer",
            "expires_in": 1800
        }
        
        assert response["token_type"] == "bearer"
        assert response["expires_in"] == 1800


class TestAuthResultTypedDict:
    """Test cases for AuthResult TypedDict."""

    def test_auth_result_success(self):
        """Test AuthResult for successful auth."""
        from infimum.engine.security.types import AuthResult
        
        result: AuthResult = {
            "success": True,
            "user_id": "123",
            "username": "testuser",
            "email": "test@example.com",
            "role": "admin",
            "roles": ["admin", "user"],
            "token_type": "access"
        }
        
        assert result["success"] is True
        assert result["user_id"] == "123"

    def test_auth_result_failure(self):
        """Test AuthResult for failed auth."""
        from infimum.engine.security.types import AuthResult
        
        result: AuthResult = {
            "success": False,
            "error": "Invalid credentials",
            "message": "Please check your username and password"
        }
        
        assert result["success"] is False
        assert result["error"] == "Invalid credentials"


class TestEmailDataTypedDict:
    """Test cases for EmailData TypedDict."""

    def test_email_data_can_be_created(self):
        """Test that EmailData dict can be created."""
        from infimum.engine.security.types import EmailData
        
        data: EmailData = {
            "to_email": "user@example.com",
            "subject": "Welcome",
            "user_name": "John",
            "token": "abc123",
            "link": "https://example.com/verify?token=abc123"
        }
        
        assert data["to_email"] == "user@example.com"
        assert data["link"].startswith("https://")


class TestBlacklistStatsTypedDict:
    """Test cases for BlacklistStats TypedDict."""

    def test_blacklist_stats_can_be_created(self):
        """Test that BlacklistStats dict can be created."""
        from infimum.engine.security.types import BlacklistStats
        
        stats: BlacklistStats = {
            "total_blacklisted_tokens": 100,
            "active_blacklisted_tokens": 50,
            "expired_blacklisted_tokens": 50
        }
        
        assert stats["total_blacklisted_tokens"] == 100
        assert stats["active_blacklisted_tokens"] + stats["expired_blacklisted_tokens"] == 100


class TestTypeImports:
    """Test that all types can be imported from the module."""

    def test_all_types_importable(self):
        """Test that all types can be imported."""
        from infimum.engine.security.types import (
            UserData,
            TokenPayload,
            TokenResponse,
            AuthResult,
            EmailData,
            BlacklistStats
        )
        
        # All should be TypedDict classes
        assert UserData is not None
        assert TokenPayload is not None
        assert TokenResponse is not None
        assert AuthResult is not None
        assert EmailData is not None
        assert BlacklistStats is not None

    def test_types_in_module_all(self):
        """Test that types are exported in __all__."""
        from infimum.engine.security import (
            UserData,
            TokenPayload,
            TokenResponse,
            AuthResult,
            EmailData,
            BlacklistStats
        )
        
        # Should be importable from main security module
        assert UserData is not None
