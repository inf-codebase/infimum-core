"""Unit tests for core.engine.security.password module."""
import pytest
from unittest.mock import patch, MagicMock

# Check if bcrypt is available
try:
    import bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False

pytestmark = pytest.mark.skipif(not HAS_BCRYPT, reason="bcrypt not installed")


class TestPasswordService:
    """Test cases for PasswordService class."""

    def test_hash_password_returns_string(self):
        """Test that hash_password returns a string."""
        from core.engine.security.password import PasswordService
        
        service = PasswordService()
        result = service.hash_password("test_password")
        
        assert isinstance(result, str)
        assert len(result) > 0

    def test_hash_password_produces_bcrypt_hash(self):
        """Test that hash_password produces a valid bcrypt hash."""
        from core.engine.security.password import PasswordService
        
        service = PasswordService()
        result = service.hash_password("test_password")
        
        # bcrypt hashes start with $2b$ or $2a$
        assert result.startswith("$2b$") or result.startswith("$2a$")

    def test_hash_password_different_for_same_input(self):
        """Test that hashing the same password twice produces different hashes (due to salt)."""
        from core.engine.security.password import PasswordService
        
        service = PasswordService()
        hash1 = service.hash_password("test_password")
        hash2 = service.hash_password("test_password")
        
        assert hash1 != hash2

    def test_hash_password_truncates_long_passwords(self):
        """Test that passwords longer than 72 bytes are truncated."""
        from core.engine.security.password import PasswordService
        
        service = PasswordService()
        long_password = "a" * 100  # 100 bytes, exceeds 72
        
        # Should not raise an error
        result = service.hash_password(long_password)
        assert isinstance(result, str)

    def test_verify_password_correct(self):
        """Test that verify_password returns True for correct password."""
        from core.engine.security.password import PasswordService
        
        service = PasswordService()
        password = "secure_password_123"
        hashed = service.hash_password(password)
        
        assert service.verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test that verify_password returns False for incorrect password."""
        from core.engine.security.password import PasswordService
        
        service = PasswordService()
        password = "secure_password_123"
        hashed = service.hash_password(password)
        
        assert service.verify_password("wrong_password", hashed) is False

    def test_verify_password_empty_password(self):
        """Test that verify_password handles empty password."""
        from core.engine.security.password import PasswordService
        
        service = PasswordService()
        hashed = service.hash_password("some_password")
        
        assert service.verify_password("", hashed) is False

    def test_verify_password_invalid_hash(self):
        """Test that verify_password returns False for invalid hash."""
        from core.engine.security.password import PasswordService
        
        service = PasswordService()
        
        assert service.verify_password("password", "invalid_hash") is False

    def test_generate_token_default_length(self):
        """Test that generate_token produces token of default length (32)."""
        from core.engine.security.password import PasswordService
        
        service = PasswordService()
        token = service.generate_token()
        
        assert len(token) == 32

    def test_generate_token_custom_length(self):
        """Test that generate_token produces token of custom length."""
        from core.engine.security.password import PasswordService
        
        service = PasswordService()
        
        for length in [16, 64, 128]:
            token = service.generate_token(length)
            assert len(token) == length

    def test_generate_token_unique(self):
        """Test that generate_token produces unique tokens."""
        from core.engine.security.password import PasswordService
        
        service = PasswordService()
        tokens = [service.generate_token() for _ in range(100)]
        
        # All tokens should be unique
        assert len(set(tokens)) == 100

    def test_generate_token_url_safe(self):
        """Test that generate_token produces URL-safe characters."""
        from core.engine.security.password import PasswordService
        import string
        
        service = PasswordService()
        token = service.generate_token(1000)  # Generate long token to test character set
        
        allowed_chars = set(string.ascii_letters + string.digits + '-_')
        assert all(c in allowed_chars for c in token)

    def test_generate_hex_token(self):
        """Test that generate_hex_token produces valid hex string."""
        from core.engine.security.password import PasswordService
        
        service = PasswordService()
        token = service.generate_hex_token(16)
        
        # 16 bytes = 32 hex characters
        assert len(token) == 32
        # Should be valid hex
        int(token, 16)  # This will raise if not valid hex

    def test_generate_urlsafe_token(self):
        """Test that generate_urlsafe_token produces URL-safe base64."""
        from core.engine.security.password import PasswordService
        
        service = PasswordService()
        token = service.generate_urlsafe_token(32)
        
        assert isinstance(token, str)
        assert len(token) > 0


class TestModuleFunctions:
    """Test cases for module-level functions."""

    def test_get_password_service_singleton(self):
        """Test that get_password_service returns singleton instance."""
        from core.engine.security.password import get_password_service
        
        service1 = get_password_service()
        service2 = get_password_service()
        
        assert service1 is service2

    def test_hash_password_function(self):
        """Test module-level hash_password function."""
        from core.engine.security.password import hash_password, verify_password
        
        password = "test_password"
        hashed = hash_password(password)
        
        assert verify_password(password, hashed)

    def test_verify_password_function(self):
        """Test module-level verify_password function."""
        from core.engine.security.password import hash_password, verify_password
        
        password = "test_password"
        hashed = hash_password(password)
        
        assert verify_password(password, hashed) is True
        assert verify_password("wrong", hashed) is False

    def test_generate_token_function(self):
        """Test module-level generate_token function."""
        from core.engine.security.password import generate_token
        
        token = generate_token(32)
        
        assert len(token) == 32
        assert isinstance(token, str)


class TestPasswordEdgeCases:
    """Test edge cases and special scenarios."""

    def test_unicode_password(self):
        """Test password with unicode characters."""
        from core.engine.security.password import PasswordService
        
        service = PasswordService()
        password = "密码🔐パスワード"
        hashed = service.hash_password(password)
        
        assert service.verify_password(password, hashed)
        assert not service.verify_password("wrong", hashed)

    def test_special_characters_password(self):
        """Test password with special characters."""
        from core.engine.security.password import PasswordService
        
        service = PasswordService()
        password = "p@$$w0rd!#$%^&*()_+-=[]{}|;':\",./<>?"
        hashed = service.hash_password(password)
        
        assert service.verify_password(password, hashed)

    def test_whitespace_password(self):
        """Test password with whitespace."""
        from core.engine.security.password import PasswordService
        
        service = PasswordService()
        password = "  password with spaces  "
        hashed = service.hash_password(password)
        
        # Should match exactly including whitespace
        assert service.verify_password(password, hashed)
        assert not service.verify_password("password with spaces", hashed)
