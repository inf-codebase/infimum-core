"""
Password hashing and token generation utilities.

This module provides secure password hashing using bcrypt and 
secure random token generation for password resets, email verification, etc.
"""

import secrets
import string
from loguru import logger

# Lazy import bcrypt to avoid loading it when not needed
_bcrypt = None


def _get_bcrypt():
    """Lazy load bcrypt module."""
    global _bcrypt
    if _bcrypt is None:
        import bcrypt
        _bcrypt = bcrypt
    return _bcrypt


class PasswordService:
    """Service for password hashing and verification using bcrypt.
    
    Bcrypt has a maximum password length of 72 bytes. Passwords longer than
    72 bytes are truncated to prevent hashing errors.
    
    Example:
        service = PasswordService()
        hashed = service.hash_password("my_password")
        is_valid = service.verify_password("my_password", hashed)
    """
    
    # Maximum password length for bcrypt
    MAX_PASSWORD_BYTES = 72
    
    # Default token character set (URL-safe)
    TOKEN_ALPHABET = string.ascii_letters + string.digits + '-_'
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt.
        
        Args:
            password: Plain text password to hash
            
        Returns:
            str: Hashed password (bcrypt hash string)
            
        Note:
            Passwords longer than 72 bytes are truncated (bcrypt limitation).
        """
        bcrypt = _get_bcrypt()
        
        # Encode password to bytes
        password_bytes = password.encode('utf-8')
        
        # Truncate to 72 bytes if longer (bcrypt limitation)
        if len(password_bytes) > self.MAX_PASSWORD_BYTES:
            logger.warning(
                f"Password exceeds {self.MAX_PASSWORD_BYTES} bytes "
                f"({len(password_bytes)} bytes), truncating for bcrypt"
            )
            password_bytes = password_bytes[:self.MAX_PASSWORD_BYTES]
        
        # Generate salt and hash password
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        
        # Return as string for storage
        return hashed.decode('utf-8')
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash using bcrypt.
        
        Args:
            plain_password: Plain text password to verify
            hashed_password: Hashed password (bcrypt hash string)
            
        Returns:
            bool: True if password matches, False otherwise
        """
        bcrypt = _get_bcrypt()
        
        try:
            # Encode password to bytes
            password_bytes = plain_password.encode('utf-8')
            
            # Truncate to 72 bytes if longer (same as hash_password)
            if len(password_bytes) > self.MAX_PASSWORD_BYTES:
                password_bytes = password_bytes[:self.MAX_PASSWORD_BYTES]
            
            # Encode hashed password to bytes
            hashed_bytes = hashed_password.encode('utf-8')
            
            # Verify password using constant-time comparison
            return bcrypt.checkpw(password_bytes, hashed_bytes)
            
        except Exception as e:
            logger.error(f"Error verifying password: {str(e)}")
            return False
    
    def generate_token(self, length: int = 32) -> str:
        """
        Generate a secure random token.
        
        Suitable for password reset tokens, email verification tokens, etc.
        
        Args:
            length: Length of the token (default: 32)
            
        Returns:
            str: Secure random token using URL-safe characters
        """
        return ''.join(secrets.choice(self.TOKEN_ALPHABET) for _ in range(length))
    
    def generate_hex_token(self, nbytes: int = 32) -> str:
        """
        Generate a secure random hex token.
        
        Args:
            nbytes: Number of random bytes (token will be 2x this length)
            
        Returns:
            str: Secure random hex string
        """
        return secrets.token_hex(nbytes)
    
    def generate_urlsafe_token(self, nbytes: int = 32) -> str:
        """
        Generate a secure random URL-safe token.
        
        Args:
            nbytes: Number of random bytes
            
        Returns:
            str: Secure random URL-safe base64 string
        """
        return secrets.token_urlsafe(nbytes)


# Module-level convenience functions
_password_service = None


def get_password_service() -> PasswordService:
    """Get singleton PasswordService instance."""
    global _password_service
    if _password_service is None:
        _password_service = PasswordService()
    return _password_service


def hash_password(password: str) -> str:
    """Hash a password using the default PasswordService."""
    return get_password_service().hash_password(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password using the default PasswordService."""
    return get_password_service().verify_password(plain_password, hashed_password)


def generate_token(length: int = 32) -> str:
    """Generate a secure random token using the default PasswordService."""
    return get_password_service().generate_token(length)
