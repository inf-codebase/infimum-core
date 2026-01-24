"""
Type definitions for the security module.

This module provides TypedDict definitions for user data, token payloads,
and authentication results, enabling type hints without coupling to specific ORM models.
"""

from typing import TypedDict, Optional, List, Any
from datetime import datetime


class UserData(TypedDict, total=False):
    """TypedDict defining expected user fields for authentication.
    
    All fields are optional (total=False) to allow partial user data.
    Services will work with whatever fields are provided.
    
    Example:
        user_data: UserData = {
            "id": 1,
            "username": "john",
            "email": "john@example.com",
            "role": "user"
        }
    """
    id: Any  # Can be int, str, UUID, etc.
    username: str
    email: str
    role: str
    roles: List[str]  # For users with multiple roles
    first_name: str
    last_name: str
    disabled: bool
    email_verified: bool


class TokenPayload(TypedDict, total=False):
    """TypedDict for JWT token payload structure.
    
    Represents the claims stored in a JWT token.
    """
    sub: str  # Subject (user ID as string)
    username: str
    email: str
    role: str
    roles: List[str]
    exp: datetime  # Expiration time
    iat: datetime  # Issued at time
    jti: str  # JWT ID (unique token identifier)
    type: str  # Token type: "access" or "refresh"


class TokenResponse(TypedDict):
    """TypedDict for token creation response."""
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int  # Seconds until access token expires


class AuthResult(TypedDict, total=False):
    """TypedDict for authentication operation results.
    
    Used to return standardized results from auth operations.
    """
    success: bool
    user_id: Any
    username: str
    email: str
    role: str
    roles: List[str]
    token_type: str
    error: Optional[str]
    message: Optional[str]


class EmailData(TypedDict, total=False):
    """TypedDict for email-related data."""
    to_email: str
    subject: str
    user_name: str
    token: str  # Verification or reset token
    link: str  # Full URL link


class BlacklistStats(TypedDict):
    """TypedDict for token blacklist statistics."""
    total_blacklisted_tokens: int
    active_blacklisted_tokens: int
    expired_blacklisted_tokens: int
