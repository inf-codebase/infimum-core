"""
Security module for authentication and authorization.

This module provides reusable security services for:
- JWT token creation and verification
- Password hashing with bcrypt
- Token blacklisting for logout
- Email sending for auth flows

All services work with generic dictionaries, not ORM-specific models.

Example:
    from core.engine.security import (
        get_jwt_service, get_password_service, get_email_service,
        hash_password, verify_password, generate_token
    )
    
    # Get singleton services
    jwt = get_jwt_service()
    pwd = get_password_service()
    email = get_email_service()
    
    # Hash password
    hashed = hash_password("my_password")
    
    # Create tokens from any dict
    user_data = {"id": 1, "username": "john", "email": "john@example.com", "role": "user"}
    access_token = jwt.create_access_token(user_data)
    refresh_token = jwt.create_refresh_token(user_data)
    
    # Or create both at once
    tokens = jwt.create_token_pair(user_data)
    
    # Verify and extract user
    result = jwt.get_user_from_token(access_token)
    
    # Blacklist on logout
    jwt.blacklist_token(access_token)
    
    # Send verification email
    await email.send_verification_email(
        to_email="user@example.com",
        verification_token=generate_token(),
        user_name="John"
    )
"""

# Type definitions
from .types import (
    UserData,
    TokenPayload,
    TokenResponse,
    AuthResult,
    EmailData,
    BlacklistStats,
)

# Password service
from .password import (
    PasswordService,
    get_password_service,
    hash_password,
    verify_password,
    generate_token,
)

# Token blacklist service
from .token_blacklist import (
    TokenBlacklistService,
    get_token_blacklist_service,
    reset_blacklist_service,
)

# JWT service
from .jwt_service import (
    JWTService,
    get_jwt_service,
    reset_jwt_service,
)

# Email service
from .email_service import (
    EmailService,
    get_email_service,
    reset_email_service,
)

__all__ = [
    # Types
    "UserData",
    "TokenPayload",
    "TokenResponse",
    "AuthResult",
    "EmailData",
    "BlacklistStats",
    # Password service
    "PasswordService",
    "get_password_service",
    "hash_password",
    "verify_password",
    "generate_token",
    # Token blacklist service
    "TokenBlacklistService",
    "get_token_blacklist_service",
    "reset_blacklist_service",
    # JWT service
    "JWTService",
    "get_jwt_service",
    "reset_jwt_service",
    # Email service
    "EmailService",
    "get_email_service",
    "reset_email_service",
]
