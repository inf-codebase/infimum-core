from typing import Optional

from src.core.utils import auto_config
from fastapi import Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

from src.features.user_management.entities.user import User

# API Key configuration
API_KEY_NAME = "X-API-KEY"

# key: role mapping
API_KEYS = auto_config.API_KEYS

# API Key header scheme
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate the API key"""
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key is missing",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key


async def get_current_user(api_key: str = Depends(get_api_key)) -> User:
    """Get user information from API key"""
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # In a real application, you'd use the API key to look up user info from a database
    # For this example, we use a simple dictionary mapping keys to roles
    role_name = API_KEYS[api_key]

    # Create and return a User object
    # Note: This is a mock user for API key auth, not a real database user
    # The role property will return empty string since no roles are assigned
    user = User(username=f"api_user_{role_name}")
    # TODO: For proper role support, this should look up an actual user from the database
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Check if the user is active"""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# Optional: Role-based authentication
def requires_role(required_role: str):
    """Create a dependency that checks if the user has the required role"""

    async def role_checker(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        if not current_user.has_role(required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Role '{required_role}' required",
            )
        return current_user

    return role_checker
