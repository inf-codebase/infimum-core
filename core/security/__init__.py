from .api_key_auth import (
    get_api_key,
    get_current_user as get_current_user_api_key,
    get_current_active_user as get_current_active_user_api_key,
    requires_role as requires_role_api_key,
)
from .jwt_auth import (
    get_token,
    get_current_user,
    get_current_active_user,
    requires_role,
)

__all__ = [
    # API Key Auth
    "get_api_key",
    "get_current_user_api_key",
    "get_current_active_user_api_key",
    "requires_role_api_key",
    # JWT Auth
    "get_token",
    "get_current_user",
    "get_current_active_user",
    "requires_role",
]

