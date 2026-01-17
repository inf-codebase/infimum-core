from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from features.user_management.entities.user import User
from features.user_management.services.jwt_service import JWTService
from features.user_management.services.auth_service import AuthService

# JWT Bearer security scheme
security = HTTPBearer()

# Initialize services
jwt_service = JWTService()
auth_service = AuthService()


def get_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Extract and validate Bearer token"""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token is missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


async def get_current_user(token: str = Depends(get_token)) -> User:
    """Get user information from JWT token"""
    # Get user info from token
    user_info = jwt_service.get_user_from_token(token)
    if user_info is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get full user from database
    user = await auth_service.get_user_by_id(int(user_info["user_id"]))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

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
