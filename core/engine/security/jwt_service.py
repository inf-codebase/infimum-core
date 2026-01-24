"""
JWT token service for authentication.

This module provides JWT token creation, verification, and management
with automatic blacklist integration for token invalidation.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Union
from loguru import logger

from .types import UserData, TokenPayload, TokenResponse, AuthResult
from .password import PasswordService, get_password_service
from .token_blacklist import TokenBlacklistService, get_token_blacklist_service

# Lazy import jose to avoid loading when not needed
_jwt = None
_JWTError = None


def _get_jwt():
    """Lazy load jose.jwt module."""
    global _jwt, _JWTError
    if _jwt is None:
        from jose import jwt, JWTError
        _jwt = jwt
        _JWTError = JWTError
    return _jwt


def _get_jwt_error():
    """Get JWTError class."""
    if _JWTError is None:
        _get_jwt()
    return _JWTError


class JWTService:
    """Service for handling JWT token operations.
    
    Creates access and refresh tokens from user data dictionaries,
    verifies tokens, and integrates with the blacklist service.
    
    Example:
        service = JWTService()
        
        user_data = {"id": 1, "username": "john", "email": "john@example.com"}
        access_token = service.create_access_token(user_data)
        refresh_token = service.create_refresh_token(user_data)
        
        payload = service.get_user_from_token(access_token)
        service.blacklist_token(access_token)  # On logout
    """
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: Optional[str] = None,
        access_token_expire_minutes: Optional[int] = None,
        refresh_token_expire_days: Optional[int] = None,
        blacklist_service: Optional[TokenBlacklistService] = None,
        password_service: Optional[PasswordService] = None
    ):
        """
        Initialize the JWT service.
        
        Args:
            secret_key: JWT signing secret (defaults to auto_config.JWT_SECRET_KEY)
            algorithm: JWT algorithm (defaults to auto_config.JWT_ALGORITHM or 'HS256')
            access_token_expire_minutes: Access token TTL (defaults to 30)
            refresh_token_expire_days: Refresh token TTL (defaults to 7)
            blacklist_service: Optional custom blacklist service
            password_service: Optional custom password service
        """
        self._secret_key = secret_key
        self._algorithm = algorithm
        self._access_token_expire_minutes = access_token_expire_minutes
        self._refresh_token_expire_days = refresh_token_expire_days
        
        # Lazy-loaded services
        self._blacklist_service = blacklist_service
        self._password_service = password_service
        
        logger.info(f"JWT Service initialized with {self.access_token_expire_minutes}min access token expiry")
    
    @property
    def secret_key(self) -> str:
        """Get JWT secret key."""
        if self._secret_key is not None:
            return self._secret_key
        try:
            from core.utils import auto_config
            key = getattr(auto_config, 'JWT_SECRET_KEY', None)
            if key:
                return key
        except ImportError:
            pass
        return 'your-secret-key-change-in-production'
    
    @secret_key.setter
    def secret_key(self, value: str) -> None:
        """Set JWT secret key."""
        self._secret_key = value
    
    @property
    def algorithm(self) -> str:
        """Get JWT algorithm."""
        if self._algorithm is not None:
            return self._algorithm
        try:
            from core.utils import auto_config
            alg = getattr(auto_config, 'JWT_ALGORITHM', None)
            return alg if alg else 'HS256'
        except ImportError:
            return 'HS256'
    
    @property
    def access_token_expire_minutes(self) -> int:
        """Get access token expiration in minutes."""
        if self._access_token_expire_minutes is not None:
            return self._access_token_expire_minutes
        try:
            from core.utils import auto_config
            return int(getattr(auto_config, 'JWT_ACCESS_TOKEN_EXPIRE_MINUTES', 30))
        except (ImportError, ValueError, TypeError):
            return 30
    
    @property
    def refresh_token_expire_days(self) -> int:
        """Get refresh token expiration in days."""
        if self._refresh_token_expire_days is not None:
            return self._refresh_token_expire_days
        try:
            from core.utils import auto_config
            return int(getattr(auto_config, 'JWT_REFRESH_TOKEN_EXPIRE_DAYS', 7))
        except (ImportError, ValueError, TypeError):
            return 7
    
    @property
    def blacklist_service(self) -> TokenBlacklistService:
        """Get token blacklist service (lazy loading)."""
        if self._blacklist_service is None:
            self._blacklist_service = get_token_blacklist_service()
        return self._blacklist_service
    
    @property
    def password_service(self) -> PasswordService:
        """Get password service (lazy loading)."""
        if self._password_service is None:
            self._password_service = get_password_service()
        return self._password_service
    
    def create_access_token(
        self,
        user_data: Union[Dict[str, Any], UserData],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create an access token for the user.
        
        Args:
            user_data: Dictionary containing user information (id, username, email, role)
            expires_delta: Optional custom expiration time
            
        Returns:
            str: Encoded JWT access token
        """
        jwt = _get_jwt()
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
        
        # Get current time for iat claim
        iat_time = datetime.now(timezone.utc)
        
        # Extract user fields with defaults
        user_id = str(user_data.get('id', ''))
        username = user_data.get('username', '')
        email = user_data.get('email', '')
        role = user_data.get('role', '')
        roles = user_data.get('roles', [])
        
        # Build payload
        to_encode: Dict[str, Any] = {
            "sub": user_id,
            "username": username,
            "email": email,
            "exp": expire,
            "iat": iat_time,
            "jti": f"{user_id}-{int(iat_time.timestamp() * 1000000)}",
            "type": "access"
        }
        
        # Include role(s) if present
        if role:
            to_encode["role"] = role
        if roles:
            to_encode["roles"] = roles
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        logger.debug(f"Created access token for user {username} (expires: {expire})")
        return encoded_jwt
    
    def create_refresh_token(
        self,
        user_data: Union[Dict[str, Any], UserData],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a refresh token for the user.
        
        Args:
            user_data: Dictionary containing user information
            expires_delta: Optional custom expiration time
            
        Returns:
            str: Encoded JWT refresh token
        """
        jwt = _get_jwt()
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)
        
        # Get current time for iat claim
        iat_time = datetime.now(timezone.utc)
        
        # Extract user fields
        user_id = str(user_data.get('id', ''))
        username = user_data.get('username', '')
        
        to_encode = {
            "sub": user_id,
            "username": username,
            "exp": expire,
            "iat": iat_time,
            "jti": f"{user_id}-{int(iat_time.timestamp() * 1000000)}-refresh",
            "type": "refresh"
        }
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        logger.debug(f"Created refresh token for user {username} (expires: {expire})")
        return encoded_jwt
    
    def create_token_pair(
        self,
        user_data: Union[Dict[str, Any], UserData]
    ) -> TokenResponse:
        """
        Create both access and refresh tokens.
        
        Args:
            user_data: Dictionary containing user information
            
        Returns:
            TokenResponse: Dictionary with access_token, refresh_token, token_type, expires_in
        """
        access_token = self.create_access_token(user_data)
        refresh_token = self.create_refresh_token(user_data)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=self.access_token_expire_minutes * 60
        )
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Dict containing token payload or None if invalid
        """
        jwt = _get_jwt()
        JWTError = _get_jwt_error()
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            logger.debug(f"Token verified successfully for user: {payload.get('username')}")
            return payload
        except JWTError as e:
            logger.warning(f"Token verification failed: {str(e)}")
            return None
    
    def is_token_expired(self, payload: Dict[str, Any]) -> bool:
        """
        Check if token payload indicates expiration.
        
        Args:
            payload: Decoded token payload
            
        Returns:
            bool: True if token is expired
        """
        exp = payload.get("exp")
        if exp is None:
            return True
        return datetime.now(timezone.utc).timestamp() > exp
    
    def get_user_from_token(self, token: str) -> Optional[AuthResult]:
        """
        Extract user information from token.
        
        Checks blacklist and validates token before returning user data.
        
        Args:
            token: JWT token string
            
        Returns:
            AuthResult with user information or None if invalid/blacklisted
        """
        # Check if token is blacklisted first
        if self.blacklist_service.is_blacklisted(token):
            logger.warning("Token is blacklisted")
            return None
        
        payload = self.verify_token(token)
        if payload is None:
            return None
        
        if self.is_token_expired(payload):
            logger.warning("Token is expired")
            return None
        
        return AuthResult(
            success=True,
            user_id=payload.get("sub"),
            username=payload.get("username"),
            email=payload.get("email"),
            role=payload.get("role"),
            roles=payload.get("roles", []),
            token_type=payload.get("type")
        )
    
    def blacklist_token(self, token: str) -> bool:
        """
        Add a token to the blacklist.
        
        Args:
            token: JWT token string to blacklist
            
        Returns:
            bool: True if successfully blacklisted, False otherwise
        """
        return self.blacklist_service.blacklist_token(token)
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """
        Create a new access token from a valid refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New access token string or None if refresh token is invalid
        """
        # Verify refresh token
        payload = self.verify_token(refresh_token)
        if payload is None:
            return None
        
        # Check it's actually a refresh token
        if payload.get("type") != "refresh":
            logger.warning("Token is not a refresh token")
            return None
        
        # Check blacklist
        if self.blacklist_service.is_blacklisted(refresh_token):
            logger.warning("Refresh token is blacklisted")
            return None
        
        # Create new access token with same user data
        user_data = {
            "id": payload.get("sub"),
            "username": payload.get("username"),
            "email": payload.get("email", ""),
            "role": payload.get("role", ""),
            "roles": payload.get("roles", [])
        }
        
        return self.create_access_token(user_data)
    
    # Password utility methods (delegated to PasswordService)
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return self.password_service.hash_password(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self.password_service.verify_password(plain_password, hashed_password)


# Singleton instance management
_jwt_service: Optional[JWTService] = None
_service_lock = None


def _get_lock():
    """Get or create the service lock."""
    global _service_lock
    if _service_lock is None:
        import threading
        _service_lock = threading.Lock()
    return _service_lock


def get_jwt_service() -> JWTService:
    """Get the singleton instance of JWTService."""
    global _jwt_service
    
    if _jwt_service is None:
        lock = _get_lock()
        with lock:
            if _jwt_service is None:
                _jwt_service = JWTService()
    
    return _jwt_service


def reset_jwt_service() -> None:
    """Reset the singleton instance (useful for testing)."""
    global _jwt_service
    lock = _get_lock()
    with lock:
        _jwt_service = None
