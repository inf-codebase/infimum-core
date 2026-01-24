"""
Token blacklist service for JWT token invalidation.

This module provides in-memory storage for blacklisted JWT tokens with
automatic cleanup of expired tokens via a background thread.
"""

from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
import threading
import time
from loguru import logger

from .types import BlacklistStats

# Lazy import jose to avoid loading when not needed
_jwt = None


def _get_jwt():
    """Lazy load jose.jwt module."""
    global _jwt
    if _jwt is None:
        from jose import jwt
        _jwt = jwt
    return _jwt


class TokenBlacklistService:
    """Service for managing blacklisted JWT tokens.
    
    Tokens are stored in-memory with their expiration times. A background
    thread periodically cleans up expired tokens to prevent memory growth.
    
    Thread-safe using RLock for concurrent access.
    
    Example:
        service = TokenBlacklistService()
        service.blacklist_token(token)
        is_blocked = service.is_blacklisted(token)
    """
    
    # Default cleanup interval (30 minutes)
    DEFAULT_CLEANUP_INTERVAL = 30 * 60
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: Optional[str] = None,
        cleanup_interval: int = DEFAULT_CLEANUP_INTERVAL,
        start_cleanup_thread: bool = True
    ):
        """
        Initialize the token blacklist service.
        
        Args:
            secret_key: JWT secret key (defaults to auto_config.JWT_SECRET_KEY)
            algorithm: JWT algorithm (defaults to auto_config.JWT_ALGORITHM or 'HS256')
            cleanup_interval: Seconds between cleanup runs (default: 30 minutes)
            start_cleanup_thread: Whether to start background cleanup thread
        """
        # In-memory storage: {jti: expiration_timestamp}
        self._blacklisted_tokens: Dict[str, float] = {}
        self._lock = threading.RLock()
        self.cleanup_interval = cleanup_interval
        
        # Store config (will be resolved lazily)
        self._secret_key = secret_key
        self._algorithm = algorithm
        
        # Start background cleanup thread
        if start_cleanup_thread:
            self._start_cleanup_thread()
        
        logger.info("Token Blacklist Service initialized")
    
    @property
    def secret_key(self) -> str:
        """Get JWT secret key from config or stored value."""
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
    
    @property
    def algorithm(self) -> str:
        """Get JWT algorithm from config or stored value."""
        if self._algorithm is not None:
            return self._algorithm
        try:
            from core.utils import auto_config
            alg = getattr(auto_config, 'JWT_ALGORITHM', None)
            return alg if alg else 'HS256'
        except ImportError:
            return 'HS256'
    
    def _start_cleanup_thread(self) -> None:
        """Start background thread for cleaning up expired tokens."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.cleanup_interval)
                    self.cleanup_expired_tokens()
                except Exception as e:
                    logger.error(f"Error in token cleanup thread: {str(e)}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        logger.debug("Token cleanup thread started")
    
    def blacklist_token(self, token: str) -> bool:
        """
        Add a token to the blacklist.
        
        Args:
            token: JWT token to blacklist
            
        Returns:
            bool: True if token was blacklisted, False if invalid or already blacklisted
        """
        jwt = _get_jwt()
        
        try:
            # Decode token to get expiration and JTI
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            exp = payload.get("exp")
            jti = payload.get("jti")
            
            if not exp or not jti:
                logger.warning("Token missing required claims (exp, jti)")
                return False
            
            with self._lock:
                if jti in self._blacklisted_tokens:
                    logger.debug(f"Token already blacklisted: {jti}")
                    return True
                
                self._blacklisted_tokens[jti] = float(exp)
                logger.info(f"Token blacklisted: {jti}")
                return True
                
        except jwt.JWTError as e:
            logger.warning(f"Invalid token for blacklisting: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error blacklisting token: {str(e)}")
            return False
    
    def is_blacklisted(self, token: str) -> bool:
        """
        Check if a token is blacklisted.
        
        Args:
            token: JWT token to check
            
        Returns:
            bool: True if token is blacklisted, False otherwise
        """
        jwt = _get_jwt()
        
        try:
            # Decode token to get JTI
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            jti = payload.get("jti")
            
            if not jti:
                logger.warning("Token missing JTI claim")
                return True  # Consider tokens without JTI as blacklisted
            
            with self._lock:
                return jti in self._blacklisted_tokens
                
        except jwt.JWTError as e:
            # Don't block on decode error - let JWT verification handle it
            logger.debug(f"Token decode error in blacklist check: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error checking token blacklist: {str(e)}")
            return True  # Consider errors as blacklisted for safety
    
    def is_jti_blacklisted(self, jti: str) -> bool:
        """
        Check if a JTI (JWT ID) is blacklisted.
        
        Args:
            jti: JWT ID to check
            
        Returns:
            bool: True if JTI is blacklisted, False otherwise
        """
        with self._lock:
            return jti in self._blacklisted_tokens
    
    def cleanup_expired_tokens(self) -> int:
        """
        Remove expired tokens from the blacklist.
        
        Returns:
            int: Number of tokens cleaned up
        """
        try:
            current_time = datetime.now(timezone.utc).timestamp()
            
            with self._lock:
                # Find expired tokens
                expired = [
                    jti for jti, exp in self._blacklisted_tokens.items()
                    if exp < current_time
                ]
                
                # Remove expired tokens
                for jti in expired:
                    del self._blacklisted_tokens[jti]
                
                if expired:
                    logger.info(f"Cleaned up {len(expired)} expired tokens")
                
                return len(expired)
                
        except Exception as e:
            logger.error(f"Error cleaning up expired tokens: {str(e)}")
            return 0
    
    def get_stats(self) -> BlacklistStats:
        """
        Get statistics about the token blacklist.
        
        Returns:
            BlacklistStats: Dictionary with blacklist statistics
        """
        try:
            with self._lock:
                current_time = datetime.now(timezone.utc).timestamp()
                active = sum(1 for exp in self._blacklisted_tokens.values() if exp >= current_time)
                expired = len(self._blacklisted_tokens) - active
                
                return BlacklistStats(
                    total_blacklisted_tokens=len(self._blacklisted_tokens),
                    active_blacklisted_tokens=active,
                    expired_blacklisted_tokens=expired
                )
                
        except Exception as e:
            logger.error(f"Error getting blacklist stats: {str(e)}")
            return BlacklistStats(
                total_blacklisted_tokens=0,
                active_blacklisted_tokens=0,
                expired_blacklisted_tokens=0
            )
    
    def force_cleanup(self) -> int:
        """
        Force cleanup of expired tokens immediately.
        
        Returns:
            int: Number of tokens cleaned up
        """
        return self.cleanup_expired_tokens()
    
    def clear(self) -> None:
        """Clear all blacklisted tokens (useful for testing)."""
        with self._lock:
            count = len(self._blacklisted_tokens)
            self._blacklisted_tokens.clear()
            logger.info(f"Cleared {count} blacklisted tokens")
    
    def _extract_jti_and_exp(self, token: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Extract JTI and expiration from token.
        
        Args:
            token: JWT token string
            
        Returns:
            Tuple of (jti, exp) or (None, None) if invalid
        """
        jwt = _get_jwt()
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            jti = payload.get("jti")
            exp = payload.get("exp")
            return jti, float(exp) if exp is not None else None
        except jwt.JWTError as e:
            logger.debug(f"Invalid token for extraction: {str(e)}")
            return None, None
        except Exception as e:
            logger.error(f"Error extracting token info: {str(e)}")
            return None, None


# Singleton instance management
_blacklist_service: Optional[TokenBlacklistService] = None
_service_lock = threading.Lock()


def get_token_blacklist_service() -> TokenBlacklistService:
    """Get the singleton instance of TokenBlacklistService."""
    global _blacklist_service
    
    if _blacklist_service is None:
        with _service_lock:
            if _blacklist_service is None:
                _blacklist_service = TokenBlacklistService()
    
    return _blacklist_service


def reset_blacklist_service() -> None:
    """Reset the singleton instance (useful for testing)."""
    global _blacklist_service
    with _service_lock:
        if _blacklist_service is not None:
            _blacklist_service.clear()
        _blacklist_service = None
