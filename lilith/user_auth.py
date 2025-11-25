"""
User authentication and identity management for multi-tenant Lilith.

Supports multiple auth modes:
- 'none': Teacher mode (no user, writes to base)
- 'simple': CLI mode (prompt for username, no validation)
- 'trusted': Chat client mode (username from external auth, trusted)
"""

from typing import Optional
from dataclasses import dataclass
from enum import Enum


class AuthMode(Enum):
    """Authentication modes for different deployment scenarios"""
    NONE = "none"          # Teacher mode - no user isolation
    SIMPLE = "simple"      # CLI - prompt for username, no password
    TRUSTED = "trusted"    # Chat client - trust external auth


@dataclass
class UserIdentity:
    """User identity information"""
    user_id: str
    auth_mode: AuthMode
    display_name: Optional[str] = None
    
    def is_teacher(self) -> bool:
        """Check if this is teacher mode (no user isolation)"""
        return self.auth_mode == AuthMode.NONE or self.user_id == "teacher"


class UserAuthenticator:
    """Manages user authentication for different deployment modes"""
    
    def __init__(self, auth_mode: AuthMode = AuthMode.NONE):
        """
        Initialize authenticator.
        
        Args:
            auth_mode: Authentication mode to use
        """
        self.auth_mode = auth_mode
    
    def authenticate(self, username: Optional[str] = None) -> UserIdentity:
        """
        Authenticate and return user identity.
        
        Args:
            username: Username (required for TRUSTED mode, optional for SIMPLE)
        
        Returns:
            UserIdentity with user_id and auth mode
        """
        if self.auth_mode == AuthMode.NONE:
            return UserIdentity(
                user_id="teacher",
                auth_mode=AuthMode.NONE,
                display_name="Teacher"
            )
        
        elif self.auth_mode == AuthMode.SIMPLE:
            # Prompt for username if not provided
            if not username:
                username = self._prompt_username()
            
            # Sanitize username for filesystem safety
            safe_username = self._sanitize_username(username)
            
            return UserIdentity(
                user_id=safe_username,
                auth_mode=AuthMode.SIMPLE,
                display_name=username
            )
        
        elif self.auth_mode == AuthMode.TRUSTED:
            # Must have username from external system
            if not username:
                raise ValueError("TRUSTED mode requires username from external auth")
            
            safe_username = self._sanitize_username(username)
            
            return UserIdentity(
                user_id=safe_username,
                auth_mode=AuthMode.TRUSTED,
                display_name=username
            )
        
        else:
            raise ValueError(f"Unknown auth mode: {self.auth_mode}")
    
    def _prompt_username(self) -> str:
        """Prompt user for username (CLI mode)"""
        while True:
            username = input("Enter your username: ").strip()
            if username:
                return username
            print("Username cannot be empty. Please try again.")
    
    def _sanitize_username(self, username: str) -> str:
        """
        Sanitize username for safe filesystem usage.
        
        Removes or replaces characters that could cause issues:
        - Path separators (/, \\)
        - Special characters that could break file systems
        - Converts to lowercase for consistency
        """
        # Remove dangerous characters
        safe = username.lower()
        safe = safe.replace("/", "_")
        safe = safe.replace("\\", "_")
        safe = safe.replace("..", "_")
        safe = "".join(c for c in safe if c.isalnum() or c in "-_")
        
        # Ensure not empty after sanitization
        if not safe:
            safe = "user"
        
        # Prevent reserved names
        if safe in ["teacher", "base", "admin", "root"]:
            safe = f"user_{safe}"
        
        return safe


def get_user_data_path(user_identity: UserIdentity, base_path: str = "data") -> str:
    """
    Get the data directory path for a user.
    
    Args:
        user_identity: User identity
        base_path: Base data directory
    
    Returns:
        Path to user's data directory (base for teacher, users/{id} for users)
    """
    if user_identity.is_teacher():
        return f"{base_path}/base"
    else:
        return f"{base_path}/users/{user_identity.user_id}"
