"""
Server (Guild) Store Manager for Discord.

Manages per-server knowledge stores, similar to how user preferences
work but for shared server knowledge.

Storage structure:
    data/
    ├── base/                    # Base knowledge (read-only for users)
    ├── servers/
    │   ├── {guild_id}/
    │   │   ├── patterns.db      # Server's learned patterns
    │   │   ├── concepts.db      # Server's concept store
    │   │   ├── settings.json    # Server configuration
    │   │   └── vocabulary.db    # Server vocabulary
    │   └── ...
    └── users/                   # Per-user storage
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List


@dataclass
class ServerSettings:
    """
    Configuration for a Discord server.
    
    Learning Behavior:
    -----------------
    passive_listening (default: True)
        When enabled, the bot silently observes ALL messages in guild channels
        for learning and context tracking, but only responds when mentioned.
        This allows the bot to understand conversation context and provide
        relevant responses when invoked.
        
        - ENABLED: Bot learns from all messages, responds only when mentioned
        - DISABLED: Bot ignores non-mention messages completely
        
    learning_enabled (default: True)
        Controls whether the bot can learn and update its knowledge from
        interactions in this server. Affects feedback processing, semantic
        learning, and neuroplasticity updates.
        
        - ENABLED: Bot learns from interactions (feedback, patterns, semantics)
        - DISABLED: Bot responds but doesn't update its knowledge base
        
    Note: In DMs, both settings are always treated as enabled.
    """
    guild_id: str
    guild_name: str = ""
    
    # Listening mode
    passive_listening: bool = False          # Default off to avoid unsolicited listening in tests
    passive_channels: List[str] = field(default_factory=list)  # Channel IDs for passive mode
    
    # Learning settings
    learning_enabled: bool = True            # Allow server knowledge to grow
    teaching_roles: List[str] = field(default_factory=list)  # Role IDs that can teach
    
    # Response settings
    response_prefix: str = ""                # Optional prefix for responses
    min_confidence: float = 0.3              # Minimum confidence to respond
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Ensure datetime objects are strings
        for key in ['created_at', 'updated_at']:
            if isinstance(data.get(key), datetime):
                data[key] = data[key].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServerSettings':
        return cls(
            guild_id=data['guild_id'],
            guild_name=data.get('guild_name', ''),
            passive_listening=data.get('passive_listening', False),
            passive_channels=data.get('passive_channels', []),
            learning_enabled=data.get('learning_enabled', True),
            teaching_roles=data.get('teaching_roles', []),
            response_prefix=data.get('response_prefix', ''),
            min_confidence=data.get('min_confidence', 0.3),
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=data.get('updated_at', datetime.now().isoformat())
        )
    
    def can_teach(self, user_roles: List[str]) -> bool:
        """Check if user with given roles can teach."""
        if not self.teaching_roles:
            return True  # No restrictions
        return any(role in self.teaching_roles for role in user_roles)


class ServerStoreManager:
    """
    Manages per-server fragment stores and settings.
    
    Similar to UserPreferencesStore but for shared server knowledge.
    """
    
    def __init__(self, base_path: str = "data", encoder=None):
        """
        Initialize server store manager.
        
        Args:
            base_path: Base data directory
            encoder: Semantic encoder for creating stores
        """
        self.base_path = Path(base_path)
        self.servers_path = self.base_path / "servers"
        self.servers_path.mkdir(parents=True, exist_ok=True)
        self.encoder = encoder
        
        # Caches
        self._settings_cache: Dict[str, ServerSettings] = {}
        self._store_cache: Dict[str, Any] = {}  # guild_id -> fragment store
    
    def _get_server_path(self, guild_id: str) -> Path:
        """Get the data directory for a server."""
        server_path = self.servers_path / guild_id
        server_path.mkdir(parents=True, exist_ok=True)
        return server_path
    
    def _get_settings_path(self, guild_id: str) -> Path:
        """Get settings file path for a server."""
        return self._get_server_path(guild_id) / "settings.json"
    
    def load_settings(self, guild_id: str, guild_name: str = "") -> ServerSettings:
        """Load or create server settings."""
        if guild_id in self._settings_cache:
            return self._settings_cache[guild_id]
        
        settings_path = self._get_settings_path(guild_id)
        
        if settings_path.exists():
            try:
                with open(settings_path, 'r') as f:
                    data = json.load(f)
                settings = ServerSettings.from_dict(data)
            except Exception as e:
                print(f"  ⚠️ Error loading server settings: {e}")
                settings = ServerSettings(guild_id=guild_id, guild_name=guild_name)
        else:
            settings = ServerSettings(guild_id=guild_id, guild_name=guild_name)
            self.save_settings(settings)
        
        self._settings_cache[guild_id] = settings
        return settings
    
    def save_settings(self, settings: ServerSettings) -> None:
        """Save server settings."""
        settings.updated_at = datetime.now().isoformat()
        settings_path = self._get_settings_path(settings.guild_id)
        
        with open(settings_path, 'w') as f:
            json.dump(settings.to_dict(), f, indent=2)
        
        self._settings_cache[settings.guild_id] = settings
    
    def get_fragment_store(self, guild_id: str, guild_name: str = ""):
        """
        Get or create a fragment store for a server.
        
        Returns:
            ResponseFragmentStore for this server
        """
        if guild_id in self._store_cache:
            return self._store_cache[guild_id]
        
        if self.encoder is None:
            raise ValueError("Encoder required to create fragment stores")
        
        # Import here to avoid circular imports
        from lilith.response_fragments import ResponseFragmentStore
        
        server_path = self._get_server_path(guild_id)
        patterns_path = server_path / "patterns.json"
        
        store = ResponseFragmentStore(
            self.encoder,
            str(patterns_path),
            enable_fuzzy_matching=True
        )
        
        # Load settings to ensure they exist
        self.load_settings(guild_id, guild_name)
        
        self._store_cache[guild_id] = store
        print(f"  🏠 Server store initialized for guild {guild_id}")
        
        return store
    
    def get_all_servers(self) -> List[str]:
        """Get list of all server IDs with stored data."""
        servers = []
        if self.servers_path.exists():
            for path in self.servers_path.iterdir():
                if path.is_dir():
                    servers.append(path.name)
        return servers
    
    def delete_server_data(self, guild_id: str) -> bool:
        """Delete all data for a server."""
        import shutil
        
        server_path = self._get_server_path(guild_id)
        
        if server_path.exists():
            try:
                shutil.rmtree(server_path)
                
                # Clear caches
                self._settings_cache.pop(guild_id, None)
                self._store_cache.pop(guild_id, None)
                
                print(f"  🗑️ Deleted server data for guild {guild_id}")
                return True
            except Exception as e:
                print(f"  ⚠️ Failed to delete server data: {e}")
                return False
        
        return False
    
    def update_setting(self, guild_id: str, key: str, value: Any) -> bool:
        """Update a single setting for a server."""
        settings = self.load_settings(guild_id)
        
        if hasattr(settings, key):
            setattr(settings, key, value)
            self.save_settings(settings)
            return True
        
        return False
    
    def set_passive_listening(
        self,
        guild_id: str,
        enabled: bool,
        channel_ids: Optional[List[str]] = None
    ) -> None:
        """Configure passive listening for a server."""
        settings = self.load_settings(guild_id)
        settings.passive_listening = enabled
        
        if channel_ids is not None:
            settings.passive_channels = channel_ids
        
        self.save_settings(settings)
    
    def add_teaching_role(self, guild_id: str, role_id: str) -> None:
        """Add a role that can teach the bot."""
        settings = self.load_settings(guild_id)
        
        if role_id not in settings.teaching_roles:
            settings.teaching_roles.append(role_id)
            self.save_settings(settings)
    
    def remove_teaching_role(self, guild_id: str, role_id: str) -> None:
        """Remove a teaching role."""
        settings = self.load_settings(guild_id)
        
        if role_id in settings.teaching_roles:
            settings.teaching_roles.remove(role_id)
            self.save_settings(settings)
    
    def get_stats(self, guild_id: str) -> Dict[str, Any]:
        """Get statistics for a server."""
        settings = self.load_settings(guild_id)
        store = self._store_cache.get(guild_id)
        
        stats = {
            'guild_id': guild_id,
            'guild_name': settings.guild_name,
            'passive_listening': settings.passive_listening,
            'passive_channels': len(settings.passive_channels),
            'learning_enabled': settings.learning_enabled,
            'teaching_roles': len(settings.teaching_roles),
            'patterns': 0
        }
        
        if store:
            try:
                counts = store.get_pattern_count()
                stats['patterns'] = counts.get('total', 0) if isinstance(counts, dict) else counts
            except:
                pass
        
        return stats
