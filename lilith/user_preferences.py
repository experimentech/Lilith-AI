"""
User Preferences Storage for Lilith AI.

Stores per-user preferences and personal information that Lilith learns
from conversation or that users set explicitly.

Examples of stored preferences:
- User's preferred name ("Call me John" → name: "John")
- Interests (learned from conversation patterns)
- Communication preferences

This enables Lilith to remember user-specific information across sessions.
"""

import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class UserPreferences:
    """User preference data."""
    user_id: str
    display_name: Optional[str] = None  # What the user wants to be called
    interests: List[str] = field(default_factory=list)
    custom_data: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_active: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, ensuring datetime objects become ISO strings."""
        data = asdict(self)
        # Convert any datetime objects to ISO strings
        for key in ['created_at', 'updated_at', 'last_active']:
            if isinstance(data.get(key), datetime):
                data[key] = data[key].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserPreferences':
        return cls(
            user_id=data['user_id'],
            display_name=data.get('display_name'),
            interests=data.get('interests', []),
            custom_data=data.get('custom_data', {}),
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=data.get('updated_at', datetime.now().isoformat()),
            last_active=data.get('last_active', data.get('updated_at', datetime.now().isoformat()))
        )
    
    def touch(self) -> None:
        """Update last_active timestamp."""
        self.last_active = datetime.now().isoformat()
    
    def days_inactive(self) -> float:
        """Get number of days since last activity."""
        try:
            # Handle both string and datetime objects
            if isinstance(self.last_active, datetime):
                last = self.last_active
            else:
                last = datetime.fromisoformat(self.last_active)
            return (datetime.now() - last).total_seconds() / 86400
        except (ValueError, TypeError):
            return 0.0


class UserPreferencesStore:
    """
    Persistent storage for user preferences.
    
    Each user has their own preferences file stored in their data directory.
    """
    
    def __init__(self, base_path: str = "data"):
        """
        Initialize preferences store.
        
        Args:
            base_path: Base data directory
        """
        self.base_path = Path(base_path)
        self._cache: Dict[str, UserPreferences] = {}
    
    def _get_prefs_path(self, user_id: str) -> Path:
        """Get the path to a user's preferences file."""
        if user_id == "teacher":
            return self.base_path / "base" / "preferences.json"
        return self.base_path / "users" / user_id / "preferences.json"
    
    def load(self, user_id: str) -> UserPreferences:
        """
        Load user preferences from disk.
        
        Args:
            user_id: User identifier
            
        Returns:
            UserPreferences object (empty if none exist)
        """
        if user_id in self._cache:
            return self._cache[user_id]
        
        path = self._get_prefs_path(user_id)
        
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                prefs = UserPreferences.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load preferences for {user_id}: {e}")
                prefs = UserPreferences(user_id=user_id)
        else:
            prefs = UserPreferences(user_id=user_id)
        
        self._cache[user_id] = prefs
        return prefs
    
    def save(self, prefs: UserPreferences) -> None:
        """
        Save user preferences to disk.
        
        Args:
            prefs: Preferences to save
        """
        prefs.updated_at = datetime.now().isoformat()
        
        path = self._get_prefs_path(prefs.user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(prefs.to_dict(), f, indent=2)
        
        self._cache[prefs.user_id] = prefs
    
    def get_display_name(self, user_id: str) -> Optional[str]:
        """Get user's preferred display name."""
        prefs = self.load(user_id)
        return prefs.display_name
    
    def set_display_name(self, user_id: str, name: str) -> None:
        """Set user's preferred display name."""
        prefs = self.load(user_id)
        prefs.display_name = name
        self.save(prefs)
    
    def add_interest(self, user_id: str, interest: str) -> None:
        """Add an interest for the user."""
        prefs = self.load(user_id)
        if interest not in prefs.interests:
            prefs.interests.append(interest)
            self.save(prefs)
    
    def set_custom(self, user_id: str, key: str, value: Any) -> None:
        """Set a custom preference value."""
        prefs = self.load(user_id)
        prefs.custom_data[key] = value
        self.save(prefs)
    
    def get_custom(self, user_id: str, key: str, default: Any = None) -> Any:
        """Get a custom preference value."""
        prefs = self.load(user_id)
        return prefs.custom_data.get(key, default)
    
    def touch_user(self, user_id: str) -> None:
        """Update user's last_active timestamp."""
        prefs = self.load(user_id)
        prefs.touch()
        self.save(prefs)
    
    def get_all_users(self) -> List[str]:
        """Get list of all user IDs with stored preferences."""
        users_dir = self.base_path / "users"
        if not users_dir.exists():
            return []
        
        user_ids = []
        for user_dir in users_dir.iterdir():
            if user_dir.is_dir():
                prefs_file = user_dir / "preferences.json"
                if prefs_file.exists():
                    user_ids.append(user_dir.name)
        return user_ids
    
    def get_inactive_users(self, days: float = 7.0) -> List[Tuple[str, float]]:
        """
        Get users inactive for more than specified days.
        
        Args:
            days: Inactivity threshold in days
            
        Returns:
            List of (user_id, days_inactive) tuples
        """
        inactive = []
        for user_id in self.get_all_users():
            prefs = self.load(user_id)
            inactive_days = prefs.days_inactive()
            if inactive_days >= days:
                inactive.append((user_id, inactive_days))
        
        # Sort by most inactive first
        inactive.sort(key=lambda x: x[1], reverse=True)
        return inactive
    
    def delete_user_data(self, user_id: str) -> bool:
        """
        Delete all data for a user.
        
        Args:
            user_id: User to delete
            
        Returns:
            True if deleted, False if not found
        """
        import shutil
        
        if user_id == "teacher":
            return False  # Never delete teacher/base data
        
        user_dir = self.base_path / "users" / user_id
        if not user_dir.exists():
            return False
        
        # Remove from cache
        self._cache.pop(user_id, None)
        
        # Delete directory
        shutil.rmtree(user_dir)
        print(f"🗑️  Deleted user data for: {user_id}")
        return True
    
    def cleanup_inactive_users(self, days: float = 7.0, dry_run: bool = True) -> List[str]:
        """
        Delete data for users inactive longer than specified days.
        
        Args:
            days: Inactivity threshold in days
            dry_run: If True, only report what would be deleted
            
        Returns:
            List of deleted (or would-be-deleted) user IDs
        """
        inactive = self.get_inactive_users(days)
        deleted = []
        
        for user_id, inactive_days in inactive:
            if dry_run:
                print(f"  Would delete: {user_id} (inactive {inactive_days:.1f} days)")
            else:
                if self.delete_user_data(user_id):
                    deleted.append(user_id)
        
        return [uid for uid, _ in inactive] if dry_run else deleted


class PreferenceExtractor:
    """
    Extracts user preferences from natural conversation.
    
    Detects patterns like:
    - "My name is John" / "Call me John" / "I'm John"
    - "I like Python" / "I'm interested in AI"
    - "I prefer dark mode"
    """
    
    # Patterns for name extraction
    NAME_PATTERNS = [
        r"\bmy name is ([A-Z][a-z]+)\b",
        r"\bcall me ([A-Z][a-z]+)\b",
        r"\bi'?m ([A-Z][a-z]+)\b(?:\s*[,.]|$)",  # "I'm John." but not "I'm happy"
        r"\bi am ([A-Z][a-z]+)\b(?:\s*[,.]|$)",
        r"\bname'?s ([A-Z][a-z]+)\b",
        r"\bthey call me ([A-Z][a-z]+)\b",
        r"\beveryone calls me ([A-Z][a-z]+)\b",
        r"\bgo by ([A-Z][a-z]+)\b",
    ]
    
    # Patterns for interest extraction  
    INTEREST_PATTERNS = [
        r"\bi (?:really )?(?:like|love|enjoy) ([a-zA-Z]+)\b",
        r"\bi'?m (?:really )?interested in ([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\b",
        r"\bi'?m (?:a|an) ([a-zA-Z]+) (?:enthusiast|fan)\b",
        r"\bmy (?:favorite|favourite) (?:thing|hobby) is ([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\b",
        r"\bi work (?:with|on|in) ([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\b",
    ]
    
    # Words that are NOT names (prevent "I'm happy" → name="Happy")
    NOT_NAMES = {
        'happy', 'sad', 'tired', 'excited', 'bored', 'confused', 'lost',
        'hungry', 'thirsty', 'ready', 'sorry', 'fine', 'good', 'great',
        'okay', 'sure', 'certain', 'positive', 'negative', 'curious',
        'interested', 'learning', 'working', 'trying', 'looking', 'going',
        'coming', 'leaving', 'staying', 'waiting', 'wondering', 'thinking',
        'here', 'there', 'back', 'home', 'done', 'finished', 'new',
        'not', 'just', 'still', 'also', 'too', 'very', 'really',
    }
    
    def __init__(self):
        """Initialize extractor with compiled patterns."""
        self.compiled_name_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.NAME_PATTERNS
        ]
        self.compiled_interest_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INTEREST_PATTERNS
        ]
    
    def extract_name(self, text: str) -> Optional[str]:
        """
        Extract user's name from text if present.
        
        Args:
            text: User input text
            
        Returns:
            Extracted name or None
        """
        for pattern in self.compiled_name_patterns:
            match = pattern.search(text)
            if match:
                name = match.group(1)
                # Validate it's likely a name
                if self._is_valid_name(name):
                    return name.capitalize()
        return None
    
    def _is_valid_name(self, name: str) -> bool:
        """Check if extracted word is likely a name."""
        name_lower = name.lower()
        
        # Must not be in our exclusion list
        if name_lower in self.NOT_NAMES:
            return False
        
        # Must be reasonable length
        if len(name) < 2 or len(name) > 20:
            return False
        
        # Must be alphabetic
        if not name.isalpha():
            return False
        
        return True
    
    def extract_interests(self, text: str) -> List[str]:
        """
        Extract interests from text.
        
        Args:
            text: User input text
            
        Returns:
            List of extracted interests
        """
        interests = []
        for pattern in self.compiled_interest_patterns:
            matches = pattern.findall(text)
            for match in matches:
                interest = match.strip().lower()
                if len(interest) > 2 and interest not in interests:
                    interests.append(interest)
        return interests
    
    def extract_all(self, text: str) -> Dict[str, Any]:
        """
        Extract all preferences from text.
        
        Args:
            text: User input text
            
        Returns:
            Dict with extracted preferences
        """
        result = {}
        
        name = self.extract_name(text)
        if name:
            result['name'] = name
        
        interests = self.extract_interests(text)
        if interests:
            result['interests'] = interests
        
        return result


class UserPreferenceLearner:
    """
    Learns and applies user preferences from conversation.
    
    Integrates preference extraction with storage and provides
    methods for applying preferences to responses.
    """
    
    def __init__(self, 
                 store: Optional[UserPreferencesStore] = None,
                 base_path: str = "data"):
        """
        Initialize preference learner.
        
        Args:
            store: Preferences store (creates one if None)
            base_path: Base data directory
        """
        self.store = store or UserPreferencesStore(base_path)
        self.extractor = PreferenceExtractor()
    
    def process_input(self, user_id: str, text: str) -> Dict[str, Any]:
        """
        Process user input for preference information.
        
        Extracts and stores any preferences found.
        
        Args:
            user_id: User identifier
            text: User input text
            
        Returns:
            Dict of newly learned preferences
        """
        extracted = self.extractor.extract_all(text)
        
        if not extracted:
            return {}
        
        learned = {}
        
        # Store name if found
        if 'name' in extracted:
            name = extracted['name']
            current_name = self.store.get_display_name(user_id)
            if current_name != name:
                self.store.set_display_name(user_id, name)
                learned['name'] = name
        
        # Store interests if found
        if 'interests' in extracted:
            for interest in extracted['interests']:
                self.store.add_interest(user_id, interest)
            learned['interests'] = extracted['interests']
        
        return learned
    
    def get_user_name(self, user_id: str) -> Optional[str]:
        """Get the user's preferred name."""
        return self.store.get_display_name(user_id)
    
    def set_user_name(self, user_id: str, name: str) -> None:
        """Explicitly set user's name."""
        self.store.set_display_name(user_id, name)
    
    def get_greeting(self, user_id: str) -> str:
        """Get a personalized greeting for the user."""
        name = self.store.get_display_name(user_id)
        if name:
            return f"Hello, {name}!"
        return "Hello!"
    
    def personalize_response(self, user_id: str, response: str) -> str:
        """
        Personalize a response with user preferences.
        
        Currently handles name substitution in greetings.
        Could be extended to include interests, etc.
        
        Args:
            user_id: User identifier
            response: Original response text
            
        Returns:
            Personalized response
        """
        name = self.store.get_display_name(user_id)
        
        if name:
            # Replace generic greetings with personalized ones
            response = re.sub(
                r'\b(Hello|Hi|Hey)(!|\.|,)',
                rf'\1, {name}\2',
                response,
                count=1
            )
        
        return response
    
    def format_name_confirmation(self, name: str) -> str:
        """Generate a response confirming name was learned."""
        responses = [
            f"Nice to meet you, {name}! I'll remember that.",
            f"Got it, {name}! I'll call you that from now on.",
            f"Hello {name}! I've made a note of your name.",
            f"Thanks for telling me, {name}! I won't forget.",
        ]
        import random
        return random.choice(responses)
