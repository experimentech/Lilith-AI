"""
Tests for the LayeredFragmentStore architecture.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from lilith.layered_store import (
    LayeredFragmentStore, 
    LayerStackBuilder, 
    LayerConfig,
    LayerPermission,
    MergeStrategy,
    create_dm_stack,
    create_server_stack
)
from lilith.server_store import ServerStoreManager, ServerSettings
from lilith.embedding import PMFlowEmbeddingEncoder


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    temp = tempfile.mkdtemp(prefix="lilith_test_")
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def encoder():
    """Create a shared encoder."""
    return PMFlowEmbeddingEncoder()


class MockStore:
    """Simple mock store for testing."""
    
    def __init__(self, name: str, patterns: dict = None):
        self.name = name
        self.patterns = patterns or {}
        self._upvotes = {}
        self._downvotes = {}
    
    def find_response(self, query: str, context: str = None):
        """Find a response for the query."""
        for pattern, response in self.patterns.items():
            if pattern.lower() in query.lower():
                return MockResponse(response, 0.8, pattern)
        return None
    
    def add_pattern(self, pattern: str, response: str, **kwargs):
        """Add a pattern."""
        self.patterns[pattern] = response
        return f"{self.name}:{pattern}"
    
    def upvote(self, pattern_id: str, strength: float = 0.1):
        """Record upvote."""
        self._upvotes[pattern_id] = self._upvotes.get(pattern_id, 0) + strength
    
    def downvote(self, pattern_id: str, strength: float = 0.1):
        """Record downvote."""
        self._downvotes[pattern_id] = self._downvotes.get(pattern_id, 0) + strength
    
    def get_pattern_count(self):
        return {'total': len(self.patterns)}


class MockResponse:
    """Mock response object."""
    
    def __init__(self, text: str, confidence: float, pattern_id: str = None):
        self.text = text
        self.confidence = confidence
        self.pattern_id = pattern_id


class TestLayerConfig:
    """Tests for LayerConfig."""
    
    def test_read_only_permission(self):
        store = MockStore("test")
        config = LayerConfig(
            name="test",
            store=store,
            permission=LayerPermission.READ_ONLY,
            priority=0
        )
        assert config.can_read()
        assert not config.can_write()
    
    def test_read_write_permission(self):
        store = MockStore("test")
        config = LayerConfig(
            name="test",
            store=store,
            permission=LayerPermission.READ_WRITE,
            priority=0
        )
        assert config.can_read()
        assert config.can_write()
    
    def test_write_only_permission(self):
        store = MockStore("test")
        config = LayerConfig(
            name="test",
            store=store,
            permission=LayerPermission.WRITE_ONLY,
            priority=0
        )
        assert not config.can_read()
        assert config.can_write()


class TestLayeredFragmentStore:
    """Tests for LayeredFragmentStore."""
    
    def test_basic_creation(self):
        """Test creating a basic layered store."""
        base = MockStore("base", {"hello": "Hi there!"})
        user = MockStore("user")
        
        layers = [
            LayerConfig("base", base, LayerPermission.READ_ONLY, priority=0),
            LayerConfig("user", user, LayerPermission.READ_WRITE, priority=10),
        ]
        
        store = LayeredFragmentStore(layers)
        assert len(store.layers) == 2
        assert store.get_layer("base") is not None
        assert store.get_layer("user") is not None
    
    def test_read_from_base_layer(self):
        """Test reading from base (read-only) layer."""
        base = MockStore("base", {"hello": "Hi there!"})
        user = MockStore("user")
        
        layers = [
            LayerConfig("base", base, LayerPermission.READ_ONLY, priority=0),
            LayerConfig("user", user, LayerPermission.READ_WRITE, priority=10),
        ]
        
        store = LayeredFragmentStore(layers)
        response, result = store.find_response("hello")
        
        assert response is not None
        assert response.text == "Hi there!"
        assert result.layer_name == "base"
    
    def test_user_layer_takes_priority(self):
        """Test that higher priority layers are preferred."""
        base = MockStore("base", {"hello": "Hi from base!"})
        user = MockStore("user", {"hello": "Hi from user!"})
        
        layers = [
            LayerConfig("base", base, LayerPermission.READ_ONLY, priority=0, weight=1.0),
            LayerConfig("user", user, LayerPermission.READ_WRITE, priority=10, weight=1.2),
        ]
        
        store = LayeredFragmentStore(layers)
        response, result = store.find_response("hello")
        
        assert response is not None
        # User layer has higher weight, so it should win
        assert "user" in result.layer_name.lower() or response.text == "Hi from user!"
    
    def test_write_to_user_layer(self):
        """Test writing goes to the writable layer."""
        base = MockStore("base", {"hello": "Hi from base!"})
        user = MockStore("user")
        
        layers = [
            LayerConfig("base", base, LayerPermission.READ_ONLY, priority=0),
            LayerConfig("user", user, LayerPermission.READ_WRITE, priority=10),
        ]
        
        store = LayeredFragmentStore(layers, default_write_layer="user")
        
        # Write a pattern
        pattern_id = store.add_pattern("what is 2+2", "2+2 equals 4")
        
        # Should be in user store
        assert "what is 2+2" in user.patterns
        assert user.patterns["what is 2+2"] == "2+2 equals 4"
        
        # Base should be unchanged
        assert "what is 2+2" not in base.patterns
    
    def test_cannot_write_to_readonly_layer(self):
        """Test that writing doesn't affect read-only layers."""
        base = MockStore("base", {"hello": "Hi from base!"})
        
        layers = [
            LayerConfig("base", base, LayerPermission.READ_ONLY, priority=0),
        ]
        
        store = LayeredFragmentStore(layers)
        
        # Try to write - should fail gracefully
        pattern_id = store.add_pattern("new pattern", "new response")
        
        # Base should be unchanged
        assert "new pattern" not in base.patterns


class TestLayerStackBuilder:
    """Tests for the fluent layer builder."""
    
    def test_basic_dm_stack(self):
        """Test building a DM context stack."""
        base = MockStore("base")
        user = MockStore("user")
        
        stack = (LayerStackBuilder()
            .add_base(base)
            .add_user(user, user_id="user123")
            .with_default_write_layer("user:user123")
            .build())
        
        assert len(stack.layers) == 2
        assert stack.get_layer("base") is not None
        assert stack.get_layer("user:user123") is not None
    
    def test_server_stack_with_user(self):
        """Test building a server context stack."""
        base = MockStore("base")
        server = MockStore("server")
        user = MockStore("user")
        
        stack = (LayerStackBuilder()
            .add_base(base)
            .add_server(server, guild_id="guild123")
            .add_user(user, user_id="user456", read_only=True)
            .with_default_write_layer("server:guild123")
            .build())
        
        assert len(stack.layers) == 3
        
        # Server should be writable
        server_layer = stack.get_layer("server:guild123")
        assert server_layer.can_write()
        
        # User should be read-only in server context
        user_layer = stack.get_layer("user:user456")
        assert user_layer.can_read()
        assert not user_layer.can_write()


class TestServerStoreManager:
    """Tests for ServerStoreManager."""
    
    def test_load_settings(self, temp_dir, encoder):
        """Test loading/creating server settings."""
        manager = ServerStoreManager(base_path=temp_dir, encoder=encoder)
        
        settings = manager.load_settings("guild123", "Test Server")
        
        assert settings.guild_id == "guild123"
        assert settings.guild_name == "Test Server"
        assert settings.learning_enabled is True
        assert settings.passive_listening is False
    
    def test_save_and_reload_settings(self, temp_dir, encoder):
        """Test saving and reloading settings."""
        manager = ServerStoreManager(base_path=temp_dir, encoder=encoder)
        
        settings = manager.load_settings("guild123", "Test Server")
        settings.learning_enabled = False
        settings.passive_listening = True
        manager.save_settings(settings)
        
        # Clear cache and reload
        manager._settings_cache.clear()
        reloaded = manager.load_settings("guild123")
        
        assert reloaded.learning_enabled is False
        assert reloaded.passive_listening is True
    
    def test_teaching_roles(self, temp_dir, encoder):
        """Test teaching role management."""
        manager = ServerStoreManager(base_path=temp_dir, encoder=encoder)
        
        manager.add_teaching_role("guild123", "role456")
        settings = manager.load_settings("guild123")
        
        assert "role456" in settings.teaching_roles
        assert settings.can_teach(["role456"])
        assert not settings.can_teach(["other_role"])
        
        manager.remove_teaching_role("guild123", "role456")
        settings = manager.load_settings("guild123")
        
        assert "role456" not in settings.teaching_roles
    
    def test_get_fragment_store(self, temp_dir, encoder):
        """Test getting a fragment store for a server."""
        manager = ServerStoreManager(base_path=temp_dir, encoder=encoder)
        
        store = manager.get_fragment_store("guild123", "Test Server")
        
        assert store is not None
        # Verify it's cached
        store2 = manager.get_fragment_store("guild123")
        assert store is store2


class TestIntegration:
    """Integration tests for the full layered system."""
    
    def test_dm_context_teaching(self, temp_dir, encoder):
        """Test teaching in DM context stores to user layer."""
        from lilith.response_fragments import ResponseFragmentStore
        
        # Create stores
        base_path = Path(temp_dir) / "base" / "patterns.json"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_store = ResponseFragmentStore(encoder, str(base_path))
        
        user_path = Path(temp_dir) / "users" / "user123" / "patterns.json"
        user_path.parent.mkdir(parents=True, exist_ok=True)
        user_store = ResponseFragmentStore(encoder, str(user_path))
        
        # Build DM stack
        stack = (LayerStackBuilder()
            .add_base(base_store)
            .add_user(user_store, user_id="user123")
            .with_default_write_layer("user:user123")
            .build())
        
        # Teach something
        pattern_id = stack.add_pattern(
            "what color is the sky",
            "The sky is blue during the day.",
            intent="taught"
        )
        
        assert pattern_id is not None
        
        # Should be findable
        response, result = stack.find_response("what color is the sky")
        assert response is not None
    
    def test_server_context_teaching(self, temp_dir, encoder):
        """Test teaching in server context stores to server layer."""
        from lilith.response_fragments import ResponseFragmentStore
        
        # Create stores
        base_path = Path(temp_dir) / "base" / "patterns.json"
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_store = ResponseFragmentStore(encoder, str(base_path))
        
        server_path = Path(temp_dir) / "servers" / "guild123" / "patterns.json"
        server_path.parent.mkdir(parents=True, exist_ok=True)
        server_store = ResponseFragmentStore(encoder, str(server_path))
        
        # Build server stack
        stack = (LayerStackBuilder()
            .add_base(base_store)
            .add_server(server_store, guild_id="guild123")
            .with_default_write_layer("server:guild123")
            .build())
        
        # Teach something
        pattern_id = stack.add_pattern(
            "what is our motto",
            "Our server motto is 'Be excellent to each other'",
            intent="taught"
        )
        
        assert pattern_id is not None
        
        # Should be findable
        response, result = stack.find_response("what is our motto")
        assert response is not None
