#!/usr/bin/env python3
"""
Manual test script for Discord bot logic without Discord.

This simulates Discord interactions to test the bot's functionality.
"""

import tempfile
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lilith.user_preferences import UserPreferencesStore, UserPreferenceLearner


def test_name_learning():
    """Test that the bot can learn user names."""
    print("\n=== Testing Name Learning ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        store = UserPreferencesStore(base_path=tmpdir)
        learner = UserPreferenceLearner(store)
        
        test_cases = [
            ("My name is Alice", "Alice"),
            ("Call me Bob", "Bob"),
            ("I'm Charlie", "Charlie"),
            ("You can call me Dave", "Dave"),
            ("I'm happy to be here", None),  # Should NOT extract "happy"
            ("Hello there", None),  # No name
        ]
        
        for i, (text, expected_name) in enumerate(test_cases):
            user_id = f"test_user_{i}"
            learned = learner.process_input(user_id, text)
            
            actual_name = store.get_display_name(user_id)
            
            if expected_name:
                if actual_name == expected_name:
                    print(f"  ✓ '{text}' → extracted '{actual_name}'")
                else:
                    print(f"  ✗ '{text}' → expected '{expected_name}', got '{actual_name}'")
            else:
                if actual_name is None:
                    print(f"  ✓ '{text}' → correctly extracted nothing")
                else:
                    print(f"  ✗ '{text}' → should not extract, got '{actual_name}'")


def test_session_cleanup():
    """Test session cleanup logic."""
    print("\n=== Testing Session Cleanup ===")
    
    from datetime import datetime, timedelta
    
    # Simulate session tracking
    user_last_active = {
        'user_active': datetime.now() - timedelta(minutes=5),
        'user_stale': datetime.now() - timedelta(minutes=45),
        'user_very_old': datetime.now() - timedelta(hours=3),
    }
    
    timeout_minutes = 30
    
    print(f"  Session timeout: {timeout_minutes} minutes")
    print(f"  Active sessions: {len(user_last_active)}")
    
    for user_id, last_active in user_last_active.items():
        age_minutes = (datetime.now() - last_active).total_seconds() / 60
        status = "WOULD FREE" if age_minutes > timeout_minutes else "active"
        print(f"    {user_id}: {age_minutes:.1f} min ago → {status}")


def test_user_retention():
    """Test user data retention logic."""
    print("\n=== Testing User Data Retention ===")
    
    from datetime import datetime, timedelta
    
    with tempfile.TemporaryDirectory() as tmpdir:
        store = UserPreferencesStore(base_path=tmpdir)
        
        # Create test users with different activity times
        from lilith.user_preferences import UserPreferences
        
        # Recent user
        recent = UserPreferences(user_id='recent_user')
        recent.display_name = 'Recent'
        store.save(recent)
        
        # Old user (10 days inactive)
        old = UserPreferences(user_id='old_user')
        old.display_name = 'Old'
        old.last_active = datetime.now() - timedelta(days=10)
        store.save(old)
        
        # Very old user (30 days inactive)
        very_old = UserPreferences(user_id='very_old_user')
        very_old.display_name = 'VeryOld'
        very_old.last_active = datetime.now() - timedelta(days=30)
        store.save(very_old)
        
        print(f"  Created 3 test users")
        
        # Check inactive users
        retention_days = 7
        inactive = store.get_inactive_users(days=retention_days)
        print(f"  Users inactive > {retention_days} days: {[u for u, _ in inactive]}")
        
        # Dry run cleanup
        would_delete = store.cleanup_inactive_users(days=retention_days, dry_run=True)
        print(f"  Would delete (dry run): {would_delete}")
        
        # Actual cleanup
        deleted = store.cleanup_inactive_users(days=retention_days, dry_run=False)
        print(f"  Actually deleted: {deleted}")
        
        remaining = store.get_all_users()
        print(f"  Remaining users: {remaining}")


def test_response_generation():
    """Test that response generation works."""
    print("\n=== Testing Response Generation ===")
    
    try:
        from lilith.embedding import PMFlowEmbeddingEncoder
        from lilith.response_fragments import ResponseFragmentStore
        from lilith.conversation_state import ConversationState
        from lilith.response_composer import ResponseComposer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = PMFlowEmbeddingEncoder()
            store = ResponseFragmentStore(encoder, str(Path(tmpdir) / "patterns.json"))
            state = ConversationState(encoder)
            composer = ResponseComposer(store, state, semantic_encoder=encoder)
            
            # Test basic response
            response = composer.compose_response(
                context="Hello",
                user_input="Hello"
            )
            
            print(f"  Input: 'Hello'")
            print(f"  Response: '{response.text}'")
            print(f"  ✓ Response generation works!")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")


def main():
    print("=" * 50)
    print("Discord Bot Logic Manual Test")
    print("=" * 50)
    
    test_name_learning()
    test_session_cleanup()
    test_user_retention()
    test_response_generation()
    
    print("\n" + "=" * 50)
    print("Manual testing complete!")
    print("=" * 50)


if __name__ == '__main__':
    main()
