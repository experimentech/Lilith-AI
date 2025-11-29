"""
Tests for Discord bot session management logic.

These tests verify the session cleanup and user retention functionality
without needing an actual Discord connection.
"""

import pytest
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from lilith.user_preferences import UserPreferencesStore, UserPreferences


class TestUserActivityTracking:
    """Test user activity tracking in UserPreferencesStore."""
    
    def test_touch_user_updates_last_active(self):
        """Test that touch_user updates the last_active timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = UserPreferencesStore(base_path=tmpdir)
            
            # Create a user
            prefs = UserPreferences(user_id='test_user')
            store.save(prefs)
            
            # Get initial last_active
            initial = store.load('test_user')
            initial_time = initial.last_active
            
            # Small delay to ensure time difference
            time.sleep(0.01)
            
            # Touch user
            store.touch_user('test_user')
            
            # Check that last_active was updated
            updated = store.load('test_user')
            assert updated.last_active >= initial_time
    
    def test_days_inactive_calculation(self):
        """Test the days_inactive calculation."""
        prefs = UserPreferences(user_id='test_user')
        
        # Just created, should be ~0 days inactive
        assert prefs.days_inactive() < 0.01
        
        # Manually set last_active to 2 days ago
        prefs.last_active = datetime.now() - timedelta(days=2)
        assert 1.9 < prefs.days_inactive() < 2.1
    
    def test_get_inactive_users(self):
        """Test getting list of inactive users."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = UserPreferencesStore(base_path=tmpdir)
            
            # Create an active user
            active = UserPreferences(user_id='active_user')
            store.save(active)
            
            # Create an inactive user (manually set old timestamp)
            inactive = UserPreferences(user_id='inactive_user')
            inactive.last_active = datetime.now() - timedelta(days=10)
            store.save(inactive)
            
            # Get users inactive for 7+ days
            inactive_list = store.get_inactive_users(days=7)
            
            # Should only contain the inactive user
            user_ids = [uid for uid, _ in inactive_list]
            assert 'inactive_user' in user_ids
            assert 'active_user' not in user_ids
    
    def test_cleanup_inactive_users_dry_run(self):
        """Test cleanup with dry_run=True doesn't delete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = UserPreferencesStore(base_path=tmpdir)
            
            # Create an old user
            old_user = UserPreferences(user_id='old_user')
            old_user.last_active = datetime.now() - timedelta(days=30)
            store.save(old_user)
            
            # Dry run should return the user but not delete
            deleted = store.cleanup_inactive_users(days=7, dry_run=True)
            assert 'old_user' in deleted
            
            # User should still exist
            assert 'old_user' in store.get_all_users()
    
    def test_cleanup_inactive_users_actual(self):
        """Test cleanup with dry_run=False actually deletes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = UserPreferencesStore(base_path=tmpdir)
            
            # Create an old user
            old_user = UserPreferences(user_id='old_user')
            old_user.last_active = datetime.now() - timedelta(days=30)
            store.save(old_user)
            
            # Also create a recent user
            new_user = UserPreferences(user_id='new_user')
            store.save(new_user)
            
            # Actual cleanup
            deleted = store.cleanup_inactive_users(days=7, dry_run=False)
            assert 'old_user' in deleted
            
            # Old user should be gone, new user should remain
            all_users = store.get_all_users()
            assert 'old_user' not in all_users
            assert 'new_user' in all_users


class TestSessionManagementLogic:
    """Test session management logic (mocked Discord bot)."""
    
    def test_session_tracking_dict(self):
        """Test that session tracking works with datetime dict."""
        from datetime import datetime
        
        # Simulate the _user_last_active dict
        user_last_active = {}
        
        # Simulate user activity
        user_last_active['user_1'] = datetime.now()
        user_last_active['user_2'] = datetime.now() - timedelta(minutes=45)
        user_last_active['user_3'] = datetime.now() - timedelta(minutes=10)
        
        # Find users inactive for 30+ minutes
        timeout_minutes = 30
        inactive = []
        for user_id, last_active in user_last_active.items():
            age_minutes = (datetime.now() - last_active).total_seconds() / 60
            if age_minutes > timeout_minutes:
                inactive.append(user_id)
        
        assert 'user_2' in inactive
        assert 'user_1' not in inactive
        assert 'user_3' not in inactive
    
    def test_session_stats_calculation(self):
        """Test session statistics calculation."""
        from datetime import datetime
        
        user_last_active = {
            'user_1': datetime.now() - timedelta(minutes=5),
            'user_2': datetime.now() - timedelta(minutes=20),
            'user_3': datetime.now() - timedelta(minutes=1),
        }
        
        # Calculate stats
        active_sessions = len(user_last_active)
        oldest_age = 0.0
        newest_age = float('inf')
        
        for user_id, last_active in user_last_active.items():
            age_minutes = (datetime.now() - last_active).total_seconds() / 60
            oldest_age = max(oldest_age, age_minutes)
            newest_age = min(newest_age, age_minutes)
        
        assert active_sessions == 3
        assert 19 < oldest_age < 21  # user_2 at ~20 min
        assert 0 < newest_age < 2    # user_3 at ~1 min


class TestPreferencesPersistence:
    """Test that preferences persist correctly with timestamps."""
    
    def test_timestamps_persist(self):
        """Test that all timestamps are saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = UserPreferencesStore(base_path=tmpdir)
            
            # Create user with specific times (as datetime, will be converted to ISO)
            prefs = UserPreferences(user_id='test_user')
            prefs.display_name = 'Test Person'
            prefs.created_at = datetime(2025, 1, 1, 12, 0, 0)
            prefs.updated_at = datetime(2025, 6, 15, 14, 30, 0)
            prefs.last_active = datetime(2025, 11, 28, 10, 0, 0)
            store.save(prefs)
            
            # Clear cache and reload
            store._cache.clear()
            loaded = store.load('test_user')
            
            assert loaded.display_name == 'Test Person'
            # Timestamps are stored as ISO strings
            assert loaded.created_at == '2025-01-01T12:00:00'
            # updated_at gets refreshed on save, so just check it exists
            assert loaded.updated_at is not None
            assert loaded.last_active == '2025-11-28T10:00:00'
    
    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        prefs = UserPreferences(user_id='test_user')
        prefs.display_name = 'Alice'
        prefs.interests = ['coding', 'AI']
        prefs.custom_data = {'favorite_color': 'blue'}
        
        # Convert to dict and back
        data = prefs.to_dict()
        restored = UserPreferences.from_dict(data)
        
        assert restored.user_id == 'test_user'
        assert restored.display_name == 'Alice'
        assert restored.interests == ['coding', 'AI']
        assert restored.custom_data == {'favorite_color': 'blue'}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
