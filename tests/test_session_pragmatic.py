#!/usr/bin/env python3
"""
Test session initialization with pragmatic templates enabled.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lilith.session import LilithSession, SessionConfig


def test_session_initialization():
    """Test that session initializes with pragmatic templates."""
    print("\n" + "="*60)
    print("TEST: Session Initialization with Pragmatic Templates")
    print("="*60)
    
    # Create config with pragmatic templates enabled
    config = SessionConfig(
        data_path="data/test_session",
        enable_pragmatic_templates=True,
        enable_compositional=True,
        composition_mode="pragmatic"
    )
    
    # Create session
    session = LilithSession(
        user_id="test_user",
        config=config
    )
    
    # Check that composer has pragmatic templates
    if hasattr(session.composer, 'pragmatic_templates') and session.composer.pragmatic_templates:
        print(f"✅ Pragmatic templates initialized")
        print(f"   Templates count: {len(session.composer.pragmatic_templates.templates)}")
    else:
        print(f"❌ Pragmatic templates NOT initialized")
    
    # Check that composer has concept store
    if hasattr(session.composer, 'concept_store') and session.composer.concept_store:
        print(f"✅ Concept store initialized")
    else:
        print(f"❌ Concept store NOT initialized")
    
    # Check composition mode
    print(f"   Composition mode: {session.composer.composition_mode}")
    
    return session


def test_pragmatic_response():
    """Test generating a response with pragmatic mode."""
    print("\n" + "="*60)
    print("TEST: Pragmatic Response Generation")
    print("="*60)
    
    # Create session
    config = SessionConfig(
        data_path="data/test_session",
        enable_pragmatic_templates=True,
        enable_compositional=True,
        composition_mode="pragmatic"
    )
    
    session = LilithSession(
        user_id="test_user",
        config=config
    )
    
    # Test greeting
    print("\n  Testing greeting:")
    response = session.process_message("Hello!")
    print(f"    User: Hello!")
    print(f"    Bot: {response.text}")
    
    # Test question (will likely fall back to pattern-based if no concepts learned)
    print("\n  Testing question:")
    response = session.process_message("What is Python?")
    print(f"    User: What is Python?")
    print(f"    Bot: {response.text}")
    print(f"    Source: {response.source}")


def main():
    """Run session tests."""
    print("\n🧪 Testing Session with Pragmatic Templates")
    
    session = test_session_initialization()
    
    # Only test responses if initialization succeeded
    if hasattr(session.composer, 'pragmatic_templates') and session.composer.pragmatic_templates:
        test_pragmatic_response()
    
    print("\n" + "="*60)
    print("✅ Session Tests Complete")
    print("="*60)


if __name__ == "__main__":
    main()
