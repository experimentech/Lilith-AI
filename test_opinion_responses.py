#!/usr/bin/env python3
"""Test opinion/preference conversational responses."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from lilith.session import LilithSession, SessionConfig

def test_opinion_responses():
    """Test that opinion questions get conversational responses."""
    
    print("=" * 60)
    print("Testing Opinion/Preference Conversational Responses")
    print("=" * 60)
    
    # Create session with Phase 2 enabled
    config = SessionConfig(
        enable_compositional=True,
        enable_pragmatic_templates=True,
        composition_mode="pragmatic"
    )
    
    session = LilithSession(
        user_id="test_user",
        config=config
    )
    
    # Test cases
    test_cases = [
        "Do you like birds?",
        "What do you think about machine learning?",
        "What's your favorite programming language?",
        "Are you interested in quantum physics?"
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {query}")
        print('-' * 60)
        
        response_obj = session.process_message(query)
        response = response_obj.text
        
        print(f"Response: {response}")
        print()
        
        # Check if response is conversational
        conversational_indicators = [
            "I find",
            "fascinating",
            "interesting",
            "appreciate",
            "thought-provoking",
            "remarkable"
        ]
        
        is_conversational = any(indicator in response.lower() for indicator in conversational_indicators)
        is_encyclopedic = response.startswith(("Birds are", "Machine learning is", "Programming languages are"))
        
        print(f"✓ Conversational: {is_conversational}")
        print(f"✗ Encyclopedic: {is_encyclopedic}")
        
        if is_conversational and not is_encyclopedic:
            print("✅ PASS - Response is conversational!")
        elif is_encyclopedic:
            print("❌ FAIL - Response is encyclopedic (search engine style)")
        else:
            print("⚠️  UNCLEAR - Response doesn't match expected patterns")
    
    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)

if __name__ == "__main__":
    test_opinion_responses()
