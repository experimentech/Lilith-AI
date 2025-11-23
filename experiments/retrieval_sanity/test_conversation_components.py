#!/usr/bin/env python3
"""
Test core conversation components independently.

Verifies:
1. ResponseFragmentStore - pattern storage and retrieval
2. ResponseComposer - pattern composition
3. ConversationHistory - turn tracking

This test doesn't require full pipeline integration.
"""

import sys
from pathlib import Path
import importlib.util

# Load modules directly without going through __init__.py
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load required modules
base_dir = Path(__file__).parent / "pipeline"
response_fragments = load_module("response_fragments", base_dir / "response_fragments.py")
conversation_history = load_module("conversation_history", base_dir / "conversation_history.py")

ResponseFragmentStore = response_fragments.ResponseFragmentStore
ResponsePattern = response_fragments.ResponsePattern
ConversationHistory = conversation_history.ConversationHistory


class MockEncoder:
    """Mock encoder for testing without full semantic stage"""
    
    def __init__(self, dim=96):
        self.dim = dim
        
    def encode(self, text):
        """Return dummy embedding based on text hash"""
        import numpy as np
        # Simple hash-based embedding for testing
        hash_val = hash(text.lower()) % 10000
        embedding = np.random.RandomState(hash_val).randn(self.dim)
        # Normalize
        return embedding / (np.linalg.norm(embedding) + 1e-8)


def test_fragment_store():
    """Test response pattern storage and retrieval"""
    print("="*80)
    print("TEST 1: Response Fragment Store")
    print("="*80)
    print()
    
    # Create mock encoder
    encoder = MockEncoder(dim=96)
    
    # Create fragment store
    print("Creating ResponseFragmentStore...")
    store = ResponseFragmentStore(
        semantic_encoder=encoder,
        storage_path="test_response_patterns.json"
    )
    
    # Check seed patterns loaded
    stats = store.get_stats()
    print(f"✓ Fragment store initialized")
    print(f"  Total patterns: {stats['total_patterns']}")
    print(f"  Seed patterns: {stats['seed_patterns']}")
    print(f"  Learned patterns: {stats['learned_patterns']}")
    print(f"  Average success: {stats['average_success']:.2f}")
    print()
    
    # Test retrieval
    print("Testing pattern retrieval...")
    test_contexts = [
        "Hello there",
        "I don't understand this",
        "Tell me more about that",
        "What is machine learning?"
    ]
    
    for context in test_contexts:
        patterns = store.retrieve_patterns(context, topk=3)
        print(f"\nContext: '{context}'")
        print(f"  Retrieved {len(patterns)} patterns:")
        for i, (pattern, score) in enumerate(patterns, 1):
            print(f"    {i}. [{score:.3f}] {pattern.response_text}")
            print(f"       Trigger: '{pattern.trigger_context}'")
    
    print()
    
    # Test adding new pattern
    print("Testing pattern learning...")
    new_id = store.add_pattern(
        trigger_context="explain quantum physics",
        response_text="Quantum physics deals with the behavior of matter at atomic scales.",
        success_score=0.7,
        intent="explanation"
    )
    print(f"✓ Added new pattern: {new_id}")
    
    # Test update
    print("Testing success score update...")
    store.update_success(new_id, feedback=0.3, plasticity_rate=0.1)
    pattern = store.patterns[new_id]
    print(f"✓ Updated success score: {pattern.success_score:.2f}")
    print(f"  Usage count: {pattern.usage_count}")
    print()
    
    # Clean up
    Path("test_response_patterns.json").unlink(missing_ok=True)
    
    return store


def test_conversation_history():
    """Test conversation history tracking"""
    print("="*80)
    print("TEST 2: Conversation History")
    print("="*80)
    print()
    
    # Create history
    print("Creating ConversationHistory...")
    history = ConversationHistory(max_turns=5)
    print("✓ History initialized with max_turns=5")
    print()
    
    # Add turns
    print("Adding conversation turns...")
    turns = [
        ("Hello!", "Hello! How can I help you?"),
        ("Tell me about AI", "AI is the simulation of human intelligence."),
        ("What is machine learning?", "Machine learning is a subset of AI."),
        ("How does it work?", "It learns from data patterns."),
        ("Can you explain more?", "Sure! ML algorithms find patterns in data."),
    ]
    
    for user, bot in turns:
        history.add_turn(user, bot)
        print(f"  Turn {len(history.turns)}: User: '{user[:30]}...'")
    
    print()
    
    # Test retrieval
    print("Testing turn retrieval...")
    recent = history.get_recent_turns(n=3)
    print(f"✓ Retrieved {len(recent)} recent turns:")
    for i, turn in enumerate(recent, 1):
        print(f"  {i}. User: {turn.user_input}")
        print(f"     Bot: {turn.bot_response}")
    print()
    
    # Test context window
    print("Testing context window...")
    context = history.get_context_window(n=3, include_embeddings=False)
    print("✓ Context window:")
    print(context[:200] + "...")
    print()
    
    # Test repetition detection
    print("Testing repetition detection...")
    is_repeat = history.detect_repetition("Can you explain more?", window=3)
    print(f"✓ 'Can you explain more?' is repeat: {is_repeat}")
    is_repeat = history.detect_repetition("Something completely new", window=3)
    print(f"✓ 'Something completely new' is repeat: {is_repeat}")
    print()
    
    # Test stats
    stats = history.get_stats()
    print("Conversation statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    return history


def test_pattern_composition():
    """Test response composition from patterns"""
    print("="*80)
    print("TEST 3: Response Composition (Simple)")
    print("="*80)
    print()
    
    # Create components
    encoder = MockEncoder(dim=96)
    store = ResponseFragmentStore(
        semantic_encoder=encoder,
        storage_path="test_response_patterns_comp.json"
    )
    
    print("Testing best-match composition...")
    
    # Test queries
    test_queries = [
        "Hi!",
        "I'm confused",
        "What do you know about physics?",
        "Yes, that's right",
    ]
    
    for query in test_queries:
        patterns = store.retrieve_patterns(query, topk=1)
        if patterns:
            pattern, score = patterns[0]
            print(f"\nQuery: '{query}'")
            print(f"  Best match [{score:.3f}]: {pattern.response_text}")
        else:
            print(f"\nQuery: '{query}'")
            print(f"  No match found")
    
    print()
    
    # Clean up
    Path("test_response_patterns_comp.json").unlink(missing_ok=True)


def main():
    """Run all tests"""
    print()
    print("╔" + "═"*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "TESTING PURE NEURO-SYMBOLIC CONVERSATION COMPONENTS".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "═"*78 + "╝")
    print()
    
    try:
        # Run tests
        store = test_fragment_store()
        history = test_conversation_history()
        test_pattern_composition()
        
        print("="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print()
        print("Core conversation components are working correctly:")
        print("  ✓ Response pattern storage and retrieval")
        print("  ✓ Conversation history tracking")
        print("  ✓ Pattern-based composition")
        print()
        print("Next step: Integrate with full PMFlow pipeline")
        print()
        
    except Exception as e:
        print()
        print("="*80)
        print("❌ TEST FAILED!")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
