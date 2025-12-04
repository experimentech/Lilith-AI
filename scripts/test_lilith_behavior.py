#!/usr/bin/env python3
"""
Comprehensive behavior test for Lilith.

Tests the key functionality:
1. Declarative learning (facts)
2. Topic extraction (BNN-based)
3. Proactive knowledge augmentation
4. Deliberation relevance validation
5. Response composition
"""

import tempfile
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def separator(title: str):
    """Print a section separator."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_topic_extractor():
    """Test the BNN-based topic extractor."""
    separator("Topic Extractor Tests")
    
    from lilith.embedding import PMFlowEmbeddingEncoder
    from lilith.topic_extractor import TopicExtractor
    
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    extractor = TopicExtractor(encoder)
    
    # Test 1: Learn some topics
    print("\n📚 Learning topics from declarations...")
    extractor.learn_topic("dogs", "Dogs are loyal animals")
    extractor.learn_topic("cats", "Cats are independent pets")
    extractor.learn_topic("elephants", "Elephants are large mammals")
    print(f"   Learned topics: {list(extractor.topics.keys())}")
    
    # Test 2: Extract known topics
    print("\n🔍 Testing extraction of known topics:")
    test_cases = [
        ("tell me about dogs", "dogs"),
        ("what are cats?", "cats"),
        ("do you know about elephants?", "elephants"),
        ("I want to learn about dogs", "dogs"),
    ]
    
    passed = 0
    for query, expected in test_cases:
        result, score = extractor.extract_topic(query)  # Returns (topic, score)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        print(f"   {status} '{query}' → '{result}' (score: {score:.2f}, expected: '{expected}')")
    
    # Test 3: Extract unknown topics (fallback)
    print("\n🔍 Testing extraction of unknown topics (fallback):")
    unknown_cases = [
        ("tell me about giraffes", "giraffes"),
        ("what is a platypus?", "platypus"),
        ("do you know about quantum physics?", "quantum physics"),
    ]
    
    for query, expected in unknown_cases:
        result, score = extractor.extract_topic(query)  # Returns (topic, score)
        # Fallback should strip scaffolding words
        matches = result and expected.lower() in result.lower()
        status = "✓" if matches else "~"
        print(f"   {status} '{query}' → '{result}' (score: {score:.2f}, expected contains: '{expected}')")
    
    print(f"\n   Known topic tests: {passed}/{len(test_cases)} passed")


def test_declarative_learning():
    """Test declarative learning (facts from user)."""
    separator("Declarative Learning Tests")
    
    from lilith.embedding import PMFlowEmbeddingEncoder
    from lilith.multi_tenant_store import MultiTenantFragmentStore
    from lilith.user_auth import UserIdentity, AuthMode
    from lilith.session import LilithSession, SessionConfig
    
    with tempfile.TemporaryDirectory() as tmpdir:
        encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
        
        user = UserIdentity(
            user_id="test_user",
            auth_mode=AuthMode.SIMPLE,
            display_name="Tester"
        )
        
        store = MultiTenantFragmentStore(
            encoder, user, base_data_path=tmpdir
        )
        
        config = SessionConfig(
            data_path=tmpdir,
            learning_enabled=True,
            enable_declarative_learning=True,
            enable_feedback_detection=False,
            plasticity_enabled=True
        )
        
        session = LilithSession(
            user_id="test_user",
            context_id="test",
            config=config,
            store=store,
            display_name="Tester"
        )
        
        # Test 1: Teach a fact
        print("\n📝 Teaching a fact...")
        response = session.process_message("Dogs are loyal animals")
        print(f"   Input: 'Dogs are loyal animals'")
        print(f"   Response: '{response.text[:100]}...'")
        print(f"   Learned fact: {response.learned_fact}")
        
        # Check if topic was learned
        if session.topic_extractor:
            topics = list(session.topic_extractor.topics.keys())
            print(f"   Topics learned: {topics}")
            learned_dogs = "dogs" in topics
            print(f"   ✓ Topic 'dogs' learned" if learned_dogs else "   ✗ Topic 'dogs' NOT learned")
        
        # Test 2: Ask about the fact
        print("\n🔍 Asking about the fact...")
        response = session.process_message("What are dogs?")
        print(f"   Input: 'What are dogs?'")
        print(f"   Response: '{response.text[:100]}...'")
        
        # Check if it's from learned knowledge
        contains_loyal = "loyal" in response.text.lower()
        print(f"   ✓ Contains learned info" if contains_loyal else "   ~ May need more context")
        
        # Test 3: Word boundary check - "dogs" shouldn't match "do"
        print("\n🔍 Testing word boundary (dogs vs do)...")
        response = session.process_message("How do you work?")
        print(f"   Input: 'How do you work?'")
        print(f"   Response: '{response.text[:100]}...'")
        
        # This should NOT contain dog-related info
        contains_dogs = "loyal" in response.text.lower() or "dogs" in response.text.lower()
        print(f"   ✓ Word boundary correct" if not contains_dogs else "   ✗ Word boundary FAILED - matched 'do' to 'dogs'")


def test_deliberation_relevance():
    """Test that deliberation rejects unrelated concepts."""
    separator("Deliberation Relevance Tests")
    
    from lilith.embedding import PMFlowEmbeddingEncoder
    from lilith.multi_tenant_store import MultiTenantFragmentStore
    from lilith.user_auth import UserIdentity, AuthMode
    from lilith.session import LilithSession, SessionConfig
    
    with tempfile.TemporaryDirectory() as tmpdir:
        encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
        
        user = UserIdentity(
            user_id="test_user",
            auth_mode=AuthMode.SIMPLE,
            display_name="Tester"
        )
        
        store = MultiTenantFragmentStore(
            encoder, user, base_data_path=tmpdir
        )
        
        config = SessionConfig(
            data_path=tmpdir,
            learning_enabled=True,
            enable_declarative_learning=True,
            enable_feedback_detection=False,
            plasticity_enabled=True
        )
        
        session = LilithSession(
            user_id="test_user",
            context_id="test",
            config=config,
            store=store,
            display_name="Tester"
        )
        
        # Teach about birds
        print("\n📝 Teaching about birds...")
        session.process_message("Birds can fly in the sky")
        print("   Taught: 'Birds can fly in the sky'")
        
        # Ask about colours (should NOT get birds response)
        print("\n🔍 Testing relevance filtering (colours vs birds)...")
        response = session.process_message("What are colours?")
        print(f"   Input: 'What are colours?'")
        print(f"   Response: '{response.text[:150]}...'")
        
        # Check for unrelated content
        contains_birds = "bird" in response.text.lower() or "fly" in response.text.lower()
        print(f"   ✓ Correctly filtered unrelated concepts" if not contains_birds else "   ✗ FAILED - returned birds for colours query")


def test_proactive_augmentation():
    """Test proactive knowledge augmentation."""
    separator("Proactive Augmentation Tests")
    
    from lilith.embedding import PMFlowEmbeddingEncoder
    from lilith.multi_tenant_store import MultiTenantFragmentStore
    from lilith.user_auth import UserIdentity, AuthMode
    from lilith.session import LilithSession, SessionConfig
    
    with tempfile.TemporaryDirectory() as tmpdir:
        encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
        
        user = UserIdentity(
            user_id="test_user",
            auth_mode=AuthMode.SIMPLE,
            display_name="Tester"
        )
        
        store = MultiTenantFragmentStore(
            encoder, user, base_data_path=tmpdir
        )
        
        config = SessionConfig(
            data_path=tmpdir,
            learning_enabled=True,
            enable_declarative_learning=True,
            enable_feedback_detection=False,
            plasticity_enabled=True
        )
        
        session = LilithSession(
            user_id="test_user",
            context_id="test",
            config=config,
            store=store,
            display_name="Tester"
        )
        
        # Ask about something completely unknown
        print("\n🔍 Asking about unknown topic (should try external sources)...")
        print("   Note: This may take a moment if fetching from Wikipedia...")
        
        response = session.process_message("Tell me about elephants")
        print(f"   Input: 'Tell me about elephants'")
        print(f"   Response: '{response.text[:200]}...'")
        print(f"   Is fallback: {response.is_fallback}")
        print(f"   Is low confidence: {response.is_low_confidence}")
        
        # If we got a response about elephants, proactive augmentation worked
        is_relevant = "elephant" in response.text.lower() or "mammal" in response.text.lower()
        print(f"   ✓ Got relevant external knowledge" if is_relevant else "   ~ No external knowledge (may be rate limited)")


def test_query_cleaning():
    """Test query cleaning for Wikipedia lookups."""
    separator("Query Cleaning Tests")
    
    from lilith.embedding import PMFlowEmbeddingEncoder
    from lilith.knowledge_augmenter import WikipediaLookup
    from lilith.topic_extractor import TopicExtractor
    
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    topic_extractor = TopicExtractor(encoder)
    
    # Learn some topics
    topic_extractor.learn_topic("dogs", "Dogs are animals")
    
    wiki = WikipediaLookup()
    wiki.topic_extractor = topic_extractor  # Set via attribute
    
    test_cases = [
        ("What is Python?", "python"),
        ("Tell me about dogs", "dogs"),
        ("Do you know about cats?", "cats"),
        ("Do you know of elephants?", "elephants"),
        ("What are quantum computers?", "quantum computers"),
        ("How does machine learning work?", "machine learning"),
    ]
    
    print("\n🔍 Testing query cleaning:")
    for query, expected_contains in test_cases:
        cleaned = wiki._clean_query(query)
        matches = expected_contains.lower() in cleaned.lower()
        status = "✓" if matches else "~"
        print(f"   {status} '{query}' → '{cleaned}' (should contain: '{expected_contains}')")


def test_full_conversation_flow():
    """Test a full conversation flow."""
    separator("Full Conversation Flow Test")
    
    from lilith.embedding import PMFlowEmbeddingEncoder
    from lilith.multi_tenant_store import MultiTenantFragmentStore
    from lilith.user_auth import UserIdentity, AuthMode
    from lilith.session import LilithSession, SessionConfig
    
    with tempfile.TemporaryDirectory() as tmpdir:
        encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
        
        user = UserIdentity(
            user_id="test_user",
            auth_mode=AuthMode.SIMPLE,
            display_name="Tester"
        )
        
        store = MultiTenantFragmentStore(
            encoder, user, base_data_path=tmpdir
        )
        
        config = SessionConfig(
            data_path=tmpdir,
            learning_enabled=True,
            enable_declarative_learning=True,
            enable_feedback_detection=False,
            plasticity_enabled=True
        )
        
        session = LilithSession(
            user_id="test_user",
            context_id="test",
            config=config,
            store=store,
            display_name="Tester"
        )
        
        conversation = [
            ("Hello!", "greeting"),
            ("My name is Alice", "name intro"),
            ("Dogs are loyal companions", "teaching fact"),
            ("What do you know about dogs?", "recall fact"),
            ("What is the color red?", "unknown topic"),
        ]
        
        print("\n🗣️  Simulating conversation:")
        for user_input, description in conversation:
            response = session.process_message(user_input)
            print(f"\n   [{description}]")
            print(f"   You: {user_input}")
            print(f"   Lilith: {response.text[:150]}{'...' if len(response.text) > 150 else ''}")
            if response.learned_fact:
                print(f"   📖 Learned: {response.learned_fact}")


def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║        LILITH BEHAVIOR TEST SUITE                        ║")
    print("╚" + "═" * 58 + "╝")
    
    try:
        test_topic_extractor()
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
    
    try:
        test_declarative_learning()
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
    
    try:
        test_deliberation_relevance()
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
    
    try:
        test_query_cleaning()
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
    
    try:
        test_proactive_augmentation()
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
    
    try:
        test_full_conversation_flow()
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
    
    separator("Test Suite Complete")
    print("\n   Run with: python scripts/test_lilith_behavior.py")
    print()


if __name__ == "__main__":
    main()
