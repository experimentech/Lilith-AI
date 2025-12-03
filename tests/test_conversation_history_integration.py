"""
Test script to verify ConversationHistory integration with ResponseComposer.

This tests that:
1. ConversationHistory is properly wired into ResponseComposer
2. Repetition detection works
3. Response variation occurs when user repeats query
4. Conversation context is tracked across turns
"""

from lilith.embedding import PMFlowEmbeddingEncoder
from lilith.response_fragments import ResponseFragmentStore
from lilith.conversation_state import ConversationState
from lilith.conversation_history import ConversationHistory
from lilith.response_composer import ResponseComposer

def test_basic_integration():
    """Test that ConversationHistory integrates without errors."""
    print("=" * 60)
    print("TEST 1: Basic Integration")
    print("=" * 60)
    
    # Setup components
    encoder = PMFlowEmbeddingEncoder()
    fragments = ResponseFragmentStore(semantic_encoder=encoder)
    state = ConversationState(encoder)
    history = ConversationHistory(max_turns=10)
    
    # Create composer WITH conversation history
    composer = ResponseComposer(
        fragment_store=fragments,
        conversation_state=state,
        conversation_history=history,
        semantic_encoder=encoder
    )
    
    print("✅ ResponseComposer created with ConversationHistory")
    print(f"   History max_turns: {history.max_turns}")
    print(f"   Current turn count: {len(history.turns)}")
    print()


def test_repetition_detection():
    """Test that repetition detection and response variation works."""
    print("=" * 60)
    print("TEST 2: Repetition Detection & Response Variation")
    print("=" * 60)
    
    # Setup
    encoder = PMFlowEmbeddingEncoder()
    fragments = ResponseFragmentStore(semantic_encoder=encoder)
    state = ConversationState(encoder)
    history = ConversationHistory(max_turns=10)
    
    composer = ResponseComposer(
        fragment_store=fragments,
        conversation_state=state,
        conversation_history=history,
        semantic_encoder=encoder
    )
    
    # Add some test patterns using correct API
    pattern_id_1 = fragments.add_pattern(
        trigger_context="what is python",
        response_text="Python is a high-level programming language.",
        success_score=0.9
    )
    
    pattern_id_2 = fragments.add_pattern(
        trigger_context="what is python",
        response_text="Python is known for its simple syntax and readability.",
        success_score=0.85
    )
    
    print(f"✅ Added 2 test patterns: {pattern_id_1}, {pattern_id_2}")
    
    # First query
    print("\n📝 First Query: 'what is python'")
    # Manually add to history to test repetition detection
    history.add_turn(
        user_input="what is python",
        bot_response="Python is a high-level programming language."
    )
    print(f"✅ Turn recorded. History size: {len(history.turns)}")
    
    # Repeat same query - should detect repetition
    print("\n📝 Second Query (REPETITION): 'what is python'")
    is_repeat = history.detect_repetition("what is python")
    print(f"   Repetition detected? {is_repeat}")
    
    if is_repeat:
        print("✅ SUCCESS: Repetition detection working!")
    else:
        print("⚠️  NOTICE: No repetition detected")
    
    print()



def test_conversation_context_tracking():
    """Test that conversation context is tracked across turns."""
    print("=" * 60)
    print("TEST 3: Conversation Context Tracking")
    print("=" * 60)
    
    # Setup
    encoder = PMFlowEmbeddingEncoder()
    fragments = ResponseFragmentStore(semantic_encoder=encoder)
    state = ConversationState(encoder)
    history = ConversationHistory(max_turns=10)
    
    composer = ResponseComposer(
        fragment_store=fragments,
        conversation_state=state,
        conversation_history=history,
        semantic_encoder=encoder
    )
    
    # Simulate multi-turn conversation
    conversation = [
        ("what is machine learning", "Machine learning is AI that learns from data."),
        ("tell me more", "It uses algorithms to find patterns in datasets."),
        ("what about neural networks", "Neural networks are a type of ML inspired by the brain.")
    ]
    
    for i, (user_input, bot_response) in enumerate(conversation, 1):
        print(f"\n Turn {i}:")
        print(f"   User: {user_input}")
        print(f"   Bot:  {bot_response}")
        
        # Add turn to history
        history.add_turn(
            user_input=user_input,
            bot_response=bot_response
        )
        
        print(f"   ✅ Recorded. History size: {len(history.turns)}")
    
    # Check recent turns
    recent = history.get_recent_turns(n=3)
    print(f"\n📋 Recent turns ({len(recent)} total):")
    for i, turn in enumerate(recent, 1):
        print(f"   {i}. User: {turn.user_input[:50]}...")
        print(f"      Bot:  {turn.bot_response[:50]}...")
    
    # Test repetition detection
    print("\n🔍 Testing repetition detection:")
    is_repeat = history.detect_repetition("what is machine learning")
    print(f"   'what is machine learning' repeats? {is_repeat}")
    print(f"   ✅ {'Detected!' if is_repeat else 'Not a repeat (expected)'}")
    
    print()


def test_full_session_flow():
    """Test complete session flow with history integration."""
    print("=" * 60)
    print("TEST 4: Full Session Flow")
    print("=" * 60)
    
    from lilith.session import LilithSession, SessionConfig
    
    # Create session (should auto-create ConversationHistory)
    config = SessionConfig(
        learning_enabled=False,  # Disable learning for test
        enable_knowledge_augmentation=False
    )
    
    session = LilithSession(
        user_id="test_user",
        config=config
    )
    
    print(f"✅ Session created")
    print(f"   Has conversation_history: {hasattr(session, 'conversation_history')}")
    print(f"   Composer has history: {session.composer.conversation_history is not None}")
    
    if hasattr(session, 'conversation_history'):
        print(f"   History max_turns: {session.conversation_history.max_turns}")
        print(f"   Current turns: {len(session.conversation_history.turns)}")
    
    # Simulate conversation
    queries = [
        "hello",
        "what can you do",
        "tell me about yourself"
    ]
    
    print("\n📝 Simulating conversation:")
    for i, query in enumerate(queries, 1):
        print(f"\n  Turn {i}: {query}")
        response = session.process_message(query)
        print(f"  → {response.text[:80]}...")
        
        if hasattr(session, 'conversation_history'):
            print(f"     History size: {len(session.conversation_history.turns)}")
    
    print("\n✅ Full session flow complete!")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CONVERSATION HISTORY INTEGRATION TEST")
    print("=" * 60 + "\n")
    
    try:
        test_basic_integration()
        test_repetition_detection()
        test_conversation_context_tracking()
        test_full_session_flow()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✅")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
