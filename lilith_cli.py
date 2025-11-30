"""
Multi-tenant CLI for Lilith with user authentication.

Supports different auth modes:
- Teacher mode: No username, writes to base
- User mode: Prompts for username, isolated storage
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lilith.embedding import PMFlowEmbeddingEncoder
from lilith.response_composer import ResponseComposer
from lilith.conversation_state import ConversationState
from lilith.multi_tenant_store import MultiTenantFragmentStore
from lilith.user_auth import UserAuthenticator, AuthMode

# Optional: Auto semantic learning
try:
    from lilith.auto_semantic_learner import AutoSemanticLearner
    AUTO_LEARNING_AVAILABLE = True
except ImportError:
    AUTO_LEARNING_AVAILABLE = False

# Optional: Automatic feedback detection
try:
    from lilith.feedback_detector import FeedbackDetector, FeedbackTracker
    FEEDBACK_DETECTION_AVAILABLE = True
except ImportError:
    FEEDBACK_DETECTION_AVAILABLE = False


def main():
    """Run multi-tenant Lilith CLI"""
    
    print("=" * 60)
    print("LILITH - Multi-Tenant Conversational AI")
    print("=" * 60)
    print()
    
    # Prompt for mode
    print("Select mode:")
    print("  1. Teacher mode (writes to base knowledge)")
    print("  2. User mode (isolated personal storage)")
    print()
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            auth_mode = AuthMode.NONE
            break
        elif choice == "2":
            auth_mode = AuthMode.SIMPLE
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    print()
    
    # Authenticate user
    authenticator = UserAuthenticator(auth_mode)
    user_identity = authenticator.authenticate()
    
    print()
    print(f"Welcome, {user_identity.display_name}!")
    print()
    
    # Create encoder
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    
    # Create multi-tenant fragment store
    fragment_store = MultiTenantFragmentStore(
        encoder,
        user_identity,
        base_data_path="data"
    )
    
    # Create conversation state
    state = ConversationState(encoder)
    
    # Create composer
    composer = ResponseComposer(
        fragment_store,
        state,
        semantic_encoder=encoder,
        enable_knowledge_augmentation=True,
        enable_modal_routing=True
    )
    
    # Load contrastive-trained semantic weights (if available)
    contrastive_path = Path("data/contrastive_learner")
    if contrastive_path.with_suffix('.json').exists():
        composer.load_contrastive_weights(str(contrastive_path))
    
    # Initialize auto semantic learner (learns from conversations)
    auto_learner = None
    if AUTO_LEARNING_AVAILABLE and composer.contrastive_learner:
        auto_learner = AutoSemanticLearner(
            contrastive_learner=composer.contrastive_learner,
            auto_train_threshold=10,  # Train after 10 new pairs
            auto_train_steps=3
        )
        # Load previous session state
        auto_state_path = Path("data/auto_learner_state.json")
        if auto_state_path.exists():
            auto_learner.load_state(auto_state_path)
    
    # Initialize automatic feedback detection
    feedback_tracker = None
    if FEEDBACK_DETECTION_AVAILABLE:
        feedback_tracker = FeedbackTracker(
            detector=FeedbackDetector(
                min_confidence=0.4,
                apply_threshold=0.5,
                strong_strength=0.2,
                weak_strength=0.1
            )
        )
    
    print("🤖 Lilith initialized")
    if feedback_tracker:
        print("   🎯 Auto-feedback detection enabled")
    if auto_learner:
        print("   📚 Auto-learning from conversations enabled")
    print()
    
    # Show pattern counts
    counts = fragment_store.get_pattern_count()
    if user_identity.is_teacher():
        print(f"   Base patterns: {counts['base']}")
    else:
        print(f"   Your patterns: {counts['user']}")
        print(f"   Base patterns: {counts['base']}")
        print(f"   Total accessible: {counts['total']}")
    
    print()
    print("Type '/quit' or '/exit' to exit")
    print("Commands: '/teach', '/stats', '/reset', '/help'")
    print("Feedback: '/+' (upvote), '/-' (downvote), '/?' (show last pattern ID)")
    print("=" * 60)
    print()
    
    # Conversation loop
    turn = 0
    last_pattern_id = None
    last_user_input = None  # Track last query for learning
    last_response_text = None  # Track last response for learning
    
    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Save auto-learner state before exit
            if auto_learner:
                auto_learner.force_train()
                auto_learner.save_state(Path("data/auto_learner_state.json"))
            print("\n\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Commands must start with '/'
        if user_input.startswith('/'):
            command = user_input[1:].lower()
            
            if command in ['quit', 'exit']:
                # Save auto-learner state before exit
                if auto_learner:
                    auto_learner.force_train()  # Train on any pending pairs
                    auto_learner.save_state(Path("data/auto_learner_state.json"))
                    stats = auto_learner.get_stats()
                    if stats['pairs_added'] > 0:
                        print(f"\n📚 Session learning: {stats['pairs_added']} semantic pairs extracted")
                print("\nGoodbye!")
                break
            
            elif command == 'stats':
                counts = fragment_store.get_pattern_count()
                print(f"\n📊 Statistics:")
                if not user_identity.is_teacher():
                    print(f"   Your patterns: {counts['user']}")
                print(f"   Base patterns: {counts['base']}")
                print(f"   Total: {counts['total']}")
                
                # Show vocabulary stats if available
                vocab_stats = fragment_store.get_vocabulary_stats()
                if vocab_stats:
                    print(f"\n📖 Vocabulary:")
                    print(f"   Total terms: {vocab_stats['total_terms']}")
                    print(f"   Technical terms: {vocab_stats['technical_terms']}")
                    print(f"   Common terms: {vocab_stats['common_terms']}")
                    
                    if vocab_stats['top_technical']:
                        print(f"\n   Top technical terms:")
                        for term, freq in vocab_stats['top_technical'][:5]:
                            print(f"     - {term} ({freq})")
                
                # Show pattern stats if available
                pattern_stats = fragment_store.get_pattern_stats()
                if pattern_stats:
                    print(f"\n📝 Syntactic Patterns:")
                    print(f"   Total patterns: {pattern_stats['total_patterns']}")
                    print(f"   Total matches: {pattern_stats['total_matches']}")
                    
                    if pattern_stats['by_intent']:
                        print(f"\n   By intent:")
                        for intent, count in pattern_stats['by_intent'].items():
                            print(f"     - {intent}: {count}")
                    
                    if pattern_stats['top_patterns']:
                        print(f"\n   Most frequent:")
                        for template, freq, intent in pattern_stats['top_patterns'][:3]:
                            print(f"     - {template} ({freq}x)")
                
                print()
                continue
            
            elif command == 'help':
                print("\n📖 Commands:")
                print("   /quit    - Exit the program")
                print("   /exit    - Exit the program")
                print("   /teach   - Teach a new question/answer pair")
                print("   /stats   - Show pattern statistics")
                print("   /feedback- Show auto-feedback stats")
                print("   /reset   - Reset your data (with backup)")
                print("   /+       - Upvote last response")
                print("   /-       - Downvote last response")
                print("   /?       - Show last pattern ID")
                print()
                continue
            
            elif command == 'teach':
                print("\n📚 Teaching Mode")
                print("=" * 50)
                
                # Get question
                question = input("Question: ").strip()
                if not question:
                    print("❌ Teaching cancelled - no question provided")
                    print()
                    continue
                
                # Get answer
                answer = input("Answer: ").strip()
                if not answer:
                    print("❌ Teaching cancelled - no answer provided")
                    print()
                    continue
                
                # Add the pattern (layered_store handles the appropriate API)
                try:
                    pattern_id = fragment_store.add_pattern(
                        pattern=question.lower(),
                        response=answer,
                        intent="taught",
                        success_rate=0.8
                    )
                    
                    if pattern_id:
                        location = "base knowledge" if user_identity.is_teacher() else "your personal knowledge"
                        print(f"\n✅ Learned!")
                        print(f"   Q: {question}")
                        print(f"   A: {answer}")
                        print(f"   Stored in: {location}")
                        print(f"   Pattern ID: {pattern_id}")
                        print(f"\n   Try asking me now! 🎓")
                    else:
                        print(f"\n❌ Failed to store pattern (no writable layer)")
                except Exception as e:
                    print(f"\n❌ Failed to learn pattern: {e}")
                
                print()
                continue
            
            elif command == 'feedback':
                if feedback_tracker:
                    stats = feedback_tracker.get_stats()
                    print("\n📊 Auto-Feedback Statistics:")
                    print(f"   Total interactions: {stats['total_interactions']}")
                    print(f"   Positive signals: {stats['positive']} ({stats['positive_rate']:.1%})")
                    print(f"   Negative signals: {stats['negative']} ({stats['negative_rate']:.1%})")
                    print(f"   Neutral: {stats['neutral']}")
                    
                    recent = feedback_tracker.get_recent_feedback(5)
                    if recent:
                        print("\n   Recent feedback:")
                        for r in recent:
                            signal = r['feedback']
                            print(f"     - '{r['input']}' → {signal}")
                else:
                    print("\n⚠️  Feedback detection not available")
                print()
                continue
            
            elif command == 'reset':
                if user_identity.is_teacher():
                    print("\n⚠️  Cannot reset in teacher mode")
                    print()
                    continue
                
                print("\n⚠️  This will reset your personal patterns")
                confirm = input("Create backup and reset? (yes/no): ").strip().lower()
                if confirm == 'yes':
                    backup = fragment_store.reset_user_data(keep_backup=True)
                    print(f"✅ Reset complete. Backup: {backup.name if backup else 'none'}")
                else:
                    print("❌ Reset cancelled")
                print()
                continue
            
            elif command == '+':
                if last_pattern_id:
                    # Check if this was an external knowledge response
                    if last_pattern_id.startswith('external_'):
                        # Learn this Wikipedia response with concept extraction
                        if last_user_input and last_response_text:
                            print("\n📚 Learning from external knowledge...")
                            new_pattern_id = fragment_store.learn_from_wikipedia(
                                query=last_user_input,
                                response_text=last_response_text,
                                success_score=0.8,
                                intent="learned_knowledge"
                            )
                            print(f"✅ Learned new pattern: {new_pattern_id}")
                            print(f"   Next time you ask '{last_user_input[:50]}...', I'll remember!")
                        else:
                            print("\n⚠️  Cannot learn - missing context")
                    else:
                        # Regular upvote for existing pattern
                        fragment_store.upvote(last_pattern_id)
                        print(f"\n👍 Upvoted pattern: {last_pattern_id}")
                        print(f"   Pattern reinforced - I'll be more confident using it!")
                else:
                    print("\n⚠️  No recent response to upvote")
                print()
                continue
            
            elif command == '-':
                if last_pattern_id:
                    fragment_store.downvote(last_pattern_id)
                    print(f"\n👎 Downvoted pattern: {last_pattern_id}")
                    print(f"   Pattern weakened - I'll use it less often")
                else:
                    print("\n⚠️  No recent response to downvote")
                print()
                continue
            
            elif command == '?':
                if last_pattern_id:
                    print(f"\n📋 Last pattern ID: {last_pattern_id}")
                else:
                    print("\n⚠️  No recent response")
                print()
                continue
            
            else:
                print(f"\n⚠️  Unknown command: /{command}")
                print("   Type '/help' for available commands")
                print()
                continue
        
        # Check for automatic feedback signals BEFORE processing current input
        if feedback_tracker and last_user_input and last_response_text:
            feedback_result = feedback_tracker.check_feedback(user_input)
            if feedback_result:
                result, pattern_id = feedback_result
                if result.should_apply and pattern_id:
                    emoji = feedback_tracker.detector.get_feedback_emoji(result)
                    if result.is_positive:
                        fragment_store.upvote(pattern_id, strength=result.strength)
                        print(f"   {emoji} Auto-feedback: {result.reason}")
                    elif result.is_negative:
                        fragment_store.downvote(pattern_id, strength=result.strength)
                        print(f"   {emoji} Auto-feedback: {result.reason}")
        
        # Generate response
        turn += 1
        response = composer.compose_response(context=user_input, user_input=user_input)
        
        # Track last interaction for learning
        last_user_input = user_input
        last_response_text = response.text
        
        # Track pattern ID if response came from learned patterns
        if response.fragment_ids:
            last_pattern_id = response.fragment_ids[0]  # First (primary) pattern
        else:
            last_pattern_id = None
        
        # Record interaction for feedback tracking
        if feedback_tracker:
            feedback_tracker.record_interaction(
                user_input=user_input,
                response_text=response.text,
                pattern_id=last_pattern_id
            )
        
        # Auto-learn semantic relationships from this conversation
        if auto_learner:
            relations = auto_learner.process_conversation(user_input, response.text)
            if relations:
                # Show what was learned (optional, can remove for cleaner output)
                pass  # Silently learn - uncomment below for verbose mode
                # print(f"   📚 Learned {len(relations)} semantic relationships")
        
        print(f"Lilith: {response.text}")
        
        if response.modality:
            print(f"   [Modality: {response.modality}]")
        
        # Enhanced fallback feedback
        if response.is_fallback:
            if response.is_low_confidence:
                # No knowledge found at all
                print(f"   💡 I don't know about this yet. You can teach me:")
                print(f"      Option 1: Use '/teach' command for direct teaching")
                print(f"      Option 2: Provide the answer, then I'll respond to it,")
                print(f"                and you can upvote MY answer with '/+'")
            else:
                # Wikipedia or external knowledge found
                print(f"   📚 This is from external knowledge - not yet in my learned patterns")
                print(f"      Upvote with '/+' to save this for faster recall next time!")
        
        print()


if __name__ == "__main__":
    main()
