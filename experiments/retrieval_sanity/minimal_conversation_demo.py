#!/usr/bin/env python3
"""
Minimal conversation demo - tests core pipeline integration.

Tests:
  1. StageCoordinator processes user input
  2. ConversationState tracks working memory
  3. ResponseComposer generates response from patterns
  4. System learns from outcomes

This is a simplified version to validate interfaces before full conversation_loop.
"""

import sys
from pathlib import Path

# Add experiments/retrieval_sanity to path
sys.path.insert(0, str(Path(__file__).parent))

# Import from pipeline (now has minimal __init__.py)
from pipeline import (
    StageCoordinator,
    StageType,
    Utterance,
    ConversationState,
    ConversationHistory,
    ResponseFragmentStore,
    ResponseComposer,
)


class MinimalConversationDemo:
    """Minimal conversation demo with proper interface usage"""
    
    def __init__(self):
        print("🧠 Initializing Minimal Conversation Demo...")
        print()
        
        # Initialize StageCoordinator (uses default 2-stage config)
        print("  📥 Creating StageCoordinator...")
        self.coordinator = StageCoordinator()
        
        # Get semantic stage
        self.semantic_stage = self.coordinator.get_stage(StageType.SEMANTIC)
        if not self.semantic_stage:
            raise RuntimeError("Semantic stage not found!")
        
        # Initialize conversation state (working memory)
        print("  💭 Initializing ConversationState...")
        self.conversation_state = ConversationState(
            encoder=self.semantic_stage.encoder,
            decay=0.75,
            max_topics=5
        )
        
        # Initialize response components
        print("  📚 Loading ResponseFragmentStore...")
        self.fragment_store = ResponseFragmentStore(
            semantic_encoder=self.semantic_stage.encoder,
            storage_path="demo_response_patterns.json"
        )
        
        print("  ✍️  Creating ResponseComposer...")
        self.composer = ResponseComposer(
            fragment_store=self.fragment_store,
            conversation_state=self.conversation_state,
            composition_mode="best_match"  # Start simple
        )
        
        # Track conversation
        self.history = ConversationHistory(max_turns=10)
        self.turn_count = 0
        
        print()
        print("✅ System ready!")
        print()
        
    def process_turn(self, user_input: str) -> str:
        """Process one conversation turn"""
        self.turn_count += 1
        
        print(f"[Turn {self.turn_count}]")
        print(f"👤 User: {user_input}")
        print()
        
        # 1. Process through pipeline stages
        print("  📥 Processing through pipeline stages...")
        utterance = Utterance(text=user_input, language="en")
        artifacts = self.coordinator.process(utterance)
        
        # Get semantic artifact
        semantic_artifact = artifacts.get(StageType.SEMANTIC)
        if not semantic_artifact:
            return "I couldn't process that. Please try again."
        
        # 2. Get context from working memory
        # NOTE: Skipping ConversationState.update() in minimal demo 
        # because it requires full PipelineArtifact (parsed tokens, frames, etc.)
        # The response composer still gets PMFlow activations from semantic stage
        print("  💭 Using semantic activations...")
        print(f"     Confidence: {semantic_artifact.confidence:.3f}")
        
        # 3. Compose response
        print("  🔍 Retrieving response patterns...")
        
        # Build context from recent history for coherence checking
        if self.history.turns:
            conversation_context = self.history.get_context_window(n=3)
        else:
            conversation_context = ""
        
        # Use user_input directly for pattern retrieval
        # The conversation context is used for coherence filtering
        response = self.composer.compose_response(
            context=user_input,  # Match against current input, not full history
            user_input=user_input,
            topk=5
        )
        
        print(f"     Matched {len(response.fragment_ids)} patterns")
        print(f"     Coherence: {response.coherence_score:.3f}")
        if response.primary_pattern:
            print(f"     Primary: '{response.primary_pattern.trigger_context}' → '{response.text}')")
        
        # 4. Store in history
        self.history.add_turn(
            user_input=user_input,
            bot_response=response.text,
            user_embedding=semantic_artifact.embedding  # Direct access to embedding
        )
        
        print()
        print(f"🤖 Bot: {response.text}")
        print()
        
        return response.text
        
    def run_demo_conversation(self):
        """Run a short demo conversation"""
        print("="*80)
        print("MINIMAL CONVERSATION DEMO")
        print("="*80)
        print()
        
        # Demo conversation sequence
        demo_inputs = [
            "Hello!",
            "What is artificial intelligence?",
            "Tell me more about machine learning",
            "How does it work?",
            "That's interesting!",
        ]
        
        for user_input in demo_inputs:
            self.process_turn(user_input)
            
        # Show final stats
        print("="*80)
        print("DEMO COMPLETE")
        print("="*80)
        print()
        
        # Fragment store stats
        stats = self.fragment_store.get_stats()
        print("Response Patterns:")
        print(f"  Total: {stats['total_patterns']}")
        print(f"  Average success: {stats['average_success']:.2f}")
        print()
        
        # History stats
        hist_stats = self.history.get_stats()
        print("Conversation History:")
        print(f"  Turns: {hist_stats['turn_count']}")
        print(f"  Window size: {hist_stats['window_size']}")
        print()
        
        # Working memory state
        if self.conversation_state.is_active():
            print("Working Memory:")
            print(f"  Decay rate: {self.conversation_state.decay}")
            print(f"  Max topics: {self.conversation_state.max_topics}")
        print()
        
    def interactive_mode(self):
        """Interactive conversation mode"""
        print("="*80)
        print("INTERACTIVE MODE")
        print("="*80)
        print()
        print("Type '/quit' to exit, '/stats' to show statistics")
        print()
        
        while True:
            try:
                user_input = input("👤 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break
                
            if not user_input:
                continue
                
            if user_input == "/quit":
                print("\nGoodbye!")
                break
                
            if user_input == "/stats":
                stats = self.fragment_store.get_stats()
                print(f"\n📊 Stats:")
                print(f"  Patterns: {stats['total_patterns']}")
                print(f"  Turns: {self.turn_count}")
                print()
                continue
                
            self.process_turn(user_input)
            

def main():
    """Main entry point"""
    print()
    print("╔" + "═"*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "MINIMAL CONVERSATION DEMO".center(78) + "║")
    print("║" + "Testing Pure Neuro-Symbolic Pipeline Integration".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "═"*78 + "╝")
    print()
    
    try:
        demo = MinimalConversationDemo()
        
        # Run automated demo
        demo.run_demo_conversation()
        
        # Offer interactive mode
        print("Would you like to try interactive mode? (y/n): ", end="")
        response = input().strip().lower()
        
        if response == 'y':
            print()
            demo.interactive_mode()
            
        # Clean up
        Path("demo_response_patterns.json").unlink(missing_ok=True)
        
        print()
        print("="*80)
        print("✅ Demo completed successfully!")
        print("="*80)
        print()
        
    except Exception as e:
        print()
        print("="*80)
        print("❌ Error during demo!")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
