#!/usr/bin/env python3
"""
Conversation Loop - Interactive Pure Neuro-Symbolic Dialogue

Full pipeline demonstration:
  Input → Intake → Semantic → Response Composition → Output
  
Shows:
  - Symmetric input/output processing
  - Working memory decay over time
  - Plasticity learning from interaction
  - PMFlow-guided response composition
  
NO LLM - pure learned behavior!
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pipeline.stage_coordinator import StageCoordinator
from pipeline.response_fragments import ResponseFragmentStore
from pipeline.response_composer import ResponseComposer
from pipeline.response_learner import ResponseLearner
from pipeline.conversation_history import ConversationHistory
from pipeline.conversation_state import ConversationState


class ConversationLoop:
    """
    Interactive conversation loop with pure neuro-symbolic processing.
    
    Demonstrates:
    1. Symmetric input/output using same PMFlow architecture
    2. Working memory with automatic decay
    3. Learning from conversation outcomes
    4. Response composition from learned patterns
    """
    
    def __init__(
        self,
        intake_dim: int = 48,
        semantic_dim: int = 96,
        history_window: int = 10,
        composition_mode: str = "weighted_blend"
    ):
        """
        Initialize conversation system.
        
        Args:
            intake_dim: Intake stage embedding dimension
            semantic_dim: Semantic stage embedding dimension
            history_window: Number of turns to keep in history
            composition_mode: How to compose responses
        """
        print("🧠 Initializing Pure Neuro-Symbolic Conversation System...")
        print()
        
        # Initialize multi-stage coordinator
        print("  📥 Loading intake stage (noise normalization)...")
        print("  🧩 Loading semantic stage (concept understanding)...")
        self.coordinator = StageCoordinator(
            intake_dim=intake_dim,
            semantic_dim=semantic_dim
        )
        
        # Initialize conversation state (working memory)
        print("  💭 Initializing working memory (PMFlow activations)...")
        self.conversation_state = ConversationState(
            encoder=self.coordinator.semantic,
            decay=0.75,        # Gradual forgetting
            max_topics=5       # Bounded capacity
        )
        
        # Initialize response generation components
        print("  📚 Loading response fragment store (learned patterns)...")
        self.fragment_store = ResponseFragmentStore(
            semantic_encoder=self.coordinator.semantic
        )
        
        print("  ✍️  Initializing response composer (PMFlow-guided)...")
        self.composer = ResponseComposer(
            fragment_store=self.fragment_store,
            conversation_state=self.conversation_state,
            composition_mode=composition_mode
        )
        
        print("  🎓 Initializing response learner (plasticity updates)...")
        self.learner = ResponseLearner(
            composer=self.composer,
            fragment_store=self.fragment_store,
            learning_rate=0.1
        )
        
        # Initialize conversation history
        print("  📜 Initializing conversation history (sliding window)...")
        self.history = ConversationHistory(max_turns=history_window)
        
        print()
        print("✅ System initialized!")
        print()
        self._print_system_info()
        
    def _print_system_info(self):
        """Print system configuration"""
        print("═" * 80)
        print("SYSTEM CONFIGURATION")
        print("═" * 80)
        print(f"  Architecture: Pure Neuro-Symbolic (no LLM)")
        print(f"  Intake dimension: {self.coordinator.intake.dim}")
        print(f"  Semantic dimension: {self.coordinator.semantic.dim}")
        print(f"  Working memory decay: {self.conversation_state.decay}")
        print(f"  Max topics: {self.conversation_state.max_topics}")
        print(f"  History window: {self.history.max_turns} turns")
        print(f"  Composition mode: {self.composer.composition_mode}")
        print(f"  Learning rate: {self.learner.learning_rate}")
        print()
        
        # Show fragment store stats
        stats = self.fragment_store.get_stats()
        print(f"  Response patterns: {stats['total_patterns']}")
        print(f"    - Seed patterns: {stats['seed_patterns']}")
        print(f"    - Learned patterns: {stats['learned_patterns']}")
        print(f"    - Average success: {stats['average_success']:.2f}")
        print("═" * 80)
        print()
        
    def process_user_input(self, user_input: str) -> str:
        """
        Process user input through full pipeline.
        
        Args:
            user_input: User's text input
            
        Returns:
            Bot's response text
        """
        # Store previous state for learning
        previous_state = ConversationState(
            encoder=self.coordinator.semantic,
            decay=self.conversation_state.decay,
            max_topics=self.conversation_state.max_topics
        )
        # Copy current state
        if hasattr(self.conversation_state, 'topics'):
            previous_state.topics = self.conversation_state.topics.copy()
            previous_state.last_activation = self.conversation_state.last_activation
        
        # 1. INPUT PIPELINE: Text → Intake → Semantic → Understanding
        print("  📥 Processing input through intake stage...")
        artifact = self.coordinator.encode_with_context(user_input)
        
        # 2. Update working memory from input
        print("  💭 Updating working memory...")
        self.conversation_state.update(artifact)
        
        # 3. OUTPUT PIPELINE: Context → Retrieve patterns → Compose response
        print("  🔍 Retrieving response patterns...")
        print("  ✍️  Composing response...")
        
        # Get context from recent history
        context = self.history.get_context_window(n=3)
        if not context or context == "(no conversation history)":
            context = user_input
        else:
            context = context + f"\nUser: {user_input}"
        
        # Compose response
        response = self.composer.compose_response(
            context=context,
            user_input=user_input
        )
        
        # 4. LEARNING: Observe outcome and update
        # (We'll update after next user input, when we see the reaction)
        
        # 5. Store in history
        user_emb = artifact.semantic_embedding
        self.history.add_turn(
            user_input=user_input,
            bot_response=response.text,
            user_embedding=user_emb,
            working_memory_state=self.conversation_state.snapshot().__dict__
        )
        
        return response.text
        
    def observe_outcome(self, user_reaction: str):
        """
        Observe user's reaction and apply learning.
        
        Args:
            user_reaction: User's next input (reaction to response)
        """
        if len(self.history.turns) < 2:
            return  # Need at least 2 turns to learn
            
        # Get last turn
        last_turn = self.history.turns[-2]  # Previous turn
        
        # Recreate response object
        # (In production, we'd store this)
        from pipeline.response_composer import ComposedResponse
        response = ComposedResponse(
            text=last_turn.bot_response,
            fragment_ids=["unknown"],  # Would need to store this
            composition_weights=[1.0],
            coherence_score=0.5,
            primary_pattern=None
        )
        
        # Previous state (from stored snapshot)
        # Current state (current working memory)
        
        # Observe and learn
        # signals = self.learner.observe_interaction(
        #     response=response,
        #     previous_state=...,
        #     current_state=self.conversation_state,
        #     user_input=user_reaction
        # )
        
        # For now, simple success estimation
        is_repetition = self.history.detect_repetition(user_reaction, window=2)
        if is_repetition:
            success = -0.5  # User is confused/repeating
        else:
            success = 0.3  # Assume positive if continuing
            
        self.history.update_last_success(success)
        
    def display_state(self):
        """Display current system state"""
        print()
        print("─" * 80)
        print("SYSTEM STATE")
        print("─" * 80)
        
        # Working memory
        snapshot = self.conversation_state.snapshot()
        print(f"💭 Working Memory:")
        print(f"  Active: {snapshot.active}")
        print(f"  Activation energy: {snapshot.activation_energy:.3f}")
        print(f"  Novelty: {snapshot.novelty:.3f}")
        print(f"  Topics ({len(snapshot.topics)}):")
        for i, topic in enumerate(snapshot.topics, 1):
            print(f"    {i}. Strength: {topic.strength:.3f}, Mentions: {topic.mentions}")
            print(f"       Summary: {topic.summary[:50]}...")
            
        # Conversation history
        hist_stats = self.history.get_stats()
        print()
        print(f"📜 Conversation History:")
        print(f"  Turns: {hist_stats['turn_count']}")
        print(f"  Average success: {hist_stats['average_success']:.2f}")
        print(f"  Has repetition: {hist_stats['has_repetition']}")
        
        # Learning progress
        learn_stats = self.learner.get_learning_stats()
        print()
        print(f"🎓 Learning Progress:")
        print(f"  Interactions: {learn_stats['interaction_count']}")
        print(f"  Average success: {learn_stats['average_success']:.2f}")
        print(f"  Recent success: {learn_stats['recent_success']:.2f}")
        print(f"  Learning trend: {learn_stats['learning_trend']:+.2f}")
        
        print("─" * 80)
        print()
        
    def run(self):
        """Run interactive conversation loop"""
        print()
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "  PURE NEURO-SYMBOLIC CONVERSATION SYSTEM".center(78) + "║")
        print("║" + "  No LLM - Pure Learned Behavior!".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "═" * 78 + "╝")
        print()
        print("Commands:")
        print("  /state  - Show system state (memory, topics, learning)")
        print("  /stats  - Show detailed statistics")
        print("  /reset  - Reset conversation")
        print("  /quit   - Exit")
        print()
        print("Type your message and press Enter to chat!")
        print("=" * 80)
        print()
        
        turn_count = 0
        
        while True:
            # Get user input
            try:
                user_input = input("\n👤 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break
                
            if not user_input:
                continue
                
            # Handle commands
            if user_input.startswith("/"):
                if user_input == "/quit":
                    print("\nGoodbye!")
                    break
                elif user_input == "/state":
                    self.display_state()
                    continue
                elif user_input == "/stats":
                    self.display_state()
                    continue
                elif user_input == "/reset":
                    self.history.clear()
                    print("\n✅ Conversation reset!")
                    continue
                else:
                    print(f"\n❌ Unknown command: {user_input}")
                    continue
                    
            # Process input
            turn_count += 1
            print()
            print(f"[Turn {turn_count}]")
            
            # If not first turn, learn from previous interaction
            if turn_count > 1:
                self.observe_outcome(user_input)
            
            # Generate response
            response_text = self.process_user_input(user_input)
            
            print()
            print(f"🤖 Bot: {response_text}")
            
            # Show brief state every 5 turns
            if turn_count % 5 == 0:
                print()
                print(f"💭 [Working memory has {len(self.conversation_state.topics)} active topics]")
                

def main():
    """Main entry point"""
    # Create conversation loop
    loop = ConversationLoop(
        intake_dim=48,
        semantic_dim=96,
        history_window=10,
        composition_mode="best_match"  # Start simple
    )
    
    # Run interactive loop
    loop.run()
    

if __name__ == "__main__":
    main()
