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
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pipeline.stage_coordinator import StageCoordinator, StageType
from pipeline.base import Utterance, PipelineArtifact, ParsedSentence, SymbolicFrame, Token
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
        history_window: int = 10,
        composition_mode: str = "best_match",
        use_grammar: bool = False
    ):
        """
        Initialize conversation system.
        
        Args:
            history_window: Number of turns to keep in history
            composition_mode: How to compose responses (best_match, weighted_blend, adaptive)
            use_grammar: Enable grammar-guided composition (experimental)
        """
        print("🧠 Initializing Pure Neuro-Symbolic Conversation System...")
        print()
        
        # Initialize multi-stage coordinator
        print("  📥 Loading intake stage (noise normalization)...")
        print("  🧩 Loading semantic stage (concept understanding)...")
        self.coordinator = StageCoordinator()  # Uses default 2-stage config
        
        # Get semantic stage for encoder
        self.semantic_stage = self.coordinator.get_stage(StageType.SEMANTIC)
        if not self.semantic_stage:
            raise RuntimeError("Semantic stage not found!")
        
        # Initialize conversation state (working memory)
        print("  💭 Initializing working memory (PMFlow activations)...")
        self.conversation_state = ConversationState(
            encoder=self.semantic_stage.encoder,
            decay=0.75,        # Gradual forgetting
            max_topics=5       # Bounded capacity
        )
        
        # Initialize response generation components
        print("  📚 Loading response fragment store (learned patterns)...")
        self.fragment_store = ResponseFragmentStore(
            semantic_encoder=self.semantic_stage.encoder,
            storage_path="conversation_patterns.json"
        )
        
        print("  ✍️  Initializing response composer (PMFlow-guided)...")
        self.composer = ResponseComposer(
            fragment_store=self.fragment_store,
            conversation_state=self.conversation_state,
            composition_mode=composition_mode,
            use_grammar=use_grammar
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
        print(f"  Stages: {len(self.coordinator.stages)}")
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
    
    def _build_pipeline_artifact(self, utterance: Utterance, artifacts: dict) -> PipelineArtifact:
        """Build a minimal PipelineArtifact from StageArtifacts for working memory.
        
        ConversationState needs PipelineArtifact with parsed/frame fields,
        but StageCoordinator produces StageArtifacts. This bridges the gap.
        """
        semantic_artifact = artifacts.get(StageType.SEMANTIC)
        if not semantic_artifact or not semantic_artifact.tokens:
            # Fallback: create minimal artifact from raw text
            token_strings = utterance.text.split()
        else:
            # StageArtifact.tokens are strings, not Token objects
            token_strings = semantic_artifact.tokens
        
        # Convert to Token objects
        tokens = [Token(text=word, position=i) for i, word in enumerate(token_strings)]
        
        # Build minimal parsed sentence
        parsed = ParsedSentence(tokens=tokens, confidence=semantic_artifact.confidence if semantic_artifact else 0.0)
        
        # Build minimal symbolic frame from tokens
        # Extract simple actor/action/target heuristics
        token_texts = [t.text.lower() for t in tokens]
        frame = SymbolicFrame(
            actor=token_texts[0] if len(token_texts) > 0 else None,
            action=token_texts[1] if len(token_texts) > 1 else None,
            target=token_texts[2] if len(token_texts) > 2 else None,
            modifiers={},
            attributes={},
            confidence=semantic_artifact.confidence if semantic_artifact else 0.5,
            raw_text=utterance.text,
            language=utterance.language,
        )
        
        return PipelineArtifact(
            utterance=utterance,
            normalised_text=utterance.text.lower().strip(),
            candidates=[utterance.text],
            parsed=parsed,
            frame=frame,
            embedding=semantic_artifact.embedding if semantic_artifact else torch.zeros(96),
            confidence=semantic_artifact.confidence if semantic_artifact else 0.5,
        )
        
    def process_user_input(self, user_input: str) -> str:
        """
        Process user input through full pipeline.
        
        Args:
            user_input: User's text input
            
        Returns:
            Bot's response text
        """
        # Save snapshot before processing (for learning)
        previous_snapshot = self.conversation_state.snapshot()
        
        # 1. INPUT PIPELINE: Text → Stages → Understanding
        print("  📥 Processing input through pipeline stages...")
        utterance = Utterance(text=user_input)
        artifacts = self.coordinator.process(utterance)
        
        # Get semantic artifact
        semantic_artifact = artifacts.get(StageType.SEMANTIC)
        if not semantic_artifact:
            return "I'm having trouble processing that. Could you try again?"
        
        # 2. Update working memory from input
        print("  💭 Updating working memory...")
        pipeline_artifact = self._build_pipeline_artifact(utterance, artifacts)
        self.conversation_state.update(pipeline_artifact)
        
        # 3. OUTPUT PIPELINE: Context → Retrieve patterns → Compose response
        print("  🔍 Retrieving response patterns...")
        
        # Use user input directly for pattern retrieval (lesson from minimal demo)
        response = self.composer.compose_response(
            context=user_input,  # Match against current input, not full history
            user_input=user_input,
            topk=5
        )
        
        # Debug: Show what pattern was selected
        if response.primary_pattern:
            print(f"     → Selected: '{response.primary_pattern.response_text[:50]}...' (intent: {response.primary_pattern.intent}, score: {response.coherence_score:.3f})")
        
        print("  ✍️  Composing response...")

        
        # 4. Store in history
        self.history.add_turn(
            user_input=user_input,
            bot_response=response.text,
            user_embedding=semantic_artifact.embedding
        )
        
        # 5. Store response for potential learning
        self.last_response = response
        self.last_snapshot = previous_snapshot
        
        # 6. If we have a previous response, learn from this interaction
        if hasattr(self, '_previous_response') and self._previous_response is not None:
            # User's current input is their reaction to our previous response
            current_snapshot = self.conversation_state.snapshot()
            self.learner.observe_interaction(
                response=self._previous_response,
                previous_state=self._previous_state,
                current_state=self.conversation_state,
                user_input=user_input
            )
        
        # Store for next iteration
        self._previous_response = response
        self._previous_state = self.conversation_state
        
        return response.text
        
    def observe_outcome(self, user_reaction: str):
        """
        Observe user's reaction and apply learning.
        
        Args:
            user_reaction: User's next input (reaction to response)
        """
        if len(self.history.turns) < 2:
            return  # Need at least 2 turns to learn
        
        # Simple heuristic learning for now
        # TODO: Use full ResponseLearner.observe_interaction when we have proper state tracking
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
                snapshot = self.conversation_state.snapshot()
                print()
                print(f"💭 [Working memory has {len(snapshot.topics)} active topics]")
                

def main():
    """Main entry point"""
    # Create conversation loop
    loop = ConversationLoop(
        history_window=10,
        composition_mode="best_match"  # Start simple
    )
    
    # Run interactive loop
    loop.run()
    

if __name__ == "__main__":
    main()
