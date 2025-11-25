"""
Response Learner - Learning from Conversation Outcomes

REFACTORED: Now uses GeneralPurposeLearner architecture.
This module provides backward-compatible wrapper around PragmaticLearner.

The actual learning logic is in:
- general_purpose_learner.py: Universal learning algorithm
- pragmatic_learner.py: Specialized for conversational responses

This file maintains the old interface for compatibility.
"""

from dataclasses import dataclass
from typing import Optional, Dict, TYPE_CHECKING
import numpy as np

from .response_composer import ResponseComposer, ComposedResponse
from .response_fragments import ResponseFragmentStore
from .conversation_state import ConversationState
from .plasticity import PlasticityController

# Import new architecture (always available in this codebase)
from .pragmatic_learner import PragmaticLearner
from .general_purpose_learner import OutcomeSignals as GPLOutcomeSignals

# Feature flag for gradual migration
NEW_ARCHITECTURE_AVAILABLE = True


@dataclass
class OutcomeSignals:
    """Conversation outcome signals for learning (legacy interface)"""
    topic_maintained: bool      # Did topic continue?
    novelty_score: float        # New information provided?
    engagement_score: float     # User engaged or confused?
    coherence_score: float      # Response fit conversation?
    overall_success: float      # Combined success (-1.0 to 1.0)
    

class ResponseLearner:
    """
    Learns successful response patterns through interaction.
    
    REFACTORED: Now delegates to PragmaticLearner (specialized GeneralPurposeLearner).
    Maintains backward compatibility with existing code.
    
    Key insight: Use the SAME learning mechanisms as semantic understanding!
    """
    
    def __init__(
        self,
        composer: ResponseComposer,
        fragment_store: ResponseFragmentStore,
        plasticity_controller: Optional[PlasticityController] = None,
        learning_rate: float = 0.1,
        learning_mode: str = "conservative"
    ):
        """
        Initialize response learner.
        
        Args:
            composer: Response composer to learn from
            fragment_store: Store to update with learned patterns
            plasticity_controller: Optional plasticity controller for embeddings
            learning_rate: How quickly to adapt (0.0-1.0)
            learning_mode: Learning strictness
                - "conservative": Only learn from highly engaging interactions (default)
                - "moderate": Learn from reasonably positive interactions
                - "eager": Learn from most interactions (for teaching/debugging)
        """
        self.composer = composer
        self.fragments = fragment_store
        self.plasticity = plasticity_controller
        self.learning_rate = learning_rate
        self.learning_mode = learning_mode
        
        # Use new architecture if available
        if NEW_ARCHITECTURE_AVAILABLE:
            self._core_learner = PragmaticLearner(
                pattern_store=fragment_store,
                learning_rate=learning_rate,
                learning_mode=learning_mode
            )
        else:
            self._core_learner = None
        
        # Set thresholds based on learning mode (legacy path)
        if learning_mode == "conservative":
            self.success_threshold = 0.4
            self.engagement_threshold = 0.7
        elif learning_mode == "moderate":
            self.success_threshold = 0.2
            self.engagement_threshold = 0.5
        elif learning_mode == "eager":
            self.success_threshold = 0.0
            self.engagement_threshold = 0.3
        else:
            # Default to conservative
            self.success_threshold = 0.4
            self.engagement_threshold = 0.7
        
        # Track learning progress
        self.interaction_count = 0
        self.success_history = []
        
    def observe_interaction(
        self,
        response: ComposedResponse,
        previous_state: ConversationState,
        current_state: ConversationState,
        user_input: str
    ) -> OutcomeSignals:
        """
        Observe conversation outcome and learn.
        
        Now delegates to PragmaticLearner if available (new architecture),
        otherwise uses legacy implementation.
        
        Args:
            response: Response that was generated
            previous_state: State before response
            current_state: State after user's reaction
            user_input: User's next input (reaction)
            
        Returns:
            Outcome signals that were observed
        """
        # Use new architecture if available
        if self._core_learner is not None:
            # Prepare context for new architecture
            context = {
                'previous_state': previous_state,
                'current_state': current_state,
                'bot_response': response.text,
                'previous_user_input': getattr(previous_state, 'last_user_input', ''),  # For teaching detection
                # Add response metadata for reliable teaching detection
                'is_fallback': response.is_fallback,
                'is_low_confidence': response.is_low_confidence,
                'confidence': response.confidence
            }
            
            # Call general-purpose learner
            gpl_signals = self._core_learner.observe_interaction(
                layer_input=user_input,
                layer_output=response,
                context=context
            )
            
            # Convert to legacy format
            layer_signals = gpl_signals.layer_signals or {}
            signals = OutcomeSignals(
                topic_maintained=layer_signals.get('topic_maintained', 0.0) > 0.5,
                novelty_score=layer_signals.get('novelty', 0.5),
                engagement_score=layer_signals.get('engagement', 0.5),
                coherence_score=layer_signals.get('coherence', 0.5),
                overall_success=gpl_signals.overall_success
            )
            
            # Track progress (sync with core learner)
            self.interaction_count = self._core_learner.interaction_count
            self.success_history = self._core_learner.success_history
            
            return signals
        
        # LEGACY PATH: Original implementation
        # Evaluate conversation outcome
        signals = self._evaluate_outcome(
            response,
            previous_state,
            current_state,
            user_input
        )
        
        # Apply learning updates (pass user input and bot response for pattern extraction)
        self._apply_learning(
            response, 
            signals,
            user_input=user_input,
            bot_response=response.text
        )
        
        # Track progress
        self.interaction_count += 1
        self.success_history.append(signals.overall_success)
        
        return signals
        
    def _evaluate_outcome(
        self,
        response: ComposedResponse,
        previous_state: ConversationState,
        current_state: ConversationState,
        user_input: str
    ) -> OutcomeSignals:
        """
        Evaluate conversation outcome from observable signals.
        
        No LLM judges - pure observable metrics!
        
        Args:
            response: Response that was generated
            previous_state: State before response
            current_state: State after user reaction
            user_input: User's reaction
            
        Returns:
            Outcome signals
        """
        prev_snapshot = previous_state.snapshot()
        curr_snapshot = current_state.snapshot()
        
        # 1. Topic Maintenance
        # Did conversation topic continue or drop?
        topic_maintained = self._check_topic_maintenance(
            prev_snapshot,
            curr_snapshot
        )
        
        # 2. Novelty Score
        # Did user provide new information (good) or repeat (bad)?
        novelty_score = curr_snapshot.novelty
        
        # 3. Engagement Score
        # Is user engaged or confused?
        engagement_score = self._evaluate_engagement(
            user_input,
            curr_snapshot
        )
        
        # 4. Coherence Score
        # Did response fit conversation flow?
        coherence_score = response.coherence_score
        
        # 5. Overall Success
        # Combine signals into overall success metric
        overall_success = self._calculate_overall_success(
            topic_maintained,
            novelty_score,
            engagement_score,
            coherence_score
        )
        
        return OutcomeSignals(
            topic_maintained=topic_maintained,
            novelty_score=novelty_score,
            engagement_score=engagement_score,
            coherence_score=coherence_score,
            overall_success=overall_success
        )
        
    def _check_topic_maintenance(
        self,
        prev_snapshot,
        curr_snapshot
    ) -> bool:
        """
        Check if conversation topic was maintained.
        
        Good sign: Topic strength maintained or increased
        Bad sign: Topic dropped sharply
        """
        if not prev_snapshot.topics or not curr_snapshot.topics:
            return True  # Can't judge without topics
            
        # Get dominant topics
        prev_dominant = max(prev_snapshot.topics, key=lambda t: t.strength)
        curr_dominant = max(curr_snapshot.topics, key=lambda t: t.strength)
        
        # Check if strength maintained
        strength_ratio = curr_dominant.strength / (prev_dominant.strength + 1e-6)
        
        # Topic maintained if strength didn't drop too much
        return strength_ratio > 0.5
        
    def _evaluate_engagement(
        self,
        user_input: str,
        curr_snapshot
    ) -> float:
        """
        Evaluate user engagement from input.
        
        Positive signals:
        - Longer input (elaborating)
        - Questions (interested)
        - New information (engaged)
        
        Negative signals:
        - Very short (disengaged)
        - Repetition (confused)
        - "What?" / "Huh?" (didn't understand)
        """
        engagement = 0.5  # Neutral baseline
        
        # Check for confusion markers
        confusion_words = ["what", "huh", "confused", "understand", "mean"]
        if any(word in user_input.lower() for word in confusion_words):
            engagement -= 0.3
            
        # Check for engagement markers
        engagement_words = ["interesting", "tell me more", "why", "how"]
        if any(word in user_input.lower() for word in engagement_words):
            engagement += 0.3
            
        # Input length suggests elaboration
        word_count = len(user_input.split())
        if word_count > 10:
            engagement += 0.2
        elif word_count < 3:
            engagement -= 0.2
            
        return np.clip(engagement, 0.0, 1.0)
        
    def _calculate_overall_success(
        self,
        topic_maintained: bool,
        novelty_score: float,
        engagement_score: float,
        coherence_score: float
    ) -> float:
        """
        Calculate overall success score from signals.
        
        Returns value from -1.0 (bad) to 1.0 (good)
        """
        # Start neutral
        success = 0.0
        
        # Topic maintenance is important
        if topic_maintained:
            success += 0.3
        else:
            success -= 0.3
            
        # Engagement is critical
        success += (engagement_score - 0.5) * 0.6
        
        # Novelty indicates information flow
        success += (novelty_score - 0.5) * 0.2
        
        # Coherence ensures response fit
        success += (coherence_score - 0.5) * 0.3
        
        return np.clip(success, -1.0, 1.0)
        
    def _apply_learning(
        self,
        response: ComposedResponse,
        signals: OutcomeSignals,
        user_input: str = "",
        bot_response: str = ""
    ):
        """
        Apply plasticity updates based on outcome.
        
        This is where learning happens!
        
        Args:
            response: Response that was generated
            signals: Observed outcome signals
            user_input: User's input that triggered learning
            bot_response: Bot's previous response (context for user input)
        """
        # Update success scores for used fragments
        for fragment_id in response.fragment_ids:
            # Scale feedback by learning rate
            feedback = signals.overall_success * self.learning_rate
            
            # Apply update to fragment store
            self.fragments.update_success(
                fragment_id=fragment_id,
                feedback=feedback,
                plasticity_rate=self.learning_rate
            )
        
        # Extract new patterns from highly engaging user inputs
        self._maybe_extract_pattern(signals, user_input, bot_response)
            
        # If plasticity controller available, update embeddings too
        if self.plasticity is not None and response.primary_pattern:
            # Determine plasticity direction
            if signals.overall_success > 0.2:
                # Positive outcome: reinforce
                recall_score = 1.0 + signals.overall_success
            elif signals.overall_success < -0.2:
                # Negative outcome: weaken
                recall_score = 0.5 + signals.overall_success
            else:
                # Neutral: slight reinforcement
                recall_score = 0.9
                
            # Apply plasticity update
            # Note: This requires coordination with semantic encoder
            # For now, we rely on fragment success scores
            pass
    
    def _maybe_extract_pattern(
        self,
        signals: OutcomeSignals,
        user_input: str,
        bot_response: str
    ):
        """
        Extract new response pattern from engaging user input.
        
        This is how the system learns NEW vocabulary and syntax!
        
        Two learning modes:
        1. Conversation flow: Bot says X → User says Y (mimics dialogue patterns)
        2. Factual recall: User's statement → Echo it back (learns facts)
        
        Args:
            signals: Outcome signals from interaction
            user_input: What the user said
            bot_response: What we said that prompted their response
        """
        # DEBUG: Log why patterns aren't extracted
        word_count = len(user_input.split())
        is_question = user_input.strip().endswith('?')
        
        if self.learning_mode in ["moderate", "eager"]:
            print(f"  🔍 Pattern extraction check ({self.learning_mode} mode):")
            print(f"     Success: {signals.overall_success:.2f} (need > {self.success_threshold})")
            print(f"     Engagement: {signals.engagement_score:.2f} (need > {self.engagement_threshold})")
            print(f"     Word count: {word_count} (need 3-20)")
            print(f"     Is question: {is_question} (must be False)")
        
        # Only extract from positive interactions (threshold depends on learning mode)
        if signals.overall_success < self.success_threshold:
            if self.learning_mode in ["moderate", "eager"]:
                print(f"     ❌ Skipped: Low success")
            return
            
        # Engagement score must be reasonable (threshold depends on learning mode)
        if signals.engagement_score < self.engagement_threshold:
            if self.learning_mode in ["moderate", "eager"]:
                print(f"     ❌ Skipped: Low engagement")
            return
            
        # Don't extract very short or very long inputs
        if word_count < 3 or word_count > 20:
            if self.learning_mode in ["moderate", "eager"]:
                print(f"     ❌ Skipped: Word count out of range")
            return
            
        # Don't extract questions (those aren't good responses)
        if is_question:
            if self.learning_mode in ["moderate", "eager"]:
                print(f"     ❌ Skipped: Is a question")
            return
            
        # Extract pattern with appropriate trigger-response mapping
        import time
        
        # Check if user input looks like a factual statement (contains "is", "are", "have", etc.)
        factual_markers = ['is', 'are', 'was', 'were', 'have', 'has', 'contain', 'include']
        user_lower = user_input.lower()
        is_factual = any(marker in user_lower.split() for marker in factual_markers)
        
        if is_factual:
            # FACTUAL LEARNING: Store user's statement as potential response
            # Extract key topic from statement for trigger
            words = user_input.split()
            # Use first 1-3 words as trigger topic (e.g., "Apples" from "Apples are red")
            if len(words) >= 3:
                topic = ' '.join(words[:min(3, len(words))])
            else:
                topic = words[0] if words else "general"
            
            fragment_id = f"learned_fact_{int(time.time() * 1000)}"
            self.fragments.add_pattern(
                fragment_id=fragment_id,
                trigger_context=topic,  # Short topic as trigger
                response_text=user_input,  # Full statement as response
                intent="learned",
                success_score=0.7
            )
            
            print(f"  ✅ 🎓 Learned fact: '{user_input[:40]}...' (trigger: '{topic}')")
        else:
            # CONVERSATIONAL LEARNING: Store as dialogue flow pattern
            # Bot says X → User might respond with Y
            fragment_id = f"learned_conv_{int(time.time() * 1000)}"
            self.fragments.add_pattern(
                fragment_id=fragment_id,
                trigger_context=bot_response,
                response_text=user_input,
                intent="learned",
                success_score=0.7
            )
            
            print(f"  ✅ 🎓 Learned response: '{user_input[:40]}...'")
            
    def get_learning_stats(self) -> Dict:
        """
        Get learning statistics.
        
        Returns:
            Dict with learning metrics
        """
        if not self.success_history:
            return {
                "interaction_count": 0,
                "average_success": 0.0,
                "recent_success": 0.0,
                "learning_trend": 0.0
            }
            
        # Calculate statistics
        avg_success = np.mean(self.success_history)
        
        # Recent success (last 10 interactions)
        recent = self.success_history[-10:]
        recent_success = np.mean(recent) if recent else 0.0
        
        # Learning trend (are we improving?)
        if len(self.success_history) > 10:
            early = np.mean(self.success_history[:10])
            late = np.mean(self.success_history[-10:])
            learning_trend = late - early
        else:
            learning_trend = 0.0
            
        return {
            "interaction_count": self.interaction_count,
            "average_success": float(avg_success),
            "recent_success": float(recent_success),
            "learning_trend": float(learning_trend)
        }
