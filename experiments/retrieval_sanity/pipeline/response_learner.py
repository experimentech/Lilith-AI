"""
Response Learner - Learning from Conversation Outcomes

Observes conversation outcomes and updates response patterns through plasticity.
Evaluates success signals from conversation state (topic strength, novelty, etc.).

This is how the system LEARNS to communicate - no programming, pure adaptation!
"""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

from .response_composer import ResponseComposer, ComposedResponse
from .response_fragments import ResponseFragmentStore
from .conversation_state import ConversationState
from .plasticity import PlasticityController


@dataclass
class OutcomeSignals:
    """Conversation outcome signals for learning"""
    topic_maintained: bool      # Did topic continue?
    novelty_score: float        # New information provided?
    engagement_score: float     # User engaged or confused?
    coherence_score: float      # Response fit conversation?
    overall_success: float      # Combined success (-1.0 to 1.0)
    

class ResponseLearner:
    """
    Learns successful response patterns through interaction.
    
    Observes conversation outcomes and applies plasticity updates
    to improve future responses.
    
    Key insight: Use the SAME learning mechanisms as semantic understanding!
    """
    
    def __init__(
        self,
        composer: ResponseComposer,
        fragment_store: ResponseFragmentStore,
        plasticity_controller: Optional[PlasticityController] = None,
        learning_rate: float = 0.1
    ):
        """
        Initialize response learner.
        
        Args:
            composer: Response composer to learn from
            fragment_store: Store to update with learned patterns
            plasticity_controller: Optional plasticity controller for embeddings
            learning_rate: How quickly to adapt (0.0-1.0)
        """
        self.composer = composer
        self.fragments = fragment_store
        self.plasticity = plasticity_controller
        self.learning_rate = learning_rate
        
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
        
        Args:
            response: Response that was generated
            previous_state: State before response
            current_state: State after user's reaction
            user_input: User's next input (reaction)
            
        Returns:
            Outcome signals that were observed
        """
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
        
        When user says something interesting/engaging, we can:
        1. Use their phrasing as a response pattern
        2. Associate it with the context (our previous response)
        
        Args:
            signals: Outcome signals from interaction
            user_input: What the user said
            bot_response: What we said that prompted their response
        """
        # Only extract from highly positive interactions
        if signals.overall_success < 0.4:
            return
            
        # Engagement score must be high (user was interested/elaborating)
        if signals.engagement_score < 0.7:
            return
            
        # Don't extract very short or very long inputs
        word_count = len(user_input.split())
        if word_count < 3 or word_count > 20:
            return
            
        # Don't extract questions (those aren't good responses)
        if user_input.strip().endswith('?'):
            return
            
        # Extract pattern: use bot's previous response as trigger context
        # and user's engaging reply as the new response
        self.fragments.add_pattern(
            trigger_context=bot_response,
            response_text=user_input,
            success_score=0.7,  # Start higher since it was engaging
            intent="learned"
        )
        
        print(f"  🎓 Learned new pattern: '{user_input[:50]}...'")
            
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
