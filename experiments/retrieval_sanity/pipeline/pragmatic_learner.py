"""
Pragmatic Layer Learner - Specialized GeneralPurposeLearner for Response/Conversation

This is GeneralPurposeLearner specialized for the PRAGMATIC/RESPONSE layer.
Shows how to adapt the universal learning algorithm to a specific cognitive level.

Learns:
- Conversational response patterns
- Factual information recall
- Dialogue flow
"""

from typing import Optional, Dict, Any, Tuple
import time

from .general_purpose_learner import (
    GeneralPurposeLearner,
    OutcomeSignals,
    LayerSpecificEvaluator,
    LayerSpecificExtractor
)
from .response_fragments import ResponseFragmentStore
from .conversation_state import ConversationState
from .response_composer import ComposedResponse


class PragmaticEvaluator(LayerSpecificEvaluator):
    """
    Evaluates success of conversational responses.
    
    Success signals:
    - Topic maintenance (did conversation stay on track?)
    - Engagement (was user interested/elaborating?)
    - Novelty (did user provide new information?)
    - Coherence (did response fit context?)
    """
    
    def evaluate(
        self,
        layer_input: Any,
        layer_output: Any,
        context: Optional[Dict[str, Any]]
    ) -> OutcomeSignals:
        """
        Evaluate conversational response quality.
        
        Args:
            layer_input: User's input text
            layer_output: ComposedResponse from system
            context: Dict with previous_state, current_state
        """
        # Extract context
        previous_state = context.get('previous_state') if context else None
        current_state = context.get('current_state') if context else None
        user_input = layer_input if isinstance(layer_input, str) else ""
        
        if not previous_state or not current_state:
            # Can't evaluate without state - return neutral
            return OutcomeSignals(
                layer_name="pragmatic",
                overall_success=0.0,
                confidence=0.5
            )
        
        prev_snapshot = previous_state.snapshot()
        curr_snapshot = current_state.snapshot()
        
        # 1. Topic Maintenance
        topic_maintained = self._check_topic_maintenance(prev_snapshot, curr_snapshot)
        
        # 2. Engagement Score
        engagement_score = self._evaluate_engagement(user_input, curr_snapshot)
        
        # 3. Novelty Score
        novelty_score = curr_snapshot.novelty
        
        # 4. Coherence Score (from response composition)
        coherence_score = layer_output.coherence_score if hasattr(layer_output, 'coherence_score') else 0.5
        
        # Calculate overall success
        overall_success = self._calculate_overall_success(
            topic_maintained,
            novelty_score,
            engagement_score,
            coherence_score
        )
        
        # TEACHING DETECTION: Boost confidence for factual statements after fallback
        bot_response = context.get('bot_response', '') if context else ''
        bot_used_fallback = bot_response and any(marker in bot_response.lower() for marker in [
            "don't have", "not sure", "don't know", "rephrase", "something else", "not quite sure"
        ])
        
        factual_markers = ['is', 'are', 'was', 'were', 'have', 'has', 'contain', 'include']
        is_factual = user_input and any(marker in user_input.lower().split() for marker in factual_markers)
        
        # If user provides factual info after fallback, this is high-confidence teaching
        is_teaching = bot_used_fallback and is_factual and len(user_input.split()) > 10
        final_confidence = 0.85 if is_teaching else engagement_score
        
        if is_teaching:
            print(f"  🎓 TEACHING DETECTED: confidence boosted to {final_confidence:.2f}")
        
        return OutcomeSignals(
            layer_name="pragmatic",
            overall_success=overall_success,
            confidence=final_confidence,
            layer_signals={
                "topic_maintained": 1.0 if topic_maintained else 0.0,
                "novelty": novelty_score,
                "engagement": engagement_score,
                "coherence": coherence_score
            }
        )
    
    def _check_topic_maintenance(self, prev_snapshot, curr_snapshot) -> bool:
        """Check if conversation topic was maintained."""
        if not prev_snapshot.topics or not curr_snapshot.topics:
            return True  # No topics to compare
        
        # Check if any previous topics are still active
        prev_topics = {t.signature for t in prev_snapshot.topics}
        curr_topics = {t.signature for t in curr_snapshot.topics}
        
        if not prev_topics:
            return True
        
        # Calculate overlap
        overlap = len(prev_topics & curr_topics) / len(prev_topics)
        
        # Also check if topic strength didn't drop too much
        if prev_snapshot.topics and curr_snapshot.topics:
            prev_strength = max(t.strength for t in prev_snapshot.topics)
            curr_strength = max(t.strength for t in curr_snapshot.topics)
            strength_ratio = curr_strength / (prev_strength + 1e-6)
        else:
            strength_ratio = 1.0
        
        # Topic maintained if strength didn't drop too much
        return strength_ratio > 0.5
    
    def _evaluate_engagement(self, user_input: str, curr_snapshot) -> float:
        """
        Evaluate user engagement from input.
        
        Positive signals:
        - Longer input (elaborating)
        - Questions (interested)
        - Engagement markers ("interesting", "tell me more")
        
        Negative signals:
        - Very short (disengaged)
        - Confusion markers ("what?", "huh?")
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
        
        import numpy as np
        return np.clip(engagement, 0.0, 1.0)
    
    def _calculate_overall_success(
        self,
        topic_maintained: bool,
        novelty_score: float,
        engagement_score: float,
        coherence_score: float
    ) -> float:
        """Calculate overall success score from signals."""
        import numpy as np
        
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


class PragmaticExtractor(LayerSpecificExtractor):
    """
    Extracts conversational patterns from successful interactions.
    
    Two types of patterns:
    1. Factual: User states fact → Store for recall
    2. Conversational: Bot says X → User responds Y
    """
    
    def extract_pattern(
        self,
        layer_input: Any,
        layer_output: Any,
        signals: OutcomeSignals,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Tuple[str, str, str, float]]:
        """
        Extract conversational pattern from interaction.
        
        Returns:
            (trigger, response, intent, initial_success) or None
        """
        user_input = layer_input if isinstance(layer_input, str) else ""
        bot_response = context.get('bot_response', "") if context else ""
        
        # Basic filters
        word_count = len(user_input.split())
        is_question = user_input.strip().endswith('?')
        
        # TEACHING DETECTION: Check if this is a teaching scenario
        # Use metadata instead of string matching for reliability
        bot_used_fallback = context.get('is_fallback', False) if context else False
        
        factual_markers = ['is', 'are', 'was', 'were', 'have', 'has', 'contain', 'include']
        user_lower = user_input.lower()
        is_factual = any(marker in user_lower.split() for marker in factual_markers)
        is_teaching = bot_used_fallback and is_factual
        
        # Don't extract very short inputs
        if word_count < 3:
            return None
        
        # Don't extract very long inputs UNLESS it's teaching
        if word_count > 20 and not is_teaching:
            return None
        
        # Allow longer teaching statements (up to 50 words)
        if is_teaching and word_count > 50:
            return None
        
        # Don't extract questions (those aren't good responses)
        if is_question:
            return None
        
        # Check if user input looks like a factual statement
        
        if is_factual:
            # FACTUAL LEARNING: Store user's statement as potential response
            # Extract key topic from statement for trigger
            # For "X is Y" or "X are Y" patterns, extract just X as the subject/topic
            
            # Find the position of factual verbs (is, are, was, were, etc.)
            words = user_input.split()
            factual_verb_pos = -1
            for i, word in enumerate(words):
                if word.lower() in factual_markers:
                    factual_verb_pos = i
                    break
            
            if factual_verb_pos > 0:
                # Extract everything BEFORE the verb as the topic/subject
                # E.g., "Episodic memory is..." -> "Episodic memory"
                # E.g., "The capital of Iceland is..." -> "The capital of Iceland"
                topic = ' '.join(words[:factual_verb_pos])
            elif len(words) >= 3:
                # Fallback to first 1-3 words if no verb found
                topic = ' '.join(words[:min(3, len(words))])
            else:
                topic = words[0] if words else "general"
            
            # TEACHING BOOST: If this follows a fallback, it's high-value teaching
            # Give it MUCH higher initial confidence to prevent override by weak patterns
            # Taught patterns are gold standard - protect them!
            initial_confidence = 0.90 if bot_used_fallback else 0.7
            
            if bot_used_fallback:
                print(f"  🎓 TEACHING DETECTED: Topic='{topic}', Teaching='{user_input[:60]}...'")
            
            return (topic, user_input, "taught" if bot_used_fallback else "learned", initial_confidence)
        else:
            # CONVERSATIONAL LEARNING: Store as dialogue flow pattern
            # But NOT if bot used fallback - that would learn wrong direction
            if bot_response and not bot_used_fallback:
                return (bot_response, user_input, "learned", 0.7)
            else:
                return None


class PragmaticLearner(GeneralPurposeLearner):
    """
    Learner for pragmatic/conversational layer.
    
    This is the specialized version of GeneralPurposeLearner for responses.
    Shows the pattern for how to adapt the universal algorithm.
    """
    
    def __init__(
        self,
        pattern_store: ResponseFragmentStore,
        learning_rate: float = 0.1,
        learning_mode: str = "conservative"
    ):
        super().__init__(
            layer_name="pragmatic",
            pattern_store=pattern_store,
            learning_rate=learning_rate,
            learning_mode=learning_mode
        )
        
        # Layer-specific helpers
        self.evaluator = PragmaticEvaluator()
        self.extractor = PragmaticExtractor()
    
    def _evaluate_outcome(
        self,
        layer_input: Any,
        layer_output: Any,
        context: Optional[Dict[str, Any]]
    ) -> OutcomeSignals:
        """Use pragmatic-specific evaluator."""
        return self.evaluator.evaluate(layer_input, layer_output, context)
    
    def _extract_and_store_pattern(
        self,
        layer_input: Any,
        layer_output: Any,
        signals: OutcomeSignals,
        context: Optional[Dict[str, Any]]
    ):
        """Use pragmatic-specific extractor."""
        pattern_data = self.extractor.extract_pattern(
            layer_input, layer_output, signals, context
        )
        
        if pattern_data is None:
            print(f"  ⚠️  Pattern extractor returned None (filtered out)")
            return
        
        trigger, response, intent, initial_success = pattern_data
        
        print(f"  ✅ Pattern extracted: Trigger='{trigger[:40]}...', Intent={intent}")
        
        # Generate unique ID
        fragment_id = f"learned_{intent}_{int(time.time() * 1000)}"
        
        # Store pattern
        self.pattern_store.add_pattern(
            fragment_id=fragment_id,
            trigger_context=trigger,
            response_text=response,
            intent=intent,
            success_score=initial_success
        )
        
        self.patterns_learned += 1
        
        # Log learning
        if self.learning_mode in ["moderate", "eager"]:
            if "learned" in intent:
                print(f"  ✅ 🎓 Learned pattern: '{response[:40]}...' (trigger: '{trigger[:30]}')")
