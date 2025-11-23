"""
Response Composer - PMFlow-Guided Response Assembly

Composes responses using learned patterns weighted by PMFlow activations.
Maintains coherence using working memory state.

This is the symmetric counterpart to semantic understanding:
  Understanding: Text → Embedding → Retrieve context
  Generation: Context → Retrieve patterns → Compose text

Pure neuro-symbolic - no LLM!
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np

from .response_fragments import ResponseFragmentStore, ResponsePattern
from .conversation_state import ConversationState


@dataclass
class ComposedResponse:
    """A response composed from learned patterns"""
    text: str                           # Final response text
    fragment_ids: List[str]            # Which patterns were used
    composition_weights: List[float]   # How much each pattern contributed
    coherence_score: float             # How well it fits working memory
    primary_pattern: ResponsePattern   # Main pattern used
    

class ResponseComposer:
    """
    Composes responses using PMFlow-guided assembly.
    
    Key insight: Response generation uses the SAME mechanisms
    as semantic understanding - retrieve, weight, compose.
    
    No templates, no LLM - pure learned behavior!
    """
    
    def __init__(
        self, 
        fragment_store: ResponseFragmentStore,
        conversation_state: ConversationState,
        composition_mode: str = "weighted_blend"
    ):
        """
        Initialize response composer.
        
        Args:
            fragment_store: Store of learned response patterns
            conversation_state: Working memory state
            composition_mode: How to compose responses
                - "best_match": Use highest-weighted pattern only
                - "weighted_blend": Blend multiple patterns
                - "adaptive": Choose based on confidence
        """
        self.fragments = fragment_store
        self.state = conversation_state
        self.composition_mode = composition_mode
        
    def compose_response(
        self, 
        context: str,
        user_input: str = "",
        topk: int = 5
    ) -> ComposedResponse:
        """
        Generate response through learned composition.
        
        Args:
            context: Current conversation context (from semantic stage)
            user_input: Raw user input (for direct references)
            topk: Number of patterns to consider
            
        Returns:
            ComposedResponse with text and metadata
        """
        # 1. Retrieve relevant response patterns
        patterns = self.fragments.retrieve_patterns(context, topk=topk)
        
        if not patterns:
            # Fallback if no patterns found
            return self._fallback_response()
            
        # 2. Get PMFlow activation signature from working memory
        activation_signature = self._get_activation_signature()
        
        # 3. Weight patterns by activation + success scores
        weighted_patterns = self._weight_patterns(
            patterns, 
            activation_signature
        )
        
        # 4. Apply coherence constraints from working memory
        coherent_patterns = self._filter_by_coherence(
            weighted_patterns,
            self.state
        )
        
        if not coherent_patterns:
            # Use original patterns if coherence filtering too strict
            coherent_patterns = weighted_patterns
            
        # 5. Compose final response
        response = self._compose_from_patterns(
            coherent_patterns,
            context,
            user_input
        )
        
        return response
        
    def _get_activation_signature(self) -> Dict[str, float]:
        """
        Get PMFlow activation strengths from working memory.
        
        Returns topic strengths as activation weights.
        """
        snapshot = self.state.snapshot()
        
        # Convert topics to activation signature
        activation_sig = {}
        for topic in snapshot.topics:
            activation_sig[topic.signature] = topic.strength
            
        return activation_sig
        
    def _weight_patterns(
        self,
        patterns: List[Tuple[ResponsePattern, float]],
        activation_signature: Dict[str, float]
    ) -> List[Tuple[ResponsePattern, float]]:
        """
        Weight patterns by PMFlow activations + learned success.
        
        This is where neural (PMFlow) meets symbolic (patterns)!
        
        Args:
            patterns: Retrieved patterns with similarity scores
            activation_signature: Current working memory activations
            
        Returns:
            Patterns weighted by: similarity * success * activation
        """
        weighted = []
        
        for pattern, similarity in patterns:
            # Base weight from retrieval similarity
            weight = similarity
            
            # Boost by learned success score
            weight *= pattern.success_score
            
            # Boost by activation if pattern relates to active topics
            # (This is a simplified version - could be more sophisticated)
            activation_boost = 1.0
            if activation_signature:
                # Use max activation as boost
                activation_boost = max(activation_signature.values()) + 0.5
                
            weight *= activation_boost
            
            weighted.append((pattern, weight))
            
        # Sort by weight descending
        weighted.sort(key=lambda x: x[1], reverse=True)
        
        return weighted
        
    def _filter_by_coherence(
        self,
        patterns: List[Tuple[ResponsePattern, float]],
        state: ConversationState
    ) -> List[Tuple[ResponsePattern, float]]:
        """
        Filter patterns to maintain conversation coherence.
        
        Uses working memory to ensure response fits current topics.
        
        Args:
            patterns: Weighted patterns
            state: Conversation state with active topics
            
        Returns:
            Coherent patterns only
        """
        snapshot = state.snapshot()
        
        # If no active topics, allow all patterns
        if not snapshot.topics:
            return patterns
            
        # For now, use simple filtering
        # Could enhance with more sophisticated coherence checking
        coherent = []
        
        for pattern, weight in patterns:
            # Calculate coherence score
            coherence = self._calculate_coherence(pattern, snapshot)
            
            # Only keep if reasonably coherent
            if coherence > 0.3:
                coherent.append((pattern, weight * coherence))
                
        return coherent
        
    def _calculate_coherence(
        self, 
        pattern: ResponsePattern,
        snapshot
    ) -> float:
        """
        Calculate how well pattern fits current conversation state.
        
        Args:
            pattern: Response pattern to evaluate
            snapshot: Current conversation state snapshot
            
        Returns:
            Coherence score (0.0-1.0)
        """
        # Start with baseline coherence
        coherence = 0.5
        
        # Boost if pattern matches dominant topic
        if snapshot.topics:
            # Simple heuristic: higher coherence if pattern has been successful
            coherence += 0.3 * pattern.success_score
            
        # Reduce if novelty is very high (pattern might be off-topic)
        if snapshot.novelty > 0.8:
            coherence *= 0.7
            
        return min(coherence, 1.0)
        
    def _compose_from_patterns(
        self,
        weighted_patterns: List[Tuple[ResponsePattern, float]],
        context: str,
        user_input: str
    ) -> ComposedResponse:
        """
        Compose final response from weighted patterns.
        
        Args:
            weighted_patterns: Patterns with composition weights
            context: Conversation context
            user_input: Raw user text
            
        Returns:
            Composed response
        """
        if not weighted_patterns:
            return self._fallback_response()
            
        if self.composition_mode == "best_match":
            return self._compose_best_match(weighted_patterns)
            
        elif self.composition_mode == "weighted_blend":
            return self._compose_weighted_blend(weighted_patterns, context)
            
        elif self.composition_mode == "adaptive":
            return self._compose_adaptive(weighted_patterns, context)
            
        else:
            # Default to best match
            return self._compose_best_match(weighted_patterns)
            
    def _compose_best_match(
        self,
        weighted_patterns: List[Tuple[ResponsePattern, float]]
    ) -> ComposedResponse:
        """
        Use the single best-matching pattern.
        
        Simple but effective strategy.
        """
        pattern, weight = weighted_patterns[0]
        
        return ComposedResponse(
            text=pattern.response_text,
            fragment_ids=[pattern.fragment_id],
            composition_weights=[1.0],
            coherence_score=weight,
            primary_pattern=pattern
        )
        
    def _compose_weighted_blend(
        self,
        weighted_patterns: List[Tuple[ResponsePattern, float]],
        context: str
    ) -> ComposedResponse:
        """
        Blend multiple patterns weighted by their scores.
        
        For now, this selects best match but could be enhanced
        to actually blend text from multiple patterns.
        """
        # Get top pattern
        primary_pattern, primary_weight = weighted_patterns[0]
        
        # For simple blending, use primary pattern
        # TODO: Implement actual text blending using multiple patterns
        # Could concatenate, interleave, or use learned blending rules
        
        response_text = primary_pattern.response_text
        
        # If confidence is low and we have alternatives, could augment
        if primary_weight < 0.6 and len(weighted_patterns) > 1:
            secondary_pattern, _ = weighted_patterns[1]
            # Could append secondary pattern for elaboration
            # response_text += " " + secondary_pattern.response_text
            pass
            
        fragment_ids = [p.fragment_id for p, _ in weighted_patterns[:3]]
        weights = [w for _, w in weighted_patterns[:3]]
        
        return ComposedResponse(
            text=response_text,
            fragment_ids=fragment_ids,
            composition_weights=weights,
            coherence_score=primary_weight,
            primary_pattern=primary_pattern
        )
        
    def _compose_adaptive(
        self,
        weighted_patterns: List[Tuple[ResponsePattern, float]],
        context: str
    ) -> ComposedResponse:
        """
        Adaptively choose composition strategy based on confidence.
        
        High confidence: Use best match
        Low confidence: Blend or ask for clarification
        """
        primary_pattern, primary_weight = weighted_patterns[0]
        
        if primary_weight > 0.7:
            # High confidence: use best match
            return self._compose_best_match(weighted_patterns)
        elif primary_weight > 0.4:
            # Medium confidence: try blending
            return self._compose_weighted_blend(weighted_patterns, context)
        else:
            # Low confidence: ask for clarification
            return ComposedResponse(
                text="Could you clarify what you mean?",
                fragment_ids=["clarification_request"],
                composition_weights=[1.0],
                coherence_score=primary_weight,
                primary_pattern=primary_pattern
            )
            
    def _fallback_response(self) -> ComposedResponse:
        """
        Fallback when no patterns available.
        
        This should rarely happen after bootstrap seeding.
        """
        return ComposedResponse(
            text="I'm not sure how to respond to that yet. I'm still learning!",
            fragment_ids=["fallback"],
            composition_weights=[1.0],
            coherence_score=0.0,
            primary_pattern=None
        )
