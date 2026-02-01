"""
Response Composer for Lilith V2.

Ported from V1's response_composer.py with v2 architectural patterns.

Key capabilities:
1. Pattern-based response retrieval (using PatternStoreAdapter)
2. Multiple composition modes (best_match, weighted_blend, adaptive)
3. Pattern blending for novel response construction
4. Success-based learning (feedback → pattern reinforcement)
5. Contrastive learning for semantic corrections
"""

import logging
import re
import hashlib
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Callable

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------

@dataclass
class ResponsePattern:
    """A stored response pattern with metadata."""
    fragment_id: str
    trigger_context: str      # What triggers this pattern (question/topic)
    response_text: str        # The response we give
    intent: str = "general"   # Intent category
    success_score: float = 0.5  # Reinforcement score (0-1)
    embedding: Optional[List[float]] = None  # Vector representation
    usage_count: int = 0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResponsePattern':
        return cls(
            fragment_id=data.get("id", data.get("fragment_id", "")),
            trigger_context=data.get("trigger_context", ""),
            response_text=data.get("response_text", ""),
            intent=data.get("intent", "general"),
            success_score=data.get("success_score", 0.5),
            embedding=data.get("embedding"),
            usage_count=data.get("usage_count", 0),
        )


class CompositionMode(Enum):
    """Available response composition strategies."""
    BEST_MATCH = "best_match"           # Use highest-scoring pattern only
    WEIGHTED_BLEND = "weighted_blend"   # Blend multiple patterns
    ADAPTIVE = "adaptive"               # Choose based on confidence
    GRAPH_FIRST = "graph_first"         # Prioritize graph traversal results


@dataclass
class ComposedResponse:
    """Result of response composition with metadata."""
    text: str                           # Final response text
    fragment_ids: List[str]             # Patterns used
    composition_weights: List[float]    # Contribution of each pattern
    coherence_score: float              # Overall confidence
    primary_pattern: Optional[ResponsePattern] = None
    confidence: float = 1.0
    is_fallback: bool = False
    is_blended: bool = False            # Was this blended from multiple patterns?
    source: str = "pattern"             # pattern, graph, template, fallback
    
    def __post_init__(self):
        if not self.fragment_ids:
            self.fragment_ids = []
        if not self.composition_weights:
            self.composition_weights = []


@dataclass 
class AdaptivePolicy:
    """
    Small learned policy for threshold adjustment (EMA-based).
    Ported from V1's AdaptiveHeuristicPolicy.
    """
    alpha: float = 0.2              # Learning rate
    confidence_ema: float = 0.65    # Exponential moving average of confidence
    success_ema: float = 0.7        # Exponential moving average of success
    
    def register_confidence(self, confidence: float) -> None:
        """Update confidence EMA after each response."""
        try:
            conf = float(confidence)
            self.confidence_ema = (1 - self.alpha) * self.confidence_ema + self.alpha * conf
        except Exception:
            pass
    
    def register_success(self, success: bool) -> None:
        """Update success EMA based on outcome."""
        target = 1.0 if success else 0.0
        self.success_ema = (1 - self.alpha) * self.success_ema + self.alpha * target
    
    @property
    def adaptive_threshold(self) -> float:
        """Dynamic confidence threshold based on recent performance."""
        # If success rate is high, lower threshold (be more adventurous)
        # If success rate is low, raise threshold (be more conservative)
        base = 0.6
        adjustment = (self.success_ema - 0.5) * 0.2  # -0.1 to +0.1
        return max(0.4, min(0.8, base - adjustment))


# -----------------------------------------------------------------
# Response Composer (The core output engine)
# -----------------------------------------------------------------

class ResponseComposer:
    """
    Composes responses using learned patterns with PMFlow-guided retrieval.
    
    Key Insight: Response generation uses the SAME mechanisms as understanding -
    retrieve patterns, weight by confidence, compose output.
    
    Ported from V1 with v2 architectural patterns.
    """
    
    def __init__(
        self,
        pattern_store: Any,  # PatternStoreAdapter
        graph_store: Any,    # RelationalGraphStore
        encoder: Any = None, # PMFlow encoder for similarity
        composition_mode: str = "adaptive",
        pragmatic_system: Any = None,
        enable_blending: bool = True,
        enable_learning: bool = True,
    ):
        """
        Initialize response composer.
        
        Args:
            pattern_store: Adapter for pattern storage/retrieval
            graph_store: Graph for inference-based responses
            encoder: PMFlow encoder for semantic similarity
            composition_mode: How to compose responses
            pragmatic_system: Optional PragmaticSystem for templates
            enable_blending: Allow pattern blending for novel responses
            enable_learning: Enable success-based learning
        """
        self.patterns = pattern_store
        self.graph = graph_store
        self.encoder = encoder
        self.mode = CompositionMode(composition_mode)
        self.pragmatics = pragmatic_system
        self.enable_blending = enable_blending
        self.enable_learning = enable_learning
        
        # Adaptive threshold learning
        self.adaptive = AdaptivePolicy()
        
        # Tracking for learning loop
        self.last_query: str = ""
        self.last_response: Optional[ComposedResponse] = None
        self.last_pattern_id: Optional[str] = None
        
        # Metrics
        self.metrics = {
            'responses_composed': 0,
            'blends_attempted': 0,
            'blends_succeeded': 0,
            'fallback_count': 0,
            'success_feedback_received': 0,
        }
        
        # Fallback responses
        self.fallbacks = [
            "I'm listening. Could you tell me more?",
            "I'd like to understand better. Please elaborate.",
            "I'm processing what you said. Can you provide more context?",
        ]
        
        logger.info(f"ResponseComposer initialized with mode={composition_mode}")
    
    # -----------------------------------------------------------------
    # Main API
    # -----------------------------------------------------------------
    
    def compose(
        self,
        thought_context: Dict[str, Any],
        user_input: str = "",
        topk: int = 5,
    ) -> ComposedResponse:
        """
        Compose a response based on cognitive context.
        
        Args:
            thought_context: Output from cognitive stage with:
                - inference: Graph traversal results
                - extracted_knowledge: Learned relations
                - external_knowledge: Augmented knowledge
                - affect: Mood state
            user_input: Original user input
            topk: Number of patterns to consider
            
        Returns:
            ComposedResponse with text and metadata
        """
        self.last_query = user_input
        self.metrics['responses_composed'] += 1
        
        # Priority 1: Graph inference results (direct answers)
        inference = thought_context.get("inference", [])
        if inference and self.mode == CompositionMode.GRAPH_FIRST:
            graph_response = self._compose_from_inference(inference, thought_context)
            if graph_response and graph_response.confidence >= 0.7:
                self.last_response = graph_response
                return graph_response
        
        # Priority 2: Pattern-based response with retrieval
        pattern_response = self._compose_from_patterns(user_input, thought_context, topk)
        if pattern_response and not pattern_response.is_fallback:
            self.last_response = pattern_response
            self.adaptive.register_confidence(pattern_response.confidence)
            return pattern_response
        
        # Priority 3: Graph inference (fallback)
        if inference:
            graph_response = self._compose_from_inference(inference, thought_context)
            if graph_response:
                self.last_response = graph_response
                return graph_response
        
        # Priority 4: Extracted knowledge acknowledgment
        extracted = thought_context.get("extracted_knowledge", [])
        if extracted:
            ack_response = self._acknowledge_learning(extracted)
            self.last_response = ack_response
            return ack_response
        
        # Priority 5: External knowledge
        external = thought_context.get("external_knowledge", [])
        if external:
            ext_response = self._compose_from_external(external)
            self.last_response = ext_response
            return ext_response
        
        # Fallback
        fallback = self._compose_fallback()
        self.last_response = fallback
        return fallback
    
    # -----------------------------------------------------------------
    # Pattern-Based Composition
    # -----------------------------------------------------------------
    
    def _compose_from_patterns(
        self,
        user_input: str,
        context: Dict[str, Any],
        topk: int = 5
    ) -> Optional[ComposedResponse]:
        """
        Compose response by retrieving and potentially blending patterns.
        """
        if not user_input or not self.patterns:
            return None
        
        # 1. Retrieve matching patterns
        patterns = self._retrieve_patterns(user_input, topk)
        if not patterns:
            return None
        
        # 2. Apply composition strategy
        if self.mode == CompositionMode.BEST_MATCH:
            return self._compose_best_match(patterns)
        elif self.mode == CompositionMode.WEIGHTED_BLEND:
            return self._compose_weighted_blend(patterns, context)
        elif self.mode == CompositionMode.ADAPTIVE:
            return self._compose_adaptive(patterns, context)
        else:
            return self._compose_best_match(patterns)
    
    def _retrieve_patterns(
        self, 
        query: str, 
        topk: int = 5
    ) -> List[Tuple[ResponsePattern, float]]:
        """
        Retrieve patterns matching query using semantic similarity.
        """
        results = []
        
        # Get query embedding
        query_embedding = None
        if self.encoder:
            try:
                tokens = query.split()
                if hasattr(self.encoder, "encode_with_components"):
                    combined, _, _ = self.encoder.encode_with_components(tokens)
                else:
                    combined = self.encoder.encode(tokens)
                
                if hasattr(combined, "detach"):
                    combined = combined.detach().cpu()
                if hasattr(combined, "squeeze"):
                    combined = combined.squeeze()
                if hasattr(combined, "tolist"):
                    query_embedding = combined.tolist()
            except Exception as e:
                logger.debug(f"Encoding failed: {e}")
        
        # Retrieve from pattern store
        try:
            for pat_dict in self.patterns.list():
                pattern = ResponsePattern.from_dict(pat_dict)
                
                # Calculate similarity
                score = self._calculate_similarity(
                    query, query_embedding, 
                    pattern.trigger_context, pattern.embedding
                )
                
                # Weight by success score (reinforcement)
                weighted_score = score * (0.5 + 0.5 * pattern.success_score)
                
                results.append((pattern, weighted_score))
        except Exception as e:
            logger.warning(f"Pattern retrieval failed: {e}")
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:topk]
    
    def _calculate_similarity(
        self,
        query_text: str,
        query_embedding: Optional[List[float]],
        pattern_text: str,
        pattern_embedding: Optional[List[float]]
    ) -> float:
        """
        Calculate similarity between query and pattern.
        Combines semantic (embedding) and lexical (keyword) similarity.
        """
        semantic_score = 0.0
        lexical_score = 0.0
        
        # Semantic similarity via embeddings
        if query_embedding and pattern_embedding:
            try:
                semantic_score = self._cosine_similarity(query_embedding, pattern_embedding)
            except Exception:
                pass
        
        # Lexical similarity via keyword overlap
        query_words = set(query_text.lower().split())
        pattern_words = set(pattern_text.lower().split())
        common = query_words & pattern_words
        if query_words and pattern_words:
            lexical_score = len(common) / min(len(query_words), len(pattern_words))
        
        # Combine (0.6 semantic, 0.4 lexical)
        if semantic_score > 0:
            return 0.6 * semantic_score + 0.4 * lexical_score
        else:
            return lexical_score
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        import math
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
        norm_b = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (norm_a * norm_b)
    
    def _compose_best_match(
        self, 
        patterns: List[Tuple[ResponsePattern, float]]
    ) -> ComposedResponse:
        """Use the single best matching pattern."""
        best, score = patterns[0]
        
        self.last_pattern_id = best.fragment_id
        
        return ComposedResponse(
            text=best.response_text,
            fragment_ids=[best.fragment_id],
            composition_weights=[score],
            coherence_score=score,
            primary_pattern=best,
            confidence=score,
            is_fallback=score < self.adaptive.adaptive_threshold,
            source="pattern",
        )
    
    def _compose_weighted_blend(
        self,
        patterns: List[Tuple[ResponsePattern, float]],
        context: Dict[str, Any]
    ) -> ComposedResponse:
        """
        Blend multiple patterns to create novel responses.
        
        This creates NEW utterances from learned pieces!
        """
        if not patterns:
            return self._compose_fallback()
        
        primary, primary_score = patterns[0]
        
        # Check if blending is appropriate
        should_blend = False
        secondary = None
        secondary_score = 0.0
        
        if self.enable_blending and len(patterns) > 1:
            secondary, secondary_score = patterns[1]
            
            # Blend if:
            # 1. Secondary is at least 60% as good as primary
            # 2. Both have reasonable confidence
            # 3. They have compatible intents
            weight_ratio = secondary_score / (primary_score + 1e-6)
            compatible = self._check_intent_compatibility(primary.intent, secondary.intent)
            
            if weight_ratio > 0.6 and secondary_score > 0.4 and compatible:
                should_blend = True
                self.metrics['blends_attempted'] += 1
        
        if should_blend and secondary:
            blended_text = self._blend_patterns(primary, secondary)
            if blended_text:
                self.metrics['blends_succeeded'] += 1
                self.last_pattern_id = primary.fragment_id
                
                return ComposedResponse(
                    text=blended_text,
                    fragment_ids=[primary.fragment_id, secondary.fragment_id],
                    composition_weights=[primary_score, secondary_score],
                    coherence_score=primary_score,
                    primary_pattern=primary,
                    confidence=primary_score,
                    is_blended=True,
                    source="blended",
                )
        
        # Fallback to best match
        return self._compose_best_match(patterns)
    
    def _compose_adaptive(
        self,
        patterns: List[Tuple[ResponsePattern, float]],
        context: Dict[str, Any]
    ) -> ComposedResponse:
        """
        Adaptively choose strategy based on confidence.
        
        High confidence → best match
        Medium confidence → try blending  
        Low confidence → ask for clarification
        """
        if not patterns:
            return self._compose_fallback()
        
        primary, primary_score = patterns[0]
        threshold = self.adaptive.adaptive_threshold
        
        if primary_score > threshold + 0.1:
            # High confidence: use best match
            return self._compose_best_match(patterns)
        elif primary_score > threshold - 0.2:
            # Medium confidence: try blending
            return self._compose_weighted_blend(patterns, context)
        else:
            # Low confidence: clarification
            return ComposedResponse(
                text="Could you clarify what you mean? I want to make sure I understand.",
                fragment_ids=["clarification_request"],
                composition_weights=[primary_score],
                coherence_score=primary_score,
                primary_pattern=primary,
                confidence=primary_score,
                is_fallback=True,
                source="clarification",
            )
    
    # -----------------------------------------------------------------
    # Pattern Blending (Novel Response Construction)
    # -----------------------------------------------------------------
    
    def _check_intent_compatibility(self, intent_a: str, intent_b: str) -> bool:
        """
        Check if two intents are compatible for blending.
        Prevents nonsensical combinations.
        """
        # Incompatible pairs
        incompatible = {
            ('greeting', 'technical'),
            ('identity', 'question'),
            ('capability', 'emotional'),
            ('factual', 'opinion'),
        }
        
        pair = tuple(sorted([intent_a.lower(), intent_b.lower()]))
        return pair not in incompatible
    
    def _blend_patterns(
        self,
        primary: ResponsePattern,
        secondary: ResponsePattern
    ) -> Optional[str]:
        """
        Blend two patterns into a novel response.
        
        Creates NEW utterances from learned fragments!
        """
        # Don't blend if both end with questions
        if (primary.response_text.strip().endswith('?') and 
            secondary.response_text.strip().endswith('?')):
            return None
        
        # Don't blend if texts are very similar
        if self._text_similarity(primary.response_text, secondary.response_text) > 0.8:
            return None
        
        # Extract clauses
        primary_clause = self._extract_first_clause(primary.response_text)
        secondary_clause = self._extract_first_clause(secondary.response_text)
        
        # Combine with appropriate connector
        connector = self._select_connector(primary.intent, secondary.intent)
        
        # Lowercase the second clause's first letter for natural flow
        # Exception: Keep "I" uppercase
        if secondary_clause and len(secondary_clause) > 1:
            first_word = secondary_clause.split()[0] if secondary_clause.split() else ""
            if first_word not in ("I", "I'm", "I've", "I'll", "I'd"):
                secondary_clause = secondary_clause[0].lower() + secondary_clause[1:]
        
        blended = f"{primary_clause}{connector}{secondary_clause}"
        
        # Ensure proper ending
        if not blended.rstrip()[-1] in '.!?':
            blended += '.'
        
        # Capitalize first letter
        blended = blended[0].upper() + blended[1:]
        
        logger.debug(f"Blended: '{primary.response_text[:30]}...' + '{secondary.response_text[:30]}...'")
        return blended
    
    def _extract_first_clause(self, text: str) -> str:
        """Extract first clause/sentence from text."""
        text = text.strip()
        
        # Short text: use as-is
        if len(text.split()) <= 6:
            # Remove trailing punctuation for blending
            return text.rstrip('.!?,')
        
        # Extract first sentence
        if '.' in text:
            clause = text.split('.')[0].strip()
        elif ',' in text:
            clause = text.split(',')[0].strip()
        else:
            # Take first 6 words
            words = text.split()[:6]
            clause = ' '.join(words)
        
        return clause.rstrip('.!?,')
    
    def _select_connector(self, intent_a: str, intent_b: str) -> str:
        """Select appropriate connector for blending."""
        # Intent-based connectors
        if intent_a == intent_b:
            return ", and "
        elif intent_b in ('question', 'clarification'):
            return ". "
        elif intent_b in ('elaboration', 'detail'):
            return " — specifically, "
        else:
            return ". Additionally, "
    
    def _text_similarity(self, a: str, b: str) -> float:
        """Simple text similarity based on word overlap."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        common = words_a & words_b
        return len(common) / max(len(words_a), len(words_b))
    
    # -----------------------------------------------------------------
    # Graph / Inference Composition
    # -----------------------------------------------------------------
    
    def _compose_from_inference(
        self,
        inference: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Optional[ComposedResponse]:
        """
        Compose response from graph traversal results.
        """
        if not inference:
            return None
        
        top = inference[0]
        
        # Case 1: Path-based inference (pattern trigger → response)
        if "path_ids" in top and len(top.get("path_ids", [])) >= 2:
            target_id = top["path_ids"][-1]  # Last node in path
            try:
                node = self.graph.get_node(target_id)
                if node and node.get("term"):
                    return ComposedResponse(
                        text=node["term"],
                        fragment_ids=[target_id],
                        composition_weights=[top.get("confidence", 0.8)],
                        coherence_score=top.get("confidence", 0.8),
                        confidence=top.get("confidence", 0.8),
                        source="graph_inference",
                    )
            except Exception as e:
                logger.debug(f"Graph node lookup failed: {e}")
        
        # Case 2: Subject-predicate-object result
        if "subject" in top and "object" in top:
            text = f"{top['subject']} is {top.get('predicate', 'related to')} {top['object']}"
            text = text[0].upper() + text[1:] + "."
            
            return ComposedResponse(
                text=text,
                fragment_ids=[],
                composition_weights=[top.get("confidence", 0.7)],
                coherence_score=top.get("confidence", 0.7),
                confidence=top.get("confidence", 0.7),
                source="graph_relation",
            )
        
        return None
    
    def _acknowledge_learning(
        self,
        extracted: List[Any]
    ) -> ComposedResponse:
        """Acknowledge that we learned something from the input."""
        rel = extracted[0]
        
        # Use pragmatics if available
        if self.pragmatics:
            template = self.pragmatics.get_template("teaching", ["subject", "object"])
            if template:
                text = self.pragmatics.fill(template, {
                    "subject": rel.subject,
                    "object": rel.object
                })
                return ComposedResponse(
                    text=text,
                    fragment_ids=["teaching_ack"],
                    composition_weights=[0.9],
                    coherence_score=0.9,
                    confidence=0.9,
                    source="learning_ack",
                )
        
        # Fallback acknowledgment
        text = f"I see! So {rel.subject} is {rel.object}. I've learned that now."
        return ComposedResponse(
            text=text,
            fragment_ids=["teaching_ack"],
            composition_weights=[0.9],
            coherence_score=0.9,
            confidence=0.9,
            source="learning_ack",
        )
    
    def _compose_from_external(
        self,
        external: List[str]
    ) -> ComposedResponse:
        """Compose from external knowledge (e.g., Wikipedia)."""
        text = external[0] if external else ""
        return ComposedResponse(
            text=text,
            fragment_ids=["external_knowledge"],
            composition_weights=[0.7],
            coherence_score=0.7,
            confidence=0.7,
            source="external",
        )
    
    def _compose_fallback(self) -> ComposedResponse:
        """Generate fallback response when nothing matched."""
        self.metrics['fallback_count'] += 1
        text = random.choice(self.fallbacks)
        
        return ComposedResponse(
            text=text,
            fragment_ids=["fallback"],
            composition_weights=[0.1],
            coherence_score=0.1,
            confidence=0.1,
            is_fallback=True,
            source="fallback",
        )
    
    # -----------------------------------------------------------------
    # Learning Loop (Feedback → Pattern Reinforcement)
    # -----------------------------------------------------------------
    
    def record_outcome(self, success: bool) -> None:
        """
        Record the outcome of the last response.
        
        This is where the system LEARNS what works!
        
        Args:
            success: True if conversation continued well, False if it broke down
                     
        Signals of success:
            - User continues the topic → True
            - User asks follow-up question → True
            - User changes topic abruptly → False
            - User says "what?" or "huh?" → False
        """
        if not self.enable_learning:
            return
        
        self.metrics['success_feedback_received'] += 1
        
        # Update adaptive policy
        self.adaptive.register_success(success)
        
        # Update pattern success score if we used a pattern
        if self.last_pattern_id and self.patterns:
            # Calculate feedback signal
            feedback = 0.2 if success else -0.15
            
            try:
                self.patterns.update_success(
                    self.last_pattern_id,
                    feedback,
                    plasticity_rate=0.1
                )
                logger.debug(
                    f"Updated pattern {self.last_pattern_id}: "
                    f"{'✓' if success else '✗'} (Δ={feedback:+.2f})"
                )
            except Exception as e:
                logger.warning(f"Failed to update pattern success: {e}")
    
    def add_pattern(
        self,
        trigger: str,
        response: str,
        intent: str = "learned",
        initial_score: float = 0.5,
    ) -> str:
        """
        Add a new response pattern.
        
        Args:
            trigger: What triggers this response
            response: The response text
            intent: Intent category
            initial_score: Starting success score
            
        Returns:
            Pattern ID
        """
        if not self.patterns:
            raise RuntimeError("No pattern store configured")
        
        # Generate embedding if encoder available
        embedding = None
        if self.encoder:
            try:
                tokens = trigger.split()
                if hasattr(self.encoder, "encode_with_components"):
                    combined, _, _ = self.encoder.encode_with_components(tokens)
                else:
                    combined = self.encoder.encode(tokens)
                
                if hasattr(combined, "detach"):
                    combined = combined.detach().cpu()
                if hasattr(combined, "squeeze"):
                    combined = combined.squeeze()
                if hasattr(combined, "tolist"):
                    embedding = combined.tolist()
            except Exception:
                pass
        
        # Generate fragment ID
        fragment_id = f"resp_{hashlib.md5(trigger.encode()).hexdigest()[:12]}"
        
        # Add to store
        pid = self.patterns.add_pattern(
            fragment_id=fragment_id,
            trigger_context=trigger,
            response_text=response,
            intent=intent,
            success_score=initial_score,
            embedding=embedding,
        )
        
        logger.info(f"Added response pattern: {fragment_id} for trigger '{trigger[:30]}...'")
        return pid
    
    def upvote(self, pattern_id: str, strength: float = 0.25) -> None:
        """Manual positive feedback for a response."""
        if self.patterns:
            self.patterns.update_success(pattern_id, strength, plasticity_rate=1.0)
            logger.info(f"👍 Upvoted pattern: {pattern_id}")
    
    def downvote(self, pattern_id: str, strength: float = 0.3) -> None:
        """Manual negative feedback for a response."""
        if self.patterns:
            self.patterns.update_success(pattern_id, -strength, plasticity_rate=1.0)
            logger.info(f"👎 Downvoted pattern: {pattern_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get composition statistics."""
        return {
            **self.metrics,
            'adaptive_threshold': self.adaptive.adaptive_threshold,
            'confidence_ema': self.adaptive.confidence_ema,
            'success_ema': self.adaptive.success_ema,
            'blend_success_rate': (
                self.metrics['blends_succeeded'] / max(1, self.metrics['blends_attempted'])
            ),
        }


__all__ = [
    'ResponsePattern',
    'CompositionMode', 
    'ComposedResponse',
    'AdaptivePolicy',
    'ResponseComposer',
]
