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

# Optional: Import grammar stage for sophisticated composition
try:
    from .syntax_stage_bnn import SyntaxStage
    GRAMMAR_AVAILABLE = True
except ImportError:
    GRAMMAR_AVAILABLE = False


@dataclass
class ComposedResponse:
    """A response composed from learned patterns"""
    text: str                           # Final response text
    fragment_ids: List[str]            # Which patterns were used
    composition_weights: List[float]   # How much each pattern contributed
    coherence_score: float             # How well it fits working memory
    primary_pattern: Optional[ResponsePattern] = None   # Main pattern used (can be None for fallback)
    

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
        composition_mode: str = "weighted_blend",
        use_grammar: bool = False
    ):
        """
        Initialize response composer.
        
        Args:
            fragment_store: Store of learned response patterns
            conversation_state: Working memory state
            composition_mode: How to compose responses
                - "best_match": Use highest-weighted pattern only
                - "weighted_blend": Blend multiple patterns
                - "grammar_guided": Use grammatical templates (requires syntax stage)
                - "adaptive": Choose based on confidence
            use_grammar: Enable grammar-guided composition
        """
        self.fragments = fragment_store
        self.state = conversation_state
        self.composition_mode = composition_mode
        
        # Initialize syntax stage if available and requested
        self.syntax_stage = None
        if use_grammar and GRAMMAR_AVAILABLE:
            self.syntax_stage = SyntaxStage()
            print("  📝 BNN-based syntax stage enabled!")
        elif use_grammar and not GRAMMAR_AVAILABLE:
            print("  ⚠️  Syntax stage not available, falling back to standard composition")
        
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
        Blend multiple patterns to create novel responses.
        
        Strategy: Combine fragments from top patterns when appropriate.
        This creates NEW utterances from learned pieces!
        """
        # Get top patterns
        primary_pattern, primary_weight = weighted_patterns[0]
        
        response_text = primary_pattern.response_text
        fragment_ids = [primary_pattern.fragment_id]
        weights = [primary_weight]
        
        # Calculate weight ratio between top two patterns
        # If they're close, blending makes sense
        should_blend = False
        if len(weighted_patterns) > 1:
            secondary_pattern, secondary_weight = weighted_patterns[1]
            
            # Blend if:
            # 1. Secondary is at least 60% as good as primary (close competition)
            # 2. Both patterns have reasonable weights
            weight_ratio = secondary_weight / (primary_weight + 1e-6)
            if weight_ratio > 0.6 and secondary_weight > 0.5:
                should_blend = True
                
                # Try blending
                blended = self._blend_patterns(primary_pattern, secondary_pattern)
                if blended:
                    response_text = blended
                    fragment_ids.append(secondary_pattern.fragment_id)
                    weights.append(secondary_weight)
                    print(f"  🎨 Blended {primary_pattern.intent} + {secondary_pattern.intent}")
        
        # If all weights are low (no good match), try composing from fragments
        if not should_blend and primary_weight < 1.5 and len(weighted_patterns) >= 3:
            # Create novel composition from multiple weak matches
            composed = self._compose_from_fragments(weighted_patterns[:3])
            if composed:
                response_text = composed
                fragment_ids = [p.fragment_id for p, _ in weighted_patterns[:3]]
                weights = [w for _, w in weighted_patterns[:3]]
                print(f"  🎨 Composed from {len(fragment_ids)} fragments")
        
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
    
    def _blend_patterns(
        self,
        primary: ResponsePattern,
        secondary: ResponsePattern
    ) -> Optional[str]:
        """
        Blend two patterns into a novel response.
        
        Strategy: Use BNN syntax stage if available for grammatically-guided blending.
        This creates NEW utterances from learned fragments!
        
        Args:
            primary: Main pattern
            secondary: Supporting pattern
            
        Returns:
            Blended text or None if blending not appropriate
        """
        # If BNN syntax stage is available, use it for intelligent composition!
        if self.syntax_stage:
            return self._blend_with_syntax_bnn(primary, secondary)
        
        # Fallback: Heuristic blending (original method)
        # Don't blend if patterns end with questions (creates awkward double-question)
        if primary.response_text.strip().endswith('?') and secondary.response_text.strip().endswith('?'):
            return None
            
        # Choose connector based on pattern types
        connectors = {
            ('question_info', 'explain'): ' ',  # Question + explanation flows naturally
            ('acknowledgment', 'interest'): ' ',  # "I see. That's interesting!"
            ('interest', 'question_detail'): ' ',  # "Interesting! Tell me more?"
            ('agreement', 'explain'): ' ',  # "Yes. The reason is..."
            ('greeting', 'question_info'): ' ',  # "Hello! What can I help with?"
            ('greeting', 'interest'): ' ',  # "Hello! Wow, that's..."
            ('capability', 'question_detail'): ' ',  # Statement + question
            ('explain', 'meta'): ' ',  # Explanation + meta-comment
            ('learned', 'capability'): ' ',  # Learned phrase + capability
            ('learned', 'meta'): ' ',  # Learned phrase + meta
            ('clarify', 'capability'): ' ',  # Question + answer
        }
        
        # Get connector (default: space for natural flow)
        intent_pair = (primary.intent, secondary.intent)
        connector = connectors.get(intent_pair, ' ')
        
        # Clean up primary text punctuation
        primary_text = primary.response_text.strip()
        
        # If primary doesn't end with strong punctuation and connector is just space,
        # we need to ensure proper sentence boundary
        if connector == ' ' and primary_text[-1] not in '!?':
            # Replace trailing period with nothing if present, we'll handle it
            if primary_text[-1] == '.':
                primary_text = primary_text[:-1]
            # Add appropriate punctuation based on context
            if secondary.response_text.strip()[0].isupper():
                primary_text += '.'  # New sentence
            else:
                primary_text += ','  # Continuation
        elif not primary_text[-1] in '.!?,':
            primary_text += '.'
            
        # Blend
        blended = primary_text + connector + secondary.response_text.strip()
        
        # Don't return if too long (over 150 chars gets verbose)
        if len(blended) > 150:
            return None
            
        return blended
    
    def _compose_from_fragments(
        self,
        patterns: List[Tuple[ResponsePattern, float]]
    ) -> Optional[str]:
        """
        Create novel composition from multiple weak pattern matches.
        
        When no single pattern is strong, combine fragments to create
        a reasonable response. This is pure compositional generation!
        
        Args:
            patterns: Multiple patterns to compose from
            
        Returns:
            Composed text or None if composition fails
        """
        if len(patterns) < 2:
            return None
            
        # Extract key phrases from patterns
        fragments = []
        for pattern, weight in patterns:
            # Take short patterns whole, extract from long ones
            text = pattern.response_text.strip()
            if len(text.split()) <= 6:
                fragments.append(text)
            else:
                # Extract first clause or sentence
                if '.' in text:
                    fragments.append(text.split('.')[0].strip())
                elif ',' in text:
                    fragments.append(text.split(',')[0].strip())
                else:
                    fragments.append(text)
        
        # Try to compose coherently
        # Strategy: Use first fragment as base, add others if they fit
        if not fragments:
            return None
            
        composed = fragments[0]
        
        # Add second fragment if it complements
        if len(fragments) > 1 and len(composed.split()) < 8:
            # Add connector
            if not composed[-1] in '.!?,':
                composed += ','
            composed += ' ' + fragments[1]
            
        # Ensure proper ending
        if not composed[-1] in '.!?':
            composed += '.'
            
        return composed
    
    def _blend_with_syntax_bnn(self, primary: ResponsePattern, secondary: ResponsePattern) -> str:
        """
        Blend two patterns using BNN-based syntax stage for grammatical composition.
        
        Args:
            primary: Primary pattern to use as base
            secondary: Secondary pattern to blend in
            
        Returns:
            Grammatically composed text using BNN-learned templates
        """
        if not self.syntax_stage:
            # Fallback if syntax stage not available
            return self._simple_blend(primary.response_text, secondary.response_text)
        
        # Process patterns through syntax stage
        tokens_a = primary.response_text.split()
        tokens_b = secondary.response_text.split()
        
        artifact_a = self.syntax_stage.process(tokens_a)
        artifact_b = self.syntax_stage.process(tokens_b)
        
        # Extract intents and retrieved patterns from artifacts
        intent_a = artifact_a.metadata.get('intent', 'statement')
        intent_b = artifact_b.metadata.get('intent', 'statement')
        
        # Get matched patterns from retrieval_info (list of dicts)
        retrieval_a = artifact_a.metadata.get('retrieval_info', [])
        retrieval_b = artifact_b.metadata.get('retrieval_info', [])
        
        # Select composition strategy based on intents
        template = None
        if intent_a == intent_b:
            # Same intent: blend templates
            if retrieval_a and retrieval_b:
                # Use higher-confidence template as base
                base_info = retrieval_a[0] if retrieval_a[0]['similarity'] > retrieval_b[0]['similarity'] else retrieval_b[0]
                template = base_info.get('template')
            elif retrieval_a:
                template = retrieval_a[0].get('template')
            elif retrieval_b:
                template = retrieval_b[0].get('template')
        else:
            # Different intents: choose dominant
            base_info = retrieval_a[0] if retrieval_a else (retrieval_b[0] if retrieval_b else None)
            template = base_info.get('template') if base_info else None
        
        # Apply template or use heuristic fallback
        if template:
            # Fill template with content from both patterns
            composed = self._apply_syntax_template(template, tokens_a, tokens_b)
        else:
            # Fallback to simpler composition
            composed = self._simple_blend(primary.response_text, secondary.response_text)
        
        return composed
    
    def _apply_syntax_template(self, template: str, tokens_a: list, tokens_b: list) -> str:
        """
        Apply a syntax template by filling slots with tokens.
        
        Args:
            template: Template string with {pos} slots (e.g., "this is {adv} {adj}")
            tokens_a: Tokens from primary pattern
            tokens_b: Tokens from secondary pattern
            
        Returns:
            Composed text with template slots filled
        """
        import re
        
        # Combine token pools
        all_tokens = tokens_a + tokens_b
        
        # Extract slot types from template
        slots = re.findall(r'\{([^}]+)\}', template)
        
        if not slots:
            # No slots, template is literal
            return template
        
        # Simple slot filling: use tokens in order
        filled = template
        token_idx = 0
        
        for slot in slots:
            if token_idx < len(all_tokens):
                filled = filled.replace(f'{{{slot}}}', all_tokens[token_idx], 1)
                token_idx += 1
            else:
                # Out of tokens, use placeholder
                filled = filled.replace(f'{{{slot}}}', slot.lower(), 1)
        
        # Ensure proper capitalization and ending
        if filled:
            filled = filled[0].upper() + filled[1:]
            if not filled[-1] in '.!?':
                filled += '.'
        
        return filled
    
    def _simple_blend(self, text_a: str, text_b: str) -> str:
        """
        Simple fallback blending when no BNN template available.
        
        Args:
            text_a: Primary text
            text_b: Secondary text
            
        Returns:
            Simple grammatical blend
        """
        # Take first clause from each
        clause_a = text_a.split('.')[0].split(',')[0].strip()
        clause_b = text_b.split('.')[0].split(',')[0].strip()
        
        # Combine with connector
        if len(clause_a.split()) + len(clause_b.split()) < 12:
            return f"{clause_a}, {clause_b}."
        else:
            return f"{clause_a}."
            
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
