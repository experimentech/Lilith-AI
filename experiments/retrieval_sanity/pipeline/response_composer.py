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
        # 1. Retrieve relevant response patterns (based on trigger context)
        patterns = self.fragments.retrieve_patterns(context, topk=topk * 3)  # Get more candidates
        
        if not patterns:
            # Fallback if no patterns found
            return self._fallback_response()
        
        # 2. Score patterns by semantic relevance to user query
        # This is the KEY fix: match response content to user's actual question!
        if user_input:
            patterns = self._score_by_query_relevance(patterns, user_input, topk=topk)
        else:
            patterns = patterns[:topk]
        
        if not patterns:
            return self._fallback_response()
        
        # 2b. Check if best pattern has sufficient relevance
        # If not, provide a graceful fallback instead of hallucinating
        best_pattern, best_score = patterns[0]
        if best_score < 0.70:  # Confidence threshold - reject weak matches
            return self._fallback_response_low_confidence(user_input, best_pattern, best_score)
        
        # 2c. Additional check: is user asking about specific topic not in training data?
        # Check for specific technical terms that aren't well-covered
        uncovered_terms = ['transformer', 'cnn', 'rnn', 'gan', 'lstm', 'gru', 'vae', 
                          'resnet', 'bert', 'gpt', 'diffusion', 'reinforcement']
        user_lower = user_input.lower()
        if any(term in user_lower for term in uncovered_terms):
            # Check if response actually addresses that term
            response_lower = best_pattern.response_text.lower()
            query_terms = set(user_lower.split())
            response_terms = set(response_lower.split())
            
            # If the specific term isn't in the response, it's probably not a good match
            uncovered_in_query = [t for t in uncovered_terms if t in user_lower]
            if uncovered_in_query and not any(t in response_lower for t in uncovered_in_query):
                return self._fallback_response_low_confidence(user_input, best_pattern, best_score)
            
        # 3. For high-relevance patterns, use top match directly
        # (Additional weighting can destroy semantic relevance order)
        if user_input and best_score > 0.75:
            # High confidence - use best pattern directly
            return ComposedResponse(
                text=best_pattern.response_text,
                fragment_ids=[best_pattern.fragment_id],
                composition_weights=[1.0],
                coherence_score=best_score,
                primary_pattern=best_pattern
            )
            
        # 4. For medium-relevance patterns, apply gentle PMFlow weighting
        activation_signature = self._get_activation_signature()
        
        # 5. Weight patterns by activation + success scores
        weighted_patterns = self._weight_patterns(
            patterns, 
            activation_signature
        )
        
        # 6. Apply coherence constraints from working memory
        coherent_patterns = self._filter_by_coherence(
            weighted_patterns,
            self.state
        )
        
        if not coherent_patterns:
            # Use original patterns if coherence filtering too strict
            coherent_patterns = weighted_patterns
            
        # 7. Compose final response
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
    
    def _score_by_query_relevance(
        self,
        patterns: List[Tuple[ResponsePattern, float]],
        user_query: str,
        topk: int = 5,
        min_relevance: float = 0.35
    ) -> List[Tuple[ResponsePattern, float]]:
        """
        Score patterns by semantic relevance between user query and response content.
        
        This is THE KEY to fixing off-topic responses!
        
        Instead of just matching trigger contexts, we check if the RESPONSE CONTENT
        is actually relevant to what the user is asking about.
        
        Args:
            patterns: Candidate patterns with trigger-based scores
            user_query: User's actual input query
            topk: Number of relevant patterns to return
            min_relevance: Minimum semantic similarity threshold
            
        Returns:
            Patterns scored by query-response semantic relevance
        """
        # Encode user query
        try:
            query_emb = self.fragments.encoder.encode(user_query)
            if hasattr(query_emb, 'cpu'):
                query_emb = query_emb.cpu().numpy()
            query_emb = query_emb.flatten()
            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        except Exception:
            # If encoding fails, return original patterns
            return patterns[:topk]
        
        # Score each pattern by semantic similarity between query and response content
        relevance_scored = []
        for pattern, trigger_score in patterns:
            try:
                # Encode the RESPONSE TEXT (not trigger context!)
                response_emb = self.fragments.encoder.encode(pattern.response_text)
                if hasattr(response_emb, 'cpu'):
                    response_emb = response_emb.cpu().numpy()
                response_emb = response_emb.flatten()
                response_norm = response_emb / (np.linalg.norm(response_emb) + 1e-8)
                
                # Semantic similarity between user query and response content
                relevance = float(np.dot(query_norm, response_norm))
                
                # Also encode trigger context for comparison
                trigger_emb = self.fragments.encoder.encode(pattern.trigger_context)
                if hasattr(trigger_emb, 'cpu'):
                    trigger_emb = trigger_emb.cpu().numpy()
                trigger_emb = trigger_emb.flatten()
                trigger_norm = trigger_emb / (np.linalg.norm(trigger_emb) + 1e-8)
                
                trigger_relevance = float(np.dot(query_norm, trigger_norm))
                
                # Combined score: max of response relevance and trigger relevance
                # This allows both "answers to queries" and "contextual responses"
                combined_relevance = max(relevance, trigger_relevance * 0.7)
                
                # Filter out low-relevance patterns
                if combined_relevance >= min_relevance:
                    relevance_scored.append((pattern, combined_relevance))
                    
            except Exception:
                # If encoding fails, use original score but penalize
                if trigger_score > min_relevance:
                    relevance_scored.append((pattern, trigger_score * 0.5))
        
        # Sort by relevance descending
        relevance_scored.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k most relevant
        result = relevance_scored[:topk]
        
        # Debug: print selected pattern
        if result:
            best_pattern, best_score = result[0]
            print(f"     → Selected: '{best_pattern.response_text[:50]}...' (intent: {best_pattern.intent}, relevance: {best_score:.3f})")
        
        return result
        
    def _weight_patterns(
        self,
        patterns: List[Tuple[ResponsePattern, float]],
        activation_signature: Dict[str, float]
    ) -> List[Tuple[ResponsePattern, float]]:
        """
        Weight patterns by PMFlow activations + learned success.
        
        This is where neural (PMFlow) meets symbolic (patterns)!
        
        IMPORTANT: For semantically-scored patterns, we apply gentle boosting
        to preserve relevance order while still rewarding successful patterns.
        
        Args:
            patterns: Retrieved patterns with similarity/relevance scores
            activation_signature: Current working memory activations
            
        Returns:
            Patterns weighted by: relevance + success_boost + activation_boost
        """
        weighted = []
        
        for pattern, base_score in patterns:
            # Start with base score (relevance or similarity)
            weight = base_score
            
            # Apply gentle success boost (additive, not multiplicative)
            # This prevents destroying relevance order
            success_boost = (pattern.success_score - 0.5) * 0.2  # -0.1 to +0.1 range
            weight += success_boost
            
            # Apply gentle activation boost
            activation_boost = 0.0
            if activation_signature:
                # Small boost from working memory
                max_activation = max(activation_signature.values())
                activation_boost = max_activation * 0.1  # Small boost
                
            weight += activation_boost
            
            weighted.append((pattern, weight))
        
        # Sort by weight descending (but order should be mostly preserved)
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
                    # Validate the blended response
                    if self._validate_response(blended):
                        response_text = blended
                        fragment_ids.append(secondary_pattern.fragment_id)
                        weights.append(secondary_weight)
                        print(f"  🎨 Blended {primary_pattern.intent} + {secondary_pattern.intent}")
                    else:
                        # Blend failed validation, use primary only
                        print(f"  ⚠️  Blend rejected (validation failed), using primary only")
                        should_blend = False
        
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
        
        Strategy: Check compatibility first, then use BNN syntax stage if available
        for grammatically-guided blending. This creates NEW utterances from learned fragments!
        
        Args:
            primary: Main pattern
            secondary: Supporting pattern
            
        Returns:
            Blended text or None if blending not appropriate
        """
        # Check if patterns are compatible for blending
        if not self._should_blend(primary, secondary):
            return None
        
        # If BNN syntax stage is available, use it for intelligent composition!
        if self.syntax_stage:
            return self._blend_with_syntax_bnn(primary, secondary)
        
        # Fallback: Heuristic blending (original method)
        # Don't blend if patterns end with questions (creates awkward double-question)
        if primary.response_text.strip().endswith('?') and secondary.response_text.strip().endswith('?'):
            return None
    
    def _should_blend(self, primary: ResponsePattern, secondary: ResponsePattern) -> bool:
        """
        Check if two patterns are compatible for blending.
        
        Prevents nonsensical combinations like "I'm doing well" + "medical diagnosis"
        
        Args:
            primary: Primary pattern
            secondary: Secondary pattern
            
        Returns:
            True if patterns can be blended coherently
        """
        # Define incompatible intent pairs
        incompatible_pairs = {
            ('greeting', 'technical_explain'),  # "Hello" + technical content
            ('greeting', 'explain'),            # "Hi" + explanation
            ('farewell', 'question_info'),      # "Goodbye" + question
            ('farewell', 'technical_explain'),  # "Bye" + technical
            ('agreement', 'greeting'),          # "Yes" + "Hello"
            ('disagreement', 'greeting'),       # "No" + "Hello"
            ('statement', 'greeting'),          # Statement + greeting is awkward
        }
        
        # Check if this pair is explicitly incompatible
        intent_pair = (primary.intent, secondary.intent)
        intent_pair_reverse = (secondary.intent, primary.intent)
        
        if intent_pair in incompatible_pairs or intent_pair_reverse in incompatible_pairs:
            return False
        
        # Don't blend if both patterns are very short (likely fragments)
        if len(primary.response_text.split()) < 3 and len(secondary.response_text.split()) < 3:
            return False
        
        # Don't blend if combined length would be too long (over 25 words)
        total_words = len(primary.response_text.split()) + len(secondary.response_text.split())
        if total_words > 25:
            return False
        
        # Check semantic compatibility using embeddings if available
        if hasattr(primary, 'embedding_cache') and hasattr(secondary, 'embedding_cache'):
            if primary.embedding_cache and secondary.embedding_cache:
                # Calculate cosine similarity between embeddings
                import numpy as np
                emb1 = np.array(primary.embedding_cache)
                emb2 = np.array(secondary.embedding_cache)
                
                # Normalize
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = np.dot(emb1, emb2) / (norm1 * norm2)
                    
                    # If patterns are too dissimilar semantically, don't blend
                    if similarity < 0.3:  # Threshold for semantic compatibility
                        return False
        
        # Define compatible intent pairs that work well together
        compatible_pairs = {
            ('question_info', 'explain'),       # Question + explanation
            ('question_info', 'technical_explain'),  # Question + technical answer
            ('acknowledgment', 'interest'),     # "I see" + "That's interesting"
            ('interest', 'question_info'),      # "Interesting!" + "How does..."
            ('agreement', 'explain'),           # "Yes" + explanation
            ('agreement', 'technical_explain'), # "Yes" + technical detail
            ('explain', 'technical_explain'),   # Explanation + more detail
            ('statement', 'explain'),           # Statement + elaboration
            ('statement', 'technical_explain'), # Statement + technical detail
        }
        
        # If it's a known good pair, allow it
        if intent_pair in compatible_pairs or intent_pair_reverse in compatible_pairs:
            return True
        
        # Same intent types usually blend well
        if primary.intent == secondary.intent:
            return True
        
        # Default: be conservative, don't blend unless we're sure
        return False
    
    def _validate_response(self, response: str) -> bool:
        """
        Validate a composed response for quality and coherence.
        
        Rejects responses that are:
        - Too short or degenerate
        - Have obvious grammar errors
        - Contain repetitive content
        
        Args:
            response: Composed response text
            
        Returns:
            True if response passes validation
        """
        if not response or not response.strip():
            return False
        
        words = response.split()
        
        # Reject if too short (less than 3 words, likely degenerate)
        if len(words) < 3:
            return False
        
        # Reject if too long (over 40 words, likely rambling)
        if len(words) > 40:
            return False
        
        # Check for repetitive content (same word repeated)
        # "Yes, Yes." or "Hello. Hello!" or "I'm doing well, I'm doing well"
        if len(words) >= 2:
            # Check for immediate repetition
            for i in range(len(words) - 1):
                word1 = words[i].strip('.,!?').lower()
                word2 = words[i + 1].strip('.,!?').lower()
                if word1 == word2 and len(word1) > 2:  # Allow "a a" but not "hello hello"
                    return False
            
            # Check for phrase repetition across sentences
            # Split by punctuation and check if phrases repeat
            parts = [p.strip() for p in response.replace('!', '.').replace('?', '.').replace(',', '.').split('.') if p.strip()]
            if len(parts) >= 2:
                # Check if any two parts are identical or very similar
                for i in range(len(parts)):
                    for j in range(i + 1, len(parts)):
                        # Compare normalized versions (lowercase, no punctuation)
                        norm_i = parts[i].lower().strip('.,!?\'\"')
                        norm_j = parts[j].lower().strip('.,!?\'\"')
                        if norm_i == norm_j:
                            return False
                        # Also check if one is a substring of the other (high overlap)
                        if len(norm_i) > 5 and len(norm_j) > 5:
                            if norm_i in norm_j or norm_j in norm_i:
                                return False
        
        # Check for obvious sentence fragments
        # If it starts with a lowercase letter and isn't a continuation word
        if response[0].islower() and not response.startswith(('and', 'but', 'or', 'so')):
            return False
        
        # Check for multiple sentences - first should end with punctuation
        sentences = [s.strip() for s in response.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        if len(sentences) > 1:
            # Each sentence should start with capital
            for sentence in sentences:
                if sentence and not sentence[0].isupper():
                    return False
        
        # Check for obvious grammar errors - multiple punctuation
        if '..' in response or '!!' in response or '??' in response:
            return False
        
        # Check for mismatched quotes or parentheses
        if response.count('"') % 2 != 0:
            return False
        if response.count('(') != response.count(')'):
            return False
        
        # All checks passed
        return True
            
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
    
    def _fallback_response_low_confidence(
        self, 
        user_input: str,
        best_pattern: ResponsePattern,
        best_score: float
    ) -> ComposedResponse:
        """
        Graceful fallback when best pattern has low relevance to user query.
        
        Instead of hallucinating or giving off-topic responses, we acknowledge
        limitation and optionally ask for clarification.
        
        Args:
            user_input: User's query
            best_pattern: Best pattern found (but with low relevance)
            best_score: Relevance score (below threshold)
        
        Returns:
            Graceful fallback response
        """
        # Analyze query to provide contextual fallback
        user_lower = user_input.lower()
        
        # Check for question markers
        is_question = any(q in user_lower for q in ['what', 'how', 'why', 'when', 'where', 'who', 'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does', '?'])
        
        # Check for technical terms
        is_technical = any(tech in user_lower for tech in ['neural', 'network', 'algorithm', 'learning', 'model', 'training', 'data', 'transformer', 'cnn', 'rnn', 'gan', 'attention'])
        
        # Generate contextual fallback
        if is_question and is_technical:
            fallback_text = "I don't have specific information about that topic yet. Could you ask about something related to what we've discussed, or rephrase your question?"
        elif is_question:
            fallback_text = "I'm not sure how to answer that. Could you rephrase or ask about something else?"
        elif is_technical:
            fallback_text = "That's an interesting topic, but I don't have enough information about it in my learned patterns. Could we discuss something related?"
        else:
            fallback_text = "I'm not quite sure how to respond to that. Could you elaborate or try rephrasing?"
        
        print(f"     ⚠️  Low confidence ({best_score:.3f}) - using graceful fallback")
        
        return ComposedResponse(
            text=fallback_text,
            fragment_ids=["low_confidence_fallback"],
            composition_weights=[1.0],
            coherence_score=0.3,
            primary_pattern=None
        )

