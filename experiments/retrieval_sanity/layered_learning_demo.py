"""
Multi-Layer Learning Architecture

Demonstrates how the SAME learning mechanism (pattern extraction + plasticity)
operates at different cognitive levels:

LAYER 1: WORDS (Intake Stage)
- What: Character sequences → normalized tokens
- Learns: Typos, slang, abbreviations
- Pattern: "teh" → "the", "lol" → "laughing out loud"
- Plasticity: BNN learns common input variations

LAYER 2: SYNTAX (Syntax Stage)  
- What: Token sequences → grammatical structures
- Learns: POS patterns, phrase templates
- Pattern: "DT JJ NN" → "The red apple"
- Plasticity: BNN learns valid syntactic combinations

LAYER 3: SEMANTICS (Semantic Stage)
- What: Word sequences → concept embeddings
- Learns: Meanings, relationships, context
- Pattern: "king - man + woman" → "queen"
- Plasticity: BNN adjusts concept clusters

LAYER 4: PRAGMATICS (Response Stage)
- What: Concepts → conversational responses
- Learns: Dialogue patterns, social context
- Pattern: "How are you?" → "I'm fine, thanks"
- Plasticity: Success scores update based on outcomes

Each layer:
1. Has dedicated PMFlow BNN encoder
2. Stores learned patterns in separate DB namespace
3. Uses SAME learning algorithm (extract pattern + plasticity update)
4. Passes structured artifacts to next layer

This is EXACTLY what brains do - same learning mechanism (hebbian plasticity)
operating at different levels of abstraction!
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class UnifiedLearningPattern:
    """
    Universal pattern structure that works at ANY cognitive layer.
    
    Same structure whether learning:
    - Typo corrections (intake)
    - Grammar rules (syntax)  
    - Word meanings (semantic)
    - Conversation flows (pragmatic)
    """
    
    pattern_id: str
    layer: str              # intake, syntax, semantic, pragmatic
    
    # Core pattern structure (universal)
    trigger: str            # Input that activates this pattern
    response: str           # Output when pattern fires
    
    # Learning metadata (universal)
    success_score: float    # How well has this worked? (0.0-1.0)
    usage_count: int        # How often used?
    confidence: float       # How certain are we? (0.0-1.0)
    
    # Layer-specific extensions
    metadata: Dict[str, Any]  # POS tags, embeddings, etc.


class GeneralPurposeLearner:
    """
    Single learning algorithm that works at ALL cognitive layers.
    
    Key insight: Learning is just pattern extraction + reinforcement.
    Same algorithm, different representations!
    """
    
    def __init__(self, layer_name: str, learning_mode: str = "moderate"):
        self.layer = layer_name
        self.learning_mode = learning_mode
        self.patterns: List[UnifiedLearningPattern] = []
        
    def observe_interaction(
        self,
        input_representation: str,
        output_representation: str,
        success_signal: float
    ):
        """
        Universal learning function.
        
        Works for:
        - Intake: observe("teh", "the", 1.0) → learn typo correction
        - Syntax: observe("DT NN", "Det Noun phrase", 0.9) → learn grammar
        - Semantic: observe("fruit", "apple/orange/banana", 0.8) → learn category
        - Pragmatic: observe("how are you", "i'm fine thanks", 0.7) → learn response
        """
        
        # Extract pattern (same algorithm for all layers!)
        pattern = UnifiedLearningPattern(
            pattern_id=f"{self.layer}_{len(self.patterns)}",
            layer=self.layer,
            trigger=input_representation,
            response=output_representation,
            success_score=success_signal,
            usage_count=1,
            confidence=success_signal,
            metadata={}
        )
        
        # Store pattern
        self.patterns.append(pattern)
        
        # Plasticity update (same mechanism for all layers!)
        self._update_neural_weights(pattern, success_signal)
        
    def _update_neural_weights(self, pattern: UnifiedLearningPattern, signal: float):
        """
        Hebbian plasticity: "Neurons that fire together, wire together"
        
        Same algorithm whether learning:
        - Character-level patterns (intake)
        - Word-level patterns (syntax)
        - Concept-level patterns (semantic)
        - Response-level patterns (pragmatic)
        """
        # In real implementation, this updates PMFlow BNN weights
        # For now, just update success score
        learning_rate = 0.1
        pattern.success_score += learning_rate * (signal - pattern.success_score)
        
    def retrieve_pattern(self, query: str, threshold: float = 0.5) -> UnifiedLearningPattern:
        """
        Universal retrieval (same for all layers).
        
        Match query to stored patterns, return best match.
        """
        best_match = None
        best_score = 0.0
        
        for pattern in self.patterns:
            # Simple keyword matching (in real system, use BNN embeddings)
            score = self._similarity(query, pattern.trigger)
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = pattern
                
        return best_match
    
    def _similarity(self, a: str, b: str) -> float:
        """Simple similarity (real system uses BNN embeddings)."""
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        if not a_words or not b_words:
            return 0.0
        return len(a_words & b_words) / len(a_words | b_words)


# ============================================================================
# DEMONSTRATION: Same learner at different cognitive layers
# ============================================================================

def demonstrate_layered_learning():
    """Show how same learning mechanism works at all layers."""
    
    print("=" * 80)
    print("UNIFIED LEARNING ACROSS COGNITIVE LAYERS")
    print("=" * 80)
    print()
    
    # LAYER 1: Intake (character-level)
    print("LAYER 1: INTAKE - Learning typo corrections")
    print("-" * 80)
    intake = GeneralPurposeLearner("intake", "eager")
    intake.observe_interaction("teh", "the", 1.0)
    intake.observe_interaction("recieve", "receive", 1.0)
    
    test = intake.retrieve_pattern("teh")
    print(f"Input: 'teh' → Corrected: '{test.response}'")
    print()
    
    # LAYER 2: Syntax (word-level patterns)
    print("LAYER 2: SYNTAX - Learning grammatical structures")
    print("-" * 80)
    syntax = GeneralPurposeLearner("syntax", "eager")
    syntax.observe_interaction("DT JJ NN", "Det Adj Noun → Noun Phrase", 0.9)
    syntax.observe_interaction("VB DT NN", "Verb Det Noun → Verb Phrase", 0.9)
    
    test = syntax.retrieve_pattern("DT JJ NN")
    print(f"Pattern: 'DT JJ NN' → Structure: '{test.response}'")
    print()
    
    # LAYER 3: Semantic (concept-level)
    print("LAYER 3: SEMANTIC - Learning word meanings")
    print("-" * 80)
    semantic = GeneralPurposeLearner("semantic", "eager")
    semantic.observe_interaction("apple", "red or green fruit", 0.8)
    semantic.observe_interaction("banana", "yellow curved fruit", 0.8)
    
    test = semantic.retrieve_pattern("apple")
    print(f"Concept: 'apple' → Meaning: '{test.response}'")
    print()
    
    # LAYER 4: Pragmatic (conversational-level)
    print("LAYER 4: PRAGMATIC - Learning conversation patterns")
    print("-" * 80)
    pragmatic = GeneralPurposeLearner("pragmatic", "eager")
    pragmatic.observe_interaction("how are you", "i'm fine thanks how about you", 0.7)
    pragmatic.observe_interaction("what's your favorite movie", "i love superbad", 0.7)
    
    test = pragmatic.retrieve_pattern("how are you")
    print(f"Query: 'how are you' → Response: '{test.response}'")
    print()
    
    print("=" * 80)
    print("KEY INSIGHT:")
    print("=" * 80)
    print("The SAME learning algorithm (observe → extract → reinforce) works at")
    print("ALL cognitive layers. Only the INPUT/OUTPUT REPRESENTATIONS change!")
    print()
    print("Layer 1: Characters → Tokens")
    print("Layer 2: Tokens → Grammar structures")
    print("Layer 3: Words → Semantic concepts")
    print("Layer 4: Concepts → Conversational responses")
    print()
    print("This is exactly how biological brains work - same plasticity mechanism,")
    print("different levels of abstraction. General purpose learning!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_layered_learning()
