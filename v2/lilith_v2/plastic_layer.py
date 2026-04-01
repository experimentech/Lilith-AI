"""
Plastic Layer - Base class for learnable processing layers.

This replaces hardcoded if/else logic with learned transformations.
Each layer:
  - Takes activation from the layer below (afferent)
  - Transforms through learned weights
  - Passes activation to the layer above (efferent)
  - Learns from feedback flowing back down

The key insight: what was hardcoded (stop words, patterns, templates)
becomes emergent from learned weights.

Architecture:
    Input → SalienceLayer → PatternLayer → SemanticLayer → IntentLayer → SynthesisLayer → Output
              ↓ learn         ↓ learn        ↓ learn         ↓ learn        ↓ learn
            (words that     (structures    (meanings        (pragmatic     (response
             matter)         that recur)    that relate)     purposes)      patterns)
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LayerActivation:
    """Activation state passed between layers."""
    
    # The raw data (text, tokens, embeddings depending on layer)
    data: Any
    
    # Activation strengths for each element (learned salience)
    activations: torch.Tensor
    
    # Confidence/certainty of this layer's processing
    confidence: float = 1.0
    
    # Provenance: which layer produced this
    source_layer: str = ""
    
    # Context accumulated through the pipeline
    context: Dict[str, Any] = field(default_factory=dict)
    
    def filter_by_activation(self, threshold: float = 0.1) -> 'LayerActivation':
        """Return only elements with activation above threshold."""
        if isinstance(self.data, list):
            mask = self.activations > threshold
            filtered_data = [d for d, m in zip(self.data, mask.tolist()) if m]
            filtered_acts = self.activations[mask]
            return LayerActivation(
                data=filtered_data,
                activations=filtered_acts,
                confidence=self.confidence,
                source_layer=self.source_layer,
                context=self.context.copy()
            )
        return self


@dataclass  
class LayerFeedback:
    """Feedback signal flowing back down through layers."""
    
    # Was the output successful? (from user or downstream)
    success: float  # -1.0 to 1.0
    
    # What specifically was good/bad?
    target_indices: Optional[List[int]] = None
    
    # Gradient-like signal for weight updates
    gradient: Optional[torch.Tensor] = None
    
    # Which layer provided this feedback
    source_layer: str = ""


class PlasticLayer(ABC):
    """
    Base class for plastic (learnable) processing layers.
    
    Replaces hardcoded processing with learned transformations.
    Each layer maintains:
      - weights: The learned transformation parameters
      - biases: Per-element baseline activations  
      - learning_rate: How fast to adapt
    """
    
    def __init__(
        self,
        layer_id: str,
        input_dim: int,
        output_dim: int,
        learning_rate: float = 0.01,
        device: str = "cpu"
    ):
        self.id = layer_id
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.device = device
        
        # Core learnable parameters
        self.weights = torch.randn(output_dim, input_dim, device=device) * 0.1
        self.biases = torch.zeros(output_dim, device=device)
        
        # Activation history for Hebbian learning
        self._last_input: Optional[torch.Tensor] = None
        self._last_output: Optional[torch.Tensor] = None
        
        # Statistics
        self._forward_count = 0
        self._learn_count = 0
        
    @abstractmethod
    def forward(self, activation: LayerActivation) -> LayerActivation:
        """
        Transform input activation to output activation.
        
        This is where the layer-specific processing happens.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def backward(self, feedback: LayerFeedback) -> LayerFeedback:
        """
        Process feedback and update weights.
        
        Returns transformed feedback for the layer below.
        """
        pass
    
    def _hebbian_update(
        self,
        pre: torch.Tensor,  # Input activation
        post: torch.Tensor,  # Output activation  
        reward: float = 1.0  # Modulation signal
    ) -> None:
        """
        Hebbian learning: strengthen connections that fire together.
        
        ΔW = η * reward * post ⊗ pre
        
        This is the core plasticity mechanism - connections that
        co-activate get strengthened, especially when rewarded.
        """
        if pre.dim() == 1:
            pre = pre.unsqueeze(0)
        if post.dim() == 1:
            post = post.unsqueeze(0)
            
        # Outer product gives connection strength change
        delta = self.learning_rate * reward * torch.outer(post.squeeze(), pre.squeeze())
        
        # Normalize to prevent runaway growth
        if delta.norm() > 0:
            delta = delta / (delta.norm() + 1e-8)
            
        # Apply update with decay (homeostatic plasticity)
        self.weights = 0.999 * self.weights + delta[:self.output_dim, :self.input_dim]
        
        self._learn_count += 1
        
    def save_state(self, path: Path) -> None:
        """Save layer weights to disk."""
        state = {
            'weights': self.weights.cpu().numpy(),
            'biases': self.biases.cpu().numpy(),
            'forward_count': self._forward_count,
            'learn_count': self._learn_count,
        }
        torch.save(state, path)
        
    def load_state(self, path: Path) -> None:
        """Load layer weights from disk."""
        if path.exists():
            state = torch.load(path, map_location=self.device)
            self.weights = torch.tensor(state['weights'], device=self.device)
            self.biases = torch.tensor(state['biases'], device=self.device)
            self._forward_count = state.get('forward_count', 0)
            self._learn_count = state.get('learn_count', 0)
            
    def stats(self) -> Dict[str, Any]:
        """Return layer statistics."""
        return {
            'id': self.id,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'forward_count': self._forward_count,
            'learn_count': self._learn_count,
            'weight_norm': float(self.weights.norm()),
            'weight_mean': float(self.weights.mean()),
            'weight_std': float(self.weights.std()),
        }


class SalienceLayer(PlasticLayer):
    """
    Layer 1: Salience - Which words/tokens matter?
    
    Replaces hardcoded stop word lists with learned salience weights.
    Words that consistently appear in meaningful contexts get high salience.
    Function words (the, is, a) naturally get low salience through learning.
    
    Input: List of tokens
    Output: Tokens with activation weights (salience scores)
    """
    
    def __init__(
        self,
        layer_id: str = "layer.salience",
        vocab_size: int = 10000,
        embedding_dim: int = 64,
        learning_rate: float = 0.01,
        device: str = "cpu"
    ):
        super().__init__(layer_id, embedding_dim, 1, learning_rate, device)
        
        # Per-word salience scores (learned, not hardcoded)
        self.word_salience = torch.zeros(vocab_size, device=device)
        
        # Word to index mapping (grows dynamically)
        self.word_to_idx: Dict[str, int] = {}
        self._next_idx = 0
        
        # Bootstrap: common function words start with low salience
        # But these can be overridden by learning!
        self._bootstrap_low_salience = {
            # Articles/determiners
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "of", "to", "in", "for", "on", "with", "at", "by", "from",
            "it", "this", "that", "these", "those",
            # Question words
            "what", "when", "where", "which", "who", "whose", "whom",
            "how", "why", "whether", "if",
            # Pronouns
            "i", "me", "my", "mine", "you", "your", "yours", 
            "he", "him", "his", "she", "her", "hers",
            "we", "us", "our", "ours", "they", "them", "their", "theirs",
            # Auxiliaries
            "do", "does", "did", "have", "has", "had", "can", "could",
            "will", "would", "shall", "should", "may", "might", "must",
            # Common function verbs (not topics)
            "know", "think", "find", "tell", "make", "take", "give",
            "want", "need", "like", "feel", "see", "look", "seem",
            "get", "got", "going", "come", "came", "went", "been",
            # Other function words
            "about", "just", "only", "also", "very", "really", "too",
            "yes", "no", "not", "okay", "please", "thank", "sorry",
        }
        
    def _get_word_idx(self, word: str) -> int:
        """Get or create index for a word."""
        word_lower = word.lower()
        if word_lower not in self.word_to_idx:
            self.word_to_idx[word_lower] = self._next_idx
            # Initialize with bootstrap bias (but learnable!)
            if word_lower in self._bootstrap_low_salience:
                self.word_salience[self._next_idx] = -2.0  # Start very low (sigmoid ≈ 0.12)
            else:
                self.word_salience[self._next_idx] = 1.0  # Start salient (sigmoid ≈ 0.73)
            self._next_idx += 1
        return self.word_to_idx[word_lower]
    
    def forward(self, activation: LayerActivation) -> LayerActivation:
        """
        Compute salience for each input token.
        
        High salience = this word matters for meaning.
        Low salience = function word, can be filtered.
        """
        self._forward_count += 1
        
        tokens = activation.data
        if isinstance(tokens, str):
            tokens = tokens.split()
            
        # Get salience for each token
        indices = [self._get_word_idx(t) for t in tokens]
        saliences = torch.sigmoid(self.word_salience[indices])
        
        # Store for learning
        self._last_input = activation.activations if activation.activations is not None else torch.ones(len(tokens))
        self._last_output = saliences
        self._last_indices = indices
        
        return LayerActivation(
            data=tokens,
            activations=saliences,
            confidence=activation.confidence,
            source_layer=self.id,
            context={**activation.context, 'token_indices': indices}
        )
    
    def backward(self, feedback: LayerFeedback) -> LayerFeedback:
        """
        Update salience weights based on feedback.
        
        If a response using certain words was successful, 
        increase salience for those words.
        """
        if self._last_indices is None:
            return feedback
            
        # Update salience for words that were used
        for idx in self._last_indices:
            # Hebbian: if word was active and outcome was good, increase salience
            self.word_salience[idx] += self.learning_rate * feedback.success
            
        self._learn_count += 1
        return feedback
    
    def get_salient_words(self, threshold: float = 0.5) -> List[str]:
        """Get words with salience above threshold."""
        salient = []
        for word, idx in self.word_to_idx.items():
            if torch.sigmoid(self.word_salience[idx]) > threshold:
                salient.append(word)
        return salient


class PatternLayer(PlasticLayer):
    """
    Layer 2: Pattern Recognition - Which structures recur?
    
    Replaces hardcoded regex patterns with learned pattern templates.
    Patterns like "X is a Y" emerge from seeing many examples,
    not from manually writing regex.
    
    Input: Tokens with salience
    Output: Recognized structural patterns with confidence
    """
    
    def __init__(
        self,
        layer_id: str = "layer.pattern",
        embedding_dim: int = 64,
        num_patterns: int = 100,
        learning_rate: float = 0.01,
        device: str = "cpu"
    ):
        super().__init__(layer_id, embedding_dim, num_patterns, learning_rate, device)
        
        # Pattern templates as embedding sequences
        # Each pattern is a sequence of slot embeddings
        self.pattern_embeddings = torch.randn(num_patterns, 5, embedding_dim, device=device) * 0.1
        
        # Pattern activation counts (for pruning unused patterns)
        self.pattern_usage = torch.zeros(num_patterns, device=device)
        
        # Named patterns (learned associations)
        self.pattern_names: Dict[int, str] = {}
        
    def forward(self, activation: LayerActivation) -> LayerActivation:
        """
        Match input against learned patterns.
        
        Returns pattern activations - which structural templates
        best match the input?
        """
        self._forward_count += 1
        
        # For now, pass through with pattern matching placeholder
        # Real implementation would do sequence alignment
        
        tokens = activation.data
        saliences = activation.activations
        
        # Compute pattern match scores (simplified)
        # In full implementation: attention over token embeddings vs pattern slots
        pattern_scores = torch.zeros(self.output_dim, device=self.device)
        
        # Store for learning
        self._last_input = saliences
        self._last_output = pattern_scores
        self._last_tokens = tokens
        
        return LayerActivation(
            data={'tokens': tokens, 'patterns': pattern_scores},
            activations=pattern_scores,
            confidence=activation.confidence * 0.9,  # Slight confidence reduction
            source_layer=self.id,
            context={**activation.context, 'saliences': saliences}
        )
    
    def backward(self, feedback: LayerFeedback) -> LayerFeedback:
        """
        Strengthen patterns that led to good outcomes.
        """
        if self._last_tokens is None:
            return feedback
            
        # TODO: Update pattern embeddings based on successful usage
        self._learn_count += 1
        return feedback
    
    def learn_pattern(
        self,
        tokens: List[str],
        pattern_name: str,
        token_embeddings: torch.Tensor
    ) -> int:
        """
        Learn a new pattern from example tokens.
        
        This is called when we explicitly want to teach a pattern.
        Returns the pattern index.
        """
        # Find least-used pattern slot
        min_idx = int(torch.argmin(self.pattern_usage))
        
        # Store pattern embedding (simplified: just use first 5 tokens)
        if len(token_embeddings) >= 5:
            self.pattern_embeddings[min_idx] = token_embeddings[:5]
        else:
            self.pattern_embeddings[min_idx, :len(token_embeddings)] = token_embeddings
            
        self.pattern_names[min_idx] = pattern_name
        self.pattern_usage[min_idx] = 1.0
        
        return min_idx


class IntentLayer(PlasticLayer):
    """
    Layer 3: Intent/Pragmatics - What is the purpose?
    
    Replaces hardcoded intent detection (if "?" in text)
    with learned intent regions in activation space.
    
    Input: Pattern activations
    Output: Intent activations (question, statement, command, etc.)
    """
    
    # Intent prototypes (learned, not hardcoded labels)
    INTENT_SLOTS = 32  # Can learn up to 32 distinct intent types
    
    def __init__(
        self,
        layer_id: str = "layer.intent",
        pattern_dim: int = 100,
        learning_rate: float = 0.01,
        device: str = "cpu"
    ):
        super().__init__(layer_id, pattern_dim, self.INTENT_SLOTS, learning_rate, device)
        
        # Intent prototypes in pattern space
        self.intent_prototypes = torch.randn(self.INTENT_SLOTS, pattern_dim, device=device) * 0.1
        
        # Intent names (emerge from learning)
        self.intent_names: Dict[int, str] = {}
        
        # Bootstrap a few basic intents (but they're still learnable!)
        self._bootstrap_intents = {
            0: "question",
            1: "statement", 
            2: "command",
            3: "teaching",
            4: "feedback_positive",
            5: "feedback_negative",
            6: "greeting",
            7: "farewell",
        }
        self.intent_names.update(self._bootstrap_intents)
        
    def forward(self, activation: LayerActivation) -> LayerActivation:
        """
        Detect intent from pattern activations.
        """
        self._forward_count += 1
        
        pattern_acts = activation.activations
        
        # Compute similarity to each intent prototype
        # (learned, not rule-based)
        if pattern_acts.dim() == 1:
            pattern_acts = pattern_acts.unsqueeze(0)
            
        # Cosine similarity to prototypes
        pattern_norm = pattern_acts / (pattern_acts.norm(dim=-1, keepdim=True) + 1e-8)
        proto_norm = self.intent_prototypes / (self.intent_prototypes.norm(dim=-1, keepdim=True) + 1e-8)
        
        intent_scores = torch.matmul(pattern_norm, proto_norm.T).squeeze()
        intent_probs = torch.softmax(intent_scores, dim=-1)
        
        # Get top intent
        top_intent = int(torch.argmax(intent_probs))
        top_name = self.intent_names.get(top_intent, f"intent_{top_intent}")
        
        self._last_input = pattern_acts
        self._last_output = intent_probs
        
        return LayerActivation(
            data={
                'patterns': activation.data,
                'intent': top_name,
                'intent_idx': top_intent,
                'intent_probs': intent_probs,
            },
            activations=intent_probs,
            confidence=float(intent_probs[top_intent]),
            source_layer=self.id,
            context=activation.context
        )
    
    def backward(self, feedback: LayerFeedback) -> LayerFeedback:
        """
        Update intent prototypes based on feedback.
        
        If we correctly identified intent (good outcome), 
        move prototype toward the pattern that triggered it.
        """
        if self._last_input is None or self._last_output is None:
            return feedback
            
        # Get the intent that was activated
        top_intent = int(torch.argmax(self._last_output))
        
        # Hebbian: move prototype toward input pattern if successful
        if feedback.success > 0:
            self.intent_prototypes[top_intent] += (
                self.learning_rate * feedback.success * 
                (self._last_input.squeeze() - self.intent_prototypes[top_intent])
            )
            
        self._learn_count += 1
        return feedback
    
    def learn_intent(self, pattern_activation: torch.Tensor, intent_name: str) -> int:
        """
        Explicitly teach a new intent type.
        """
        # Find empty or least-used slot
        for i in range(self.INTENT_SLOTS):
            if i not in self.intent_names or i >= 8:  # Keep bootstrap intents
                self.intent_names[i] = intent_name
                self.intent_prototypes[i] = pattern_activation.squeeze()
                return i
        return -1


class LayerStack:
    """
    A stack of plastic layers that process input through the pipeline.
    
    Manages:
      - Forward pass: input → layer1 → layer2 → ... → output
      - Backward pass: feedback flows back, updating weights
      - Persistence: save/load the whole stack
    """
    
    def __init__(self, layers: List[PlasticLayer]):
        self.layers = layers
        self._layer_by_id = {layer.id: layer for layer in layers}
        
    def forward(self, input_data: Any) -> LayerActivation:
        """Process input through all layers."""
        # Create initial activation
        if isinstance(input_data, LayerActivation):
            activation = input_data
        else:
            if isinstance(input_data, str):
                tokens = input_data.split()
            else:
                tokens = input_data
            activation = LayerActivation(
                data=tokens,
                activations=torch.ones(len(tokens) if isinstance(tokens, list) else 1),
                confidence=1.0,
                source_layer="input"
            )
        
        # Forward through each layer
        for layer in self.layers:
            activation = layer.forward(activation)
            
        return activation
    
    def backward(self, feedback: LayerFeedback) -> None:
        """Propagate feedback back through all layers."""
        # Backward through layers in reverse order
        for layer in reversed(self.layers):
            feedback = layer.backward(feedback)
            
    def learn(self, input_data: Any, success: float) -> None:
        """
        Complete learning cycle: forward then backward.
        
        This is the main training method.
        """
        # Forward pass (stores activations)
        _ = self.forward(input_data)
        
        # Backward pass (updates weights)
        feedback = LayerFeedback(success=success, source_layer="external")
        self.backward(feedback)
        
    def save_state(self, directory: Path) -> None:
        """Save all layer states."""
        directory.mkdir(parents=True, exist_ok=True)
        for layer in self.layers:
            layer.save_state(directory / f"{layer.id}.pt")
            
    def load_state(self, directory: Path) -> None:
        """Load all layer states."""
        for layer in self.layers:
            layer.load_state(directory / f"{layer.id}.pt")
            
    def stats(self) -> Dict[str, Any]:
        """Get stats for all layers."""
        return {layer.id: layer.stats() for layer in self.layers}


# Factory function to create the standard layer stack
def create_standard_stack(
    vocab_size: int = 10000,
    embedding_dim: int = 64,
    device: str = "cpu"
) -> LayerStack:
    """
    Create the standard plastic layer stack.
    
    Layer 1: Salience (which words matter)
    Layer 2: Pattern (which structures recur)  
    Layer 3: Intent (what is the purpose)
    
    More layers can be added for:
    - Semantic association
    - Response synthesis
    """
    return LayerStack([
        SalienceLayer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            device=device
        ),
        PatternLayer(
            embedding_dim=embedding_dim,
            device=device
        ),
        IntentLayer(
            device=device
        ),
    ])
