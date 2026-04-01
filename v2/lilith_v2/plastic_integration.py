"""
Plastic Processing Integration

Bridges the plastic layer system with CognitiveStage.
Provides drop-in replacements for hardcoded processing.

V1 ARCHITECTURE PRINCIPLE:
Neural networks stay lightweight - they determine HOW to process.
Databases store WHAT was learned - facts, salience, patterns.
This gives infinite scalability at the cost of lookup speed.

The key insight: we don't rip out all the static code at once.
Instead, we gradually replace each hardcoded component with
calls to the plastic layer stack, which learns from usage.

Migration Strategy:
1. Create plastic layer equivalents that shadow static behavior
2. Route through plastic layers, falling back to static if needed
3. Collect training data from successful interactions
4. Gradually reduce reliance on static fallbacks as layers learn
"""

import torch
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from .plastic_layer import (
    LayerStack, LayerActivation, LayerFeedback,
    SalienceLayer, PatternLayer, IntentLayer,
    create_standard_stack
)
from .db_backed_layers import DBBackedLayerStack, DBSalienceLayer

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result from plastic processing pipeline."""
    
    # Filtered tokens (high salience only)
    salient_tokens: List[str]
    
    # Token salience scores
    saliences: torch.Tensor
    
    # Detected intent
    intent: str
    intent_confidence: float
    
    # Pattern matches (if any)
    patterns: List[str]
    
    # Full layer activation chain
    final_activation: LayerActivation
    
    # Did we use plastic processing or fall back to static?
    used_plastic: bool = True


class PlasticProcessor:
    """
    Plastic processing system that replaces hardcoded logic.
    
    This class provides the same API as the static processing in CognitiveStage
    but routes through learnable plastic layers.
    
    INTEGRATION POINTS:
    - encoder: Uses SemanticPMFlowEncoder for word embeddings (shared vocabulary)
    - graph_store: Persists learned patterns as graph edges
    - pmflow: Modulates activations based on field dynamics
    
    Usage:
        processor = PlasticProcessor(
            encoder=semantic_encoder,
            graph_store=graph,
            data_dir="/path/to/layers"
        )
        
        # Process input (replaces hardcoded stop words, patterns, etc.)
        result = processor.process("Do you know what a dog is?")
        
        # result.salient_tokens = ["dog"]  (not ["know", "dog"])
        # result.intent = "question"
        
        # Learn from feedback
        processor.learn_from_feedback(success=1.0)
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        vocab_size: int = 10000,
        embedding_dim: int = 64,
        device: str = "cpu",
        use_fallback: bool = True,  # Fall back to static if layers untrained
        encoder: Any = None,  # SemanticPMFlowEncoder for shared embeddings
        graph_store: Any = None,  # Graph store for DB-backed layers
        pmflow: Any = None,  # PMField for activation modulation
        use_db_backed: bool = True,  # Use DB-backed layers (v1 principle)
    ):
        self.data_dir = Path(data_dir) if data_dir else None
        self.device = device
        self.use_fallback = use_fallback
        
        # Integration with existing systems
        self.encoder = encoder  # Shared semantic encoder
        self.graph_store = graph_store  # Knowledge graph persistence
        self.pmflow = pmflow  # Physics-based dynamics
        self.use_db_backed = use_db_backed and graph_store is not None
        
        # Match embedding dimension to encoder if provided
        if encoder and hasattr(encoder, 'dimension'):
            embedding_dim = encoder.dimension
        elif encoder and hasattr(encoder, 'base_encoder'):
            embedding_dim = getattr(encoder.base_encoder, 'dimension', embedding_dim)
        
        # === V1 ARCHITECTURE ===
        # If graph_store provided, use DB-backed layers (stores WHAT in DB)
        # Otherwise fall back to tensor-based layers for compatibility
        if self.use_db_backed:
            self.stack = DBBackedLayerStack(graph_store=graph_store)
            self._salience_layer = self.stack.salience
            self._intent_layer = self.stack.intent
            logger.info("Using DB-backed layers (v1 architecture: DB stores WHAT)")
        else:
            # Legacy tensor-based layers
            self.stack = create_standard_stack(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                device=device
            )
            self._salience_layer = self.stack.layers[0]
            self._intent_layer = self.stack.layers[2]
            
            # Load saved state if available
            if self.data_dir and self.data_dir.exists():
                self.stack.load_state(self.data_dir)
                logger.info(f"Loaded plastic layer state from {self.data_dir}")
            
        # Cache last processing for learning
        self._last_input: Optional[str] = None
        self._last_result: Optional[ProcessingResult] = None
        self._last_tenant_id: Optional[str] = None
        
        # Training statistics
        self._process_count = 0
        self._learn_count = 0
        self._plastic_used_count = 0
        self._fallback_used_count = 0
        
    def process(self, text: str, tenant_id: Optional[str] = None) -> ProcessingResult:
        """
        Process text through plastic layers.
        
        Returns salient tokens, detected intent, and patterns.
        
        Integration:
        - If encoder is set, uses real semantic embeddings
        - If pmflow is set, modulates activation with field dynamics
        - If DB-backed, queries graph for learned salience (v1 architecture)
        """
        self._process_count += 1
        self._last_input = text
        self._last_tenant_id = tenant_id
        
        tokens = text.split()
        
        # === ENCODER INTEGRATION ===
        # Get real embeddings from the semantic encoder if available
        token_embeddings = None
        if self.encoder:
            try:
                # Encode each token to get semantic embeddings
                token_embeddings = self.encoder.encode(tokens)
                if hasattr(token_embeddings, 'detach'):
                    token_embeddings = token_embeddings.detach()
            except Exception as e:
                logger.debug(f"Encoder failed, using plastic layers only: {e}")
        
        # === V1 ARCHITECTURE: DB-BACKED OR TENSOR-BASED ===
        if self.use_db_backed:
            # Forward through DB-backed stack (queries database for learned info)
            activation = self.stack.forward(text, tenant_id=tenant_id)
            
            # Get salient tokens directly from DB-backed salience layer
            salient_tokens = self._salience_layer.get_salient_words(text, tenant_id=tenant_id)
            saliences = torch.tensor([0.8 if t.lower() in [s.lower() for s in salient_tokens] else 0.2 
                                      for t in tokens])
        else:
            # Forward through tensor-based layer stack
            activation = self.stack.forward(text)
            
            # Get salient tokens from first layer's output
            salience_layer = self.stack.layers[0]
            if hasattr(salience_layer, '_last_output') and salience_layer._last_output is not None:
                saliences = salience_layer._last_output
                tokens = salience_layer._last_tokens if hasattr(salience_layer, '_last_tokens') else text.split()
                
                # Filter to salient tokens
                salient_tokens = []
                for token, sal in zip(tokens, saliences):
                    if sal > 0.5:  # Threshold
                        salient_tokens.append(token)
            else:
                salient_tokens = text.split()
                saliences = torch.ones(len(salient_tokens))
        
        # === PMFLOW INTEGRATION ===
        # Modulate salience with field dynamics if available
        if self.pmflow and token_embeddings is not None:
            try:
                # Evolve embeddings through the field to get activated tokens
                evolved = self.pmflow.evolve(token_embeddings, steps=3)
                # Use magnitude of evolution as additional salience signal
                evolution_magnitude = (evolved - token_embeddings).norm(dim=-1)
                # Normalize and blend with plastic salience
                if evolution_magnitude.numel() > 0:
                    evo_salience = evolution_magnitude / (evolution_magnitude.max() + 1e-8)
                    # Store for blending
                    self._last_pmflow_salience = evo_salience
            except Exception as e:
                logger.debug(f"PMFlow evolution failed: {e}")
        
        # Extract results from final activation
        intent_data = activation.data
        intent = intent_data.get('intent', 'unknown') if isinstance(intent_data, dict) else 'unknown'
        intent_confidence = float(activation.confidence)
            
        # Determine if we're confident enough to use plastic result
        # or need to fall back to static processing
        used_plastic = True
        
        if self.use_fallback and self._process_count < 100:
            # During early training, blend with static processing
            # to bootstrap the system
            if intent_confidence < 0.3:
                # Low confidence - might want to fall back
                used_plastic = False
                self._fallback_used_count += 1
            else:
                self._plastic_used_count += 1
        else:
            self._plastic_used_count += 1
            
        result = ProcessingResult(
            salient_tokens=salient_tokens,
            saliences=saliences,
            intent=intent,
            intent_confidence=intent_confidence,
            patterns=[],  # TODO: extract from pattern layer
            final_activation=activation,
            used_plastic=used_plastic
        )
        
        self._last_result = result
        return result
    
    def is_salient_word(self, word: str, tenant_id: Optional[str] = None) -> bool:
        """
        Check if a word is salient (plastic replacement for stop word check).
        
        This is the drop-in replacement for:
            if word not in stop_words:  # OLD
            if processor.is_salient_word(word):  # NEW
        """
        if self.use_db_backed:
            # Query graph database for salience
            return self._salience_layer.is_salient(word, tenant_id=tenant_id)
        else:
            # Query tensor weights
            idx = self._salience_layer._get_word_idx(word)
            salience = torch.sigmoid(self._salience_layer.word_salience[idx])
            return float(salience) > 0.5
    
    def get_salient_words(self, text: str, threshold: float = 0.5, tenant_id: Optional[str] = None) -> List[str]:
        """
        Get salient words from text (plastic replacement for stop word filtering).
        
        This is the drop-in replacement for:
            [w for w in words if w not in stop_words]  # OLD
            processor.get_salient_words(text)  # NEW
        """
        if self.use_db_backed:
            return self._salience_layer.get_salient_words(text, tenant_id=tenant_id)
        else:
            # Original tensor-based implementation
            tokens = text.lower().split()
            salient = []
            
            for token in tokens:
                clean = token.rstrip("?!.,;:'\"")
                if len(clean) > 2 and self.is_salient_word(clean, tenant_id=tenant_id):
                    salient.append(clean)
                    
            return salient
    
    def detect_intent(self, text: str, tenant_id: Optional[str] = None) -> Tuple[str, float]:
        """
        Detect intent from text (plastic replacement for if/else intent detection).
        
        This is the drop-in replacement for:
            if "?" in text:  # OLD
                intent = "question"
            intent, confidence = processor.detect_intent(text)  # NEW
        """
        result = self.process(text, tenant_id=tenant_id)
        return result.intent, result.intent_confidence
    
    def learn_from_feedback(self, success: float, tenant_id: Optional[str] = None) -> None:
        """
        Learn from feedback on the last processed input.
        
        Called after a response was generated and evaluated.
        success: -1.0 (bad) to 1.0 (good)
        
        V1 Architecture:
        - DB-backed: Updates graph database directly (stores WHAT)
        - Tensor-based: Updates NN weights (legacy)
        
        Integration:
        - Updates layer weights or DB (depending on mode)
        - Stores successful patterns in graph_store (if available)
        - Trains encoder on salient word pairs (if available)
        """
        if self._last_input is None:
            return
        
        # Use stored tenant_id if not provided
        tenant_id = tenant_id or self._last_tenant_id
            
        self._learn_count += 1
        
        # === V1 ARCHITECTURE: DB-BACKED OR TENSOR-BASED ===
        if self.use_db_backed:
            # DB-backed layers: learning updates database
            self.stack.learn(success=success, tenant_id=tenant_id)
        else:
            # Tensor-based: backward pass through layer stack
            feedback = LayerFeedback(success=success, source_layer="user")
            self.stack.backward(feedback)
            
            # === GRAPH INTEGRATION (legacy path) ===
            # Store learned salience relationships if successful
            if success > 0.5 and self.graph_store and self._last_result:
                try:
                    salient = self._last_result.salient_tokens
                    if len(salient) >= 2:
                        # First, ensure nodes exist for each salient word
                        for word in salient:
                            word_clean = word.lower().rstrip("?!.,")
                            self.graph_store.add_node(
                                f"word:{word_clean}", "word", word_clean,
                                confidence=0.6, tenant_id=tenant_id
                            )
                        
                        # Create "co_occurs" edges between salient words
                        for i, word1 in enumerate(salient[:-1]):
                            word2 = salient[i + 1]
                            word1_clean = word1.lower().rstrip("?!.,")
                            word2_clean = word2.lower().rstrip("?!.,")
                            self.graph_store.add_edge(
                                f"word:{word1_clean}", f"word:{word2_clean}",
                                "co_occurs", confidence=0.5 + success * 0.3,
                                tenant_id=tenant_id
                            )
                            logger.debug(f"Learned co-occurrence: {word1_clean} <-> {word2_clean}")
                except Exception as e:
                    logger.debug(f"Graph storage failed: {e}")
        
        # === ENCODER INTEGRATION ===
        # Train encoder on salient word relationships if available
        if success > 0.5 and self.encoder and self._last_result:
            salient = self._last_result.salient_tokens
            if len(salient) >= 2 and hasattr(self.encoder, 'add_words'):
                try:
                    self.encoder.add_words(salient)
                except Exception as e:
                    logger.debug(f"Encoder update failed: {e}")
        
        # Periodic save (only for tensor-based)
        if not self.use_db_backed and self.data_dir and self._learn_count % 100 == 0:
            self.save_state()
            
    def teach_word_salience(self, word: str, is_salient: bool, tenant_id: Optional[str] = None) -> None:
        """
        Explicitly teach that a word is/isn't salient.
        
        This is for direct teaching:
            "The word 'know' is not a topic, it's a verb."
        """
        if self.use_db_backed:
            # DB-backed: store directly in database
            self._salience_layer.teach_salience(word, is_salient, tenant_id=tenant_id)
        else:
            # Tensor-based: update weights
            idx = self._salience_layer._get_word_idx(word)
            target = 1.0 if is_salient else -1.0
            
            # Strong update toward target
            current = float(self._salience_layer.word_salience[idx])
            self._salience_layer.word_salience[idx] = 0.5 * current + 0.5 * target
            
        logger.debug(f"Taught salience: '{word}' -> {is_salient}")
        
    def teach_intent(self, text: str, intent_name: str, tenant_id: Optional[str] = None) -> None:
        """
        Explicitly teach an intent for a text pattern.
        
        This is for direct teaching:
            "When I say 'Can you...', that's a request, not a question about ability."
        """
        if self.use_db_backed:
            # DB-backed: store pattern in database
            self._intent_layer.teach_intent(text, intent_name, tenant_id=tenant_id)
        else:
            # Tensor-based: update prototype weights
            # Process to get pattern activation
            result = self.process(text)
            
            # Get pattern layer activation
            pattern_layer = self.stack.layers[1]
            if hasattr(pattern_layer, '_last_output'):
                pattern_activation = pattern_layer._last_output
                self._intent_layer.learn_intent(pattern_activation, intent_name)
                
        logger.debug(f"Taught intent: '{text[:30]}...' -> {intent_name}")
    
    def save_state(self) -> None:
        """Save layer states to disk."""
        if self.data_dir:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.stack.save_state(self.data_dir)
            logger.info(f"Saved plastic layer state to {self.data_dir}")
            
    def load_state(self) -> None:
        """Load layer states from disk."""
        if self.data_dir and self.data_dir.exists():
            self.stack.load_state(self.data_dir)
            logger.info(f"Loaded plastic layer state from {self.data_dir}")
            
    def stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'process_count': self._process_count,
            'learn_count': self._learn_count,
            'plastic_used': self._plastic_used_count,
            'fallback_used': self._fallback_used_count,
            'plastic_ratio': self._plastic_used_count / max(1, self._process_count),
            'layers': self.stack.stats(),
        }


# Factory function for CognitiveStage integration
def create_plastic_processor_for_stage(
    stage_id: str,
    data_dir: Optional[Path] = None,
    device: str = "cpu",
    encoder: Any = None,
    graph_store: Any = None,
    pmflow: Any = None,
) -> PlasticProcessor:
    """
    Create a PlasticProcessor configured for a CognitiveStage.
    
    This is the recommended way to add plastic processing to a stage.
    
    Args:
        stage_id: Unique identifier for the stage
        data_dir: Where to persist layer states
        device: "cpu" or "cuda"
        encoder: SemanticPMFlowEncoder for shared embeddings
        graph_store: RelationalGraphStore for pattern persistence
        pmflow: PMField for activation modulation
    """
    if data_dir is None:
        data_dir = Path(f"/tmp/lilith_plastic_{stage_id}")
        
    return PlasticProcessor(
        data_dir=data_dir,
        device=device,
        use_fallback=True,  # Safe: falls back during early training
        encoder=encoder,
        graph_store=graph_store,
        pmflow=pmflow,
    )


# Convenience function for gradual migration
def is_salient_word_plastic(
    word: str,
    processor: Optional[PlasticProcessor] = None,
    static_stop_words: Optional[set] = None
) -> bool:
    """
    Check word salience with fallback to static stop words.
    
    This is the transition function for migration:
    
    OLD:
        if word not in stop_words:
            
    TRANSITION:
        if is_salient_word_plastic(word, processor, stop_words):
            
    NEW (after training):
        if processor.is_salient_word(word):
    """
    if processor is not None:
        plastic_result = processor.is_salient_word(word)
        
        # If processor is well-trained, trust it
        if processor._learn_count > 100:
            return plastic_result
            
        # Otherwise, blend with static
        if static_stop_words:
            static_result = word.lower() not in static_stop_words
            # Use plastic if they agree, otherwise prefer static during training
            return plastic_result if plastic_result == static_result else static_result
        return plastic_result
        
    # No processor, fall back to static
    if static_stop_words:
        return word.lower() not in static_stop_words
    return True  # No filtering
