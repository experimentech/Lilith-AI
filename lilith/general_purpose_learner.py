"""
General Purpose Learner - Universal Learning Algorithm for All Cognitive Layers

This is the core learning mechanism that works at EVERY level of abstraction:
- Intake: typo → correction
- Syntax: POS pattern → grammatical structure
- Semantic: word → concept/meaning
- Pragmatic: context → conversational response
- Reasoning: premises → conclusion

Key insight: Same algorithm (observe → evaluate → extract → store → reinforce)
just different input/output representations!

This is biologically inspired - brains use the same Hebbian plasticity mechanism
at all levels (synaptic, neural assembly, cortical column).
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Protocol, runtime_checkable
from abc import ABC, abstractmethod
import numpy as np
import torch
# Avoid circular import at runtime if possible, or use TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .embedding import PMFlowEmbeddingEncoder


@dataclass
class OutcomeSignals:
    """
    Universal outcome signals for learning at any layer.
    
    Different layers can use different evaluation criteria, but all
    produce these standardized signals.
    """
    layer_name: str              # Which cognitive layer
    overall_success: float       # Combined success (-1.0 to 1.0)
    confidence: float            # How confident are we? (0.0 to 1.0)
    
    # Optional layer-specific signals
    layer_signals: Dict[str, float] = None
    
    def __post_init__(self):
        if self.layer_signals is None:
            self.layer_signals = {}

@runtime_checkable
class PatternStore(Protocol):
    """
    Interface for pattern storage.
    
    Different layers can use different storage backends
    (database, memory, etc.) as long as they implement this.
    """
    
    def add_pattern(
        self,
        fragment_id: str,
        trigger_context: str,
        response_text: str,
        intent: str,
        success_score: float
    ) -> str:
        """Store a new learned pattern."""
        ...
    
    def update_success(
        self,
        fragment_id: str,
        feedback: float,
        plasticity_rate: float
    ):
        """Update pattern success score based on feedback."""
        ...


class GeneralPurposeLearner(ABC):
    """
    Universal learning algorithm that works at ANY cognitive layer.
    
    Core mechanism:
    1. OBSERVE: Watch outcomes of layer's operations
    2. EVALUATE: Calculate success signals
    3. EXTRACT: Identify patterns from successful operations
    4. STORE: Save patterns for future retrieval
    5. REINFORCE: Update neural weights based on success
    
    Subclasses specialize by:
    - Defining what counts as "input" and "output" at their layer
    - Implementing layer-specific evaluation criteria
    - Customizing pattern extraction logic
    """
    
    def __init__(
        self,
        layer_name: str,
        pattern_store: PatternStore,
        learning_rate: float = 0.1,
        learning_mode: str = "conservative"
    ):
        """
        Initialize general-purpose learner.
        
        Args:
            layer_name: Which cognitive layer (intake, syntax, semantic, pragmatic)
            pattern_store: Storage backend for learned patterns
            learning_rate: How quickly to adapt (0.0-1.0)
            learning_mode: Learning strictness
                - "conservative": Only learn from highly successful operations
                - "moderate": Learn from reasonably positive outcomes
                - "eager": Learn from most operations (for teaching/debugging)
        """
        self.layer_name = layer_name
        self.pattern_store = pattern_store
        self.learning_rate = learning_rate
        self.learning_mode = learning_mode
        
        # Set thresholds based on learning mode
        if learning_mode == "conservative":
            self.success_threshold = 0.4
            self.confidence_threshold = 0.7
        elif learning_mode == "moderate":
            self.success_threshold = 0.2
            self.confidence_threshold = 0.5
        elif learning_mode == "eager":
            self.success_threshold = 0.0
            self.confidence_threshold = 0.3
        else:
            # Default to conservative
            self.success_threshold = 0.4
            self.confidence_threshold = 0.7
        
        # Track learning progress
        self.interaction_count = 0
        self.success_history: List[float] = []
        self.patterns_learned = 0
    
    def observe_interaction(
        self,
        layer_input: Any,
        layer_output: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> OutcomeSignals:
        """
        Observe outcome of layer operation and learn.
        
        This is the MAIN entry point for learning at any layer.
        
        Args:
            layer_input: Input to this layer's operation
            layer_output: Output produced by this layer
            context: Additional context for evaluation
            
        Returns:
            Outcome signals that were observed
        """
        # 1. EVALUATE: Calculate success signals
        signals = self._evaluate_outcome(layer_input, layer_output, context)
        
        # 2. EXTRACT & STORE: Learn patterns if successful
        if self._should_learn(signals):
            self._extract_and_store_pattern(layer_input, layer_output, signals, context)
        
        # 3. REINFORCE: Update existing patterns
        self._apply_reinforcement(layer_output, signals)
        
        # Track progress
        self.interaction_count += 1
        self.success_history.append(signals.overall_success)
        
        return signals
    
    @abstractmethod
    def _evaluate_outcome(
        self,
        layer_input: Any,
        layer_output: Any,
        context: Optional[Dict[str, Any]]
    ) -> OutcomeSignals:
        """
        Evaluate success of layer operation.
        
        Each layer defines its own success criteria:
        - Intake: Was normalization correct?
        - Syntax: Was grammar valid?
        - Semantic: Did meaning make sense?
        - Pragmatic: Was response appropriate?
        
        Returns:
            OutcomeSignals with success scores
        """
        pass
    
    @abstractmethod
    def _extract_and_store_pattern(
        self,
        layer_input: Any,
        layer_output: Any,
        signals: OutcomeSignals,
        context: Optional[Dict[str, Any]]
    ):
        """
        Extract pattern from successful operation and store it.
        
        Each layer defines what constitutes a "pattern":
        - Intake: typo → correction mapping
        - Syntax: POS sequence → grammatical structure
        - Semantic: word → concept relationship
        - Pragmatic: context → response mapping
        """
        pass
    
    def _should_learn(self, signals: OutcomeSignals) -> bool:
        """
        Decide whether to extract pattern based on signals.
        
        Universal across all layers - only learn from successful operations.
        Thresholds vary by learning_mode.
        """
        should_learn = (
            signals.overall_success >= self.success_threshold and
            signals.confidence >= self.confidence_threshold
        )
        
        # Debug logging for teaching scenarios
        if not should_learn and signals.overall_success > 0.2:
            print(f"  ⚠️  Not learning: success={signals.overall_success:.2f} (need >={self.success_threshold}), "
                  f"confidence={signals.confidence:.2f} (need >={self.confidence_threshold})")
        
        return should_learn
    
    def _apply_reinforcement(
        self,
        layer_output: Any,
        signals: OutcomeSignals
    ):
        """
        Reinforce existing patterns based on outcome.
        
        Universal Hebbian-style plasticity:
        - Successful outcomes strengthen patterns
        - Failed outcomes weaken patterns
        
        Subclasses can override for layer-specific reinforcement.
        """
        # Scale feedback by learning rate
        feedback = signals.overall_success * self.learning_rate
        
        # Update pattern store (if output came from a stored pattern)
        # This is a hook for subclasses to implement
        pass
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get learning statistics.
        
        Universal across all layers.
        
        Returns:
            Dict with learning metrics
        """
        if not self.success_history:
            return {
                "layer": self.layer_name,
                "interaction_count": 0,
                "patterns_learned": 0,
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
            "layer": self.layer_name,
            "interaction_count": self.interaction_count,
            "patterns_learned": self.patterns_learned,
            "average_success": float(avg_success),
            "recent_success": float(recent_success),
            "learning_trend": float(learning_trend),
            "learning_mode": self.learning_mode,
            "success_threshold": self.success_threshold,
            "confidence_threshold": self.confidence_threshold
        }


class LayerSpecificEvaluator(ABC):
    """
    Helper class for layer-specific outcome evaluation.
    
    Each cognitive layer can implement its own evaluator
    with domain-specific success criteria.
    """
    
    @abstractmethod
    def evaluate(
        self,
        layer_input: Any,
        layer_output: Any,
        context: Optional[Dict[str, Any]]
    ) -> OutcomeSignals:
        """Evaluate outcome with layer-specific criteria."""
        pass


class LayerSpecificExtractor(ABC):
    """
    Helper class for layer-specific pattern extraction.
    
    Each cognitive layer can implement its own extractor
    with domain-specific pattern identification logic.
    """
    
    @abstractmethod
    def extract_pattern(
        self,
        layer_input: Any,
        layer_output: Any,
        signals: OutcomeSignals,
        context: Optional[Dict[str, Any]]
    ) -> Tuple[str, str, str, float]:
        """
        Extract pattern from layer operation.
        
        Returns:
            (trigger, response, intent, initial_success)
        """
        pass

class AgenticReasoningLearner(GeneralPurposeLearner):
    """
    Cognitive layer that uses Agentic Physics (PMFlow) for reasoning.
    
    This layer implements 'Thinking' as a physical process:
    1. Intent Injection: Modifying the 'spin' (omegas) of concepts to drag the frame of reference.
    2. Reasoning Chain: A temporal trajectory (geodesic) through the concept manifold.
    3. Mental Effort: The specific action (path length/energy) required to reach a conclusion.
    """
    
    def __init__(self, 
                 pattern_store: PatternStore,
                 encoder, # PMFlowEmbeddingEncoder 
                 learning_rate: float = 0.1):
        super().__init__("reasoning", pattern_store, learning_rate, "moderate")
        self.encoder = encoder
        
    def set_active_will(self, intent_text: str, strength: float = 0.5):
        """
        Inject active will (Intent) into the cognitive field.
        
        This mechanically alters the 'spin' (omegas) of the concepts associated with
        the intent, causing them to create a frame-dragging vortex that biases 
        future reasoning trajectories.
        """
        if not hasattr(self.encoder, 'pm_field'):
            return

        with torch.no_grad():
             # Get activation profile of intent text against field centers
             # We look at the latent representation
             _, latent, _ = self.encoder.encode_with_components(intent_text)
             latent = latent.to(self.encoder.device)
             
             # Access underlying field
             pm_field = self.encoder.pm_field
             
             # Handle MultiScale vs Standard
             if hasattr(pm_field, 'fine_field'):
                 target_field = pm_field.fine_field
             else:
                 target_field = pm_field
             
             # Calculate distance to all centers
             dists = torch.cdist(latent, target_field.centers)
             
             # Activate spin for nearby centers (Gaussian falloff)
             # Use a wider standard deviation (2.0) to ensure interaction in high-D space
             proximity = torch.exp(-dists[0]**2 / (2 * 2.0**2))
             
             # Ensure omegas exist (they should if initialized correctly)
             if hasattr(target_field, 'omegas'):
                 # We add magnitude to the spin, effectively "energizing" these concepts
                 # to act as active agents in the flow field.
                 # Current direction of spin preserves existing bias or random init.
                 current_spin = target_field.omegas.data
                 
                 # If spin is zero, give it a random direction
                 zero_mask = (current_spin == 0)
                 if zero_mask.any():
                     current_spin[zero_mask] = torch.randn_like(current_spin[zero_mask])
                 
                 # Strengthen the spin
                 # We normalize proximity to distribute 'strength' amount of energy
                 prox_norm = proximity / (proximity.sum() + 1e-6)
                 target_field.omegas.data += strength * prox_norm * torch.sign(current_spin)

    def think(self, prompt: str, steps: int = 10) -> Dict[str, Any]:
        """
        Perform a reasoning step (Thinking) using physics.
        Returns the trajectory and the final conclusion stats.
        """
        with torch.no_grad():
            _, latent, _ = self.encoder.encode_with_components(prompt)
            latent = latent.to(self.encoder.device)
            
            # Trace trajectory
            pm_field = self.encoder.pm_field
            if hasattr(pm_field, 'fine_field'):
                # For multiscale, we currently trace the fine field for detailed reasoning
                # Ideally we'd trace both, but fine field captures specific concept dynamics
                trajectory = pm_field.fine_field(latent, return_trajectory=True)
            else:
                trajectory = pm_field(latent, return_trajectory=True)
                
            # Trajectory shape: (1, Steps+1, D)
            
            # Analyze Result
            start_point = trajectory[0, 0]
            end_point = trajectory[0, -1]
            
            # Calculate path length (Mental Effort)
            diffs = trajectory[0, 1:] - trajectory[0, :-1]
            segment_lengths = torch.norm(diffs, dim=1)
            path_length = segment_lengths.sum().item()
            
            # Did we move?
            displacement = torch.norm(end_point - start_point).item()
            
            return {
                "trajectory": trajectory.cpu(),
                "path_length": path_length,
                "displacement": displacement,
                "steps": steps,
                "final_latent": end_point.cpu()
            }

    def _evaluate_outcome(self, layer_input, layer_output, context) -> OutcomeSignals:
        """
        Evaluate the quality of the thought process using physics metrics.
        """
        stats = layer_output
        path_length = stats['path_length']
        displacement = stats['displacement']
        
        # Efficiency Ratio (0.0 to 1.0)
        # 1.0 = Straight line (High confidence/Clarity)
        # <0.5 = Meandering/Struggling
        efficiency = displacement / (path_length + 1e-6)
        
        # Confidence derived from physical efficiency
        confidence = min(1.0, efficiency)
        
        # Success: Did we arrive somewhere useful?
        # A very short path (displacement ~ 0) means we didn't think or got stuck.
        # A very long path might be good if efficiency is decent.
        success = 1.0 if displacement > 0.1 else 0.0
        
        return OutcomeSignals(
            layer_name="reasoning",
            overall_success=success,
            confidence=confidence,
            layer_signals={
                "mental_effort": path_length,
                "cognitive_efficiency": efficiency
            }
        )

    def _extract_and_store_pattern(self, layer_input, layer_output, signals, context):
        """
        For now, we don't store explicit reasoning patterns as text.
        Future work: Store successful (start -> end) vector pairs.
        """
        pass
