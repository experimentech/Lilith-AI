"""
CognitiveCycle: Continuous loop with three-tier action selection.

This implements the "always-on" architecture for Lilith v2:
- Reflex tier: Fast, immediate responses (rate-limited)
- Deliberative tier: BioNN-driven goal selection (PMFlow physics)
- Homeostatic tier: Decay prevention, internal stimulation

The cycle monitors neural health (attractor strength, field entropy) and
triggers maintenance when energy landscape flattens.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class ActionTier(Enum):
    """Three-tier action classification."""
    REFLEX = "reflex"           # Fast, immediate, pattern-matched
    DELIBERATIVE = "deliberative"  # PMFlow physics, goal-directed
    HOMEOSTATIC = "homeostatic"    # Internal maintenance, decay prevention


@dataclass
class EndpointBudget:
    """Rate limiting and cost tracking for external endpoints."""
    name: str
    calls_per_minute: int = 60
    cooldown_seconds: float = 0.0
    priority_score: float = 1.0
    cost_per_call: float = 0.0
    
    # Runtime state
    call_count: int = 0
    last_call_time: float = 0.0
    total_cost: float = 0.0
    window_start: float = field(default_factory=time.time)
    
    def can_call(self) -> bool:
        """Check if endpoint is available within budget."""
        now = time.time()
        
        # Reset window if minute passed
        if now - self.window_start > 60:
            self.call_count = 0
            self.window_start = now
        
        # Check rate limit
        if self.call_count >= self.calls_per_minute:
            return False
        
        # Check cooldown
        if now - self.last_call_time < self.cooldown_seconds:
            return False
        
        return True
    
    def record_call(self):
        """Record a call to this endpoint."""
        self.call_count += 1
        self.last_call_time = time.time()
        self.total_cost += self.cost_per_call


@dataclass
class NeuralHealthMetrics:
    """Metrics for detecting neural decay."""
    attractor_strength_mean: float = 1.0      # Average mu of attractors
    attractor_strength_variance: float = 0.1  # Variance in mus
    field_entropy: float = 0.5                # Entropy of attractor distribution
    hebbian_weight_variance: float = 0.1      # Variance in Hebbian weights
    pattern_store_size: int = 0               # Number of learned patterns
    recent_success_rate: float = 0.5          # Success rate in recent interactions
    
    # Thresholds for triggering homeostatic maintenance
    MIN_ATTRACTOR_VARIANCE: float = 0.05      # Below this = flat landscape
    MAX_FIELD_ENTROPY: float = 0.9            # Above this = uniform distribution
    MIN_HEBBIAN_VARIANCE: float = 0.02        # Below this = dead weights
    
    def needs_maintenance(self) -> bool:
        """Check if neural health requires homeostatic intervention."""
        if self.attractor_strength_variance < self.MIN_ATTRACTOR_VARIANCE:
            logger.debug("Low attractor variance - landscape flattening")
            return True
        if self.field_entropy > self.MAX_FIELD_ENTROPY:
            logger.debug("High field entropy - attractors dispersing")
            return True
        if self.hebbian_weight_variance < self.MIN_HEBBIAN_VARIANCE:
            logger.debug("Low Hebbian variance - weights converging")
            return True
        return False


@dataclass
class IntrinsicDrive:
    """Internal motivation powering idle behavior."""
    name: str
    strength: float = 0.5       # 0.0 - 1.0
    satiation: float = 0.5      # 0.0 (hungry) - 1.0 (satiated)
    decay_rate: float = 0.01    # How fast satiation decays
    
    def tick(self, dt: float = 1.0):
        """Decay satiation over time, increasing drive."""
        self.satiation = max(0.0, self.satiation - self.decay_rate * dt)
    
    def satisfy(self, amount: float = 0.2):
        """Satisfy the drive (after completing drive-related action)."""
        self.satiation = min(1.0, self.satiation + amount)
    
    @property
    def urgency(self) -> float:
        """How urgently this drive needs satisfaction."""
        return self.strength * (1.0 - self.satiation)


class CognitiveCycle:
    """
    Continuous cognitive loop with three-tier action selection.
    
    Architecture:
    1. Reflex tier: Pattern-matched responses, minimal latency
    2. Deliberative tier: PMFlow physics simulation, goal selection
    3. Homeostatic tier: Decay detection, internal stimulation
    
    The cycle runs continuously, selecting actions based on:
    - External stimuli (user input, sensor input)
    - Internal drives (curiosity, social connection)
    - Neural health (preventing attractor decay)
    """
    
    def __init__(
        self,
        cognitive_stage,       # CognitiveStage instance
        affective_system,      # AffectiveSystem instance
        somatic_layer = None,  # SomaticLayer instance (optional)
        config: Optional[Dict] = None
    ):
        self.cognitive_stage = cognitive_stage
        self.affective_system = affective_system
        self.somatic_layer = somatic_layer
        
        config = config or {}
        
        # Timing
        self.tick_interval = config.get("tick_interval", 0.1)  # 100ms default
        self.idle_threshold = config.get("idle_threshold", 30.0)  # 30s until idle
        
        # Endpoint budgets
        self.endpoint_budgets: Dict[str, EndpointBudget] = {}
        self._init_default_budgets()
        
        # Neural health tracking
        self.health = NeuralHealthMetrics()
        self.last_health_check = 0.0
        self.health_check_interval = config.get("health_check_interval", 5.0)
        
        # Intrinsic drives
        self.drives: Dict[str, IntrinsicDrive] = {
            "curiosity": IntrinsicDrive("curiosity", strength=0.8),
            "social": IntrinsicDrive("social", strength=0.6),
            "consolidation": IntrinsicDrive("consolidation", strength=0.4),
        }
        
        # State
        self.running = False
        self.last_input_time = time.time()
        self.last_action_time = time.time()
        self._action_queue: List[Dict[str, Any]] = []
        
        # Callbacks
        self.on_action: Optional[Callable[[str, Dict], None]] = None
    
    def _init_default_budgets(self):
        """Initialize default endpoint budgets."""
        defaults = {
            "wikipedia": EndpointBudget("wikipedia", calls_per_minute=20, cooldown_seconds=1.0),
            "mcp_search": EndpointBudget("mcp_search", calls_per_minute=10, cooldown_seconds=2.0),
            "external_api": EndpointBudget("external_api", calls_per_minute=30, priority_score=0.8),
        }
        self.endpoint_budgets.update(defaults)
    
    # ============ Main Loop ============
    
    async def run(self):
        """Main cognitive loop - runs continuously."""
        self.running = True
        logger.info("CognitiveCycle started")
        
        try:
            while self.running:
                await self._tick()
                await asyncio.sleep(self.tick_interval)
        except asyncio.CancelledError:
            logger.info("CognitiveCycle cancelled")
        finally:
            self.running = False
            logger.info("CognitiveCycle stopped")
    
    def stop(self):
        """Signal the loop to stop."""
        self.running = False
    
    async def _tick(self):
        """Single heartbeat of the cognitive cycle."""
        now = time.time()
        
        # 1. Process any queued external actions (reflex tier)
        if self._action_queue:
            action = self._action_queue.pop(0)
            await self._process_action(action, tier=ActionTier.REFLEX)
            return
        
        # 2. Update intrinsic drives
        dt = now - self.last_action_time
        for drive in self.drives.values():
            drive.tick(dt)
        
        # 3. Check neural health periodically (homeostatic tier)
        if now - self.last_health_check > self.health_check_interval:
            self._update_health_metrics()
            self.last_health_check = now
            
            if self.health.needs_maintenance():
                await self._homeostatic_maintenance()
                return
        
        # 4. Idle behavior - drive-powered actions
        idle_time = now - self.last_input_time
        if idle_time > self.idle_threshold:
            most_urgent = self._select_urgent_drive()
            if most_urgent and most_urgent.urgency > 0.3:
                await self._drive_powered_action(most_urgent)
                return
        
        # 5. Deliberative tier - PMFlow-driven goal selection
        # Only if no reflex/homeostatic actions needed
        # This runs the physics simulation to find next conceptual target
        await self._deliberative_step()
    
    # ============ Tier Handlers ============
    
    async def _process_action(self, action: Dict, tier: ActionTier):
        """Process an action at the specified tier."""
        action_type = action.get("type", "unknown")
        
        logger.debug(f"[{tier.value}] Processing action: {action_type}")
        
        if action_type == "respond":
            # Generate response through CognitiveStage
            response = self.cognitive_stage.learn(
                action.get("input", ""),
                ctx=action.get("ctx", {})
            )
            if self.on_action:
                self.on_action("response", {"text": response, "tier": tier.value})
        
        elif action_type == "lookup":
            # External lookup with budget check
            endpoint = action.get("endpoint", "external_api")
            budget = self.endpoint_budgets.get(endpoint)
            
            if budget and budget.can_call():
                budget.record_call()
                # Actual lookup would happen here via MCP
                logger.debug(f"Budget OK for {endpoint}, executing lookup")
            else:
                logger.debug(f"Budget exhausted for {endpoint}, deferring")
        
        elif action_type == "learn":
            # Explicit learning/pattern storage
            pattern = action.get("pattern")
            if pattern:
                self.cognitive_stage._response_composer.add_pattern(**pattern)
        
        self.last_action_time = time.time()
    
    async def _homeostatic_maintenance(self):
        """Perform neural maintenance to prevent decay."""
        logger.info("Homeostatic maintenance triggered")
        
        maintenance_actions = []
        
        # 1. Replay successful patterns (Hebbian consolidation)
        if hasattr(self.cognitive_stage, '_response_composer'):
            top_patterns = self.cognitive_stage._response_composer.get_top_patterns(n=5)
            for pattern in top_patterns:
                # Internal replay - activate the pattern without external output
                logger.debug(f"Replaying pattern: {pattern.get('trigger', 'unknown')[:30]}")
                maintenance_actions.append("pattern_replay")
        
        # 2. Prune low-success patterns
        if hasattr(self.cognitive_stage, '_response_composer'):
            pruned = self.cognitive_stage._response_composer.prune_low_scoring(threshold=0.2)
            if pruned > 0:
                logger.debug(f"Pruned {pruned} low-scoring patterns")
                maintenance_actions.append("pattern_prune")
        
        # 3. Random walk through concept space (exploration)
        # This stimulates the PMFlow field with random inputs
        if HAS_TORCH and hasattr(self.cognitive_stage, 'physics'):
            assert torch is not None
            random_point = torch.randn(1, self.cognitive_stage.physics.latent_dim)
            trajectory = self.cognitive_stage.physics.forward(random_point, return_trajectory=True)
            logger.debug(f"Random walk trajectory length: {trajectory.shape[1] if len(trajectory.shape) > 1 else 1}")
            maintenance_actions.append("random_walk")
        
        # 4. Contrastive refresh - strengthen distinctions
        # Pick two random attractors and ensure they're separated
        if HAS_TORCH and hasattr(self.cognitive_stage, 'physics'):
            assert torch is not None
            mus = self.cognitive_stage.physics.mus
            if mus.sum().abs() > 0.1:  # Has active attractors
                # Boost the variance by slightly adjusting strengths
                noise = mus.std() * 0.1 * torch.randn_like(mus)
                self.cognitive_stage.physics.mus = mus + noise
                maintenance_actions.append("contrastive_refresh")
        
        # Satisfy consolidation drive
        self.drives["consolidation"].satisfy(0.5)
        
        logger.info(f"Maintenance complete: {maintenance_actions}")
    
    async def _drive_powered_action(self, drive: IntrinsicDrive):
        """Execute action powered by intrinsic drive."""
        logger.debug(f"Drive-powered action: {drive.name} (urgency={drive.urgency:.2f})")
        
        if drive.name == "curiosity":
            # Explore: random association or concept linking
            # This would query the semantic encoder with a random pattern
            exploration_result = self._explore_concept_space()
            drive.satisfy(0.3 if exploration_result else 0.1)
        
        elif drive.name == "social":
            # Social: check if any channels have pending interactions
            # In full implementation, would check message queues
            pass  # Placeholder - social drive satisfied by user input
        
        elif drive.name == "consolidation":
            # Consolidation: memory maintenance
            await self._homeostatic_maintenance()
    
    def _explore_concept_space(self) -> bool:
        """
        Curiosity-driven exploration of concept space.
        Returns True if interesting pattern found.
        """
        # This is a learning-driven exploration:
        # 1. Pick a random point in the physics field
        # 2. Simulate trajectory to see where it lands
        # 3. If it converges to an attractor, record the association
        
        if not HAS_TORCH or not hasattr(self.cognitive_stage, 'physics'):
            return False
        
        assert torch is not None
        try:
            random_start = torch.randn(1, self.cognitive_stage.physics.latent_dim) * 2
            trajectory = self.cognitive_stage.physics.forward(random_start, return_trajectory=True)
            
            # Check if trajectory converged (velocity decreased)
            if trajectory.shape[1] > 2:
                start_velocity = (trajectory[0, 1] - trajectory[0, 0]).norm()
                end_velocity = (trajectory[0, -1] - trajectory[0, -2]).norm()
                
                converged = end_velocity < start_velocity * 0.5
                if converged:
                    logger.debug("Exploration found attractor basin")
                    return True
            
        except Exception as e:
            logger.debug(f"Exploration failed: {e}")
        
        return False
    
    async def _deliberative_step(self):
        """
        Deliberative tier: PMFlow physics-driven goal selection.
        
        This simulates the current cognitive state through the physics
        field to find the next conceptual attractor (goal).
        """
        # Only deliberate if there's something to think about
        if not hasattr(self.cognitive_stage, 'last_interaction'):
            return
        
        last = self.cognitive_stage.last_interaction
        if not last:
            return
        
        # Check if we're mid-thought (have an active trajectory)
        # In full implementation, would maintain working memory state
        pass  # Placeholder for deliberative processing
    
    # ============ Health Monitoring ============
    
    def _update_health_metrics(self):
        """Update neural health metrics from current state."""
        if not HAS_TORCH:
            return
        
        assert torch is not None
        
        # 1. Attractor statistics
        if hasattr(self.cognitive_stage, 'physics'):
            mus = self.cognitive_stage.physics.mus
            self.health.attractor_strength_mean = mus.abs().mean().item()
            self.health.attractor_strength_variance = mus.var().item() if mus.numel() > 1 else 0.0
            
            # Field entropy approximation using attractor distribution
            if mus.abs().sum() > 0:
                probs = mus.abs() / mus.abs().sum()
                entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                max_entropy = torch.log(torch.tensor(float(mus.numel()))).item()
                self.health.field_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # 2. Pattern store statistics
        if hasattr(self.cognitive_stage, '_response_composer'):
            composer = self.cognitive_stage._response_composer
            if hasattr(composer, 'pattern_store') and composer.pattern_store:
                store = composer.pattern_store
                self.health.pattern_store_size = len(store.patterns) if hasattr(store, 'patterns') else 0
        
        # 3. Hebbian weights (if available)
        # This would check the semantic encoder's Hebbian layer
        # Placeholder for now
        self.health.hebbian_weight_variance = 0.1  # Default healthy value
        
        logger.debug(f"Health: attractor_var={self.health.attractor_strength_variance:.4f}, "
                    f"entropy={self.health.field_entropy:.4f}")
    
    # ============ External Interface ============
    
    def queue_input(self, text: str, ctx: Optional[Dict] = None):
        """Queue external input for processing (reflex tier)."""
        self._action_queue.append({
            "type": "respond",
            "input": text,
            "ctx": ctx or {}
        })
        self.last_input_time = time.time()
        
        # Satisfy social drive on input
        self.drives["social"].satisfy(0.3)
    
    def _select_urgent_drive(self) -> Optional[IntrinsicDrive]:
        """Select the most urgent intrinsic drive."""
        sorted_drives = sorted(self.drives.values(), key=lambda d: d.urgency, reverse=True)
        return sorted_drives[0] if sorted_drives else None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current cycle status for debugging."""
        return {
            "running": self.running,
            "last_input_ago": time.time() - self.last_input_time,
            "last_action_ago": time.time() - self.last_action_time,
            "action_queue_size": len(self._action_queue),
            "drives": {name: {"urgency": d.urgency, "satiation": d.satiation} 
                      for name, d in self.drives.items()},
            "health": {
                "attractor_variance": self.health.attractor_strength_variance,
                "field_entropy": self.health.field_entropy,
                "needs_maintenance": self.health.needs_maintenance()
            },
            "endpoint_budgets": {name: {"remaining": b.calls_per_minute - b.call_count, 
                                        "can_call": b.can_call()} 
                                for name, b in self.endpoint_budgets.items()}
        }
