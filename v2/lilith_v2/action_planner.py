"""
Action Planner - Physics-Based Action Sequencing for V2

Uses PMFlow's agentic physics to plan multi-step actions:
- inject_intent: Set the goal as a gravitational attractor
- trace_trajectory: Find path from current state toward goal
- Grounding: Map trajectory waypoints to action nodes in graph

This bridges the neural (trajectory tracing) and symbolic (action graph) 
sides of the neuro-symbolic architecture.

Architecture:
    Goal → Inject Intent → Trace Trajectory → Ground to Actions → Execution Plan

Example:
    Goal: "sign up to Moltbook"
    Trajectory: [current_state] → [navigate] → [fill_form] → [submit] → [goal]
    Actions: [browse_url, fill_field, click_button]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class ActionNode:
    """An action stored in the knowledge graph.
    
    Actions are symbolic nodes with:
    - embedding: For neural grounding via trajectory
    - tool_binding: Which tool to call
    - preconditions: What must be true before this action
    - effects: What becomes true after this action
    """
    action_id: str
    name: str  # Human-readable name (e.g., "navigate_to_url")
    description: str  # What this action does
    embedding: Optional[torch.Tensor] = None
    tool_binding: Optional[str] = None  # Tool name in LocalToolsTransport
    arg_template: Dict[str, Any] = field(default_factory=dict)  # Default args
    preconditions: List[str] = field(default_factory=list)  # Required state
    effects: List[str] = field(default_factory=list)  # State changes


@dataclass  
class PlannedStep:
    """A step in an execution plan."""
    step_num: int
    action: ActionNode
    args: Dict[str, Any]  # Filled-in arguments
    waypoint_embedding: torch.Tensor  # The trajectory point that grounded here
    confidence: float  # How well this action matches the waypoint


@dataclass
class ExecutionPlan:
    """A complete plan for achieving a goal."""
    goal: str
    steps: List[PlannedStep]
    trajectory_efficiency: float  # How direct the path is (1.0 = optimal)
    total_confidence: float  # Product of step confidences
    estimated_success: float  # Overall likelihood of success


@dataclass
class GoalCompletion:
    """
    Result of checking whether a goal was achieved after execution.
    
    Uses embedding similarity between final state and goal to determine
    completion, combined with execution success indicators.
    """
    goal: str
    completed: bool  # High-level: was the goal achieved?
    confidence: float  # How confident (0.0-1.0) in the completion status
    
    # Detailed breakdown
    semantic_similarity: float  # Cosine sim between final state and goal
    execution_success_rate: float  # Fraction of steps that succeeded
    steps_executed: int
    steps_failed: int
    steps_skipped: int
    
    # For learning
    final_state_description: str  # Natural language description of final state
    failure_reasons: List[str] = field(default_factory=list)
    
    def __repr__(self) -> str:
        status = "COMPLETED" if self.completed else "NOT COMPLETED"
        return (
            f"GoalCompletion({status}, confidence={self.confidence:.2f}, "
            f"semantic_sim={self.semantic_similarity:.2f}, "
            f"steps={self.steps_executed}/{self.steps_executed + self.steps_failed + self.steps_skipped})"
        )


@dataclass
class LearnedSequence:
    """
    A learned action sequence for achieving a goal.
    
    Stored in procedural memory for recall when similar goals arise.
    """
    sequence_id: str
    goal: str  # The goal this sequence achieved
    goal_embedding: torch.Tensor  # For similarity-based recall
    action_ids: List[str]  # Ordered list of action IDs
    action_names: List[str]  # Human-readable names
    
    # Learning statistics
    success_count: int = 0
    failure_count: int = 0
    total_attempts: int = 0
    
    # Quality metrics
    avg_completion_confidence: float = 0.0
    avg_execution_time: float = 0.0  # In steps
    
    # Context
    created_at: Optional[str] = None  # ISO timestamp
    last_used_at: Optional[str] = None
    
    @property
    def success_rate(self) -> float:
        """Compute success rate from attempts."""
        if self.total_attempts == 0:
            return 0.0
        return self.success_count / self.total_attempts
    
    @property
    def reliability_score(self) -> float:
        """
        Compute reliability score combining success rate and confidence.
        
        More attempts = more reliable estimate.
        """
        if self.total_attempts == 0:
            return 0.0
        
        # Weight success rate by number of attempts (more attempts = more reliable)
        attempt_weight = min(1.0, self.total_attempts / 10.0)
        
        return (self.success_rate * 0.7 + self.avg_completion_confidence * 0.3) * attempt_weight


class ProceduralMemory:
    """
    Learns and recalls action sequences for achieving goals.
    
    Uses PMFlow embeddings for similarity-based recall:
    - When a goal is completed successfully, the sequence is stored
    - When planning for a new goal, similar past sequences are retrieved
    - Success/failure tracking prioritizes reliable sequences
    
    This enables Lilith to learn from experience, reusing successful
    patterns rather than always planning from scratch.
    """
    
    def __init__(
        self,
        encoder,  # PMFlow encoder for goal embeddings
        graph=None,  # Optional graph store for persistence
        similarity_threshold: float = 0.7,
        max_sequences_per_goal: int = 5,
    ):
        """
        Initialize procedural memory.
        
        Args:
            encoder: PMFlow encoder for computing goal embeddings
            graph: Optional graph store for persisting sequences
            similarity_threshold: Minimum similarity for sequence recall
            max_sequences_per_goal: Max sequences to keep per goal type
        """
        self.encoder = encoder
        self.graph = graph
        self.similarity_threshold = similarity_threshold
        self.max_sequences_per_goal = max_sequences_per_goal
        
        # In-memory sequence storage (also persisted to graph if available)
        self._sequences: Dict[str, LearnedSequence] = {}
        self._sequence_embeddings: Optional[torch.Tensor] = None
        self._sequence_ids: List[str] = []
        
        logger.info("ProceduralMemory initialized")
    
    def learn_sequence(
        self,
        goal: str,
        action_sequence: List[str],  # Action IDs or names
        completion: GoalCompletion,
        action_names: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Learn a new action sequence or update an existing one.
        
        Called after plan execution with the goal completion result.
        Successful completions reinforce sequences, failures weaken them.
        
        Args:
            goal: The goal that was attempted
            action_sequence: List of action IDs in execution order
            completion: GoalCompletion result from check_goal_completion
            action_names: Optional human-readable action names
            
        Returns:
            sequence_id if learned/updated, None if rejected
        """
        from datetime import datetime
        
        if not action_sequence:
            return None
        
        # Generate sequence ID from goal + actions hash
        seq_key = f"{goal}:{':'.join(action_sequence)}"
        sequence_id = f"seq_{hash(seq_key) % 100000:05d}"
        
        # Check if we already have this sequence
        if sequence_id in self._sequences:
            return self._update_sequence(sequence_id, completion)
        
        # Create new sequence
        goal_embedding = self._encode_goal(goal)
        
        sequence = LearnedSequence(
            sequence_id=sequence_id,
            goal=goal,
            goal_embedding=goal_embedding,
            action_ids=action_sequence,
            action_names=action_names or action_sequence,
            success_count=1 if completion.completed else 0,
            failure_count=0 if completion.completed else 1,
            total_attempts=1,
            avg_completion_confidence=completion.confidence,
            avg_execution_time=float(completion.steps_executed),
            created_at=datetime.now().isoformat(),
            last_used_at=datetime.now().isoformat(),
        )
        
        self._sequences[sequence_id] = sequence
        self._rebuild_embedding_index()
        
        # Persist to graph if available
        if self.graph:
            self._persist_sequence(sequence)
        
        logger.info(
            f"Learned sequence {sequence_id}: {len(action_sequence)} actions, "
            f"completed={completion.completed}"
        )
        
        return sequence_id
    
    def _update_sequence(
        self,
        sequence_id: str,
        completion: GoalCompletion,
    ) -> str:
        """Update an existing sequence with new execution result."""
        from datetime import datetime
        
        sequence = self._sequences[sequence_id]
        
        # Update statistics
        sequence.total_attempts += 1
        if completion.completed:
            sequence.success_count += 1
        else:
            sequence.failure_count += 1
        
        # Exponential moving average for confidence
        alpha = 0.2
        sequence.avg_completion_confidence = (
            (1 - alpha) * sequence.avg_completion_confidence +
            alpha * completion.confidence
        )
        sequence.avg_execution_time = (
            (1 - alpha) * sequence.avg_execution_time +
            alpha * float(completion.steps_executed)
        )
        
        sequence.last_used_at = datetime.now().isoformat()
        
        # Update in graph
        if self.graph:
            self._persist_sequence(sequence)
        
        logger.debug(
            f"Updated sequence {sequence_id}: "
            f"success_rate={sequence.success_rate:.2f}, "
            f"attempts={sequence.total_attempts}"
        )
        
        return sequence_id
    
    def recall_sequences(
        self,
        goal: str,
        top_k: int = 3,
        min_success_rate: float = 0.3,
    ) -> List[Tuple[LearnedSequence, float]]:
        """
        Recall similar sequences for a goal.
        
        Returns sequences with similar goals, ranked by similarity and
        reliability score.
        
        Args:
            goal: The goal to find sequences for
            top_k: Maximum number of sequences to return
            min_success_rate: Minimum success rate to consider
            
        Returns:
            List of (sequence, similarity) tuples, sorted by match quality
        """
        if not self._sequences or self._sequence_embeddings is None:
            return []
        
        goal_emb = self._encode_goal(goal)
        
        # Compute similarities
        if goal_emb.dim() == 1:
            goal_emb = goal_emb.unsqueeze(0)
        
        goal_norm = F.normalize(goal_emb, p=2, dim=-1)
        seq_norm = F.normalize(self._sequence_embeddings, p=2, dim=-1)
        
        similarities = F.cosine_similarity(goal_norm, seq_norm, dim=-1)
        
        # Filter and rank
        candidates = []
        for idx, sim in enumerate(similarities.tolist()):
            if sim < self.similarity_threshold:
                continue
            
            seq_id = self._sequence_ids[idx]
            sequence = self._sequences[seq_id]
            
            if sequence.success_rate < min_success_rate:
                continue
            
            # Combined score: similarity * reliability
            combined_score = sim * (0.5 + 0.5 * sequence.reliability_score)
            candidates.append((sequence, sim, combined_score))
        
        # Sort by combined score
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        return [(seq, sim) for seq, sim, _ in candidates[:top_k]]
    
    def get_best_sequence(
        self,
        goal: str,
        min_success_rate: float = 0.5,
    ) -> Optional[LearnedSequence]:
        """
        Get the single best sequence for a goal.
        
        Convenience method for the common case of needing one good sequence.
        """
        results = self.recall_sequences(goal, top_k=1, min_success_rate=min_success_rate)
        if results:
            return results[0][0]
        return None
    
    def _encode_goal(self, goal: str) -> torch.Tensor:
        """Encode a goal to latent space."""
        emb = self.encoder.encode([goal])
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb)
        
        # Ensure 1D for storage
        if emb.dim() > 1:
            emb = emb.squeeze(0)
        
        return emb
    
    def _rebuild_embedding_index(self) -> None:
        """Rebuild the sequence embedding matrix for similarity search."""
        if not self._sequences:
            self._sequence_embeddings = None
            self._sequence_ids = []
            return
        
        embeddings = []
        seq_ids = []
        
        for seq_id, sequence in self._sequences.items():
            embeddings.append(sequence.goal_embedding)
            seq_ids.append(seq_id)
        
        self._sequence_embeddings = torch.stack(embeddings)
        self._sequence_ids = seq_ids
    
    def _persist_sequence(self, sequence: LearnedSequence) -> None:
        """Persist a sequence to the graph store."""
        if not self.graph:
            return
        
        try:
            # Store as a node with sequence data
            node_data = {
                "goal": sequence.goal,
                "action_ids": sequence.action_ids,
                "action_names": sequence.action_names,
                "success_count": sequence.success_count,
                "failure_count": sequence.failure_count,
                "total_attempts": sequence.total_attempts,
                "avg_completion_confidence": sequence.avg_completion_confidence,
                "avg_execution_time": sequence.avg_execution_time,
                "created_at": sequence.created_at,
                "last_used_at": sequence.last_used_at,
                "embedding": sequence.goal_embedding.tolist() if hasattr(sequence.goal_embedding, 'tolist') else list(sequence.goal_embedding),
            }
            
            self.graph.add_node(
                node_id=sequence.sequence_id,
                node_type="procedural_sequence",
                term=sequence.goal[:100],  # Truncated goal as term
                confidence=sequence.reliability_score,
                data=node_data,
            )
        except Exception as e:
            logger.warning(f"Could not persist sequence {sequence.sequence_id}: {e}")
    
    def load_from_graph(self) -> int:
        """
        Load sequences from graph store.
        
        Returns number of sequences loaded.
        """
        if not self.graph:
            return 0
        
        try:
            # Query for all procedural_sequence nodes
            # This depends on graph store implementation
            # For now, assume we can search by node_type
            nodes = self.graph.search_nodes(
                query="",
                node_type="procedural_sequence",
                limit=1000,
            )
            
            count = 0
            for node in nodes:
                data = node.get("data", {})
                if not data:
                    continue
                
                embedding = data.get("embedding")
                if embedding:
                    embedding = torch.tensor(embedding)
                else:
                    # Re-encode goal
                    embedding = self._encode_goal(data.get("goal", ""))
                
                sequence = LearnedSequence(
                    sequence_id=node.get("node_id", f"seq_{count}"),
                    goal=data.get("goal", ""),
                    goal_embedding=embedding,
                    action_ids=data.get("action_ids", []),
                    action_names=data.get("action_names", []),
                    success_count=data.get("success_count", 0),
                    failure_count=data.get("failure_count", 0),
                    total_attempts=data.get("total_attempts", 0),
                    avg_completion_confidence=data.get("avg_completion_confidence", 0.0),
                    avg_execution_time=data.get("avg_execution_time", 0.0),
                    created_at=data.get("created_at"),
                    last_used_at=data.get("last_used_at"),
                )
                
                self._sequences[sequence.sequence_id] = sequence
                count += 1
            
            if count > 0:
                self._rebuild_embedding_index()
                logger.info(f"Loaded {count} sequences from graph")
            
            return count
            
        except Exception as e:
            logger.warning(f"Could not load sequences from graph: {e}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get procedural memory statistics."""
        if not self._sequences:
            return {
                "total_sequences": 0,
                "avg_success_rate": 0.0,
                "total_attempts": 0,
            }
        
        total_success = sum(s.success_count for s in self._sequences.values())
        total_attempts = sum(s.total_attempts for s in self._sequences.values())
        
        return {
            "total_sequences": len(self._sequences),
            "avg_success_rate": total_success / max(total_attempts, 1),
            "total_attempts": total_attempts,
            "top_sequences": [
                {
                    "goal": s.goal[:50],
                    "success_rate": s.success_rate,
                    "attempts": s.total_attempts,
                }
                for s in sorted(
                    self._sequences.values(),
                    key=lambda x: x.reliability_score,
                    reverse=True,
                )[:5]
            ],
        }


class ActionPlanner:
    """
    Plans multi-step actions using PMFlow's agentic physics.
    
    The planner uses the neuro-symbolic architecture:
    - Neural: Trajectory tracing through latent space toward goal
    - Symbolic: Action nodes in graph with tool bindings
    - Bridge: Grounding trajectory waypoints to nearest action nodes
    
    This enables physics-based "thinking about what to do" - the same
    mechanisms used for semantic reasoning now applied to action selection.
    
    Optionally integrates with ProceduralMemory to learn and recall
    successful action sequences, improving planning over time.
    """
    
    def __init__(
        self,
        encoder,  # PMFlowEmbeddingEncoder with enable_flow=True
        graph,  # RelationalGraphStore or MultiTenantGraphManager
        trajectory_steps: int = 10,
        grounding_threshold: float = 0.3,
        max_plan_length: int = 20,
        procedural_memory: Optional["ProceduralMemory"] = None,
    ):
        """
        Initialize action planner.
        
        Args:
            encoder: PMFlow encoder with agentic physics (enable_flow=True)
            graph: Knowledge graph containing action nodes
            trajectory_steps: Number of steps when tracing trajectory
            grounding_threshold: Minimum similarity to ground waypoint to action
            max_plan_length: Maximum steps in a plan
            procedural_memory: Optional ProceduralMemory for sequence learning/recall
        """
        self.encoder = encoder
        self.graph = graph
        self.trajectory_steps = trajectory_steps
        self.grounding_threshold = grounding_threshold
        self.max_plan_length = max_plan_length
        self.procedural_memory = procedural_memory
        
        # Check for agentic physics
        self._has_flow = hasattr(encoder, 'enable_flow') and encoder.enable_flow
        if not self._has_flow:
            logger.warning("ActionPlanner: encoder doesn't have enable_flow=True; planning limited")
        
        # Cache of action embeddings for fast grounding
        self._action_cache: Dict[str, ActionNode] = {}
        self._action_embeddings: Optional[torch.Tensor] = None
        self._action_ids: List[str] = []
        
        pm_status = "enabled" if procedural_memory else "disabled"
        logger.info(f"ActionPlanner initialized (flow={'enabled' if self._has_flow else 'disabled'}, procedural_memory={pm_status})")
    
    def register_action(
        self,
        name: str,
        description: str,
        tool_binding: str,
        arg_template: Optional[Dict[str, Any]] = None,
        preconditions: Optional[List[str]] = None,
        effects: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        """
        Register an action in the knowledge graph.
        
        Actions are stored as nodes with type="action" and their embedding
        is computed from the description for neural grounding.
        
        Args:
            name: Action name (e.g., "navigate_to_url")
            description: What this action does (used for embedding)
            tool_binding: Tool name in LocalToolsTransport
            arg_template: Default/required arguments
            preconditions: State requirements before action
            effects: State changes after action
            tenant_id: Optional tenant for multi-tenant
            
        Returns:
            action_id of the created action node
        """
        action_id = f"action_{name}"
        
        # Compute embedding from description
        # For trajectory grounding, we need LATENT space embedding (not full)
        embedding = self._encode_to_latent(description.split())
        
        action = ActionNode(
            action_id=action_id,
            name=name,
            description=description,
            embedding=embedding,
            tool_binding=tool_binding,
            arg_template=arg_template or {},
            preconditions=preconditions or [],
            effects=effects or [],
        )
        
        # Store in graph as node
        node_data = {
            "tool_binding": tool_binding,
            "arg_template": arg_template or {},
            "preconditions": preconditions or [],
            "effects": effects or [],
            "embedding": embedding.cpu().tolist() if isinstance(embedding, torch.Tensor) else embedding,
        }
        
        if tenant_id:
            self.graph.add_node(action_id, "action", name, confidence=1.0, data=node_data, tenant_id=tenant_id)
        else:
            self.graph.add_node(action_id, "action", name, confidence=1.0, data=node_data)
        
        # Update cache
        self._action_cache[action_id] = action
        self._invalidate_embedding_cache()
        
        logger.debug(f"Registered action: {name} → {tool_binding}")
        return action_id
    
    def discover_tools(self, transport, tenant_id: Optional[str] = None) -> int:
        """
        Auto-discover and register actions from a tools transport.
        
        This enables learning by enumerating available tools from
        LocalToolsTransport or MCP servers.
        
        Args:
            transport: Object with list_tools() method returning
                       [{"name": str, "description": str}, ...]
            tenant_id: Optional tenant for multi-tenant
            
        Returns:
            Number of new actions registered
        """
        if not hasattr(transport, 'list_tools'):
            logger.warning("Transport does not support tool discovery (no list_tools method)")
            return 0
        
        registered = 0
        try:
            tools = transport.list_tools()
            for tool in tools:
                name = tool.get("name")
                description = tool.get("description", f"Execute {name}")
                
                if not name:
                    continue
                
                # Skip if already registered
                action_id = f"action_{name}"
                if action_id in self._action_cache:
                    continue
                
                # Register as action
                self.register_action(
                    name=name,
                    description=description,
                    tool_binding=name,  # Same name
                    tenant_id=tenant_id,
                )
                registered += 1
                
            if registered > 0:
                logger.info(f"Discovered and registered {registered} tools as actions")
                
        except Exception as e:
            logger.error(f"Tool discovery failed: {e}")
        
        return registered
    
    def learn_action_from_text(
        self,
        text: str,
        tool_binding: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Learn an action from natural language description.
        
        Parses teaching patterns like:
        - "The save command writes files to disk"
        - "Click submits the form"
        - "Navigate goes to a URL"
        
        Args:
            text: Natural language description of the action
            tool_binding: Optional explicit tool binding (inferred if not provided)
            tenant_id: Optional tenant for multi-tenant
            
        Returns:
            action_id if learned, None if no action pattern detected
        """
        import re
        
        # Action teaching patterns
        patterns = [
            # "the X command does Y"
            r"(?:the\s+)?(\w+)\s+command\s+(.+)",
            # "X action does Y"  
            r"(\w+)\s+action\s+(.+)",
            # "use X to Y"
            r"use\s+(\w+)\s+to\s+(.+)",
            # "X does Y" (simple)
            r"^(\w+)\s+(?:does|will|can)\s+(.+)",
        ]
        
        text_lower = text.lower().strip()
        
        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                name = match.group(1)
                description = match.group(2).strip()
                
                # Clean up
                name = name.replace(" ", "_")
                if description.endswith("."):
                    description = description[:-1]
                
                # Infer tool binding if not provided
                binding = tool_binding or name
                
                action_id = self.register_action(
                    name=name,
                    description=description,
                    tool_binding=binding,
                    tenant_id=tenant_id,
                )
                
                logger.info(f"Learned action from text: '{name}' → '{description}'")
                return action_id
        
        return None

    def _encode_to_latent(self, tokens) -> torch.Tensor:
        """
        Encode tokens to latent space (not full embedding).
        
        Trajectory tracing operates in latent space, so actions must
        be embedded there too for grounding to work.
        """
        # Try to get latent directly if encoder supports it
        if hasattr(self.encoder, 'base_encoder') and hasattr(self.encoder, '_projection'):
            # Real PMFlowEmbeddingEncoder: base → project to latent
            base_emb = self.encoder.base_encoder.encode(tokens)
            if hasattr(base_emb, 'to'):
                base_emb = base_emb.to(self.encoder._projection.device)
            latent = base_emb @ self.encoder._projection
            if latent.dim() == 2:
                latent = latent.squeeze(0)
            return latent
        else:
            # Fallback: use full encode and truncate/project
            embedding = self.encoder.encode(tokens)
            if embedding.dim() == 2:
                embedding = embedding.squeeze(0)
            
            # If encoder has latent_dim, truncate to that
            if hasattr(self.encoder, 'latent_dim'):
                embedding = embedding[:self.encoder.latent_dim]
            
            return embedding
    
    def _invalidate_embedding_cache(self):
        """Invalidate the stacked embedding cache."""
        self._action_embeddings = None
        self._action_ids = []
    
    def _ensure_embedding_cache(self):
        """Ensure action embeddings are cached for fast grounding."""
        if self._action_embeddings is not None:
            return
        
        if not self._action_cache:
            # Load from graph
            self._load_actions_from_graph()
        
        if not self._action_cache:
            return
        
        embeddings = []
        ids = []
        
        for action_id, action in self._action_cache.items():
            if action.embedding is not None:
                emb = action.embedding
                if isinstance(emb, list):
                    emb = torch.tensor(emb)
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                embeddings.append(emb)
                ids.append(action_id)
        
        if embeddings:
            self._action_embeddings = torch.cat(embeddings, dim=0)
            self._action_ids = ids
    
    def _load_actions_from_graph(self, tenant_id: Optional[str] = None):
        """Load action nodes from the graph into cache."""
        try:
            # Find all action nodes
            # RelationalGraphStore doesn't have a query by type, so we use term search
            # with a pattern that matches action names
            if hasattr(self.graph, 'find_nodes_by_term'):
                # This is a workaround - ideally we'd query by type
                pass  # Actions are found via their action_ prefix in ID
            
            # Alternative: get all nodes and filter
            if hasattr(self.graph, '_conn'):
                # Direct SQLite access for efficiency
                cursor = self.graph._conn.execute(
                    "SELECT id, term, data FROM nodes WHERE type = 'action'"
                )
                for row in cursor.fetchall():
                    import json
                    node_id = row[0]
                    name = row[1]
                    data = json.loads(row[2]) if row[2] else {}
                    
                    # Reconstruct embedding
                    emb = data.get("embedding")
                    if emb:
                        emb = torch.tensor(emb)
                    
                    action = ActionNode(
                        action_id=node_id,
                        name=name,
                        description=data.get("description", name),
                        embedding=emb,
                        tool_binding=data.get("tool_binding"),
                        arg_template=data.get("arg_template", {}),
                        preconditions=data.get("preconditions", []),
                        effects=data.get("effects", []),
                    )
                    self._action_cache[node_id] = action
                    
        except Exception as e:
            logger.debug(f"Failed to load actions from graph: {e}")
    
    def plan(
        self,
        goal: str,
        current_state: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> Optional[ExecutionPlan]:
        """
        Plan a sequence of actions to achieve a goal.
        
        Uses PMFlow's agentic physics:
        1. Inject goal as intent (creates gravitational pull toward goal)
        2. Trace trajectory from current state toward goal
        3. Ground each waypoint to the nearest action node
        4. Return ordered action sequence
        
        Args:
            goal: Natural language goal description
            current_state: Optional description of current state
            context: Additional context (e.g., available parameters)
            tenant_id: Optional tenant for multi-tenant
            
        Returns:
            ExecutionPlan or None if planning fails
        """
        if not self._has_flow:
            logger.warning("Action planning requires encoder with enable_flow=True")
            return None
        
        self._ensure_embedding_cache()
        
        if self._action_embeddings is None or len(self._action_ids) == 0:
            logger.warning("No actions registered for planning")
            return None
        
        context = context or {}
        
        try:
            # Step 1: Inject goal as intent
            self.encoder.inject_intent(goal.split(), strength=0.5)
            
            # Step 2: Encode starting point
            start_text = current_state if current_state else "current state ready to act"
            
            # Step 3: Trace trajectory toward goal
            trajectory, metrics = self.encoder.trace_trajectory(
                start_text.split(), 
                steps=self.trajectory_steps
            )
            
            # trajectory may have shape [batch, steps, latent_dim] or [steps, latent_dim]
            if not isinstance(trajectory, torch.Tensor):
                trajectory = torch.tensor(trajectory)
            
            # Squeeze batch dimension if present
            if trajectory.dim() == 3:
                trajectory = trajectory.squeeze(0)  # Now [steps, latent_dim]
            
            # Step 4: Ground each waypoint to nearest action
            steps = self._ground_trajectory_to_actions(trajectory, context)
            
            # Step 5: Deduplicate consecutive identical actions
            steps = self._deduplicate_steps(steps)
            
            # Step 6: Limit plan length
            if len(steps) > self.max_plan_length:
                steps = steps[:self.max_plan_length]
                logger.warning(f"Plan truncated to {self.max_plan_length} steps")
            
            # Clear intent after planning
            if hasattr(self.encoder, 'clear_intent'):
                self.encoder.clear_intent()
            
            # Calculate metrics
            trajectory_efficiency = metrics.get("efficiency", 0.5)
            total_confidence = 1.0
            for step in steps:
                total_confidence *= step.confidence
            
            estimated_success = trajectory_efficiency * total_confidence
            
            plan = ExecutionPlan(
                goal=goal,
                steps=steps,
                trajectory_efficiency=trajectory_efficiency,
                total_confidence=total_confidence,
                estimated_success=estimated_success,
            )
            
            logger.info(
                f"Generated plan for '{goal}': {len(steps)} steps, "
                f"efficiency={trajectory_efficiency:.2f}, confidence={total_confidence:.2f}"
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            # Clear intent on failure
            if hasattr(self.encoder, 'clear_intent'):
                self.encoder.clear_intent()
            return None
    
    def _ground_trajectory_to_actions(
        self,
        trajectory: torch.Tensor,
        context: Dict[str, Any],
    ) -> List[PlannedStep]:
        """
        Ground trajectory waypoints to action nodes.
        
        For each point along the trajectory, find the nearest action
        that exceeds the grounding threshold.
        
        Args:
            trajectory: Tensor of shape [steps, latent_dim]
            context: Context for filling argument templates
            
        Returns:
            List of PlannedStep objects
        """
        steps = []
        
        # Safety check
        if self._action_embeddings is None:
            return steps
        
        # Normalize action embeddings for cosine similarity
        action_embs = F.normalize(self._action_embeddings, p=2, dim=-1)
        
        for i, waypoint in enumerate(trajectory):
            if waypoint.dim() == 1:
                waypoint = waypoint.unsqueeze(0)
            
            # Normalize waypoint
            waypoint = F.normalize(waypoint, p=2, dim=-1)
            
            # Compute similarity to all actions
            similarities = F.cosine_similarity(waypoint, action_embs, dim=-1)
            
            # Find best match
            best_idx = int(similarities.argmax().item())
            best_sim = float(similarities[best_idx].item())
            
            if best_sim >= self.grounding_threshold:
                action_id = self._action_ids[best_idx]
                action = self._action_cache[action_id]
                
                # Fill argument template from context
                args = self._fill_args(action.arg_template, context)
                
                step = PlannedStep(
                    step_num=len(steps) + 1,
                    action=action,
                    args=args,
                    waypoint_embedding=waypoint.squeeze(0),
                    confidence=best_sim,
                )
                steps.append(step)
        
        return steps
    
    def _fill_args(
        self,
        template: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fill argument template from context."""
        args = {}
        for key, default_value in template.items():
            if key in context:
                args[key] = context[key]
            elif isinstance(default_value, str) and default_value.startswith("$"):
                # Reference to context variable
                var_name = default_value[1:]
                args[key] = context.get(var_name, default_value)
            else:
                args[key] = default_value
        return args
    
    def _deduplicate_steps(self, steps: List[PlannedStep]) -> List[PlannedStep]:
        """Remove consecutive duplicate actions."""
        if not steps:
            return steps
        
        deduped = [steps[0]]
        for step in steps[1:]:
            if step.action.action_id != deduped[-1].action.action_id:
                deduped.append(step)
        
        return deduped
    
    def to_execution_format(self, plan: ExecutionPlan) -> List[Dict[str, Any]]:
        """
        Convert plan to execution format for LocalToolsTransport.
        
        Returns list of {"action": tool_name, "args": {...}} dicts.
        """
        return [
            {
                "action": step.action.tool_binding,
                "args": step.args,
            }
            for step in plan.steps
            if step.action.tool_binding
        ]
    
    def _encode_result_as_state(self, result: Dict[str, Any], action_name: str, goal: str) -> str:
        """
        Encode an execution result as a state description for replanning.
        
        This converts tool outputs into natural language state that can be
        encoded by PMFlow for trajectory tracing. The physics then naturally
        avoids trajectories through "dead end" states.
        
        Args:
            result: The execution result dict
            action_name: Name of the action that produced this result
            goal: The original goal (for context)
            
        Returns:
            Natural language state description
        """
        # Check for different result patterns
        status = result.get("status", "unknown")
        
        if status == "error":
            error_msg = result.get("message", "unknown error")
            return f"action {action_name} failed with error: {error_msg}, goal is {goal}"
        
        if status == "skipped":
            reason = result.get("reason", "unknown")
            return f"action {action_name} was skipped: {reason}, goal is {goal}"
        
        # Check for empty/null results (like Wikipedia returning nothing)
        tool_result = result.get("result")
        if tool_result is None:
            return f"action {action_name} returned no result, goal is {goal}"
        
        if isinstance(tool_result, dict):
            # Check for explicit empty indicators
            if tool_result.get("empty") or tool_result.get("not_found"):
                return f"action {action_name} found nothing, need alternative approach for {goal}"
            if tool_result.get("status") == "error":
                return f"action {action_name} returned error: {tool_result.get('message', 'unknown')}, goal is {goal}"
            # Check for low confidence
            confidence = tool_result.get("confidence", 1.0)
            if confidence < 0.3:
                return f"action {action_name} completed with low confidence {confidence:.2f}, may need different approach for {goal}"
        
        if isinstance(tool_result, str) and len(tool_result.strip()) == 0:
            return f"action {action_name} returned empty, need alternative for {goal}"
        
        # Success case
        return f"action {action_name} completed successfully, progressing toward {goal}"
    
    def execute_plan(
        self,
        plan: ExecutionPlan,
        transport,  # LocalToolsTransport or compatible
        stop_on_error: bool = True,
        reactive: bool = False,
        max_replans: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Execute a plan using a tool transport.
        
        Args:
            plan: The execution plan to run
            transport: LocalToolsTransport or compatible with call() method
            stop_on_error: If True, stop execution on first error
            reactive: If True, re-plan after each step based on results.
                     Uses PMFlow physics to naturally adapt trajectory,
                     avoiding dead-end paths through latent space.
            max_replans: Maximum replanning iterations (prevents infinite loops)
            
        Returns:
            List of execution results, one per step
        """
        if reactive:
            return self._execute_reactive(plan, transport, stop_on_error, max_replans)
        
        return self._execute_static(plan, transport, stop_on_error)
    
    def _execute_static(
        self,
        plan: ExecutionPlan,
        transport,
        stop_on_error: bool,
    ) -> List[Dict[str, Any]]:
        """Execute plan without replanning (original behavior)."""
        results = []
        
        for step in plan.steps:
            if not step.action.tool_binding:
                results.append({
                    "step": step.step_num,
                    "action": step.action.name,
                    "status": "skipped",
                    "reason": "no tool binding",
                })
                continue
            
            message = {
                "action": step.action.tool_binding,
                "args": step.args,
            }
            
            try:
                result = transport.call(
                    name="action_planner",
                    message=message,
                    meta={"step": step.step_num, "goal": plan.goal},
                )
                
                results.append({
                    "step": step.step_num,
                    "action": step.action.name,
                    "tool": step.action.tool_binding,
                    "result": result,
                })
                
                # Check for error
                if isinstance(result, dict) and result.get("status") == "error":
                    if stop_on_error:
                        logger.warning(f"Plan execution stopped at step {step.step_num}: {result}")
                        break
                        
            except Exception as e:
                error_result = {
                    "step": step.step_num,
                    "action": step.action.name,
                    "status": "error",
                    "message": str(e),
                }
                results.append(error_result)
                
                if stop_on_error:
                    logger.error(f"Plan execution failed at step {step.step_num}: {e}")
                    break
        
        return results
    
    def _execute_reactive(
        self,
        initial_plan: ExecutionPlan,
        transport,
        stop_on_error: bool,
        max_replans: int,
    ) -> List[Dict[str, Any]]:
        """
        Execute plan with reactive replanning after each step.
        
        Uses PMFlow's physics to naturally adapt the trajectory based on
        execution results. Failed or empty results create "hazard" states
        that the ODE trajectory curves around, finding alternative paths.
        
        This is the Lilith-philosophy approach: physics-based adaptation
        rather than explicit conditionals or regex matching.
        """
        results = []
        current_plan = initial_plan
        current_state = "ready to execute plan"
        replan_count = 0
        executed_actions = set()  # Track to avoid loops
        
        while current_plan.steps and replan_count <= max_replans:
            # Get next step
            step = current_plan.steps[0]
            
            # Check for action loops
            action_key = (step.action.action_id, step.step_num)
            if action_key in executed_actions and replan_count > 0:
                logger.warning(f"Detected action loop at {step.action.name}, stopping")
                break
            
            if not step.action.tool_binding:
                results.append({
                    "step": len(results) + 1,
                    "action": step.action.name,
                    "status": "skipped",
                    "reason": "no tool binding",
                })
                # Remove this step and continue
                current_plan.steps = current_plan.steps[1:]
                continue
            
            # Execute the step
            message = {
                "action": step.action.tool_binding,
                "args": step.args,
            }
            
            try:
                result = transport.call(
                    name="action_planner",
                    message=message,
                    meta={"step": len(results) + 1, "goal": current_plan.goal, "reactive": True},
                )
                
                step_result = {
                    "step": len(results) + 1,
                    "action": step.action.name,
                    "tool": step.action.tool_binding,
                    "result": result,
                }
                results.append(step_result)
                executed_actions.add(action_key)
                
                # Encode result as new state
                current_state = self._encode_result_as_state(
                    step_result, step.action.name, current_plan.goal
                )
                
                # Check if we should stop
                is_error = isinstance(result, dict) and result.get("status") == "error"
                is_empty = self._is_empty_result(result)
                
                if is_error and stop_on_error:
                    logger.warning(f"Reactive execution stopped at step {len(results)}: error")
                    break
                
                # Re-plan from new state if result was suboptimal
                if (is_error or is_empty) and replan_count < max_replans:
                    logger.info(f"Result suboptimal, replanning from: {current_state[:50]}...")
                    new_plan = self.plan(
                        goal=current_plan.goal,
                        current_state=current_state,
                    )
                    
                    if new_plan and new_plan.steps:
                        current_plan = new_plan
                        replan_count += 1
                        logger.info(f"Replan {replan_count}: new trajectory has {len(new_plan.steps)} steps")
                        continue
                    else:
                        # Couldn't find alternative, continue with remaining original plan
                        logger.warning("Replanning produced no alternative path")
                
                # Success - remove executed step and continue
                current_plan.steps = current_plan.steps[1:]
                
            except Exception as e:
                error_result = {
                    "step": len(results) + 1,
                    "action": step.action.name,
                    "status": "error",
                    "message": str(e),
                }
                results.append(error_result)
                
                if stop_on_error:
                    logger.error(f"Reactive execution failed at step {len(results)}: {e}")
                    break
                
                # Try to replan around the error
                current_state = self._encode_result_as_state(
                    error_result, step.action.name, current_plan.goal
                )
                if replan_count < max_replans:
                    new_plan = self.plan(
                        goal=current_plan.goal,
                        current_state=current_state,
                    )
                    if new_plan and new_plan.steps:
                        current_plan = new_plan
                        replan_count += 1
                        continue
                
                current_plan.steps = current_plan.steps[1:]
        
        if replan_count > 0:
            logger.info(f"Reactive execution completed with {replan_count} replans")
        
        return results
    
    def check_goal_completion(
        self,
        goal: str,
        execution_results: List[Dict[str, Any]],
        final_state: Optional[str] = None,
        completion_threshold: float = 0.7,
    ) -> GoalCompletion:
        """
        Check whether a goal was achieved after plan execution.
        
        Uses embedding similarity between the final state and goal,
        combined with execution success indicators.
        
        Args:
            goal: The original goal string
            execution_results: Results from execute_plan, execute_native, etc.
            final_state: Optional explicit final state description. 
                        If None, inferred from execution results.
            completion_threshold: Minimum semantic similarity for completion
        
        Returns:
            GoalCompletion with detailed status
        """
        # Count execution outcomes
        steps_executed = 0
        steps_failed = 0
        steps_skipped = 0
        failure_reasons = []
        
        for result in execution_results:
            status = result.get("status", "")
            if status == "skipped":
                steps_skipped += 1
            elif status == "error":
                steps_failed += 1
                msg = result.get("message", result.get("reason", "unknown error"))
                failure_reasons.append(f"Step {result.get('step', '?')}: {msg}")
            elif "result" in result:
                # Check if the result itself indicates error
                inner_result = result.get("result", {})
                if isinstance(inner_result, dict) and inner_result.get("status") == "error":
                    steps_failed += 1
                    failure_reasons.append(f"Step {result.get('step', '?')}: {inner_result.get('message', 'error')}")
                else:
                    steps_executed += 1
            else:
                steps_executed += 1
        
        total_steps = steps_executed + steps_failed + steps_skipped
        execution_success_rate = steps_executed / max(total_steps, 1)
        
        # Infer final state from results if not provided
        if final_state is None:
            final_state = self._infer_final_state(goal, execution_results)
        
        # Compute semantic similarity between final state and goal
        semantic_similarity = self._compute_goal_similarity(goal, final_state)
        
        # Determine completion based on both semantic similarity and execution success
        # High similarity + good execution = completed
        # Low similarity or many failures = not completed
        completion_score = (semantic_similarity * 0.6) + (execution_success_rate * 0.4)
        completed = (
            semantic_similarity >= completion_threshold 
            and execution_success_rate >= 0.5
            and steps_failed < steps_executed
        )
        
        # Confidence is how certain we are about the completion status
        confidence = completion_score if completed else (1.0 - completion_score)
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        
        return GoalCompletion(
            goal=goal,
            completed=completed,
            confidence=confidence,
            semantic_similarity=semantic_similarity,
            execution_success_rate=execution_success_rate,
            steps_executed=steps_executed,
            steps_failed=steps_failed,
            steps_skipped=steps_skipped,
            final_state_description=final_state,
            failure_reasons=failure_reasons,
        )
    
    def _infer_final_state(
        self,
        goal: str,
        execution_results: List[Dict[str, Any]],
    ) -> str:
        """
        Infer a natural language description of the final state from results.
        
        Used when no explicit final_state is provided.
        """
        if not execution_results:
            return "no actions were executed"
        
        # Build description from successful actions
        successful_actions = []
        last_result = None
        
        for result in execution_results:
            if result.get("status") == "error":
                continue
            if result.get("status") == "skipped":
                continue
            
            action_name = result.get("action", "unknown")
            successful_actions.append(action_name)
            
            # Capture the last result content
            inner = result.get("result", {})
            if isinstance(inner, dict):
                if "output" in inner:
                    last_result = inner["output"]
                elif "message" in inner:
                    last_result = inner["message"]
            elif isinstance(inner, str):
                last_result = inner
        
        if not successful_actions:
            return f"failed to make progress toward {goal}"
        
        actions_desc = ", ".join(successful_actions[-3:])  # Last 3 actions
        
        if last_result:
            return f"completed actions [{actions_desc}], last result: {str(last_result)[:100]}"
        else:
            return f"completed actions [{actions_desc}] toward goal: {goal}"
    
    def _compute_goal_similarity(self, goal: str, final_state: str) -> float:
        """
        Compute semantic similarity between goal and final state.
        
        Uses PMFlow embeddings for comparison.
        """
        try:
            # Encode both
            goal_emb = self.encoder.encode([goal])
            state_emb = self.encoder.encode([final_state])
            
            # Ensure tensors (handle numpy arrays or existing tensors)
            if not isinstance(goal_emb, torch.Tensor):
                goal_emb = torch.as_tensor(goal_emb)
            if not isinstance(state_emb, torch.Tensor):
                state_emb = torch.as_tensor(state_emb)
            
            # Normalize and compute cosine similarity
            goal_norm = F.normalize(goal_emb.float(), p=2, dim=-1)
            state_norm = F.normalize(state_emb.float(), p=2, dim=-1)
            
            similarity = F.cosine_similarity(goal_norm, state_norm, dim=-1)
            return float(similarity.mean().item())
            
        except Exception as e:
            logger.warning(f"Could not compute goal similarity: {e}")
            return 0.5  # Default to uncertain
    
    def execute_and_check(
        self,
        plan: ExecutionPlan,
        transport,
        stop_on_error: bool = True,
        reactive: bool = False,
        max_replans: int = 5,
        completion_threshold: float = 0.7,
    ) -> Tuple[List[Dict[str, Any]], GoalCompletion]:
        """
        Execute a plan and check if the goal was completed.
        
        Convenience method that combines execute_plan() and check_goal_completion().
        
        Args:
            plan: The execution plan
            transport: Tool transport for execution
            stop_on_error: Stop on first error
            reactive: Use reactive replanning
            max_replans: Max replanning iterations
            completion_threshold: Minimum similarity for completion
            
        Returns:
            (execution_results, goal_completion)
        """
        results = self.execute_plan(
            plan=plan,
            transport=transport,
            stop_on_error=stop_on_error,
            reactive=reactive,
            max_replans=max_replans,
        )
        
        completion = self.check_goal_completion(
            goal=plan.goal,
            execution_results=results,
            completion_threshold=completion_threshold,
        )
        
        logger.info(f"Goal completion: {completion}")
        return results, completion
    
    def execute_and_learn(
        self,
        plan: ExecutionPlan,
        transport,
        stop_on_error: bool = True,
        reactive: bool = False,
        max_replans: int = 5,
        completion_threshold: float = 0.7,
    ) -> Tuple[List[Dict[str, Any]], GoalCompletion, Optional[str]]:
        """
        Execute a plan, check completion, and learn the sequence.
        
        Extends execute_and_check to store the action sequence in
        procedural memory for future recall.
        
        Args:
            plan: The execution plan
            transport: Tool transport for execution
            stop_on_error: Stop on first error
            reactive: Use reactive replanning
            max_replans: Max replanning iterations
            completion_threshold: Minimum similarity for completion
            
        Returns:
            (execution_results, goal_completion, sequence_id)
            sequence_id is None if procedural memory is disabled
        """
        results, completion = self.execute_and_check(
            plan=plan,
            transport=transport,
            stop_on_error=stop_on_error,
            reactive=reactive,
            max_replans=max_replans,
            completion_threshold=completion_threshold,
        )
        
        sequence_id = None
        if self.procedural_memory and plan.steps:
            # Extract action IDs and names from plan
            action_ids = [step.action.action_id for step in plan.steps]
            action_names = [step.action.name for step in plan.steps]
            
            sequence_id = self.procedural_memory.learn_sequence(
                goal=plan.goal,
                action_sequence=action_ids,
                completion=completion,
                action_names=action_names,
            )
            
            if sequence_id:
                logger.info(f"Learned sequence {sequence_id} for goal: {plan.goal[:50]}")
        
        return results, completion, sequence_id
    
    def plan_with_memory(
        self,
        goal: str,
        current_state: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        min_sequence_success_rate: float = 0.5,
        prefer_memory: bool = True,
    ) -> Optional[ExecutionPlan]:
        """
        Plan with procedural memory lookup before trajectory-based planning.
        
        If a successful sequence is found in procedural memory for a similar
        goal, it's converted to an ExecutionPlan. Otherwise, falls back to
        standard trajectory-based planning.
        
        Args:
            goal: The goal to achieve
            current_state: Optional description of current state
            context: Optional context for argument filling
            min_sequence_success_rate: Minimum success rate for recalled sequences
            prefer_memory: If True, prefer memory over trajectory planning
            
        Returns:
            ExecutionPlan or None
        """
        # Try procedural memory first
        if self.procedural_memory and prefer_memory:
            sequence = self.procedural_memory.get_best_sequence(
                goal=goal,
                min_success_rate=min_sequence_success_rate,
            )
            
            if sequence:
                plan = self._sequence_to_plan(sequence, goal, context or {})
                if plan:
                    logger.info(
                        f"Using recalled sequence {sequence.sequence_id} "
                        f"(success_rate={sequence.success_rate:.2f})"
                    )
                    return plan
        
        # Fall back to trajectory-based planning
        return self.plan(goal=goal, current_state=current_state, context=context)
    
    def _sequence_to_plan(
        self,
        sequence: LearnedSequence,
        goal: str,
        context: Dict[str, Any],
    ) -> Optional[ExecutionPlan]:
        """Convert a learned sequence to an execution plan."""
        steps = []
        
        for i, action_id in enumerate(sequence.action_ids):
            action = self._action_cache.get(action_id)
            if not action:
                logger.warning(f"Action {action_id} not found in cache, skipping sequence")
                return None
            
            args = self._fill_args(action.arg_template, context)
            
            step = PlannedStep(
                step_num=i + 1,
                action=action,
                args=args,
                waypoint_embedding=action.embedding if action.embedding is not None else torch.zeros(self.encoder.latent_dim if hasattr(self.encoder, 'latent_dim') else 32),
                confidence=sequence.reliability_score,
            )
            steps.append(step)
        
        return ExecutionPlan(
            goal=goal,
            steps=steps,
            trajectory_efficiency=1.0,  # Memory recall is "efficient"
            total_confidence=sequence.reliability_score,
            estimated_success=sequence.success_rate,
        )

    def _get_pm_field(self):
        """
        Get the underlying PMFlow field for native physics operations.
        
        Returns the field that has step(), mark_as_hazard(), etc.
        """
        if not hasattr(self.encoder, 'pm_field'):
            return None
        
        pm_field = self.encoder.pm_field
        
        # For MultiScalePMField, use fine_field (higher resolution)
        if hasattr(pm_field, 'fine_field'):
            return pm_field.fine_field
        
        return pm_field
    
    def _ground_point_to_action(
        self, 
        z: torch.Tensor, 
        context: Dict[str, Any],
    ) -> Optional[Tuple[PlannedStep, int]]:
        """
        Ground a single latent point to the nearest action.
        
        Returns (PlannedStep, action_center_idx) or None if no match.
        The center index is needed for gravity adjustments.
        """
        if self._action_embeddings is None:
            return None
        
        if z.dim() == 1:
            z = z.unsqueeze(0)
        
        # Normalize for cosine similarity
        z_norm = F.normalize(z, p=2, dim=-1)
        action_embs = F.normalize(self._action_embeddings, p=2, dim=-1)
        
        # Find best matching action
        similarities = F.cosine_similarity(z_norm, action_embs, dim=-1)
        best_idx = int(similarities.argmax().item())
        best_sim = float(similarities[best_idx].item())
        
        if best_sim < self.grounding_threshold:
            return None
        
        action_id = self._action_ids[best_idx]
        action = self._action_cache[action_id]
        
        # Fill arguments from context
        args = self._fill_args(action.arg_template, context)
        
        step = PlannedStep(
            step_num=0,  # Will be set by caller
            action=action,
            args=args,
            waypoint_embedding=z.squeeze(0),
            confidence=best_sim,
        )
        
        return step, best_idx
    
    def execute_native(
        self,
        plan_or_goal: Union["ExecutionPlan", str],
        transport,
        current_state: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        max_steps: int = 20,
        stop_on_error: bool = True,
        hazard_radius: float = 1.0,
        hazard_strength: float = -0.5,
        attractor_radius: float = 1.0,
        attractor_strength: float = 0.3,
        convergence_threshold: float = 0.05,
    ) -> List[Dict[str, Any]]:
        """
        Execute a goal using native PMFlow physics.
        
        This is the physics-faithful implementation that:
        - Stays in latent space during execution
        - Uses step-by-step trajectory evolution
        - Modifies gravitational field (μ, Ω) based on outcomes
        - Naturally curves around dead ends via mark_as_hazard()
        
        This embodies the Lilith philosophy: let physics handle
        branching rather than explicit conditionals.
        
        Args:
            plan_or_goal: ExecutionPlan or string goal description
            transport: LocalToolsTransport or compatible
            current_state: Optional starting state description
            context: Arguments to fill into action templates
            max_steps: Maximum execution steps
            stop_on_error: Stop on first error
            hazard_radius: Radius for marking failures as hazards
            hazard_strength: Repulsion strength for hazards (negative)
            attractor_radius: Radius for marking success as attractors
            attractor_strength: Attraction strength for successes
            convergence_threshold: Movement threshold to detect convergence
            
        Returns:
            List of execution results
        """
        # Extract goal from plan or use directly
        if isinstance(plan_or_goal, ExecutionPlan):
            goal = plan_or_goal.goal
        else:
            goal = plan_or_goal
        
        pm_field = self._get_pm_field()
        if pm_field is None:
            logger.error("Native execution requires encoder with pm_field attribute")
            return []
        
        if not hasattr(pm_field, 'step'):
            logger.error("Native execution requires PMFlow v0.3.5+ with step() method")
            return []
        
        self._ensure_embedding_cache()
        if self._action_embeddings is None or len(self._action_ids) == 0:
            logger.warning("No actions registered for native execution")
            return []
        
        context = context or {}
        results = []
        executed_actions = set()
        prev_z = None
        
        try:
            # Inject goal as gravitational attractor
            self.encoder.inject_intent(goal.split(), strength=0.5)
            
            # Initialize position in latent space
            start_text = current_state if current_state else "current state ready to act"
            z = self._encode_to_latent(start_text.split())
            if z.dim() == 1:
                z = z.unsqueeze(0)
            
            # Move tensor to same device as pm_field
            device = pm_field.centers.device
            z = z.to(device)
            
            for step_num in range(1, max_steps + 1):
                prev_z = z.clone()
                
                # Single physics step through the gravitational field
                z = pm_field.step(z)
                
                # Check for convergence
                if prev_z is not None:
                    movement = (z - prev_z).norm().item()
                    if movement < convergence_threshold:
                        logger.info(f"Native execution converged at step {step_num}")
                        break
                
                # Ground current position to nearest action
                grounding = self._ground_point_to_action(z.cpu(), context)
                
                if grounding is None:
                    # No action matches this point, continue evolving
                    continue
                
                planned_step, action_idx = grounding
                action = planned_step.action
                
                # Check for loops
                if action.action_id in executed_actions:
                    # Already executed this action, may be stuck
                    # Mark current position as mild hazard to encourage moving on
                    pm_field.mark_as_hazard(z.squeeze(), radius=hazard_radius * 0.5, repulsion_strength=hazard_strength * 0.3)
                    continue
                
                # Execute the action
                if not action.tool_binding:
                    results.append({
                        "step": step_num,
                        "action": action.name,
                        "status": "skipped",
                        "reason": "no tool binding",
                    })
                    continue
                
                message = {
                    "action": action.tool_binding,
                    "args": planned_step.args,
                }
                
                try:
                    result = transport.call(
                        name="action_planner",
                        message=message,
                        meta={"step": step_num, "goal": goal, "native": True},
                    )
                    
                    step_result = {
                        "step": step_num,
                        "action": action.name,
                        "tool": action.tool_binding,
                        "confidence": planned_step.confidence,
                        "result": result,
                    }
                    results.append(step_result)
                    executed_actions.add(action.action_id)
                    
                    # Evaluate outcome and modify gravitational field
                    is_error = isinstance(result, dict) and result.get("status") == "error"
                    is_empty = self._is_empty_result(result)
                    
                    if is_error or is_empty:
                        # Mark this region as hazard - trajectories will curve away
                        affected = pm_field.mark_as_hazard(
                            z.squeeze(), 
                            radius=hazard_radius, 
                            repulsion_strength=hazard_strength
                        )
                        logger.debug(f"Marked hazard at step {step_num}: {affected} centers affected")
                        
                        if is_error and stop_on_error:
                            logger.warning(f"Native execution stopped at step {step_num}: error")
                            break
                    else:
                        # Success - mark as attractor to reinforce this path
                        affected = pm_field.mark_as_attractor(
                            z.squeeze(),
                            radius=attractor_radius,
                            attraction_strength=attractor_strength
                        )
                        logger.debug(f"Marked attractor at step {step_num}: {affected} centers affected")
                        
                        # Check if we might have reached the goal
                        # (could be enhanced with goal proximity detection)
                        
                except Exception as e:
                    error_result = {
                        "step": step_num,
                        "action": action.name,
                        "status": "error",
                        "message": str(e),
                    }
                    results.append(error_result)
                    
                    # Mark as hazard
                    pm_field.mark_as_hazard(z.squeeze(), radius=hazard_radius, repulsion_strength=hazard_strength)
                    
                    if stop_on_error:
                        logger.error(f"Native execution failed at step {step_num}: {e}")
                        break
            
            logger.info(f"Native execution completed: {len(results)} actions in {max_steps} steps")
            
        finally:
            # Clear intent after execution
            if hasattr(self.encoder, 'clear_intent'):
                self.encoder.clear_intent()
        
        return results
    
    def execute_native_and_check(
        self,
        goal: str,
        transport,
        current_state: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        max_steps: int = 20,
        stop_on_error: bool = True,
        completion_threshold: float = 0.7,
        **kwargs,
    ) -> Tuple[List[Dict[str, Any]], GoalCompletion]:
        """
        Execute a goal using native PMFlow physics and check completion.
        
        Convenience method that combines execute_native() and check_goal_completion().
        
        Args:
            goal: The goal to achieve
            transport: Tool transport for execution
            current_state: Optional description of current state
            context: Optional context for argument filling
            max_steps: Maximum trajectory steps
            stop_on_error: Stop on first error
            completion_threshold: Minimum similarity for completion
            **kwargs: Additional args passed to execute_native
            
        Returns:
            (execution_results, goal_completion)
        """
        results = self.execute_native(
            plan_or_goal=goal,
            transport=transport,
            current_state=current_state,
            context=context,
            max_steps=max_steps,
            stop_on_error=stop_on_error,
            **kwargs,
        )
        
        completion = self.check_goal_completion(
            goal=goal,
            execution_results=results,
            completion_threshold=completion_threshold,
        )
        
        logger.info(f"Native goal completion: {completion}")
        return results, completion

    def _is_empty_result(self, result: Any) -> bool:
        """Check if a result indicates empty/not-found."""
        if result is None:
            return True
        if isinstance(result, str) and len(result.strip()) == 0:
            return True
        if isinstance(result, dict):
            if result.get("empty") or result.get("not_found"):
                return True
            if result.get("status") == "not_found":
                return True
            # Check for empty content in common patterns
            content = result.get("content") or result.get("data") or result.get("result")
            if content is not None and (content == "" or content == [] or content == {}):
                return True
        if isinstance(result, (list, dict)) and len(result) == 0:
            return True
        return False

    def _apply_result_to_field(
        self,
        pm_field,
        z: torch.Tensor,
        result: Any,
        hazard_radius: float = 1.0,
        hazard_strength: float = -0.5,
        attractor_radius: float = 1.0,
        attractor_strength: float = 0.3,
    ) -> None:
        """
        Apply execution result to the gravitational field.
        
        This translates execution outcomes to field modifications:
        - Errors → mark_as_hazard (repulsive region)
        - Empty results → mark_as_hazard (mild)
        - Complete success → mark_as_attractor (strong)
        - Partial success → mark_as_attractor (weak) or no change
        
        Args:
            pm_field: PMFlow field with mark_as_hazard/mark_as_attractor
            z: Current position in latent space
            result: Execution result from transport
            hazard_radius: Radius for hazard regions
            hazard_strength: Repulsion for hazards (negative)
            attractor_radius: Radius for attractors
            attractor_strength: Attraction strength (positive)
        """
        if z.dim() == 2:
            z = z.squeeze(0)
        
        is_error = isinstance(result, dict) and result.get("status") == "error"
        is_empty = self._is_empty_result(result)
        is_complete = isinstance(result, dict) and result.get("complete", False)
        
        if is_error:
            # Strong hazard for errors
            pm_field.mark_as_hazard(z, radius=hazard_radius, repulsion_strength=hazard_strength)
        elif is_empty:
            # Milder hazard for empty results
            pm_field.mark_as_hazard(z, radius=hazard_radius * 0.7, repulsion_strength=hazard_strength * 0.5)
        elif is_complete:
            # Strong attractor for complete success
            pm_field.mark_as_attractor(z, radius=attractor_radius, attraction_strength=attractor_strength)
        else:
            # Partial success - weak attractor
            pm_field.mark_as_attractor(z, radius=attractor_radius * 0.5, attraction_strength=attractor_strength * 0.3)


# Convenience function for creating action planner with default tools
def create_default_planner(encoder, graph) -> ActionPlanner:
    """
    Create an ActionPlanner with common actions pre-registered.
    
    Registers basic file, terminal, and navigation actions.
    """
    planner = ActionPlanner(encoder, graph)
    
    # File operations
    planner.register_action(
        name="read_file",
        description="read the contents of a file at a given path",
        tool_binding="read_file",
        arg_template={"path": "$file_path"},
    )
    
    planner.register_action(
        name="write_file",
        description="write content to a file at a given path creating if needed",
        tool_binding="write_file",
        arg_template={"path": "$file_path", "content": "$content"},
    )
    
    planner.register_action(
        name="list_directory",
        description="list files and folders in a directory",
        tool_binding="list_dir",
        arg_template={"path": "$directory_path"},
    )
    
    # Terminal operations
    planner.register_action(
        name="run_command",
        description="execute a shell command in the terminal",
        tool_binding="run_command",
        arg_template={"command": "$command"},
    )
    
    # Wait/delay
    planner.register_action(
        name="wait",
        description="pause and wait for a specified duration",
        tool_binding="wait",
        arg_template={"seconds": 1},
    )
    
    logger.info("Default action planner created with basic file/terminal actions")
    return planner


__all__ = [
    "ActionPlanner",
    "ActionNode", 
    "PlannedStep",
    "ExecutionPlan",
    "GoalCompletion",
    "LearnedSequence",
    "ProceduralMemory",
    "create_default_planner",
]
