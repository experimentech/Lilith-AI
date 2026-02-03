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
from typing import Any, Dict, List, Optional, Tuple
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


class ActionPlanner:
    """
    Plans multi-step actions using PMFlow's agentic physics.
    
    The planner uses the neuro-symbolic architecture:
    - Neural: Trajectory tracing through latent space toward goal
    - Symbolic: Action nodes in graph with tool bindings
    - Bridge: Grounding trajectory waypoints to nearest action nodes
    
    This enables physics-based "thinking about what to do" - the same
    mechanisms used for semantic reasoning now applied to action selection.
    """
    
    def __init__(
        self,
        encoder,  # PMFlowEmbeddingEncoder with enable_flow=True
        graph,  # RelationalGraphStore or MultiTenantGraphManager
        trajectory_steps: int = 10,
        grounding_threshold: float = 0.3,
        max_plan_length: int = 20,
    ):
        """
        Initialize action planner.
        
        Args:
            encoder: PMFlow encoder with agentic physics (enable_flow=True)
            graph: Knowledge graph containing action nodes
            trajectory_steps: Number of steps when tracing trajectory
            grounding_threshold: Minimum similarity to ground waypoint to action
            max_plan_length: Maximum steps in a plan
        """
        self.encoder = encoder
        self.graph = graph
        self.trajectory_steps = trajectory_steps
        self.grounding_threshold = grounding_threshold
        self.max_plan_length = max_plan_length
        
        # Check for agentic physics
        self._has_flow = hasattr(encoder, 'enable_flow') and encoder.enable_flow
        if not self._has_flow:
            logger.warning("ActionPlanner: encoder doesn't have enable_flow=True; planning limited")
        
        # Cache of action embeddings for fast grounding
        self._action_cache: Dict[str, ActionNode] = {}
        self._action_embeddings: Optional[torch.Tensor] = None
        self._action_ids: List[str] = []
        
        logger.info(f"ActionPlanner initialized (flow={'enabled' if self._has_flow else 'disabled'})")
    
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
    "create_default_planner",
]
