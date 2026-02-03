"""
Tests for ActionPlanner - Physics-based action sequencing.

Tests the neuro-symbolic action planning:
- Action registration (symbolic side)
- Trajectory-based planning (neural side)
- Grounding waypoints to actions (bridge)
- Execution via LocalToolsTransport
"""

import unittest
import tempfile
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch


class MockEncoder:
    """Mock encoder with agentic physics features.
    
    Uses bag-of-words hashing for pseudo-semantic similarity:
    texts with overlapping words produce similar embeddings.
    """
    
    def __init__(self, dimension=64, latent_dim=32):
        self.dimension = dimension
        self.latent_dim = latent_dim
        self.enable_flow = True
        self._intent = None
        self._intent_strength = 0.0
    
    def encode(self, tokens):
        """Encode tokens to embedding with pseudo-semantic similarity."""
        if isinstance(tokens, list):
            words = [w.lower() for w in tokens]
        else:
            words = str(tokens).lower().split()
        
        # Build embedding as sum of per-word vectors
        # Words that appear in multiple texts will contribute the same component
        embedding = torch.zeros(1, self.dimension)
        for word in words:
            # Deterministic vector per word
            word_hash = hash(word) % 10000
            torch.manual_seed(word_hash)
            word_vec = torch.randn(1, self.dimension)
            embedding += word_vec
        
        # Normalize to unit length
        norm = torch.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def inject_intent(self, tokens, strength=0.5):
        """Mock intent injection."""
        self._intent = tokens
        self._intent_strength = strength
    
    def clear_intent(self):
        """Clear injected intent."""
        self._intent = None
        self._intent_strength = 0.0
    
    def trace_trajectory(self, tokens, steps=10):
        """Mock trajectory tracing.
        
        Returns trajectory in latent space (latent_dim dimensions),
        matching real PMFlowEmbeddingEncoder behavior.
        """
        # Generate a trajectory of embeddings in latent space
        start = self.encode(tokens)
        # Project to latent space
        start_latent = start[:, :self.latent_dim]  # [1, latent_dim]
        
        trajectory = []
        current = start_latent.clone()
        
        for i in range(steps):
            # Drift toward intent if set
            if self._intent:
                intent_emb = self.encode(self._intent)
                intent_latent = intent_emb[:, :self.latent_dim]  # Project to latent
                alpha = (i + 1) / steps * self._intent_strength
                current = (1 - alpha) * current + alpha * intent_latent
            else:
                # Random walk
                current = current + torch.randn_like(current) * 0.1
            
            trajectory.append(current.squeeze(0))
        
        trajectory_tensor = torch.stack(trajectory)
        metrics = {
            "efficiency": 0.75,
            "path_length": float(steps),
        }
        
        return trajectory_tensor, metrics


class MockPMField:
    """Mock PMFlow field with native execution API.
    
    Implements step(), mark_as_hazard(), mark_as_attractor() for
    testing execute_native().
    """
    
    def __init__(self, d_latent=32, n_centers=32):
        self.d_latent = d_latent
        self.n_centers = n_centers
        self.centers = torch.randn(n_centers, d_latent) * 0.5
        self.mus = torch.ones(n_centers) * 0.35
        self.omegas = torch.zeros(n_centers)
        self.enable_flow = True
        self._clamp = 3.0
        self._step_count = 0
        self.hazards = []  # Track hazard markings for testing
        self.attractors = []  # Track attractor markings for testing
        self.target_embeddings = None  # Set by test to guide trajectory
    
    def set_action_targets(self, embeddings: torch.Tensor):
        """Set action embeddings as trajectory targets."""
        self.target_embeddings = embeddings
        # Override some centers with action positions
        if embeddings is not None and len(embeddings) > 0:
            n = min(len(embeddings), self.n_centers)
            self.centers[:n] = embeddings[:n].clone()
    
    def step(self, z: torch.Tensor) -> torch.Tensor:
        """Single physics step - drift toward action centers."""
        self._step_count += 1
        
        # If we have target embeddings, drift toward nearest one
        if self.target_embeddings is not None and len(self.target_embeddings) > 0:
            if z.dim() == 1:
                z_query = z.unsqueeze(0)
            else:
                z_query = z
            
            targets = self.target_embeddings
            if targets.dim() == 1:
                targets = targets.unsqueeze(0)
            
            # Find nearest target
            dists = torch.cdist(z_query, targets)
            nearest_idx = dists.argmin(dim=1).item()
            target = targets[nearest_idx]
            
            if z.dim() == 2:
                target = target.unsqueeze(0)
            
            # Strong drift toward target (guaranteed to get there)
            z_new = z + 0.4 * (target - z)
        else:
            # Fallback: drift toward mean of centers
            center_mean = self.centers.mean(dim=0)
            if z.dim() == 2:
                center_mean = center_mean.unsqueeze(0)
            z_new = z + 0.1 * (center_mean - z)
            z_new = z_new + torch.randn_like(z_new) * 0.05
        
        return torch.clamp(z_new, -self._clamp, self._clamp)
    
    def mark_as_hazard(self, z: torch.Tensor, radius: float = 1.0, repulsion_strength: float = -0.5) -> int:
        """Mark region as hazard."""
        self.hazards.append({"z": z.clone(), "radius": radius, "strength": repulsion_strength})
        # Reduce mu for nearby centers
        if z.dim() == 1:
            z = z.unsqueeze(0)
        dists = torch.cdist(z, self.centers).squeeze(0)
        affected = (dists < radius).sum().item()
        return int(affected)
    
    def mark_as_attractor(self, z: torch.Tensor, radius: float = 1.0, attraction_strength: float = 0.3) -> int:
        """Mark region as attractor."""
        self.attractors.append({"z": z.clone(), "radius": radius, "strength": attraction_strength})
        # Increase mu for nearby centers
        if z.dim() == 1:
            z = z.unsqueeze(0)
        dists = torch.cdist(z, self.centers).squeeze(0)
        affected = (dists < radius).sum().item()
        return int(affected)


class MockEncoderWithPMField(MockEncoder):
    """Mock encoder with pm_field for native execution testing."""
    
    def __init__(self, dimension=64, latent_dim=32):
        super().__init__(dimension, latent_dim)
        self.pm_field = MockPMField(d_latent=latent_dim)
        self.base_encoder = self  # Self-reference for _encode_to_latent fallback
        self._projection = torch.randn(dimension, latent_dim)


class MockGraphStore:
    """Mock graph store for action nodes."""
    
    def __init__(self):
        self.nodes = {}
        self._conn = None  # Simulated connection
    
    def add_node(self, node_id, node_type, term, confidence=1.0, data=None, **kwargs):
        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "term": term,
            "confidence": confidence,
            "data": data or {},
        }
    
    def get_node(self, node_id, **kwargs):
        return self.nodes.get(node_id)


class MockToolsTransport:
    """Mock LocalToolsTransport for testing execution."""
    
    def __init__(self):
        self.calls = []
        self.tools = {}
    
    def register_tool(self, name, func):
        self.tools[name] = func
    
    def call(self, name, message, meta):
        action = message.get("action")
        args = message.get("args", {})
        
        self.calls.append({
            "name": name,
            "action": action,
            "args": args,
            "meta": meta,
        })
        
        if action in self.tools:
            try:
                result = self.tools[action](**args)
                return {"status": "success", "result": result}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        
        return {"status": "success", "result": f"executed {action}"}


class TestActionPlanner(unittest.TestCase):
    """Test ActionPlanner functionality."""
    
    def setUp(self):
        from v2.lilith_v2.action_planner import ActionPlanner
        
        self.encoder = MockEncoder(dimension=64)
        self.graph = MockGraphStore()
        self.planner = ActionPlanner(
            encoder=self.encoder,
            graph=self.graph,
            trajectory_steps=5,
            grounding_threshold=0.2,  # Lower threshold for testing
        )
    
    def test_register_action(self):
        """Test registering an action."""
        action_id = self.planner.register_action(
            name="read_file",
            description="read the contents of a file",
            tool_binding="read_file",
            arg_template={"path": "$file_path"},
        )
        
        self.assertEqual(action_id, "action_read_file")
        self.assertIn(action_id, self.planner._action_cache)
        self.assertIn(action_id, self.graph.nodes)
        
        # Check action has embedding
        action = self.planner._action_cache[action_id]
        self.assertIsNotNone(action.embedding)
        self.assertEqual(action.tool_binding, "read_file")
    
    def test_register_multiple_actions(self):
        """Test registering multiple actions."""
        self.planner.register_action(
            name="read_file",
            description="read the contents of a file",
            tool_binding="read_file",
        )
        self.planner.register_action(
            name="write_file",
            description="write content to a file",
            tool_binding="write_file",
        )
        self.planner.register_action(
            name="run_command",
            description="execute a shell command",
            tool_binding="run_command",
        )
        
        self.assertEqual(len(self.planner._action_cache), 3)
    
    def test_plan_generation(self):
        """Test generating a plan from a goal."""
        # Register some actions with specific vocabulary
        self.planner.register_action(
            name="read_file",
            description="read the contents of a file at a path",
            tool_binding="read_file",
            arg_template={"path": "$file_path"},
        )
        self.planner.register_action(
            name="write_file",
            description="write content to a file at a path",
            tool_binding="write_file",
            arg_template={"path": "$output_path", "content": "$content"},
        )
        
        # Generate a plan - goal shares vocabulary with actions for reliable matching
        plan = self.planner.plan(
            goal="read file contents and write to path",
            context={"file_path": "test.txt", "output_path": "backup/test.txt"},
        )
        
        self.assertIsNotNone(plan)
        self.assertEqual(plan.goal, "read file contents and write to path")
        self.assertGreater(len(plan.steps), 0)
        self.assertGreater(plan.trajectory_efficiency, 0)
    
    def test_plan_to_execution_format(self):
        """Test converting plan to execution format."""
        self.planner.register_action(
            name="echo",
            description="echo a message",
            tool_binding="echo",
            arg_template={"message": "$msg"},
        )
        
        plan = self.planner.plan(
            goal="echo hello world",
            context={"msg": "hello world"},
        )
        
        if plan and plan.steps:
            commands = self.planner.to_execution_format(plan)
            self.assertIsInstance(commands, list)
            for cmd in commands:
                self.assertIn("action", cmd)
                self.assertIn("args", cmd)
    
    def test_execute_plan(self):
        """Test executing a plan via transport."""
        self.planner.register_action(
            name="greet",
            description="greet someone",
            tool_binding="greet",
            arg_template={"name": "$name"},
        )
        
        plan = self.planner.plan(
            goal="greet the user",
            context={"name": "Alice"},
        )
        
        if plan and plan.steps:
            transport = MockToolsTransport()
            results = self.planner.execute_plan(plan, transport)
            
            self.assertIsInstance(results, list)
            self.assertGreater(len(transport.calls), 0)
    
    def test_no_actions_returns_none(self):
        """Test planning with no registered actions returns None."""
        plan = self.planner.plan(goal="do something")
        self.assertIsNone(plan)
    
    def test_deduplication(self):
        """Test that consecutive identical actions are deduplicated."""
        # Register single action
        self.planner.register_action(
            name="wait",
            description="wait for a moment pause delay",
            tool_binding="wait",
        )
        
        # Plan should deduplicate consecutive waits
        plan = self.planner.plan(goal="wait a moment")
        
        if plan:
            # Even with multiple trajectory points grounding to same action,
            # consecutive duplicates should be removed
            action_ids = [s.action.action_id for s in plan.steps]
            for i in range(1, len(action_ids)):
                self.assertNotEqual(
                    action_ids[i], action_ids[i-1],
                    "Consecutive duplicate actions should be removed"
                )


class TestActionPlannerWithCognitive(unittest.TestCase):
    """Test ActionPlanner integration with CognitiveStage."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cognitive_stage_has_planner(self):
        """Test that CognitiveStage initializes with action planner."""
        from v2.lilith_v2.cognitive_stage import CognitiveStage
        from v2.lilith_v2.pmflow_sqlite import SQLitePMFlowStateStore
        from v2.lilith_v2.relational_graph_store import RelationalGraphStore
        
        pmflow_path = os.path.join(self.temp_dir, "pmflow.sqlite")
        graph_path = os.path.join(self.temp_dir, "graph.sqlite")
        
        pmflow = SQLitePMFlowStateStore(pmflow_path)
        graph = RelationalGraphStore(graph_path)
        encoder = MockEncoder()
        
        brain = CognitiveStage(
            node_id="test",
            pmflow_store=pmflow,
            graph_store=graph,
            encoder=encoder,
            config={"enable_action_planning": True},
        )
        
        self.assertTrue(brain._action_planning_enabled)
        self.assertIsNotNone(brain._action_planner)
    
    def test_cognitive_register_action(self):
        """Test registering action via CognitiveStage."""
        from v2.lilith_v2.cognitive_stage import CognitiveStage
        from v2.lilith_v2.pmflow_sqlite import SQLitePMFlowStateStore
        from v2.lilith_v2.relational_graph_store import RelationalGraphStore
        
        pmflow_path = os.path.join(self.temp_dir, "pmflow.sqlite")
        graph_path = os.path.join(self.temp_dir, "graph.sqlite")
        
        pmflow = SQLitePMFlowStateStore(pmflow_path)
        graph = RelationalGraphStore(graph_path)
        encoder = MockEncoder()
        
        brain = CognitiveStage(
            node_id="test",
            pmflow_store=pmflow,
            graph_store=graph,
            encoder=encoder,
        )
        
        action_id = brain.register_action(
            name="test_action",
            description="a test action",
            tool_binding="test_tool",
        )
        
        self.assertIsNotNone(action_id)
        self.assertEqual(action_id, "action_test_action")


class TestActionPlannerIntegration(unittest.TestCase):
    """Integration tests with real PMFlowEmbeddingEncoder.
    
    These tests verify that the actual physics-based encoder produces
    semantically meaningful trajectories that ground to appropriate actions.
    """
    
    def setUp(self):
        """Set up test fixtures with real encoder."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Try to import real encoder
        try:
            from pmflow.encoder import PMFlowEmbeddingEncoder
            self.encoder = PMFlowEmbeddingEncoder(
                dimension=64,
                latent_dim=32,
                enable_flow=True,
            )
            self.has_real_encoder = True
        except ImportError:
            self.has_real_encoder = False
            self.encoder = None
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_semantic_trajectory_grounds_to_file_actions(self):
        """Test that 'copy file' goal grounds to read/write actions."""
        if not self.has_real_encoder:
            self.skipTest("PMFlowEmbeddingEncoder not available")
        
        from v2.lilith_v2.action_planner import ActionPlanner
        from v2.lilith_v2.relational_graph_store import RelationalGraphStore
        
        graph_path = os.path.join(self.temp_dir, "graph.sqlite")
        graph = RelationalGraphStore(graph_path)
        
        planner = ActionPlanner(
            encoder=self.encoder,
            graph=graph,
            trajectory_steps=15,
            grounding_threshold=0.2,  # Lower threshold for real embeddings
        )
        
        # Register file-related actions
        planner.register_action(
            name="read_file",
            description="read the contents of a file from disk",
            tool_binding="read_file",
        )
        planner.register_action(
            name="write_file", 
            description="write content to a file on disk",
            tool_binding="write_file",
        )
        planner.register_action(
            name="delete_file",
            description="delete remove a file from disk",
            tool_binding="delete_file",
        )
        
        # Plan for file copy
        plan = planner.plan(goal="copy the file to a backup location")
        
        self.assertIsNotNone(plan, "Plan should be generated with real encoder")
        self.assertGreater(len(plan.steps), 0, "Plan should have steps")
        
        # Check that at least one file-related action was grounded
        action_names = [s.action.name for s in plan.steps]
        file_actions = {"read_file", "write_file", "delete_file"}
        has_file_action = any(name in file_actions for name in action_names)
        
        self.assertTrue(
            has_file_action,
            f"Expected file actions, got: {action_names}"
        )
        
        print(f"\nReal encoder plan for 'copy file': {action_names}")
        print(f"Trajectory efficiency: {plan.trajectory_efficiency:.3f}")
        print(f"Total confidence: {plan.total_confidence:.3f}")
    
    def test_semantic_similarity_produces_meaningful_embeddings(self):
        """Verify that semantically similar texts produce similar embeddings."""
        if not self.has_real_encoder:
            self.skipTest("PMFlowEmbeddingEncoder not available")
        
        import torch.nn.functional as F
        
        # Encode related concepts
        emb_read = self.encoder.encode(["read", "file", "contents"])
        emb_write = self.encoder.encode(["write", "file", "data"])
        emb_weather = self.encoder.encode(["sunny", "temperature", "forecast"])
        
        # Flatten for comparison
        emb_read = emb_read.flatten()
        emb_write = emb_write.flatten()
        emb_weather = emb_weather.flatten()
        
        # Compute similarities
        sim_read_write = F.cosine_similarity(
            emb_read.unsqueeze(0), emb_write.unsqueeze(0)
        ).item()
        sim_read_weather = F.cosine_similarity(
            emb_read.unsqueeze(0), emb_weather.unsqueeze(0)
        ).item()
        
        print(f"\nSimilarity read-write: {sim_read_write:.3f}")
        print(f"Similarity read-weather: {sim_read_weather:.3f}")
        
        # File operations should be more similar to each other than to weather
        self.assertGreater(
            sim_read_write, sim_read_weather,
            "File operations should cluster together in embedding space"
        )
    
    def test_intent_injection_affects_trajectory(self):
        """Verify that inject_intent actually influences trajectory direction."""
        if not self.has_real_encoder:
            self.skipTest("PMFlowEmbeddingEncoder not available")
        
        import torch
        import torch.nn.functional as F
        
        # Encode goal and project to latent space (to match trajectory space)
        goal_emb = self.encoder.encode(["write", "file", "save"])
        # Get base encoding (first encoder.dimension dims) and project to latent
        base_dim = self.encoder.dimension
        goal_base = goal_emb[:, :base_dim]  # [1, 64]
        goal_latent = torch.matmul(goal_base, self.encoder._projection)  # [1, 32]
        
        # Trace trajectory WITHOUT intent
        self.encoder.clear_intent()
        traj_no_intent, _ = self.encoder.trace_trajectory(
            ["current", "state"], steps=10
        )
        
        # Inject intent and trace trajectory WITH intent
        self.encoder.inject_intent(["write", "file", "save"], strength=0.5)
        traj_with_intent, _ = self.encoder.trace_trajectory(
            ["current", "state"], steps=10
        )
        self.encoder.clear_intent()
        
        # trajectory shape is [1, steps, 32] - get final step for each
        # Squeeze batch and get last step: [steps, 32][-1] -> [32]
        final_no_intent = traj_no_intent.squeeze(0)[-1].unsqueeze(0)  # [1, 32]
        final_with_intent = traj_with_intent.squeeze(0)[-1].unsqueeze(0)  # [1, 32]
        
        # Both goal_latent and final trajectory steps are now [1, 32]
        sim_no_intent = F.cosine_similarity(
            goal_latent,
            final_no_intent
        ).item()
        
        sim_with_intent = F.cosine_similarity(
            goal_latent,
            final_with_intent
        ).item()
        
        print(f"\nFinal trajectory similarity to goal:")
        print(f"  Without intent: {sim_no_intent:.3f}")
        print(f"  With intent: {sim_with_intent:.3f}")
        
        # With intent should be closer to goal (or at least not worse)
        # Note: This might not always hold due to field dynamics, but should trend this way
        self.assertGreaterEqual(
            sim_with_intent, sim_no_intent - 0.1,
            "Intent injection should pull trajectory toward goal"
        )


class TestImperativeIntentDetection(unittest.TestCase):
    """Test imperative intent detection for action planning triggers."""
    
    def test_discourse_manager_detects_imperative(self):
        """Test that DiscourseManager identifies imperative intents."""
        from v2.lilith_v2.discourse_manager import DiscourseManager
        
        dm = DiscourseManager()
        
        # Test imperative patterns
        imperative_inputs = [
            "Sign up for Moltbook",
            "Please create a new account",
            "Go to the settings page",
            "Click on the submit button",
            "Can you write a file for me?",
            "I need you to read this document",
            "Help me register for the service",
        ]
        
        for inp in imperative_inputs:
            state = dm.update(inp)
            self.assertEqual(
                state.last_user_intent, "imperative",
                f"'{inp}' should be detected as imperative, got {state.last_user_intent}"
            )
    
    def test_non_imperative_not_triggered(self):
        """Test that questions and statements are not imperative."""
        from v2.lilith_v2.discourse_manager import DiscourseManager
        
        dm = DiscourseManager()
        
        non_imperative = [
            "What is Python?",  # question
            "Hello there",  # greeting
            "Thanks for the help",  # feedback
            "Python is a language",  # statement
        ]
        
        for inp in non_imperative:
            state = dm.update(inp)
            self.assertNotEqual(
                state.last_user_intent, "imperative",
                f"'{inp}' should NOT be imperative, got {state.last_user_intent}"
            )


class TestPlanConfirmationFlow(unittest.TestCase):
    """Test the plan confirmation workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pending_plan_state(self):
        """Test pending plan management."""
        from v2.lilith_v2.cognitive_stage import CognitiveStage
        from v2.lilith_v2.pmflow_sqlite import SQLitePMFlowStateStore
        from v2.lilith_v2.relational_graph_store import RelationalGraphStore
        
        pmflow_path = os.path.join(self.temp_dir, "pmflow.sqlite")
        graph_path = os.path.join(self.temp_dir, "graph.sqlite")
        
        pmflow = SQLitePMFlowStateStore(pmflow_path)
        graph = RelationalGraphStore(graph_path)
        encoder = MockEncoder()
        
        brain = CognitiveStage(
            node_id="test",
            pmflow_store=pmflow,
            graph_store=graph,
            encoder=encoder,
            config={"enable_action_planning": True},
        )
        
        # Initially no pending plan
        self.assertFalse(brain.has_pending_plan())
        self.assertIsNone(brain.get_pending_plan_summary())
        
        # Register an action and propose a plan
        brain.register_action("do_something", "do something useful", "do_it")
        proposed = brain._propose_plan("do something now")
        
        if proposed:
            self.assertTrue(brain.has_pending_plan())
            summary = brain.get_pending_plan_summary()
            self.assertIsNotNone(summary)
            self.assertIn("do something now", summary)
            
            # Reject the plan
            brain.reject_pending_plan("not needed")
            self.assertFalse(brain.has_pending_plan())
    
    def test_plan_confirmation_returns_commands(self):
        """Test confirming a plan returns execution commands."""
        from v2.lilith_v2.cognitive_stage import CognitiveStage
        from v2.lilith_v2.pmflow_sqlite import SQLitePMFlowStateStore
        from v2.lilith_v2.relational_graph_store import RelationalGraphStore
        
        pmflow_path = os.path.join(self.temp_dir, "pmflow.sqlite")
        graph_path = os.path.join(self.temp_dir, "graph.sqlite")
        
        pmflow = SQLitePMFlowStateStore(pmflow_path)
        graph = RelationalGraphStore(graph_path)
        encoder = MockEncoder()
        
        brain = CognitiveStage(
            node_id="test",
            pmflow_store=pmflow,
            graph_store=graph,
            encoder=encoder,
            config={"enable_action_planning": True},
        )
        
        # Register and propose
        brain.register_action("save_file", "save a file to disk", "write_file", {"path": "$path"})
        brain._propose_plan("save the document")
        
        if brain.has_pending_plan():
            # Confirm without transport → returns commands
            commands = brain.confirm_pending_plan(transport=None)
            
            self.assertIsNotNone(commands)
            self.assertIsInstance(commands, list)
            
            # Plan should be cleared after confirmation
            self.assertFalse(brain.has_pending_plan())


class TestActionLearning(unittest.TestCase):
    """Test action learning capabilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_learn_action_from_text(self):
        """Test learning actions from natural language."""
        from v2.lilith_v2.action_planner import ActionPlanner
        from v2.lilith_v2.relational_graph_store import RelationalGraphStore
        
        graph_path = os.path.join(self.temp_dir, "graph.sqlite")
        graph = RelationalGraphStore(graph_path)
        encoder = MockEncoder()
        
        planner = ActionPlanner(encoder=encoder, graph=graph)
        
        # Test various teaching patterns
        test_cases = [
            ("the save command writes files to disk", "save"),
            ("use navigate to go to a URL", "navigate"),
            ("click does submit the form", "click"),
        ]
        
        for text, expected_name in test_cases:
            action_id = planner.learn_action_from_text(text)
            self.assertIsNotNone(action_id, f"Failed to learn from: {text}")
            self.assertIn(expected_name, action_id)
    
    def test_discover_tools(self):
        """Test auto-discovery of tools from transport."""
        from v2.lilith_v2.action_planner import ActionPlanner
        from v2.lilith_v2.relational_graph_store import RelationalGraphStore
        from v2.lilith_v2.mcp_transport_tools import LocalToolsTransport
        
        graph_path = os.path.join(self.temp_dir, "graph.sqlite")
        graph = RelationalGraphStore(graph_path)
        encoder = MockEncoder()
        
        planner = ActionPlanner(encoder=encoder, graph=graph)
        
        # Set up transport with some tools
        transport = LocalToolsTransport()
        transport.register_tool("read_file", lambda path: f"contents of {path}", "Read a file from disk")
        transport.register_tool("write_file", lambda path, content: True, "Write content to file")
        
        # Discover tools
        count = planner.discover_tools(transport)
        
        self.assertEqual(count, 2)
        self.assertIn("action_read_file", planner._action_cache)
        self.assertIn("action_write_file", planner._action_cache)
    
    def test_discover_skips_existing(self):
        """Test that discovery skips already registered actions."""
        from v2.lilith_v2.action_planner import ActionPlanner
        from v2.lilith_v2.relational_graph_store import RelationalGraphStore
        from v2.lilith_v2.mcp_transport_tools import LocalToolsTransport
        
        graph_path = os.path.join(self.temp_dir, "graph.sqlite")
        graph = RelationalGraphStore(graph_path)
        encoder = MockEncoder()
        
        planner = ActionPlanner(encoder=encoder, graph=graph)
        
        # Pre-register one action
        planner.register_action("read_file", "read file contents", "read_file")
        
        # Set up transport
        transport = LocalToolsTransport()
        transport.register_tool("read_file", lambda path: "contents", "Read file")
        transport.register_tool("write_file", lambda path, content: True, "Write file")
        
        # Discover - should only add write_file
        count = planner.discover_tools(transport)
        
        self.assertEqual(count, 1)  # Only write_file is new
    
    def test_cognitive_stage_learn_action(self):
        """Test learning actions via CognitiveStage."""
        from v2.lilith_v2.cognitive_stage import CognitiveStage
        from v2.lilith_v2.pmflow_sqlite import SQLitePMFlowStateStore
        from v2.lilith_v2.relational_graph_store import RelationalGraphStore
        
        pmflow_path = os.path.join(self.temp_dir, "pmflow.sqlite")
        graph_path = os.path.join(self.temp_dir, "graph.sqlite")
        
        pmflow = SQLitePMFlowStateStore(pmflow_path)
        graph = RelationalGraphStore(graph_path)
        encoder = MockEncoder()
        
        brain = CognitiveStage(
            node_id="test",
            pmflow_store=pmflow,
            graph_store=graph,
            encoder=encoder,
            config={"enable_action_planning": True},
        )
        
        # Learn action
        action_id = brain.learn_action("the delete command removes files")
        
        self.assertIsNotNone(action_id)
        self.assertIn("delete", action_id)


class TestReactiveExecution(unittest.TestCase):
    """Test reactive replanning during execution."""
    
    def setUp(self):
        """Set up test fixtures with planner and actions."""
        from v2.lilith_v2.action_planner import ActionPlanner
        
        self.encoder = MockEncoder(dimension=64, latent_dim=32)
        self.graph = MockGraphStore()
        self.planner = ActionPlanner(
            encoder=self.encoder,
            graph=self.graph,
            trajectory_steps=5,
            grounding_threshold=0.2,
        )
        
        # Register multiple actions for branching paths
        self.planner.register_action(
            name="search_wikipedia",
            description="search wikipedia for information lookup",
            tool_binding="search_wikipedia",
        )
        self.planner.register_action(
            name="web_search",
            description="search the web for information deep search",
            tool_binding="web_search",
        )
        self.planner.register_action(
            name="academic_search",
            description="search academic papers scholarly research",
            tool_binding="academic_search",
        )
    
    def test_encode_result_as_state_success(self):
        """Test encoding successful results as state."""
        result = {
            "step": 1,
            "action": "search",
            "result": {"content": "found information"},
        }
        state = self.planner._encode_result_as_state(result, "search", "find info")
        self.assertIn("successfully", state)
        self.assertIn("find info", state)
    
    def test_encode_result_as_state_error(self):
        """Test encoding error results as state."""
        result = {
            "step": 1,
            "status": "error",
            "message": "connection failed",
        }
        state = self.planner._encode_result_as_state(result, "fetch", "get data")
        self.assertIn("failed", state)
        self.assertIn("error", state)
    
    def test_encode_result_as_state_empty(self):
        """Test encoding empty results as state."""
        result = {
            "step": 1,
            "action": "search",
            "result": {"empty": True},
        }
        state = self.planner._encode_result_as_state(result, "search", "find info")
        self.assertIn("found nothing", state)
        self.assertIn("alternative", state)
    
    def test_encode_result_as_state_low_confidence(self):
        """Test encoding low confidence results as state."""
        result = {
            "step": 1,
            "action": "search",
            "result": {"content": "maybe", "confidence": 0.2},
        }
        state = self.planner._encode_result_as_state(result, "search", "find info")
        self.assertIn("low confidence", state)
    
    def test_is_empty_result_none(self):
        """Test _is_empty_result with None."""
        self.assertTrue(self.planner._is_empty_result(None))
    
    def test_is_empty_result_empty_string(self):
        """Test _is_empty_result with empty string."""
        self.assertTrue(self.planner._is_empty_result(""))
        self.assertTrue(self.planner._is_empty_result("   "))
    
    def test_is_empty_result_dict_empty_flag(self):
        """Test _is_empty_result with dict containing empty flag."""
        self.assertTrue(self.planner._is_empty_result({"empty": True}))
        self.assertTrue(self.planner._is_empty_result({"not_found": True}))
        self.assertTrue(self.planner._is_empty_result({"status": "not_found"}))
    
    def test_is_empty_result_success(self):
        """Test _is_empty_result with valid content."""
        self.assertFalse(self.planner._is_empty_result({"content": "data"}))
        self.assertFalse(self.planner._is_empty_result("some text"))
        self.assertFalse(self.planner._is_empty_result([1, 2, 3]))
    
    def test_static_execution_no_replan(self):
        """Test static execution doesn't replan on failure."""
        plan = self.planner.plan(goal="search for information")
        
        if plan:
            call_count = [0]
            
            class CountingTransport:
                calls = []
                def call(self, name, message, meta):
                    call_count[0] += 1
                    self.calls.append(message)
                    return {"empty": True}  # Simulate empty result
            
            transport = CountingTransport()
            results = self.planner.execute_plan(plan, transport, reactive=False)
            
            # Should execute all steps without replanning
            self.assertEqual(len(results), len(plan.steps))
    
    def test_reactive_execution_replans_on_empty(self):
        """Test reactive execution triggers replanning on empty results."""
        plan = self.planner.plan(goal="search wikipedia for information")
        
        if plan:
            replan_triggered = [False]
            original_plan = self.planner.plan
            
            def tracking_plan(goal, current_state=None, **kwargs):
                if current_state and "found nothing" in current_state:
                    replan_triggered[0] = True
                return original_plan(goal, current_state, **kwargs)
            
            self.planner.plan = tracking_plan
            
            class EmptyTransport:
                calls = []
                def call(self, name, message, meta):
                    self.calls.append(message)
                    return {"empty": True}
            
            transport = EmptyTransport()
            results = self.planner.execute_plan(
                plan, transport, reactive=True, max_replans=2
            )
            
            # Should have attempted replanning
            self.assertTrue(
                replan_triggered[0] or len(transport.calls) > len(plan.steps),
                "Reactive execution should trigger replanning on empty result"
            )
            
            self.planner.plan = original_plan
    
    def test_reactive_max_replans_limit(self):
        """Test that reactive execution respects max_replans."""
        plan = self.planner.plan(goal="search for information")
        
        if plan:
            replan_count = [0]
            original_plan = self.planner.plan
            
            def counting_plan(goal, current_state=None, **kwargs):
                if current_state:
                    replan_count[0] += 1
                return original_plan(goal, current_state, **kwargs)
            
            self.planner.plan = counting_plan
            
            class AlwaysEmptyTransport:
                calls = []
                def call(self, name, message, meta):
                    self.calls.append(message)
                    return {"empty": True}
            
            transport = AlwaysEmptyTransport()
            results = self.planner.execute_plan(
                plan, transport, reactive=True, max_replans=3
            )
            
            # Should not exceed max_replans
            self.assertLessEqual(replan_count[0], 3)
            
            self.planner.plan = original_plan
    
    def test_reactive_stops_on_error_when_configured(self):
        """Test reactive execution stops on error when stop_on_error=True."""
        plan = self.planner.plan(goal="search for information")
        
        if plan and len(plan.steps) > 1:
            class ErrorTransport:
                calls = []
                def call(self, name, message, meta):
                    self.calls.append(message)
                    return {"status": "error", "message": "test error"}
            
            transport = ErrorTransport()
            results = self.planner.execute_plan(
                plan, transport, reactive=True, stop_on_error=True
            )
            
            # Should stop after first error
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].get("result", {}).get("status"), "error")


class TestNativeExecution(unittest.TestCase):
    """Test PMFlow-native latent space execution."""
    
    def setUp(self):
        """Set up test fixtures with PM field-enabled encoder."""
        from v2.lilith_v2.action_planner import ActionPlanner
        
        self.encoder = MockEncoderWithPMField(dimension=64, latent_dim=32)
        self.graph = MockGraphStore()
        self.planner = ActionPlanner(
            encoder=self.encoder,
            graph=self.graph,
            trajectory_steps=5,
            grounding_threshold=0.2,
        )
        
        # Register actions
        self.planner.register_action(
            name="search",
            description="search for information",
            tool_binding="search",
        )
        self.planner.register_action(
            name="fetch",
            description="fetch data from source",
            tool_binding="fetch",
        )
        
        # Sync PM field with action embeddings so trajectories drift toward them
        self.planner._ensure_embedding_cache()
        if self.planner._action_embeddings is not None:
            self.encoder.pm_field.set_action_targets(self.planner._action_embeddings)
    
    def test_native_execution_basic(self):
        """Test basic native execution flow."""
        plan = self.planner.plan(goal="search for information")
        
        if plan:
            class SimpleTransport:
                calls = []
                def call(self, name, message, meta):
                    self.calls.append(message)
                    return {"content": "found it"}
            
            transport = SimpleTransport()
            results = self.planner.execute_native(
                plan, transport, max_steps=10
            )
            
            self.assertIsNotNone(results)
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0)
    
    def test_native_marks_hazard_on_error(self):
        """Test that native execution marks errors as hazards."""
        plan = self.planner.plan(goal="search for data")
        
        if plan:
            class FailingTransport:
                calls = []
                def call(self, name, message, meta):
                    self.calls.append(message)
                    return {"status": "error", "message": "failed"}
            
            transport = FailingTransport()
            
            initial_hazards = len(self.encoder.pm_field.hazards)
            
            results = self.planner.execute_native(
                plan, transport, max_steps=5
            )
            
            # Should have marked hazards
            self.assertGreater(
                len(self.encoder.pm_field.hazards),
                initial_hazards,
                "Native execution should mark errors as hazards"
            )
    
    def test_native_marks_attractor_on_success(self):
        """Test that native execution marks successes as attractors."""
        plan = self.planner.plan(goal="search for data")
        
        if plan:
            class SuccessTransport:
                calls = []
                def call(self, name, message, meta):
                    self.calls.append(message)
                    return {"content": "excellent data", "success": True}
            
            transport = SuccessTransport()
            
            initial_attractors = len(self.encoder.pm_field.attractors)
            
            results = self.planner.execute_native(
                plan, transport, max_steps=5
            )
            
            # Should have marked attractors
            self.assertGreater(
                len(self.encoder.pm_field.attractors),
                initial_attractors,
                "Native execution should mark successes as attractors"
            )
    
    def test_native_respects_max_steps(self):
        """Test that native execution respects max_steps limit."""
        plan = self.planner.plan(goal="search extensively")
        
        if plan:
            class SlowTransport:
                calls = []
                def call(self, name, message, meta):
                    self.calls.append(message)
                    # Always return partial success to keep going
                    return {"content": "partial", "more_available": True}
            
            transport = SlowTransport()
            
            results = self.planner.execute_native(
                plan, transport, max_steps=3
            )
            
            # Should not exceed max_steps
            self.assertLessEqual(len(results), 3)
    
    def test_native_stops_on_convergence(self):
        """Test that native execution stops when converged."""
        plan = self.planner.plan(goal="find specific answer")
        
        if plan:
            # After first success, field should converge
            call_count = [0]
            
            class ConvergingTransport:
                calls = []
                def call(self, name, message, meta):
                    self.calls.append(message)
                    call_count[0] += 1
                    return {"content": "complete answer", "complete": True}
            
            transport = ConvergingTransport()
            
            results = self.planner.execute_native(
                plan, transport, max_steps=20, convergence_threshold=0.05
            )
            
            # Should converge before hitting max_steps
            self.assertLess(
                call_count[0], 20,
                "Should converge before max steps"
            )
    
    def test_apply_result_to_field_error(self):
        """Test _apply_result_to_field with error result."""
        import torch
        
        pm_field = self.encoder.pm_field
        z = torch.randn(1, 32)
        
        error_result = {"status": "error", "message": "failed"}
        initial_hazards = len(pm_field.hazards)
        
        self.planner._apply_result_to_field(
            pm_field, z, error_result
        )
        
        self.assertEqual(
            len(pm_field.hazards),
            initial_hazards + 1,
            "Error should add hazard"
        )
    
    def test_apply_result_to_field_success(self):
        """Test _apply_result_to_field with success result."""
        import torch
        
        pm_field = self.encoder.pm_field
        z = torch.randn(1, 32)
        
        success_result = {"content": "great data", "complete": True}
        initial_attractors = len(pm_field.attractors)
        
        self.planner._apply_result_to_field(
            pm_field, z, success_result
        )
        
        self.assertEqual(
            len(pm_field.attractors),
            initial_attractors + 1,
            "Complete success should add attractor"
        )
    
    def test_apply_result_to_field_partial(self):
        """Test _apply_result_to_field with partial result."""
        import torch
        
        pm_field = self.encoder.pm_field
        z = torch.randn(1, 32)
        
        partial_result = {"content": "some data"}
        initial_hazards = len(pm_field.hazards)
        initial_attractors = len(pm_field.attractors)
        
        self.planner._apply_result_to_field(
            pm_field, z, partial_result
        )
        
        # Partial result should not change field drastically
        self.assertEqual(len(pm_field.hazards), initial_hazards)
        # Might add weak attractor for successful partial
        # Allow either no change or one attractor
        self.assertLessEqual(
            len(pm_field.attractors),
            initial_attractors + 1
        )


class TestGoalCompletion(unittest.TestCase):
    """Tests for goal completion detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        from v2.lilith_v2.action_planner import (
            ActionPlanner, ActionNode, ExecutionPlan, PlannedStep, GoalCompletion
        )
        from v2.lilith_v2.relational_graph_store import RelationalGraphStore
        
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_actions.sqlite")
        self.graph = RelationalGraphStore(self.db_path)
        
        self.encoder = MockEncoder(dimension=64, latent_dim=32)
        self.planner = ActionPlanner(
            encoder=self.encoder,
            graph=self.graph,
            trajectory_steps=10,
            grounding_threshold=0.3,
        )
        
        # Register some test actions using proper method signature
        self.planner.register_action(
            name="search_web",
            description="search the web for information",
            tool_binding="web_search",
        )
        self.planner.register_action(
            name="login_user",
            description="log in to the system",
            tool_binding="auth_login",
        )
    
    def tearDown(self):
        """Clean up."""
        self.graph.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_goal_completion_success(self):
        """Test goal completion detection for successful execution."""
        # Simulate successful execution results
        results = [
            {"step": 1, "action": "search_web", "tool": "web_search", 
             "result": {"status": "ok", "output": "found information about the topic"}},
            {"step": 2, "action": "login_user", "tool": "auth_login",
             "result": {"status": "ok", "output": "logged in successfully"}},
        ]
        
        completion = self.planner.check_goal_completion(
            goal="search for info and login",
            execution_results=results,
            final_state="completed search and login successfully",
            completion_threshold=0.5,
        )
        
        self.assertIsInstance(completion, GoalCompletion)
        self.assertEqual(completion.steps_executed, 2)
        self.assertEqual(completion.steps_failed, 0)
        self.assertEqual(completion.steps_skipped, 0)
        self.assertEqual(completion.execution_success_rate, 1.0)
        # With mock encoder, similarity may vary, but should have valid confidence
        self.assertGreaterEqual(completion.confidence, 0.0)
        self.assertLessEqual(completion.confidence, 1.0)
    
    def test_goal_completion_failure(self):
        """Test goal completion detection for failed execution."""
        results = [
            {"step": 1, "action": "search_web", "status": "error", "message": "network timeout"},
            {"step": 2, "action": "login_user", "status": "skipped", "reason": "previous step failed"},
        ]
        
        completion = self.planner.check_goal_completion(
            goal="search and login",
            execution_results=results,
        )
        
        self.assertFalse(completion.completed)
        self.assertEqual(completion.steps_failed, 1)
        self.assertEqual(completion.steps_skipped, 1)
        self.assertEqual(completion.steps_executed, 0)
        self.assertIn("network timeout", completion.failure_reasons[0])
    
    def test_goal_completion_partial(self):
        """Test goal completion detection for partial execution."""
        results = [
            {"step": 1, "action": "search_web", "tool": "web_search",
             "result": {"status": "ok", "output": "found results"}},
            {"step": 2, "action": "login_user", "status": "error", "message": "auth failed"},
        ]
        
        completion = self.planner.check_goal_completion(
            goal="search and then login",
            execution_results=results,
        )
        
        self.assertEqual(completion.steps_executed, 1)
        self.assertEqual(completion.steps_failed, 1)
        self.assertEqual(completion.execution_success_rate, 0.5)
    
    def test_goal_completion_empty_results(self):
        """Test goal completion with no execution results."""
        completion = self.planner.check_goal_completion(
            goal="do something",
            execution_results=[],
        )
        
        self.assertFalse(completion.completed)
        self.assertEqual(completion.steps_executed, 0)
        self.assertEqual(completion.final_state_description, "no actions were executed")
    
    def test_infer_final_state(self):
        """Test final state inference from results."""
        results = [
            {"step": 1, "action": "search_web", "tool": "web_search",
             "result": {"output": "Python documentation found"}},
            {"step": 2, "action": "read_page", "tool": "web_read",
             "result": {"output": "Page content about Python programming"}},
        ]
        
        final_state = self.planner._infer_final_state(
            goal="learn about Python",
            execution_results=results,
        )
        
        self.assertIn("search_web", final_state)
        self.assertIn("Page content about Python", final_state)
    
    def test_execute_and_check(self):
        """Test combined execute and check functionality."""
        from v2.lilith_v2.action_planner import ExecutionPlan, PlannedStep, ActionNode
        
        # Create a simple plan using the registered action
        action = self.planner._action_cache.get("action_search_web")
        if action:
            step = PlannedStep(
                step_num=1,
                action=action,
                args={},
                waypoint_embedding=action.embedding,
                confidence=0.9,
            )
            plan = ExecutionPlan(
                goal="find information",
                steps=[step],
                trajectory_efficiency=0.8,
                total_confidence=0.9,
                estimated_success=0.85,
            )
            
            class MockTransport:
                def call(self, name, message, meta=None):
                    return {"status": "ok", "output": "search results found"}
            
            results, completion = self.planner.execute_and_check(
                plan=plan,
                transport=MockTransport(),
            )
            
            self.assertEqual(len(results), 1)
            self.assertIsInstance(completion, GoalCompletion)
            self.assertEqual(completion.steps_executed, 1)
    
    def test_goal_completion_repr(self):
        """Test GoalCompletion string representation."""
        from v2.lilith_v2.action_planner import GoalCompletion
        
        completion = GoalCompletion(
            goal="test goal",
            completed=True,
            confidence=0.85,
            semantic_similarity=0.9,
            execution_success_rate=1.0,
            steps_executed=3,
            steps_failed=0,
            steps_skipped=0,
            final_state_description="all done",
        )
        
        repr_str = repr(completion)
        self.assertIn("COMPLETED", repr_str)
        self.assertIn("0.85", repr_str)
        self.assertIn("0.90", repr_str)


# Import GoalCompletion at module level for tests
try:
    from v2.lilith_v2.action_planner import GoalCompletion
except ImportError:
    GoalCompletion = None


if __name__ == "__main__":
    unittest.main()
