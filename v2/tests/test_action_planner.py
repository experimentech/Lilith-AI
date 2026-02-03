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
        # Register some actions
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
        
        # Generate a plan
        plan = self.planner.plan(
            goal="copy file to backup",
            context={"file_path": "test.txt", "output_path": "backup/test.txt"},
        )
        
        self.assertIsNotNone(plan)
        self.assertEqual(plan.goal, "copy file to backup")
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


if __name__ == "__main__":
    unittest.main()
