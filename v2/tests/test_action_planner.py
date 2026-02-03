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
    """Mock encoder with agentic physics features."""
    
    def __init__(self, dimension=64, latent_dim=32):
        self.dimension = dimension
        self.latent_dim = latent_dim
        self.enable_flow = True
        self._intent = None
        self._intent_strength = 0.0
    
    def encode(self, tokens):
        """Encode tokens to embedding."""
        # Create deterministic embedding based on token content
        if isinstance(tokens, list):
            text = " ".join(tokens)
        else:
            text = str(tokens)
        
        # Hash-based pseudo-embedding
        hash_val = hash(text) % 10000
        torch.manual_seed(hash_val)
        return torch.randn(1, self.dimension)
    
    def inject_intent(self, tokens, strength=0.5):
        """Mock intent injection."""
        self._intent = tokens
        self._intent_strength = strength
    
    def clear_intent(self):
        """Clear injected intent."""
        self._intent = None
        self._intent_strength = 0.0
    
    def trace_trajectory(self, tokens, steps=10):
        """Mock trajectory tracing."""
        # Generate a trajectory of embeddings
        start = self.encode(tokens)
        
        trajectory = []
        current = start.clone()
        
        for i in range(steps):
            # Drift toward intent if set
            if self._intent:
                intent_emb = self.encode(self._intent)
                alpha = (i + 1) / steps * self._intent_strength
                current = (1 - alpha) * current + alpha * intent_emb
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


if __name__ == "__main__":
    unittest.main()
