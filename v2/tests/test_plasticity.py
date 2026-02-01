import unittest
import torch
import shutil
import tempfile
import os
from unittest.mock import MagicMock
from v2.lilith_v2.cognitive_stage import CognitiveStage
from v2.lilith_v2.pmflow_sqlite import SQLitePMFlowStateStore
from v2.lilith_v2.relational_graph_store import RelationalGraphStore

class MockEncoder:
    def __init__(self):
        self.embedding_dim = 4 # Small dim for testing
    def encode(self, text):
        # Deterministic deterministic encoding for testing
        # Ruby -> [0.1, 0.2, 0.3, 0.4]
        return torch.tensor([0.1, 0.2, 0.3, 0.4])

class MockGraph(RelationalGraphStore):
    def __init__(self):
        self.nodes = {}
    def add_node(self, node_id, node_type, term, confidence=1.0, data=None):
        self.nodes[node_id] = term
    def add_edge(self, *args, **kwargs): pass
    def traverse_bfs(self, *args, **kwargs): return []
    def get_node(self, node_id): return {"term": self.nodes.get(node_id)}
    def get_all_terms(self):
        # Used by ConceptGrounder hydrate
        # Return list of (id, term) tuples
        return [(nid, term) for nid, term in self.nodes.items()]
    def find_nodes(self, term):
        # Support grounding
        found = []
        for nid, val in self.nodes.items():
            if val.lower() in term.lower():
                # Mock a node object
                m = MagicMock()
                m.concept_id = nid
                m.confidence = 1.0
                found.append(m)
        return found
    def get_edges(self, *args): return []

class TestPlasticity(unittest.TestCase):
    def setUp(self):
        # Temp DB for PMFlow
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "pmflow_plasticity.db")
        self.pmflow = SQLitePMFlowStateStore(self.db_path)
        
        self.graph = MockGraph()
        self.encoder = MockEncoder()
        
        self.stage = CognitiveStage(
            node_id="lilith_core",
            pmflow_store=self.pmflow,
            graph_store=self.graph,
            encoder=self.encoder,
            config={"knowledge_enabled": False}
        )
        from pathlib import Path
        # Fix TopicExtractor dimension mismatch by clearing loaded topics
        # and pointing to temp file (Must be Path object)
        self.stage.topic_extractor.topics = {}
        self.stage.topic_extractor.storage_path = Path(self.test_dir) / "topics.json"
        
        # Mock grounder to ensure regex/exact match works against our MockGraph
        # The default implementation calls self.graph.find_nodes which we mocked

    def tearDown(self):
        self.pmflow.close()
        shutil.rmtree(self.test_dir)

    def test_hebbian_reinforcement(self):
        """Test that learning a concept twice increases its attractor mass."""
        
        # 1. Initial State: No attractors
        state = self.pmflow.load_state("main")
        self.assertEqual(len(state.get("attractors", [])), 0)
        
        # 2. Learn a new fact: "Ruby is a red gem"
        # Use stronger phrasing to ensure high confidence (>0.8) extraction
        self.stage.learn("Ruby is a type of red gem")
        
        # Check State Genesis
        state_1 = self.pmflow.load_state("main")
        attractors_1 = state_1.get("attractors", [])
        
        # We expect attractors for 'ruby' and maybe 'red_gem'/'gem' depending on extraction
        # Extractor: "Ruby is a ..." -> Subject=Ruby
        ruby_attr = next((a for a in attractors_1 if a.get("concept_id") == "ruby"), None)
        
        self.assertIsNotNone(ruby_attr, "Should have created an attractor for 'Ruby'")
        initial_mass = ruby_attr["weight"]
        self.assertGreater(initial_mass, 0.0)
        
        # 3. Learn again (Reinforcement)
        input_text = "Ruby is indeed a gem."
        # Ensure graph has it so Grounding finds it
        self.graph.add_node("ruby", "concept", "Ruby") 
        
        self.stage.learn(input_text)
        
        # Check State Evolution
        state_2 = self.pmflow.load_state("main")
        attractors_2 = state_2.get("attractors", [])
        ruby_attr_2 = next((a for a in attractors_2 if a.get("concept_id") == "ruby"), None)
        
        reinforce_mass = ruby_attr_2["weight"]
        
        print(f"Mass Evolution: {initial_mass} -> {reinforce_mass}")
        self.assertGreater(reinforce_mass, initial_mass, "Mass should increase after reinforcement")
        
        # Verify vector is valid length
        self.assertEqual(len(ruby_attr_2["vector"]), 4)

if __name__ == "__main__":
    unittest.main()
