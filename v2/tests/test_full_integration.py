import unittest
import torch
import shutil
import tempfile
import os
import time
from pathlib import Path
from v2.lilith_v2.cognitive_stage import CognitiveStage
from v2.lilith_v2.pmflow_sqlite import SQLitePMFlowStateStore
from v2.lilith_v2.relational_graph_store import RelationalGraphStore
from v2.lilith_v2.topic_extractor import TopicExtractor

# --------------------------------------------------------------------------
# MOCKS
# --------------------------------------------------------------------------

class MockEncoder:
    """Deterministic embedding generator for testing concept consistency."""
    def __init__(self):
        self.embedding_dim = 4 
    
    def encode(self, text):
        # Very simple hashing to vectors for stability
        # "Ruby" -> [0.1, 0, 0, 0] roughly
        val = sum(ord(c) for c in text) % 100 / 100.0
        return torch.tensor([val, 0.5, 0.5, 0.5])

class InMemoryGraph(RelationalGraphStore):
    """Simple Graph Store for the Integration Test."""
    def __init__(self):
        self.nodes = {}
        self.edges = []
    
    def add_node(self, node_id, node_type, term, confidence=1.0, data=None):
        self.nodes[node_id] = {"term": term, "type": node_type}
        
    def add_edge(self, source, target, type, confidence=1.0, metadata=None):
        self.edges.append((source, target, type))
        
    def traverse_bfs(self, start_node_id, max_depth=1):
        # Simple immediate neighbor return
        results = []
        for src, tgt, rel in self.edges:
            if src == start_node_id:
                tgt_term = self.nodes.get(tgt, {}).get("term", tgt)
                results.append({"subject": start_node_id, "predicate": rel, "object": tgt_term})
        return results

    def get_all_terms(self):
        return [(nid, d["term"]) for nid, d in self.nodes.items()]

    def find_nodes(self, term):
        # Used by Grounder if FuzzyUtils is disabled, but Grounder has manual cache logic
        # This is strictly not part of RelationalGraphStore Protocol usually, but our tests rely on impl details sometimes
        pass
        
# --------------------------------------------------------------------------
# INTEGRATION TEST
# --------------------------------------------------------------------------

class TestCoreIntegration(unittest.TestCase):
    """
    Tests the Full Cognitive Loop:
    Input -> Pragmatics (Teaching) -> Extraction -> Graph -> Plasticity -> Output.
    """
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "pmflow_core.db")
        self.topics_path = os.path.join(self.test_dir, "topics.json")
        
        self.pmflow = SQLitePMFlowStateStore(self.db_path)
        self.graph = InMemoryGraph()
        self.encoder = MockEncoder()
        
        self.stage = CognitiveStage(
            node_id="lilith_core",
            pmflow_store=self.pmflow,
            graph_store=self.graph,
            encoder=self.encoder,
            config={"knowledge_enabled": False}
        )
        # Fix TopicExtractor path
        self.stage.topic_extractor.storage_path = Path(self.topics_path)
        self.stage.topic_extractor.topics = {} # clear cache

    def tearDown(self):
        self.pmflow.close()
        shutil.rmtree(self.test_dir)

    def test_full_learning_conversation(self):
        print("\n--- Starting Full Core Integration Test ---")
        
        # 1. First interaction: Uncertainty
        # Lilith starts blank.
        print("\n[Step 1] User asks unknown question.")
        self.stage.learn("What is a Ruby?")
        
        response_1 = self.stage.last_thought["response"]
        print(f"Lilith: {response_1}")
        
        # She doesn't know, so she should give a generic or fail response.
        # Likely "I am listening" or similar since graph is empty.
        
        # 2. Teaching Interaction
        # User provides new information.
        print("\n[Step 2] User teaches: 'A Ruby is a red gemstone.'")
        # Note: Previous output was low confidence/fallback, so teaching heuristic should trigger.
        self.stage.learn("A Ruby is a red gemstone")
        
        response_2 = self.stage.last_thought["response"]
        print(f"Lilith: {response_2}")
        
        # A) Check Pragmatics: Did she acknowledge the teaching?
        is_ack = "I see" in response_2 or "learned" in response_2 or "updating my understanding" in response_2
        self.assertTrue(is_ack, "Lilith failed to pragmatically acknowledge the teaching lesson.")
        
        # B) Check Graph: Is the knowledge stored?
        # Expect nodes: ruby, red_gemstone
        self.assertIn("ruby", self.graph.nodes)
        self.assertIn("red_gemstone", self.graph.nodes)
        
        # C) Check Plasticity: Was an attractor created?
        state_2 = self.pmflow.load_state("main")
        attractors_2 = state_2.get("attractors", [])
        ruby_attr = next((a for a in attractors_2 if a.get("concept_id") == "ruby"), None)
        self.assertIsNotNone(ruby_attr, "Plasticity failed: No intuitive attractor created for 'Ruby'")
        
        # 3. Recall / Topic Following
        # User asks again, differently.
        print("\n[Step 3] User asks: 'Tell me about Ruby.'")
        self.stage.learn("Tell me about Ruby")
        
        response_3 = self.stage.last_thought["response"]
        print(f"Lilith: {response_3}")
        
        # Should contain the definition from graph
        is_recall = "red gemstone" in response_3.lower() or "red_gemstone" in response_3.lower()
        self.assertTrue(is_recall, f"Lilith failed to recall the learned fact. Got: {response_3}")
        
        # D) Check Topic Extraction
        topic = self.stage.topic_extractor.extract_topic("Tell me about Ruby")
        # Note: extract_topic relies on BioNN, which uses our MockEncoder.
        # "Ruby" was learned in Step 2.
        self.assertEqual(topic, "ruby", "Topic Extractor failed to identify 'Ruby'")
        
        # 4. Reinforcement
        # Mentioning it again should strengthen the attractor
        mass_before = ruby_attr["weight"]
        self.stage.learn("Ruby is definitely a type of red gemstone")
        
        state_4 = self.pmflow.load_state("main")
        attractors_4 = state_4.get("attractors", [])
        ruby_attr_4 = next((a for a in attractors_4 if a.get("concept_id") == "ruby"), None)
        
        print(f"Attractor Mass: {mass_before} -> {ruby_attr_4['weight']}")
        self.assertGreater(ruby_attr_4["weight"], mass_before, "Hebbian learning failed to reinforce mass.")

if __name__ == "__main__":
    unittest.main()
