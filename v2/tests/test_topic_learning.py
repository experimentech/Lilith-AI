import sys
import os
import unittest
import numpy as np
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from v2.lilith_v2.topic_extractor import TopicExtractor
from v2.lilith_v2.cognitive_stage import CognitiveStage
from v2.lilith_v2.pmflow_sqlite import PMFlowStateStore
from v2.lilith_v2.relational_graph_store import RelationalGraphStore

class MockEncoder:
    def encode(self, text):
        # Create a deterministic fake embedding based on string hash
        # This ensures "Rust" always gets same vector, "Java" different
        seed = abs(hash(text)) % 1000
        np.random.seed(seed)
        return np.random.rand(10).astype(np.float32)

class TestTopicLearning(unittest.TestCase):
    def setUp(self):
        self.encoder = MockEncoder()
        # Use simple file path for testing
        test_db = "data/test_topics.json"
        if os.path.exists(test_db):
            os.remove(test_db)
        self.extractor = TopicExtractor(self.encoder, storage_path=test_db)

    def test_basic_learning_extraction(self):
        # 1. Learn
        self.extractor.learn_topic("Rust", "Rust is a fast language.")
        
        # 2. Extract from exact match
        topic = self.extractor.extract_topic("Do you know about Rust?")
        self.assertEqual(topic, "Rust")
        
        # 3. Extract from noisy query
        # "tell me about" is in scaffolding, so it should be stripped
        topic = self.extractor.extract_topic("Please tell me about Rust !!!")
        self.assertEqual(topic, "Rust")

    def test_differentiation(self):
        self.extractor.learn_topic("Rust", "context")
        self.extractor.learn_topic("Python", "context")
        
        # Should distinguish based on vector similarity (random vectors are orthogonal-ish in high dim, 
        # but low dim 10 might have collisions. Let's hope hash func is distinct enough.)
        
        res_rust = self.extractor.extract_topic("Rust")
        res_python = self.extractor.extract_topic("Python")
        
        self.assertEqual(res_rust, "Rust")
        self.assertEqual(res_python, "Python")

    def tearDown(self):
        if os.path.exists("data/test_topics.json"):
            os.remove("data/test_topics.json")

if __name__ == '__main__':
    unittest.main()
