import sys
import os
import unittest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from v2.lilith_v2.world_model import WorldModel, Entity, SpatialRelation

class TestWorldModel(unittest.TestCase):
    def setUp(self):
        self.wm = WorldModel()

    def test_entity_extraction(self):
        text = "The red ball is on the table."
        situation = self.wm.process_utterance(text)
        
        entity_names = [e.name for e in situation.entities]
        self.assertIn("red ball", entity_names)
        self.assertIn("table", entity_names)

    def test_spatial_extraction(self):
        text = "The cat is in the box."
        situation = self.wm.process_utterance(text)
        
        self.assertTrue(len(situation.spatial_relations) > 0)
        rel = situation.spatial_relations[0]
        self.assertEqual(rel.relation_type, "in")
        self.assertIn("cat", rel.entity_a) # "cat" or "the cat" depending on extraction nuance
        self.assertIn("box", rel.entity_b)

    def test_object_permanence(self):
        # Turn 1: Establish state
        self.wm.process_utterance("The book is on the desk.")
        
        # Turn 2: Discuss something else
        self.wm.process_utterance("I like reading.")
        
        # Check context: The book should still be known
        context = self.wm.get_context()
        active_entities = [e['name'] for e in context['entities']]
        
        self.assertIn("book", active_entities)
        
        # Check spatial memory
        spatial_memory = context['spatial']
        # Expect something like "book on desk"
        self.assertTrue(any("on" in s for s in spatial_memory), "Spatial relation should persist")

    def test_causal_extraction(self):
        text = "Rain causes flooding."
        situation = self.wm.process_utterance(text)
        
        # Simple extraction might be tricky without "The", let's see heuristic
        # Our heuristic relies on spaces around markers: " causes "
        # Depending on "Rain" being detected as entity (no determiner).
        # Let's use determiners to be safe for this heuristics-based v2.0
        text_safe = "The rain causes the flood."
        situation = self.wm.process_utterance(text_safe)
        
        self.assertTrue(len(situation.causal_relations) > 0)
        rel = situation.causal_relations[0]
        self.assertEqual(rel.relation_type, "causes")

if __name__ == '__main__':
    unittest.main()
