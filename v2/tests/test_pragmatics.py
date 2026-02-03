import sys
import os
import unittest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from v2.lilith_v2.pragmatic_system import PragmaticSystem, PragmaticTemplate
from v2.lilith_v2.generative_system import GenerativeSystem

class MockGraph:
    def __init__(self):
        self.nodes = {}
    def add_node(self, **kwargs): pass

class TestPragmatics(unittest.TestCase):
    def setUp(self):
        self.pragmatics = PragmaticSystem()
        self.gen = GenerativeSystem(MockGraph())

    def test_template_retrieval(self):
        # We expect to find a definition template
        tpl = self.pragmatics.get_template("definition", ["concept", "property"])
        self.assertIsNotNone(tpl)
        self.assertEqual(tpl.category, "definition")
        self.assertIn("concept", tpl.slots)

    def test_template_fill(self):
        tpl = PragmaticTemplate("test", "test", "Hello {name}!", ["name"])
        res = self.pragmatics.fill(tpl, {"name": "Lilith"})
        self.assertEqual(res, "Hello Lilith!")

    def test_generative_system_integration(self):
        # Simulate teaching Lilith
        # User: "Python is a language." -> Subject=Python, Object=Language
        class Rel:
            subject = "Python"
            object = "Language"
            predicate = "is_a"

        context = {
            "extract_knowledge": [Rel()], # Typo intended to test empty first
            "inference": [],
            "affect": {}
        }
        
        # 1. Fallback (No context = listening state)
        res = self.gen.compose(context)
        # Should return a fallback message when no knowledge is available
        self.assertIsInstance(res, str)
        self.assertGreater(len(res), 0)
        
        # 2. Teaching Acknowledgment
        context["extracted_knowledge"] = [Rel()]
        res = self.gen.compose(context)
        # Should match teaching_ack template: "I see! So {subject} is {object}..."
        self.assertIn("Python", res)
        self.assertIn("Language", res)

if __name__ == '__main__':
    unittest.main()
