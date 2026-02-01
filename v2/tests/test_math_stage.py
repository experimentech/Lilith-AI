import unittest
from v2.lilith_v2.math_stage import MathStage

class TestMathStage(unittest.TestCase):
    def setUp(self):
        self.stage = MathStage("branch.math")
        
    def test_arithmetic(self):
        self.stage.learn("What is 2 + 2?")
        res = self.stage.last_interaction
        self.assertIsNotNone(res)
        # 4.0000.. or 4
        # SymPy usually exact integer 4
        self.assertIn("4", res['response'])
        
    def test_equation(self):
        # x + 2 = 10 -> x = 8
        self.stage.learn("Solve x + 2 = 10")
        res = self.stage.last_interaction
        self.assertIn("8", res['response'])
        
    def test_invalid(self):
        self.stage.learn("Hello world")
        res = self.stage.last_interaction
        self.assertEqual(res['response'], "I don't see any math here.")

if __name__ == "__main__":
    unittest.main()
