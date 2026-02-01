from typing import Any, Dict, Optional, List
from .stage import Stage
from .math_system import MathSystem

class MathStage(Stage):
    """
    A specialized stage for the 'Math Branch'.
    It does not use Hebbian learning. It simply computes and responds.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.system = MathSystem()
        self.last_interaction: Optional[Dict[str, Any]] = None
        
    def learn(self, input_signal: Any, ctx: Optional[Dict[str, Any]] = None) -> None:
        """
        Process input as a math query.
        """
        if not isinstance(input_signal, str):
            # Try to extract text from structured signal
            if isinstance(input_signal, dict) and "message" in input_signal:
                text = str(input_signal["message"])
            else:
                text = str(input_signal)
        else:
            text = input_signal
            
        # Check confidence
        conf = self.system.check_confidence(text)
        if conf < 0.5:
            # Low confidence - maybe ignore or error?
            # For now, simplistic response
            self.last_interaction = {"response": "I don't see any math here."}
            return

        result = self.system.compute(text)
        if result:
            self.last_interaction = {
                "response": f"{result.result}",
                "meta": {
                    "steps": result.steps, 
                    "latex": result.latex,
                    "expression": result.expression
                }
            }
        else:
            self.last_interaction = {"response": "Could not compute."}
    
    def sleep(self) -> None:
        pass
