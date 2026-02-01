import unittest
from v2.lilith_v2.app import V2Runtime
from v2.lilith_v2.mcp_router import MCPDescriptor, EndpointType
from v2.lilith_v2.in_memory_port import InMemoryIOPort

class TestMathIntegration(unittest.TestCase):
    def test_full_math_pipeline(self):
        # 1. Config with Math Branch
        config = {
            "bindings": [
                {"node_id": "trunk.tools", "modalities": ["text"]},
                {"node_id": "branch.math", "modalities": ["math"], "stage": "math"}
            ]
        }
        
        runtime = V2Runtime.from_config(config)
        
        # 2. Mock Binding Resolution (since InMemoryBindingResolver created inside from_config is fresh)
        # Actually `from_config` creates everything correctly.
        # We need to grab the ports to check output.
        decision = runtime.route_and_dispatch(
            descriptor=MCPDescriptor("test_req", EndpointType.ACTION),
            context={"modality": "math", "tenant": "test_user"},
            payload="What is 2 + 2?"
        )
        
        # 3. Verify Routing
        self.assertIn("branch.math", decision.target_nodes)
        self.assertTrue("branch.math" in decision.ports)
        
        math_port = decision.ports["branch.math"]
        # Since V2Runtime wraps ports in make_vscode_binding which uses VSCodeMCPAdapter (InMemoryIOPort),
        # we can check its queue.
        # BUT `route_and_dispatch` sends input into `port.send()`.
        # AND with my new change, it sends output into `port.send()` too?
        # Typically `port.send()` is OUTBOUND from the runtime perspective?
        # Or `port.send()` is INBOUND (from external)?
        # VSCodeMCPAdapter.send(message) queues it.
        # `route_and_dispatch` calls `port.send(body)`. This simulates INPUT arriving at the port?
        # Wait. `IOPort` is usually the boundary.
        # If I call `port.send()`, I am sending data TO the port.
        # If the port represents a connection to VS Code, then `send` means "Send to VS Code".
        # BUT `route_and_dispatch` does: `port.send(body)`...
        
        # In `route_and_dispatch` (existing code I read):
        #   port.send(body, meta=meta)
        #   stage.learn(body)
        # This implies `port.send` was being used to "Log" or "Echo" the input into the port's internal queue?
        # A `VSCodeMCPAdapter` (subclass of InMemoryIOPort) appends to `_queue`.
        # `InMemoryIOPort.receive` pops from `_queue`.
        
        # So `route_and_dispatch` pushes to queue. `SomaticLayer.tick` (if running) would pop from queue.
        # But `stage.learn` is called DIRECTLY in `route_and_dispatch`.
        # This means `route_and_dispatch` bypasses the `receive()` mechanism and injects straight to stage.
        
        # Now, my added code:
        #   response = stage.last_interaction
        #   port.send(response) 
        
        # This puts the RESPONSE into the `_queue` of the port.
        # If `VSCodeMCPAdapter` represents the connection, then `_queue` is the "Network Wire".
        # So finding the response in `_queue` means it was sent out.
        
        # Let's inspect the port queue
        # It should contain:
        # 1. The Input echo ("What is 2 + 2?")
        # 2. The Output response ("4")
        
        queue = list(math_port._queue) # Access underlying deque
        self.assertEqual(len(queue), 2)
        
        input_msg = queue[0]
        output_msg = queue[1]
        
        self.assertEqual(input_msg['message'], "What is 2 + 2?")
        
        # Check Response
        # The output payload is unwrapped from `{"response": "4"}` -> `"4"`
        # SymPy float might be 4.0000...
        val = float(output_msg['message'])
        self.assertAlmostEqual(val, 4.0)
        
        self.assertEqual(output_msg['meta']['source'], 'branch.math')

if __name__ == "__main__":
    unittest.main()
