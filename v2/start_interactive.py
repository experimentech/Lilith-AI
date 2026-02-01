#!/usr/bin/env python3
import sys
import os
import argparse
import logging
from typing import Dict, Any

# Ensure we can import lilith_v2
sys.path.insert(0, os.getcwd())

from v2.lilith_v2.app import V2Runtime
from v2.lilith_v2.cognitive_stage import CognitiveStage
from v2.lilith_v2.multi_tenant_store import MultiTenantPMFlowManager, MultiTenantGraphManager
from v2.lilith_v2.mcp_router import MCPDescriptor
from v2.lilith_v2.relational_store import RelationalStore

# Try to import PMFlow encoder for agentic physics
try:
    from pmflow import PMFlowEmbeddingEncoder
    HAS_PMFLOW = True
except ImportError:
    HAS_PMFLOW = False

# Simple fallback encoder if PMFlow is missing
class SimpleEncoder:
    """Fallback encoder without agentic physics."""
    def __init__(self):
        self.embedding_dim = 64
        self.enable_flow = False  # No agentic physics
        
    def encode(self, text):
        import torch
        # Deterministic simple embedding for demo
        seed = sum(ord(c) for c in str(text)) 
        torch.manual_seed(seed)
        return torch.randn(64)


def create_encoder(enable_flow: bool = True, verbose: bool = False):
    """Create the best available encoder."""
    if HAS_PMFLOW:
        encoder = PMFlowEmbeddingEncoder(
            dimension=96,
            latent_dim=48,
            enable_flow=enable_flow,
        )
        if verbose:
            flow_status = "enabled" if enable_flow else "disabled"
            print(f"  PMFlowEmbeddingEncoder loaded (agentic flow: {flow_status})")
        return encoder
    else:
        if verbose:
            print("  Fallback to SimpleEncoder (PMFlow not available)")
        return SimpleEncoder()

def main():
    parser = argparse.ArgumentParser(description="Lilith V2 Interactive Shell")
    parser.add_argument("--user", default="user", help="User/Tenant ID")
    parser.add_argument("--teacher", action="store_true", help="Run as Teacher")
    parser.add_argument("--verbose", action="store_true", help="Verbose logs")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    tenant_id = "teacher" if args.teacher else args.user
    print(f"\033[1;36mBooting Lilith V2 as tenant: '{tenant_id}'\033[0m")

    # 1. Setup Data Stores
    data_root = os.path.join(os.getcwd(), "data")
    base_root = os.path.join(data_root, "base")
    users_root = os.path.join(data_root, "users")
    
    print(f"Data Root: {data_root}")
    
    pmflow = MultiTenantPMFlowManager(base_root, users_root)
    graph = MultiTenantGraphManager(base_root, users_root)
    
    # 2. Setup Runtime (Limbs)
    # Using default() gives us FS, Terminal, Weather integration
    runtime = V2Runtime.default()
    
    # 3. Inject Brain (CognitiveStage)
    # Replace NoopStage on 'trunk.vscode'
    
    # Create encoder with agentic physics for reasoning
    encoder = create_encoder(enable_flow=True, verbose=args.verbose)
    
    # Response Store for pattern-based response generation with learning
    response_db_path = os.path.join(users_root, tenant_id, "responses.db")
    os.makedirs(os.path.dirname(response_db_path), exist_ok=True)
    response_store = RelationalStore(response_db_path)

    cognitive = CognitiveStage(
        node_id="trunk.vscode",
        pmflow_store=pmflow,
        graph_store=graph,
        encoder=encoder,
        config={
            "knowledge_enabled": True,
            # Response Composer settings
            "response_store": response_store,
            "composition_mode": "adaptive",  # best_match, weighted_blend, adaptive, graph_first
            "enable_blending": True,          # Allow novel response construction
            "enable_learning": True,          # Learn from feedback
            # Reasoning Stage settings
            "enable_reasoning": True,         # Enable deliberative reasoning
            "deliberation_steps": 10,         # Steps per reasoning cycle
            "max_working_memory": 7,          # Miller's number
        }
    )
    
    if args.verbose:
        print(f"  ResponseComposer enabled with store: {response_db_path}")
        print(f"  ReasoningStage enabled (deliberation_steps=10)")
    
    runtime.stages["trunk.vscode"] = cognitive
    
    print("\033[1;32mSystem Online.\033[0m")
    print("Commands: 'exit' to quit, 'health' for status, 'stats' for cognitive stats.")
    print("Feedback: 'feedback+' = good response, 'feedback-' = bad response.")
    print("Debug: 'reasoning' to show last deliberation result.")
    
    # 4. Interaction Loop
    ctx = {"tenant": tenant_id, "modality": "text"}
    port = runtime.bindings["trunk.vscode"].ports[0] # The text port
    
    # Clear any startup noise
    list(port.receive({}))

    while True:
        try:
            # Colorized Prompt
            prompt = f"\033[1;33m{tenant_id}>\033[0m "
            user_input = input(prompt)
            
            if not user_input.strip():
                continue
                
            if user_input.strip().lower() in ["exit", "quit"]:
                break
                
            if user_input.strip().lower() == "health":
                print(runtime.health_check(tenant=tenant_id))
                continue
            
            if user_input.strip().lower() == "stats":
                stats = cognitive.stats()
                print(f"\033[1;34mCognitive Stats:\033[0m")
                for k, v in stats.items():
                    if isinstance(v, dict):
                        print(f"  {k}:")
                        for k2, v2 in v.items():
                            print(f"    {k2}: {v2}")
                    else:
                        print(f"  {k}: {v}")
                continue
            
            if user_input.strip().lower() == "feedback+":
                cognitive.record_response_outcome(success=True)
                print("\033[1;32m✓ Positive feedback recorded\033[0m")
                continue
            
            if user_input.strip().lower() == "feedback-":
                cognitive.record_response_outcome(success=False)
                print("\033[1;31m✗ Negative feedback recorded\033[0m")
                continue
            
            if user_input.strip().lower() == "reasoning":
                # Show last deliberation result
                if hasattr(cognitive, '_reasoning') and cognitive._reasoning:
                    result = cognitive._reasoning.get_last_result()
                    if result:
                        print(f"\033[1;34mLast Deliberation:\033[0m")
                        print(f"  Steps: {result.deliberation_steps}")
                        print(f"  Confidence: {result.confidence:.2f}")
                        print(f"  Efficiency: {result.trajectory_efficiency:.2f}")
                        print(f"  Mental Effort: {result.mental_effort:.3f}")
                        print(f"  Focus: {result.focus_concept}")
                        print(f"  Intent: {result.resolved_intent}")
                        print(f"  Inferences: {len(result.inferences)}")
                        for inf in result.inferences[:5]:  # Show first 5
                            print(f"    - [{inf.inference_type}] {inf.conclusion[:60]}...")
                    else:
                        print("No deliberation result yet.")
                else:
                    print("Reasoning stage not enabled.")
                continue

            # Dispatch
            descriptor = MCPDescriptor(name="user_input", type="action")
            
            # This triggers: Port Send (Input) -> Stage Learn -> Stage Think -> Port Send (Output)
            runtime.route_and_dispatch(
                descriptor, 
                context=ctx, 
                payload=user_input
            )
            
            # Collect outputs
            # We expect at least the input echo and the output
            messages = list(port.receive({}))
            
            for msg_pack in messages:
                msg = msg_pack.get("message")
                meta = msg_pack.get("meta", {})
                
                # Filter out the input echo (heuristic: source is usually empty vs 'trunk.vscode')
                # Or just check equality
                if msg == user_input:
                    continue
                    
                print(f"\033[1;35mLilith:\033[0m {msg}")
                
                # Debug info if verbose
                if args.verbose:
                    print(f"[DEBUG] Meta: {meta}")

        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"\033[1;31mError: {e}\033[0m")
            import traceback
            traceback.print_exc()

    # Cleanup (SQLite handles cleanup automatically)
    pmflow.close()
    graph.close()

if __name__ == "__main__":
    main()
