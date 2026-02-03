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

# Try to import SemanticPMFlowEncoder (preferred - trainable word embeddings)
try:
    from lilith.learned_vocabulary_encoder import SemanticPMFlowEncoder
    HAS_SEMANTIC_ENCODER = True
except ImportError:
    HAS_SEMANTIC_ENCODER = False

# Try to import PMFlow encoder for agentic physics (fallback)
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


def create_encoder(enable_flow: bool = True, verbose: bool = False, vocab_path=None):
    """Create the best available encoder.
    
    Priority:
    1. SemanticPMFlowEncoder - trainable word embeddings + PMFlow physics
    2. PMFlowEmbeddingEncoder - hashed embeddings + PMFlow physics  
    3. SimpleEncoder - basic fallback
    """
    if HAS_SEMANTIC_ENCODER:
        encoder = SemanticPMFlowEncoder(
            dimension=96,
            latent_dim=48,
            enable_flow=enable_flow,
            vocab_path=vocab_path,
            bootstrap_semantics=True,  # Learn core semantic relationships
        )
        if verbose:
            flow_status = "enabled" if enable_flow else "disabled"
            print(f"  SemanticPMFlowEncoder loaded (semantic learning + agentic flow: {flow_status})")
            print(f"    Vocabulary: {encoder.vocab_size()} words")
        return encoder
    elif HAS_PMFLOW:
        encoder = PMFlowEmbeddingEncoder(
            dimension=96,
            latent_dim=48,
            enable_flow=enable_flow,
        )
        if verbose:
            flow_status = "enabled" if enable_flow else "disabled"
            print(f"  PMFlowEmbeddingEncoder loaded (agentic flow: {flow_status})")
            print(f"    ⚠️  Using hash-based encoding (no semantic learning)")
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
    parser.add_argument("--production", action="store_true", 
                        help="Use production databases (data/production/)")
    parser.add_argument("--ingest", type=str, metavar="FILE",
                        help="Ingest a corpus file before starting")
    parser.add_argument("--ingest-field", type=str, default="text",
                        help="Text field name for JSON/CSV ingestion")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    tenant_id = "teacher" if args.teacher else args.user
    print(f"\033[1;36mBooting Lilith V2 as tenant: '{tenant_id}'\033[0m")

    # 1. Setup Data Stores
    data_root = os.path.join(os.getcwd(), "data")
    base_root = os.path.join(data_root, "base")
    users_root = os.path.join(data_root, "users")
    production_root = os.path.join(data_root, "production") if args.production else None
    
    # Check for production databases
    if production_root and os.path.exists(production_root):
        print(f"\033[1;33mUsing production databases: {production_root}\033[0m")
    elif args.production:
        print(f"\033[1;33mProduction enabled but {production_root} doesn't exist\033[0m")
        production_root = None
    
    print(f"Data Root: {data_root}")
    
    # Handle corpus ingestion before starting
    if args.ingest:
        print(f"\033[1;35mIngesting corpus: {args.ingest}\033[0m")
        from v2.lilith_v2.corpus_ingester import ingest_corpus
        try:
            stats = ingest_corpus(
                args.ingest,
                data_dir=production_root or os.path.join(data_root, "production"),
                text_field=args.ingest_field,
            )
            print(f"\033[1;32m{stats.summary()}\033[0m")
            # If we just ingested, enable production mode
            production_root = os.path.join(data_root, "production")
        except Exception as e:
            print(f"\033[1;31mIngestion failed: {e}\033[0m")
            import traceback
            traceback.print_exc()
    
    pmflow = MultiTenantPMFlowManager(base_root, users_root, production_root)
    graph = MultiTenantGraphManager(base_root, users_root, production_root)
    
    # 2. Setup Runtime (Limbs)
    # Using default() gives us FS, Terminal, Weather integration
    runtime = V2Runtime.default()
    
    # 3. Inject Brain (CognitiveStage)
    # Replace NoopStage on 'trunk.vscode'
    
    # Create encoder with semantic learning + agentic physics
    user_data_path = os.path.join(users_root, tenant_id)
    os.makedirs(user_data_path, exist_ok=True)
    vocab_path = os.path.join(user_data_path, "semantic_vocab")
    encoder = create_encoder(enable_flow=True, verbose=args.verbose, vocab_path=vocab_path)
    
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
    
    print("\033[1;32mSystem Online.\033[0m Type /help for commands.")
    
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
                
            if user_input.strip().lower() in ["/exit", "/quit"]:
                break
            
            if user_input.strip().lower() == "/help":
                print("\033[1;34mCommands:\033[0m")
                print("  /exit, /quit     - Exit the CLI")
                print("  /health          - System health check")
                print("  /stats           - Cognitive stats")
                print("  /feedback+       - Record positive feedback")
                print("  /feedback-       - Record negative feedback")
                print("  /reasoning       - Show last deliberation")
                print("  /explain [style] - Explain reasoning (narrative/step_by_step/technical)")
                print("  /consolidate     - Run memory consolidation")
                continue
                
            if user_input.strip().lower() == "/health":
                print(runtime.health_check(tenant=tenant_id))
                continue
            
            if user_input.strip().lower() == "/stats":
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
            
            if user_input.strip().lower() == "/feedback+":
                cognitive.record_response_outcome(success=True)
                print("\033[1;32m✓ Positive feedback recorded\033[0m")
                continue
            
            if user_input.strip().lower() == "/feedback-":
                cognitive.record_response_outcome(success=False)
                print("\033[1;31m✗ Negative feedback recorded\033[0m")
                continue
            
            if user_input.strip().lower() == "/reasoning":
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
                        
                        # Show concept chains if any were discovered
                        if result.concept_chains:
                            print(f"  \033[1;33mConcept Chains: {len(result.concept_chains)}\033[0m")
                            for chain in result.concept_chains[:3]:  # Show top 3
                                chain_str = " -> ".join(f"{t}[{r}]" for t, r in zip(chain.terms[:-1], chain.relations))
                                chain_str += f" -> {chain.terms[-1]}" if chain.terms else ""
                                print(f"    [{chain.confidence:.2f}] {chain_str}")
                        
                        # Show abstractions if any were formed
                        if result.abstractions:
                            print(f"  \033[1;36mAbstractions: {len(result.abstractions)}\033[0m")
                            for abs in result.abstractions[:3]:  # Show top 3
                                pattern_str = " -> ".join(abs.pattern)
                                print(f"    [{abs.confidence:.2f}] {abs.name}: {pattern_str}")
                                print(f"      Seen {abs.occurrence_count}x, exemplars: {', '.join(abs.exemplar_terms[:3])}")
                        
                        print(f"  Inferences: {len(result.inferences)}")
                        for inf in result.inferences[:5]:  # Show first 5
                            marker = "⛓" if inf.inference_type == "chain" else "•"
                            print(f"    {marker} [{inf.inference_type}] {inf.conclusion[:60]}...")
                    else:
                        print("No deliberation result yet.")
                else:
                    print("Reasoning stage not enabled.")
                continue
            
            if user_input.strip().lower() == "/consolidate":
                # Run memory consolidation
                if hasattr(cognitive, '_reasoning') and cognitive._reasoning:
                    stats = cognitive._reasoning.consolidate_memory(tenant_id=tenant_id)
                    print(f"\033[1;35mMemory Consolidation:\033[0m")
                    print(f"  Strengthened connections: {stats.get('strengthened_connections', 0)}")
                    print(f"  New attractors: {stats.get('new_attractors', 0)}")
                    print(f"  Merged abstractions: {stats.get('merged_abstractions', 0)}")
                    print(f"  Decayed: {stats.get('decayed_connections', 0)} working memory items")
                else:
                    print("Reasoning stage not enabled.")
                continue
            
            if user_input.strip().lower().startswith("/explain"):
                # Generate explanation of last reasoning
                if hasattr(cognitive, '_reasoning') and cognitive._reasoning:
                    parts = user_input.strip().split()
                    style = parts[1] if len(parts) > 1 else "narrative"
                    if style not in ("narrative", "step_by_step", "technical"):
                        style = "narrative"
                    
                    explanation = cognitive._reasoning.explain(style=style)
                    style_label = style.replace("_", "-")
                    print(f"\033[1;36mExplanation ({style_label}):\033[0m")
                    print(explanation)
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
