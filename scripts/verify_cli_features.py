#!/usr/bin/env python3
"""
Quick verification that CLI gets all Phase 2 features.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from lilith.session import SessionConfig


def main():
    """Check what the CLI is getting."""
    print("\n" + "="*70)
    print("CLI FEATURE VERIFICATION")
    print("="*70)
    
    # This is what the CLI creates (minimal config)
    cli_config = SessionConfig(
        data_path="data",
        learning_enabled=True,
        enable_declarative_learning=True,
        enable_feedback_detection=True,
        plasticity_enabled=True
    )
    
    print("\nCLI Session Configuration:")
    print("-" * 70)
    
    # Core features
    print("\n📚 Core Features:")
    print(f"  data_path:                    {cli_config.data_path}")
    print(f"  learning_enabled:             {cli_config.learning_enabled}")
    print(f"  use_grammar:                  {cli_config.use_grammar}")
    
    # Phase 2 features
    print("\n🎯 Phase 2 Features (NEW):")
    print(f"  enable_compositional:         {cli_config.enable_compositional}")
    print(f"  enable_pragmatic_templates:   {cli_config.enable_pragmatic_templates}")
    print(f"  composition_mode:             {cli_config.composition_mode}")
    
    # Knowledge augmentation
    print("\n🌐 Knowledge Augmentation:")
    print(f"  enable_knowledge_augmentation: {cli_config.enable_knowledge_augmentation}")
    
    # Modal routing
    print("\n🔢 Modal Routing:")
    print(f"  enable_modal_routing:         {cli_config.enable_modal_routing}")
    
    # Learning features
    print("\n📖 Learning Features:")
    print(f"  enable_declarative_learning:  {cli_config.enable_declarative_learning}")
    print(f"  enable_auto_learning:         {cli_config.enable_auto_learning}")
    print(f"  enable_feedback_detection:    {cli_config.enable_feedback_detection}")
    
    # Plasticity
    print("\n🧠 Neuroplasticity:")
    print(f"  plasticity_enabled:           {cli_config.plasticity_enabled}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    phase2_features = []
    if cli_config.enable_compositional:
        phase2_features.append("✅ Concept store (semantic knowledge)")
    if cli_config.enable_pragmatic_templates:
        phase2_features.append("✅ Pragmatic templates (26 conversational patterns)")
    if cli_config.composition_mode == "pragmatic":
        phase2_features.append("✅ Compositional response generation")
    if cli_config.enable_knowledge_augmentation:
        phase2_features.append("✅ Wikipedia integration (with disambiguation)")
    if cli_config.enable_modal_routing:
        phase2_features.append("✅ Math backend (symbolic computation)")
    
    print("\nPhase 2 features active in CLI:")
    for feature in phase2_features:
        print(f"  {feature}")
    
    print(f"\n🎉 CLI has {len(phase2_features)}/5 Phase 2 features enabled!")
    print("\nWhat this means for CLI users:")
    print("  • Novel compositional responses (not verbatim)")
    print("  • Learn from Wikipedia with '/+' to save")
    print("  • Conversation continuity ('Tell me more' works)")
    print("  • Math queries: '2+2', 'sqrt(16)', etc.")
    print("  • Efficient storage (26 templates + concepts)")
    

if __name__ == "__main__":
    main()
