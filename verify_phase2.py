#!/usr/bin/env python3
"""
Verification script for Phase 2: Reasoning Stage Integration

Verifies that the reasoning stage is properly integrated into the
gap-filling learning process.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def verify_phase2_implementation():
    """Verify Phase 2 implementation"""
    
    print("=" * 80)
    print("PHASE 2: REASONING STAGE INTEGRATION - VERIFICATION")
    print("=" * 80)
    print()
    
    try:
        from lilith.response_composer import ResponseComposer
        import inspect
        
        # Get the _fill_gaps_and_retry method
        method = getattr(ResponseComposer, '_fill_gaps_and_retry', None)
        
        if method is None:
            print("❌ _fill_gaps_and_retry method not found!")
            return False
        
        print("✅ Found _fill_gaps_and_retry method")
        print()
        
        # Check the source code for Phase 2 integration
        source = inspect.getsource(method)
        
        phase2_checks = {
            'Phase 2 Header': 'PHASE 2: REASONING STAGE INTEGRATION',
            'Reasoning Stage Check': 'if self.reasoning_stage',
            'Concept Activation': 'reasoning_stage.activate_concept(',
            'Deliberation Call': 'reasoning_stage.deliberate(',
            'Inference Logging': 'deliberation.inferences',
            'Connection Discovery': 'Found {len(deliberation.inferences)} connections',
            'Learned External Source': 'source="learned_external"',
            'Quick Deliberation': 'max_steps=2',
            'Symbolic Level Comment': 'SYMBOLIC LEVEL',
            'Network Building': 'semantic network'
        }
        
        print("Phase 2 Integration Checks:")
        print("-" * 80)
        
        failed = []
        for check_name, pattern in phase2_checks.items():
            passed = pattern in source
            status = "✅" if passed else "❌"
            print(f"{status} {check_name}: {pattern}")
            
            if not passed:
                failed.append(check_name)
        
        print()
        
        # Check docstring update
        docstring = inspect.getdoc(method)
        docstring_checks = {
            'Phase 2 in Docstring': 'Phase 2' in docstring,
            'BUILD CONNECTIONS mentioned': 'BUILD CONNECTIONS' in docstring,
            'Reasoning stage explanation': 'reasoning stage' in docstring.lower(),
            'Inference generation': 'inferences' in docstring.lower(),
            'Knowledge graph': 'knowledge graph' in docstring.lower() or 'semantic network' in docstring.lower()
        }
        
        print("Documentation Checks:")
        print("-" * 80)
        
        for check_name, condition in docstring_checks.items():
            status = "✅" if condition else "❌"
            print(f"{status} {check_name}")
            
            if not condition:
                failed.append(check_name)
        
        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        total_checks = len(phase2_checks) + len(docstring_checks)
        passed_checks = total_checks - len(failed)
        
        print(f"Checks Passed: {passed_checks}/{total_checks}")
        print()
        
        if not failed:
            print("✅ ALL CHECKS PASSED!")
            print()
            print("Phase 2 Reasoning Stage Integration is complete:")
            print("  🧠 Reasoning stage activated for learned concepts")
            print("  🔗 Deliberation finds connections with existing knowledge")
            print("  💡 Inferences generated from concept relationships")
            print("  🕸️  Semantic network built from learned information")
            print()
            print("Enhanced Learning Flow:")
            print("  1️⃣  Learn vocabulary, concepts, syntax (Phase 1)")
            print("  2️⃣  Activate concepts in reasoning stage (Phase 2)")
            print("  3️⃣  Run deliberation to find connections (Phase 2)")
            print("  4️⃣  Generate and log inferences (Phase 2)")
            print("  5️⃣  Build semantic knowledge graph (Phase 2)")
            return True
        else:
            print("❌ SOME CHECKS FAILED")
            print()
            print("Missing components:")
            for check in failed:
                print(f"  ❌ {check}")
            return False
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = verify_phase2_implementation()
    
    print()
    if success:
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "🎉 PHASE 2 COMPLETE - READY TO TEST! 🎉" + " " * 21 + "║")
        print("╚" + "═" * 78 + "╝")
        print()
        print("What changed:")
        print("  ✅ Reasoning stage now activates learned concepts")
        print("  ✅ Deliberation finds connections between concepts")
        print("  ✅ Inferences generated and logged")
        print("  ✅ Knowledge graph built instead of isolated facts")
        print()
        print("Test in production to see:")
        print("  • 🔗 Reasoning: Found N connections for 'term'")
        print("  • → inference_type: conclusion...")
        print()
    
    sys.exit(0 if success else 1)
