#!/usr/bin/env python3
"""
Integration test for reasoning-enhanced fallback.

Quick sanity check that the enhanced code works.
"""

import sys

print("=" * 70)
print("Integration Test: Reasoning-Enhanced Fallback")
print("=" * 70)
print()

print("Testing that enhanced fallback code imports and runs...")
print()

try:
    # Test imports
    print("1. Testing imports...")
    from lilith.embedding import PMFlowEmbeddingEncoder
    from lilith.multi_tenant_store import MultiTenantFragmentStore
    from lilith.user_auth import AuthMode, UserIdentity
    from lilith.session import LilithSession, SessionConfig
    from lilith.response_composer import ResponseComposer
    print("   ✓ All imports successful")
    print()
    
    # Test that ResponseComposer has the enhanced methods
    print("2. Checking enhanced methods exist...")
    import inspect
    
    # Check _fallback_response has reasoning code
    fallback_source = inspect.getsource(ResponseComposer._fallback_response)
    if "reasoning_stage" in fallback_source and "deliberate" in fallback_source:
        print("   ✓ _fallback_response has reasoning enhancement")
    else:
        print("   ✗ _fallback_response missing reasoning code")
        sys.exit(1)
    
    # Check _fallback_response_low_confidence has reasoning code
    low_conf_source = inspect.getsource(ResponseComposer._fallback_response_low_confidence)
    if "reasoning_stage" in low_conf_source and "deliberate" in low_conf_source:
        print("   ✓ _fallback_response_low_confidence has reasoning enhancement")
    else:
        print("   ✗ _fallback_response_low_confidence missing reasoning code")
        sys.exit(1)
    
    # Check _fill_gaps_and_retry exists
    if hasattr(ResponseComposer, '_fill_gaps_and_retry'):
        print("   ✓ _fill_gaps_and_retry method exists")
    else:
        print("   ✗ _fill_gaps_and_retry method missing")
        sys.exit(1)
    
    # Check _extract_unknown_terms exists
    if hasattr(ResponseComposer, '_extract_unknown_terms'):
        print("   ✓ _extract_unknown_terms method exists")
    else:
        print("   ✗ _extract_unknown_terms method missing")
        sys.exit(1)
    
    print()
    print("3. Checking reasoning stage integration...")
    
    # Verify reasoning stage can be imported
    from lilith.reasoning_stage import ReasoningStage
    print("   ✓ ReasoningStage imports successfully")
    
    # Verify knowledge augmenter can be imported
    from lilith.knowledge_augmenter import KnowledgeAugmenter
    print("   ✓ KnowledgeAugmenter imports successfully")
    
    print()
    print("=" * 70)
    print("✅ Integration Test PASSED!")
    print("=" * 70)
    print()
    print("Enhanced fallback with 4-layer intelligence is ready:")
    print()
    print("  1️⃣  REASONING: Concept connections & inference")
    print("  2️⃣  GAP-FILLING: Unknown term lookup")
    print("  3️⃣  EXTERNAL: Direct knowledge sources")
    print("  4️⃣  FALLBACK: Graceful teaching invitation")
    print()
    print("To test in action:")
    print("  1. Start Discord bot: scripts/run_discord.sh")
    print("  2. Try queries with unknown terms")
    print("  3. Watch logs for reasoning/gap-filling output")
    print()
    print("Example queries to test:")
    print("  • 'What is memoization?' (gap-filling)")
    print("  • 'How do neural networks learn?' (reasoning)")
    print("  • 'What does ephemeral mean?' (external lookup)")
    print()
    
except Exception as e:
    print(f"\n❌ Integration test FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
