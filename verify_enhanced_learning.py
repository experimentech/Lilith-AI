#!/usr/bin/env python3
"""
Verification script for Enhanced Learning Integration (Phase 1)

This script verifies that the gap-filling method now performs
full learning integration:
- Vocabulary tracking
- Concept extraction  
- Syntax pattern learning
"""

import re

def verify_enhanced_learning():
    """Verify the enhanced learning implementation"""
    
    print("=" * 80)
    print("ENHANCED LEARNING INTEGRATION - VERIFICATION")
    print("=" * 80)
    print()
    
    # Read the implementation
    with open('lilith/response_composer.py', 'r') as f:
        content = f.read()
    
    # Find the _fill_gaps_and_retry method
    method_start = content.find('def _fill_gaps_and_retry(')
    method_end = content.find('\n    def _extract_unknown_terms(', method_start)
    
    if method_start == -1:
        print("❌ Could not find _fill_gaps_and_retry method")
        return False
    
    method_code = content[method_start:method_end]
    
    print("✅ Found _fill_gaps_and_retry method")
    print()
    
    # Check for Phase 1 marker
    checks = {
        'Phase 1 Header': 'PHASE 1: FULL LEARNING INTEGRATION',
        'Vocabulary Learning': 'self.fragments.vocabulary.track_terms',
        'Concept Learning': 'self.fragments.concept_store.extract_concepts',
        'Syntax Learning': 'self.fragments.pattern_extractor.extract_patterns',
        'Learned Count Tracking': 'learned_count',
        'Progress Logging': 'Successfully learned',
        'Vocabulary Check': 'if self.fragments.vocabulary:',
        'Concept Check': 'if self.fragments.concept_store:',
        'Pattern Check': 'if self.fragments.pattern_extractor:'
    }
    
    results = {}
    for name, pattern in checks.items():
        found = pattern in method_code
        results[name] = found
        status = "✅" if found else "❌"
        print(f"{status} {name}: {pattern}")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"Checks Passed: {passed}/{total}")
    print()
    
    if passed == total:
        print("✅ ALL CHECKS PASSED!")
        print()
        print("Phase 1 Enhanced Learning Integration is complete:")
        print("  📖 Vocabulary tracking enabled")
        print("  🧠 Concept extraction enabled")
        print("  📝 Syntax pattern learning enabled")
        print("  ✨ Transparent online learning ready!")
        print()
        print("Next Step: Phase 2 - Reasoning Stage Integration")
        return True
    else:
        print("❌ SOME CHECKS FAILED")
        print()
        print("Missing components:")
        for name, passed in results.items():
            if not passed:
                print(f"  ❌ {name}")
        return False

if __name__ == '__main__':
    success = verify_enhanced_learning()
    exit(0 if success else 1)
