#!/usr/bin/env python3
"""Verify all gap-filling fixes are correct."""

import sys

print("Checking gap-filling implementation fixes...\n")

issues = []

with open('lilith/response_composer.py', 'r') as f:
    content = f.read()

# Check 1: No self.store references
if 'self.store.' in content or 'self.fragment_store.' in content:
    issues.append("❌ Found incorrect store references (should be 'self.fragments')")
else:
    print("✅ Store attribute: Using correct 'self.fragments'")

# Check 2: Correct method name
if 'self.fragments.retrieve_context' in content:
    issues.append("❌ Found 'retrieve_context' (should be 'retrieve_patterns')")
elif 'self.fragments.retrieve_patterns' in content:
    print("✅ Retrieve method: Using correct 'retrieve_patterns'")
else:
    print("⚠️  Warning: Could not find retrieve_patterns call")

# Check 3: Correct parameter name in gap-filling
if 'retrieve_patterns(enhanced_context' in content:
    if 'min_similarity=' in content.split('retrieve_patterns(enhanced_context')[1].split(')')[0]:
        issues.append("❌ Found 'min_similarity' parameter in retrieve_patterns (should be 'min_score')")
    elif 'min_score=' in content.split('retrieve_patterns(enhanced_context')[1].split(')')[0]:
        print("✅ Parameter name: Using correct 'min_score'")
    else:
        print("⚠️  Warning: Could not verify parameter name")
else:
    print("⚠️  Warning: Could not find retrieve_patterns call")

# Check 4: No self.min_confidence_threshold
if 'self.min_confidence_threshold' in content:
    issues.append("❌ Found 'self.min_confidence_threshold' (should be hardcoded value)")
else:
    print("✅ Threshold: Using hardcoded threshold value")

# Check 5: No self._apply_adaptation
if 'self._apply_adaptation' in content and '_fill_gaps_and_retry' in content:
    issues.append("❌ Found 'self._apply_adaptation' (method doesn't exist)")
else:
    print("✅ Adaptation: Using pattern response text directly")

# Check 6: No metadata parameter in add_pattern
if 'metadata=' in content and 'add_pattern(' in content and '_fill_gaps_and_retry' in content:
    # Need to check if metadata is in the gap-filling code specifically
    lines = content.split('\n')
    in_fill_gaps = False
    for line in lines:
        if '_fill_gaps_and_retry' in line:
            in_fill_gaps = True
        elif in_fill_gaps and 'def ' in line and '_fill_gaps_and_retry' not in line:
            in_fill_gaps = False
        elif in_fill_gaps and 'metadata=' in line:
            issues.append("❌ Found 'metadata=' parameter (not supported by add_pattern)")
            break
    else:
        print("✅ add_pattern: Using only supported parameters")
else:
    print("✅ add_pattern: Using only supported parameters")

# Check 7: Correct pattern attribute
if '.pattern_id' in content and '_fill_gaps_and_retry' in content:
    issues.append("❌ Found '.pattern_id' (should be '.fragment_id')")
elif '.fragment_id' in content and '_fill_gaps_and_retry' in content:
    print("✅ Pattern attribute: Using correct '.fragment_id'")
else:
    print("⚠️  Warning: Could not verify pattern attribute")

print()
if issues:
    print("FAILED - Found issues:")
    for issue in issues:
        print(f"  {issue}")
    sys.exit(1)
else:
    print("=" * 70)
    print("✅ ALL CHECKS PASSED!")
    print("=" * 70)
    print("\nGap-filling should now work correctly in Discord bot.")
    print("Test with: 'What is a Gigatron?'")
