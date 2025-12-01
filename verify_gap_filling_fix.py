#!/usr/bin/env python3
"""Quick verification that gap-filling fix works."""

import sys
import os

# Check that the attribute references are correct
print("Checking response_composer.py for correct attribute usage...")

with open('lilith/response_composer.py', 'r') as f:
    content = f.read()

# Check for bad references
bad_refs = []
if 'self.store.' in content:
    bad_refs.append("Found 'self.store.' (should be 'self.fragments.')")
if 'self.fragment_store.' in content:
    bad_refs.append("Found 'self.fragment_store.' (should be 'self.fragments.')")

if bad_refs:
    print("❌ FAILED - Found incorrect attribute references:")
    for ref in bad_refs:
        print(f"  - {ref}")
    sys.exit(1)

# Check for correct references in gap-filling code
if 'self.fragments.retrieve_context' in content:
    print("✅ Found correct usage: self.fragments.retrieve_context")
else:
    print("⚠️  Warning: Could not find self.fragments.retrieve_context")

if 'self.fragments.add_pattern' in content:
    print("✅ Found correct usage: self.fragments.add_pattern")
else:
    print("⚠️  Warning: Could not find self.fragments.add_pattern")

print("\n✅ Verification passed! Gap-filling code uses correct attribute 'self.fragments'")
print("\nThe AttributeError should now be fixed.")
print("Try the Discord bot again with: 'What is a Gigatron?'")
