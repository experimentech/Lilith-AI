#!/usr/bin/env python3
"""Quick test of the /teach command"""

import subprocess
import sys

print("Testing /teach command in CLI...")
print("=" * 60)

# Simulate user input: select teacher mode, use /help, and quit
user_input = "1\n/help\n/quit\n"

try:
    result = subprocess.run(
        [sys.executable, 'lilith_cli.py'],
        input=user_input,
        text=True,
        capture_output=True,
        timeout=15
    )
    
    # Check for /teach in help output
    if '/teach' in result.stdout:
        print("✅ /teach command found in help menu!")
        print()
        print("Help menu excerpt:")
        print("-" * 60)
        
        # Extract help section
        lines = result.stdout.split('\n')
        for i, line in enumerate(lines):
            if '📖 Commands:' in line:
                # Print the next 12 lines (the help menu)
                for j in range(i, min(i + 12, len(lines))):
                    print(lines[j])
                break
        print("-" * 60)
    else:
        print("❌ /teach command NOT found in help")
        print()
        print("Output preview:")
        print(result.stdout[:500])
        
except subprocess.TimeoutExpired:
    print("⚠️  Test timed out - CLI might be hanging")
except Exception as e:
    print(f"❌ Test failed: {e}")

print()
print("To manually test /teach:")
print("  1. Run: python lilith_cli.py")
print("  2. Select mode (1 or 2)")
print("  3. Type: /teach")
print("  4. Enter a question and answer")
