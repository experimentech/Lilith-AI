# Quick Start: Interactive Testing with Lilith

## Running the Interactive CLI

### Option 1: User Mode (Recommended for Testing)

```bash
cd /home/tmumford/Coding/LLM/lilith
python lilith_cli.py
```

**When prompted:**
1. Choose `2` for User mode
2. Enter a username (e.g., "alice", "testuser", "bob")
3. Start chatting!

**Example session:**
```
============================================================
LILITH - Multi-Tenant Conversational AI
============================================================

Select mode:
  1. Teacher mode (writes to base knowledge)
  2. User mode (isolated personal storage)

Enter choice (1 or 2): 2

Username: alice
Display name (or Enter to use 'alice'): Alice

Welcome, Alice!

🔍 Fuzzy matching enabled for typo tolerance!
👤 User mode: Alice (isolated storage)
🤖 Lilith initialized

   Your patterns: 0
   Base patterns: 4
   Total accessible: 4

Type 'quit' to exit
Commands: 'stats', 'reset', 'help'
Feedback: '+' (upvote), '-' (downvote), '?' (show last pattern ID)
============================================================

You: hello
Lilith: Hello! How can I help you?

You: _
```

### Option 2: Teacher Mode (For Managing Base Knowledge)

```bash
python lilith_cli.py
```

Choose `1` for Teacher mode - writes directly to base knowledge that all users can see.

## Interactive Commands

While chatting, you can use these commands (all commands except `quit` start with `/`):

| Command | What it does |
|---------|-------------|
| `/help` | Show all commands |
| `/stats` | Display pattern statistics |
| `/+` | Upvote last response (makes it more likely) |
| `/-` | Downvote last response (suppresses it) |
| `/?` | Show last pattern ID |
| `/reset` | Reset your data with backup |
| `quit` | Exit |

## Testing Scenarios

### Scenario 1: Teaching Lilith

```
You: My name is Alice
Lilith: I'm not sure I understand. Could you rephrase that?
   [Fallback response - teach me!]

You: Remember, my name is Alice
Lilith: Let me try to explain that differently.
   [Fallback response - teach me!]

You: stats
📊 Statistics:
   Your patterns: 0
   Base patterns: 4
   Total: 4
```

Currently Lilith learns through the `ResponseComposer` automatically when you chat. To see learning in action, you'll need to interact multiple times with similar contexts.

### Scenario 2: Using Feedback

```
You: what is 2+2
Lilith: 2+2 equals 4
   [Modality: MATH]

You: +
👍 Upvoted pattern: pattern_math_12345

You: what is 1+1
Lilith: 1+1 equals 2
   [Modality: MATH]

You: -
👎 Downvoted pattern: pattern_math_67890
```

### Scenario 3: Math vs Linguistic

Lilith has modal routing - it detects MATH queries and uses symbolic computation:

```
You: what is 5 * 7
Lilith: 5 * 7 = 35
   [Modality: MATH]

You: what is your favorite color
Lilith: I'm not sure I understand. Could you rephrase that?
   [Fallback response - teach me!]
```

### Scenario 4: Reset Your Data

```
You: reset
⚠️  This will reset your personal patterns
Create backup and reset? (yes/no): yes
💾 Backup created: response_patterns.backup.20251126_123456.db
🔄 Database reset complete
📊 Old patterns: 15
📊 New patterns: 0
✅ Reset complete
```

## Testing Learning Behavior

To see Lilith learn, you need to use the teaching interface. Let me create a simple teaching script:

### Quick Teaching Script

Create `test_teaching.py`:

```python
from lilith.embedding import PMFlowEmbeddingEncoder
from lilith.multi_tenant_store import MultiTenantFragmentStore
from lilith.user_auth import UserIdentity, AuthMode

# Setup
encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
user = UserIdentity("alice", AuthMode.SIMPLE, "Alice")
store = MultiTenantFragmentStore(encoder, user)

# Teach Lilith some facts
facts = [
    ("what is your name", "My name is Lilith", 0.9, "identity"),
    ("who created you", "I was created by the experimentech team", 0.9, "identity"),
    ("what can you do", "I can learn from conversations and answer questions", 0.8, "capability"),
    ("my name is alice", "Nice to meet you, Alice!", 0.9, "greeting"),
    ("my favorite color is blue", "I'll remember that your favorite color is blue", 0.8, "personal"),
]

print("Teaching Lilith...")
for trigger, response, score, intent in facts:
    pattern_id = store.add_pattern(trigger, response, score, intent)
    print(f"✓ Taught: {trigger} → {response[:30]}...")

print(f"\n✅ Taught {len(facts)} patterns")

# Test retrieval
print("\nTesting retrieval:")
test_queries = ["what's your name", "who made you", "what is my favorite color"]

for query in test_queries:
    patterns = store.retrieve_patterns(query, topk=1, min_score=0.5)
    if patterns:
        pattern, confidence = patterns[0]
        print(f"\nQ: {query}")
        print(f"A: {pattern.response_text} (confidence: {confidence:.2f})")
    else:
        print(f"\nQ: {query}")
        print(f"A: No match found")
```

Run it:
```bash
python test_teaching.py
```

Then test in CLI:
```bash
python lilith_cli.py
# Choose user mode, username: alice

You: what is your name
Lilith: My name is Lilith

You: who created you  
Lilith: I was created by the experimentech team

You: what is my favorite color
Lilith: I'll remember that your favorite color is blue
```

## Multi-User Testing

Test user isolation:

**Terminal 1 (Alice):**
```bash
python lilith_cli.py
# Mode: 2 (User)
# Username: alice

You: my favorite food is pizza
Lilith: [learns this]
```

**Terminal 2 (Bob):**
```bash
python lilith_cli.py
# Mode: 2 (User)  
# Username: bob

You: what is alice's favorite food
Lilith: I'm not sure I understand
   [Bob can't see Alice's patterns]
```

## Checking Your Data

```bash
# List all users
python tools/manage_users.py list

# View stats
python tools/manage_users.py stats

# Export your data
python tools/manage_users.py export alice alice_patterns.json
```

## Database Locations

Your data is stored in:
```
data/
├── base/
│   └── response_patterns.db    # Shared base knowledge (teacher mode)
└── users/
    ├── alice/
    │   └── response_patterns.db
    ├── bob/
    │   └── response_patterns.db
    └── ...
```

## Troubleshooting

### "No module named 'lilith'"

Make sure you're running from the repository root:
```bash
cd /home/tmumford/Coding/LLM/lilith
python lilith_cli.py
```

### Virtual Environment

If you have a virtual environment:
```bash
source .venv/bin/activate
python lilith_cli.py
```

### Can't Find Patterns

Check if patterns exist:
```python
from pathlib import Path
import sqlite3

db = Path("data/users/alice/response_patterns.db")
if db.exists():
    conn = sqlite3.connect(str(db))
    cursor = conn.execute("SELECT COUNT(*) FROM response_patterns")
    print(f"Patterns: {cursor.fetchone()[0]}")
    conn.close()
```

## Next Steps

1. **Run the CLI**: `python lilith_cli.py`
2. **Choose user mode** and create a test user
3. **Try the teaching script** above to add patterns
4. **Test retrieval** by asking questions
5. **Use feedback** (`+` and `-`) to tune responses
6. **Check stats** to see your learning progress

Have fun testing! 🚀
