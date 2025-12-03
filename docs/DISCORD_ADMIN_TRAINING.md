# Discord Admin Training Guide

A practical guide for Discord server admins who want to train Lilith for their server.

## Quick Start

### 1. Set Up Teaching Permissions

By default, anyone can teach Lilith. To restrict teaching to specific roles:

```
/teachrole add @Trainers
/teachrole add @Moderators
```

To remove a teaching role:
```
/teachrole remove @Trainers
```

### 2. Enable/Disable Learning

```
/settings setting:learning_enabled value:true    # Enable learning
/settings setting:learning_enabled value:false   # Disable learning
/settings setting:view_all                       # View all settings
```

---

## Training Methods

### Method 1: Interactive Teaching (Easiest)

Use the `/teach` command directly in Discord:

```
/teach question:"What is your refund policy?" answer:"We offer full refunds within 30 days of purchase with original receipt."
```

**Best for:** Quick fixes, small additions, corrections

### Method 2: Q&A File (Bulk Training)

Create a text file with question-answer pairs:

```text
# Store FAQ - Training Data
# Format: Q: question followed by A: answer

Q: What are your business hours?
A: We're open Monday-Friday, 9am-6pm, and Saturday 10am-4pm.

Q: How can I contact support?
A: Email support@example.com or call 1-800-EXAMPLE.

Q: Do you offer gift cards?
A: Yes! Gift cards are available in $25, $50, and $100 denominations.
```

Save as `my_training.txt` and run:

```bash
# Copy to qa_bootstrap.txt and run
cp my_training.txt data/qa_bootstrap.txt
python scripts/bootstrap_qa.py --data-dir data/servers/YOUR_SERVER_ID
```

**Best for:** Initial setup, large knowledge bases

### Method 3: Document Ingestion (Unstructured Content)

Train from existing documents like manuals, FAQs, or articles:

```bash
# From a single file
python scripts/train_from_document.py --file docs/employee_handbook.pdf --server-id 123456789

# From a directory
python scripts/train_from_document.py --dir docs/knowledge_base/ --server-id 123456789

# From a URL
python scripts/train_from_document.py --url "https://example.com/faq" --server-id 123456789
```

**Supported formats:** `.txt`, `.md`, `.html`, `.pdf`, `.docx`

**Best for:** Existing documentation, manuals, articles

---

## Training for Different Use Cases

### Customer Support Bot

**Key training areas:**
- Product/service information
- Pricing and policies
- Common issues and solutions
- Contact information

**Example Q&A file:**
```text
# Product Information
Q: What products do you sell?
A: We sell premium coffee beans, brewing equipment, and accessories.

Q: What's your best-selling coffee?
A: Our Ethiopian Yirgacheffe is our most popular single-origin coffee.

# Policies
Q: What's your return policy?
A: We accept returns within 30 days for unopened products.

Q: Do you offer free shipping?
A: Yes! Free shipping on orders over $50.

# Support
Q: How do I track my order?
A: You'll receive a tracking link via email once your order ships.

Q: My order is damaged, what do I do?
A: Please email support@example.com with photos of the damage for a replacement.
```

### Educational Bot

**Key training areas:**
- Subject matter definitions
- Explanations of concepts
- Examples and illustrations
- Related topics

**Example Q&A file:**
```text
# Biology Basics
Q: What is photosynthesis?
A: Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.

Q: Why is photosynthesis important?
A: It produces oxygen for animals to breathe and is the foundation of most food chains on Earth.

Q: Where does photosynthesis occur?
A: Photosynthesis occurs in chloroplasts, specifically in the chlorophyll-containing parts of plant leaves.
```

### Community/Gaming Bot

**Key training areas:**
- Server rules and guidelines
- Event information
- Game-specific knowledge
- Community inside jokes (if appropriate)

**Example Q&A file:**
```text
# Server Rules
Q: What are the server rules?
A: Our main rules are: 1) Be respectful, 2) No spam, 3) Keep content appropriate, 4) No advertising.

Q: What happens if I break the rules?
A: First offense is a warning, second is a 24-hour timeout, third is a permanent ban.

# Events
Q: When is game night?
A: Game night is every Friday at 8pm EST in the Gaming voice channel.

Q: How do I join events?
A: React with ✅ to event announcements to sign up!
```

---

## Best Practices

### Writing Good Training Data

| Do | Don't |
|-----|-------|
| Use natural questions | Use keywords only |
| Provide complete answers | Give one-word answers |
| Include variations | Only train one phrasing |
| Organize by topic | Mix unrelated topics |

**Good:**
```
Q: What are your store hours?
A: We're open Monday through Friday from 9am to 6pm, and Saturday from 10am to 4pm. We're closed on Sundays.
```

**Bad:**
```
Q: hours
A: 9-6
```

### Include Question Variations

Train multiple phrasings of the same question:

```text
Q: What are your hours?
A: We're open Monday-Friday 9am-6pm, Saturday 10am-4pm.

Q: When are you open?
A: We're open Monday-Friday 9am-6pm, Saturday 10am-4pm.

Q: What time do you close?
A: We close at 6pm on weekdays and 4pm on Saturdays.

Q: Are you open on weekends?
A: We're open Saturday 10am-4pm but closed on Sundays.
```

### Use the Feedback System

Encourage users to react to responses:
- 👍 Upvotes good responses (reinforces the pattern)
- 👎 Downvotes bad responses (weakens the pattern)

Over time, this improves response quality automatically.

---

## Where Data Is Stored

| Scope | Location | Description |
|-------|----------|-------------|
| Server | `data/servers/{server_id}/` | Shared by all users in server |
| User (DM) | `data/users/{user_id}/` | Private to individual user |
| Base | `data/base/` | Shared baseline knowledge |

Each location contains:
- `patterns.db` - Response patterns
- `concepts.db` - Semantic concepts
- `vocabulary.db` - Learned terms

---

## Troubleshooting

### "You don't have permission to teach"

**Solution:** Ask a server admin to add your role:
```
/teachrole add @YourRole
```

### Bot gives wrong answers

**Solutions:**
1. 👎 React to downvote the bad response
2. Re-teach with correct answer:
   ```
   /teach question:"[the question]" answer:"[correct answer]"
   ```

### Teaching doesn't seem to work

**Check:**
1. Learning is enabled: `/settings setting:view_all`
2. You have teaching permissions
3. The server has write access to data directory

### Bot is too verbose/quiet

**Adjust settings:**
```
/settings setting:min_confidence value:0.5   # Lower = more responses
/settings setting:min_confidence value:0.8   # Higher = fewer, more confident responses
```

---

## Advanced: Bulk Import Script

For very large training sets, create a Python script:

```python
#!/usr/bin/env python3
"""Bulk training script for Discord server."""

import json
from lilith.session import LilithSession, SessionConfig

# Your server ID
SERVER_ID = "123456789012345678"

# Training data
training_data = [
    {"q": "What is X?", "a": "X is..."},
    {"q": "How does Y work?", "a": "Y works by..."},
    # ... more pairs
]

# Initialize session for server
config = SessionConfig(
    data_path=f"data/servers/{SERVER_ID}",
    use_grammar=True,
    enable_concepts=True
)
session = LilithSession(config)

# Train
for item in training_data:
    session.teach(item["q"], item["a"])
    print(f"Taught: {item['q'][:50]}...")

print(f"✅ Trained {len(training_data)} patterns")
```

---

## Getting Help

- Check `/help` in Discord for available commands
- See [USER_GUIDE.md](USER_GUIDE.md) for general usage
- See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for technical details
- Open an issue on GitHub for bugs
