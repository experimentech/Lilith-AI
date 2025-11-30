# Discord Bot Setup Guide

This guide walks you through setting up and running the Lilith Discord bot.

## Prerequisites

- Python 3.10+
- A Discord account
- A server where you have admin permissions (or create a new one for testing)

## Step 1: Create a Discord Application

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications/)
2. Click **"New Application"**
3. Give it a name (e.g., "Lilith AI")
4. Click **Create**

## Step 2: Create the Bot

1. In your application, click **"Bot"** in the left sidebar
2. Click **"Add Bot"** (if not already created)
3. Under **Token**, click **"Reset Token"**
4. **Copy the token** - you'll need this! (It looks like: `MTIzNDU2Nzg5MDEyMzQ1Njc4.GAbCdE.abcdef...`)

> ⚠️ **IMPORTANT:** Never share your bot token! It's like a password. If you accidentally expose it, click "Reset Token" immediately.

## Step 3: Enable Privileged Intents

Still on the **Bot** page, scroll down to **"Privileged Gateway Intents"** and enable:

- ✅ **MESSAGE CONTENT INTENT** (required - lets the bot read messages)
- ✅ **SERVER MEMBERS INTENT** (optional but recommended)

Click **Save Changes**.

## Step 4: Generate Invite URL

1. Click **"OAuth2"** in the left sidebar
2. Click **"URL Generator"**
3. Under **Scopes**, check:
   - ✅ `bot`
   - ✅ `applications.commands`
4. Under **Bot Permissions**, check:
   - ✅ `Send Messages`
   - ✅ `Read Message History`
   - ✅ `Use Slash Commands`
   - ✅ `Add Reactions`
   - ✅ `Embed Links` (for rich responses)
5. Copy the **Generated URL** at the bottom

## Step 5: Add Bot to Your Server

1. Paste the generated URL in your browser
2. Select **Guild Install** (standard server installation)
3. Choose a server you own or have admin access to
4. Click **Authorize**
5. Complete the CAPTCHA if prompted

The bot should now appear in your server's member list (usually offline until you start it).

## Step 6: Configure the Bot Locally

### Option A: Using Setup Script (Recommended)

```bash
# Clone the repo (if you haven't)
git clone https://github.com/experimentech/Lilith-AI.git
cd Lilith-AI

# Run setup (creates venv, installs dependencies)
./scripts/setup.sh
# When prompted, say 'y' to install Discord dependencies
```

### Option B: Manual Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install discord.py python-dotenv

# Copy the environment template
cp .env.example .env
```

## Step 7: Add Your Bot Token

Edit the `.env` file and add your bot token:

```bash
# Discord Bot Configuration
DISCORD_TOKEN=your_actual_bot_token_here
```

Replace `your_actual_bot_token_here` with the token you copied in Step 2.

## Step 8: Run the Bot

```bash
./scripts/run_discord.sh
```

You should see:
```
🚀 Starting Lilith Discord Bot...

🤖 Lilith Discord Bot is ready!
   Logged in as: YourBotName#1234
   Servers: 1
   Session timeout: 30 minutes
   User retention: 7 days
   Synced N slash commands
```

The bot is now online and ready!

## Testing the Bot

### Direct Message
1. Find the bot in your server's member list
2. Right-click → **Message**
3. Chat with it in DMs

### @Mention in Server
```
@Lilith hello!
@Lilith my name is John
@Lilith what's my name?
```

### Slash Commands
Type `/` in any channel to see commands:
- `/help` - Show help information
- `/setname <name>` - Set your preferred name
- `/whoami` - See what Lilith knows about you
- `/stats` - View session statistics
- `/forget` - Clear your stored data

### Reaction Feedback
React to Lilith's responses:
- 👍 ❤️ ✅ 🎉 💯 = Positive feedback (improves similar responses)
- 👎 ❌ 😕 🤔 = Negative feedback (reduces similar responses)

## Troubleshooting

### "Privileged Intents Required" Error
Go back to Step 3 and enable the MESSAGE CONTENT INTENT.

### Bot Doesn't Respond
- Make sure the bot is online (green dot in member list)
- Check that you're either DMing it or @mentioning it
- Verify the bot has permission to read/send in the channel

### Slash Commands Don't Appear
- Commands can take up to an hour to sync globally
- Try using them in the server where you added the bot first
- Restart the bot to force a sync

### "Token Invalid" Error
- Make sure you copied the full token
- Check there are no extra spaces in your `.env` file
- Try resetting the token in the Developer Portal

## Configuration Options

The bot supports these configuration options in `discord_bot.py`:

```python
bot = LilithDiscordBot(
    data_path="data",              # Where to store user data
    session_timeout_minutes=30,    # Free memory after inactivity
    user_retention_days=7          # Delete user data after N days
)
```

---

## Server Settings & Behavior

Use `/settings` command to configure how the bot behaves in your server.

### 🎧 Passive Listening (Default: **ENABLED**)

**What it does:**
- **ENABLED** ✅: Bot silently observes ALL messages for learning and context
  - Learns from conversations happening in the channel
  - Builds contextual understanding of topics being discussed
  - **Only responds when mentioned (@Lilith)** - won't spam!
  - When mentioned, has full conversation context for relevant responses
  
- **DISABLED** ❌: Bot ignores messages unless directly mentioned
  - No background learning from conversations
  - No contextual awareness of channel discussions
  - Still responds when mentioned, but without prior context

**Example with passive listening enabled:**
```
User A: "I'm debugging a Python async issue"
User B: "Have you tried asyncio.gather()?"
[Bot learns: Python, async, asyncio context]

User C: "@Lilith what are we talking about?"
Bot: "You're discussing Python async/await patterns and asyncio.gather()."
```

**Toggle:**
```
/settings passive_listening true   # Enable (recommended)
/settings passive_listening false  # Disable
```

**Recommendation:** Keep **enabled** for better context-aware responses.

---

### 🧠 Learning Enabled (Default: **ENABLED**)

**What it does:**
- **ENABLED** ✅: Bot learns and evolves from interactions
  - Processes feedback (👍/👎 reactions)
  - Semantic learning (word associations, concepts)
  - Neuroplasticity (pattern refinement, grammar improvements)
  - Knowledge accumulation over time
  
- **DISABLED** ❌: Bot responds but doesn't update knowledge
  - No learning from feedback or interactions
  - No pattern updates or knowledge growth
  - Still generates responses using existing knowledge
  - Useful for testing or "read-only" mode

**Toggle:**
```
/settings learning_enabled true    # Enable (recommended)
/settings learning_enabled false   # Disable for testing
```

**Recommendation:** Keep **enabled** for continuous improvement.

---

### 🎯 Minimum Confidence (Default: **0.3** / 30%)

Controls how confident the bot must be before responding.

**Adjust:**
```
/settings min_confidence 0.5   # Require 50% confidence
/settings view_all             # View all settings
```

---

### 👨‍🏫 Teaching Roles

Restrict who can teach the bot:

```
/teachrole add @TrustedRole
/teachrole remove @TrustedRole
```

---

## Teaching the Bot

**Method 1: `/teach` Command**
```
/teach
Question: What is the capital of France?
Answer: Paris
```

**Method 2: Reactions**
- 👍 ❤️ ✅ = Upvote
- 👎 ❌ = Downvote

**Method 3: Implicit Feedback**
```
Bot: "Python is functional only."
You: "That's wrong, it's multi-paradigm."
[Bot learns from correction]
```

---

## Privacy & Data

**Stored:** Message content (for learning), user preferences, server knowledge

**Not stored:** Edit history, deleted messages, private info

**Location:** `data/servers/{guild_id}/`, `data/users/{user_id}/`

---

## DM Behavior

In Direct Messages:
- ✅ Always responds
- ✅ Learning always enabled
- ✅ Settings don't apply

---

## Security Notes

- Each Discord user gets isolated storage (can't see others' data)
- Users cannot modify the base AI knowledge
- The `.env` file is gitignored - your token stays local
- User data is automatically cleaned up after the retention period

## Stopping the Bot

Press `Ctrl+C` in the terminal where the bot is running.

## Running as a Background Service

For production deployment, consider using:
- `systemd` service (Linux)
- `pm2` process manager
- Docker container
- Cloud hosting (Railway, Fly.io, etc.)

---

## Quick Reference

| Action | Command |
|--------|---------|
| Start bot | `./scripts/run_discord.sh` |
| Run CLI instead | `./scripts/run_cli.sh` |
| Run tests | `./scripts/run_tests.sh` |
| View logs | Check terminal output |
| Stop bot | `Ctrl+C` |
