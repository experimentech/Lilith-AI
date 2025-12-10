# I/O Modularity Architecture

This document describes the modular input/output architecture for Lilith, supporting multiple interface types while maintaining a consistent core.

## Design Principles

### Separation of Concerns
- **Core**: Session management, pattern storage, learning logic (interface-agnostic)
- **Interface**: CLI, Discord, API endpoints (swappable I/O handlers)
- **Storage**: Database backend (pluggable storage layer)

### Interface Independence
The core `LilithSession` class provides interface-agnostic conversation processing:

```python
from lilith.session import LilithSession, SessionConfig

config = SessionConfig(user_id="alice", mode="user")
session = LilithSession(config)
response = session.process_message("Hello")
```

Any interface can wrap this core functionality.

## Current Interfaces

### 1. Command-Line Interface (CLI)
**File**: `lilith_cli.py`

**Features**:
- Interactive REPL
- Command history
- Teacher/user modes
- /teach, /stats, /quit commands

**Usage**:
```bash
python lilith_cli.py --mode user --user alice
```

### 2. Discord Bot
**File**: `discord_bot.py`

**Features**:
- Multi-server support
- Per-user isolation
- Reaction-based feedback
- Slash commands
- Server-specific settings

**Usage**:
```bash
python discord_bot.py
```

### 3. Programmatic API (Internal)
**File**: `lilith/session.py`

**Usage**:
```python
session = LilithSession(config)
response = session.process_message(user_input)
```

## Storage Modularity

### Backend Abstraction
Storage is abstracted through the `MultiTenantFragmentStore` interface:

```python
# SQLite (current)
from lilith.multi_tenant_store import MultiTenantFragmentStore

# Future: PostgreSQL, MongoDB, etc.
# Just swap the implementation
```

### Isolation Levels
- **Base**: Shared knowledge accessible to all users
- **User**: Personal knowledge (isolated per user_id)
- **Server**: Discord server-specific knowledge (isolated per guild_id)

## Adding New Interfaces

### Step 1: Create Interface Module

```python
# interfaces/telegram_bot.py
from lilith.session import LilithSession, SessionConfig

class TelegramBot:
    def __init__(self):
        self.sessions = {}
    
    def handle_message(self, user_id: str, message: str):
        # Get or create session
        if user_id not in self.sessions:
            config = SessionConfig(user_id=user_id, mode="user")
            self.sessions[user_id] = LilithSession(config)
        
        # Process message
        response = self.sessions[user_id].process_message(message)
        return response
```

### Step 2: Handle Interface-Specific Features

```python
# Feedback (upvote/downvote)
def handle_reaction(self, message_id: str, reaction: str):
    pattern_id = self.get_pattern_id_for_message(message_id)
    if reaction == "👍":
        self.session.upvote(pattern_id)
    elif reaction == "👎":
        self.session.downvote(pattern_id)
```

### Step 3: Session Management

```python
# Timeout inactive sessions
def cleanup_sessions(self):
    for user_id, session in list(self.sessions.items()):
        if session.inactive_for(minutes=30):
            session.save_state()
            del self.sessions[user_id]
```

## Future Interfaces

### REST API
```python
# interfaces/rest_api.py (planned)
from fastapi import FastAPI
from lilith.session import LilithSession

app = FastAPI()

@app.post("/chat")
async def chat(user_id: str, message: str):
    session = get_or_create_session(user_id)
    return {"response": session.process_message(message)}
```

### Voice Interface
```python
# interfaces/voice.py (planned)
from speech_recognition import Recognizer
from lilith.session import LilithSession

def voice_chat():
    recognizer = Recognizer()
    session = LilithSession(config)
    
    while True:
        audio = recognizer.listen(microphone)
        text = recognizer.recognize_google(audio)
        response = session.process_message(text)
        speak(response)
```

### Web UI
- React/Vue frontend
- WebSocket for real-time chat
- Same backend session logic

## Benefits

### Reusability
Core logic is shared across all interfaces:
- Pattern retrieval
- Learning mechanisms
- Knowledge augmentation
- Multi-turn coherence

### Testability
Core can be tested independently:
```python
def test_learning():
    session = LilithSession(config)
    session.process_message("Python is a programming language")
    response = session.process_message("What is Python?")
    assert "programming language" in response
```

### Maintainability
Changes to core logic automatically benefit all interfaces.

### Scalability
New interfaces can be added without touching core code.

## Architecture Diagram

```
┌─────────────────────────────────────────┐
│         Interface Layer                 │
├──────────┬──────────┬──────────┬────────┤
│   CLI    │ Discord  │   API    │  Web   │
└──────────┴──────────┴──────────┴────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│      Core Session Logic                 │
│  - LilithSession                        │
│  - Pattern retrieval                    │
│  - Learning mechanisms                  │
│  - Multi-turn coherence                 │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│       Storage Layer                     │
│  - MultiTenantFragmentStore             │
│  - User isolation                       │
│  - Base/user/server knowledge           │
└─────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│      Database Backend                   │
│  - SQLite (current)                     │
│  - PostgreSQL (planned)                 │
└─────────────────────────────────────────┘
```

## Conclusion

The modular architecture allows Lilith to support multiple interfaces while maintaining a single source of truth for core functionality. This design enables rapid addition of new interfaces and ensures consistency across all interaction methods.
