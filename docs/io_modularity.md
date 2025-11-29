# I/O Modularity in Lilith

## Architecture Overview

Lilith's cognitive core is **completely modality-agnostic**. The I/O layer is just a thin wrapper that converts input to text and displays output.

```
┌─────────────────────────────────────────────────────────────────┐
│                       I/O LAYER (Thin)                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │   CLI   │  │  Voice  │  │   API   │  │ Discord │  ...       │
│  │ (text)  │  │ (STT)   │  │ (JSON)  │  │  (bot)  │            │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│       │            │            │            │                  │
│       └────────────┴─────┬──────┴────────────┘                  │
│                          │ text                                 │
│                          ▼                                      │
├─────────────────────────────────────────────────────────────────┤
│                    COGNITIVE CORE                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  AutoSemanticLearner.process_text(text)                  │  │
│  │       │                                                   │  │
│  │       ▼                                                   │  │
│  │  ResponseComposer.compose_response(text, context)        │  │
│  │       │                                                   │  │
│  │       ├─► QueryPatternMatcher (intent/slots)             │  │
│  │       ├─► ReasoningStage (deliberation)                  │  │
│  │       ├─► ContrastiveLearner (semantic similarity)       │  │
│  │       ├─► FragmentStore (pattern retrieval)              │  │
│  │       └─► ConceptStore (knowledge base)                  │  │
│  │                                                           │  │
│  │       ▼                                                   │  │
│  │  ComposedResponse (text + metadata)                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          │ text                                 │
│                          ▼                                      │
├─────────────────────────────────────────────────────────────────┤
│                       I/O LAYER                                 │
│       ┌──────────────────┴──────────────────┐                  │
│       │           │           │             │                   │
│       ▼           ▼           ▼             ▼                   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│  │  print  │ │   TTS   │ │  JSON   │ │  embed  │  ...         │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

## Adding a New I/O Modality

To add a new input/output modality (voice, Discord, API, etc.), you only need:

### 1. Input Adapter (converts to text)

```python
class VoiceAdapter:
    def __init__(self, cognitive_core):
        self.core = cognitive_core
        self.auto_learner = AutoSemanticLearner(cognitive_core.contrastive_learner)
    
    def process_audio(self, audio_data) -> str:
        # Convert speech to text (using any STT)
        text = speech_to_text(audio_data)
        
        # Feed to cognitive core (SAME as CLI)
        response = self.core.compose_response(text)
        
        # Auto-learn from this interaction
        self.auto_learner.process_conversation(text, response.text)
        
        return response.text
```

### 2. Output Adapter (converts from text)

```python
    def respond(self, response_text: str):
        # Convert text to speech (using any TTS)
        audio = text_to_speech(response_text)
        play_audio(audio)
```

### 3. That's it!

No changes needed to:
- ResponseComposer
- ContrastiveLearner  
- ReasoningStage
- FragmentStore
- ConceptStore
- Any cognitive component

## What Each Layer Does

### I/O Layer (Modality-Specific)
| Component | Responsibility |
|-----------|----------------|
| `lilith_cli.py` | Terminal text I/O |
| Voice adapter | STT → text, text → TTS |
| API adapter | HTTP → text, text → JSON |
| Discord bot | Discord events → text |

### Cognitive Core (Modality-Agnostic)
| Component | Responsibility |
|-----------|----------------|
| `AutoSemanticLearner` | Extract semantic pairs from ANY text |
| `ContrastiveLearner` | Learn semantic similarity |
| `ReasoningStage` | Deliberate on concepts |
| `ResponseComposer` | Compose responses |
| `FragmentStore` | Store/retrieve patterns |
| `ConceptStore` | Store/retrieve concepts |

## Auto-Learning from Any Modality

The key insight: **all learning flows through text**.

```python
# In any I/O adapter:
from lilith.auto_semantic_learner import AutoSemanticLearner

# 1. Initialize once
auto_learner = AutoSemanticLearner(
    contrastive_learner=core.contrastive_learner,
    auto_train_threshold=10,  # Train after 10 new pairs
)

# 2. Call on every interaction (regardless of source)
auto_learner.process_conversation(
    user_input=transcribed_text,  # From voice, chat, etc.
    response=response_text
)
```

The `AutoSemanticLearner`:
1. Extracts semantic relationships from text
2. Adds pairs to ContrastiveLearner
3. Triggers incremental training automatically
4. Works identically for CLI, voice, API, etc.

## Example: Adding Voice Support

```python
# voice_adapter.py
import speech_recognition as sr
from gtts import gTTS

from lilith.response_composer import ResponseComposer
from lilith.auto_semantic_learner import AutoSemanticLearner

class VoiceLilith:
    def __init__(self, composer: ResponseComposer):
        self.composer = composer
        self.recognizer = sr.Recognizer()
        
        # Auto-learning works for voice too!
        self.auto_learner = AutoSemanticLearner(
            contrastive_learner=composer.contrastive_learner
        )
    
    def listen(self) -> str:
        with sr.Microphone() as source:
            audio = self.recognizer.listen(source)
        return self.recognizer.recognize_google(audio)
    
    def speak(self, text: str):
        tts = gTTS(text=text, lang='en')
        # ... play audio ...
    
    def conversation_turn(self):
        # Listen
        user_text = self.listen()
        
        # Think (SAME cognitive core as CLI)
        response = self.composer.compose_response(
            user_input=user_text,
            context=user_text
        )
        
        # Learn (SAME learning as CLI)
        self.auto_learner.process_conversation(user_text, response.text)
        
        # Speak
        self.speak(response.text)
```

## No Re-implementation Needed

| New Modality | What You Write | What's Reused |
|--------------|----------------|---------------|
| Voice | STT/TTS wrappers (~50 lines) | Everything else |
| Discord | Event handlers (~100 lines) | Everything else |
| REST API | Flask routes (~100 lines) | Everything else |
| Slack | Event adapter (~80 lines) | Everything else |

The cognitive architecture is:
- **Modality-agnostic**: Works with any text input
- **Self-learning**: Extracts patterns from any conversation
- **Persistent**: Same databases/models across all modalities
- **Composable**: Mix modalities (voice input → text response, etc.)
