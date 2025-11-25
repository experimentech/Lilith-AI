# Multi-Modal Pipeline Architecture

## Overview

The neuro-symbolic pipeline architecture is designed to be **modality-agnostic** at the semantic layer. This enables integration of diverse input sources (vision, audio, web data) and output actions (calendar, device control, web interactions) without modifying core cognitive processing.

## Core Principle

All modalities converge to a shared semantic representation space, allowing the same learning and reasoning mechanisms to work across different input/output formats.

```
Input Modality → Intake Stage → SemanticArtifact → Semantic/Pragmatic Processing 
→ ComposedResponse → Output Stage → Action Modality
```

The middle layers (semantic understanding, pragmatic response generation) operate on standardized `Artifact` objects, remaining independent of the source or destination modality.

---

## Input Side: Multi-Modal Perception

### Current Implementation
```python
# Text input
text_intake = IntakeStage()
artifact = text_intake.process(user_text)  # → SemanticArtifact
```

### Vision Input Extension
```python
class VisionIntakeStage:
    """Process images into semantic artifacts."""
    
    def __init__(self, vision_model, semantic_encoder):
        self.vision_model = vision_model  # CLIP, ViT, etc.
        self.semantic_encoder = semantic_encoder
    
    def process(self, image) -> SemanticArtifact:
        # 1. Extract visual features
        image_features = self.vision_model.encode(image)
        
        # 2. Generate caption/description
        caption = self.vision_model.caption(image)
        
        # 3. Create semantic embedding in shared text-vision space
        embedding = self.semantic_encoder.encode(caption)
        
        # 4. Extract visual concepts (objects, colors, scenes)
        concepts = self._extract_visual_concepts(image_features)
        
        # 5. Return standard artifact - same interface as text!
        return SemanticArtifact(
            text=caption,
            embedding=embedding,
            concepts=concepts,
            metadata={'modality': 'vision', 'confidence': self._compute_confidence()}
        )
    
    def _extract_visual_concepts(self, features) -> List[Concept]:
        """Map visual features to semantic concepts."""
        # Object detection → concept nodes
        # Color analysis → attribute concepts
        # Scene classification → context concepts
        pass
```

### Web Scraper Input Extension
```python
class WebScraperIntakeStage:
    """Fetch and process web content into semantic artifacts."""
    
    def __init__(self, semantic_encoder):
        self.semantic_encoder = semantic_encoder
        self.scraper = WebScraper()
    
    def process(self, url: str) -> SemanticArtifact:
        # 1. Fetch and clean web content
        raw_html = self.scraper.fetch(url)
        clean_text = self.scraper.extract_main_content(raw_html)
        
        # 2. Chunk into semantic units
        chunks = self._chunk_text(clean_text)
        
        # 3. Encode each chunk
        embeddings = [self.semantic_encoder.encode(chunk) for chunk in chunks]
        
        # 4. Extract key facts/concepts
        concepts = self._extract_web_concepts(clean_text)
        
        # 5. Return as standard artifact
        return SemanticArtifact(
            text=clean_text,
            embedding=self._aggregate_embeddings(embeddings),
            concepts=concepts,
            metadata={'source': url, 'modality': 'web'}
        )
```

### Audio Input Extension
```python
class AudioIntakeStage:
    """Process speech audio into semantic artifacts."""
    
    def __init__(self, speech_to_text, semantic_encoder):
        self.stt = speech_to_text  # Whisper, etc.
        self.semantic_encoder = semantic_encoder
    
    def process(self, audio_data) -> SemanticArtifact:
        # 1. Transcribe speech to text
        transcript = self.stt.transcribe(audio_data)
        
        # 2. Extract prosody/emotion (optional)
        emotion = self._analyze_prosody(audio_data)
        
        # 3. Create semantic embedding
        embedding = self.semantic_encoder.encode(transcript)
        
        # 4. Extract concepts from transcript
        concepts = self._extract_speech_concepts(transcript)
        
        # 5. Return standard artifact
        return SemanticArtifact(
            text=transcript,
            embedding=embedding,
            concepts=concepts,
            metadata={'emotion': emotion, 'modality': 'audio'}
        )
```

---

## Output Side: Multi-Modal Actions

### Current Implementation
```python
# Text response output
response_text = composer.compose_response(context, user_input)
print(response_text)  # Display to user
```

### Calendar Integration
```python
class CalendarOutputStage:
    """Execute calendar actions based on response intent."""
    
    def __init__(self, calendar_api):
        self.calendar = calendar_api
    
    def execute(self, composed_response: ComposedResponse) -> Optional[CalendarEvent]:
        # Check if response intent indicates scheduling
        if composed_response.intent not in ['schedule', 'reminder', 'appointment']:
            return None
        
        # Extract temporal and event information
        event_data = self._extract_event_details(composed_response)
        
        if event_data:
            # Create calendar event
            event = self.calendar.create_event(
                title=event_data['title'],
                start_time=event_data['start'],
                end_time=event_data['end'],
                description=composed_response.text
            )
            return event
        
        return None
    
    def _extract_event_details(self, response: ComposedResponse) -> Dict:
        """Parse response for date/time/title information."""
        # Use temporal extraction from semantic concepts
        # Look for date/time entities in response text
        # Extract action/subject as event title
        pass
```

### Web Action Output
```python
class WebActionStage:
    """Execute web searches or API calls based on response intent."""
    
    def __init__(self, search_engine):
        self.search = search_engine
    
    def execute(self, composed_response: ComposedResponse) -> Optional[SearchResults]:
        # Check for search/lookup intent
        if composed_response.intent not in ['web_search', 'lookup', 'research']:
            return None
        
        # Extract search query
        query = self._extract_search_query(composed_response)
        
        if query:
            # Execute search
            results = self.search.query(query, max_results=5)
            
            # Convert results to SemanticArtifacts for re-ingestion
            artifacts = [self._result_to_artifact(r) for r in results]
            
            return SearchResults(query=query, artifacts=artifacts)
        
        return None
    
    def _result_to_artifact(self, search_result) -> SemanticArtifact:
        """Convert search result to semantic artifact for synthesis."""
        # This enables the system to "learn" from search results
        # and incorporate them into its knowledge base
        pass
```

### Smart Home Control
```python
class HomeAutomationStage:
    """Control smart home devices based on response intent."""
    
    def __init__(self, device_controller):
        self.controller = device_controller
    
    def execute(self, composed_response: ComposedResponse) -> Optional[DeviceCommand]:
        # Check for device control intent
        if composed_response.intent not in ['device_control', 'automation']:
            return None
        
        # Parse device and action from response
        command = self._parse_device_command(composed_response)
        
        if command:
            # Execute on device
            result = self.controller.send_command(
                device=command.device,
                action=command.action,
                parameters=command.params
            )
            return result
        
        return None
    
    def _parse_device_command(self, response: ComposedResponse) -> Optional[DeviceCommand]:
        """Extract device type and action from response."""
        # Example: "turn on living room lights" 
        # → DeviceCommand(device='lights', room='living_room', action='on')
        pass
```

### Tool Use Orchestration
```python
class ToolUseStage:
    """Orchestrate multiple tools based on task requirements."""
    
    def __init__(self):
        self.tools = {
            'search': WebSearchTool(),
            'calculator': CalculatorTool(),
            'calendar': CalendarTool(),
            'weather': WeatherAPITool(),
            'translator': TranslationTool()
        }
    
    def execute(self, composed_response: ComposedResponse) -> ToolResult:
        # Determine which tool(s) are needed
        required_tools = self._identify_tools(composed_response)
        
        results = []
        for tool_name in required_tools:
            tool = self.tools.get(tool_name)
            if tool:
                # Execute tool with extracted parameters
                params = self._extract_tool_params(composed_response, tool_name)
                result = tool.execute(**params)
                results.append(result)
        
        # Aggregate results and feed back to intake for synthesis
        return self._synthesize_results(results)
    
    def _identify_tools(self, response: ComposedResponse) -> List[str]:
        """Determine which tools are needed based on intent and content."""
        # Intent-based routing:
        # "calculate" → calculator
        # "search" → web search
        # "weather" → weather API
        # "translate" → translator
        pass
```

---

## Cross-Modal Learning

### The Key Insight

Because all modalities map to the same semantic embedding space, **learning transfers across modalities**:

```python
# Learn from text
system.learn_pattern(
    trigger="red apple",
    response="It's a fruit",
    intent="classification"
)

# Same pattern matches vision input!
image = load_image("red_apple.jpg")
vision_artifact = vision_intake.process(image)
# vision_artifact.embedding ≈ text_embedding("red apple")

response = composer.compose_response(vision_artifact.text)
# → "It's a fruit" (learned pattern applies!)
```

### Multi-Modal Pattern Storage

```python
@dataclass
class MultiModalPattern:
    """Pattern that can be triggered by any modality."""
    
    trigger_embedding: np.ndarray  # Shared semantic space
    response_template: str
    modality_hints: Dict[str, Any]  # Modality-specific metadata
    
    # Can match:
    # - Text: "show me a red apple"
    # - Vision: [image of red apple]
    # - Audio: "what's this?" + [image context]
```

### Feedback Loops Across Modalities

```python
# User shows image
vision_input = vision_intake.process(user_image)
response = composer.compose("What is this?", context=vision_input)
# → "It's a red apple"

# User corrects
user_says("No, it's a tomato")

# Learn correction - updates pattern for BOTH text and vision
learner.observe_correction(
    original_response=response,
    correction="tomato",
    embedding=vision_input.embedding  # Links correction to visual features
)

# Next time, similar images trigger corrected response
```

---

## Integration Patterns

### 1. Input Multiplexing
Multiple inputs converge to single semantic representation:
```python
class MultiModalIntake:
    def process(self, text=None, image=None, audio=None) -> SemanticArtifact:
        artifacts = []
        
        if text:
            artifacts.append(text_intake.process(text))
        if image:
            artifacts.append(vision_intake.process(image))
        if audio:
            artifacts.append(audio_intake.process(audio))
        
        # Fuse into unified semantic representation
        return self._fuse_artifacts(artifacts)
```

### 2. Output Demultiplexing
Single response triggers multiple output actions:
```python
class MultiModalOutput:
    def execute(self, composed_response: ComposedResponse):
        results = []
        
        # Text output (always)
        results.append(text_output.execute(composed_response))
        
        # Conditional outputs based on intent
        if 'schedule' in composed_response.intent:
            results.append(calendar_output.execute(composed_response))
        
        if 'search' in composed_response.intent:
            results.append(web_output.execute(composed_response))
        
        return results
```

### 3. Recursive Processing
Output feeds back as input for multi-step reasoning:
```python
# Step 1: User asks question requiring lookup
user_input = "What's the weather in Tokyo?"
response1 = composer.compose_response(context, user_input)
# Intent: 'web_search'

# Step 2: Execute search
search_results = web_action.execute(response1)

# Step 3: Re-ingest results as new semantic input
result_artifact = web_intake.process(search_results.content)

# Step 4: Synthesize final response
response2 = composer.compose_response(result_artifact.text, user_input)
# → "The weather in Tokyo is 18°C and partly cloudy."
```

---

## Implementation Roadmap

### Phase 1: Abstraction
- [ ] Define `ModalityStage` base class
- [ ] Standardize `Artifact` interface across modalities
- [ ] Create `OutputAction` protocol

### Phase 2: Vision Integration
- [ ] Implement `VisionIntakeStage` with CLIP
- [ ] Test cross-modal learning (text ↔ vision)
- [ ] Validate semantic space alignment

### Phase 3: Action Outputs
- [ ] Implement `CalendarOutputStage`
- [ ] Implement `WebActionStage`
- [ ] Test intent-based output routing

### Phase 4: Tool Use
- [ ] Create `ToolUseStage` orchestrator
- [ ] Implement recursive processing loop
- [ ] Add result synthesis

### Phase 5: Full Integration
- [ ] Multi-modal input fusion
- [ ] Multi-modal output execution
- [ ] Cross-modal learning validation

---

## Benefits

1. **Modularity**: Add new modalities without changing core logic
2. **Transferability**: Learning in one modality applies to others
3. **Composability**: Combine inputs and outputs flexibly
4. **Scalability**: Each modality can be optimized independently
5. **Consistency**: Same reasoning/learning mechanisms everywhere

---

## Example Use Cases

### Personal Assistant
- **Input**: Voice command + camera image
- **Processing**: Understand "What's this?" + image of object
- **Output**: Text response + calendar event if scheduling mentioned

### Smart Home Controller
- **Input**: Text command "Turn on lights when I arrive home"
- **Processing**: Extract intent, location, condition
- **Output**: Device command + confirmation text

### Research Assistant
- **Input**: Text query "Summarize recent papers on transformers"
- **Processing**: Identify research intent
- **Output**: Web search → scrape papers → synthesize summary → text response

### Multimodal Conversation
- **Input**: "Show me examples of this" [points at object in image]
- **Processing**: Vision recognition + text understanding
- **Output**: Web search for similar items → present image results + descriptions
