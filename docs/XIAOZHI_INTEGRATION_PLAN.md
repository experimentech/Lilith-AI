# Xiaozhi Integration Plan

## Goals
- Expose a host-side Xiaozhi-compatible endpoint that drives `LilithSession` without requiring physical hardware.
- Keep Xiaozhi as a branch off the cognitive tree: trunk stays LilithSession (state, memory, learning), branch handles voice/device transport and schemas.
- Provide a minimal, testable WebSocket MVP first; stage MQTT/UDP parity as an advanced mode.

## Protocol snapshot (client expectations)
- WebSocket path: client sends hello `{type: "hello", version: 1, features: {mcp: true}, transport: "websocket", audio_params: {format: "opus", sample_rate, channels, frame_duration}}` with headers `Authorization: Bearer <token>`, `Protocol-Version: 1`, `Device-Id`, `Client-Id`. Server replies `type: "hello", transport: "websocket"` and keeps the socket for JSON + audio frames.
- MQTT/UDP path: client publishes hello `{type: "hello", version: 3, features: {mcp: true}, transport: "udp", audio_params: {...}}`; server replies with `session_id` and `udp {server, port, key, nonce}`. Audio is AES-CTR over UDP; nonce = prefix + length + nonce tail + seq; packet = 16-byte nonce + ciphertext.
- Common JSON messages after hello (all carry `session_id`): `listen` (start/stop/detect, mode realtime|auto|manual), `abort` (optional reason `wake_word_detected`), `mcp {payload}`, `iot {descriptors|states}`. Audio frames follow the transport.

## Cognitive tree fit (branching)
- Trunk stays unchanged: LilithSession handles dialog state, retrieval, augmentation, feedback.
- New branch: "Voice/Device client" that
  - Maps listen/abort/wake to session lifecycle cues and audio capture boundaries.
  - Passes audio/text into existing modal routing (speech/text) without bypassing safety.
  - Bridges MCP payloads to existing MCP adapter/server plumbing where possible.
  - Exposes optional IoT state hooks (descriptors/states) for future device control, stored transiently unless explicitly promoted.

## Minimum viable server surface (WebSocket-first)
- File: `xiaozhi_server.py` (or a FastAPI router) that:
  - Accepts WebSocket with auth headers; validates token (configurable noop for dev); sends server hello `{type: "hello", transport: "websocket", session_id, features:{mcp:true}}`.
  - Generates session IDs and instantiates a LilithSession per connection with timeouts and cache limits consistent with mcp_server.
  - Handles JSON messages: `listen start/stop/detect` toggles capture; `abort` maps to cancellation; `mcp` forwards payload into existing MCP adapter; `iot` optionally echoed/logged for now.
  - Accepts binary audio frames (Opus) and forwards to speech pipeline (stub until wired). Returns text or audio responses as JSON for MVP; streaming audio reply is stretch.
  - Graceful close: send `goodbye` and release session resources.

### Implemented (MVP)
- WebSocket endpoint lives at `/xiaozhi/ws` inside the FastAPI app (`lilith.mcp_server:app`).
- Env flags: `XIAOZHI_AUTH_TOKEN` (optional bearer token check) and `XIAOZHI_AUDIO_ENABLED` (default off; if off, audio frames return `audio_disabled`).
- Behaviors: server hello on connect; acks for `listen`/`abort`/`iot`; `mcp` payloads support JSON-RPC 2.0 (`initialize`, `tools/list`, `tools/call`) via a Xiaozhi tool registry, plus a simple chat passthrough for non-RPC payloads. Binary audio frames are accepted but not processed when audio is disabled.
- Tests: `tests/test_xiaozhi_websocket.py` covers hello, mcp chat round-trip, listen ack, auth guard, audio-disabled ack, JSON-RPC initialize/tools/list/tools/call, and unknown tool error.

### Audio bridge abstraction
- `lilith/speech_bridge.py` defines `SpeechBridge` (transcribe/synthesize/supports_tts) and `NullSpeechBridge` (default no-op). `set_speech_bridge()` lets deployments plug ASR/TTS without touching core code. Audio remains disabled by default; when enabled, ASR→chat→optional TTS flows through the bridge.

## Toward feature parity (MQTT/UDP mode)
- Add optional MQTT broker support that mirrors `mcp_server` auth config; publish/subscribe topics from config.
- Implement UDP audio channel with AES-CTR using server-provided `key` and `nonce`; maintain per-session sequence and length fields to match client expectations.
- Allow transport selection via config (websocket|mqtt_udp); default to websocket.

## Libraries and infra needed
- WebSocket MVP: `fastapi` + `uvicorn` (already present) or `websockets` for a minimal server; `python-multipart` only if file uploads appear (not required now).
- MQTT/UDP: `paho-mqtt`, `cryptography` for AES-CTR; `socket` from stdlib for UDP; optional `aiohttp`/`asyncio` helpers if we want async MQTT bridge.
- Testing: `websockets` client in tests for handshake/JSON; `pytest-asyncio` for async fixtures; `paho-mqtt` and `cryptography` for UDP tests.

## Testing plan (no hardware)
- WebSocket handshake test: client connects, verifies server hello, sends `listen start` and receives ack or state transition, sends `mcp` payload and gets routed stub response.
- Abort/wake routing test: `abort` with `wake_word_detected` cancels pending generation; `listen detect` registers wake state.
- MCP passthrough test: ensure `mcp` payload reaches MCP adapter (mock) and response is returned; observe rate limits/ttl if enabled.
- UDP encryption test (later): given key/nonce, encrypt/decrypt a frame matches client formula; verify session hello advertises UDP params.
- Resource cleanup test: server closes session and cancels tasks on disconnect.

## Open decisions
- Auth: use bearer token from config vs. dev-mode no-auth flag.
- Audio handling: do we return audio (Opus) to client or text-only for first cut?
- IoT surface: echo-only for now vs. simple device registry adapter.
- Transport choice: if FastAPI is preferred, implement WebSocket endpoint in existing app; otherwise keep a standalone `websockets` server for simplicity.
