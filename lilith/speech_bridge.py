"""
Pluggable speech bridge abstraction for Xiaozhi audio flows.

Implementations can provide ASR (transcribe) and optional TTS (synthesize)
without changing Lilith core logic. A NullSpeechBridge is used by default so
text-only deployments pay no dependency cost.
"""
from typing import Optional


class SpeechBridge:
    """Interface for speech backends.

    Implementations should be thread-safe or reentrant because calls may be
    dispatched via thread pools.
    """

    def transcribe(self, audio_bytes: bytes, session_id: str, client_id: str) -> Optional[str]:
        """Convert audio bytes to text. Return None if no transcript is available."""

        raise NotImplementedError

    def synthesize(self, text: str, session_id: str, client_id: str) -> Optional[bytes]:
        """Convert text to audio bytes. Return None if TTS is unavailable."""

        raise NotImplementedError

    def supports_tts(self) -> bool:
        return False


class NullSpeechBridge(SpeechBridge):
    """Default placeholder that performs no audio work."""

    def transcribe(self, audio_bytes: bytes, session_id: str, client_id: str) -> Optional[str]:
        return None

    def synthesize(self, text: str, session_id: str, client_id: str) -> Optional[bytes]:
        return None

    def supports_tts(self) -> bool:
        return False


__all__ = ["SpeechBridge", "NullSpeechBridge"]
