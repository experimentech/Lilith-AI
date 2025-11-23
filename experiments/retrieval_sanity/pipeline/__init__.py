"""Pipeline package - minimal init for conversation demos."""

# Only import what's safe (no storage_bridge or responder)
from .base import PipelineArtifact, SymbolicFrame, Utterance
from .stage_coordinator import (
    StageCoordinator,
    StageType,
    StageConfig,
    StageArtifact,
)
from .conversation_state import ConversationState, ConversationStateSnapshot
from .conversation_history import ConversationHistory
from .response_fragments import ResponseFragmentStore, ResponsePattern
from .response_composer import ResponseComposer, ComposedResponse
from .response_learner import ResponseLearner

__all__ = [
    "PipelineArtifact",
    "SymbolicFrame",
    "Utterance",
    "StageCoordinator",
    "StageType",
    "StageConfig",
    "StageArtifact",
    "ConversationState",
    "ConversationStateSnapshot",
    "ConversationHistory",
    "ResponseFragmentStore",
    "ResponsePattern",
    "ResponseComposer",
    "ComposedResponse",
    "ResponseLearner",
]
