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
from .responder import ConversationResponder
from .embedding import PMFlowEmbeddingEncoder
from .plasticity import PlasticityController
from .pipeline import SymbolicPipeline
from .storage_bridge import SymbolicStore
from .decoder import TemplateDecoder
from .trace import TraceLogger

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
    "ConversationResponder",
    "PMFlowEmbeddingEncoder",
    "PlasticityController",
    "SymbolicPipeline",
    "SymbolicStore",
    "TemplateDecoder",
    "TraceLogger",
]
