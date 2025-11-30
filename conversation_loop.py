"""
Top-level shim for ConversationLoop used by tests and CLI.

This file exposes an importable ConversationLoop due to tests expecting
`from conversation_loop import ConversationLoop` to work.
It imports the ConversationLoop from `experiments.retrieval_sanity.conversation_loop`.
"""
from experiments.retrieval_sanity.conversation_loop import ConversationLoop  # noqa: F401
