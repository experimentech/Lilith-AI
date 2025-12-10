# Conversation Context Bugfixes

This document summarizes bugfixes applied to conversation context handling and the rationales for each change. Dialogue transcripts have been removed and replaced with concise descriptions of issues and resolutions.

Issues addressed:
- Disambiguation errors where follow-up questions incorrectly referenced unrelated entities.
- Incorrect template selection for information queries due to keyword collisions.

Resolutions:
- Improved context windowing and disambiguation heuristics.
- Adjusted retrieval ranking to prefer embedding similarity for context-sensitive queries.

Revision history:
- 2025-12-10: Removed raw transcripts and provided a summary of fixes.
