# Hybrid Retrieval Implementation

This document details the hybrid retrieval approach combining keyword matching and embedding similarity.

Implementation considerations:
- Decide whether to retire legacy fuzzy matchers based on empirical performance.
- Log scoring breakdowns for auditing and threshold selection.
- Cache embeddings where appropriate to reduce latency.

Revision history:
- 2025-12-10: Converted question-style headings into declarative implementation considerations.
