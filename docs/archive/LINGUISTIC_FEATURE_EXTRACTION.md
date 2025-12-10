# Linguistic Feature Extraction

This document describes the extraction and normalization of linguistic features used by the pipeline. It outlines recommended feature sets and considerations for merging related concepts.

Feature categories:
- Token-level: normalization, orthographic features, character n-grams.
- Semantic: concept embeddings, topic activations.
- Syntactic: POS sequences and dependency relations.

Merge policy:
- Similar or overlapping concepts should be consolidated when their usage distributions and embedding centroids are highly similar; define thresholds empirically.

Revision history:
- 2025-12-10: Removed conversational phrasing and clarified merge guidance.
