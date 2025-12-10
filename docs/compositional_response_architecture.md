# Compositional Response Architecture

This document describes the design for composing responses from modular components, including canonicalization, paraphrase handling, and policy enforcement.

Design notes:
- Preserve canonical properties that should not be paraphrased during composition.
- Use modular composers for substituting validated slots and paraphrase templates.

Revision history:
- 2025-12-10: Removed informal Q/A and standardized architectural notes.
