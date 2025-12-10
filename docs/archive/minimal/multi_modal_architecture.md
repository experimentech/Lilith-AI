# Multi-Modal Architecture

This document describes the integration points between vision and language components and the expected data flow for multi-modal inputs.

Usage:
- Visual inputs are converted into a shared representation (vision embeddings) that the composer can incorporate with textual context.
- The composer consumes vision and text embeddings to produce an intent or response representation.

Example call:
- composer.compose("<text>", context=vision_embedding)

Revision history:
- 2025-12-10: Removed informal query examples and standardized API description.
