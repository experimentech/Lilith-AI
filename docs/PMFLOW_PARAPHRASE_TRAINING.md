# PMFlow Paraphrase Training

This document describes the paraphrase training objectives, metrics, and expected outcomes for PMFlow models. Example paraphrase pairs have been removed in favour of metric definitions and training protocols.

Training objectives:
- Increase positive similarity for paraphrase pairs toward target thresholds.
- Increase margin between positive and negative similarity.

Metrics:
- paraphrase_match_rate: fraction of paraphrase pairs above threshold.
- semantic_score_avg: average semantic similarity.
- pos_neg_margin: average margin between positive and negative pairs.

Protocol:
- Use held-out validation sets to tune margin and learning rate.

Revision history:
- 2025-12-10: Removed in-line conversational examples and clarified metrics.
