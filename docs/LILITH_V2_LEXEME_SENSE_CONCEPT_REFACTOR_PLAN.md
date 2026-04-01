# Lilith v2 Lexeme-Sense-Concept Refactor Plan

Date: 2026-04-02
Status: Ready for implementation
Owner: Lilith v2 core

## Goal

Enforce strict separation between:
- Lexeme: written word or phrase form
- Sense: context-specific meaning of a lexeme
- Concept: canonical semantic entity used for reasoning

This plan addresses polysemy failures (same word, multiple meanings) while preserving existing graph, PMFlow, and stage architecture.

## Current Risks (Observed)

1. Grounding candidate pool is mixed across node types
- Concept grounding currently pulls all node terms without node_type filtering.
- Effect: lexical tokens, utterance artifacts, syntax patterns, and concepts can all compete during concept activation.

2. Concept IDs are frequently surface-form-derived
- In multiple paths, concept IDs are built from text normalization only (lowercase + underscore).
- Effect: different senses can collapse into one concept node.

3. Lexical artifacts and semantic entities coexist in same retrieval namespace
- Node type exists, but grounding and retrieval paths do not consistently enforce semantic boundaries.

## Target Model

1. Lexeme layer
- Node type: lexeme
- Examples: bank, bat, run
- Stores: language form data (surface, lemma, POS priors, variants)

2. Sense layer
- Node type: sense
- Examples: bank#finance, bank#river_edge
- Stores: disambiguated meaning, contextual signals, confidence

3. Concept layer
- Node type: concept
- Canonical entity for reasoning and graph traversal

4. Required edge types
- lexeme -> sense: possible_meaning, preferred_meaning
- sense -> concept: maps_to
- concept <-> concept: is_a, part_of, opposite_of, related_to, etc.

## Ticket Breakdown

### T1: Grounding Type Filter (Immediate safety patch)

Scope:
- Add type-filtered term retrieval for concept grounding.

Files:
- v2/lilith_v2/relational_graph_store.py
- v2/lilith_v2/concept_grounding.py

Changes:
1. Add new graph API method to fetch terms by allowed node types.
2. Update concept grounder cache builder to call filtered API.
3. Default allowed types for grounding:
- concept
- learned_concept
- entity
- word (optional, low priority)
4. Explicitly exclude:
- lexical_token
- utterance_parsed
- symbolic_frame
- pos_pattern
- syntax_pattern

Acceptance criteria:
- Grounder candidate list contains only allowed semantic node types.
- Existing tests still pass.
- No regression in basic concept retrieval.

---

### T2: Introduce Sense Node Type and Schema Contract

Scope:
- Add first-class sense nodes and edge conventions.

Files:
- v2/lilith_v2/relational_graph_store.py
- v2/lilith_v2/docs/ARCHITECTURE.md (or equivalent docs)

Changes:
1. Define node_type=sense semantics in documentation.
2. Define edge contracts:
- possible_meaning (lexeme -> sense)
- maps_to (sense -> concept)
3. Add helper methods:
- add_sense(lexeme, sense_id, metadata)
- link_lexeme_to_sense(...)
- link_sense_to_concept(...)

Acceptance criteria:
- Sense nodes persist and can be traversed.
- Contract is documented and used by grounding pipeline.

---

### T3: Lexeme Store Normalization Pass

Scope:
- Normalize lexical artifact writes into lexeme nodes.

Files:
- v2/lilith_v2/linguistic_processor.py

Changes:
1. Keep lexical_token if needed for syntactic telemetry.
2. Add or update lexeme nodes keyed by lemma and POS profile.
3. Link utterances to lexeme nodes, not directly to concept nodes.

Acceptance criteria:
- Lexical writes are isolated to lexeme namespace.
- Lexical telemetry remains available for syntax learning.

---

### T4: Sense Resolver Service

Scope:
- New resolver to select candidate senses from context.

Files:
- v2/lilith_v2/sense_resolver.py (new)
- v2/lilith_v2/concept_grounding.py
- v2/lilith_v2/cognitive_stage.py

Changes:
1. Candidate generation:
- from lexeme -> possible senses
- from concept prior neighborhood
2. Candidate scoring features:
- local context overlap
- discourse recency/topic state
- graph neighborhood coherence
- relation compatibility
3. Ambiguity handling:
- if top1-top2 margin below threshold, keep explicit ambiguity

Acceptance criteria:
- Resolver returns sense candidates with confidence and provenance.
- Grounding path supports unresolved ambiguity state.

---

### T5: PMFlow LM Reranker (Optional but recommended)

Scope:
- Use PMFlow LM for contextual reranking of ambiguous senses.

Files:
- v2/lilith_v2/pmflow_lm_adapter.py
- v2/lilith_v2/sense_resolver.py (new)
- v2/lilith_v2/cognitive_stage.py

Changes:
1. Add optional reranker call when ambiguity is high.
2. Build short candidate prompts from local context and candidate gloss/relations.
3. Fuse scores:
- final = graph_score * a + discourse_score * b + lm_score * c
4. Add config flags:
- enable_sense_lm_rerank
- lm_rerank_min_ambiguity

Acceptance criteria:
- With LM reranker enabled, disambiguation improves on ambiguous benchmark set.
- With reranker disabled, baseline behavior unchanged.

---

### T6: Concept ID Stabilization

Scope:
- Stop uncontrolled concept ID generation from raw strings.

Files:
- v2/lilith_v2/cognitive_stage.py
- v2/lilith_v2/linguistic_relation_extractor.py
- v2/lilith_v2/corpus_ingester.py

Changes:
1. Introduce canonical concept ID allocator.
2. Maintain aliases in node data or alias edges.
3. Route new relations through resolver before concept creation.

Acceptance criteria:
- Same lexeme in different contexts can map to different senses without concept ID collisions.
- Canonical concept IDs remain stable across sessions.

---

### T7: Confidence and Pruning Policy by Layer

Scope:
- Separate confidence policies for lexeme, sense, concept.

Files:
- v2/lilith_v2/cognitive_stage.py
- v2/lilith_v2/feedback_system.py
- v2/lilith_v2/relational_graph_store.py

Changes:
1. Lexeme confidence: low-stakes, frequency-informed.
2. Sense confidence: context-sensitive and revisable.
3. Concept confidence: conservative decay, slower pruning.
4. Add anti-overpruning protection for rare but repeatedly verified senses.

Acceptance criteria:
- Conflicting evidence decreases sense confidence without deleting canonical concept prematurely.
- Rare true senses survive longer under sparse observations.

---

### T8: Migration Utility

Scope:
- Backfill existing graph data into lexeme/sense/concept split.

Files:
- v2/scripts/migrate_lexeme_sense_concept.py (new)

Changes:
1. Detect ambiguous terms currently mapped to single concept IDs.
2. Create provisional senses and remap edges.
3. Preserve provenance and rollback log.

Acceptance criteria:
- Migration is idempotent.
- Dry-run mode reports planned remaps.
- Rollback metadata is generated.

---

### T9: Evaluation Harness

Scope:
- Add ambiguity-focused evaluation set and metrics.

Files:
- v2/tests/test_sense_disambiguation.py (new)
- v2/tests/test_grounding_type_filter.py (new)
- v2/data/test/polysemy_cases.json (new)

Metrics:
1. Sense accuracy on ambiguous terms.
2. Wrong-concept activation rate.
3. Ambiguity deferral correctness.
4. Non-ambiguous regression rate.

Acceptance criteria:
- CI includes ambiguity tests.
- Quality gates prevent reintroduction of mixed-type grounding.

---

### T10: Observability and Traceability

Scope:
- Log grounding decisions with layer provenance.

Files:
- v2/lilith_v2/observability.py
- v2/lilith_v2/cognitive_stage.py

Changes:
1. Emit decision trace fields:
- lexeme
- candidate_senses
- selected_sense
- mapped_concept
- score breakdown (graph, discourse, LM)
2. Add counters for fallback and ambiguity deferrals.

Acceptance criteria:
- Each grounding decision can be audited end-to-end.
- Debugging polysemy issues is reproducible from logs.

## Execution Order

1. T1 Grounding Type Filter
2. T2 Sense Schema Contract
3. T4 Sense Resolver Service
4. T6 Concept ID Stabilization
5. T7 Confidence and Pruning Policy
6. T9 Evaluation Harness
7. T5 PMFlow LM Reranker
8. T8 Migration Utility
9. T10 Observability
10. T3 Lexeme Store Normalization pass (can run in parallel after T2)

## Rollout Strategy

1. Feature flags
- enable_sense_layer
- enable_sense_lm_rerank
- strict_grounding_types

2. Safe default mode
- strict grounding enabled
- sense layer enabled
- LM reranker disabled by default

3. Incremental deployment
- shadow mode scoring first
- then active disambiguation for selected tenants

## PMFlow LM Role (Specific)

PMFlow LM is not the source of truth for ontology.
It is a contextual signal provider for hard disambiguation cases.

Use PMFlow LM when:
- graph/discourse scores are near-tied
- lexical context is sparse
- user query is short and ambiguous

Do not use PMFlow LM to:
- create concept IDs directly
- bypass canonical graph relation checks

## Definition of Done

The refactor is complete when:
1. Concept grounding never uses disallowed node types.
2. Polysemous lexemes map through sense nodes before concepts.
3. Ambiguity is explicitly represented when unresolved.
4. PMFlow LM reranker improves ambiguous cases without non-ambiguous regression.
5. Decision traces make each grounding outcome explainable.
