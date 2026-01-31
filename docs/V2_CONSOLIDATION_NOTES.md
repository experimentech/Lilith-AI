# Lilith v2 Consolidation Notes

Goal: capture ideas to reduce code complexity by unifying storage/retrieval via `cognitive_stage_base.py` while keeping relational SQL as a sidecar.

## Opportunities
- **Concept store:** Replace bespoke embedding cache/retrieval/usage/decay in `production_concept_store.py` with `CognitiveStageBase`; keep relational joins in a side helper (chains stay SQL).
- **Fragment/pattern store:** Wrap `response_fragments_sqlite.py` with a stage adapter so latent retrieval, success scoring, plasticity, and persistence all come from the base.
- **Shared hygiene/decay:** Use a single sanitizer/decay path (length caps, term mention, success/usage decay) instead of ad hoc per-store logic.
- **Reasoning/composer:** Provide a thin compatibility layer so `ResponseComposer` consumes the unified stage API while relational results remain a separate source.

## Constraints / Caveats
- Schemas differ (concepts/properties/relations vs. patterns); need adapters or a migration to a common layout.
- Relational SQL joins are orthogonal to latent retrieval; keep `relational_concept_store` as a sidecar.
- Migration should be flag-gated to avoid breaking existing DBs and tests.

## Rough Migration Path
1) Define stage adapters for concepts/fragments on top of `CognitiveStageBase` (SQLite-backed) with backward-compatible methods.
2) Keep `relational_concept_store` for joins/chains; route concept retrieval through the stage adapter + relational helper.
3) Swap `ResponseComposer` to consume the adapter interfaces; gate with a feature flag.
4) Migrate existing DBs or add shims for old schema; run hygiene (sanitizer) pre-migration.
5) Add tests covering adapter retrieval, relational joins, and composer integration.

## Expected Payoff
- Remove duplicated retrieval/embedding/decay code (hundreds of LOC) across concept and fragment stores.
- Centralize plasticity/learning and reduce maintenance surface.
