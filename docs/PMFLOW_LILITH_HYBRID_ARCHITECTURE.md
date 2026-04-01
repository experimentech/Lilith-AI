# PMFlow + Lilith Hybrid Architecture

Date: 2026-04-02
Status: Concept draft for implementation planning

## Purpose

Define a practical hybrid architecture that combines:
- PMFlow's adaptive latent dynamics and language modeling
- Lilith v2's staged cognition, embodiment, and persistent external memory

The goal is a learning, adapting AI that avoids forcing all learning into model weights.

## Core Idea

Use a two-memory approach in one cognitive loop:
- Parametric memory (PMFlow): compressed behavioral priors in field geometry and model parameters
- External memory (Lilith): episodic, semantic, relational, and tenant state in database-backed stores

Rule of thumb:
- New information lands in external memory first
- Only stable, repeated, high-confidence patterns are distilled into PMFlow updates

## System Architecture

1. Input and staging (Lilith)
- Route input through cognitive stages (semantic, pragmatic, planning, affective, response).
- Persist per-stage artifacts and traces.

2. Deliberative dynamics (PMFlow)
- Build a latent state from staged context + retrieved memory.
- Run PMFlow trajectory/flow dynamics to evaluate candidate paths and goals.
- Optionally inject intent for goal-directed bias.

3. Generation (PMFlow LM + Lilith policy)
- Generate candidate responses from PMFlowLanguageModel conditioned on staged context.
- Score and filter candidates with Lilith constraints (pragmatics, safety, task policy).
- Emit final response through somatic/IO layer.

4. Learning and consolidation
- Online: write traces to DB; update graph/indexes immediately.
- Controlled online PMFlow updates only on strong learning signals.
- Periodic consolidation replay from DB traces to tune PMFlow components.

## Cognitive-Cycle Mapping

Map Lilith's three-tier cycle to PMFlow operations:

1. Reflex tier
- Fast pattern and branch routing using existing Lilith mechanisms.
- Minimal or no PMFlow trajectory simulation when latency-sensitive.

2. Deliberative tier
- PMFlow field evolution for intent-aware planning and response selection.
- Use trajectory metrics (path length, efficiency, attractor proximity) as planner features.

3. Homeostatic tier
- Use Lilith neural-health metrics to trigger maintenance.
- Run PMFlow maintenance: entropy control, attractor refresh, selective plasticity, replay.

## Memory Contract

The hybrid depends on explicit memory contracts:

1. Episodic store
- Raw interaction events, tool calls, outcomes, user corrections.

2. Semantic store
- Distilled claims and concept summaries with confidence/provenance.

3. Relational graph
- Entity-relation edges and temporal updates.

4. Procedural store
- Successful plans, action templates, and route patterns.

5. PMFlow state snapshots
- Versioned field/model checkpoints tied to replay cohorts.

## Update Policy

To prevent drift and catastrophic forgetting:

1. Immediate writes
- Always persist interaction evidence externally.

2. PMFlow micro-updates
- Allow only when confidence exceeds threshold and evidence is non-contradictory.

3. Batch distillation
- Periodic replay from curated traces to consolidate into parameters.

4. Rollbackability
- Keep PMFlow checkpoints and memory snapshots aligned by version.

## Minimal Implementation Path (v1)

1. Keep Lilith v2 orchestration as-is
- Reuse existing stage pipeline, stores, and routing.

2. Add a PMFlow deliberation adapter
- Input: staged context packet + retrieved memory bundle.
- Output: candidate latent trajectory metrics + generation conditioning.

3. Integrate PMFlowLanguageModel for response synthesis
- Preserve Lilith policy layer as final arbiter.

4. Add conservative learning gates
- Accept PMFlow updates only from explicit feedback, task success, or repeated confirmations.

5. Add nightly consolidation job
- Replay selected traces, tune PMFlow, checkpoint, and log changes.

## Success Criteria

1. Adaptation speed
- New user facts influence behavior immediately via external memory.

2. Stability
- Lower regression/forgetting after online updates.

3. Interpretability
- Ability to trace response origin across stage artifacts, retrievals, and PMFlow decisions.

4. Personalization quality
- Per-tenant behavior improves without degrading global behavior.

## Open Questions

1. Conditioning interface
- Should staged context enter PMFlow LM via prefix tokens, latent adapters, or both?

2. Distillation thresholding
- Which confidence and contradiction criteria gate PMFlow updates?

3. Retrieval arbitration
- How should conflicts between semantic store and relational graph be resolved?

4. Evaluation protocol
- What benchmark suite best captures adaptation + stability jointly?

## Recommended Next Step

Define a narrow adapter contract and ship an MVP integration:
- `ContextPacket -> PMFlowDeliberation -> CandidateResponses -> LilithPolicySelection`

Then instrument end-to-end traces before expanding online learning depth.
