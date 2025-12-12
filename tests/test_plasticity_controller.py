"""Unit tests for the PMFlow plasticity controller."""

from __future__ import annotations

import torch
import pytest

from experiments.retrieval_sanity.pipeline.base import Utterance
from experiments.retrieval_sanity.pipeline.plasticity import PlasticityController
from experiments.retrieval_sanity.pipeline.pipeline import SymbolicPipeline
from experiments.retrieval_sanity.pipeline.embedding import PMFlowEmbeddingEncoder
from experiments.retrieval_sanity.pipeline.storage_bridge import SymbolicStore
from experiments.retrieval_sanity.pipeline.decoder import TemplateDecoder
from experiments.retrieval_sanity.pipeline.responder import ConversationResponder
from experiments.retrieval_sanity.pipeline.conversation_state import ConversationState


pytest.importorskip("pmflow.pmflow", reason="pmflow runtime is required")


def test_plasticity_controller_updates_pmfield() -> None:
    pipeline = SymbolicPipeline()
    assert isinstance(pipeline.encoder, PMFlowEmbeddingEncoder)

    artefact = pipeline.process(Utterance(text="visit the hospital today", language="en"))

    controller = PlasticityController(
        pipeline.encoder,
        threshold=0.9,
        mu_lr=1e-2,
        center_lr=1e-2,
    )

    pm_field = pipeline.encoder.pm_field
    target_field = pm_field.fine_field if hasattr(pm_field, "fine_field") else pm_field
    before_centers = target_field.centers.detach().clone()
    before_mus = target_field.mus.detach().clone()

    report = controller.maybe_update(artefact, recall_score=0.1)

    assert report is not None
    assert report.delta_centers > 0.0 or report.delta_mus > 0.0
    assert not torch.allclose(target_field.centers, before_centers)
    assert not torch.allclose(target_field.mus, before_mus)


def test_plasticity_controller_respects_threshold() -> None:
    pytest.importorskip("pmflow.pmflow", reason="pmflow runtime is required")
    pipeline = SymbolicPipeline()
    assert isinstance(pipeline.encoder, PMFlowEmbeddingEncoder)

    artefact = pipeline.process(Utterance(text="visit the hospital today", language="en"))

    controller = PlasticityController(pipeline.encoder, threshold=0.05)

    report = controller.maybe_update(artefact, recall_score=0.5)
    assert report is None

    report_none = controller.maybe_update(artefact, recall_score=None)
    assert report_none is None


def test_plasticity_refreshes_store_vectors(tmp_path) -> None:
    pytest.importorskip("pmflow.pmflow", reason="pmflow runtime is required")

    sqlite_path = tmp_path / "vectors.db"
    scenario = f"unit-{tmp_path.name}"
    store = SymbolicStore(sqlite_path, scenario=scenario)
    store._frames_path = tmp_path / "frames.json"
    pipeline = SymbolicPipeline()
    assert isinstance(pipeline.encoder, PMFlowEmbeddingEncoder)
    decoder = TemplateDecoder()
    controller = PlasticityController(pipeline.encoder, threshold=1.1, mu_lr=1e-2, center_lr=1e-2)
    conversation_state = ConversationState(pipeline.encoder)
    responder = ConversationResponder(
        store,
        decoder,
        topk=1,
        min_score=0.0,
        plasticity_controller=controller,
        conversation_state=conversation_state,
    )

    first = pipeline.process(Utterance(text="we visited the hospital", language="en"))
    store.persist([first])
    vectors_before, labels = store.vector_store.fetch(scenario=scenario)
    assert vectors_before.shape[0] == 1
    first_tokens = [token.text for token in first.parsed.tokens]

    second = pipeline.process(Utterance(text="we might go elsewhere", language="en"))
    response = responder.reply(second)

    assert response.plasticity is not None
    assert response.plasticity.get("refreshed", 0.0) >= 1.0
    assert (
        response.plasticity.get("delta_centers", 0.0) > 0.0
        or response.plasticity.get("delta_mus", 0.0) > 0.0
    )

    vectors_after, labels_after = store.vector_store.fetch(scenario=scenario)
    assert torch.equal(labels, labels_after)

    updated_expected = pipeline.encoder.encode(first_tokens).squeeze(0)
    assert torch.allclose(vectors_after[0], updated_expected, atol=1e-5)


def test_pmflow_state_persists_across_sessions(tmp_path) -> None:
    pytest.importorskip("pmflow.pmflow", reason="pmflow runtime is required")

    state_path = tmp_path / "pmflow_state.pt"
    sqlite_path = tmp_path / "vectors.db"
    frames_path = tmp_path / "frames.json"
    scenario = f"persist-{tmp_path.name}"

    pipeline = SymbolicPipeline(pmflow_state_path=state_path)
    assert isinstance(pipeline.encoder, PMFlowEmbeddingEncoder)
    target_field = pipeline.encoder.pm_field.fine_field if hasattr(pipeline.encoder.pm_field, "fine_field") else pipeline.encoder.pm_field
    before_centers = target_field.centers.detach().clone()

    store = SymbolicStore(sqlite_path, scenario=scenario)
    store._frames_path = frames_path
    decoder = TemplateDecoder()
    controller = PlasticityController(pipeline.encoder, threshold=1.1, mu_lr=1e-2, center_lr=1e-2)
    conversation_state = ConversationState(pipeline.encoder)
    responder = ConversationResponder(
        store,
        decoder,
        topk=1,
        min_score=0.0,
        plasticity_controller=controller,
        conversation_state=conversation_state,
    )

    first = pipeline.process(Utterance(text="we visited the hospital", language="en"))
    store.persist([first])
    second = pipeline.process(Utterance(text="we might go elsewhere", language="en"))
    response = responder.reply(second)

    assert response.plasticity is not None
    assert state_path.exists()

    mutated_centers = target_field.centers.detach().clone()
    assert not torch.allclose(before_centers, mutated_centers)

    pipeline_reloaded = SymbolicPipeline(pmflow_state_path=state_path)
    assert isinstance(pipeline_reloaded.encoder, PMFlowEmbeddingEncoder)
    reloaded_field = pipeline_reloaded.encoder.pm_field
    reloaded_target = reloaded_field.fine_field if hasattr(reloaded_field, "fine_field") else reloaded_field
    reloaded_centers = reloaded_target.centers.detach()

    assert torch.allclose(mutated_centers, reloaded_centers)
