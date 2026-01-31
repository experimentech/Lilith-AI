import pytest

try:
    import pmflow  # noqa: F401
    from pmflow.core.pmflow import MultiScalePMField
except ImportError:  # pragma: no cover - pmflow optional
    pmflow = None
    MultiScalePMField = None  # type: ignore

from lilith.embedding import PMFlowEmbeddingEncoder


@pytest.mark.skipif(pmflow is None, reason="pmflow not installed")
def test_pmflow_encoder_uses_multiscale():
    encoder = PMFlowEmbeddingEncoder(dimension=64, latent_dim=32)
    assert isinstance(encoder.pm_field, MultiScalePMField)
