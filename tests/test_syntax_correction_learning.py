import tempfile
from pathlib import Path

import torch

from lilith.syntax_stage_bnn import SyntaxStage


class _DummyEncoder:
    """Minimal encoder stub so SyntaxStage can bootstrap without pmflow."""

    def encode(self, _text: str) -> torch.Tensor:
        return torch.zeros(8)

    def encode_with_components(self, _text: str):
        emb = torch.zeros(8)
        latent = torch.zeros(4)
        activations = torch.zeros(8)
        return emb, latent, activations


def test_learned_correction_applies_in_check_and_correct():
    # Pick a correction not already handled by hardcoded rules.
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Path(tmpdir) / "syntax_patterns.json"
        stage = SyntaxStage(storage_path=storage, encoder=_DummyEncoder())
        stage.learn_correction("we was going home", "we were going home")

        assert stage.check_and_correct("we was going home") == "We were going home"


def test_learned_correction_persists_across_reload():
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Path(tmpdir) / "syntax_patterns.json"

        stage1 = SyntaxStage(storage_path=storage, encoder=_DummyEncoder())
        stage1.learn_correction("we was going home", "we were going home")

        stage2 = SyntaxStage(storage_path=storage, encoder=_DummyEncoder())
        assert stage2.check_and_correct("we was going home") == "We were going home"
