from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.retrieval_sanity.config_utils import load_labeled_overrides, load_overrides


def test_load_overrides_returns_strings(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"label": "demo", "overrides": {"epochs": 200, "metric": "cosine"}}),
        encoding="utf-8",
    )

    overrides = load_overrides(config)

    assert overrides == {"epochs": "200", "metric": "cosine"}


def test_load_labeled_overrides(tmp_path):
    config = tmp_path / "config.json"
    payload = {"label": "demo", "overrides": {"epochs": 200}}
    config.write_text(json.dumps(payload), encoding="utf-8")

    label, overrides = load_labeled_overrides(config)

    assert label == "demo"
    assert overrides == {"epochs": "200"}


def test_load_overrides_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_overrides(tmp_path / "missing.json")


def test_load_overrides_missing_section(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"label": "demo"}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_overrides(config)