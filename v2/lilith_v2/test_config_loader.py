import json
from pathlib import Path

import pytest

from .config_loader import load_config


def test_load_config_json(tmp_path: Path):
    cfg = {"bindings": [{"node_id": "trunk.tools"}]}
    f = tmp_path / "config.json"
    f.write_text(json.dumps(cfg), encoding="utf-8")

    loaded = load_config(str(f))
    assert loaded == cfg


def test_load_config_yaml_if_available(tmp_path: Path):
    try:
        import yaml  # type: ignore
    except ImportError:
        pytest.skip("PyYAML not installed")

    cfg = {"bindings": [{"node_id": "trunk.tools"}]}
    f = tmp_path / "config.yaml"
    f.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    loaded = load_config(str(f))
    assert loaded == cfg


def test_unsupported_extension(tmp_path: Path):
    f = tmp_path / "config.txt"
    f.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(str(f))
