from __future__ import annotations

from experiments.retrieval_sanity import provenance


def test_collect_provenance_without_git(monkeypatch, tmp_path):
    # run inside a temporary directory without a git repo
    monkeypatch.chdir(tmp_path)
    data = provenance.collect_provenance(runner="test", device="cuda:0", extra={"foo": "bar"})

    assert data["runner"] == "test"
    assert data["device"] == "cuda:0"
    assert data["extra"] == {"foo": "bar"}
    assert data["git"] == {"commit": None, "branch": None, "dirty": None}
    assert data["workspace_root"] is None


def test_provenance_json_roundtrip(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    payload = provenance.collect_provenance(runner="roundtrip")
    rendered = provenance.provenance_json(payload)

    assert "\"runner\": \"roundtrip\"" in rendered
    assert rendered.strip().startswith("{")
    assert rendered.strip().endswith("}")


def test_collect_provenance_uses_git_status(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    fake_root = tmp_path / "repo"
    fake_root.mkdir()

    monkeypatch.setattr(provenance, "_find_workspace_root", lambda start: fake_root)
    monkeypatch.setattr(provenance, "_git_status", lambda root: ("deadbeef", "main", True))

    data = provenance.collect_provenance(runner="git-check")

    assert data["git"] == {"commit": "deadbeef", "branch": "main", "dirty": True}
    assert data["workspace_root"] == str(fake_root)
