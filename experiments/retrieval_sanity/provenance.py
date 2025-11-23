"""Utilities for capturing provenance metadata around retrieval experiments."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for metadata
    torch = None  # type: ignore


@dataclass
class Provenance:
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_dirty: Optional[bool] = None
    workspace_root: Optional[Path] = None
    runner: Optional[str] = None
    python_version: str = field(default_factory=lambda: platform.python_version())
    platform: str = field(default_factory=platform.platform)
    torch_version: Optional[str] = field(default_factory=lambda: getattr(torch, "__version__", None) if torch else None)
    device: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "git": {
                "commit": self.git_commit,
                "branch": self.git_branch,
                "dirty": self.git_dirty,
            },
            "workspace_root": str(self.workspace_root) if self.workspace_root else None,
            "runner": self.runner,
            "python_version": self.python_version,
            "platform": self.platform,
            "torch_version": self.torch_version,
            "device": self.device,
            "extra": self.extra or None,
        }
        return payload


def collect_provenance(*, runner: Optional[str] = None, device: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Collect provenance metadata for the current workspace."""

    root = _find_workspace_root(Path.cwd())
    commit, branch, dirty = _git_status(root) if root else (None, None, None)

    provenance = Provenance(
        git_commit=commit,
        git_branch=branch,
        git_dirty=dirty,
        workspace_root=root,
        runner=runner,
        device=device,
        extra=extra or {},
    )
    return provenance.to_dict()


def _find_workspace_root(start: Path) -> Optional[Path]:
    for path in [start] + list(start.parents):
        if (path / ".git").exists():
            return path
    return None


def _git_status(root: Optional[Path]) -> tuple[Optional[str], Optional[str], Optional[bool]]:
    if root is None:
        return None, None, None

    env = os.environ.copy()
    env["GIT_DIR"] = str(root / ".git")

    def _run(cmd: list[str]) -> Optional[str]:
        try:
            completed = subprocess.run(
                cmd,
                cwd=root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
            return completed.stdout.strip()
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty_output = _run(["git", "status", "--porcelain"])
    dirty = bool(dirty_output) if dirty_output is not None else None
    return commit, branch, dirty


def provenance_json(provenance: Dict[str, Any]) -> str:
    return json.dumps(provenance, indent=2, default=str)


__all__ = ["collect_provenance", "provenance_json", "Provenance"]
