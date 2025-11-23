"""Helpers for loading and validating experiment override configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

CONFIG_FIELD = "overrides"


def load_overrides(path: Path) -> Dict[str, str]:
    """Load a JSON config file and return the overrides mapping.

    Raises:
        FileNotFoundError: if the file is missing.
        ValueError: if the payload does not contain the required fields.
    """

    resolved = path.expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Override file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    overrides = payload.get(CONFIG_FIELD)
    if not isinstance(overrides, dict):
        raise ValueError(f"Config file {resolved} is missing '{CONFIG_FIELD}' dict")

    return {str(key): str(value) for key, value in overrides.items()}


def load_labeled_overrides(path: Path) -> Tuple[str, Dict[str, str]]:
    """Return the config label and overrides mapping."""

    resolved = path.expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Override file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    overrides = payload.get(CONFIG_FIELD)
    if not isinstance(overrides, dict):
        raise ValueError(f"Config file {resolved} is missing '{CONFIG_FIELD}' dict")

    label = payload.get("label") or resolved.stem
    return label, {str(key): str(value) for key, value in overrides.items()}


__all__ = ["load_overrides", "load_labeled_overrides"]
