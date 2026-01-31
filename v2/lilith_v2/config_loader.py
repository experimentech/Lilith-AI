import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """Load a config file (JSON or YAML if PyYAML is installed)."""

    p = Path(path)
    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")

    if suffix in {".json", ""}:  # default to JSON if no suffix
        return json.loads(text)

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ValueError("PyYAML not installed; cannot load YAML configs") from exc
        return yaml.safe_load(text) or {}

    raise ValueError(f"Unsupported config extension: {suffix}")
