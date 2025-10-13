from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_yaml_or_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    text = p.read_text(encoding="utf-8")
    # Try YAML first
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except Exception:
        return json.loads(text)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

