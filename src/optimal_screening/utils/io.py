from __future__ import annotations

from pathlib import Path
from typing import Any


try:  # Conditional import hoisted to module top
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - surfaced at runtime
    yaml = None  # type: ignore[assignment]


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    text = p.read_text(encoding="utf-8")
    if yaml is None:
        raise RuntimeError(
            "PyYAML is not installed. Install it with 'pip install PyYAML' or install the project.",
        )
    try:
        data = yaml.safe_load(text)
        return data or {}
    except Exception as exc:
        raise RuntimeError("Failed to parse YAML config; ensure valid YAML syntax.") from exc


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
