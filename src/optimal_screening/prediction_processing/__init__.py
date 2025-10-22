from __future__ import annotations

from typing import Any


def process_predictions(raw: Any) -> float:
    """Process raw model output into a single probability in [0,1].

    - Accepts a float already in range, or attempts to coerce from strings.
    - Clamps to [0,1] for safety.
    """

    try:
        prob = float(raw)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError("Model returned an unsupported probability format") from exc

    if prob < 0.0:
        prob = 0.0
    if prob > 1.0:
        prob = 1.0
    return prob
