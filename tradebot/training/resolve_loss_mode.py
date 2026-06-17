from __future__ import annotations

from .shared import *  # noqa: F401,F403

def resolve_loss_mode(_architecture: str, requested_mode: str) -> str:
    if _architecture in ("chronos_bolt", "timesfm"):
        return "zero-shot"
    if requested_mode != "auto":
        return requested_mode
    return DEFAULT_LOSS_MODE
