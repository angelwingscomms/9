from __future__ import annotations

from .shared import *  # noqa: F401,F403
from .resolve_active_config_path import resolve_active_config_path

def configured_symbol(config_path: Path | None = None) -> str:
    """Read the active symbol from the user-editable root config.

    When config_path is None, resolves through the ``.active_config`` pointer
    file so that training, testing, and data-export all agree on which config
    is active.
    """
    if config_path is None:
        config_path = resolve_active_config_path()
    values = load_define_file(config_path)
    return str(values.get("SYMBOL", "XAUUSD")).strip() or "XAUUSD"
