from __future__ import annotations

from .shared import *  # noqa: F401,F403
from tradebot.workspace_parts.resolve_active_config_path import resolve_active_config_path

def resolve_symbol_config(requested_symbol: str) -> tuple[str, Path]:
    requested = requested_symbol.strip()
    if requested:
        config_path = symbol_default_config_path(requested)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found for symbol '{requested}': {config_path}")
        shared = load_define_file(config_path)
        symbol = str(shared.get("SYMBOL", requested)).strip() or requested
        return symbol, config_path

    active_cfg = resolve_active_config_path()
    active_symbol = configured_symbol(config_path=active_cfg)
    config_path = symbol_default_config_path(active_symbol)
    if config_path.exists():
        shared = load_define_file(config_path)
        symbol = str(shared.get("SYMBOL", active_symbol)).strip() or active_symbol
        return symbol, config_path
    return active_symbol, active_cfg
