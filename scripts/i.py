from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tradebot.training as _impl
from tradebot.workspace import ROOT_DIR
from tradebot.workspace_parts.resolve_active_config_path import set_override_config_path
from tradebot.config_io import load_define_file
from tradebot.config_io_parts.shared import warn_missing_config_keys

log = logging.getLogger("i.py")

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)


def _warn_missing_configs(config_path: Path) -> None:
    if config_path.exists():
        user_values = load_define_file(config_path)
    else:
        user_values = {}
    warn_missing_config_keys(user_values, source_label=str(config_path))


def _override_from_argv() -> Path | None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="config file (relative to config/)")
    args, _ = parser.parse_known_args()
    if args.config:
        config_path = ROOT_DIR / "config" / args.config
        if config_path.exists():
            set_override_config_path(config_path)
            return config_path
        else:
            config_yaml = config_path.with_suffix(".yaml")
            if config_yaml.exists():
                set_override_config_path(config_yaml)
                return config_yaml
            else:
                raise FileNotFoundError(f"Config not found: {config_path}")
    return None


if __name__ == "__main__":
    _config_path = _override_from_argv() or (ROOT_DIR / "config" / "default.yaml")
    _warn_missing_configs(_config_path)
    main()
