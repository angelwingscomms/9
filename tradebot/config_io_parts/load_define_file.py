from __future__ import annotations

import logging
from pathlib import Path

from .parse_define_value import parse_define_value
from .shared import DEFINE_PATTERN, REQUIRED_CONFIG_KEYS, Scalar, warn_unknown_yaml_keys

_log = logging.getLogger(__name__)


def load_define_file(path: Path) -> dict[str, Scalar]:
    """Load all `#define` entries from a config file, directory, or YAML file.

    If path is a .yaml file, loads YAML and flattens nested keys.
    If path is a directory, recursively loads all config files.
    If path is a .mqh/.ini file, loads #define entries.
    """
    if path.suffix in (".yaml", ".yml"):
        return _load_yaml_file(path)
    if path.is_dir():
        return _load_config_dir(path)
    return _load_config_file(path)


def _load_yaml_file(path: Path) -> dict[str, Scalar]:
    """Load a YAML file and flatten nested keys to uppercase with underscores."""
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data:
        return {}

    result: dict[str, Scalar] = {}
    _flatten_yaml(data, [], result)
    warn_unknown_yaml_keys(set(result.keys()), source_label=str(path))
    return result


def _flatten_yaml(obj: dict | list | any, path: list[str], result: dict[str, Scalar]) -> None:
    """Recursively flatten nested YAML into flat uppercase keys (deepest key only)."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            _flatten_yaml(value, path + [key], result)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _flatten_yaml(item, path + [str(i)], result)
    else:
        final_key = path[-1].upper() if path else ""
        result[final_key] = obj


def _load_config_dir(dir_path: Path) -> dict[str, Scalar]:
    """Recursively load all config entries from a directory."""
    result: dict[str, Scalar] = {}
    items = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    for item in items:
        if item.name.startswith('.'):
            continue
        if item.is_dir():
            result.update(_load_config_dir(item))
        else:
            result.update(_load_config_file(item))
    return result


def _load_config_file(path: Path) -> dict[str, Scalar]:
    """Load all `#define` entries from a single config file."""
    values: dict[str, Scalar] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = DEFINE_PATTERN.match(line)
        if not match:
            continue
        name, raw_value = match.groups()
        values[name] = parse_define_value(raw_value, values)
    return values
