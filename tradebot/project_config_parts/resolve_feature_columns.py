from __future__ import annotations

from common.past_dir_features import parse_past_dir_features

from .shared import *  # noqa: F401,F403


def resolve_feature_columns(
    values: dict[str, Scalar], architecture: str
) -> tuple[str, ...]:
    """Resolve the exact ordered feature list used for training and export."""

    if architecture in ("chronos_bolt", "timesfm"):
        return MINIMAL_FEATURE_COLUMNS

    if bool(values.get("USE_MAIN_FEATURE_SET", False)):
        feature_cols = list(MAIN_FEATURE_COLUMNS)
        if not bool(values.get("USE_GOLD_CONTEXT", False)):
            for col in MAIN_GOLD_CONTEXT_FEATURE_COLUMNS:
                if col in feature_cols:
                    feature_cols.remove(col)
        feature_cols.extend(parse_past_dir_features(values))
        return tuple(feature_cols)

    if bool(values.get("USE_MINIMAL_FEATURE_SET", False)):
        selected = list(
            feature_name
            for feature_name in MINIMAL_FEATURE_COLUMNS
            if _minimal_feature_enabled(values, feature_name)
        )
        if not selected:
            raise ValueError(
                "USE_MINIMAL_FEATURE_SET is true but no minimal features are enabled."
            )
        selected.extend(parse_past_dir_features(values))
        return tuple(selected)

    selected = list(MINIMAL_FEATURE_COLUMNS)
    if bool(values.get("USE_GOLD_CONTEXT", False)):
        for feature_name in GOLD_CONTEXT_FEATURE_COLUMNS:
            if _feature_enabled(values, feature_name):
                selected.append(feature_name)
    for feature_name in EXTRA_FEATURE_COLUMNS:
        if _feature_enabled(values, feature_name):
            selected.append(feature_name)
    selected.extend(parse_past_dir_features(values))
    return tuple(selected)
