"""Helpers for reading and resolving MQL-style config files.

The repo stores user-editable configuration in `.mqh`/`.config` files so the
same values can be consumed by Python training code and the MQL5 runtime.
This module keeps that bridge in one place.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Final


Scalar = bool | int | float | str

DEFINE_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\s*#define\s+([A-Z0-9_]+)\s+(.+?)\s*$")
SAFE_SYMBOL_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_.-]+")

# ---------------------------------------------------------------------------
# Known config keys  –  every key a YAML or #define file can legitimately emit
# ---------------------------------------------------------------------------
_CONFIG_KEYS: set[str] = {
    # Project
    "SYMBOL", "DATA_FILE", "MODEL_NAME", "ARCHITECTURE_CONFIG",
    # Architecture
    "MODEL_ARCHITECTURE", "SEQUENCE_DROPOUT", "SEQUENCE_HIDDEN_SIZE",
    "SEQUENCE_LAYERS", "ATTENTION_DIM", "ATTENTION_DROPOUT",
    "ATTENTION_HEADS", "ATTENTION_LAYERS", "USE_MULTIHEAD_ATTENTION",
    "KAN_PROJ_DIM", "L1_LAMBDA", "TCN_LEVELS", "TCN_KERNEL_SIZE",
    # Target
    "DEFAULT_FIXED_MOVE", "LABEL_TIMEOUT_BARS", "LABEL_SL_MULTIPLIER",
    "LABEL_TP_MULTIPLIER", "DEFAULT_SL_MULTIPLIER", "DEFAULT_TP_MULTIPLIER",
    "TARGET_ATR_PERIOD", "USE_FIXED_TARGETS", "DEFAULT_FIXED_SL",
    "DEFAULT_FIXED_TP",
    # Trade sizing
    "DEFAULT_LOT_SIZE", "DEFAULT_LOT_SIZE_CAP", "DEFAULT_RISK_PERCENT",
    "DEFAULT_BROKER_MIN_LOT_SIZE", "USE_LOT_SIZE_CAP", "USE_RISK_PERCENT",
    "USE_BROKER_MIN_LOT_SIZE",
    # Training
    "SEQ_LEN", "DEFAULT_BATCH_SIZE", "DEFAULT_EPOCHS", "DEFAULT_PATIENCE",
    "DEFAULT_MAX_TRAIN_WINDOWS", "DEFAULT_MAX_EVAL_WINDOWS",
    "DEFAULT_LOSS_MODE", "LEARNING_RATE", "WEIGHT_DECAY", "FOCAL_GAMMA",
    "USE_ALL_WINDOWS", "USE_CUSTOM_LEARNING_RATE", "USE_CUSTOM_WEIGHT_DECAY",
    "LOSS_MODE", "CONFIDENCE_SEARCH_MIN", "CONFIDENCE_SEARCH_MAX",
    "CONFIDENCE_SEARCH_STEPS", "MIN_SELECTED_TRADES", "MIN_TRADE_PRECISION",
    "USE_CONFIDENCE_THRESHOLD", "USE_BALANCED_SAMPLING", "LABEL_SMOOTHING",
    # Feature periods
    "FEATURE_ATR_PERIOD", "FEATURE_ATR_RATIO_PERIOD",
    "FEATURE_BOLLINGER_PERIOD", "FEATURE_DONCHIAN_FAST_PERIOD",
    "FEATURE_DONCHIAN_SLOW_PERIOD", "FEATURE_RET_2_PERIOD",
    "FEATURE_RET_3_PERIOD", "FEATURE_RET_6_PERIOD", "FEATURE_RET_12_PERIOD",
    "FEATURE_RET_20_PERIOD", "FEATURE_RSI_FAST_PERIOD", "FEATURE_RSI_SLOW_PERIOD",
    "FEATURE_RV_LONG_PERIOD", "FEATURE_SMA_FAST_PERIOD", "FEATURE_SMA_MID_PERIOD",
    "FEATURE_SMA_SLOW_PERIOD", "FEATURE_SMA_SLOPE_SHIFT",
    "FEATURE_SMA_TREND_FAST_PERIOD", "FEATURE_SPREAD_Z_PERIOD",
    "FEATURE_STOCH_PERIOD", "FEATURE_STOCH_SMOOTH_PERIOD",
    "FEATURE_TICK_COUNT_PERIOD", "FEATURE_TICK_IMBALANCE_FAST_PERIOD",
    "FEATURE_TICK_IMBALANCE_SLOW_PERIOD", "TARGET_ATR_PERIOD", "RV_PERIOD",
    "RETURN_PERIOD", "FEATURE_MAIN_SHORT_PERIOD", "FEATURE_MAIN_MEDIUM_PERIOD",
    "FEATURE_MAIN_LONG_PERIOD", "FEATURE_MAIN_XLONG_PERIOD",
    "FEATURE_MAIN_XXLONG_PERIOD", "FEATURE_MACD_FAST_PERIOD",
    "FEATURE_MACD_SLOW_PERIOD", "FEATURE_MACD_SIGNAL_PERIOD",
    # Feature-set toggles
    "USE_MINIMAL_FEATURE_SET", "USE_MAIN_FEATURE_SET", "USE_GOLD_CONTEXT",
    "MINIMAL_FEATURE_ATR_REL", "MINIMAL_FEATURE_CLOSE_IN_RANGE",
    "MINIMAL_FEATURE_HIGH_REL_PREV", "MINIMAL_FEATURE_LOW_REL_PREV",
    "MINIMAL_FEATURE_RET1", "MINIMAL_FEATURE_RETURN_N",
    "MINIMAL_FEATURE_RV", "MINIMAL_FEATURE_SPREAD_REL",
    "MINIMAL_FEATURE_TICK_IMBALANCE", "MINIMAL_FEATURE_RET_N_PERIOD",
    # Feature toggles (price_action)
    "FEATURE_BODY_REL", "FEATURE_CLOSE_IN_RANGE", "FEATURE_CLOSE_REL_SMA_20",
    "FEATURE_CLOSE_REL_SMA_3", "FEATURE_CLOSE_REL_SMA_9",
    "FEATURE_HIGH_REL_PREV", "FEATURE_LOWER_WICK_REL", "FEATURE_LOW_REL_PREV",
    "FEATURE_OPEN_REL_PREV", "FEATURE_RANGE_REL", "FEATURE_RET1",
    "FEATURE_RET_12", "FEATURE_RET_2", "FEATURE_RET_20", "FEATURE_RET_3",
    "FEATURE_RET_6", "FEATURE_RETURN_N", "FEATURE_RV", "FEATURE_RV_18",
    "FEATURE_TICK_COUNT_CHG", "FEATURE_TICK_COUNT_REL_9",
    "FEATURE_TICK_COUNT_Z_9", "FEATURE_TICK_IMBALANCE",
    "FEATURE_TICK_IMBALANCE_SMA_5", "FEATURE_TICK_IMBALANCE_SMA_9",
    "FEATURE_UPPER_WICK_REL", "FEATURE_USDJPY_RET1", "FEATURE_USDX_RET1",
    # Feature toggles (oscillators)
    "FEATURE_STOCH_D_3", "FEATURE_STOCH_GAP", "FEATURE_STOCH_K_9",
    "FEATURE_RSI_14", "FEATURE_RSI_6",
    # Feature toggles (volatility)
    "FEATURE_ATR_RATIO_20", "FEATURE_ATR_REL",
    "FEATURE_BOLLINGER_POS_20", "FEATURE_BOLLINGER_WIDTH_20",
    "FEATURE_DONCHIAN_POS_20", "FEATURE_DONCHIAN_POS_9",
    "FEATURE_DONCHIAN_WIDTH_20", "FEATURE_DONCHIAN_WIDTH_9",
    "FEATURE_SMA_3_9_GAP", "FEATURE_SMA_5_20_GAP", "FEATURE_SMA_9_20_GAP",
    "FEATURE_SMA_SLOPE_20", "FEATURE_SMA_SLOPE_9", "FEATURE_SPREAD_REL",
    "FEATURE_SPREAD_Z_9",
    # Feature normalization
    "FEATURE_CLOSE_Z_250", "FEATURE_RET_Z_250", "FEATURE_NORMALIZE_PERIOD",
    # Main feature set
    "FEATURE_SPREAD_ABS", "FEATURE_BAR_DURATION_MS",
    "FEATURE_RSI_9", "FEATURE_RSI_18", "FEATURE_RSI_27",
    "FEATURE_ATR_9", "FEATURE_ATR_18", "FEATURE_ATR_27",
    "FEATURE_MACD_LINE", "FEATURE_MACD_SIGNAL", "FEATURE_MACD_HIST",
    "FEATURE_EMA_GAP_9", "FEATURE_EMA_GAP_18", "FEATURE_EMA_GAP_27",
    "FEATURE_EMA_GAP_54", "FEATURE_EMA_GAP_144",
    "FEATURE_CCI_9", "FEATURE_CCI_18", "FEATURE_CCI_27",
    "FEATURE_WILLR_9", "FEATURE_WILLR_18", "FEATURE_WILLR_27",
    "FEATURE_MOM_9", "FEATURE_MOM_18", "FEATURE_MOM_27",
    "FEATURE_USDX_PCT_CHANGE", "FEATURE_USDJPY_PCT_CHANGE",
    "FEATURE_BOLLINGER_WIDTH_9", "FEATURE_BOLLINGER_WIDTH_18",
    "FEATURE_BOLLINGER_WIDTH_27", "FEATURE_HOUR_SIN", "FEATURE_HOUR_COS",
    "FEATURE_MINUTE_SIN", "FEATURE_MINUTE_COS", "FEATURE_DAY_OF_WEEK_SCALED",
    # System
    "DEVICE", "FLIP", "MAX_BARS", "METAEDITOR_PATH", "SKIP_LIVE_COMPILE",
    "USE_MAX_BARS", "USE_NO_HOLD", "USE_CONFIDENCE_THRESHOLD",
    # Bars
    "IMBALANCE_EMA_SPAN", "IMBALANCE_MIN_TICKS",
    "USE_IMBALANCE_EMA_THRESHOLD", "USE_IMBALANCE_MIN_TICKS_DIV3_THRESHOLD",
    "PRIMARY_BAR_SECONDS", "PRIMARY_TICK_DENSITY", "BAR_TYPE",
    "USE_FIXED_TICK_BARS", "USE_SECOND_BARS",
    # Chronos / TimesFM / MiniRocket
    "CHRONOS_BOLT_MODEL", "TIMESFM_MODEL",
    "USE_CHRONOS_AUTO_CONTEXT", "USE_CHRONOS_PATCH_ALIGNED_CONTEXT",
    "USE_CHRONOS_ENSEMBLE_CONTEXTS", "MINIROCKET_FEATURES",
    # Past_dir toggles (bar-based)
    "USE_PAST_DIR_1_T", "USE_PAST_DIR_2_T", "USE_PAST_DIR_3_T",
    "USE_PAST_DIR_5_T", "USE_PAST_DIR_9_T", "USE_PAST_DIR_12_T",
    "USE_PAST_DIR_18_T", "USE_PAST_DIR_27_T", "USE_PAST_DIR_36_T",
    "USE_PAST_DIR_54_T", "USE_PAST_DIR_72_T", "USE_PAST_DIR_100_T",
    "USE_PAST_DIR_144_T", "USE_PAST_DIR_200_T", "USE_PAST_DIR_288_T",
    "USE_PAST_DIR_360_T",
    # Past_dir toggles (second-based)
    "USE_PAST_DIR_60_S", "USE_PAST_DIR_120_S", "USE_PAST_DIR_300_S",
    "USE_PAST_DIR_600_S", "USE_PAST_DIR_900_S", "USE_PAST_DIR_1800_S",
    "USE_PAST_DIR_3600_S", "USE_PAST_DIR_5400_S", "USE_PAST_DIR_7200_S",
    "USE_PAST_DIR_10800_S", "USE_PAST_DIR_14400_S", "USE_PAST_DIR_21600_S",
    "USE_PAST_DIR_43200_S", "USE_PAST_DIR_86400_S",
}

REQUIRED_CONFIG_KEYS: Final[frozenset[str]] = frozenset(_CONFIG_KEYS)

_log = logging.getLogger(__name__)


def validate_config_keys(values: dict[str, Scalar]) -> list[str]:
    """Return a sorted list of required config keys that are missing from *values*."""
    return sorted(REQUIRED_CONFIG_KEYS - values.keys())


def warn_unknown_yaml_keys(
    flat_keys: set[str],
    source_label: str = "",
) -> None:
    """Log a warning for every flattened YAML key not in the known set."""
    unknown = flat_keys - REQUIRED_CONFIG_KEYS
    if unknown:
        label = f" in {source_label}" if source_label else ""
        for key in sorted(unknown):
            _log.warning("Unknown config key%s: %r (typo?)", label, key)


def warn_missing_config_keys(
    values: dict[str, Scalar],
    source_label: str = "",
) -> None:
    """Log a warning for every required config key that is missing."""
    missing = validate_config_keys(values)
    if missing:
        label = f" in {source_label}" if source_label else ""
        for key in missing:
            _log.warning("Missing config key%s: %r", label, key)
        _log.warning("%d required key(s) not provided%s", len(missing), label)
