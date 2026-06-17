"""Train, export, archive, and activate the MT5 model from `config.mqh`."""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    from .tqdm_fallback import tqdm

from common.bars import resolve_imbalance_base_threshold
from tradebot.config_io import (
    load_define_file,
    read_text_best_effort,
    render_define_value,
)
from tradebot.models.sequence import (
    AuLSTMMultiheadAttentionClassifier,
    FusionLSTMClassifier,
    GoldLegacyLSTMAttentionClassifier,
    GoldNewTemporalClassifier,
    LegacyLSTMAttentionClassifier,
    RecurrentSequenceClassifier,
    ScalperMicrostructureClassifier,
    TCNClassifier,
    TemporalLSTMAttentionClassifier,
    TKAN,
    EmbTCNClassifier,
)
from tradebot.pipeline.diagnostics import DiagnosticsConfig
from tradebot.pipeline.diagnostics import write_diagnostics as write_diagnostics_report
from tradebot.pipeline.feature_builder import FeatureEngineeringConfig
from tradebot.pipeline.feature_builder import compute_features as build_feature_array
from tradebot.pipeline.market_data import (
    build_market_bars as build_market_bars_frame,
)
from tradebot.pipeline.market_data import (
    fixed_move_price_distance as fixed_move_distance,
)
from tradebot.pipeline.market_data import (
    get_triple_barrier_labels as build_triple_barrier_labels,
)
from tradebot.pipeline.mql_config import build_mql_config as render_mql_config
from tradebot.pipeline.training_utils import (
    FocalLoss as PipelineFocalLoss,
)
from tradebot.pipeline.training_utils import (
    choose_confidence_threshold as select_confidence_threshold,
)
from tradebot.pipeline.training_utils import (
    evaluate_model as run_model_evaluation,
)
from tradebot.pipeline.training_utils import fit_robust_scaler
from tradebot.pipeline.training_utils import (
    gate_metrics as compute_gate_metrics,
)
from tradebot.pipeline.training_utils import (
    make_class_weights as build_class_weights,
)
from tradebot.pipeline.training_utils import (
    make_loader as build_loader,
)
from tradebot.pipeline.training_utils import (
    make_sample_weights as build_sample_weights,
)
from tradebot.pipeline.training_utils import (
    softmax as compute_softmax,
)
from tradebot.pipeline.training_utils import (
    summarize_gate as log_gate_summary,
)
from tradebot.pipeline.windowing import (
    build_segment_end_indices,
    build_windows,
    maybe_cap_windows,
)
from tradebot.project_config import (
    EXTRA_FEATURE_COLUMNS,
    GOLD_CONTEXT_FEATURE_COLUMNS,
    MINIMAL_FEATURE_COLUMNS,
    ResolvedProjectConfig,
    max_feature_lookback,
    resolve_active_project_config,
)
from tradebot.project_config import (
    feature_macro_name as project_feature_macro_name,
)
from tradebot.root_modules.castor_lite import CastorClassifier
from tradebot.root_modules.chronos_backend import (
    CHRONOS_BOLT_MODEL_IDS,
    DEFAULT_CHRONOS_BOLT_MODEL_ID,
    load_chronos_bolt_barrier_model,
)
from tradebot.root_modules.timesfm_backend import (
    DEFAULT_TIMESFM_MODEL_ID,
    TIMESFM_MODEL_IDS,
    load_timesfm_barrier_model,
)
from tradebot.root_modules.mamba_lite import MambaLiteClassifier
from tradebot.root_modules.minirocket_classifier import (
    MiniRocketClassifier,
    MiniRocketMultiAttentionHead,
    fit_minirocket,
    transform_sequence_tokens,
    transform_sequences,
)
from tradebot.root_modules.mt5_runtime import resolve_mt5_runtime
from tradebot.workspace import (
    ACTIVE_CONFIG_PATH,
    ACTIVE_DIAGNOSTICS_DIR,
    compile_live_expert,
    ensure_default_test_config,
    format_model_dir_name,
    model_diagnostics_dir,
    resolve_active_config_path,
    sanitize_model_name,
    set_live_model_reference,
    symbol_models_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nn")

EPS = 1e-10
DEFAULT_DATA_FILE = ""
DEFAULT_OUTPUT_FILE = "model.onnx"
SHARED_CONFIG_PATH = ACTIVE_CONFIG_PATH
DEFAULT_MINIROCKET_FEATURES = 10_080
DEFAULT_FOCAL_GAMMA = 2.0
DEFAULT_MIN_SELECTED_TRADES = 12
DEFAULT_MIN_TRADE_PRECISION = 0.50
DEFAULT_MAMBA_LR = 6e-4
DEFAULT_MINIROCKET_LR = 1e-4
DEFAULT_SEQUENCE_LR = 1e-3
DEFAULT_MAMBA_WEIGHT_DECAY = 1e-4
DEFAULT_MINIROCKET_WEIGHT_DECAY = 0.0
DEFAULT_SEQUENCE_WEIGHT_DECAY = 1e-4
DEFAULT_MINIROCKET_ATTENTION_DIM = 128
DEFAULT_ATTENTION_HEADS = 4
DEFAULT_ATTENTION_LAYERS = 2
DEFAULT_ATTENTION_DROPOUT = 0.1
DEFAULT_SEQUENCE_HIDDEN_SIZE = 64
DEFAULT_SEQUENCE_LAYERS = 2
DEFAULT_SEQUENCE_DROPOUT = 0.1
DEFAULT_TCN_LEVELS = 3
DEFAULT_TCN_KERNEL_SIZE = 3
DEFAULT_L1_LAMBDA = 1e-4
DEFAULT_KAN_PROJ_DIM = 512
DEFAULT_CONFIDENCE_SEARCH_MIN = 0.40
DEFAULT_CONFIDENCE_SEARCH_MAX = 0.99
DEFAULT_CONFIDENCE_SEARCH_STEPS = 60
DEFAULT_PRIMARY_TICK_DENSITY = 27
GOLD_CONTEXT_FEATURE_COLUMNS = (
    "usdx_ret1",
    "usdjpy_ret1",
)
GOLD_CONTEXT_TICK_COLUMNS = (
    "usdx_bid",
    "usdjpy_bid",
)
GOLD_WARNING_SYMBOL = "XAUUSD"
GOLD_PROFILE_SEQ_LEN = 120
GOLD_PROFILE_TICK_DENSITY = 27
GOLD_LEGACY_OLD_TICKS = 2_160_000
GOLD_LEGACY_OLD_TICK_DENSITY = 144
GOLD_LEGACY_OLD_RAW_BARS = 15_000
GOLD_LEGACY_OLD_CANDLES = 72000
GOLD_LEGACY_OLD_WINDOWS = 72000
GOLD_LEGACY_EQUIVALENT_27_TICK_BARS = 79_856

CURRENT_CONFIG_PATH = ACTIVE_CONFIG_PATH
ACTIVE_PROJECT: ResolvedProjectConfig | None = None
SHARED: dict[str, bool | int | float | str] = {}
SYMBOL = "XAUUSD"
SEQ_LEN = 0
LABEL_TIMEOUT_BARS = 0
FEATURE_ATR_PERIOD = 0
FEATURE_ATR_RATIO_PERIOD = 0
FEATURE_BOLLINGER_PERIOD = 0
FEATURE_DONCHIAN_FAST_PERIOD = 0
FEATURE_DONCHIAN_SLOW_PERIOD = 0
FEATURE_RET_2_PERIOD = 0
FEATURE_RET_3_PERIOD = 0
FEATURE_RET_6_PERIOD = 0
FEATURE_RET_12_PERIOD = 0
FEATURE_RET_20_PERIOD = 0
FEATURE_RSI_FAST_PERIOD = 0
FEATURE_RSI_SLOW_PERIOD = 0
FEATURE_RV_LONG_PERIOD = 0
FEATURE_SMA_FAST_PERIOD = 0
FEATURE_SMA_MID_PERIOD = 0
FEATURE_SMA_SLOW_PERIOD = 0
FEATURE_SMA_SLOPE_SHIFT = 0
FEATURE_SMA_TREND_FAST_PERIOD = 0
FEATURE_SPREAD_Z_PERIOD = 0
FEATURE_STOCH_PERIOD = 0
FEATURE_STOCH_SMOOTH_PERIOD = 0
FEATURE_TICK_COUNT_PERIOD = 0
FEATURE_TICK_IMBALANCE_FAST_PERIOD = 0
FEATURE_TICK_IMBALANCE_SLOW_PERIOD = 0
TARGET_ATR_PERIOD = 0
RV_PERIOD = 0
RETURN_PERIOD = 0
MAX_FEATURE_LOOKBACK = 0
WARMUP_BARS = 0
IMBALANCE_MIN_TICKS = 0
IMBALANCE_EMA_SPAN = 0
USE_IMBALANCE_EMA_THRESHOLD = False
USE_IMBALANCE_MIN_TICKS_DIV3_THRESHOLD = False
PRIMARY_BAR_SECONDS = 0
PRIMARY_TICK_DENSITY = 0
BAR_DURATION_MS = 0
DEFAULT_FIXED_MOVE = 0.0
LABEL_FIXED_SL = 0.0  # separate fixed-point SL for labeling (price units); 0.0 = use DEFAULT_FIXED_MOVE
LABEL_FIXED_TP = 0.0  # separate fixed-point TP for labeling (price units); 0.0 = use DEFAULT_FIXED_MOVE
LABEL_SL_MULTIPLIER = 0.0
LABEL_TP_MULTIPLIER = 0.0
EXECUTION_SL_MULTIPLIER = 0.0
EXECUTION_TP_MULTIPLIER = 0.0
USE_ALL_WINDOWS = False
DEFAULT_EPOCHS = 0
DEFAULT_BATCH_SIZE = 0
DEFAULT_MAX_TRAIN_WINDOWS = 0
DEFAULT_MAX_EVAL_WINDOWS = 0
DEFAULT_PATIENCE = 0
DEFAULT_LOSS_MODE = "cross-entropy"
USE_NO_HOLD = False
FLIP = False
LABEL_NAMES = ("HOLD", "BUY", "SELL")
LABEL_NAMES_BINARY = ("BUY", "SELL")
