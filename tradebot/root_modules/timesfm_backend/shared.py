from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

DEFAULT_TIMESFM_MODEL_ID = "google/timesfm-2.5-200m-pytorch"
TIMESFM_MODEL_IDS = (
    "google/timesfm-2.5-200m-pytorch",
    "google/timesfm-2.5-200m-flax",
    "google/timesfm-2.5-200m-transformers",
)
TIMESFM_REQUIRED_FEATURES = (
    "ret1",
    "spread_rel",
    "atr_rel",
)
LOGIT_EPS = 1e-6
TIMESFM_QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
