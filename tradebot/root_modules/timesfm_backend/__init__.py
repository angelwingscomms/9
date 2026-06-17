"""TimesFM backend package."""
from __future__ import annotations

from .shared import *  # noqa: F401,F403
from . import _feature_index_map as _module_0
from ._feature_index_map import _feature_index_map
from . import _quantile_weights as _module_1
from ._quantile_weights import _quantile_weights
from . import _normalize_context_tail_lengths as _module_2
from ._normalize_context_tail_lengths import _normalize_context_tail_lengths
from . import TimesFmBarrierClassifier as _module_3
from .TimesFmBarrierClassifier import TimesFmBarrierClassifier
from . import load_timesfm_barrier_model as _module_4
from .load_timesfm_barrier_model import load_timesfm_barrier_model

_MODULES = [
    _module_0,
    _module_1,
    _module_2,
    _module_3,
    _module_4,
]
_PACKAGE_GLOBALS = globals()
for _module in _MODULES:
    _module.__dict__.update(_PACKAGE_GLOBALS)
