from __future__ import annotations

from .shared import *  # noqa: F401,F403


def _quantile_weights(quantile_levels: Sequence[float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    quantiles = torch.tensor(list(quantile_levels), device=device, dtype=dtype)
    boundaries = torch.cat(
        [
            torch.tensor([0.0], device=device, dtype=dtype),
            quantiles,
            torch.tensor([1.0], device=device, dtype=dtype),
        ]
    )
    masses = (boundaries[2:] - boundaries[:-2]) / 2
    return masses / masses.sum()
