from __future__ import annotations

from .shared import *  # noqa: F401,F403


def load_timesfm_barrier_model(
    *,
    device: torch.device,
    model_id: str,
    median: Sequence[float],
    iqr: Sequence[float],
    feature_columns: Sequence[str],
    prediction_length: int,
    use_atr_risk: bool,
    label_tp_multiplier: float,
    label_sl_multiplier: float,
    context_tail_lengths: Sequence[int] | None = None,
) -> TimesFmBarrierClassifier:
    if not use_atr_risk:
        raise ValueError(
            "TimesFM backend currently supports ATR-based label risk only. "
            "Fixed-risk labels require absolute price scale, which is not available in the exported MT5 feature tensor."
        )

    try:
        import timesfm
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TimesFM backend requires the `timesfm` package. "
            "Run `pip install timesfm[torch]` and rerun with MODEL_ARCHITECTURE=timesfm."
        ) from exc

    try:
        tfm_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            model_id,
            force_download=False,
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to load TimesFM checkpoint {model_id}: {exc}") from exc

    compile_kwargs: dict[str, object] = {
        "max_context": 1024,
        "max_horizon": max(256, prediction_length),
        "normalize_inputs": True,
        "use_continuous_quantile_head": True,
        "force_flip_invariance": True,
        "infer_is_positive": False,
        "fix_quantile_crossing": True,
    }

    forecast_config = type("ForecastConfig", (), {})()
    for k, v in compile_kwargs.items():
        setattr(forecast_config, k, v)

    tfm_model.compile(timesfm.ForecastConfig(**compile_kwargs))

    tfm_model = tfm_model.to(device).eval()
    for parameter in tfm_model.parameters():
        parameter.requires_grad_(False)

    return TimesFmBarrierClassifier(
        tfm_model=tfm_model,
        forecast_config=forecast_config,
        quantile_levels=TIMESFM_QUANTILE_LEVELS,
        median=median,
        iqr=iqr,
        feature_columns=tuple(feature_columns),
        prediction_length=prediction_length,
        label_tp_multiplier=label_tp_multiplier,
        label_sl_multiplier=label_sl_multiplier,
        context_tail_lengths=context_tail_lengths,
    )
