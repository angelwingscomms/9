from __future__ import annotations

from .shared import *  # noqa: F401,F403


class TimesFmBarrierClassifier(nn.Module):
    def __init__(
        self,
        tfm_model: nn.Module,
        forecast_config: object,
        quantile_levels: Sequence[float],
        median: Sequence[float],
        iqr: Sequence[float],
        feature_columns: Sequence[str],
        prediction_length: int,
        label_tp_multiplier: float,
        label_sl_multiplier: float,
        context_tail_lengths: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.tfm = tfm_model
        self._forecast_config = forecast_config
        self.prediction_length = int(prediction_length)
        self.label_tp_multiplier = float(label_tp_multiplier)
        self.label_sl_multiplier = float(label_sl_multiplier)

        index_map = _feature_index_map(feature_columns)
        self.ret1_index = index_map["ret1"]
        self.spread_feature_index = index_map["spread_rel"]
        self.atr_feature_index = index_map["atr_rel"]

        self.context_tail_lengths = (0,)
        self.set_context_tail_lengths(context_tail_lengths)

        self.register_buffer("input_median", torch.as_tensor(median, dtype=torch.float32))
        self.register_buffer("input_iqr", torch.as_tensor(iqr, dtype=torch.float32))
        self.register_buffer(
            "quantile_weights",
            _quantile_weights(quantile_levels, device=torch.device("cpu"), dtype=torch.float32),
        )

    def set_context_tail_lengths(self, context_tail_lengths: Sequence[int] | None) -> None:
        self.context_tail_lengths = _normalize_context_tail_lengths(context_tail_lengths)
        tail_label = "full" if self.context_tail_lengths == (0,) else "_".join(
            "full" if tail_length <= 0 else str(tail_length) for tail_length in self.context_tail_lengths
        )
        self.backend_name = f"timesfm-zero-shot-close-barrier-context-{tail_label}"

    def _raw_features(self, x_scaled: torch.Tensor) -> torch.Tensor:
        median = self.input_median.to(device=x_scaled.device, dtype=x_scaled.dtype)
        iqr = self.input_iqr.to(device=x_scaled.device, dtype=x_scaled.dtype)
        return x_scaled * iqr.view(1, 1, -1) + median.view(1, 1, -1)

    def _build_context_series(self, x_raw: torch.Tensor) -> torch.Tensor:
        raw_ret1 = x_raw[:, :, self.ret1_index]
        return torch.cumsum(raw_ret1, dim=1)

    def _crop_context_series(self, context_series: torch.Tensor, tail_length: int) -> torch.Tensor:
        if tail_length <= 0 or tail_length >= context_series.shape[1]:
            return context_series
        return context_series[:, -tail_length:]

    def _predict_price_quantiles(self, context_series: torch.Tensor) -> torch.Tensor:
        device = context_series.device
        dtype = context_series.dtype
        context_np = context_series.detach().cpu().numpy()
        inputs_list = [ctx.astype(np.float64) for ctx in context_np]
        _, quantile_forecast = self.tfm.forecast(
            horizon=self.prediction_length,
            inputs=inputs_list,
        )
        # quantile_forecast shape: (batch, horizon, 10) -> mean + [p10..p90]
        # Take quantiles (indices 1-9), permute to (batch, 9, horizon)
        quantile_preds = quantile_forecast[:, :, 1:]
        return torch.from_numpy(quantile_preds).to(device=device, dtype=dtype).permute(0, 2, 1)

    def _quantile_signal_probs(self, x_raw: torch.Tensor, context_series: torch.Tensor, predicted_levels: torch.Tensor) -> torch.Tensor:
        device = predicted_levels.device
        dtype = predicted_levels.dtype
        batch_size, quantile_count, _horizon = predicted_levels.shape
        scenario_weights = self.quantile_weights.to(device=device, dtype=dtype).view(1, quantile_count)

        current_level = context_series[:, -1].unsqueeze(1).unsqueeze(2)
        future_close_rel = torch.exp(torch.clamp(predicted_levels - current_level, min=-20.0, max=20.0))

        last_bar = x_raw[:, -1, :]
        current_spread_rel = torch.clamp_min(last_bar[:, self.spread_feature_index], 0.0)
        current_atr_rel = torch.clamp_min(last_bar[:, self.atr_feature_index], LOGIT_EPS)

        long_tp = 1.0 + current_spread_rel + self.label_tp_multiplier * current_atr_rel
        long_sl = 1.0 + current_spread_rel - self.label_sl_multiplier * current_atr_rel
        short_tp = 1.0 - self.label_tp_multiplier * current_atr_rel
        short_sl = 1.0 + self.label_sl_multiplier * current_atr_rel
        valid_thresholds = (long_tp > long_sl) & (long_sl > 0.0) & (short_tp > 0.0) & (short_sl > 1.0)

        long_result = torch.zeros(batch_size, quantile_count, device=device, dtype=torch.int64)
        short_result = torch.zeros(batch_size, quantile_count, device=device, dtype=torch.int64)
        one = torch.ones_like(long_result)
        neg_one = -one

        for step in range(self.prediction_length):
            future_close = future_close_rel[:, :, step]

            unresolved_long = long_result == 0
            long_hit_tp = future_close >= long_tp.unsqueeze(1)
            long_hit_sl = future_close <= long_sl.unsqueeze(1)
            long_result = torch.where(unresolved_long & long_hit_tp & ~long_hit_sl, one, long_result)
            long_result = torch.where(unresolved_long & long_hit_sl, neg_one, long_result)

            unresolved_short = short_result == 0
            short_hit_tp = future_close <= short_tp.unsqueeze(1)
            short_hit_sl = future_close >= short_sl.unsqueeze(1)
            short_result = torch.where(unresolved_short & short_hit_tp & ~short_hit_sl, one, short_result)
            short_result = torch.where(unresolved_short & short_hit_sl, neg_one, short_result)

        buy_mask = (long_result == 1) & (short_result != 1)
        sell_mask = (short_result == 1) & (long_result != 1)
        hold_mask = ~(buy_mask | sell_mask)

        hold_prob = (hold_mask.to(dtype) * scenario_weights).sum(dim=1)
        buy_prob = (buy_mask.to(dtype) * scenario_weights).sum(dim=1)
        sell_prob = (sell_mask.to(dtype) * scenario_weights).sum(dim=1)
        probs = torch.stack([hold_prob, buy_prob, sell_prob], dim=1)

        invalid_probs = torch.zeros_like(probs)
        invalid_probs[:, 0] = 1.0
        probs = torch.where(valid_thresholds.unsqueeze(1), probs, invalid_probs)
        probs = torch.clamp_min(probs, LOGIT_EPS)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs

    def forward(self, x_scaled: torch.Tensor) -> torch.Tensor:
        x_raw = self._raw_features(x_scaled)
        full_context_series = self._build_context_series(x_raw)
        member_probs = []

        for tail_length in self.context_tail_lengths:
            context_series = self._crop_context_series(full_context_series, tail_length)
            predicted_levels = self._predict_price_quantiles(context_series)
            probs = self._quantile_signal_probs(x_raw, context_series, predicted_levels)
            member_probs.append(probs)

        probs = torch.stack(member_probs, dim=0).mean(dim=0)
        probs = torch.clamp_min(probs, LOGIT_EPS)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return torch.log(probs)
