# AWC Pipeline Fix Plan

> Written by Claude Fable 5 (orchestrator). Execute steps **in order**. Every step is
> self-contained: file, exact change, why, and how to verify. Do not improvise beyond
> what is written here. All Python runs use `source env/bin/activate` first.

## 0. Executive summary

The models fail for two separate reasons. **(1) The learning problem is unlearnable:**
labels ask "will gold move $36 within ~8 minutes?" — that almost never happens, so ~99%
of bars are HOLD, and then no-hold mode throws HOLD away and trains a huge TKAN network
on the leftover handful of freak news bars. It memorizes them (train accuracy up, val
loss up = overfitting). **(2) The live EA executes models incorrectly:** it always
applies the label-flip (a `#ifdef` bug), hard-codes 3 output classes for 2-class models,
uses a stop-loss 100× too wide in fixed mode, and deploys models that *failed* the
quality gate (live.mq5 currently points at `0423-070412-next-fail`). This plan fixes the
EA first (so nothing bad can trade), then fixes labeling/regularization/validation so a
small model on ATR-scaled 3-class labels can actually pass an honest walk-forward gate.

## 1. Diagnosis table

Severity: 🔴 CRITICAL (wrong live behavior / unlearnable target), 🟠 HIGH (badly degrades
model quality), 🟡 MEDIUM (silent drift or latent crash), ⚪ LOW (hygiene).

| # | Stage | File:line | Problem | Severity | Root cause |
|---|-------|-----------|---------|----------|------------|
| 1 | Labeling | `config/default.yaml:42-49`, `config/i.yaml:42-48` | `use_fixed_targets: true` + `default_fixed_move: 3600` points (= $36 at point 0.01) as TP/SL on 54-second bars with a 9-bar (~8 min) timeout. Per-bar volatility is ~$1 (see archived `iqrs`: ret1 IQR ≈ 5e-4 × $2400 ≈ $1.2), so a $36 barrier within 9 bars is a >10-sigma event. ~99%+ of labels are HOLD. | 🔴 | Barrier size is fixed in points and wildly out of scale with bar volatility. |
| 2 | Labeling | `tradebot/training/main.py:401-416`, `config/default.yaml:15` | `use_no_hold: true` deletes all HOLD windows from train/val/test. Combined with #1, the model trains on only the rare barrier-hit bars (news spikes), then live it must call BUY/SELL on *every* bar — a total train/live distribution mismatch. | 🔴 | Forcing a binary always-trade target on a mostly-no-signal market. |
| 3 | Loss | `tradebot/training/main.py:916-918` | `class_weights` built only when `len(active_label_names) >= 3` → binary no-hold mode trains with **no class weights** at all. | 🟠 | Condition written for the 3-class case only. |
| 4 | Regularization / architecture | `config/i.yaml:24-39`, `tradebot/models/sequence/tkan_classifier.py` | Default TKAN = BiLSTM(64)×2dir + MHA + 256-latent + KAN(256→512→C) ≈ 1M params fed 9 timesteps × 9 features. `label_smoothing 0.0`. YAML `weight_decay: 0.0` is *ignored* (`use_custom_weight_decay: false` → silent fallback to 1e-4, `parse_args.py:81`), still weak. A memorization machine on a noisy target. | 🟠 | Oversized head + near-zero regularization + confusing custom-flag config semantics. |
| 5 | Validation | `tradebot/training/main.py:352-356`, `1114-1146` | One chronological 70/15/15 split. Confidence threshold is *selected* on validation and the quality gate is *scored* on the same validation. Holdout gate is computed but never enforced. | 🟠 | Selection set == scoring set; no walk-forward. |
| 6 | Validation | `config/default.yaml:75-76` | Gate = ≥15 selected trades at ≥0.52 precision. On 15 trades the 95% CI on precision is ±~0.25 — the gate is noise. | 🟠 | Sample size far too small for the claim being tested. |
| 7 | Deployment | `tradebot/training/main.py:1148-1154`, `1275`; `live/live.mq5:4-6` | On gate failure, the selected confidence is still written (`PRIMARY_CONFIDENCE 0.0` in the `-fail` config.mqh) and `set_live_model_reference()` still repoints `live.mq5` — **live.mq5 currently references `0423-070412-next-fail`**. `OnInit.mqh:55-57` expects a `>1.0` sentinel that the trainer never writes. | 🔴 | Deployment not conditioned on the gate; sentinel convention implemented on only one side. |
| 8 | Live EA | `tradebot/pipeline/mql_config.py:78` + `live/functions/Predict.mqh:30-37` | `FLIP` is always `#define`d (as 0 or 1), and MQL5 `#ifdef` tests *existence*, not value → **the flip branch always compiles → every prediction's direction is inverted**, even with `flip: false`. Same broken pattern for `USE_NO_HOLD` and `USE_CONFIDENCE_THRESHOLD` (`#define X false` still triggers `#ifdef X`). | 🔴 | Boolean encoded as macro *value* but consumed as macro *existence*. |
| 9 | Live EA | `live/functions/Predict.mqh:28,79` vs `live/functions/Execute.mqh:19` | Binary model classes are 0=BUY, 1=SELL (`SignalName.mqh`), but `Execute()` treats `signal==1` as BUY. Today bug #8's always-flip *accidentally cancels* this in binary mode — fix them together or trades invert. Also the confidence gate reads `probs[signal]` **after** flipping, i.e. the losing class's probability. | 🔴 | Two independent index conventions, never reconciled. |
| 10 | Live EA | `live/functions/OnInit.mqh:14-16`, `live/live.mq5:92` | ONNX output shape hard-coded to `[1,3]` and `float output_data[3]`, but no-hold models output 2 logits → `OnnxSetOutputShape`/`OnnxRun` fails and the EA never produces a valid prediction. | 🔴 | Class count not carried from training into the EA. |
| 11 | Live EA | `live/functions/StopDistance.mqh`, `TargetDistance.mqh` | Fixed mode returns `FIXED_SL` (3600 **points**) as a raw **price** distance — $3600 instead of $36. Training converts points→price via `fixed_move_price_distance()`; the EA never multiplies by point size. | 🔴 | Unit mismatch (points vs price). |
| 12 | Live EA | `live/functions/RollingStdReturn.mqh:2` | Buffer `double values[RV_PERIOD]` but callers pass windows up to `FEATURE_RV_LONG_PERIOD` (14 > 5) → array-out-of-range runtime error the moment `rv_18` is enabled. | 🟡 | Fixed-size buffer sized to the wrong constant. |
| 13 | Live EA | `live/functions/ProcessTick.mqh:24-25`, `ResolveAuxBid.mqh` | If USDX/USDJPY is unavailable and no value was ever seen, the aux bid falls back to **gold's own bid** → `usdx_ret1` becomes gold's return. Training drops NaN-aux rows, so the model never saw this regime. | 🟡 | Fabricated fallback instead of refusing to run. |
| 14 | Preprocessing | `tradebot/training/apply_shared_settings.py:117-118` | `WARMUP_BARS = MAX_FEATURE_LOOKBACK` (7), but Wilder ATR (`wilder_atr.py`) is recursive — it needs ~4× its period to converge. Early training bars carry unconverged ATR features/labels that live never produces. | 🟡 | Lookback table assumes finite windows; ATR is exponential. |
| 15 | Preprocessing | `tradebot/pipeline/feature_builder_parts/compute_feature_frame.py:20-52` | Wavelet regime features run `pywt.wavedec` over the **whole series** and replace OHLC → every bar is denoised using *future* bars (classic look-ahead leak). Currently inactive (minimal set), but one config toggle away. | 🟡 | Non-causal transform in a causal pipeline. |
| 16 | Split | `tradebot/training/main.py:350` | `embargo = max(SEQ_LEN, LABEL_TIMEOUT_BARS)`; correct purge needs `SEQ_LEN + LABEL_TIMEOUT_BARS` (window overlap + label horizon). Borderline-safe today only because both are 9. | ⚪ | Off-by-formula. |
| 17 | Config | `config/default.yaml:14` | `use_confidence_threshold: false` → threshold 0.0 → EA trades on *every* bar close (with #2 forcing a direction every time). | 🟠 | Safety gate disabled by default. |
| 18 | Config | `config/i.yaml:3` | `flip: true` — training on inverted labels is sign-mining; if a model only works flipped, it has no edge, you found noise. | 🟠 | Data-mining artifact kept as a feature. |
| 19 | Geometry | `config/i.yaml:44-45` vs `55-56` | Label geometry (TP/SL used to create labels) ≠ execution geometry (`default_sl_multiplier 2.0 / tp 2.5`). The model predicts "TP hits before SL" for barriers the EA doesn't actually use. | 🟡 | Two configs for one concept. |
| 20 | Optimization | `tradebot/training/main.py:712-717` vs `890-892` | LR scheduler exists only for MiniRocket; all sequence models train at constant LR. | ⚪ | Branch-local code. |

**What is already correct (leave alone):** robust scaler fit on train only
(`main.py:360`); all features causal (`compute_feature_frame.py` uses only
`shift`/rolling); tick export uses a proper as-of join for aux symbols
(`mt5/scripts/data_gold.mq5:61-75`); the labeler is spread-aware and scans forward only
(`get_triple_barrier_labels.py`); gradient clipping at 1.0 (`main.py:1007`); MQL feature
math (`ExtractFeatures.mqh`) matches Python formula-for-formula including the
median/IQR/clip scaling (`ScaleAndClip.mqh`), with medians/iqrs baked into `config.mqh`
so no external scaler is needed at inference.

## 2. Decisions (locked in — do not re-litigate)

- **Labeling scheme:** ATR-scaled triple-barrier, **3 classes (HOLD kept)** — `use_fixed_targets: false`, `TP = SL = 1.0 × WilderATR(14)`, `label_timeout_bars: 24`. Barriers must scale with how much the market is actually wiggling; a fixed $36 wall is unreachable and a fixed tiny wall is spread noise. Symmetric 1.0× makes "precision > breakeven" directly meaningful.
- **No-hold:** OFF everywhere. HOLD is a real answer ("no trade here") and the model must be allowed to say it.
- **Flip:** OFF everywhere (config *and* the EA bug). A model that only works inverted is noise.
- **Architecture:** primary `gru` (RecurrentSequenceClassifier: hidden 64, 1 layer, MHA 4 heads × 1 layer ≈ 60k params — has instance-norm, LayerNorm, dropout knobs); fallback `au` (≈85k params). TKAN retired as default: ~1M params on 9 features is a memorization machine.
- **Regularization:** `weight_decay 0.001` (with `use_custom_weight_decay: true` so it actually applies), `label_smoothing 0.05`, `sequence_dropout 0.2`, `attention_dropout 0.1`. Loss: plain cross-entropy + class weights (always built, bug #3 fixed); focal loss stays available but not default; balanced sampling OFF (class weights and a weighted sampler double-correct).
- **Sequence:** `SEQ_LEN 48` (≈ 43 min of 54s bars — enough to see a regime, and MT5 inference cost at 48×9 inputs is microseconds); `batch 64`, `epochs 100`, `patience 10`, LR default 1e-3 + ReduceLROnPlateau(0.5, patience//3) for all architectures.
- **Confidence gate:** ON, search 0.40–0.90 in 26 steps, `min_selected_trades 200`, `min_trade_precision 0.55` (breakeven for symmetric barriers ≈ 0.5 + spread/(TP+SL) ≈ 0.53; 0.55 leaves margin). Holdout gate **enforced**.
- **Validation:** purged expanding walk-forward, 5 folds over the first 85% of bars, embargo = `SEQ_LEN + LABEL_TIMEOUT_BARS` = 72 bars, final 15% is an untouched holdout scored once by `i.py`. Pass = ≥4/5 folds at precision ≥0.55 with ≥100 selected trades each.
- **Deployment:** a model that fails the gate is archived with `-fail`, gets `PRIMARY_CONFIDENCE 2.0` (EA-side kill sentinel), and **never** repoints `live.mq5`.
- **ONNX parity:** torch vs onnxruntime max abs logit diff < 1e-4 on 64 validation windows, enforced at export time.
- **Class count / point size:** carried into `config.mqh` as `MODEL_CLASS_COUNT` and `MODEL_POINT_SIZE` overrides; the EA consumes *values*, never `#ifdef`-existence, for anything boolean.

## 3. Step-by-step implementation

Phase A (steps 1–8) makes the live side safe and correct. Phase B (steps 9–15) fixes
training. Phase C (steps 16–18) adds honest validation. Ship in this order.

> MQL5 note for the implementer: MQL5's preprocessor has `#ifdef`/`#ifndef` but **no
> value-testing `#if`**. `#define X false` still makes `#ifdef X` true. That's why every
> boolean the EA branches on becomes an integer macro tested with a normal runtime
> `if(X != 0)` — the compiler constant-folds it, so there is no runtime cost.

### Step 1: Stop deploying failed models; write the kill sentinel
- File: `tradebot/training/main.py`
- Change: (a) on gate failure set `deployed_primary_confidence = 2.0`; (b) only call `set_live_model_reference` / compile when the gate passed.
- Exact code:
  ```python
  # Replace (currently ~lines 1148-1154):
      deployed_primary_confidence = selected_primary_confidence
      if not quality_gate_passed:
          log.warning(
              "Model failed the live quality gate (%s). Keeping PRIMARY_CONFIDENCE=%.2f.",
              quality_gate_reason,
              deployed_primary_confidence,
          )
  # With:
      deployed_primary_confidence = selected_primary_confidence
      if not quality_gate_passed:
          deployed_primary_confidence = 2.0
          log.warning(
              "Model failed the live quality gate (%s). Writing PRIMARY_CONFIDENCE=2.0 "
              "so the EA can never trade this model.",
              quality_gate_reason,
          )
  ```
  ```python
  # Replace (currently ~line 1275):
      set_live_model_reference(model_dir)

      if not archive_only and not args.skip_live_compile:
  # With:
      if quality_gate_passed:
          set_live_model_reference(model_dir)
      else:
          log.warning("Quality gate failed; live.mq5 was NOT repointed to this model.")

      if quality_gate_passed and not archive_only and not args.skip_live_compile:
  ```
- Why: today a failing model is wired straight into the live robot. This makes "failed" actually mean "cannot trade": the EA only trades when confidence ≥ `PRIMARY_CONFIDENCE`, and no probability can reach 2.0.
- Verify: `source env/bin/activate && python scripts/i.py -c testrun` → if the run fails the gate, output shows the new warning, the archive folder ends in `-fail`, its `config.mqh` contains `PRIMARY_CONFIDENCE 2.00000000`, and `live/live.mq5`'s `@active-model-reference` block is unchanged (`git diff live/live.mq5` is empty).

### Step 2: Enforce the holdout gate
- File: `tradebot/training/main.py`
- Change: add holdout checks to `quality_gate_reasons` before `quality_gate_passed` is computed (holdout_gate already exists at that point).
- Exact code:
  ```python
  # Insert immediately after the existing validation-precision checks
  # (after the `elif validation_precision < args.min_trade_precision:` block,
  #  currently ~line 1145) and BEFORE `quality_gate_passed = ...`:
      holdout_min_trades = max(1, args.min_selected_trades // 2)
      if int(holdout_gate["selected_trades"]) < holdout_min_trades:
          quality_gate_reasons.append(
              "holdout selected trades "
              f"{int(holdout_gate['selected_trades'])} < required {holdout_min_trades}"
          )
      holdout_precision = float(holdout_gate["precision"])
      if not np.isfinite(holdout_precision):
          quality_gate_reasons.append("holdout selected-trade precision unavailable")
      elif holdout_precision < args.min_trade_precision - 0.02:
          quality_gate_reasons.append(
              f"holdout selected-trade precision {holdout_precision:.4f} "
              f"< required {args.min_trade_precision - 0.02:.4f}"
          )
  ```
- Why: the confidence threshold is *chosen* on validation, so validation precision is flattered by construction. The holdout never influenced any choice — it's the only honest number. (Holdout is 15% vs val's 15% but sits at the series end, hence the halved trade count and 2pt slack.)
- Verify: `source env/bin/activate && python scripts/i.py -c testrun` → log line `holdout | ...` followed by gate pass/fail that now mentions holdout reasons when applicable.

### Step 3: Emit MODEL_CLASS_COUNT and MODEL_POINT_SIZE into config.mqh
- Files: `tradebot/pipeline/mql_config.py`, `tradebot/training/main.py`
- Change: add `point_size` parameter to `build_mql_config`; add two overrides to the "Resolved Model Overrides" tuple.
- Exact code:
  ```python
  # mql_config.py — add parameter (after `flip: bool = False,`):
      flip: bool = False,
      point_size: float = 0.0,
  # mql_config.py — in the overrides tuple, after ("FLIP", 1 if flip else 0), add:
          ("MODEL_CLASS_COUNT", 2 if bool(project.values.get("USE_NO_HOLD", False)) else 3),
          ("MODEL_POINT_SIZE", float(point_size)),
  ```
  ```python
  # main.py — in the render_mql_config(...) call (~line 1182), add the kwarg:
              flip=flip,
              point_size=point_size,
  ```
- Why: the EA must know how many outputs the network has (2 or 3) and what one "point" was worth in the training data, instead of hard-coding 3 and guessing units.
- Verify: `source env/bin/activate && python scripts/i.py -c testrun` → new archive's `config.mqh` contains `#define MODEL_CLASS_COUNT 3` and `#define MODEL_POINT_SIZE 0.01` (value from the data).

### Step 4: Rewrite the EA prediction path — value-tested booleans, correct class count, correct signal mapping
- Files: `live/live.mq5`, `live/functions/Predict.mqh`, `live/functions/Softmax.mqh`, `live/functions/SignalName.mqh`, `live/functions/OnInit.mqh`
- Change: canonical signal convention everywhere: **0=HOLD, 1=BUY, 2=SELL**. Binary models map argmax 0→1(BUY), 1→2(SELL). No `#ifdef` on `USE_NO_HOLD`/`FLIP`/`USE_CONFIDENCE_THRESHOLD` anywhere.
- Exact code:
  ```mql5
  // live.mq5 — after the existing `#ifndef MODEL_USE_FIXED_TICK_BARS` guard block, add:
  #ifndef MODEL_CLASS_COUNT
  #define MODEL_CLASS_COUNT 3
  #endif
  #ifndef MODEL_POINT_SIZE
  #define MODEL_POINT_SIZE _Point
  #endif

  // live.mq5 — replace `float output_data[3];` with:
  float output_data[];
  ```
  ```mql5
  // Predict.mqh — full replacement:
  void Predict() {
     prediction_count++;
     for(int i = 0; i < SEQ_LEN; i++) {
        int h = SEQ_LEN - 1 - i;
        int offset = i * MODEL_FEATURE_COUNT;
        float features[MODEL_FEATURE_COUNT];
        ExtractFeatures(h, features);
        for(int k = 0; k < MODEL_FEATURE_COUNT; k++) {
           input_data[offset + k] = features[k];
        }
     }

     if(!OnnxRun(onnx_handle, ONNX_DEFAULT, input_data, output_data)) {
        DebugPrint(StringFormat("OnnxRun failed err=%d", GetLastError()));
        return;
     }

     float probs[];
     ArrayResize(probs, MODEL_CLASS_COUNT);
     Softmax(output_data, probs);
     int best = ArrayMaximum(probs);
     float confidence = probs[best];

     // Canonical signal: 0=HOLD, 1=BUY, 2=SELL.
     // Binary models emit classes 0=BUY, 1=SELL (training maps y-1).
     int signal = (MODEL_CLASS_COUNT == 2) ? (best == 0 ? 1 : 2) : best;
     if(FLIP != 0) {
        if(signal == 1) signal = 2;
        else if(signal == 2) signal = 1;
     }

     DebugPrint(StringFormat("predict best=%d conf=%.4f signal=%s", best, confidence, SignalName(signal)));

     if(signal == 0) {
        hold_skip_count++;
        DebugPrint("skip trade: model chose HOLD");
        return;
     }
     if(confidence < PRIMARY_CONFIDENCE) {
        confidence_skip_count++;
        DebugPrint(StringFormat("skip trade: confidence %.4f below threshold %.4f", confidence, PRIMARY_CONFIDENCE));
        return;
     }
     Execute(signal);
  }
  ```
  ```mql5
  // Softmax.mqh — full replacement:
  void Softmax(const float &logits[], float &probs[]) {
     double max_logit = logits[0];
     for(int i = 1; i < MODEL_CLASS_COUNT; i++) {
        max_logit = MathMax(max_logit, logits[i]);
     }
     double sum = 0.0;
     for(int i = 0; i < MODEL_CLASS_COUNT; i++) {
        probs[i] = (float)MathExp(logits[i] - max_logit);
        sum += probs[i];
     }
     for(int i = 0; i < MODEL_CLASS_COUNT; i++) {
        probs[i] = (float)(probs[i] / sum);
     }
  }
  ```
  ```mql5
  // SignalName.mqh — full replacement (canonical convention only):
  string SignalName(int signal) {
     if(signal == 1) return "BUY";
     if(signal == 2) return "SELL";
     return "HOLD";
  }
  ```
  ```mql5
  // OnInit.mqh — replace the shape block:
     long input_shape[3];
     long output_shape[2];
     input_shape[0] = 1;
     input_shape[1] = SEQ_LEN;
     input_shape[2] = MODEL_FEATURE_COUNT;
     output_shape[0] = 1;
     output_shape[1] = MODEL_CLASS_COUNT;
     ArrayResize(output_data, MODEL_CLASS_COUNT);
  // OnInit.mqh — remove the `#ifdef USE_CONFIDENCE_THRESHOLD` / `#endif` lines around
  // the PRIMARY_CONFIDENCE > 1.0 info print (keep the print itself).
  ```
- Why: `#ifdef` fires on *existence*, so `#define FLIP 0` still enabled flipping — every live signal was inverted (masked only by a second inversion bug in binary mode). The network's output size (2 vs 3) must match what we tell ONNX, or inference fails. One signal convention removes the Predict-vs-Execute disagreement, and the confidence check now reads the *chosen* class's probability before nothing — flips happen after, on the signal only.
- Verify: on the MT5 machine, compile `live.mq5` in MetaEditor (0 errors). In the strategy tester with `DEBUG_LOG=true`, `predict best=... signal=...` lines appear and `OnnxRun failed` does not. `Execute` is unchanged and only ever receives 1 (BUY) or 2 (SELL).

### Step 5: Fix fixed-mode SL/TP unit conversion
- Files: `live/functions/StopDistance.mqh`, `live/functions/TargetDistance.mqh`
- Exact code:
  ```mql5
  // StopDistance.mqh — full replacement:
  double StopDistance() {
     if(R) {
        return FIXED_SL * MODEL_POINT_SIZE;
     }
     return history[0].atr_trade * SL_MULTIPLIER;
  }
  ```
  ```mql5
  // TargetDistance.mqh — full replacement:
  double TargetDistance() {
     if(R) {
        return FIXED_TP * MODEL_POINT_SIZE;
     }
     return history[0].atr_trade * TP_MULTIPLIER;
  }
  ```
- Why: `FIXED_SL` is 3600 *points*; training turns points into price by multiplying by point size ($36). The EA skipped that multiply, producing a $3600 stop — 100× off. `MODEL_POINT_SIZE` is the exact value training used, so labels and execution now agree to the cent.
- Verify: compile; in the tester log, `Intent to place trade: ... sl=... tp=...` shows SL/TP ≈ $36 from entry (with 3600-point config), not $3600.

### Step 6: Fix the RollingStdReturn buffer overflow
- File: `live/functions/RollingStdReturn.mqh`
- Exact code:
  ```mql5
  double RollingStdReturn(int h, int window) {
     double values[];
     ArrayResize(values, window);
     double mean = 0.0;
     for(int i = 0; i < window; i++) {
        values[i] = LogReturnAt(h + i);
        mean += values[i];
     }
     mean /= window;

     double var = 0.0;
     for(int i = 0; i < window; i++) {
        double diff = values[i] - mean;
        var += diff * diff;
     }
     return MathSqrt(var / window);
  }
  ```
- Why: the buffer was sized `RV_PERIOD` (5) but `rv_18` calls it with window 14 — the moment that feature is enabled the EA hard-crashes with array out of range.
- Verify: compile; enable `feature_rv_18` in a test model and confirm no "array out of range" in the tester journal.

### Step 7: Refuse to run a gold-context model without the aux symbols
- File: `live/functions/OnInit.mqh`
- Change: after `usdx_available = SymbolSelect(...)` / `usdjpy_available = SymbolSelect(...)`, add hard checks (the `FEATURE_IDX_*` macros are only defined when the model actually uses the feature, so `#ifdef` is correct here):
- Exact code:
  ```mql5
     #ifdef FEATURE_IDX_USDX_RET1
     if(!usdx_available) {
        Print("[FATAL] Model requires ", USDX_SYMBOL, " but it is not available on this account.");
        return INIT_FAILED;
     }
     #endif
     #ifdef FEATURE_IDX_USDJPY_RET1
     if(!usdjpy_available) {
        Print("[FATAL] Model requires ", USDJPY_SYMBOL, " but it is not available on this account.");
        return INIT_FAILED;
     }
     #endif
  ```
- Why: `ResolveAuxBid` falls back to gold's own bid when an aux symbol never ticks, silently feeding the model a fake dollar-index. Training dropped those rows, so the model has never seen that input; better to refuse to start than to trade on fabricated features.
- Verify: compile; on an account without `$USDX`, a gold-context model logs the FATAL line and the EA does not attach; the minimal-set model (no aux features) is unaffected.

### Step 8: Manually point live.mq5 at a non-fail model (or park it)
- File: `live/live.mq5`
- Change: the `@active-model-reference` block currently references `0423-070412-next-fail`. Until a model passes the new gate, either revert the block to the last known-good passing model, or leave it but do not attach the EA to a live chart. After Phase B produces a passing model, `i.py` repoints it automatically (Step 1 now guarantees only passing models do this).
- Why: a failed model must never be the active live reference.
- Verify: `grep ACTIVE_MODEL_VERSION live/live.mq5` shows a non-`-fail` version.

### Step 9: Always build class weights
- File: `tradebot/training/main.py`
- Exact code:
  ```python
  # Replace (currently ~lines 916-918):
          class_weights = None
          if len(active_label_names) >= 3:
              class_weights = build_class_weights(y_train, class_count=3).to(device)
  # With:
          class_weights = build_class_weights(
              y_train, class_count=len(active_label_names)
          ).to(device)
  ```
- Why: class weights make rare classes count more in the loss so the network can't win by always predicting the common class. The old guard silently gave binary models no weights at all.
- Verify: `source env/bin/activate && python scripts/i.py -c testrun` runs clean; add a temporary `log.info("class_weights=%s", class_weights)` if you want to see the tensor, then remove it.

### Step 10: Correct embargo and warmup
- Files: `tradebot/training/main.py`, `tradebot/training/apply_shared_settings.py`
- Exact code:
  ```python
  # main.py — replace (currently ~line 350):
      embargo = max(SEQ_LEN, LABEL_TIMEOUT_BARS)
  # With:
      embargo = SEQ_LEN + LABEL_TIMEOUT_BARS
  ```
  ```python
  # apply_shared_settings.py — replace (currently ~line 118):
      _sm.WARMUP_BARS = _sm.MAX_FEATURE_LOOKBACK
  # With (Wilder ATR is recursive; ~4x its period ≈ 98% converged):
      _sm.WARMUP_BARS = max(
          _sm.MAX_FEATURE_LOOKBACK,
          4 * max(_sm.FEATURE_ATR_PERIOD, _sm.TARGET_ATR_PERIOD),
      )
  ```
- Why: embargo is the "quiet gap" between train and validation so no training label peeks into validation bars *and* no window overlaps — that needs window length **plus** label horizon. Warmup: the ATR is an average that starts "cold"; the first ~4×period bars have a wrong ATR, so we throw them away like live (which warms up from 3 days of replayed ticks) effectively does.
- Verify: `source env/bin/activate && python scripts/i.py -c testrun` → config dump shows `WARMUP_BARS = 56` (testrun: max ATR period 14) and the run still produces non-empty splits.

### Step 11: LR scheduler for all trained architectures
- File: `tradebot/training/main.py`
- Change: in the non-MiniRocket branch, right after the `optimizer = torch.optim.AdamW(...)` at ~line 890, add:
- Exact code:
  ```python
              scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                  optimizer,
                  factor=0.5,
                  min_lr=1e-6,
                  patience=max(1, args.patience // 3),
              )
  ```
  (The epoch loop already calls `scheduler.step(val_loss)` when `scheduler is not None`.)
- Why: when validation stops improving, halving the learning rate lets the model settle into a better minimum instead of bouncing.
- Verify: `source env/bin/activate && python scripts/i.py -c testrun` runs; no `scheduler` NameError (it's pre-initialized to `None` at line 600).

### Step 12: Hard-disable the non-causal wavelet features
- File: `tradebot/pipeline/feature_builder_parts/compute_feature_frame.py`
- Exact code:
  ```python
  # Replace:
      if needs_wavelet_regime_timing:
          df = _apply_wavelet_regime_timing_features(df)
  # With:
      if needs_wavelet_regime_timing:
          raise ValueError(
              "Wavelet regime/timing features denoise the full series (they read future "
              "bars) and would leak label information. Disabled until a causal "
              "implementation exists."
          )
  ```
- Why: full-series wavelet denoising rewrites each bar using bars that come *after* it. A model trained on that sees the future in training and fails live. Nobody uses these features today; make sure nobody can by accident.
- Verify: `source env/bin/activate && python scripts/i.py -c testrun` (minimal set) still runs; a config requesting a wavelet feature fails fast with the message above.

### Step 13: ONNX ⇔ torch numerical parity gate at export
- File: `tradebot/training/main.py` (plus `pip install onnxruntime` into `env`, and add `onnxruntime>=1.18` to README requirements)
- Change: immediately after `export_onnx_model(export_model, dummy, archive_output_path)` (~line 1179), add:
- Exact code:
  ```python
      try:
          import onnxruntime as ort
      except ModuleNotFoundError as exc:
          raise RuntimeError(
              "onnxruntime is required for the export parity check: "
              "source env/bin/activate && pip install onnxruntime"
          ) from exc
      parity_session = ort.InferenceSession(
          str(archive_output_path), providers=["CPUExecutionProvider"]
      )
      parity_input_name = parity_session.get_inputs()[0].name
      parity_count = min(64, len(x_val))
      parity_max_abs_diff = 0.0
      with torch.no_grad():
          for parity_idx in range(parity_count):
              parity_window = x_val[parity_idx : parity_idx + 1].astype(np.float32)
              torch_logits = export_model(torch.from_numpy(parity_window)).numpy()
              onnx_logits = parity_session.run(None, {parity_input_name: parity_window})[0]
              parity_max_abs_diff = max(
                  parity_max_abs_diff,
                  float(np.max(np.abs(torch_logits - onnx_logits))),
              )
      log.info(
          "ONNX parity | max_abs_logit_diff=%.3g over %d windows",
          parity_max_abs_diff,
          parity_count,
      )
      if parity_max_abs_diff > 1e-4:
          raise RuntimeError(
              f"ONNX export diverges from torch (max abs logit diff "
              f"{parity_max_abs_diff:.3g} > 1e-4); refusing to archive a broken model."
          )
  ```
  (The export uses a fixed batch dimension of 1 — that is exactly how the EA calls it, so per-window comparison is the honest test. Do not add dynamic axes.)
- Why: if the saved ONNX file computes even slightly different numbers than the trained network, live behavior silently diverges from everything we validated. This proves, per training run, that the file MT5 loads *is* the model we measured.
- Verify: `source env/bin/activate && pip install onnxruntime && python scripts/i.py -c testrun` → log shows `ONNX parity | max_abs_logit_diff=...` with a value ≤ 1e-4.

### Step 14: New default training configuration
- Files: `config/default.yaml`, `config/i.yaml`
- Change: apply these exact values (same keys in both files; leave all keys not listed untouched; `i.yaml` keeps its own `bars:` section):
- Exact config:
  ```yaml
  system:
    flip: false
    use_confidence_threshold: true
    use_no_hold: false
    use_balanced_sampling: false
    use_custom_weight_decay: true
    label_smoothing: 0.05

  architecture:
    main:
      model_architecture: gru
    sequence:
      sequence_dropout: 0.2
      sequence_hidden_size: 64
      sequence_layers: 1
    attention:
      attention_dim: 64
      attention_dropout: 0.10
      attention_heads: 4
      attention_layers: 1
      use_multihead_attention: true

  target:
    use_fixed_targets: false
    target_atr_period: 14
    label_timeout_bars: 24
    label_sl_multiplier: 1.0
    label_tp_multiplier: 1.0
    default_sl_multiplier: 1.0    # execution geometry == label geometry
    default_tp_multiplier: 1.0

  training:
    seq_len: 48
    default_batch_size: 64
    default_epochs: 100
    default_patience: 10
    weight_decay: 0.001
    learning_rate: 0.0            # keep use_custom_learning_rate: false → 1e-3 default
    confidence_search_min: 0.40
    confidence_search_max: 0.90
    confidence_search_steps: 26
    min_selected_trades: 200
    min_trade_precision: 0.55
    default_max_train_windows: 50000
    default_max_eval_windows: 10000
  ```
- Why (plain language): barriers now stretch and shrink with the market's own wiggle size (ATR) so BUY/SELL/HOLD are all reachable and roughly balanced; the model sees 43 minutes of context instead of 8; a smaller network with real weight decay and label smoothing can't just memorize; and "good enough" now means 200+ trades at 55%+ precision, which is a real statistical statement instead of a coin-flip on 15 trades.
- Verify: `source env/bin/activate && python scripts/i.py -c default.yaml` (full run on the training machine with `data/gold.csv`) → config dump shows the new values; label distribution in the log/diagnostics shows all three classes present with HOLD ≈ 30–60% (if HOLD > 85%, raise `label_timeout_bars` to 36 before anything else).

### Step 15: Point the testrun smoke config at the ATR path and the new default architecture
- Files: `config/testrun/target`, `config/testrun/architecture/main`
- Exact config:
  ```
  // config/testrun/target — full replacement (single change: USE_FIXED_TARGETS true → false):
  #define DEFAULT_FIXED_MOVE 1080
  #define LABEL_SL_MULTIPLIER 1.0
  #define LABEL_TP_MULTIPLIER 1.0
  #define SEQ_LEN 9
  #define TARGET_ATR_PERIOD 14
  #define LABEL_TIMEOUT_BARS 10
  #define USE_FIXED_TARGETS false
  ```
  ```
  // config/testrun/architecture/main — full replacement (was "tkan"):
  #define MODEL_ARCHITECTURE "gru"
  ```
  (testrun already has `USE_NO_HOLD false`.)
- Why: the smoke test must exercise the same labeling code path and architecture production uses — and `scripts/walkforward.py` (Step 16) supports gru/bilstm/au, so `-c testrun` must resolve to one of those.
- Verify: `source env/bin/activate && python scripts/i.py -c testrun` completes end-to-end (1 epoch, 512 bars) and archives a model folder.

### Step 16: Walk-forward validation script
- File: `scripts/walkforward.py` (new)
- Change: create the script below verbatim.
- Exact code:
  ```python
  """Purged expanding walk-forward validation for the active config.

  Usage: source env/bin/activate && python scripts/walkforward.py -c default.yaml
  Trains one model per fold on everything before the fold, scores precision at the
  fold-selected confidence threshold. The final 15% of bars is never touched here.
  """
  from __future__ import annotations

  import argparse
  import sys
  from pathlib import Path

  _ROOT = Path(__file__).resolve().parent.parent
  if str(_ROOT) not in sys.path:
      sys.path.insert(0, str(_ROOT))

  from tradebot.workspace import ROOT_DIR
  from tradebot.workspace_parts.resolve_active_config_path import set_override_config_path


  def _apply_config_override() -> None:
      parser = argparse.ArgumentParser()
      parser.add_argument("-c", "--config", type=str)
      args, _ = parser.parse_known_args()
      if args.config:
          path = ROOT_DIR / "config" / args.config
          if not path.exists():
              path = path.with_suffix(".yaml")
          if not path.exists():
              raise FileNotFoundError(f"Config not found: {path}")
          set_override_config_path(path)


  _apply_config_override()

  import numpy as np
  import torch
  from torch import nn

  import tradebot.training.shared as S
  from tradebot.training.parse_args import parse_args
  from tradebot.training.resolve_architecture import resolve_architecture
  from tradebot.training.resolve_local_path import resolve_local_path
  from tradebot.models.sequence import (
      AuLSTMMultiheadAttentionClassifier,
      RecurrentSequenceClassifier,
  )
  from tradebot.pipeline.feature_builder import FeatureEngineeringConfig, compute_features
  from tradebot.pipeline.market_data import (
      build_market_bars,
      fixed_move_price_distance,
      get_triple_barrier_labels,
  )
  from tradebot.pipeline.training_utils import (
      choose_confidence_threshold,
      evaluate_model,
      fit_robust_scaler,
      gate_metrics,
      make_class_weights,
      make_loader,
      softmax,
  )
  from tradebot.pipeline.windowing import (
      build_segment_end_indices,
      build_windows,
      maybe_cap_windows,
  )

  N_FOLDS = 5
  HOLDOUT_FRACTION = 0.15
  FOLD_MIN_TRADES = 100
  FOLD_MIN_PRECISION = 0.55
  FOLD_EPOCHS = 40
  FOLD_PATIENCE = 6


  def build_model(architecture: str, feature_count: int, n_classes: int, args) -> nn.Module:
      if architecture == "au":
          return AuLSTMMultiheadAttentionClassifier(
              n_features=feature_count, n_classes=n_classes
          )
      if architecture in ("gru", "bilstm"):
          return RecurrentSequenceClassifier(
              n_features=feature_count,
              cell_type="gru" if architecture == "gru" else "lstm",
              hidden_size=args.sequence_hidden_size,
              hidden=max(args.sequence_hidden_size, feature_count * 4),
              n_classes=n_classes,
              dropout=args.sequence_dropout,
              num_layers=args.sequence_layers,
              bidirectional=architecture == "bilstm",
              use_multihead_attention=True,
              attention_heads=args.attention_heads,
              attention_layers=args.attention_layers,
              attention_dropout=args.attention_dropout,
          )
      raise ValueError(
          f"walkforward.py supports gru/bilstm/au; got {architecture!r}"
      )


  def train_fold(model, x_train, y_train, x_val, y_val, n_classes, args, device):
      class_weights = make_class_weights(y_train, class_count=n_classes).to(device)
      criterion = nn.CrossEntropyLoss(
          weight=class_weights, label_smoothing=args.label_smoothing
      ).to(device)
      optimizer = torch.optim.AdamW(
          model.parameters(),
          lr=1e-3,
          weight_decay=args.weight_decay if args.weight_decay >= 0.0 else 1e-3,
      )
      train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
      val_loader = make_loader(x_val, y_val, max(args.batch_size, 256), shuffle=False)
      best_val_loss, best_state, wait = float("inf"), None, 0
      for _epoch in range(FOLD_EPOCHS):
          model.train()
          for xb, yb in train_loader:
              loss = criterion(model(xb.to(device)), yb.to(device))
              optimizer.zero_grad()
              loss.backward()
              torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
              optimizer.step()
          val_logits, val_labels = evaluate_model(model, val_loader, device)
          val_loss = float(
              criterion(
                  torch.tensor(val_logits, dtype=torch.float32, device=device),
                  torch.tensor(val_labels, dtype=torch.long, device=device),
              ).item()
          )
          if val_loss < best_val_loss:
              best_val_loss = val_loss
              best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
              wait = 0
          else:
              wait += 1
              if wait >= FOLD_PATIENCE:
                  break
      model.load_state_dict(best_state)
      return evaluate_model(model, val_loader, device)


  def main() -> None:
      torch.manual_seed(42)
      np.random.seed(42)
      args = parse_args()
      architecture = resolve_architecture(args)
      device = torch.device(str(args.device).strip() or "cpu")
      if bool(args.no_hold):
          raise ValueError("Walk-forward assumes 3-class labels; set use_no_hold: false.")

      bars, point_size = build_market_bars(
          resolve_local_path(args.data_file),
          bar_type=str(args.bar_type).strip().lower(),
          tick_density=args.primary_tick_density,
          max_bars=int(args.max_bars),
          bar_duration_ms=S.BAR_DURATION_MS,
          imbalance_min_ticks=S.IMBALANCE_MIN_TICKS,
          imbalance_ema_span=S.IMBALANCE_EMA_SPAN,
          use_imbalance_ema_threshold=S.USE_IMBALANCE_EMA_THRESHOLD,
          use_imbalance_min_ticks_div3_threshold=S.USE_IMBALANCE_MIN_TICKS_DIV3_THRESHOLD,
          require_gold_context=bool(args.gold_context)
          and not bool(S.SHARED.get("USE_MINIMAL_FEATURE_SET", False)),
      )
      feature_columns = args.config_project.feature_columns
      x_all = compute_features(
          bars,
          feature_columns=feature_columns,
          config=FeatureEngineeringConfig.from_values(S.SHARED),
      )
      fixed_move_price = fixed_move_price_distance(S.DEFAULT_FIXED_MOVE, point_size)
      y_all = get_triple_barrier_labels(
          bars,
          use_atr_risk=not bool(args.use_fixed_risk),
          fixed_move_price=fixed_move_price,
          label_timeout_bars=S.LABEL_TIMEOUT_BARS,
          target_atr_period=S.TARGET_ATR_PERIOD,
          label_tp_multiplier=S.LABEL_TP_MULTIPLIER,
          label_sl_multiplier=S.LABEL_SL_MULTIPLIER,
      )
      x = x_all[S.WARMUP_BARS:]
      y = y_all[S.WARMUP_BARS:]
      n = len(x)
      holdout_start = int(n * (1.0 - HOLDOUT_FRACTION))
      val_len = holdout_start // 10
      embargo = S.SEQ_LEN + S.LABEL_TIMEOUT_BARS
      counts = np.bincount(y[:holdout_start], minlength=3)
      print(
          f"bars={n} holdout_start={holdout_start} val_len={val_len} embargo={embargo} "
          f"labels HOLD/BUY/SELL={counts.tolist()}"
      )

      passes = 0
      for fold in range(N_FOLDS):
          val_start = holdout_start - (N_FOLDS - fold) * val_len
          val_end = val_start + val_len
          train_end = val_start - embargo
          median, iqr = fit_robust_scaler(x[:train_end])
          x_scaled = np.clip((x - median) / iqr, -10.0, 10.0).astype(np.float32)
          valid = ~np.isnan(x_scaled).any(axis=1)
          train_idx = maybe_cap_windows(
              build_segment_end_indices(valid, 0, train_end, S.SEQ_LEN, S.LABEL_TIMEOUT_BARS),
              args.max_train_windows,
              S.USE_ALL_WINDOWS,
          )
          val_idx = build_segment_end_indices(
              valid, val_start, val_end, S.SEQ_LEN, S.LABEL_TIMEOUT_BARS
          )
          x_train, y_train = build_windows(x_scaled, y, train_idx, S.SEQ_LEN)
          x_val, y_val = build_windows(x_scaled, y, val_idx, S.SEQ_LEN)
          model = build_model(architecture, len(feature_columns), 3, args).to(device)
          val_logits, val_labels = train_fold(
              model, x_train, y_train, x_val, y_val, 3, args, device
          )
          probs = softmax(val_logits)
          threshold = choose_confidence_threshold(
              probs,
              val_labels,
              min_selected=FOLD_MIN_TRADES,
              threshold_min=args.confidence_search_min,
              threshold_max=args.confidence_search_max,
              threshold_steps=args.confidence_search_steps,
          )
          gate = gate_metrics(val_labels, probs, threshold)
          fold_pass = (
              int(gate["selected_trades"]) >= FOLD_MIN_TRADES
              and np.isfinite(float(gate["precision"]))
              and float(gate["precision"]) >= FOLD_MIN_PRECISION
          )
          passes += int(fold_pass)
          print(
              f"fold {fold + 1}/{N_FOLDS} | train={len(x_train)} val={len(x_val)} "
              f"thr={threshold:.2f} trades={int(gate['selected_trades'])} "
              f"precision={float(gate['precision']):.4f} "
              f"coverage={float(gate['trade_coverage']):.4f} "
              f"{'PASS' if fold_pass else 'FAIL'}"
          )

      verdict = "WALK-FORWARD PASS" if passes >= 4 else "WALK-FORWARD FAIL"
      print(f"{verdict}: {passes}/{N_FOLDS} folds passed "
            f"(need >=4 at precision >= {FOLD_MIN_PRECISION} on >= {FOLD_MIN_TRADES} trades)")
      if passes < 4:
          sys.exit(1)


  if __name__ == "__main__":
      main()
  ```
- Why: one lucky split proves nothing. Five separate "train on the past, test on the unseen next slice" exams, with a quiet gap (embargo) so no answers leak, is how you know the edge is real and stable over time. The last 15% of data stays locked away for the final `i.py` run.
- Verify: `source env/bin/activate && python scripts/walkforward.py -c testrun` runs all 5 folds and prints a verdict (on 512 testrun bars the folds are tiny — the point is the plumbing works; the real run uses `-c default.yaml` on the full dataset).

### Step 17: Full retrain + gate
- Files: none (operational step, on the machine that has `data/gold.csv`)
- Change: run, in order:
  1. `source env/bin/activate && pip install onnxruntime`
  2. `python scripts/walkforward.py -c default.yaml` → must print `WALK-FORWARD PASS`.
  3. `python scripts/i.py -c default.yaml` → trains on the same config, gates on validation **and** holdout (Step 2), archives, and — only on pass — repoints `live.mq5`.
  4. If walk-forward fails: try `model_architecture: au` (fallback). If both fail, the edge isn't there at these barriers — adjust `label_timeout_bars` to 36 and rerun *walk-forward only* (never tune on the holdout).
- Why: this is the actual acceptance test for the whole plan.
- Verify: archive folder without `-fail`; diagnostics `report.md` shows all three classes, holdout precision ≥ 0.53, ONNX parity line present.

### Step 18: Compile and dry-run on MT5
- Files: none (operational, Windows/MT5 machine)
- Change: pull the repo, open MetaEditor, compile `live/live.mq5` (0 errors), then run the EA in the **strategy tester** on XAUUSD ticks for ≥1 week of data with `DEBUG_LOG=true` before any live attach.
- Verify in the tester journal: `init seq=48 ... primary_conf=<0.40-0.90 value>`; `predict best=...` lines on every bar close; HOLD skips present (model can decline); trade SL/TP distances ≈ `atr_trade × 1.0`; win rate over ≥100 tester trades within ~5pts of the holdout precision from Step 17 (if it's wildly lower, a feature-parity bug remains — compare one bar's `input_data` values against Python's scaled features for the same timestamp).

## 4. Validation protocol

- **Walk-forward folds (exact):** let `n` = bars after warmup. `holdout_start = int(n*0.85)`, `val_len = holdout_start // 10`, `embargo = SEQ_LEN + LABEL_TIMEOUT_BARS = 72`. Fold *i* (i = 0..4): validation = `[holdout_start − (5−i)·val_len, holdout_start − (4−i)·val_len)`, training = `[0, val_start − embargo)`, scaler refit per fold on that fold's training slice only. With ~60 days of 54s bars (n ≈ 65–70k), each fold validates on ≈5.5k bars and the earliest fold still trains on ≈29k bars.
- **Out-of-sample holdout:** final 15% of bars, touched exactly once, by the final `i.py` run. Nothing — no threshold, no architecture choice, no label tweak — may be chosen using holdout results. If you tune anything after seeing the holdout, that holdout is burnt; collect new data.
- **Quality gate ("good enough to go live"):** all of — walk-forward ≥4/5 folds at precision ≥0.55 on ≥100 selected trades each; final-run validation ≥200 selected trades at precision ≥0.55; holdout ≥100 selected trades at precision ≥0.53; trade coverage between 1% and 25% (below 1% the model is useless, above 25% with 3-class labels it's probably ignoring HOLD); ONNX parity ≤1e-4. Raw accuracy is *not* a gate metric — a model that's right 54% on trades it chooses beats one that's "60% accurate" by always guessing the majority class.
- **ONNX == torch:** enforced automatically at every export (Step 13): 64 validation windows, batch-of-1 (exactly how MT5 calls it), max abs logit difference ≤ 1e-4 or the run aborts.
- **Live parity spot-check (once, after Step 18):** log one bar's `input_data` from the EA (DebugPrint) and compare element-wise to Python's scaled features for the same bar timestamp; every value should match to ~1e-3 (float32 + tick-feed differences).

## 5. Rollout & risk

- **Order:** Phase A first (Steps 1–8) and push — this is pure de-risking: even the *existing* archived models become non-dangerous (failed models can't trade, flip bug gone, correct stops). Then Phase B (9–15), then Phase C (16–18). Do not retrain before Phase A is merged, because Step 1/3 changes what gets written into every new `config.mqh`.
- **Interaction warning:** bugs #8 (always-flip) and #9 (binary signal mapping) currently cancel each other for binary models. Step 4 fixes both *simultaneously* — never ship one without the other.
- **Old archives:** models archived before this plan have `config.mqh` files without `MODEL_CLASS_COUNT`/`MODEL_POINT_SIZE`; the `#ifndef` fallbacks in Step 4/5 (`3` and `_Point`) keep them compiling, but re-activating a pre-plan *binary* model is unsupported — retrain instead.
- **What to watch in live MT5 (first two weeks, demo account or minimum lot):** trades/day vs holdout `trade_coverage × bars/day` (should match within ~2×); rolling precision after every 50 closed trades (alarm below 0.50); `confidence_skip_count` and `hold_skip_count` growing (proves the gates work); journal free of `OnnxRun failed` and "array out of range"; SL/TP distances on tickets ≈ ATR-sized (dollars, not thousands).
- **Kill switch:** remove the EA from the chart (positions keep their broker-side SL/TP). To force-disable without detaching: set the EA input so no trade passes — or redeploy by reverting `live.mq5`'s model block to a known-good version via git (`git log -p live/live.mq5`) and recompiling. Every model the EA ever ran is archived under `symbols/xauusd/models/<version>/` with its exact config, so rollback = repoint + recompile, nothing to rebuild.
- **Revert steps:** each plan step is a small, independent commit; `git revert <sha>` any of them. Python-side regressions are caught by `python scripts/i.py -c testrun`; EA-side by recompiling in MetaEditor.
