# Codebase Analysis: /home/ed/i/awc/9

## Overview

End-to-end ML pipeline for Forex algo trading. Trains neural networks on tick data, exports to ONNX, executes live in MetaTrader 5. ~160 Python source files + ~50 MQL5 include files. 6 config presets, 45+ archived models (mostly XAUUSD, all failed quality gate).

## Architecture

```
scripts/i.py          → entry: train → export ONNX → deploy → compile
scripts/test.py       → entry: backtest archived models in MT5
scripts/export_data.py → entry: export tick data from MT5
config/*.yaml         → YAML config presets (resolved → #define lines)
config/.active_config → pointer file → which preset is active
tradebot/training/    → training loop, ONNX export, diagnostics
tradebot/pipeline/    → bars, features, windowing, labels, MQL config rendering
tradebot/models/      → PyTorch model definitions (TKAN, LSTM, GRU, TCN, etc.)
tradebot/root_modules/→ Mamba, Castor, MiniRocket, Chronos, TimesFM backends
tradebot/workspace/   → file layout, model archiving, live.mq5 reference injection
live/                 → MQL5 EA source + 50 function includes
common/               → shared feature definitions, bar building, config I/O
```

## Flaws Found

### 1. F1 (SafeTy) — Failed model goes live
`tradebot/training/main.py:1275`: `set_live_model_reference(model_dir)` called unconditionally after training, even when `quality_gate_passed=False`. A failed model (below precision/trade-count threshold) silently becomes the live EA.

### 2. F2 (Safety) — Backtest mutates live state
`tradebot/root_modules/test_cli/main.py:16`: `activate_model(model_dir)` rewrites `live/live.mq5` and clobbers `diagnostics/`. A read-only backtest command should never touch live deployment state.

### 3. F3 (Correctness) — Dual config resolution paths
`resolve_active_config_path()` reads `config/.active_config` → `config/active.mqh` fallback. `resolve_symbol_config()` (used by `export_data`) reads `config/active.mqh` directly via `configured_symbol()`, completely ignoring `.active_config`. Training and data-export can silently use different configs.

### 4. F10 (Safety) — MT5 process kill is too broad
`stop_terminal_best_effort.py:9`: kills ALL `terminal64.exe` processes via `taskkill /IM terminal64.exe /F`. On a machine with multiple MT5 instances (live + paper + backtest), this kills unrelated sessions.

### 5. F11 (Reliability) — Compile fallback uses fragile GUI automation
`_compile_via_metaeditor_ui_wine.py`: drives MetaEditor via X11 `xdotool` as last resort. Breaks in headless, SSH, or locked-down sessions. No timeout on the window-search loop.

### 6. F12 (Bug) — keyboard hotkey registered at module import
`tradebot/training/main.py:22`: `keyboard.add_hotkey("ctrl+k", ...)` executes at import time. On headless servers/Wine, this silently fails or raises because no keyboard device exists.

### 7. F13 (Ops) — Shared repo state mutated by transient commands
`export_data.py` overwrites `config/active.mqh`. `test.py` rewrites `live/live.mq5` + `diagnostics/`. Training overwrites `diagnostics/`. Interrupted runs leave repo inconsistent with no rollback.

### 8. F14 (Reliability) — Backtest success is heuristic
`wait_for_tester_completion.py`: success/failure determined by scanning log substrings + file size > 100 bytes. Stale logs, partial writes, or MT5 crashes produce false positives/negatives.

### 9. F18 (ML) — IQR normalisation leaks val data
`tradebot/training/main.py:360`: `median` and `iqr` computed over `x[train_start:train_end]` combined train+val slice. Strictly should be only training portion to prevent lookahead.

### 10. F22 (Config) — bitcoin.config references wrong data path
`config/bitcoin.config` has `#define DATA_FILE "data/bitcoin.csv"`, inconsistent with canonical `data/<SYMBOL>/ticks.csv` convention. Data exporter writes `data/BTCUSD/ticks.csv` so this config silently fails.

### 11. F26 (Deploy) — Windows install path hardcoded
`mt5_runtime/shared.py:15`: `DEFAULT_WINDOWS_INSTALL_DIR = Path("C:\Program Files\MetaTrader 5")`. No auto-discovery; Wine paths must be manually set via CLI/env.

### 12. F27 (Deploy) — No lockfile, loose requirements
`requirements.txt` uses `>=` (not exact versions). No `pyproject.toml` or lockfile. Two installs at different times can produce different environments.

### 13. F28 (Tech debt) — skills-lock.json has no purpose
`meta/skills-lock.json`: no code reads it, no docs reference it. Dead file.

### 14. Config system design flaws
- YAML → `#define` conversion is done once at load time, with no validation that all required keys exist. Missing keys silently produce zeros.
- Two config representations coexist: YAML (`config/*.yaml`) and MQL `#define` lines (`.config`/`.mqh`). The bridge is fragile.
- `config.py` (interactive editor) duplicates the full config schema as Python dataclasses. Schema drift between `config/default.yaml` and `scripts/config.py` is inevitable.

### 15. Module import pattern is dangerous
~10 files use `globals().update({name: getattr(impl, name) for name in dir(impl) if not name.startswith("__")})`. This dumps ALL public names from a submodule into the parent namespace. Hides dependencies, breaks IDE navigation, makes `grep` unreliable.

### 16. Model architecture sprawl
16 architectures: `legacy_lstm_attention`, `gold_legacy`, `gold_new`, `au`, `i9`, `tkan`, `fusion_lstm`, `tcn`, `embtcn`, `tla`, `ela`, `bilstm`, `gru`, `castor`, `mamba_lite`, `shared_mamba`, plus `minirocket`, `chronos_bolt`, `timesfm`. Many share 90% identical training logic with different hardcoded constants. Massive `if/elif` chain in `main.py:738-889`.

### 17. Repetitive boilerplate in apply_shared_settings
`apply_shared_settings.py`: 146 lines of `global` declarations + manual `int()`/`float()`/`bool()` casts for ~50 config keys. One line per key. Extremely fragile — adding a config key requires touching 5+ files.

### 18. MQL5 feature extraction is unmaintainable
`ExtractFeatures.mqh`: 533 lines of `#ifdef FEATURE_IDX_*` blocks, each manually computing one feature with duplicated `ScaleAndClip` calls. No abstraction. Adding a feature requires editing both Python and MQL5.

### 19. MQL5 ONNX inference flaw
`Softmax.mqh`: computes softmax in MQL5 manually. If ONNX model changes output dimension (e.g., binary vs ternary), this silently breaks — the output buffer size is hardcoded.

### 20. No test suite
Zero unit tests. Zero integration tests. The only "testing" is running MT5 backtests, which take minutes each and produce heuristic results.

### 21. Training main() is 1300+ lines
Single monolithic function handling: config loading, data loading, feature engineering, model construction (20+ architectures), training loop, evaluation, confidence threshold search, quality gates, ONNX export, diagnostics, deployment. Impossible to reason about, test, or modify safely.

### 22. No type safety on config values
Config values are `dict[str, bool | int | float | str]`. `apply_shared_settings` casts with bare `int()`/`float()` — a string like `"abc"` where a number is expected silently produces `0` at runtime.

### 23. Bar construction modes have implicit state machine
`OnTick.mqh`: three bar modes (time, tick, imbalance) with mutually exclusive branches. But `bar_started`, `ticks_in_bar`, `tick_imbalance_sum` are global variables — no encapsulation. Tick processing is a monolithic 104-line function.

### 24. Data export overwrites without backup
`move_to_data_dir.py`: always writes `data/<SYMBOL>/ticks.csv` with no versioning or backup. Failed/partial export silently corrupts training dataset (partially addressed by F17 fix adding immutable snapshots).

### 25. No experiment tracking
Model archives store ONNX + config + diagnostics. But no metrics history, no hyperparameter search tracking, no model comparison dashboard. Evaluating which of 45+ models is best requires manual log inspection.

### 26. Config files have no schema validation
YAML configs are validated only by which `#define` keys happen to be present. Typos in key names are silently ignored (e.g., `sequence_droput` instead of `sequence_dropout` in `i.yaml:28`).

### 27. Duplicate config values across presets
`default.yaml`, `ed.yaml`, `i.yaml`, `next.yaml`, `min.yaml` have massive duplication. Changing a feature toggle requires editing 5 files. No inheritance/composition mechanism.

### 28. Hardcoded paths in multiple places
- `live/live.mq5:3-6`: hardcoded model reference injected by script
- `config/default.yaml:20`: `data_file: data/gold.csv` hardcoded
- `mt5_runtime/shared.py:15`: C:\Program Files\MetaTrader 5 hardcoded
- `export_data/main.py:7-8`: prints raw step strings with no i18n or config

### 29. i.yaml typo
`config/i.yaml:28`: `sequence_droput` instead of `sequence_dropout`. This means the dropout value silently defaults to whatever the parser produces for undefined keys.

### 30. No graceful handling of missing data files
`main.py:297`: `build_market_bars_frame(data_path, ...)` — if `data_path` doesn't exist, generic file-not-found error, no suggestion of `export_data.py`.

### 31. Chronos/TimesFM backends are complex but only used for zero-shot
`load_chronos_bolt_barrier_model.py`: wraps a full 2B+ parameter foundation model just to do zero-shot forecasting as a feature extractor. This is massive overhead for what a small learned projection could do. No training happens.

### 32. MiniRocket pipeline is overly complex
`minirocket_classifier/`: 14 files for a random convolution + linear head approach. The attention variant adds a full transformer head on top of 10K+ random features. High memory, dubious value.

### 33. No data pipeline caching
Features are recomputed from scratch on every training run. No parquet/cache of precomputed features. With 50+ feature periods, this adds minutes per run.

### 34. MQL5 history buffer is fixed-size array
`live.mq5:73`: `Bar history[HISTORY_SIZE]` — compile-time constant. If `SEQ_LEN` or `REQUIRED_HISTORY_INDEX` exceeds this, silent memory corruption.

### 35. Magic constants everywhere
`live.mq5`: MAGIC_NUMBER=777777, `live/functions/`: `1e-10` and `1e-8` are scattered as literal values. No named constants.

### 36. Missing ONNX export validation
`export_onnx_model(main.py:1179)`: dummy input is `torch.randn(1, SEQ_LEN, feature_count)`. If the model uses dynamic shapes or the actual inference shapes differ, ONNX runtime silently produces garbage.

### 37. Plot/diagnostics only text-based
Diagnostics output is CSV/markdown reports only. No learning curves, no confusion matrix images, no feature importance plots.

### 38. `warmup_count` in live EA is unused
`live.mq5:89`: `warmup_count` is declared and incremented but never read.

### 39. Redundant `USE_ALL_WINDOWS` logic
`maybe_cap_windows` and `choose_evenly_spaced` — two separate capping mechanisms with overlapping semantics. Confusing.

### 40. No MQL5 build system
Compilation requires MetaEditor UI (Wine xdotool hack) or pre-compiled `.ex5` binaries checked into git. No command-line `metaeditor64.exe /compile` CI pipeline.
