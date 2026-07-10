# Improvement Plan: /home/ed/i/awc/9

## Phase 0 — Safety Gates (must fix before anything else)

| # | What | Why | Effort |
|---|------|-----|--------|
| 0.1 | Gate `set_live_model_reference` behind `quality_gate_passed` in `main.py:1275`. If quality gate fails, log and skip deployment. | F1: failed model silently goes live | 1 file, ~5 lines |
| 0.2 | Remove `activate_model(model_dir)` from `test_cli/main.py:16`. Pass `model_dir` to tester without rewriting `live.mq5`. | F2: backtest mutates live state | 1 file, ~3 lines |
| 0.3 | Move `keyboard.add_hotkey` from module scope into `main()` function. Wrap in try/except. | F12: import-time crash on headless | 1 file, ~5 lines |
| 0.4 | In `stop_terminal_best_effort.py`, filter by `--instance-root` or process PID. Kill only the terminal matching our runtime. | F10: kills all MT5 instances | 1 file, ~10 lines |

## Phase 1 — Correctness Fixes

| # | What | Why | Effort |
|---|------|-----|--------|
| 1.1 | Unify config resolution: `resolve_symbol_config` should delegate to `resolve_active_config_path()` instead of reading `active.mqh` directly. | F3: dual paths diverge silently | 1 file, ~5 lines |
| 1.2 | Fix `i.yaml:28`: change `sequence_droput` → `sequence_dropout`. Add YAML schema validation that warns on unrecognized keys. | Config typo silently defaults | 1 file, +1 validation util |
| 1.3 | IQR normalization: compute `median`/`iqr` on `x[:train_range[0]]` (training-only) instead of `x[:train_range[1]]` (train+val). | F18: val data leaks into scaler | 1 line change |
| 1.4 | Add config key existence validation at load time. Raise clear error if required keys are missing. | Silent zero defaults | +1 validation function |

## Phase 2 — Architecture Consolidation

| # | What | Why | Effort |
|---|------|-----|--------|
| 2.1 | Create `ArchitectureRegistry`: a dict mapping arch name → (model class, default hparams, supported features). Replace 150-line `if/elif` chain in `main.py:738-889`. | Arch sprawl, massive duplication | +1 file, ~50 lines, edits to main.py |
| 2.2 | Create `SharedSettings` dataclass. Replace `apply_shared_settings` (146 lines of globals) with a single `SharedSettings.from_dict(values)` that validates types. | Boilerplate, fragility | +1 file, replace module-level globals |
| 2.3 | Eliminate `globals().update(...)` pattern across all proxy modules. Use explicit imports. | Hides deps, breaks IDE/grep | ~10 files, ~2 lines each |
| 2.4 | Decompose `main()` into focused functions: `load_data()`, `build_features()`, `build_model()`, `train()`, `evaluate()`, `export()`, `deploy()`. | 1300-line monolith | +5 files, medium effort |

## Phase 3 — MQL5/EA Improvements

| # | What | Why | Effort |
|---|------|-----|--------|
| 3.1 | Refactor `ExtractFeatures.mqh` to use a table-driven pattern: array of (feature_index, compute_fn_ptr) instead of 533 lines of `#ifdef` blocks. | Unmaintainable, duplicated logic | 1 file rewrite |
| 3.2 | Add HISTORY_SIZE bounds check on all array accesses. Return default value instead of corrupting memory. | F34: silent memory corruption | ~5 MQL5 files |
| 3.3 | Deduplicate `1e-10`/`1e-8` into `#define EPSILON 1e-10` in a shared constants header. | Magic numbers | +1 header, update ~10 files |
| 3.4 | Remove unused `warmup_count` variable from `live.mq5`. | Dead code | 1 line |
| 3.5 | Make `OnTick.mqh` bar-mode logic into separate functions per mode instead of a single 104-line function with early-return branches. | Implicit state machine | 1 file, ~30 lines |

## Phase 4 — Config System Redesign

| # | What | Why | Effort |
|---|------|-----|--------|
| 4.1 | Implement YAML schema validation using a library (pydantic or a simple JSON Schema). Validate on load, warn on unknown keys. | Silent typos, no structure | +1 schema file, +1 validation function |
| 4.2 | Add config inheritance: `config/default.yaml` as base, overlays extend/override specific sections. Eliminate duplicated presets. | 5 near-identical YAML files | Update config loader |
| 4.3 | Replace `scripts/config.py` dataclass schema with auto-generation from YAML schema. Eliminate drift. | 2 config schemas | +1 script |
| 4.4 | Add config hash/fingerprint to model archive for reproducibility. | Traceability | +5 lines in export |

## Phase 5 — ML Pipeline

| # | What | Why | Effort |
|---|------|-----|--------|
| 5.1 | Cache precomputed features to parquet after first build. Invalidate on config change via hash. | F33: recomputes every run | +1 cache layer |
| 5.2 | Add feature importance analysis (e.g., ablation or permutation importance) to diagnostics output. | Understand what drives predictions | +1 diagnostics module |
| 5.3 | Plot learning curves (train/val loss, accuracy) and confusion matrix as images. Save to diagnostics. | Text-only reports | +1 plotting util |
| 5.4 | Add experiment tracking: log hyperparams, metrics, data fingerprint to a simple SQLite/CSV ledger. | No comparison across 45+ models | +1 ledger module |
| 5.5 | Parallel/early-fail quality gate by evaluating on 3 validation folds instead of 1. | High variance in gate passing | +1 eval module |

## Phase 6 — Testing & CI

| # | What | Why | Effort |
|---|------|-----|--------|
| 6.1 | Unit tests for bar construction (test that 54s time bars produce correct indices from tick CSV). | Zero tests | +test_bar_construction.py |
| 6.2 | Unit tests for feature computation (test numerical parity between Python and MQL5 feature extraction on same data). | Feature drift between train and live | +test_feature_parity.py |
| 6.3 | Unit tests for label generation (triple barrier, SL/TP distances, timeout). | Label logic is correctness-critical | +test_label_generation.py |
| 6.4 | Integration test: train a tiny model (SEQ_LEN=3, 2 epochs, minimal features), export ONNX, validate shapes. | End-to-end pipeline validation | +test_e2e_minimal.py |
| 6.5 | Add `pyproject.toml` with exact version pinning + `pip freeze > requirements.lock`. | F27: environment drift | +2 files |

## Phase 7 — Deployment & Ops

| # | What | Why | Effort |
|---|------|-----|--------|
| 7.1 | Add CLI compile path: `metaeditor64.exe /compile:..."` as first attempt before GUI automation. | F11: GUI automation fragile | 1 file |
| 7.2 | Add rollback mechanism: snapshot `live.mq5`, `diagnostics/`, `config/active.mqh` before mutating. Restore on failure. | F13: no rollback | +1 snapshot util |
| 7.3 | Add data export with versioned timestamped snapshots + symlink `ticks.csv` → latest. | F17 partial-fix: still in progress | 1 file update |
| 7.4 | Discover MT5 install path automatically via registry (Wine) or common install dirs. | F26: hardcoded path | 1 file update |
| 7.5 | Remove or document `meta/skills-lock.json`. | F28: dead file | 1 line |
| 7.6 | Add data-freshness check: compare data file mtime vs model archive timestamp. Warn if data is newer. | Training on stale data | +1 check in training |

## Phase 8 — Long-term Architectural

| # | What | Why | Effort |
|---|------|-----|--------|
| 8.1 | Design a plugin-based model architecture: each model class registers itself with metadata (supported features, default hparams, ONNX export shape). No more hardcoded if/elif. | Sustainable model growth | +1 registry, rewrite model loading |
| 8.2 | Replace `#define`-based MQL5 config with a structured config that the EA parser reads. | Fragile string parsing | Large: changes both Python and MQL5 |
| 8.3 | Add ONNX output shape validation against expected number of classes during export. | F semantic mismatch at inference | +5 lines |
| 8.4 | Implement ensemble inference: run N models and average their softmax outputs. | Higher robustness | +1 ensemble module |
| 8.5 | Evaluate whether Chronos/TimesFM zero-shot adds value over a learned 2-layer MLP on the same features. Drop if not. | F31: massive overhead | Research task |

## Priority Summary

```
IMMEDIATE (Phase 0):    Safety — do before any other change
SHORT-TERM (Phase 1):   Correctness — fix real bugs
MEDIUM (Phase 2-3):     Architecture consolidation + MQL5 cleanup
LONG-TERM (Phase 4-5):  Config system + ML pipeline maturity
ONGOING (Phase 6-8):    Testing, deployment, architectural debt
```

## Quick Wins (< 10 lines change each)

| File | Change |
|------|--------|
| `tradebot/training/main.py:1275` | Gate `set_live_model_reference` + `compile_live_expert` behind `if quality_gate_passed:` |
| `tradebot/root_modules/test_cli/main.py:16` | Remove `activate_model(model_dir)`, pass dir directly |
| `tradebot/training/main.py:22` | Move `keyboard.add_hotkey` inside `main()`, wrap ImportError |
| `config/i.yaml:28` | Fix typo: `sequence_droput` → `sequence_dropout` |
| `tradebot/training/main.py:360` | Change `x[:train_range[1]]` → `x[train_range[0]:train_range[1]]` |
| `tradebot/root_modules/export_data/resolve_symbol_config.py` | Use `resolve_active_config_path()` not raw `active.mqh` |
| `tradebot/root_modules/mt5_runtime/stop_terminal_best_effort.py` | Filter by instance root before killing |
| `tradebot/root_modules/minirocket_search/main.py` | Remove or mark deprecated (duplicate of training) |
