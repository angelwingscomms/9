# Fable 5 — Orchestrator Prompt: Fix the AWC Trading Pipeline

> Run this with `claude --model claude-fable-5` from the repo root (`~/i/awc/9`).
> Fable 5 reads the filesystem itself. Do NOT paste files manually.

---

## WHY THIS MATTERS (the reason, not just the request)

This is a live FOREX trading system. It scrapes/exports market data (XAUUSD, BTCUSD),
builds features, trains neural networks (TKAN/LSTM/Mamba/Chronos/MiniRocket), exports
them to ONNX, and executes them inside MetaTrader 5 via an MQL5 EA. **Real capital is at
risk when a model goes live.** Today the system is broken: a large fraction of trained
models are archived with a `-fail` suffix, validation loss is high while validation
accuracy still climbs (the classic signature of overfitting), and the pipeline has
structural flaws across data collection, preprocessing, labeling, regularization,
validation methodology, and export.

The user is not an ML expert. They need two things from you:
1. A correct **diagnosis** of every flaw, explained simply (like teaching a smart 9-year-old).
2. An **execution-ready plan** so a smaller, cheaper model (Sonnet 5, or any code agent)
   can implement the fix *exactly the way you would have* — capturing your architectural
   judgment, not just your conclusions.

You are the **orchestrator / principal engineer**. You do NOT write the final code into
the repo. You write one plan file that a downstream model executes. Your intelligence must
live in that file: every decision, every tradeoff, every exact value, and the reasoning
behind each — so the implementer never has to guess.

---

## YOUR ROLE & WORKFLOW

1. **Explore first.** Read the whole repo: `tradebot/pipeline/*`, `tradebot/training/*`,
   `common/*`, `config/*`, `scripts/*`, `live/*`, `symbols/xauusd/models/*/config.mqh`
   (especially the `-fail` ones), `README.md`, `AGENTS.md`. Understand the current
   data flow end-to-end before forming opinions.
2. **Diagnose.** Identify every place the pipeline can overfit, leak, or produce a model
   that fails live. Be specific — cite file and line.
3. **Decide.** For each flaw, choose the concrete fix and the exact parameter/value. Do not
   leave choices open; pick the professional-standard default and justify it.
4. **Write the plan.** Output a single markdown file (see format below) that is so detailed
   a model with no domain knowledge can implement it step by step and get Fable-5-quality
   results.

Do not edit source files. Do not train models. Your only output artifact is the plan file.

---

## WHAT TO INVESTIGATE (the full surface area)

Work through every stage. For each, state whether it is correct, flawed, or missing — and
if flawed/missing, the exact fix.

1. **Data collection / scraping** (`mt5/`, `scripts/export_data.py`)
   - Tick-export integrity, point-size inference, gaps, duplicate/zero-volume bars.
   - Whether cross-asset context (USDX, USDJPY) is aligned in time or leaks future bars.

2. **Preprocessing & leakage** (`tradebot/pipeline/`, `common/`)
   - Robust scaler is fit on train only (good) — verify nothing else fits on full data.
   - Feature lookback (`MAX_FEATURE_LOOKBACK`, `WARMUP_BARS`) vs. `SEQ_LEN` — confirm no
     feature at window `t` uses information from `t+1` or from the label's future.
   - Bar construction (time/tick/imbalance) and whether it can peek ahead.

3. **Labeling / targets** (`build_triple_barrier_labels`, `common/features.py`)
   - Triple-barrier vs fixed-move (`DEFAULT_FIXED_MOVE`, `LABEL_TIMEOUT_BARS`): is the
     target噪声-dominated? Is `USE_NO_HOLD` (forcing BUY/SELL, discarding HOLD) the cause of
     the model memorizing spurious patterns to always trade? Recommend the correct labeling
     scheme and whether HOLD should be a real class.
   - Class balance: note that `main.py` only builds `class_weights` when
     `len(active_label_names) >= 3` (line ~917), so **binary no-hold mode gets NO class
     weights** — this is a real bug to fix.

4. **Train / val / test split & validation methodology**
   - Current split is chronological 70/15/15 with an embargo (good). Verify embargo length
     covers `SEQ_LEN` + `LABEL_TIMEOUT_BARS` + max feature lookback.
   - Require **walk-forward / purged k-fold cross-validation** as the primary validation,
     with out-of-sample holdout. Specify the exact folds and embargo for this dataset size.
   - Define the live quality gate metrics that actually matter (precision@coverage,
     trade_coverage, drawdown) vs. raw accuracy.

5. **Regularization (prime suspect for the overfitting)**
   - The failing config has `WEIGHT_DECAY 0.0`, `LABEL_SMOOTHING 0.0`, no focal loss, and
     `SEQUENCE_DROPOUT 0.2` only. Prescribe correct L2, label smoothing, and whether focal
     loss should be default. Give exact values per architecture.
   - Sequence length `SEQ_LEN 9` may be too short to capture regime — recommend a value and
     the tradeoff vs. inference latency in MT5.

6. **Model architecture** (`tradebot/models/`, `resolve_architecture`)
   - Which architecture is appropriate for this task and why; whether TKAN is a good default
     or a trap. Recommend one primary architecture + one fallback, with sizes.

7. **Loss & optimization**
   - Correct loss for the chosen label scheme; proper class weighting (fix the binary bug);
     LR schedule; gradient clipping (already present at 1.0 — confirm).

8. **ONNX export** (`export_onnx_model.py`, `tradebot/training/export_onnx_model.py`)
   - Verify the exported graph matches training preprocessing exactly (median/iqr constants
     baked in, no external scaler needed at inference — confirm `config.mqh` carries them).
   - Dynamic axes, op-set, and numerical parity between torch and ONNX outputs.

9. **MetaTrader import / live EA** (`live/live.mq5`, `live/functions/*.mqh`)
   - Confirm the EA applies the same median/iqr, feature order, and sequence assembly as
     training. Flag any mismatch that would silently degrade live performance.
   - Confirm `REQUIRED_HISTORY_INDEX`, `MODEL_FEATURE_COUNT`, `FEATURE_IDX_*` match the
     Python side — a mismatch here is a silent live failure.

---

## REQUIRED PLAN FILE FORMAT

Write the plan to `~/i/awc/9/PIPELINE_FIX_PLAN.md`. It must contain these sections, in order:

```
# AWC Pipeline Fix Plan

## 0. Executive summary (3-5 lines, plain language: what's broken, what to do)

## 1. Diagnosis table
| # | Stage | File:line | Problem | Severity | Root cause |
|...|

## 2. Decisions (your architectural judgment, locked in)
- Labeling scheme: <choice> — reason
- Architecture: <choice> — reason
- Regularization defaults: <exact values> — reason
- Validation: <walk-forward spec> — reason
- (one line each, no hedging)

## 3. Step-by-step implementation
For EACH step use this exact shape:
### Step N: <one-line goal>
- File: <exact path>
- Change: <what to modify — function/class/config key>
- Exact code / config:
  ```python
  # the exact new code or diff
  ```
  ```yaml
  # exact config values
  ```
- Why: <1-2 line rationale, in 9-year-old terms where ML concepts appear>
- Verify: <exact command + expected output, e.g. `source env/bin/activate && python scripts/i.py -c testrun`>

## 4. Validation protocol
- Walk-forward fold definition (exact windows)
- Out-of-sample holdout procedure
- Quality-gate thresholds that define "good enough to go live"
- How to confirm ONNX == torch numerically

## 5. Rollout & risk
- Order of changes (what to ship first to de-risk)
- What to watch in live MT5
- Kill-switch / revert steps
```

The plan must be **minimalist** (per `AGENTS.md`): smallest change that fixes the flaw, no
refactors beyond what's needed. But it must be **complete**: a model reading only this file
can implement the whole fix without asking you anything.

---

## CONSTRAINTS

- Follow the repo's `AGENTS.md`: minimalist changes, explain ML like the user is 9, code like
  a principal deep-learning engineer with zero tolerance for sloppiness.
- Never suggest adding `.onnx` to `.gitignore`.
- Preserve the existing config system (YAML + `.mqh`) — fixes go through it, not around it.
- The plan's verification commands must use `source env/bin/activate` and, for training,
  `python scripts/i.py -c testrun`.
- Do not overplan beyond what fixes the pipeline. Once the plan is written and self-contained,
  stop.

## DIFFICULTY & EFFORT

This is a hard, multi-system problem at the top of your range: it needs causal reasoning
about leakage, regularization theory, and production ML validation — not just code edits.
Spend the effort to get the diagnosis and the exact values right. The downstream model will
execute mechanically; your job is to be unambiguously correct.
