# trade-bot

End-to-end ML pipeline for Forex algo trading. Trains neural networks on tick data, exports to ONNX, and executes live in MetaTrader 5.

## Layout

```
config/          # Config presets (YAML) + active config pointer
scripts/         # CLI entry points (run from repo root)
  i.py           #   train → export ONNX → update live EA → compile
  test.py        #   backtest archived models
  export_data.py #   export ticks from MT5 to CSV
  i.sh / i.ps1   #   train + backtest in one command
live/            # MQL5 EA source
  live.mq5       #   main EA
  functions/     #   50+ .mqh includes (features, risk, execution)
mt5/             # MT5 data export scripts + compiled binaries
common/          # Python feature definitions, bar building, config
tradebot/        # Python pipeline
  models/        #   TKAN, LSTM, GRU, Mamba, Chronos, MiniRocket
  training/      #   trainer, ONNX export, diagnostics
  pipeline/      #   data processing, windowing, feature building
symbols/         # Per-symbol model archives
  xauusd/models/ #   30+ trained XAUUSD model snapshots
  btcusd/models/ #   15+ trained BTCUSD model snapshots
```

## Usage

Run from repo root:

```bash
python scripts/i.py -c default.yaml     # train + export
python scripts/test.py                   # backtest archived models
python scripts/export_data.py --profile gold  # export XAUUSD ticks
./scripts/i.sh                            # train + backtest (Linux)
```

## Config System

Configs are YAML files in `config/`. The active config is resolved as:

1. `config/.active_config` — first line is a relative path to a preset
2. `config/active.mqh` — fallback

To switch config:

```bash
echo "config/default.yaml" > config/.active_config
```

Key config sections:

| Section | What it controls |
|---------|-----------------|
| `architecture` | TKAN, LSTM, Mamba, Chronos, MiniRocket |
| `target` | Fixed-point or ATR-based SL/TP labels |
| `features.toggles` | Which technical features to include (100+ toggles) |
| `training` | Sequence length, epochs, batch size, confidence |
| `bars` | Time bar, tick bar, or imbalance bar construction |
| `trade_sizing` | Lot size, risk %, broker min lot |

## Model Archive

Each trained model lives under `symbols/<SYMBOL>/models/<stamp>-<name>/`:

```
model.onnx            # ONNX for MT5 inference
config.mqh            # Combined config
diagnostics/
  report.md
  validation_predictions.csv
  holdout_predictions.csv
  confusion_matrix.csv
  active_features.txt
```

The active model is referenced in `live/live.mq5` via a model path and ONNX resource buffer — the training script (`i.py`) updates this automatically.

## Live EA

`live/live.mq5` + 50 function files. Supports:

- Time, tick, and imbalance bar construction
- 100+ technical features with cross-asset context (USDX, USDJPY)
- ONNX inference with softmax (BUY/HOLD/SELL or BUY/SELL)
- ATR-based or fixed-point SL/TP
- Stop/freeze level validation, broker min stop checks
- Risk-based or fixed-lot position sizing
- Full trade logging and run summary

## Requirements

```
torch>=2.5
onnx>=1.16
pandas>=2.2
numpy>=1.26
chronos-forecasting>=2.2
tqdm
```

## Symbols

- **XAUUSD** — 30+ trained model snapshots, 54s/1m time bars
- **BTCUSD** — 15+ trained model snapshots
