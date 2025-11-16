# Multi-Token Modeling — Run Steps

This note documents the end-to-end run steps for the multi-token pipeline that mirrors the single-token BTC notebook logic.

## Overview

- Scope: EDA → Feature engineering → Labeling → TimeSeries cross-validated training for many tokens.
- Data source: Parquet files saved earlier (one file per symbol, interval, and date range).
- Core modules:
  - `simplest_ml/w1/multi_token_modeling.py`
  - `simplest_ml/w1/pretty_helpers.py`

## Prerequisites

- Installed packages: `pandas`, `numpy`, `scikit-learn`, `pyarrow` (or `fastparquet`), `prettytable`.
- Parquet files exist in the data folder, named like:
  - `data/binance_{symbol}_{interval}_{start}_{end}.parquet`
  - Example: `data/binance_btcusdt_1h_20231030_20251030.parquet`

## Quick Start (Notebook cell)

```python
from multi_token_modeling import run_per_token_pipeline, summarize_results

tokens = ["BTC", "BNB", "SOL", "ETH", "AVAX", "TRON", "ARBtrium", "Sui", "Polygon"]

results = run_per_token_pipeline(
    tokens=tokens,
    data_dir="data",      # use "simplest_ml/w1/data" if running from repo root
    interval="1h",
    start="20231030",
    end="20251030",
    windows=(3, 6, 12, 24),
    n_splits=5,
    C=0.5,
    max_iter=2000,
    show_eda=False,        # True to see per-token samples and describe()
    verbose=True           # detailed step-by-step logs
)

summarize_results(results)
```

## Step-by-Step (what the pipeline does)

The pipeline runs the following steps per token (with a clear `[TOKEN]` log prefix):

1) LOAD
- Loader: `pretty_helpers.load_token_df()` chooses the best Parquet file for the symbol, preferring the exact `start/end` if provided.
- Logs: row/column counts, time span, and whether required columns `open, high, low, close, volume` are present.

2) EDA (optional)
- Function: `eda_brief(df, token, symbol)`.
- Shows a small head/tail sample using `pretty_print_sample()` and `pandas.describe()` for price/volume columns.
- Use `show_eda=True` to enable in `run_per_token_pipeline()`.

3) FEATURES + LABEL
- Function: `prepare_Xy()` which calls:
  - `add_features()` to compute:
    - Momentum/returns: `ret_1`, `ret_w`
    - Moving averages: `sma_w`, `ema_w`
    - Volatility: `vol_w` (std of returns)
    - Extremes: `max_w`, `min_w`
    - Oscillator: `rsi_w`
  - All features are then lagged by 1 to prevent leakage.
  - Label: `y = (close[t+1] > close[t])` (binary next-step up).
- Logs:
  - Number of generated features, total NaNs after lagging.
  - Rows before/after `dropna()` and the selected feature count.

4) TRAIN
- Method: TimeSeries cross-validation with `StandardScaler` and `LogisticRegression`.
- Parameters: `n_splits`, `C`, `max_iter`.
- Logs:
  - Per-fold train/test sizes, fold AUC and ACC.
  - Final out-of-fold metrics (AUC, ACC), classification report, and confusion matrix.

5) SUMMARY
- Use `summarize_results(results)` to print a compact table across tokens: rows used, AUC, ACC, Precision, Recall, and span.

## Key Parameters

- `tokens`: list of names or symbols (e.g., `"BTC"` or `"BTCUSDT"`).
- `data_dir`: where the Parquet files live (match your save path).
- `interval`: `"1h"` (or other if your files use a different interval).
- `start`, `end`: date tags used in filenames (yyyymmdd). Exact-match preferred; falls back to the latest available range.
- `windows`: feature windows, default `(3, 6, 12, 24)`.
- `n_splits`, `C`, `max_iter`: model and CV settings.
- `show_eda`, `verbose`: control printing and logs; set `verbose=False` for quieter runs.

## Troubleshooting

- No Parquet found: ensure filenames match the pattern and `data_dir` path is correct.
- Missing Parquet engine: install `pyarrow` (or `fastparquet`).
- `prettytable` missing: `pip install prettytable`.
- Too few rows after feature prep: expand the date range, or reduce feature windows.
- Class imbalance: tune threshold or try different models; this baseline uses 0.5.

## File Map

- Pipeline: `simplest_ml/w1/multi_token_modeling.py`
- Helpers (load/print): `simplest_ml/w1/pretty_helpers.py`

