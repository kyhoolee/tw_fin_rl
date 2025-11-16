# v4 Plan — Minute + Hour Multi‑Scale Modeling

- This plan extends the baseline (v1–v3) 
  - with 1‑minute data, 
  - multi‑scale features (minute→hour), 
  - and additional labels for volatility/volume regimes. 

- It keeps 
  - leak‑safe engineering, 
  - walk‑forward validation, 
  - and fee‑aware backtests.

## Scope

- Ingest 1m candles in addition to 1h.
- Build short‑term (1m) + mid‑term (1h) features aligned at hourly decision times.
- Add extra labels: volatility/volume regimes alongside direction.
- Keep OOS‑only backtests and charts with fee scenarios.

## Data

- Source: Binance klines.
- Storage (Parquet):
  - Hourly: `data/hourly/binance_{symbol}_1h_YYYYMMDD_YYYYMMDD.parquet` (existing)
  - Minute: `data/minute/binance_{symbol}_1m_YYYYMM.parquet` (monthly shards)
- Hygiene:
  - UTC timezone; monotonic index; drop duplicates.
  - No imputation for modeling; allow gaps (features must respect available data).
  - Ingestion report per symbol: rows, span, %missing.

## Alignment (No‑Leak)

- Decision time: end of each 1h bar (right edge).
- Minute features for hour `t` may use only minutes ≤ `t`.
- Resample minutes → hourly using right‑closed windows, then lag:
  - `.resample('1h', label='right', closed='right').agg(...)` → `.shift(1)`
- Unit checks: for random timestamps, assert max source minute ≤ target hour.

## Features

- Hourly (mid‑term, keep v3 blocks + a few extras):
  - Returns/moving averages/EMA/volatility/min/max/RSI‑like (existing)
  - Range & vol: `true_range_z`, `bb_width_z`
  - Trend strength: ADX/DI±(14) (simple formula)
  - Acceleration: `ret_1.diff()`
- Minute (short‑term, aggregated per hour t using minutes up to t):
  - Momentum: `ret_1m_last_5m/15m/60m` ; skew/kurt of 1m returns over 15/60m
  - Volatility: realized vol of 1m returns over 15/60m; bipower variation; 60m range percentile
  - Volume: 1m volume sum/mean over 5/15/60m; minute volume z vs 7‑day at same hour; burst flags (top X%)
  - Microstructure from candles: sign(close−open)·(volume_z) mean over last 15m; VWAP divergence vs hourly close
- Cross‑scale interactions: ratios/differences like `rv_1m_60m / rv_hour_24h`, `vol_1m_60m_z − vol_hour_z`
- Time: hour‑of‑day/day‑of‑week sin/cos.
- Normalization: rolling z‑score with `shift(1)`; per‑fold StandardScaler.

## Labels

- Direction (main, 3‑class): `y_dir ∈ {−1, 0, +1}` via H‑step cum return with epsilon band.
- Volatility regime (binary or 3‑class):
  - Realized vol of 1h returns over `Wv` (e.g., 24h).
  - Percentile bins: low/normal/high or `y_vol_high=1` if > 80%ile (computed on train folds only).
- Volume regime (binary): `y_volm_up=1` if hourly volume > 80%ile of rolling 30d at same hour.
- Optional event labels: Range expansion next hour; Large move over next H (|cumret_H| > ε_big).

## Modeling

- Baseline: multinomial Logistic for `y_dir`; LightGBM/HistGB optional for nonlinearity.
- Multi‑task: train separate classifiers for `y_vol_high`, `y_volm_up`; use as trading filters.
- Threshold tuning: per token, on validation folds, for both `y_dir` and filter thresholds.
- Regularization/selection: drop near‑constant; prune |corr|>0.98 on train folds only.

## Validation

- Default: walk‑forward TimeSeriesSplit (same as v3).
- Options:
  - `split_mode='final_fold'`: report only most recent OOS block.
  - `split_mode='dates'`: e.g., train 2023‑10‑30→2024‑03‑31; tune 2024‑04‑01→2024‑06‑30; OOS 2024‑07‑01→end.
- Logs: per‑fold metrics, tuned thresholds, alignment assertions.

## Backtest

- Hourly decisions at bar close; fee scenarios: gross, VIP9=2.0bps, VIP4=5.4bps.
- Filters: only trade if aux predictions indicate vol_high=1 and/or volume_rising=1.
- Risk knobs: min_hold/cooldown; cap trades/day.
- Metrics: CAGR/Sharpe, fee‑drag curve, trades/day, avg_hold, turnover; pooled equity across tokens.

## Charts

- Two‑panel retained: equity (with OOS shading + hold baseline) + trades/day.
- Add fee‑drag overlay/annotation and optional vol_high rate over time.

## Deliverables

- Code:
  - `simplest_ml/w1/data_minute.py` — 1m fetch + Parquet writer (monthly shards, backoff).
  - `simplest_ml/w1/features_minute.py` — minute→hour aggregations (no‑leak helpers).
  - `simplest_ml/w1/multi_token_modeling_v4.py` — v3 + minute features + extra labels + split options.
- Docs:
  - Update `doc/run_step.md` with multi‑scale pipeline.
  - Add `doc/labels.md` with definitions and thresholds.
  - Add `doc/features_multiscale.md` (recipes + alignment diagrams).
- Artifacts:
  - Per‑token PnL CSVs & equity PNGs (fee tiers + fee‑drag, filters noted).
  - Per‑fold CSV: trades/day, avg_hold, net Sharpe, tuned thresholds.

## Implementation Steps

1) Minute ingestion
- Add `fetch_klines(..., interval='1m')` batching; write Parquet to `data/minute/` by month.

2) Aggregation utilities
- `agg_minute_to_hour(df_1m, end_index_1h)` → aligned feature frame; right‑closed, `shift(1)`; unit tests.

3) Feature wiring
- Extend v3 `add_features(..., include_minute=True, minute_windows=(5,15,60))` and join to hourly index.

4) Labels (multi‑task)
- Implement `make_label_volatility`, `make_label_volume` with percentiles computed on train folds.

5) Validation & backtest
- Add `split_mode` options; implement filters from aux predictions; keep fee tiers and charts.

6) Reporting
- Fee‑drag line; per‑fold table; pooled equity. Persist configs + metrics (JSON/CSV) under `_exp/`.

## Runtime Notes

- 2y of 1m per symbol ≈ ~1M rows → use column pruning, Snappy Parquet; cache derived hour features to `data/derived/`.
- Rate limiting: ~200–300ms between API calls; implement retry with exponential backoff.
- Memory: compute minute aggregations in monthly chunks, then append.

---

If you want, I can start with steps (1) and (2): minute fetcher + leak‑safe aggregators, then wire a minimal v4 that adds a small set of minute features and a volatility label to compare against v3.

