# Feature Processing Plan (Hourly + Minute)

This doc consolidates how we will 
  - build, normalize, and cache features for 
  - both hourly and minute candles, 
  - and lists concrete implementation actions.

## Objectives

- Produce 
  - leak‑safe, 
  - stable features 
  - for modeling at 1h frequency.

- Support 
  - multi‑scale inputs by aggregating 1m into 1h features 
  - with right‑closed windows.

- Cache all 
  - expensive transforms so experiments don’t recompute.

- Use basic per‑series normalization (no cross‑sectional at this stage).

## Data Sources

- Hourly klines (existing): `data/hourly/` or `simplest_ml/w1/data` (notebooks may differ by CWD).
- Minute klines (v4): monthly shards: `data/minute/binance_{symbol}_1m_YYYYMM.parquet`.

## Hourly Features (mid‑term)

Blocks (all lagged by 1 at the end):
- Returns: `ret_k` (k ∈ {1,3,6,12,24}).
- MAs/EMAs: `sma_k`, `ema_k`.
- Volatility/Range: `vol_k = std(ret_1, k)`, ATR(k), `bb_width_k`.
- Candle anatomy: `body`, `range`, `body_ratio`, `upper`, `lower`, `true_range`, `gap`.
- Momentum/Trend: RSI‑like, optional ADX/DI±, MACD set when needed.
- Volume/Flow: volume z vs rolling mean/std, OBV.
- Time: hour‑of‑day/day‑of‑week sin/cos.

Normalization (basic):
- Rolling z‑score per feature with W ∈ {48, 168} (applied inside fold, then `shift(1)`).
- Vol‑scaled returns: `ret_1 / (rv_24h + eps)` where `rv_24h = std(ret_1, 24h)`.
- Optional winsorization at past‑window percentiles to cap outliers.

Caching:
- Keep computed hourly features in memory within a run; heavy caching is mainly needed on minute→hour path.

## Minute→Hour Features (short‑term)

Aggregation per hour t using minutes ≤ t (then 1h lag on join):
- Momentum: `ret_1m_last_5m/15m/60m`, skew/kurt of 1m returns over 15/60m.
- Volatility: realized vol over 15/60m; bipower variation 60m; 60m range percentile (trailing 30d at same hour).
- Volume: sum/mean over 5/15/60m; volume z vs trailing 7d at same hour; burst flags (top X%).
- Microstructure: mean(sign(close−open) * volume_z) last 15m; VWAP divergence vs hourly close.

Normalization (basic):
- Apply rolling z‑scores over recent windows (e.g., 7d for volume z, 48/168h for others) with `shift(1)`.
- Vol‑scale minute returns by hourly past vol when combining with hourly features.

Caching & Versioning:
- Derived shards per month: `data/derived/{symbol}_hour_from_1m_{feature_set_id}_{YYYYMM}.parquet`.
- `feature_set_id` encodes windows/flags; e.g., `v4m_w=(5,15,60)_z7d`.
- Manifest per symbol to track source shard mtimes and invalidate only changed months.
- Halo: when building month M, prepend max(window)−1 minutes from month M−1 so rolling windows cross the boundary; drop halo rows before save.

## Split & Leakage Rules

- Resampling: right‑closed (`label='right', closed='right'`) so the hour at time t uses minutes up to and including t.
- Final step: shift all engineered features by 1 hour before modeling to prevent look‑ahead.
- In CV: fit normalizers and any per‑feature stats on train folds only; transform test folds.

## Outputs Delivered to Modeling

- Hourly index DataFrame per symbol with columns:
  - Hourly engineered features
  - Minute‑aggregated features
  - Normalized variants (z/robust, vol‑scaled) per configuration
- Plus metadata: `feature_set_id`, normalization params (JSON), source ranges.

## Implementation Actions

1) Minute ingestion (DONE)
- `data_minute.py`: downloads 1m shards with logs and monthly Parquet outputs.

2) Minute→Hour aggregation utilities (NEW)
- Create `features_minute.py`:
  - `default_params()`: windows and flags for feature_set_id.
  - `make_feature_set_id(params)`: stable string hash from params.
  - `iter_months(start, end)` and `load_minute_shard(symbol, yyyymm, minute_dir)` (read only `open, high, low, close, volume`).
  - `aggregate_hour_from_minute(df_1m, params, halo_state=None)` → hourly frame for the month, plus updated halo_state.
  - `build_hour_from_minute(symbol, start, end, params, minute_dir, derived_dir)` → write monthly derived shards, update manifest, print logs.
  - `get_hour_features(symbol, start, end, feature_set_id, derived_dir)` → concat monthly shards for modeling.

3) Normalizer (BASIC)
- Implement `Normalizer` class:
  - per‑feature rolling z with windows W; winsor percentiles optional.
  - `fit(train_df)` (stores per‑feature mean/std histories or uses rolling transform with shift inside fold)
  - `transform(df)` returns normalized df + diagnostics (NaN count, std zeros)
  - serialize chosen config to JSON alongside derived shards.

4) Integration into v4 modeling
- Add `include_minute=True` path in v4 pipeline that calls `get_hour_features(...)` and joins to hourly OHLCV.
- Apply Normalizer inside each CV train fold; transform test fold; then `shift(1)`.
- Log drift metrics (basic): std/mean ranges per fold.

5) Caching hygiene
- Respect manifest invalidation (rebuild only when source minute shard mtime > derived mtime or params changed).
- Use `float32` for numeric columns to reduce size.

6) Notebooks
- Add `4_2_minute_to_hour_features.ipynb` sketch:
  - Build/rebuild derived shards for a small token set.
  - Load and visualize a couple of features over time (post‑lag) to verify stability.

## Notes

- We intentionally defer cross‑sectional normalization to keep v4 simple and fast to iterate.
- For heavy percentile features (30d/7d), consider building by quarter to keep sufficient halo in memory.

