# Minute Features — Prepare & Cache Plan

Goal: 
  - Prepare leak‑safe short‑term (1m) features 
    - aligned to the hourly modeling 
  - index with strong caching 
    - so experiments don’t recompute expensive transforms.

## Inputs and Outputs

- Inputs
  - Raw 1m shards (monthly): 
    - `data/minute/binance_{symbol}_1m_YYYYMM.parquet`
  - Hourly modeling index 
    - (from 1h klines or from v3 pipeline)

- Outputs (cache layers)
  - L1 Cleaned 1m shard (optional): 
    - normalized dtypes/index, 
    - columns pruned 
      - → same file path or `data/minute_clean/...`

  - L2 Hour‑aggregated from minutes (monthly/day shards): 
    - `data/derived/{symbol}_hour_from_1m_{feature_set_id}_{YYYYMM}.parquet`

  - L3 Union view for modeling: 
    - `get_hour_features(symbol, start, end, feature_set_id)` 
    - returns a single hourly‑indexed frame to join into v4 pipeline

## Feature Set (initial)

- Compute per hour t using only minutes ≤ t 
  - (then lag by 1 hour in modeling step):

- Momentum
  - `ret_1m_5m`, 
  - `ret_1m_15m`, 
  - `ret_1m_60m` 
  - (close[t] / close[t−k] − 1 using 1m closes)
  - skew/kurtosis of 1m returns over 15m/60m

- Volatility
  - realized volatility: 
    - std of 1m returns over 15m/60m

  - bipower variation proxy over 60m

  - 60m high−low range percentile (within trailing 30d at the same hour)

- Volume
  - sum/mean volume over 5/15/60m
  - volume z‑score vs trailing 7d at same hour
  - burst flag: top X% minute volumes in last 24h

- Microstructure (from candles only)
  - mean of sign(close−open)*volume_z over last 15m
  - VWAP divergence: (close_hourly − VWAP_1m_last_60m) / VWAP


- All aggregations are right‑closed and aligned to the hour’s right edge; 
  - when joined to 1h modeling frame, we still apply a 1‑bar lag to prevent leakage.

## Caching & Versioning

- `feature_set_id`: 
  - stable identifier computed from a JSON of parameters (windows, zscore windows, flags). 
    - Example: `v4m_w=(5,15,60)_z7d`.

- Output shards: 
  - `data/derived/{symbol}_hour_from_1m_{feature_set_id}_{YYYYMM}.parquet`

- Manifest per symbol:
  - `data/derived/{symbol}_hour_from_1m_{feature_set_id}.manifest.json` 
    - with entries:
      - month, 
      - rows, 
      - time_span, 
      - created_at, 
      - src_shard_path, 
      - src_mtime

- Invalidation rules
  - If source shard is newer (mtime) than derived shard → rebuild month
  - If `feature_set_id` changes → new set of output files; old ones are preserved

## Rolling Window Boundaries (Halo)

- Monthly shards cut rolling windows; 
  - metrics like 60m volatility need the previous 59 minutes from the prior shard.

- Strategy: load each month with a left halo from the previous month
  - halo minutes = max_window_minutes − 1 (e.g., 59 for 60m)
  - compute aggregates on (halo + month), 
    - then drop halo rows before writing the monthly output
- The 30d/7d percentiles use longer horizons; 
  - compute using expanding stats on the union of shards loaded in sequence 
    - (keep running state in memory), 
  - or keep a longer halo (e.g., 7d) across shard boundaries when building a quarter at a time.

## API Design (features_minute.py)

- `load_minute_shard(symbol, yyyymm, minute_dir) -> df_1m`

- `iter_months(start, end) -> [(start_dt, end_dt, yyyymm)]`

- `aggregate_hour_from_minute(df_1m, halo_prev=None, params) -> df_hour`
  - Resample to `'1h'` (label='right', closed='right'), 
    - compute windows per params
  - Return hourly‑indexed frame for this month (without halo)

- `build_hour_from_minute(symbol, start, end, params, minute_dir, derived_dir) -> List[str]`
  - Stream months in order with halo, 
    - write `..._{YYYYMM}.parquet`, 
    - update manifest

- `get_hour_features(symbol, start, end, feature_set_id, derived_dir) -> pd.DataFrame`
  - Concatenate the necessary shards and trim to `[start,end)`


## Performance Tactics

- Read only needed columns from minute shards: 
  - `open, high, low, close, volume`
- Use `float32` where feasible to reduce memory
- Vectorize with NumPy as much as possible; avoid Python loops
- Build per month in chronological order to reuse rolling state; drop references to free memory
- Optional: 
  - use Polars for faster groupby/resample if installed (but default to pandas for portability)

## Logging & Safety Checks

- Print per‑month progress: time span, input rows, output rows
- Check monotonic index and duplicate minutes; warn and drop
- After aggregation, verify
  - max minute timestamp per output hour ≤ that hour edge
  - no NaNs created unexpectedly (apart from initial warm‑up hours)
- Update manifest at the end of each month shard

## Usage Pattern (Notebook)

1) Build/refresh derived hour features from minutes

```python
from features_minute import build_hour_from_minute, default_params
params = default_params()  # returns dict with windows/zscore settings
build_hour_from_minute(
    symbol="BTCUSDT",
    start="2023-10-30",
    end="2025-10-30",
    params=params,
    minute_dir="simplest_ml/w1/data/minute",   # or "data/minute" if running inside w1
    derived_dir="simplest_ml/w1/data/derived",
)
```

2) Load features for modeling

```python
from features_minute import get_hour_features, make_feature_set_id
fid = make_feature_set_id(params)
df_hour = get_hour_features(
    symbol="BTCUSDT",
    start="2023-10-30",
    end="2025-10-30",
    feature_set_id=fid,
    derived_dir="simplest_ml/w1/data/derived",
)
```

## Implementation Steps

1) Scaffold `features_minute.py` with the API above, plus `default_params()` and `make_feature_set_id()`
2) Implement month iterator + halo logic; write one shard end‑to‑end with small window set
3) Add rolling stats and percentiles; keep running state between months where needed
4) Add manifest writing and invalidation checks
5) Wire minimal integration into v4 pipeline (join to hourly frame, lag, train)

---

This plan keeps the cost of recomputation low by writing derived hour‑level shards keyed by a feature‑set identifier and by rebuilding only the months whose minute sources changed.

