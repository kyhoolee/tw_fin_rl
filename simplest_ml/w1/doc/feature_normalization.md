# Feature Preparation Under Non‑Stationarity

- Trading data is non‑stationary: 
  - levels, 
  - variance, 
  - and distributions 
    - drift 
  - over time and across regimes. 

- A model trained on unnormalized levels 
  - often just learns regime state 
  - rather than repeatable patterns. 

- This note proposes 
  - practical, leak‑safe normalization strategies 
  - and feature designs 
  - to stabilize inputs 
  - so models can discover recurring structure.

## Goals
- Reduce sensitivity 
  -to level/volatility drift 
  - without leaking future information.

- Make features comparable 
  - across time, tokens, and regimes.

- Preserve informative signals 
  - (direction, structure) 
  - while taming heavy tails/outliers.

## Principles

- Normalize to recent history only 
  - (rolling/EW) 
  - and always lag by 1 bar.

- Prefer 
  - relative vs absolute 
  - (returns, ratios, spreads over levels).

- Detrend/neutralize known drivers 
  - (market beta, time‑of‑day seasonality).

- Use robust statistics 
  - (median/MAD, winsorization) to handle fat tails.

- Validate stability 
  - with drift metrics (PSI/KS) 
  - and backtest impact.

## Normalization Toolkit

1) Returnization (levels → returns)

- Use log or percent returns for price‑dependent features: 
  - `ret_k = close/close.shift(k)-1`.

- Candle anatomy normalized by range: 
  - `body/range`, `upper/range`, `lower/range`.

2) Volatility Scaling
- Scale returns by recent vol to equalize signal scale across regimes:
  - `ret_1_volscaled = ret_1 / (rv_24h + eps)` 
    - where `rv_24h = std(ret_1, 24h)`.

- For minute‑to‑hour features, 
  - scale minute aggregates by hourly vol (or ATR) computed on past data.

3) Rolling Z‑Score (leak‑safe)
- `z_t = (x_t − mean_{t−W}) / std_{t−W}` 
  - with both mean/std computed on a right‑closed window 
  - and shifted by 1 bar.

- Use multiple horizons 
  - (e.g., W∈{48, 168} hours) 
  - to stabilize both short/weekly regimes.

- Fallback when `std≈0` (use epsilon, or drop feature locally).


4) Robust Scaling
- Median/MAD: 
  - `z_robust = (x−median_{t−W})/(MAD_{t−W}+eps)`.

- Rolling winsorization (clipping) 
  - at percentile bands (e.g., 0.5%/99.5%) 
  - within the past W bars to cap outliers.

5) Seasonality Adjustment (Time‑of‑Day, Day‑of‑Week)
- Demean/scale per hour‑of‑day 
  - using rolling stats for that hour bucket only.

- Ex: 
  - `x_adj = (x−mean_{HOD, past K weeks})/std_{HOD, past K weeks}` 
  - to remove intraday recurring structure.

6) Cross‑Sectional Normalization (multi‑token)
- At each timestamp, 
  - convert features to percentiles/ranks across tokens 
  - to reduce cross‑asset scaling differences.

- Ex: 
  - cross‑sectional z‑score per hour; 
  - or use rank‑transform to [0,1].

- Caution: 
  - compute using only tokens available at that time; 
  - avoid peeking ahead.

7) Market/Beta Neutralization
- Regress token hourly returns 
  - on BTC return over rolling window W: 
  - `r_i = α + β*r_BTC + ε` using past data.

- Use residuals `ε` (or store β to create hedged features). 
  - Equivalent: subtract `β*r_BTC` from returns.

8) Spread/Relative Features
- Use ratios vs moving averages: 
  - `close/EMA_w − 1`, `VWAP_hour / EMA_w − 1`.

- Pair/relative returns: 
  - `ret_token − ret_BTC` to remove common shocks.


9) Probability Integral Transform (rolling quantiles)
- Map feature to its rolling CDF estimate over W: 
  - `u_t ≈ rank(x; past W)/W`. 
  - This creates stable uniform[0,1] inputs 
  - and can tame heavy tails.

10) Whitening / PCA (optional)
- Rolling PCA on selected correlated features; 
  - drop first PC (market mode) 
  - and use residual components. 

- Keep state per fold to avoid leakage.

## Label Stabilization (brief)

- Use H‑step cumulative return labels 
  - with epsilon bands to ignore tiny moves.

- Volatility/volume regime labels 
  - via rolling percentiles computed on train folds only; 

- apply to validation/OOS.

## Leak‑Safe Implementation Patterns

- All rolling/EW statistics 
  - must be computed on past data only 
  - and applied with `shift(1)` at the modeling frequency.

- For minute→hour aggregation, 
  - resample with right‑closed windows
  - and then lag the aggregated features 1 hour when joining the hourly frame.

- In cross‑validation, 
  - compute normalizers on the train fold and apply to its test fold. 
  
- Do not reuse train state across folds unless it mimics live expanding training.

## Drift Monitoring

- Population Stability Index (PSI): 
  - compare recent window vs baseline window per feature.

- KS distance / Wasserstein distance 
  - between past and recent windows.

- Alert on features whose drift exceeds thresholds; 
  - consider re‑normalization or dropping.

## Practical Recipes (by block)

- Returns
  - Use vol‑scaled returns: 
    - `ret_1 / (rv_24h+eps)` and `ret_k / (rv_24h+eps)`.

- Candle anatomy
  - Normalize by range and add z‑scores of `body/range` with W∈{48,168}.

- Volatility features
  - Use z‑scores of realized vol (e.g., `rv_24h_z`) 
    - and relative vol `rv_24h / rv_168h`.

- Volume
  - Volume z vs rolling same‑hour baseline; 
    - include rank within day to remove intraday profile.

- Time features
  - Use sin/cos, 
    - and optionally include a de‑seasonalized residual of returns after HOD adjustment.

- Cross‑asset
  - Keep residual returns after BTC beta removal and BTC vol context z‑scored.

## Caching the Normalization State

- For each derived shard (hourly features), 
  - persist normalization metadata alongside the Parquet:
    - rolling window length(s), 
    - method 
      - (mean/std or median/MAD), 
      - epsilon, 
      - winsor clip percentiles.

- In v4, implement a `Normalizer` object with:
  - `partial_fit(df_train)` 
    - → updates internal state (means/stds or quantile bins) per feature

  - `transform(df)` 
    - → applies to new data; returns normalized df and diagnostics

  - Serialize parameters to JSON next to artifacts for reproducibility.

## Integration into Pipeline

1) Compute raw engineered features at minute/hour scale.

2) Apply within‑fold normalizer (train→fit, test→transform) with 1‑bar lag at the end.

3) Optionally add cross‑sectional rank transforms at each timestamp.

4) Monitor drift metrics and log per‑fold stats; 
  - drop unstable features if needed.

## Sanity Checks

- Histogram tails after normalization should be bounded (post‑winsor) and roughly comparable across time.
- Vol‑scaled returns should have near‑constant variance across regimes.
- Cross‑token ranks should look uniform[0,1] at each timestamp.

---

This strategy yields features with more stable distributions across time/tokens while avoiding leakage. With the stabilized inputs, logistic/GBDT models are more likely to learn patterns that repeat across regimes rather than memorize regimes themselves.

