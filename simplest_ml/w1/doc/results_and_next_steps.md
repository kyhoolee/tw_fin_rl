# Multi-Token Baseline — Results & Next Steps

This note summarizes the baseline results in `simplest_ml/w1/2_2_result.md` and proposes focused improvements.

## Current Baseline

- Data: 
  - 1h candles, 
  - span 2023-10-30 → 2025-10-30 (per symbol), 
  - saved as Parquet.

- Tokens: 
  - BTC, BNB, SOL, ETH, AVAX, TRON, ARBtrium, Sui, Polygon (Polygon has shorter span).

- Features: 
  - TA-style rolling features (ret_w, sma_w, ema_w, vol_w, max/min_w, rsi_w); 
  - all lagged by 1 to avoid leakage.

- Label: 
  - Binary next-bar direction (close[t+1] > close[t]).

- Model: 
  - TimeSeriesSplit(5) 
  - + StandardScaler 
  - + LogisticRegression(C=0.5, max_iter=2000).

- Metrics snapshot (see full logs in `simplest_ml/w1/2_2_result.md`):
  - AUC ~ 0.49 – 0.53 across tokens; Accuracy ~ 0.50 – 0.52.
  - Examples: BTC AUC 0.514 ACC 0.515; ETH AUC 0.510 ACC 0.514; SOL AUC 0.513 ACC 0.510.

## Observations

- Signal is weak at the 1h next-bar horizon (near-random overall performance).
- Class balance roughly even; recall/precision can be imbalanced per token (e.g., AVAX low recall(1)).
- Polygon has fewer rows (shorter history), reducing stability.
- No regime awareness or cross-asset context in features.

## Key Conclusions From Results

- Non-stationarity: 
  - crypto is regime-driven; 
  - distributions drift over time.

- Next-hour direction is close to random with basic TA features and a 0.5 threshold.

- Simple descriptive stats (mean/var/min/max) fluctuate strongly; 
  - naive normalization is insufficient.

- Label close[t+1] > close[t] is noisy; 
  - tiny moves and microstructure noise dominate at 1h.

Implication: 
- We need 
  - robust targets, 
  - adaptive normalization, 
  - regime/context features, 
  - and validation that respects time.

## Likely Causes

- 1h next-bar direction is noisy; single-step target amplifies noise.
- Fixed 0.5 threshold; no threshold tuning or probability calibration.
- Feature set lacks candle anatomy, volatility regime, cross-asset signals, and time-of-day effects.

## Next Actions (Prioritized)

- Targets/Labels
  - Multi-horizon: predict sign of cumulative return over H bars (e.g., H=4, 12).
  - Noise filter: ignore tiny moves (|ret_next_H| ≤ ε) or use 3-class target.
  - Threshold tuning: select decision threshold on validation (maximize F1/Youden J).

- Features
  - Candle anatomy: body, wicks, body/range ratio, true range (TR), gaps.
  - Volatility/Range: ATR, Bollinger band width, realized volatility.
  - Momentum/Trend: MACD (diff, signal, hist), ADX/DI±, CCI, Stoch %K/%D, Williams %R.
  - Volume: OBV, volume z-scores vs rolling mean/std, volume spikes.
  - Cross-asset: BTC returns/vol as context; alt/BTC relative performance.
  - Time features: hour-of-day, day-of-week (sin/cos encoding).
  - Per-token rolling standardization (z-score) inside each fold.

- Modeling
  - LogisticRegression upgrades: `class_weight='balanced'`, L1 penalty for selection.
  - Add a tree model baseline (LightGBM/XGBoost) to capture nonlinearity and interactions.
  - Probability calibration (Platt/Isotonic) and per-token threshold selection.
  - Simple ensemble (avg probabilities from LogReg + GBDT).

- Validation & Trading
  - Walk-forward evaluation emphasizing recent periods.
  - Backtest with costs (e.g., 4–10 bps), confidence filters (|p-0.5| > m), volatility-adjusted sizing.
  - Regime slicing: evaluate metrics by volatility deciles and bull/bear segments.

## Proposed Implementation Plan

1) Label Extensions (quick win)
- Add horizon H and ε filter options to the pipeline.
- Implement threshold tuning on validation folds; report tuned threshold per token.

2) Feature Expansion
- Extend `add_features()` with candle anatomy, ATR, BB width, MACD set, simple time features.
- Add an option to compute rolling z-scores per feature.

3) Modeling Variants
- Add LightGBM baseline (optional dependency); keep LogReg path for minimal setup.
- Optionally enable `class_weight='balanced'` and L1.

4) Simple Backtest
- Translate probabilities to positions with upper/lower thresholds; include fees.
- Report PnL, hit rate, Sharpe, max drawdown, turnover.

## Remedies Mapped To Conclusions

- Non-stationarity
  - Walk-forward CV; recent-weighted validation; rolling re-fit cadence.
  - Rolling z-score features; log-returns; volatility scaling.
  - Regime features (volatility deciles, trend state), model per regime if beneficial.

- Traditional logic ~ random
  - Expand nonlinear models (LightGBM/XGB) and interaction-rich features.
  - Cross-sectional lens: predict relative rank across tokens instead of absolute direction.

- Feature fluctuation / normalization
  - Per-token rolling standardization within folds; robust stats (median/MAD).
  - Range/ATR/Bollinger width to capture state, not just levels.

- Noisy label (close[t+1])
  - Multi-horizon H (e.g., 4/12 bars) cumulative return sign.
  - Epsilon band to ignore tiny moves; or 3-class (up/flat/down).
  - Triple-barrier labeling (take-profit/stop/timeout) for path-aware labels.

## Next Experiments Shortlist

1) Horizon+Noise Filter
- Label = sign(sum of returns over H=4); ignore |ret| ≤ ε=0.1%.
- Tune threshold on validation; LightGBM baseline vs LogReg.

2) Triple-Barrier + Backtest
- Volatility-scaled barriers (e.g., 1.5σ TP, 1.0σ SL, timeout H=12).
- Convert probabilities to trades with fees; report Sharpe/CAGR/DD.

## Open Questions

- Which horizons to target first (H=4, 12)?
- ε band for noise filtering (e.g., 0.05%–0.15% on 1h)?
- Fee/slippage assumptions and trading frequency limits?
- Priority tokens/universes (only top liquidity?)

## References (Code)

- Pipeline: `simplest_ml/w1/multi_token_modeling.py`
- Data helpers: `simplest_ml/w1/pretty_helpers.py`
- Full baseline logs: `simplest_ml/w1/2_2_result.md`
- Run guide: `simplest_ml/w1/doc/run_step.md`
