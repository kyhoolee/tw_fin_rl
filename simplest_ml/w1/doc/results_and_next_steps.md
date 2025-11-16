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

## Immediate Action Plan (v2 follow-ups)

- Proper OOS Backtest (no refit bias)
  - Use out-of-fold probabilities to construct an aggregate OOS PnL series per token.
  - Report token-level and pooled OOS: CAGR, Sharpe, maxDD, hit-rate, turnover.
  - Save OOS equity curves (CSV) under `simplest_ml/w1/_exp/` for inspection.

- Threshold Tuning per Token
  - On validation folds, tune upper/lower thresholds (t_long, t_short) by maximizing F1(±1) or Youden J.
  - Lock tuned thresholds and apply to OOS proba; log chosen values.

- Multiclass Macro AUC
  - Compute final macro AUC (one-vs-rest, OVR) using full probability matrices.
  - Include macro AUC in the summary PrettyTable and per-token JSON.

- Sensitivity Sweeps
  - Thresholds: t_long/t_short ∈ {0.55, 0.60, 0.65}; Fees: {4, 6, 10} bps; Min-hold: {1, 4} bars.
  - Emit a compact grid summary (CSV) for each token with metrics; highlight robust regions.

- Run Logging & Reproducibility
  - Persist run config (tokens, label H/ε, feature flags, thresholds, fees, splits) to JSON with a timestamp.
  - Save per-token metrics and tuned thresholds to CSV; include fold-level metrics for diagnostics.
  - Record library versions and random seeds.

- Optional Baselines
  - Add LightGBM classifier (shallow, early stopping) and report identical OOS/backtest metrics.
  - Compare to multinomial logistic; optionally add probability calibration.

## Visualization (Charts)

What to plot and why, plus quick code snippets.

- Equity Curve (PnL)
  - Plot cumulative equity: `eq = (1 + pnl).cumprod()`; baseline at `y=1.0`.
  - Per token, and an average/pooled curve.
  - Save: `_exp/{run_id}/{token}_equity.png`.

- Drawdown
  - `dd = eq / eq.cummax() - 1`; plot alongside equity or as a separate panel.
  - Useful to see tail risk and regime sensitivity.

- Rolling Sharpe / Volatility
  - 30D or 7D windows (in hours): `roll = pnl.rolling(24*30).apply(lambda x: x.mean()/(x.std(ddof=0)+1e-12)*np.sqrt(8760))`.

- Threshold Sensitivity
  - For grids of `t_long/t_short`, plot metric vs threshold (ACC/F1/Sharpe) to find robust ranges.

- Probability Diagnostics
  - Histograms of `P(up)`, `P(down)`; reliability (calibration) plot for the up-class (OVR bins).
  - Confusion matrix heatmap (normalized) to see class mix and flat-class filtering.

- Cross-Token Comparison
  - Bar charts for CAGR/Sharpe/HitRate; heatmaps for metrics across tokens and fees.

Minimal example (matplotlib)

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_equity_and_drawdown(pnl, title="", save_path=None):
    pnl = pnl.fillna(0)
    eq = (1 + pnl).cumprod()
    dd = eq / eq.cummax() - 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(eq.index, eq.values, label="Equity")
    ax1.axhline(1.0, color="gray", lw=1, ls="--")
    ax1.set_ylabel("Equity (x)")
    ax1.set_title(title or "Equity & Drawdown")
    ax1.legend(loc="best")

    ax2.fill_between(dd.index, dd.values, 0, color="tab:red", alpha=0.4)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Time")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.show()
```

Usage: after `backtest_from_proba(...)` returns `pnl`, call

```python
plot_equity_and_drawdown(pnl, title=f"{token} strategy", save_path=f"_exp/{run_id}/{token}_equity.png")
```

Note: use a timestamped `run_id` to group plots and CSVs for each run.

## Fees: With vs Without

Measure both gross (no fees) and net (with fees) performance. Use Binance spot fee tiers as reference scenarios:

- No-fee (gross): fee_bps = 0.0 (upper bound on strategy edge)
- VIP 4: fee_bps = 5.4 bps (0.054%)
- VIP 9: fee_bps = 2.0 bps (0.02%)

Implementation notes

- Per our backtest, fees are applied on position changes: `fee = fee_bps/1e4 * |Δposition|` per bar.
- Report both gross and net metrics:
  - Gross PnL: set `fee_bps=0.0`
  - Net PnL: set `fee_bps` to 5.4 or 2.0 bps and recompute.
- Include fees in sensitivity sweeps and the summary table (add columns or separate rows).

Example (v2 backtest)

```python
# Gross
pnl_gross, stats_gross = backtest_from_proba(xdf, proba, pred, classes, t_long=0.60, t_short=0.60, fee_bps=0.0)
# VIP 4
pnl_vip4,  stats_vip4  = backtest_from_proba(xdf, proba, pred, classes, t_long=0.60, t_short=0.60, fee_bps=5.4)
# VIP 9
pnl_vip9,  stats_vip9  = backtest_from_proba(xdf, proba, pred, classes, t_long=0.60, t_short=0.60, fee_bps=2.0)
```

Recommended reporting

- Add a small fee table per token: CAGR/Sharpe/HitRate under (0.0, 2.0, 5.4) bps.
- Plot three equity curves on the same chart for a quick visual of fee impact.

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
