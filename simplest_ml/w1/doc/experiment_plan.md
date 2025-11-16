# Next Experiments — Component-Wise Plan

This plan enumerates concrete experiments per component (labels, features, normalization, modeling, validation/backtest). Each item includes: goal, design, key params, success criteria, and notes to implement in `multi_token_modeling.py`.

## 1) Targets / Labels

- H-step Direction (noise-filtered)
  - Goal: 
    - Reduce label noise at 1h.
  - Design: 
    - y = sign(sum_{i=1..H} ret_{t+i}); 
    - ignore if |sum_ret| ≤ ε.
  - Params: 
    - H ∈ {4, 12}, ε ∈ {0.05%, 0.10%, 0.20%}.
  - Success: 
    - +AUC ≥ +0.02 vs baseline on ≥50% tokens; 
    - improved F1(1).

  - Notes: 
    - Add `make_label_horizon(H, eps)` 
    - and hook into `prepare_Xy()`.

- 3-Class Up/Flat/Down
  - Goal: Model uncertainty around tiny moves.
  - Design: y ∈ {-1,0,+1} with band ε around 0; use one-vs-rest or reduce to binary by dropping flats.
  - Params: ε ∈ {0.05%, 0.15%}.
  - Success: Higher precision on non-flat trades; backtest hit-rate ↑.
  - Notes: Keep evaluation comparable by mapping back to binary for AUC when needed.

- Triple-Barrier Labeling
  - Goal: Path-aware profit-taking/stop-loss labels.
  - Design: TP = a·σ, SL = b·σ, timeout = H; σ = rolling std of returns.
  - Params: a ∈ {1.0,1.5}, b ∈ {1.0}, H ∈ {12,24}.
  - Success: Backtest metrics after fees (Sharpe, CAGR) beat baseline logic.
  - Notes: Requires small helper to scan forward within H bars.

## 2) Features

- Candle Anatomy
  - Body = |close-open|; Range = high-low; Upper/Lower wick; Body/Range; Gap = open - prev_close.
  - Windows: w ∈ {3,6,12,24}; lag by 1.

- Volatility & Range
  - ATR(w), 
  - Bollinger bandwidth (2σ), 
  - realized vol (std of ret_1 over w), 
  - Parkinson high-low vol.

- Momentum & Trend
  - MACD(12,26,9): 
    - macd, 
    - signal, 
    - hist; 
  - ADX/DI±(14); 
  - CCI(20); 
  - Stoch %K/%D(14,3); 
  - Williams %R(14).

- Volume & Flow
  - OBV; 
  - volume z-score vs rolling mean/std; 
  - spike flags (volume_z > 2).

- Time Features
  - Hour-of-day, 
  - day-of-week as sin/cos cyclic encodings; 
  - holiday/weekend flag if relevant.

- Cross-Asset Context
  - BTC ret/vol features joined to all tokens; 
  - alt/BTC relative return; 
  - simple beta vs BTC over rolling w.

- Success: 
  - +AUC ≥ +0.02 on median token 
  - or improved backtest metrics with thresholds.

- Notes: 
  - Implement as optional blocks in `add_features(..., use_candle=True, use_vol=True, ...)` 
  - and ensure all features are lagged (no leakage).

## 3) Normalization & Regularization

- Rolling Z-Score per Feature
  - Design: z_t = (x_t - mean_{t-w}) / std_{t-w}, computed inside each train fold.
  - Params: w ∈ {48, 168} (2d, 7d hours).
  - Success: Stabilized training; AUC variance ↓ across folds.

- Robust Scaling
  - Design: median/MAD scaling or quantile transform per fold.

- Feature Selection / Shrinkage
  - L1 Logistic (C grid) and/or tree feature importance pruning.

## 4) Modeling

- Logistic Tweaks
  - `class_weight='balanced'`, penalty ∈ {l2, l1}, C ∈ {0.25, 0.5, 1.0}.

- Gradient Boosted Trees
  - LightGBM/XGBoost classifier; 
    - shallow trees; 
    - early stopping on fold.

  - Params: 
    - num_leaves ∈ {15, 31}, depth ∈ {3, 5}, 
    - learning_rate ∈ {0.03, 0.1}.

  - Success: AUC gain ≥ +0.02 median token.

- Threshold Tuning & Calibration
  - Tune threshold per token using validation (maximize F1 or Youden J).
  - Calibrate probabilities (Platt/Isotonic) if beneficial.

## 5) Validation & Backtest

- Walk-Forward / Expanding Window
  - Emphasize recent periods; 
  - report last-split OOS separately.

- Backtest with Costs
  - Rule: 
    - long if p > t_hi, 
    - short if p < t_lo; 
    - else flat.

  - Params: 
    - t_hi ∈ {0.55,0.60}, 
    - t_lo ∈ {0.45,0.40}; 
    - fee ∈ {4,10} bps; 
    - min holding H_min ∈ {1,4}.

  - Metrics: 
    - CAGR, 
    - Sharpe, 
    - maxDD, 
    - hit-rate, 
    - turnover; 
  - per-token and pooled.

## 6) Experiment Management

- Logging
  - Persist per-token 
    - metrics, 
    - thresholds, 
    - params 
  - to CSV/JSON in `simplest_ml/w1/_exp/`.

- Reproducibility
  - Fix seeds; 
  - record library versions; 
  - store config used for each run.

- Reporting
  - Auto-generate PrettyTable summary 
  - and minimal plots (equity curve, ROC) per token.

## 7) Prioritization (Suggested Order)

1) Labels: 
  - H-step + ε filter; 
  - Threshold tuning.

2) Features: 
  - Candle anatomy + volatility block; 
  - rolling z-scores.

3) Modeling: 
  - LightGBM baseline; 
  - class_weight for LogReg.

4) Backtest: 
  - basic thresholds with fees.

5) Extend to triple-barrier if promising.

## Implementation Hooks

- `prepare_Xy(...)`: add args `label_h=1, label_eps=0.0, label_mode={'direction','triple_barrier'}`.
- `add_features(...)`: flags for feature blocks and `zscore_window`.
- `train_timeseries_*`: return tuned threshold; optionally calibrated model.
- New: `backtest_from_proba(...)` with fees and thresholds.

