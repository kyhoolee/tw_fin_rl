# -*- coding: utf-8 -*-
"""
Extra baselines on synchronized OHLCV bundles (no-leak, fees-aware):
- EW with volatility targeting (exponential vol, leverage cap)
- Time-series breakout (Donchian), long-only cash fallback
- Risk-parity (min-var static from rolling cov), periodic rebalance
- Momentum top-k with turnover cap

Conventions:
- All signals at t computed with data up to t-1.
- Execution at close[t].
- Costs modeled via per-side fee (bps) + slippage (bps).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from pathlib import Path

# --------- Common loaders & helpers (compatible with your existing module) ---------

def load_close_tensor(split_parquet_path: str):
    df = pd.read_parquet(split_parquet_path)
    N = df["symbol"].nunique()
    assert df["open_time"].nunique() * N == len(df), "Split must be synchronized per timestamp"
    piv = df.pivot(index="open_time", columns="symbol", values="Close").sort_index()
    syms = list(piv.columns)
    close = piv.to_numpy(dtype=np.float64)  # shape (T, N)
    ts = piv.index.to_pydatetime()
    return close, ts, syms

def apply_txn_costs(w_prev: np.ndarray, w_new: np.ndarray, fee_bps=7.5, slippage_bps=0.0) -> float:
    delta = float(np.abs(w_new - w_prev).sum())
    return delta * ((fee_bps + slippage_bps) / 1e4)

def _safe(x, eps=1e-12): return np.maximum(x, eps)

def rolling_vol(r: np.ndarray, half_life: int) -> np.ndarray:
    """Exponentially weighted stdev per asset; r shape (T, N)."""
    lam = np.exp(np.log(0.5) / half_life)
    T, N = r.shape
    var = np.zeros((T, N))
    for t in range(1, T):
        var[t] = lam * var[t-1] + (1-lam) * (r[t-1]**2)  # use t-1
    return np.sqrt(_safe(var))

def rolling_cov(close: np.ndarray, window: int) -> np.ndarray:
    """Return list of covariance matrices Σ_{t-1} (T, N, N) using log-returns up to t-1."""
    logp = np.log(close)
    r = np.diff(logp, axis=0)  # shape (T-1, N)
    Tm1, N = r.shape
    covs = np.zeros((Tm1, N, N))
    for t in range(window, Tm1+1):
        x = r[t-window:t]
        covs[t-1] = np.cov(x.T, bias=False)
    # pad early with last available (or identity if empty)
    last = np.eye(N)
    for t in range(Tm1):
        if not covs[t].any():
            covs[t] = last
        else:
            last = covs[t]
    # align to close index length T by pre-pending an identity row
    I = np.eye(N)
    covs_full = np.concatenate([I[None, ...], covs], axis=0)  # shape (T, N, N)
    return covs_full

def step_nav(nav_prev: float, w_prev: np.ndarray, ret_t: np.ndarray) -> float:
    return nav_prev * (1.0 + float((w_prev * ret_t).sum()))

# --------- Baselines ---------

def ew_vol_target(close: np.ndarray, ann_vol_target=0.20, hl=1440, fee_bps=7.5, slippage_bps=0.0,
                  leverage_cap=2.0, rebalance_every=60) -> np.ndarray:
    """
    Equal-weight then scale to target volatility using EWMA vol.
    - half-life `hl` in bars; annualization assumes 525600 minute-bars by default.
    """
    T, N = close.shape
    ret = close[1:] / close[:-1] - 1.0
    vol = rolling_vol(ret, half_life=hl)          # shape (T-1, N)
    nav = np.ones(T)
    w_prev = np.zeros(N)

    for t in range(1, T):
        # periodic rebalance
        if t % rebalance_every == 0:
            eq = np.ones(N) / N
            # per-asset risk ~ EWMA vol at t-1
            vol_vec = vol[t-1-1] if t-1-1 >= 0 else np.ones(N) * 1e-3
            port_vol = float(np.sqrt((eq**2 * (vol_vec**2)).sum()))
            scale = (ann_vol_target / _safe(port_vol)) if port_vol > 0 else 1.0
            scale = float(np.clip(scale, 0.0, leverage_cap))
            w_tgt = eq * scale
            cost = apply_txn_costs(w_prev, w_tgt, fee_bps, slippage_bps)
            nav[t-1] *= (1 - cost)
            w_prev = w_tgt
        # evolve NAV
        r_t = close[t] / close[t-1] - 1.0
        nav[t] = step_nav(nav[t-1], w_prev, r_t)
    return nav

def ts_breakout_donchian(close: np.ndarray, up_win=240, dn_win=240, fee_bps=7.5, slippage_bps=0.0,
                         rebalance_every=30) -> np.ndarray:
    """
    Time-series breakout long-only per asset:
    - Signal at t: long if close[t-1] >= max(close[t-1-up_win:t-1]), else 0.
    - Portfolio = equal-weight among signaled assets; cash if none.
    """
    T, N = close.shape
    nav = np.ones(T)
    w_prev = np.zeros(N)
    roll_max = pd.DataFrame(close).shift(1).rolling(up_win).max().to_numpy()
    for t in range(1, T):
        if t % rebalance_every == 0 and t-1 >= max(up_win, dn_win):
            signal = (close[t-1] >= roll_max[t-1]).astype(np.float64)
            if signal.sum() > 0:
                w_tgt = signal / signal.sum()
            else:
                w_tgt = np.zeros(N)
            cost = apply_txn_costs(w_prev, w_tgt, fee_bps, slippage_bps)
            nav[t-1] *= (1 - cost)
            w_prev = w_tgt
        r_t = close[t] / close[t-1] - 1.0
        nav[t] = step_nav(nav[t-1], w_prev, r_t)
    return nav

def risk_parity_minvar(close: np.ndarray, cov_win=720, fee_bps=7.5, slippage_bps=0.0,
                       rebalance_every=60, reg_eps=1e-6) -> np.ndarray:
    """
    Min-variance weights from rolling cov (t-1), normalized to sum 1 (long-only not enforced).
    """
    T, N = close.shape
    covs = rolling_cov(close, window=cov_win)   # (T, N, N), uses up to t-1
    nav = np.ones(T)
    w_prev = np.zeros(N)
    one = np.ones((N, 1))
    for t in range(1, T):
        if t % rebalance_every == 0:
            Sigma = covs[t-1].copy()
            Sigma[np.diag_indices(N)] += reg_eps
            try:
                inv = np.linalg.pinv(Sigma)
            except Exception:
                inv = np.eye(N)
            w = (inv @ one)
            w = (w / (one.T @ inv @ one)).ravel()  # minimum variance (sum=1)
            w_tgt = w.astype(np.float64)
            cost = apply_txn_costs(w_prev, w_tgt, fee_bps, slippage_bps)
            nav[t-1] *= (1 - cost)
            w_prev = w_tgt
        r_t = close[t] / close[t-1] - 1.0
        nav[t] = step_nav(nav[t-1], w_prev, r_t)
    return nav

def momentum_topk_turnover_capped(close: np.ndarray, lookback=360, topk=2, fee_bps=7.5,
                                  rebalance_every=60, slippage_bps=0.0, max_turnover=0.5) -> np.ndarray:
    """
    Cross-sectional momentum with a turnover cap:
    - Target weights = equal weight among top-k on trailing return t-1 / t-1-lookback.
    - If L1(w_tgt - w_prev) > max_turnover, move partially toward w_tgt.
    """
    T, N = close.shape
    nav = np.ones(T)
    w_prev = np.zeros(N)
    for t in range(1, T):
        if (t % rebalance_every == 0) and (t-1-lookback >= 0):
            past = (close[t-1] / close[t-1-lookback]).astype(np.float64)
            sel = np.argsort(-past)[:max(1, min(topk, N))]
            w_tgt = np.zeros(N); w_tgt[sel] = 1.0 / len(sel)
            l1 = float(np.abs(w_tgt - w_prev).sum())
            if l1 > max_turnover and l1 > 1e-9:
                alpha = max_turnover / l1
                w_adj = w_prev + alpha * (w_tgt - w_prev)
            else:
                w_adj = w_tgt
            cost = apply_txn_costs(w_prev, w_adj, fee_bps, slippage_bps)
            nav[t-1] *= (1 - cost)
            w_prev = w_adj
        r_t = close[t] / close[t-1] - 1.0
        nav[t] = step_nav(nav[t-1], w_prev, r_t)
    return nav

# --------- Batch runner ---------

def run_ext_baselines(split_parquet_path: str, fee_bps=7.5, slippage_bps=0.0):
    close, ts, syms = load_close_tensor(split_parquet_path)
    out = {}
    out["EW_VolTarget_hl1440"]     = ew_vol_target(close, ann_vol_target=0.20, hl=1440, fee_bps=fee_bps, slippage_bps=slippage_bps, leverage_cap=2.0, rebalance_every=60)
    out["TS_Breakout_240"]         = ts_breakout_donchian(close, up_win=240, dn_win=240, fee_bps=fee_bps, slippage_bps=slippage_bps, rebalance_every=30)
    out["RiskParity_MinVar_720"]   = risk_parity_minvar(close, cov_win=720, fee_bps=fee_bps, slippage_bps=slippage_bps, rebalance_every=60)
    out["MOM_Top2_Turnover50%_60"] = momentum_topk_turnover_capped(close, lookback=360, topk=2, fee_bps=fee_bps,
                                                                   rebalance_every=60, slippage_bps=slippage_bps, max_turnover=0.5)
    return syms, ts, out
