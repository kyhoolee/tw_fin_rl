# -*- coding: utf-8 -*-
"""Portfolio baselines for 4 synchronized tokens (no-leak):
- EW equal-weight (periodic rebalance)
- Momentum top-k (cross-sectional)
- Mean-reversion pair spread (delta-neutral)

Assumptions:
- Signals at t are computed using data up to t-1.
- Execution occurs at close[t].
- Fees model: Binance taker per-side fee (default 7.5 bps = 0.075%).
"""
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- Loader ----------

def load_tensor(split_path, cols=("Open","High","Low","Close","Volume")):
    df = pd.read_parquet(split_path)
    N = df["symbol"].nunique()
    assert df["open_time"].nunique() * N == len(df), "Split is not per-timestamp synchronized"
    piv = df.pivot(index="open_time", columns="symbol", values=list(cols)).sort_index()
    symbols = sorted(df["symbol"].unique())
    features = list(cols)
    piv = piv.reindex(sorted(piv.columns, key=lambda c: (c[1], c[0])), axis=1)
    T, N, F = len(piv), len(symbols), len(features)
    x = piv.to_numpy().reshape(T, N, F).astype(np.float32)
    return x, piv.index.to_pydatetime(), symbols, features

# ---------- Costs & helpers ----------

def safe_div(a, b, eps=1e-12):
    return a / (b + eps)

def price_to_ret(close):
    r = np.zeros_like(close, dtype=np.float64)
    r[1:] = safe_div(close[1:], close[:-1]) - 1.0
    return r

def apply_txn_costs(weight_prev, weight_new, fee_bps=7.5, slippage_bps=0.0):
    """
    Binance-style per-side fee model.
    - delta L1 = sum |w_new - w_prev| (buys + sells as fraction of NAV)
    - cost = delta * (fee_rate + slippage_rate)
    """
    delta = float(np.abs(weight_new - weight_prev).sum())
    fee_rate = fee_bps / 1e4
    slip_rate = slippage_bps / 1e4
    return delta * (fee_rate + slip_rate)

# ---------- Metrics ----------

def perf_metrics(nav, dt_index=None):
    nav = np.asarray(nav, dtype=np.float64)
    rets = np.diff(nav) / (nav[:-1] + 1e-12)
    if dt_index is None:
        ann = 525600.0  # minute bars per ~year
    else:
        dt = pd.to_datetime(dt_index)
        minutes = (dt[-1] - dt[0]).total_seconds() / 60.0
        ann = max(minutes, 1.0)
    mean = rets.mean()
    std  = rets.std(ddof=0) + 1e-12
    sharpe = mean / std * np.sqrt(ann)
    cagr = nav[-1] - 1.0
    dd = 1.0 - safe_div(nav, np.maximum.accumulate(nav))
    maxdd = dd.max()
    return {"CAGR≈": float(cagr), "Sharpe≈": float(sharpe), "MaxDD": float(maxdd)}

# ---------- Baselines ----------

def ew_baseline(close, fee_bps=7.5, rebalance_every=1440, slippage_bps=0.0):
    T, N = close.shape
    nav = np.ones(T, dtype=np.float64)
    w_prev = np.ones(N) / N
    last_rb = 0
    for t in range(1, T):
        if (t - last_rb) % rebalance_every == 0:
            w_tgt = np.ones(N) / N
            cost = apply_txn_costs(w_prev, w_tgt, fee_bps, slippage_bps)
            nav[t-1] *= (1 - cost)
            w_prev = w_tgt
            last_rb = t
        r_t = close[t] / close[t-1] - 1.0
        nav[t] = nav[t-1] * (1.0 + (w_prev * r_t).sum())
    return nav

def momentum_topk(close, lookback=360, topk=2, fee_bps=7.5, rebalance_every=60, slippage_bps=0.0):
    T, N = close.shape
    nav = np.ones(T, dtype=np.float64)
    w_prev = np.zeros(N, dtype=np.float64)
    for t in range(1, T):
        if (t % rebalance_every == 0) and (t - 1 - lookback >= 0):
            past = safe_div(close[t-1], close[t-1-lookback])
            ranks = np.argsort(-past)
            sel = ranks[:max(1, min(topk, N))]
            w_tgt = np.zeros(N, dtype=np.float64)
            w_tgt[sel] = 1.0 / len(sel)
            cost = apply_txn_costs(w_prev, w_tgt, fee_bps, slippage_bps)
            nav[t-1] *= (1 - cost)
            w_prev = w_tgt
        r_t = close[t] / close[t-1] - 1.0
        nav[t] = nav[t-1] * (1.0 + (w_prev * r_t).sum())
    return nav

def pair_meanrev_spread(close, pair=(0,1), z_window=180, entry_z=1.5, exit_z=0.5,
                        fee_bps=7.5, slippage_bps=0.0, max_gross=1.0):
    i, j = pair
    logp = np.log(close)
    T = close.shape[0]
    nav = np.ones(T, dtype=np.float64)
    pos_i = 0.0; pos_j = 0.0
    for t in range(1, T):
        if t - z_window >= 1:
            s = logp[:t, i] - logp[:t, j]  # up to t-1
            m = pd.Series(s).rolling(z_window, min_periods=z_window).mean().iloc[-1]
            v = pd.Series(s).rolling(z_window, min_periods=z_window).std(ddof=0).iloc[-1] + 1e-12
            z = (s[-1] - m) / v
            w_i_new = w_j_new = 0.0
            if abs(z) > entry_z:
                if z > 0:  # i rich vs j → short i, long j
                    w_i_new, w_j_new = -max_gross/2, +max_gross/2
                else:
                    w_i_new, w_j_new = +max_gross/2, -max_gross/2
            elif abs(z) < exit_z:
                w_i_new = w_j_new = 0.0
            else:
                w_i_new, w_j_new = pos_i, pos_j
            cost = apply_txn_costs(np.array([pos_i, pos_j]), np.array([w_i_new, w_j_new]), fee_bps, slippage_bps)
            nav[t-1] *= (1 - cost)
            pos_i, pos_j = w_i_new, w_j_new
        r_i = close[t, i] / close[t-1, i] - 1.0
        r_j = close[t, j] / close[t-1, j] - 1.0
        nav[t] = nav[t-1] * (1.0 + pos_i * r_i + pos_j * r_j)
    return nav

# ---------- Convenience ----------

def run_all_baselines(split_parquet_path, fee_bps=7.5, slippage_bps=0.0):
    x, ts, syms, feats = load_tensor(split_parquet_path, cols=("Close",))
    close = x[:, :, 0].astype(np.float64)
    out = {}
    out["EW_daily"] = ew_baseline(close, fee_bps=fee_bps, rebalance_every=1440, slippage_bps=slippage_bps)
    out["MOM_top2_look360_rb60"] = momentum_topk(close, lookback=360, topk=2, fee_bps=fee_bps, rebalance_every=60, slippage_bps=slippage_bps)
    out["PAIR_MR_BTC_ETH"] = pair_meanrev_spread(close, pair=(syms.index("BTCUSDT"), syms.index("ETHUSDT")),
                                                 z_window=180, entry_z=1.5, exit_z=0.5, fee_bps=fee_bps, slippage_bps=slippage_bps, max_gross=1.0)
    metrics = {k: perf_metrics(nav, dt_index=ts) for k, nav in out.items()}
    return syms, out, metrics