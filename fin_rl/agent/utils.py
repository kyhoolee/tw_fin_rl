import yaml
import numpy as np
import pandas as pd

def perf_metrics(nav, dt_index=None):
    nav = np.asarray(nav, dtype=np.float64)
    rets = np.diff(nav) / (nav[:-1] + 1e-12)
    if dt_index is not None:
        minutes = (pd.to_datetime(dt_index[-1]) - pd.to_datetime(dt_index[0])).total_seconds()/60.0
        ann_fac = max(minutes, 1.0)
    else:
        ann_fac = 525600.0
    mean, std = rets.mean(), rets.std(ddof=0) + 1e-12
    sharpe = mean / std * np.sqrt(ann_fac)
    cagr = nav[-1] - 1.0
    roll_max = np.maximum.accumulate(nav)
    mdd = float((1 - nav/roll_max).max())
    # Sortino & Calmar (optional)
    downside = rets[rets < 0]
    sortino = mean / (downside.std(ddof=0) + 1e-12) * np.sqrt(ann_fac)
    calmar = (cagr + 1e-12) / (mdd + 1e-12)
    return {"CAGR≈": float(cagr), "Sharpe≈": float(sharpe), "Sortino≈": float(sortino), "MaxDD": mdd, "Calmar≈": float(calmar)}

def l1_turnover(weights: np.ndarray) -> float:
    """
    L1 turnover across the backtest (sum of |Δw|).
    weights shape (T, N) with NaN where not rebalanced; fill forward before pass-in.
    """
    w = np.nan_to_num(weights, nan=0.0)
    return float(np.abs(np.diff(w, axis=0)).sum())


def load_fee_cfg(path: str, name: str | None = None):
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    if name is None:
        name = y.get("default")
    cfg = y["presets"][name]
    return float(cfg["fee_bps"]), float(cfg.get("slippage_bps", 0.0))