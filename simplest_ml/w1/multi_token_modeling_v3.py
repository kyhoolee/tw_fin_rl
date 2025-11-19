"""
Multi‑token modeling — v3

This version implements the ideas captured in the docs:
- Three‑class labeling with horizon H and epsilon band (reduce noise)
- Expanded, leakage‑safe features (candle/volatility/volume/time + rolling z‑scores)
- TimeSeriesSplit with multinomial LogisticRegression
- Out‑of‑fold (OOS) probability matrix retained for honest evaluation/backtest
- Threshold tuning on validation (grid over t_long/t_short)
- OOS backtest only (no refit bias) with fee scenarios (gross/VIP9/VIP4)
- Chart artifacts: equity curves for different fees, buy‑and‑hold baseline, shaded OOS windows

The focus of comments is to make the flow and design choices explicit for review and iteration.
"""

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score

try:
    from .pretty_helpers import (
        DEFAULT_TOKEN_TO_SYMBOL,
        load_token_df,
        pretty_print_sample,
    )
except Exception:
    import sys as _sys, os as _os
    _sys.path.append(_os.path.dirname(__file__))
    from pretty_helpers import (
        DEFAULT_TOKEN_TO_SYMBOL,
        load_token_df,
        pretty_print_sample,
    )


@dataclass
class TokenResult:
    """Per‑token summary for one pipeline run.

    Fields capture classification quality (macroAUC/ACC), tuned thresholds,
    and fee‑aware backtest results so we can compare across runs.
    """
    token: str
    symbol: str
    rows: int
    macro_auc: float
    accuracy: float
    precision_up: float
    recall_up: float
    t_long: float
    t_short: float
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    # Fee scenarios
    cagr_gross: Optional[float] = None
    sharpe_gross: Optional[float] = None
    cagr_vip9: Optional[float] = None
    sharpe_vip9: Optional[float] = None
    cagr_vip4: Optional[float] = None
    sharpe_vip4: Optional[float] = None


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    """Leakage‑safe rolling z‑score.

    We shift the rolling mean/std by 1 bar so the current row only uses
    past information. This stabilizes distributions under non‑stationarity.
    """
    mean = s.rolling(window).mean().shift(1)
    std = s.rolling(window).std(ddof=0).shift(1)
    return (s - mean) / (std + 1e-9)


def add_features(
    df: pd.DataFrame,
    windows: Sequence[int] = (3, 6, 12, 24),
    *,
    verbose: bool = False,
    log_prefix: str = "",
    use_candle: bool = True,
    use_volatility: bool = True,
    use_momentum: bool = True,
    use_volume: bool = True,
    use_time: bool = True,
    zscore_windows: Sequence[int] = (48,),
) -> Tuple[pd.DataFrame, List[str]]:
    """Build the feature matrix from raw OHLCV.

    Blocks (all lagged by 1 at the end to avoid leakage):
    - Core returns/moving averages/volatility/min/max/RSI‑like
    - Candle anatomy (body/range/wicks/gap/true range)
    - Volatility & range indicators (ATR, BB width, realized vol)
    - Volume/flow (volume z‑scores, OBV)
    - Time features (hour/day sin/cos)
    - Rolling z‑scores for stability
    """
    if verbose:
        print(f"{log_prefix}add_features: start, rows={len(df)}, windows={tuple(windows)}")

    x = df.copy()
    x["ret_1"] = x["close"].pct_change(1)

    for w in windows:
        x[f"ret_{w}"] = x["close"].pct_change(w)
        x[f"sma_{w}"] = x["close"].rolling(w).mean()
        x[f"ema_{w}"] = x["close"].ewm(span=w, adjust=False).mean()
        x[f"vol_{w}"] = x["ret_1"].rolling(w).std()
        x[f"max_{w}"] = x["high"].rolling(w).max()
        x[f"min_{w}"] = x["low"].rolling(w).min()
        up = np.clip(x["ret_1"], 0, None)
        down = np.clip(-x["ret_1"], 0, None)
        rs = up.rolling(w).mean() / (down.rolling(w).mean() + 1e-9)
        x[f"rsi_{w}"] = 100 - 100 / (1 + rs)

    if use_candle:
        body = (x["close"] - x["open"]).abs()
        rng = (x["high"] - x["low"]).clip(lower=1e-12)
        upper = (x["high"] - x[["open", "close"]].max(axis=1)).clip(lower=0)
        lower = (x[["open", "close"]].min(axis=1) - x["low"]).clip(lower=0)
        x["c_body"] = body
        x["c_range"] = rng
        x["c_body_ratio"] = body / rng
        x["c_upper"] = upper
        x["c_lower"] = lower
        tr = (x["high"] - x["low"]).combine(
            (x["high"] - x["close"].shift(1)).abs(), max
        ).combine((x["low"] - x["close"].shift(1)).abs(), max)
        x["c_true_range"] = tr
        x["c_gap"] = x["open"] - x["close"].shift(1)

    if use_volatility:
        for w in windows:
            tr = (x["high"] - x["low"]).combine(
                (x["high"] - x["close"].shift(1)).abs(), max
            ).combine((x["low"] - x["close"].shift(1)).abs(), max)
            x[f"atr_{w}"] = tr.rolling(w).mean()
            ma = x["close"].rolling(w).mean()
            sd = x["close"].rolling(w).std()
            x[f"bb_width_{w}"] = (2 * sd) / (ma.abs() + 1e-9)
            x[f"rv_{w}"] = x["ret_1"].rolling(w).std()

    if use_volume:
        for w in windows:
            v_ma = x["volume"].rolling(w).mean()
            v_sd = x["volume"].rolling(w).std()
            x[f"vol_z_{w}"] = (x["volume"] - v_ma) / (v_sd + 1e-9)
        direction = np.sign(x["close"].diff()).fillna(0.0)
        x["obv"] = (direction * x["volume"]).cumsum()

    if use_time and isinstance(x.index, pd.DatetimeIndex):
        hr = x.index.hour
        dw = x.index.dayofweek
        x["hr_sin"] = np.sin(2 * np.pi * hr / 24)
        x["hr_cos"] = np.cos(2 * np.pi * hr / 24)
        x["dw_sin"] = np.sin(2 * np.pi * dw / 7)
        x["dw_cos"] = np.cos(2 * np.pi * dw / 7)

    # Collect base feature names
    feature_cols = [
        c
        for c in x.columns
        if any(k in c for k in ["ret_", "sma_", "ema_", "vol_", "max_", "min_", "rsi_"])
    ]
    for k in [
        "c_body",
        "c_range",
        "c_body_ratio",
        "c_upper",
        "c_lower",
        "c_true_range",
        "c_gap",
        "atr_",
        "bb_width_",
        "rv_",
        "vol_z_",
        "obv",
        "hr_",
        "dw_",
    ]:
        feature_cols += [c for c in x.columns if c.startswith(k)]

    # A small set of base series to z‑score; expands to columns z_<name>_<win>
    base_for_z = {
        "close": x["close"],
        "ret_1": x["ret_1"],
        "c_body": x.get("c_body"),
        "c_range": x.get("c_range"),
        "volume": x["volume"],
    }
    for w in zscore_windows:
        for name, series in base_for_z.items():
            if series is None:
                continue
            x[f"z_{name}_{w}"] = _rolling_zscore(series, w)
            feature_cols.append(f"z_{name}_{w}")

    for c in feature_cols:
        x[c] = x[c].shift(1)

    if verbose:
        na_counts = int(x[feature_cols].isna().sum().sum())
        print(
            f"{log_prefix}add_features: generated {len(feature_cols)} features; total NaNs after lag={na_counts}"
        )
    return x, feature_cols


def make_label_three_class(
    df: pd.DataFrame,
    *,
    H: int = 4,
    eps: float = 0.001,
    verbose: bool = False,
    log_prefix: str = "",
) -> pd.Series:
    """Three‑class target based on an H‑step forward return.

    - y = +1  if ret_fwd >  eps
    - y =  0  if |ret_fwd| <= eps (tiny move → ambiguous/no‑trade)
    - y = −1  if ret_fwd < −eps
    """
    ret_fwd = df["close"].shift(-H) / df["close"] - 1.0
    y = pd.Series(index=df.index, dtype=int)
    y[ret_fwd > eps] = 1
    y[ret_fwd < -eps] = -1
    y[(ret_fwd >= -eps) & (ret_fwd <= eps)] = 0
    if verbose:
        counts = y.value_counts().to_dict()
        print(f"{log_prefix}make_label(3-class): H={H} eps={eps} counts={counts}")
    return y


def prepare_Xy(
    df: pd.DataFrame,
    windows: Sequence[int] = (3, 6, 12, 24),
    *,
    verbose: bool = False,
    log_prefix: str = "",
    label_h: int = 4,
    label_eps: float = 0.001,
    feature_kwargs: Optional[dict] = None,
    extra_hourly_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """Feature/label preparation with leakage‑safe transforms.

    Returns the numpy arrays plus the aligned pandas frame `xdf` so we can
    compute metrics/backtests and draw charts on the original time index.
    """
    if verbose:
        idx_info = (
            f"[{df.index.min()} → {df.index.max()}]" if isinstance(df.index, pd.DatetimeIndex) else ""
        )
        print(f"{log_prefix}prepare_Xy: input rows={len(df)} {idx_info}")

    feature_kwargs = feature_kwargs or {}
    x, feature_cols = add_features(
        df,
        windows=windows,
        verbose=verbose,
        log_prefix=log_prefix,
        **feature_kwargs,
    )
    # Join extra precomputed hourly features (e.g., minute->hour aggregates)
    if extra_hourly_df is not None and not extra_hourly_df.empty:
        try:
            ex = extra_hourly_df.reindex(x.index)
            ex = ex.shift(1)  # safety lag
            ex.columns = [f"mf_{c}" for c in ex.columns]
            x = pd.concat([x, ex], axis=1)
            feature_cols += list(ex.columns)
            if verbose:
                print(f"{log_prefix}prepare_Xy: added minute features: {ex.shape}")
        except Exception as e:
            if verbose:
                print(f"{log_prefix}prepare_Xy: minute feature join failed: {e}")
    y = make_label_three_class(x, H=label_h, eps=label_eps, verbose=verbose, log_prefix=log_prefix)
    before_drop = len(x)
    x = x.dropna().copy()
    y = y.loc[x.index]
    X = x[feature_cols].values
    y = y.values
    if verbose:
        print(
            f"{log_prefix}prepare_Xy: dropped NaNs -> rows {before_drop} → {len(x)}; features={len(feature_cols)}"
        )
    return X, y, feature_cols, x


def train_timeseries_logreg(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    C: float = 0.5,
    max_iter: int = 2000,
    *,
    verbose: bool = False,
    log_prefix: str = "",
    index: Optional[Sequence] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, List[Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]]:
    """TimeSeriesSplit + StandardScaler + multinomial LogisticRegression.

    We fit per fold, predict on the contiguous test window, and stitch
    together an out‑of‑fold (OOS) probability matrix. This is our honest
    input for backtesting. We also record the time span of each test fold
    for chart shading.

    Returns
    - oof_pred: int array of predicted classes for every row
    - oof_proba_full: [n_samples, n_classes] probabilities (aligned to `classes`)
    - classes: sorted unique label values (expected [-1,0,1])
    - macro_auc: macro one‑vs‑rest AUC on OOF probs
    - accuracy: overall OOF accuracy
    - fold_ranges: list[(start_ts, end_ts)] for each fold's test window
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    n = len(y)
    classes = np.unique(y)
    n_classes = len(classes)
    oof_pred = np.zeros(n, dtype=int)
    oof_proba_full = np.zeros((n, n_classes), dtype=float)
    fold_ranges: List[Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        if verbose:
            print(f"{log_prefix}train: fold {fold}/{n_splits} | train={len(train_idx)} test={len(test_idx)}")
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        clf = LogisticRegression(max_iter=max_iter, C=C, n_jobs=None, multi_class="multinomial", solver="lbfgs")
        clf.fit(Xtr, ytr)

        proba = clf.predict_proba(Xte)
        pred = clf.predict(Xte)

        # Map proba columns to our sorted classes order
        clf_classes = clf.classes_.tolist()
        for j, c in enumerate(classes):
            oof_proba_full[test_idx, j] = proba[:, clf_classes.index(c)]
        oof_pred[test_idx] = pred

        if verbose:
            try:
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(yte, classes=classes)
                fold_auc = roc_auc_score(y_bin, proba[:, [clf_classes.index(c) for c in classes]], average="macro", multi_class="ovr")
            except Exception:
                fold_auc = float("nan")
            fold_acc = float((pred == yte).mean())
            print(f"{log_prefix}train: fold {fold} metrics | macroAUC={fold_auc:.4f} ACC={fold_acc:.4f}")

        # Record OOS time span for visualization (TimeSeriesSplit gives contiguous test)
        if index is not None:
            try:
                start_ts = index[test_idx[0]]
                end_ts = index[test_idx[-1]]
            except Exception:
                start_ts = None
                end_ts = None
            fold_ranges.append((start_ts, end_ts))

    # Macro AUC on OOF
    try:
        from sklearn.preprocessing import label_binarize
        y_bin_all = label_binarize(y, classes=classes)
        macro_auc = float(roc_auc_score(y_bin_all, oof_proba_full, average="macro", multi_class="ovr"))
    except Exception:
        macro_auc = float("nan")
    acc = float((oof_pred == y).mean())
    if verbose:
        print(f"{log_prefix}train: OOF metrics | macroAUC={macro_auc:.4f} ACC={acc:.4f}")
    return oof_pred, oof_proba_full, classes, macro_auc, acc, fold_ranges


def tune_thresholds_three_class(
    proba: np.ndarray,
    y: np.ndarray,
    classes: Sequence[int],
    *,
    grid: Sequence[float] = (0.55, 0.60, 0.65),
) -> Tuple[float, float, Dict[str, float]]:
    classes = list(classes)
    idx_up = classes.index(1)
    idx_dn = classes.index(-1)
    best = (0.60, 0.60)
    best_score = -1.0
    for t_long in grid:
        for t_short in grid:
            pred = np.zeros_like(y)
            p_up = proba[:, idx_up]
            p_dn = proba[:, idx_dn]
            pred[p_up >= t_long] = 1
            pred[p_dn >= t_short] = -1
            # F1 for up and down; ignore zeros
            f1_up = f1_score(y == 1, pred == 1, zero_division=0)
            f1_dn = f1_score(y == -1, pred == -1, zero_division=0)
            score = 0.5 * (f1_up + f1_dn)
            if score > best_score:
                best_score = score
                best = (t_long, t_short)
    return best[0], best[1], {"f1_avg": best_score}


def _equity_metrics(pnl: pd.Series, index: Optional[pd.DatetimeIndex]) -> Tuple[float, float]:
    eq = (1 + pnl.fillna(0)).cumprod()
    if index is not None and len(index) > 1:
        dt_days = (index[-1] - index[0]).total_seconds() / (3600 * 24)
        years = dt_days / 365.0 if dt_days > 0 else np.nan
    else:
        years = np.nan
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if years and years > 0 else float("nan")
    ret = pnl.fillna(0)
    ann = 8760.0
    sharpe = float(ret.mean() / (ret.std(ddof=0) + 1e-12) * np.sqrt(ann)) if ret.std(ddof=0) > 0 else float("nan")
    return cagr, sharpe


def backtest_oof_from_proba(
    xdf: pd.DataFrame,
    proba: np.ndarray,
    classes: Sequence[int],
    *,
    t_long: float,
    t_short: float,
    fee_bps: float,
) -> Tuple[pd.Series, Dict[str, float], pd.Series]:
    """Construct an OOS position series from probabilities and compute PnL.

    Positions:
    - long if P(up) >= t_long
    - short if P(down) >= t_short
    - flat otherwise

    Fees are charged on position changes only to approximate spot maker/taker
    costs at the bar level (simplification). Returns a per‑bar PnL series and
    summary metrics (CAGR/Sharpe).
    """
    classes = list(classes)
    idx_up = classes.index(1)
    idx_dn = classes.index(-1)
    p_up = proba[:, idx_up]
    p_dn = proba[:, idx_dn]
    pos = pd.Series(0.0, index=xdf.index)
    pos[p_up >= t_long] = 1.0
    pos[p_dn >= t_short] = -1.0
    ret_1 = xdf["close"].pct_change().fillna(0.0)
    trade = pos.diff().abs().fillna(0.0)
    fee = (fee_bps / 1e4) * trade
    pnl = pos.shift(1).fillna(0.0) * ret_1 - fee
    cagr, sharpe = _equity_metrics(pnl, xdf.index if isinstance(xdf.index, pd.DatetimeIndex) else None)
    # Extra trade diagnostics
    opens = ((pos.shift(1).fillna(0.0) == 0) & (pos != 0)).sum()
    changes = (trade > 0).sum()
    # average holding bars for non-zero position segments
    seg_change = pos.ne(pos.shift(1))
    idxs = np.flatnonzero(seg_change.values)
    if len(idxs) > 0:
        lens = np.diff(np.r_[idxs, len(pos)])
        holds = [l for i, l in zip(idxs, lens) if pos.iloc[i] != 0]
        avg_hold = float(np.mean(holds)) if holds else 0.0
    else:
        avg_hold = 0.0
    if isinstance(xdf.index, pd.DatetimeIndex) and len(xdf.index) > 1:
        days = (xdf.index[-1] - xdf.index[0]).days or 1
        trades_per_day = float(opens) / days
    else:
        trades_per_day = float("nan")
    turnover = float(trade.mean())
    stats = {
        "cagr": cagr,
        "sharpe": sharpe,
        "trades": float(opens),
        "changes": float(changes),
        "avg_hold": avg_hold,
        "trades_per_day": trades_per_day,
        "turnover": turnover,
    }
    return pnl, stats, pos


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _save_pnl_artifacts(
    token: str,
    index: pd.DatetimeIndex,
    pnl_map: Dict[str, pd.Series],
    out_dir: str,
    run_id: str,
    *,
    title_suffix: str = "",
    fold_ranges: Optional[List[Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]]] = None,
    price: Optional[pd.Series] = None,
    position: Optional[pd.Series] = None,
    stats_by_fee: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """Save PnL CSV and equity chart.

    - CSV columns: keys of `pnl_map` (e.g., gross/vip9/vip4)
    - Equity plot: overlays fee scenarios, optional buy‑and‑hold baseline,
      and shades OOS test windows for clarity.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib not available: {e}")
        return

    out_base = os.path.join(out_dir, run_id)
    _ensure_dir(out_base)

    # Save CSV
    df = pd.DataFrame({k: v.reindex(index).fillna(0.0) for k, v in pnl_map.items()}, index=index)
    csv_path = os.path.join(out_base, f"{token}_pnl.csv")
    df.to_csv(csv_path)

    # Plot equity curves (top) and optional trade frequency (bottom)
    if position is not None and isinstance(index, pd.DatetimeIndex):
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax2 = None
    for k, s in pnl_map.items():
        eq = (1 + s.reindex(index).fillna(0.0)).cumprod()
        ax.plot(eq.index, eq.values, label=k)
    # Optional buy & hold baseline (start at first OOS fold start for fair comparison)
    if price is not None:
        try:
            px = price.reindex(index).ffill()
            # Determine first OOS start timestamp
            oos_start = None
            if fold_ranges:
                for (s_ts, e_ts) in fold_ranges:
                    if s_ts is not None:
                        oos_start = s_ts
                        break
            # Normalize hold from first OOS start; don't plot earlier part
            if oos_start is None:
                hold_base = float(px.iloc[0]) if len(px) else 1.0
                eq_hold = px / hold_base
                hold_x = index
            else:
                # Find base price at or immediately after oos_start
                try:
                    hold_base = float(px.loc[oos_start])
                except KeyError:
                    # use first index >= oos_start
                    mask = px.index >= oos_start
                    hold_base = float(px[mask].iloc[0]) if mask.any() else float(px.iloc[0])
                eq_hold = (px / hold_base).where(px.index >= oos_start)
                hold_x = eq_hold.index
            ax.plot(hold_x, eq_hold.values, color="black", ls=":", lw=1.5, label="hold")
        except Exception:
            pass
    ax.axhline(1.0, color="gray", lw=1, ls="--")
    ax.set_title(f"{token} Equity Curves {title_suffix}".strip())
    ax.set_ylabel("Equity (x)")
    ax.legend(loc="best")
    # Shade OOS test fold ranges
    if fold_ranges:
        for (s_ts, e_ts) in fold_ranges:
            if s_ts is None or e_ts is None:
                continue
            ax.axvspan(s_ts, e_ts, color="k", alpha=0.08)
    fig.tight_layout()
    # Bottom panel: trade frequency and a small text box of metrics
    if ax2 is not None and position is not None and isinstance(index, pd.DatetimeIndex):
        trades_daily = (position.diff().abs() > 0).astype(int).reindex(index).resample('1D').sum()
        ax2.bar(trades_daily.index, trades_daily.values, width=1.0, color='tab:gray', alpha=0.7)
        ax2.set_ylabel("Trades/day")
        # Add text metrics
        if stats_by_fee:
            lines = []
            tf = stats_by_fee.get('trade', {})
            lines.append(f"trades={int(tf.get('trades', 0))}, avg_hold={tf.get('avg_hold', float('nan')):.1f} bars, t/day={tf.get('trades_per_day', float('nan')):.2f}")
            g = stats_by_fee.get('gross', {})
            v9 = stats_by_fee.get('vip9', {})
            v4 = stats_by_fee.get('vip4', {})
            lines.append(f"CAGR g/v9/v4 = {g.get('cagr', float('nan')):.2f} / {v9.get('cagr', float('nan')):.2f} / {v4.get('cagr', float('nan')):.2f}")
            lines.append(f"Sharpe g/v9/v4 = {g.get('sharpe', float('nan')):.2f} / {v9.get('sharpe', float('nan')):.2f} / {v4.get('sharpe', float('nan')):.2f}")
            ax2.text(0.01, 0.95, "\n".join(lines), transform=ax2.transAxes, va='top', ha='left', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

    png_path = os.path.join(out_base, f"{token}_equity.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def run_per_token_pipeline(
    tokens: Iterable[str],
    data_dir: str = "data",
    interval: str = "1h",
    start: Optional[str] = "20231030",
    end: Optional[str] = "20251030",
    windows: Sequence[int] = (3, 6, 12, 24),
    *,
    n_splits: int = 5,
    C: float = 0.5,
    max_iter: int = 2000,
    show_eda: bool = False,
    verbose: bool = True,
    label_h: int = 4,
    label_eps: float = 0.001,
    feature_kwargs: Optional[dict] = None,
    # minute feature integration
    include_minute: bool = False,
    minute_feature_set_id: Optional[str] = None,
    minute_derived_dir: str = "data/derived",
    minute_auto_build: bool = False,
    minute_params: Optional[dict] = None,
    minute_dir: str = "data/minute",
    tune_grid: Sequence[float] = (0.55, 0.60, 0.65),
    # artifacts
    save_plots: bool = True,
    out_dir: str = "simplest_ml/w1/_exp",
    run_id: Optional[str] = None,
) -> List[TokenResult]:
    """End‑to‑end per‑token pipeline with logging and artifacts.

    1) Load parquet → 2) Feature engineering + label → 3) TSCV train/OOS proba
    4) Tune thresholds on validation (OOF) → 5) OOS backtest for fee scenarios
    6) Save charts/CSVs and return a compact summary in TokenResult objects.
    """
    from datetime import datetime
    results: List[TokenResult] = []
    if save_plots and run_id is None:
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    for t in tokens:
        prefix = f"[{t}] "
        if verbose:
            print(f"\n{prefix}==== PIPELINE START (v3) ====")
        try:
            df = load_token_df(t, data_dir=data_dir, interval=interval, start=start, end=end)
        except Exception as e:
            print(f"{prefix}Skip: {e}")
            continue

        symbol = DEFAULT_TOKEN_TO_SYMBOL.get(t, t if t.endswith("USDT") else f"{t}USDT")
        if verbose:
            span = (
                f"{df.index.min()} → {df.index.max()}" if isinstance(df.index, pd.DatetimeIndex) else "N/A"
            )
            print(
                f"{prefix}LOAD: rows={len(df)}, cols={df.shape[1]}, span={span}, cols_req_present="
                f"{all(c in df.columns for c in ['open','high','low','close','volume'])}"
            )

        if show_eda:
            print(f"{prefix}EDA: brief stats & sample")
            eda_brief(df, t, symbol)

        if verbose:
            print(f"{prefix}FEATURES+LABEL: windows={tuple(windows)} H={label_h} eps={label_eps}")
        # optional minute->hour features
        extra_df = None
        if include_minute:
            try:
                from features_minute import (
                    default_params as _min_default_params,
                    make_feature_set_id as _min_make_fid,
                    build_hour_from_minute as _min_build,
                    get_hour_features as _min_get,
                )
                if minute_feature_set_id is None:
                    p = minute_params or _min_default_params()
                    minute_feature_set_id = _min_make_fid(p)
                if minute_auto_build:
                    _min_build(
                        symbol=symbol,
                        start=start,
                        end=end,
                        params=minute_params or _min_default_params(),
                        minute_dir=minute_dir,
                        derived_dir=minute_derived_dir,
                        overwrite=False,
                        verbose=verbose,
                    )
                extra_df = _min_get(
                    symbol=symbol,
                    start=start,
                    end=end,
                    feature_set_id=minute_feature_set_id,
                    derived_dir=minute_derived_dir,
                )
                if verbose:
                    print(f"{prefix}MINUTE FEATS: {extra_df.shape} from {minute_feature_set_id}")
            except Exception as e:
                print(f"{prefix}Minute feature load failed: {e}")
        X, y, feature_cols, xdf = prepare_Xy(
            df,
            windows=windows,
            verbose=verbose,
            log_prefix=prefix,
            label_h=label_h,
            label_eps=label_eps,
            feature_kwargs=feature_kwargs,
            extra_hourly_df=extra_df,
        )
        if len(xdf) < 300:
            print(f"{prefix}Too few rows after feature prep: {len(xdf)} (skip)")
            continue

        if verbose:
            print(f"{prefix}TRAIN: TSCV n_splits={n_splits}, C={C}")
        oof_pred, oof_proba, classes, macro_auc, acc, fold_ranges = train_timeseries_logreg(
            X, y, n_splits=n_splits, C=C, max_iter=max_iter, verbose=verbose, log_prefix=prefix, index=xdf.index
        )
        if verbose:
            print(f"{prefix}OOF macroAUC={macro_auc:.4f} ACC={acc:.4f}")

        # Report classification summary (focus on class 1)
        report = classification_report(y, oof_pred, output_dict=True, zero_division=0)
        precision_up = float(report.get("1", {}).get("precision", 0.0))
        recall_up = float(report.get("1", {}).get("recall", 0.0))
        if verbose:
            print(classification_report(y, oof_pred, digits=3, zero_division=0))
            print("Confusion matrix:\n", confusion_matrix(y, oof_pred))

        # Threshold tuning on OOF (validation-style) and OOS backtest from OOF probabilities
        t_long, t_short, tune_info = tune_thresholds_three_class(oof_proba, y, classes, grid=tune_grid)
        if verbose:
            print(f"{prefix}TUNED THRESHOLDS: t_long={t_long:.2f} t_short={t_short:.2f} | f1_avg={tune_info['f1_avg']:.3f}")

        # Gross and net backtests (using OOF probabilities only)
        pnl_gross, stats_gross, pos_oof = backtest_oof_from_proba(xdf, oof_proba, classes, t_long=t_long, t_short=t_short, fee_bps=0.0)
        pnl_vip9,  stats_vip9,  _      = backtest_oof_from_proba(xdf, oof_proba, classes, t_long=t_long, t_short=t_short, fee_bps=2.0)
        pnl_vip4,  stats_vip4,  _      = backtest_oof_from_proba(xdf, oof_proba, classes, t_long=t_long, t_short=t_short, fee_bps=5.4)
        if verbose:
            print(f"{prefix}BACKTEST (gross):  CAGR={stats_gross['cagr']:.4f} Sharpe={stats_gross['sharpe']:.2f}")
            print(f"{prefix}BACKTEST (VIP9 2.0bps): CAGR={stats_vip9['cagr']:.4f} Sharpe={stats_vip9['sharpe']:.2f}")
            print(f"{prefix}BACKTEST (VIP4 5.4bps): CAGR={stats_vip4['cagr']:.4f} Sharpe={stats_vip4['sharpe']:.2f}")
        if save_plots:
            _save_pnl_artifacts(
                token=t,
                index=xdf.index if isinstance(xdf.index, pd.DatetimeIndex) else pd.RangeIndex(len(xdf)),
                pnl_map={"gross": pnl_gross, "vip9": pnl_vip9, "vip4": pnl_vip4},
                out_dir=out_dir,
                run_id=run_id or "run",
                title_suffix=f"(t=[{t_long:.2f},{t_short:.2f}])",
                fold_ranges=fold_ranges,
                price=xdf["close"] if "close" in xdf.columns else None,
                position=pos_oof,
                stats_by_fee={
                    'gross': stats_gross,
                    'vip9': stats_vip9,
                    'vip4': stats_vip4,
                    'trade': {k: stats_gross.get(k) for k in ['trades','avg_hold','trades_per_day','turnover']}
                }
            )

        start_i = xdf.index.min() if isinstance(xdf.index, pd.DatetimeIndex) else None
        end_i = xdf.index.max() if isinstance(xdf.index, pd.DatetimeIndex) else None

        results.append(
            TokenResult(
                token=t,
                symbol=symbol,
                rows=len(xdf),
                macro_auc=macro_auc,
                accuracy=acc,
                precision_up=precision_up,
                recall_up=recall_up,
                t_long=t_long,
                t_short=t_short,
                start=start_i,
                end=end_i,
                cagr_gross=stats_gross["cagr"],
                sharpe_gross=stats_gross["sharpe"],
                cagr_vip9=stats_vip9["cagr"],
                sharpe_vip9=stats_vip9["sharpe"],
                cagr_vip4=stats_vip4["cagr"],
                sharpe_vip4=stats_vip4["sharpe"],
            )
        )
    return results


def summarize_results(results: List[TokenResult]) -> None:
    try:
        from prettytable import PrettyTable
    except Exception:
        PrettyTable = None

    if not results:
        print("No results to summarize.")
        return

    if PrettyTable is None:
        for r in results:
            print(
                f"{r.token:10s} {r.symbol:12s} rows={r.rows:6d} macroAUC={r.macro_auc:.3f} acc={r.accuracy:.3f} "
                f"t=[{r.t_long:.2f},{r.t_short:.2f}] grossCAGR={r.cagr_gross:.3f} vip9CAGR={r.cagr_vip9:.3f} vip4CAGR={r.cagr_vip4:.3f}"
            )
        return

    tbl = PrettyTable()
    tbl.field_names = [
        "Token",
        "Symbol",
        "Rows",
        "macroAUC",
        "Accuracy",
        "Prec(1)",
        "Rec(1)",
        "t_long",
        "t_short",
        "CAGR(gross)",
        "Sharpe(g)",
        "CAGR(VIP9)",
        "CAGR(VIP4)",
        "Start",
        "End",
    ]
    for r in results:
        tbl.add_row(
            [
                r.token,
                r.symbol,
                r.rows,
                ("nan" if r.macro_auc != r.macro_auc else f"{r.macro_auc:.3f}"),
                f"{r.accuracy:.3f}",
                f"{r.precision_up:.3f}",
                f"{r.recall_up:.3f}",
                f"{r.t_long:.2f}",
                f"{r.t_short:.2f}",
                ("nan" if r.cagr_gross != r.cagr_gross else f"{r.cagr_gross:.3f}"),
                ("nan" if r.sharpe_gross != r.sharpe_gross else f"{r.sharpe_gross:.2f}"),
                ("nan" if r.cagr_vip9 != r.cagr_vip9 else f"{r.cagr_vip9:.3f}"),
                ("nan" if r.cagr_vip4 != r.cagr_vip4 else f"{r.cagr_vip4:.3f}"),
                str(r.start),
                str(r.end),
            ]
        )
    print(tbl)
