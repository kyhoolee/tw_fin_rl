import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

try:
    # When imported as a module within the package
    from .pretty_helpers import (
        DEFAULT_TOKEN_TO_SYMBOL,
        load_token_df,
        pretty_print_sample,
    )
except Exception:
    # When used from a notebook in the same directory
    import sys as _sys, os as _os
    _sys.path.append(_os.path.dirname(__file__))
    from pretty_helpers import (
        DEFAULT_TOKEN_TO_SYMBOL,
        load_token_df,
        pretty_print_sample,
    )


@dataclass
class TokenResult:
    token: str
    symbol: str
    rows: int
    auc: float
    accuracy: float
    precision: float
    recall: float
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    # Optional backtest summary
    cagr: Optional[float] = None
    sharpe: Optional[float] = None
    hit_rate: Optional[float] = None


def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
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
    """
    Add rolling/momentum features similar to the single-token notebook.
    Returns (df_with_features, feature_columns).
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

    # Candle anatomy
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

    # Volatility & range blocks
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

    # Volume features
    if use_volume:
        for w in windows:
            v_ma = x["volume"].rolling(w).mean()
            v_sd = x["volume"].rolling(w).std()
            x[f"vol_z_{w}"] = (x["volume"] - v_ma) / (v_sd + 1e-9)
        direction = np.sign(x["close"].diff()).fillna(0.0)
        x["obv"] = (direction * x["volume"]).cumsum()

    # Time features (cyclical)
    if use_time and isinstance(x.index, pd.DatetimeIndex):
        hr = x.index.hour
        dw = x.index.dayofweek
        x["hr_sin"] = np.sin(2 * np.pi * hr / 24)
        x["hr_cos"] = np.cos(2 * np.pi * hr / 24)
        x["dw_sin"] = np.sin(2 * np.pi * dw / 7)
        x["dw_cos"] = np.cos(2 * np.pi * dw / 7)

    feature_cols = [
        c
        for c in x.columns
        if any(k in c for k in ["ret_", "sma_", "ema_", "vol_", "max_", "min_", "rsi_"])
    ]
    # Extended blocks
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

    # Optional rolling z-scores for stability
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
    # Lag all features to avoid leakage
    for c in feature_cols:
        x[c] = x[c].shift(1)

    if verbose:
        na_counts = int(x[feature_cols].isna().sum().sum())
        print(
            f"{log_prefix}add_features: generated {len(feature_cols)} features; total NaNs after lag={na_counts}"
        )
    return x, feature_cols


def make_label_binary(df: pd.DataFrame, *, verbose: bool = False, log_prefix: str = "") -> pd.Series:
    """Binary label: 1 if next close > current close, else 0."""
    ret_1 = df["close"].pct_change(1)
    y = (ret_1.shift(-1) > 0).astype(int)
    if verbose:
        ones = int(y.sum())
        print(f"{log_prefix}make_label(binary): positives={ones} / {len(y)}")
    return y


def make_label_three_class(
    df: pd.DataFrame,
    *,
    H: int = 4,
    eps: float = 0.001,
    verbose: bool = False,
    log_prefix: str = "",
) -> pd.Series:
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
    label_mode: str = "binary",  # 'binary' or 'three_class'
    label_h: int = 4,
    label_eps: float = 0.001,
    feature_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
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
    if label_mode == "binary":
        y = make_label_binary(x, verbose=verbose, log_prefix=log_prefix)
    elif label_mode == "three_class":
        y = make_label_three_class(x, H=label_h, eps=label_eps, verbose=verbose, log_prefix=log_prefix)
    else:
        raise ValueError("label_mode must be 'binary' or 'three_class'")
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
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    TimeSeriesSplit + StandardScaler + LogisticRegression.
    Returns (oof_pred, oof_proba, auc, accuracy, threshold_used).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_pred = np.zeros_like(y)
    oof_proba = np.zeros_like(y, dtype=float)

    classes = np.unique(y)
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        if verbose:
            print(
                f"{log_prefix}train: fold {fold}/{n_splits} | train={len(train_idx)} test={len(test_idx)}"
            )
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        multi = len(np.unique(y)) > 2
        if multi:
            clf = LogisticRegression(max_iter=max_iter, C=C, n_jobs=None, multi_class="multinomial", solver="lbfgs")
        else:
            clf = LogisticRegression(max_iter=max_iter, C=C, n_jobs=None)
        clf.fit(Xtr, ytr)

        proba = clf.predict_proba(Xte)
        if multi:
            pred = clf.predict(Xte)
        else:
            proba_1 = proba[:, 1]
            pred = (proba_1 >= 0.5).astype(int)

        oof_pred[test_idx] = pred
        if proba.ndim == 1 or proba.shape[1] == 1:
            oof_proba[test_idx] = proba
        else:
            if proba.shape[1] == 2:
                oof_proba[test_idx] = proba[:, 1]
            else:
                oof_proba[test_idx] = proba.max(axis=1)
        if verbose:
            try:
                if proba.ndim == 1 or proba.shape[1] == 1:
                    fold_auc = roc_auc_score(yte, proba)
                elif proba.shape[1] == 2:
                    fold_auc = roc_auc_score(yte, proba[:, 1])
                else:
                    from sklearn.preprocessing import label_binarize
                    y_bin = label_binarize(yte, classes=np.sort(classes))
                    fold_auc = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
            except Exception:
                fold_auc = float("nan")
            fold_acc = float((pred == yte).mean())
            print(
                f"{log_prefix}train: fold {fold} metrics | AUC={fold_auc:.4f} ACC={fold_acc:.4f}"
            )

    try:
        if len(np.unique(y)) == 2:
            auc = float(roc_auc_score(y, oof_proba))
        else:
            auc = float("nan")
    except Exception:
        auc = float("nan")
    acc = float((oof_pred == y).mean())
    if verbose:
        print(f"{log_prefix}train: OOF metrics | AUC={auc:.4f} ACC={acc:.4f}")
    return oof_pred, oof_proba, auc, acc, 0.5


def eda_brief(df: pd.DataFrame, token: str, symbol: str) -> None:
    start = df.index.min() if isinstance(df.index, pd.DatetimeIndex) else None
    end = df.index.max() if isinstance(df.index, pd.DatetimeIndex) else None
    print(f"{token} ({symbol}) | rows={len(df)} | span= {start} → {end}")
    # Show quick sample
    pretty_print_sample(df, head=3, tail=3)
    # Basic stats
    s = df[["open", "high", "low", "close", "volume"]].describe()
    print(s)


def _equity_metrics(pnl: pd.Series, index: Optional[pd.DatetimeIndex]) -> Tuple[float, float, float]:
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
    hr = float((ret[ret != 0] > 0).mean()) if (ret != 0).any() else float("nan")
    return cagr, sharpe, hr


def backtest_from_proba(
    xdf: pd.DataFrame,
    proba: np.ndarray,
    pred: np.ndarray,
    classes: Sequence[int],
    *,
    t_long: float = 0.55,
    t_short: float = 0.55,
    fee_bps: float = 5.0,
    min_hold: int = 1,
) -> Tuple[pd.Series, Dict[str, float]]:
    classes = list(classes)
    if len(classes) == 3 and -1 in classes and 1 in classes:
        idx_up = classes.index(1)
        idx_dn = classes.index(-1)
        p_up = proba[:, idx_up]
        p_dn = proba[:, idx_dn]
        pos = pd.Series(0, index=xdf.index, dtype=float)
        pos[p_up >= t_long] = 1.0
        pos[p_dn >= t_short] = -1.0
    elif len(classes) == 2:
        p_up = proba
        pos = pd.Series((p_up >= t_long).astype(float), index=xdf.index)
    else:
        pos = pd.Series(np.where(pred > 0, 1.0, np.where(pred < 0, -1.0, 0.0)), index=xdf.index)

    if min_hold > 1:
        last = 0.0
        hold = 0
        out = []
        for v in pos.values:
            if v != last:
                last = v
                hold = min_hold
            if hold > 0:
                out.append(last)
                hold -= 1
            else:
                out.append(v)
        pos = pd.Series(out, index=pos.index)

    ret_1 = xdf["close"].pct_change().fillna(0.0)
    trade = pos.diff().abs().fillna(0.0)
    fee = (fee_bps / 1e4) * trade
    pnl = pos.shift(1).fillna(0.0) * ret_1 - fee

    cagr, sharpe, hr = _equity_metrics(pnl, xdf.index if isinstance(xdf.index, pd.DatetimeIndex) else None)
    stats = {"cagr": cagr, "sharpe": sharpe, "hit_rate": hr}
    return pnl, stats


def run_per_token_pipeline(
    tokens: Iterable[str],
    data_dir: str = "data",
    interval: str = "1h",
    start: Optional[str] = "20231030",
    end: Optional[str] = "20251030",
    windows: Sequence[int] = (3, 6, 12, 24),
    n_splits: int = 5,
    C: float = 0.5,
    max_iter: int = 2000,
    show_eda: bool = True,
    verbose: bool = True,
    label_mode: str = "three_class",
    label_h: int = 4,
    label_eps: float = 0.001,
    feature_kwargs: Optional[dict] = None,
    do_backtest: bool = True,
    t_long: float = 0.60,
    t_short: float = 0.60,
    fee_bps: float = 5.0,
    min_hold: int = 1,
) -> List[TokenResult]:
    results: List[TokenResult] = []
    for t in tokens:
        prefix = f"[{t}] "
        if verbose:
            print(f"\n{prefix}==== PIPELINE START ====")
        try:
            df = load_token_df(t, data_dir=data_dir, interval=interval, start=start, end=end)
        except Exception as e:
            print(f"{prefix}Skip: {e}")
            continue

        symbol = DEFAULT_TOKEN_TO_SYMBOL.get(t, t if t.endswith("USDT") else f"{t}USDT")
        # Step 1: Data info
        if verbose:
            span = (
                f"{df.index.min()} → {df.index.max()}" if isinstance(df.index, pd.DatetimeIndex) else "N/A"
            )
            print(
                f"{prefix}Step 1/4 LOAD: rows={len(df)}, cols={df.shape[1]}, span={span}, cols_req_present="
                f"{all(c in df.columns for c in ['open','high','low','close','volume'])}"
            )

        # EDA (brief)
        if show_eda:
            print(f"{prefix}Step 2/4 EDA: brief stats & sample")
            eda_brief(df, t, symbol)

        # Features + label
        if verbose:
            print(f"{prefix}Step 3/5 FEATURES+LABEL: windows={tuple(windows)} mode={label_mode} H={label_h} eps={label_eps}")
        X, y, feature_cols, xdf = prepare_Xy(
            df,
            windows=windows,
            verbose=verbose,
            log_prefix=prefix,
            label_mode=label_mode,
            label_h=label_h,
            label_eps=label_eps,
            feature_kwargs=feature_kwargs,
        )
        if len(xdf) < 300:
            print(f"{prefix}Too few rows after feature prep: {len(xdf)} (skip)")
            continue

        # Train
        if verbose:
            print(f"{prefix}Step 4/5 TRAIN: TSCV n_splits={n_splits}, C={C}, max_iter={max_iter}")
        pred, proba, auc, acc, thr = train_timeseries_logreg(
            X, y, n_splits=n_splits, C=C, max_iter=max_iter, verbose=verbose, log_prefix=prefix
        )

        # Detailed report
        if verbose:
            print(f"{prefix}FINAL METRICS: AUC={auc:.4f} ACC={acc:.4f}")
            print(classification_report(y, pred, digits=3, zero_division=0))
            print("Confusion matrix:\n", confusion_matrix(y, pred))

        # Optional backtest
        cagr = sharpe = hit = None
        if do_backtest:
            if verbose:
                print(f"{prefix}Step 5/5 BACKTEST: t_long={t_long} t_short={t_short} fee_bps={fee_bps} min_hold={min_hold}")
            try:
                scaler = StandardScaler().fit(X)
                Xs = scaler.transform(X)
                classes = np.unique(y)
                multi = len(classes) > 2
                if multi:
                    clf = LogisticRegression(max_iter=max_iter, C=C, n_jobs=None, multi_class="multinomial", solver="lbfgs")
                else:
                    clf = LogisticRegression(max_iter=max_iter, C=C, n_jobs=None)
                clf.fit(Xs, y)
                full_proba = clf.predict_proba(Xs)
                full_pred = clf.predict(Xs)
                pnl, stats = backtest_from_proba(
                    xdf, full_proba, full_pred, list(np.sort(classes)), t_long=t_long, t_short=t_short, fee_bps=fee_bps, min_hold=min_hold
                )
                cagr, sharpe, hit = stats["cagr"], stats["sharpe"], stats["hit_rate"]
                if verbose:
                    print(f"{prefix}BACKTEST: CAGR={cagr:.4f} Sharpe={sharpe:.2f} HitRate={hit:.2f}")
            except Exception as e:
                print(f"{prefix}Backtest failed: {e}")

        # Collect summary
        start_i = xdf.index.min() if isinstance(xdf.index, pd.DatetimeIndex) else None
        end_i = xdf.index.max() if isinstance(xdf.index, pd.DatetimeIndex) else None
        # precision/recall from report
        report = classification_report(y, pred, output_dict=True, zero_division=0)
        key_up = "1" if "1" in report else list(report.keys())[0]
        precision = float(report[key_up]["precision"]) if key_up in report else 0.0
        recall = float(report[key_up]["recall"]) if key_up in report else 0.0

        results.append(
            TokenResult(
                token=t,
                symbol=symbol,
                rows=len(xdf),
                auc=auc,
                accuracy=acc,
                precision=precision,
                recall=recall,
                start=start_i,
                end=end_i,
                cagr=cagr,
                sharpe=sharpe,
                hit_rate=hit,
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
                f"{r.token:10s} {r.symbol:12s} rows={r.rows:6d} auc={r.auc:.3f} acc={r.accuracy:.3f} "
                f"prec={r.precision:.3f} rec={r.recall:.3f} span={r.start}→{r.end}"
            )
        return

    tbl = PrettyTable()
    tbl.field_names = [
        "Token",
        "Symbol",
        "Rows",
        "AUC",
        "Accuracy",
        "Precision(1)",
        "Recall(1)",
        "CAGR",
        "Sharpe",
        "HitRate",
        "Start",
        "End",
    ]
    for r in results:
        tbl.add_row(
            [
                r.token,
                r.symbol,
                r.rows,
                ("nan" if r.auc != r.auc else f"{r.auc:.3f}"),
                f"{r.accuracy:.3f}",
                f"{r.precision:.3f}",
                f"{r.recall:.3f}",
                ("nan" if r.cagr is None or r.cagr != r.cagr else f"{r.cagr:.3f}"),
                ("nan" if r.sharpe is None or r.sharpe != r.sharpe else f"{r.sharpe:.2f}"),
                ("nan" if r.hit_rate is None or r.hit_rate != r.hit_rate else f"{r.hit_rate:.2f}"),
                str(r.start),
                str(r.end),
            ]
        )
    print(tbl)
