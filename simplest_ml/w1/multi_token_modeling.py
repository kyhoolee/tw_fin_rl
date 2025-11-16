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


def add_features(
    df: pd.DataFrame,
    windows: Sequence[int] = (3, 6, 12, 24),
    *,
    verbose: bool = False,
    log_prefix: str = "",
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

    feature_cols = [
        c
        for c in x.columns
        if any(k in c for k in ["ret_", "sma_", "ema_", "vol_", "max_", "min_", "rsi_"])
    ]
    # Lag all features to avoid leakage
    for c in feature_cols:
        x[c] = x[c].shift(1)

    if verbose:
        na_counts = int(x[feature_cols].isna().sum().sum())
        print(
            f"{log_prefix}add_features: generated {len(feature_cols)} features; total NaNs after lag={na_counts}"
        )
    return x, feature_cols


def make_label(df: pd.DataFrame, *, verbose: bool = False, log_prefix: str = "") -> pd.Series:
    """Binary label: 1 if next close > current close, else 0."""
    ret_1 = df["close"].pct_change(1)
    y = (ret_1.shift(-1) > 0).astype(int)
    if verbose:
        ones = int(y.sum())
        print(f"{log_prefix}make_label: positives={ones} / {len(y)}")
    return y


def prepare_Xy(
    df: pd.DataFrame,
    windows: Sequence[int] = (3, 6, 12, 24),
    *,
    verbose: bool = False,
    log_prefix: str = "",
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    if verbose:
        idx_info = (
            f"[{df.index.min()} → {df.index.max()}]" if isinstance(df.index, pd.DatetimeIndex) else ""
        )
        print(f"{log_prefix}prepare_Xy: input rows={len(df)} {idx_info}")

    x, feature_cols = add_features(df, windows=windows, verbose=verbose, log_prefix=log_prefix)
    y = make_label(x, verbose=verbose, log_prefix=log_prefix)
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

        clf = LogisticRegression(max_iter=max_iter, C=C, n_jobs=None)
        clf.fit(Xtr, ytr)

        proba = clf.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)

        oof_pred[test_idx] = pred
        oof_proba[test_idx] = proba
        if verbose:
            try:
                fold_auc = roc_auc_score(yte, proba)
            except Exception:
                fold_auc = float("nan")
            fold_acc = float((pred == yte).mean())
            print(
                f"{log_prefix}train: fold {fold} metrics | AUC={fold_auc:.4f} ACC={fold_acc:.4f}"
            )

    auc = float(roc_auc_score(y, oof_proba))
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
            print(f"{prefix}Step 3/4 FEATURES: windows={tuple(windows)}")
        X, y, feature_cols, xdf = prepare_Xy(
            df, windows=windows, verbose=verbose, log_prefix=prefix
        )
        if len(xdf) < 300:
            print(f"{prefix}Too few rows after feature prep: {len(xdf)} (skip)")
            continue

        # Train
        if verbose:
            print(f"{prefix}Step 4/4 TRAIN: TSCV n_splits={n_splits}, C={C}, max_iter={max_iter}")
        pred, proba, auc, acc, thr = train_timeseries_logreg(
            X, y, n_splits=n_splits, C=C, max_iter=max_iter, verbose=verbose, log_prefix=prefix
        )

        # Detailed report
        if verbose:
            print(f"{prefix}FINAL METRICS: AUC={auc:.4f} ACC={acc:.4f}")
            print(classification_report(y, pred, digits=3))
            print("Confusion matrix:\n", confusion_matrix(y, pred))

        # Collect summary
        start_i = xdf.index.min() if isinstance(xdf.index, pd.DatetimeIndex) else None
        end_i = xdf.index.max() if isinstance(xdf.index, pd.DatetimeIndex) else None
        # precision/recall from report
        report = classification_report(y, pred, output_dict=True, zero_division=0)
        precision = float(report["1"]["precision"]) if "1" in report else 0.0
        recall = float(report["1"]["recall"]) if "1" in report else 0.0

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
        "Start",
        "End",
    ]
    for r in results:
        tbl.add_row(
            [
                r.token,
                r.symbol,
                r.rows,
                f"{r.auc:.3f}",
                f"{r.accuracy:.3f}",
                f"{r.precision:.3f}",
                f"{r.recall:.3f}",
                str(r.start),
                str(r.end),
            ]
        )
    print(tbl)
