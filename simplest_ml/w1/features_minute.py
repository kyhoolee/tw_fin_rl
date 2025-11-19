"""
Minute -> Hour feature aggregation with caching and leak-safe alignment.

Outputs monthly derived shards per symbol:
  data/derived/{symbol}_hour_from_1m_{feature_set_id}_{YYYYMM}.parquet

See doc/minute_prepare_plan.md and doc/feature_processing_plan.md.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Params & identifiers
# ----------------------------


def default_params() -> Dict:
    """Default feature parameters for 1m->1h aggregation.

    feature_set_id is derived from this dict (stable order JSON dump).
    """
    return {
        "ret_windows_min": [5, 15, 60],
        "vol_windows_min": [15, 60],
        "volsum_windows_min": [5, 15, 60],
        "skew_kurt_windows_min": [15, 60],
        "bipower_window_min": 60,
        "vwap_window_min": 60,
        # normalization windows (hours)
        "z_windows_hours": [48, 168],
        # use float32 to reduce size
        "float32": True,
    }


def make_feature_set_id(params: Dict) -> str:
    key_order = [
        "ret_windows_min",
        "vol_windows_min",
        "volsum_windows_min",
        "skew_kurt_windows_min",
        "bipower_window_min",
        "vwap_window_min",
        "z_windows_hours",
    ]
    sig = {k: params.get(k) for k in key_order}
    return "v4m_" + json.dumps(sig, sort_keys=True).replace(" ", "")


# ----------------------------
# Month iteration & IO
# ----------------------------


def _parse_dt(x: str | datetime) -> datetime:
    if isinstance(x, datetime):
        dt = x
    else:
        try:
            dt = datetime.fromisoformat(x)
        except Exception:
            dt = datetime.strptime(x, "%Y-%m-%d")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def iter_months(start: str | datetime, end: str | datetime) -> List[Tuple[datetime, datetime, str]]:
    s = _parse_dt(start)
    e = _parse_dt(end)
    s0 = datetime(s.year, s.month, 1, tzinfo=timezone.utc)
    e0 = datetime(e.year, e.month, 1, tzinfo=timezone.utc)
    cur = s0
    out = []
    while cur <= e0:
        if cur.month == 12:
            nxt = datetime(cur.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            nxt = datetime(cur.year, cur.month + 1, 1, tzinfo=timezone.utc)
        seg_start = max(cur, s)
        seg_end = min(nxt, e)
        out.append((seg_start, seg_end, f"{cur.year:04d}{cur.month:02d}"))
        cur = nxt
    return out


def load_minute_shard(symbol: str, yyyymm: str, minute_dir: str) -> pd.DataFrame:
    fn = os.path.join(minute_dir, f"binance_{symbol.lower()}_1m_{yyyymm}.parquet")
    if not os.path.exists(fn):
        raise FileNotFoundError(fn)
    cols = ["open", "high", "low", "close", "volume"]
    df = pd.read_parquet(fn, columns=cols)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Minute shard must have DatetimeIndex")
    df = df.sort_index().tz_convert("UTC") if df.index.tz is not None else df.tz_localize("UTC")
    # basic cleaning
    df = df[~df.index.duplicated(keep="last")]
    return df


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ----------------------------
# Core aggregation
# ----------------------------


def _resample_last(s: pd.Series) -> pd.Series:
    return s.resample("1H", label="right", closed="right").last()


def aggregate_hour_from_minute(
    df_1m: pd.DataFrame,
    *,
    params: Dict,
    halo_prev: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate 1m frame to 1h features for the provided chunk.

    If `halo_prev` is provided, it will be concatenated to the front to allow
    rolling windows to cross boundaries; the returned hourly frame excludes
    halo bars. Returns (df_hour, halo_for_next_chunk).
    """
    x = df_1m
    if halo_prev is not None and not halo_prev.empty:
        x = pd.concat([halo_prev, x], axis=0).sort_index()

    # minute returns
    r1 = x["close"].pct_change()

    feats: Dict[str, pd.Series] = {}

    # Momentum windows
    for w in params.get("ret_windows_min", [5, 15, 60]):
        s = x["close"] / x["close"].shift(w) - 1.0
        feats[f"m_ret_{w}m"] = _resample_last(s)

    # Volatility windows
    for w in params.get("vol_windows_min", [15, 60]):
        s = r1.rolling(w).std()
        feats[f"m_rv_{w}m"] = _resample_last(s)

    # Skew/Kurt
    for w in params.get("skew_kurt_windows_min", [15, 60]):
        feats[f"m_skew_{w}m"] = _resample_last(r1.rolling(w).skew())
        feats[f"m_kurt_{w}m"] = _resample_last(r1.rolling(w).kurt())

    # Bipower variation proxy
    w = params.get("bipower_window_min", 60)
    bp = (r1.abs() * r1.abs().shift(1)).rolling(w).mean()
    feats[f"m_bipower_{w}m"] = _resample_last(bp)

    # Volume aggregates
    for w in params.get("volsum_windows_min", [5, 15, 60]):
        feats[f"m_volsum_{w}m"] = _resample_last(x["volume"].rolling(w).sum())
        feats[f"m_volmean_{w}m"] = _resample_last(x["volume"].rolling(w).mean())

    # VWAP divergence vs hourly close
    vwap_w = params.get("vwap_window_min", 60)
    pv = (x["close"] * x["volume"]).rolling(vwap_w).sum()
    vv = x["volume"].rolling(vwap_w).sum()
    vwap = pv / (vv + 1e-12)
    vwap_h = _resample_last(vwap)
    hour_close = x["close"].resample("1H", label="right", closed="right").last()
    feats["m_vwap_div_60m"] = (hour_close - vwap_h) / (vwap_h + 1e-12)

    # Combine
    dfh = pd.concat(feats, axis=1)
    # drop halo rows if any
    if halo_prev is not None and not halo_prev.empty:
        first_ts = df_1m.index[0]
        dfh = dfh[dfh.index >= first_ts]

    # Build halo for next chunk: keep last (max_window-1) minutes
    max_w = max(
        params.get("ret_windows_min", [60]) +
        params.get("vol_windows_min", [60]) +
        params.get("volsum_windows_min", [60]) +
        params.get("skew_kurt_windows_min", [60]) +
        [params.get("bipower_window_min", 60), params.get("vwap_window_min", 60)]
    )
    halo = df_1m.tail(max_w - 1) if max_w > 1 else df_1m.iloc[0:0]

    if params.get("float32", True):
        dfh = dfh.astype(np.float32)

    return dfh, halo


# ----------------------------
# Build monthly derived shards with manifest
# ----------------------------


def _manifest_path(symbol: str, feature_set_id: str, derived_dir: str) -> str:
    return os.path.join(derived_dir, f"{symbol.lower()}_hour_from_1m_{feature_set_id}.manifest.json")


def _load_manifest(symbol: str, feature_set_id: str, derived_dir: str) -> Dict:
    fn = _manifest_path(symbol, feature_set_id, derived_dir)
    if os.path.exists(fn):
        try:
            return json.loads(open(fn, "r").read())
        except Exception:
            return {"files": []}
    return {"files": []}


def _save_manifest(symbol: str, feature_set_id: str, derived_dir: str, manifest: Dict) -> None:
    fn = _manifest_path(symbol, feature_set_id, derived_dir)
    _ensure_dir(os.path.dirname(fn))
    with open(fn, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)


def build_hour_from_minute(
    *,
    symbol: str,
    start: str | datetime,
    end: str | datetime,
    params: Dict,
    minute_dir: str = "data/minute",
    derived_dir: str = "data/derived",
    overwrite: bool = False,
    verbose: bool = True,
) -> List[str]:
    """Build monthly derived 1h feature shards from 1m inputs for [start, end).

    Returns list of paths written or kept.
    """
    _ensure_dir(derived_dir)
    fid = make_feature_set_id(params)
    manifest = _load_manifest(symbol, fid, derived_dir)
    written: List[str] = []

    # iterate months with halo
    months = iter_months(start, end)
    halo: Optional[pd.DataFrame] = None

    for seg_start, seg_end, yyyymm in months:
        out_fn = os.path.join(derived_dir, f"{symbol.lower()}_hour_from_1m_{fid}_{yyyymm}.parquet")
        src_fn = os.path.join(minute_dir, f"binance_{symbol.lower()}_1m_{yyyymm}.parquet")
        # invalidate if overwrite or src newer than out
        need = True
        if os.path.exists(out_fn) and not overwrite:
            try:
                need = os.path.getmtime(src_fn) > os.path.getmtime(out_fn)
            except Exception:
                need = False
        if not need:
            if verbose:
                print(f"[keep] {symbol} {yyyymm}: {out_fn}")
            written.append(out_fn)
            # load a small tail from src to continue halo
            try:
                this_min = load_minute_shard(symbol, yyyymm, minute_dir)
                max_w = max(params.get("ret_windows_min", [60]) + params.get("vol_windows_min", [60]) + params.get("volsum_windows_min", [60]) + params.get("skew_kurt_windows_min", [60]) + [params.get("bipower_window_min", 60), params.get("vwap_window_min", 60)])
                halo = this_min.tail(max_w - 1)
            except Exception:
                halo = None
            continue

        if verbose:
            print(f"[build] {symbol} {yyyymm}: {seg_start.date()} → {seg_end.date()}")
        # load with halo
        df_min = load_minute_shard(symbol, yyyymm, minute_dir)
        dfh, halo = aggregate_hour_from_minute(df_min, params=params, halo_prev=halo)
        # Trim to this month span on hourly index
        mask = (dfh.index >= seg_start) & (dfh.index < seg_end)
        dfh = dfh.loc[mask]
        if not dfh.empty:
            dfh.to_parquet(out_fn)
            written.append(out_fn)
            if verbose:
                print(f"  -> rows={len(dfh):,} saved: {out_fn}")
        else:
            if verbose:
                print(f"  -> no hourly rows for month {yyyymm}")

    # update manifest
    manifest["files"] = sorted(list(set(manifest.get("files", []) + written)))
    _save_manifest(symbol, fid, derived_dir, manifest)
    return written


def get_hour_features(
    *,
    symbol: str,
    start: str | datetime,
    end: str | datetime,
    feature_set_id: str,
    derived_dir: str = "data/derived",
) -> pd.DataFrame:
    """Load and concatenate monthly derived shards for [start, end)."""
    s = _parse_dt(start)
    e = _parse_dt(end)
    frames: List[pd.DataFrame] = []
    for _, _, yyyymm in iter_months(s, e):
        fn = os.path.join(derived_dir, f"{symbol.lower()}_hour_from_1m_{feature_set_id}_{yyyymm}.parquet")
        if not os.path.exists(fn):
            continue
        frames.append(pd.read_parquet(fn))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    df = df[(df.index >= s) & (df.index < e)]
    return df

