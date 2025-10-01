"""Barrier-based labeling schemes (triple barrier, time-to-hit)."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import xarray as xr

from ace_rl.core.cube_feature import make_name
from ace_rl.core.label.label_forward import _finalize_dataset

EPS = 1e-12

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba optional
    njit = None


def _triple_barrier_python(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    horizon: int,
    entry_lag: int,
    up_pct: float,
    down_pct: float,
    min_hold: int,
    log_return: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = close.shape[0]
    label = np.zeros(n, dtype=np.int8)
    holding = np.empty(n, dtype=np.float32)
    realized = np.empty(n, dtype=np.float64)
    exit_type = np.zeros(n, dtype=np.int8)

    holding[:] = np.nan
    realized[:] = np.nan

    for t in range(n):
        entry_idx = t + entry_lag
        if entry_idx >= n:
            continue
        entry_price = close[entry_idx]
        if np.isnan(entry_price):
            continue

        up_level = entry_price * (1.0 + up_pct) if up_pct > 0 else np.nan
        down_level = entry_price * (1.0 - down_pct) if down_pct > 0 else np.nan

        end_idx = entry_idx + horizon
        if end_idx >= n:
            end_idx = n - 1
        max_step = end_idx - entry_idx
        if max_step <= 0:
            holding[t] = np.nan
            realized[t] = np.nan
            label[t] = 0
            exit_type[t] = 0
            continue

        hit_step = 0
        choice = 0
        realized_ret = np.nan

        for step in range(1, max_step + 1):
            idx = entry_idx + step
            if step < min_hold:
                continue
            hi = high[idx]
            lo = low[idx]
            if choice == 0 and up_pct > 0 and not np.isnan(hi) and not np.isnan(up_level) and hi >= up_level:
                hit_step = step
                choice = 1
                if log_return:
                    realized_ret = np.log((up_level + EPS) / (entry_price + EPS))
                else:
                    realized_ret = (up_level - entry_price) / (entry_price + EPS)
                break
            if choice == 0 and down_pct > 0 and not np.isnan(lo) and not np.isnan(down_level) and lo <= down_level:
                hit_step = step
                choice = -1
                if log_return:
                    realized_ret = np.log((down_level + EPS) / (entry_price + EPS))
                else:
                    realized_ret = (down_level - entry_price) / (entry_price + EPS)
                break

        if hit_step == 0:
            exit_idx = entry_idx + max_step
            exit_price = close[exit_idx]
            if np.isnan(exit_price):
                realized_ret = np.nan
            else:
                if log_return:
                    realized_ret = np.log((exit_price + EPS) / (entry_price + EPS))
                else:
                    realized_ret = (exit_price - entry_price) / (entry_price + EPS)
            choice = 0
            hit_step = max_step
        else:
            exit_idx = entry_idx + hit_step

        if choice == 0:
            if np.isnan(realized_ret):
                label[t] = 0
            elif realized_ret > 0:
                label[t] = 1
            elif realized_ret < 0:
                label[t] = -1
            else:
                label[t] = 0
        else:
            label[t] = choice

        holding[t] = float(hit_step)
        realized[t] = realized_ret
        exit_type[t] = choice

    return label, holding, realized, exit_type


if njit is not None:  # pragma: no branch

    @njit(cache=True)
    def _triple_barrier_core(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        horizon: int,
        entry_lag: int,
        up_pct: float,
        down_pct: float,
        min_hold: int,
        log_return: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return _triple_barrier_python(
            close,
            high,
            low,
            horizon,
            entry_lag,
            up_pct,
            down_pct,
            min_hold,
            log_return,
        )

else:  # pragma: no cover - fallback when numba is absent

    def _triple_barrier_core(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        horizon: int,
        entry_lag: int,
        up_pct: float,
        down_pct: float,
        min_hold: int,
        log_return: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return _triple_barrier_python(
            close,
            high,
            low,
            horizon,
            entry_lag,
            up_pct,
            down_pct,
            min_hold,
            log_return,
        )


def make_triple_barrier(
    cube: xr.Dataset,
    horizon: int = 24,
    entry_lag: int = 1,
    up_pct: float = 0.01,
    down_pct: float = 0.01,
    min_hold: int = 0,
    price_feature: str = "Close",
    high_feature: str = "High",
    low_feature: str = "Low",
    log_return: bool = True,
) -> Tuple[str, xr.Dataset]:
    """Triple-barrier label assuming fill at t+entry_lag close."""
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if entry_lag <= 0:
        raise ValueError("entry_lag must be >= 1")
    if min_hold < 0:
        raise ValueError("min_hold must be >= 0")

    close = cube["values"].sel(feature=price_feature)
    high = cube["values"].sel(feature=high_feature)
    low = cube["values"].sel(feature=low_feature)

    time_size = close.sizes["time"]
    symbol_size = close.sizes["symbol"]

    label_store = np.full((time_size, symbol_size), np.nan, dtype=np.float32)
    hold_store = np.full((time_size, symbol_size), np.nan, dtype=np.float32)
    ret_store = np.full((time_size, symbol_size), np.nan, dtype=np.float32)
    type_store = np.full((time_size, symbol_size), np.nan, dtype=np.float32)

    for idx, symbol in enumerate(close.coords["symbol"].values):
        lbl, hold, realized, exit_type = _triple_barrier_core(
            close.sel(symbol=symbol).values,
            high.sel(symbol=symbol).values,
            low.sel(symbol=symbol).values,
            horizon,
            entry_lag,
            up_pct,
            down_pct,
            min_hold,
            log_return,
        )
        label_store[:, idx] = lbl.astype(np.float32)
        hold_store[:, idx] = hold
        ret_store[:, idx] = realized.astype(np.float32)
        type_store[:, idx] = exit_type.astype(np.float32)

    coords = dict(time=close.coords["time"], symbol=close.coords["symbol"])
    arrays: Dict[str, xr.DataArray] = {
        "triple_barrier_label": xr.DataArray(label_store, coords=coords, dims=("time", "symbol")),
        "triple_barrier_holding": xr.DataArray(hold_store, coords=coords, dims=("time", "symbol")),
        "triple_barrier_return": xr.DataArray(ret_store, coords=coords, dims=("time", "symbol")),
        "triple_barrier_exit_type": xr.DataArray(type_store, coords=coords, dims=("time", "symbol")),
    }

    params: Dict[str, object] = {
        "type": "triple_barrier",
        "price_feature": price_feature,
        "high_feature": high_feature,
        "low_feature": low_feature,
        "horizon": horizon,
        "entry_lag": entry_lag,
        "up_pct": up_pct,
        "down_pct": down_pct,
        "min_hold": min_hold,
        "log_return": log_return,
        "numba": bool(njit is not None),
    }
    name = make_name("label", "triple_barrier", params)
    ds = _finalize_dataset(cube, arrays, params)
    return name, ds


def make_time_to_hit(
    cube: xr.Dataset,
    levels: Tuple[float, float],
    entry_lag: int = 1,
    horizon: int = 24,
    price_feature: str = "Close",
    high_feature: str = "High",
    low_feature: str = "Low",
) -> Tuple[str, xr.Dataset]:
    """Return time (bars) to reach +/- thresholds in future window."""
    upper, lower = map(float, levels)
    if upper <= 0 or lower >= 0:
        raise ValueError("levels must be (positive_upper, negative_lower)")
    close = cube["values"].sel(feature=price_feature)
    high = cube["values"].sel(feature=high_feature)
    low = cube["values"].sel(feature=low_feature)

    time_size = close.sizes["time"]
    symbol_size = close.sizes["symbol"]
    hold_store = np.full((time_size, symbol_size), np.nan, dtype=np.float32)

    for idx, symbol in enumerate(close.coords["symbol"].values):
        _, hold, _, _ = _triple_barrier_core(
            close.sel(symbol=symbol).values,
            high.sel(symbol=symbol).values,
            low.sel(symbol=symbol).values,
            horizon,
            entry_lag,
            upper,
            abs(lower),
            0,
            False,
        )
        hold_store[:, idx] = hold

    coords = dict(time=close.coords["time"], symbol=close.coords["symbol"])
    arrays = {"time_to_hit": xr.DataArray(hold_store, coords=coords, dims=("time", "symbol"))}
    params = {
        "type": "time_to_hit",
        "levels": (upper, lower),
        "entry_lag": entry_lag,
        "horizon": horizon,
        "price_feature": price_feature,
        "high_feature": high_feature,
        "low_feature": low_feature,
    }
    name = make_name("label", "time_to_hit", params)
    ds = _finalize_dataset(cube, arrays, params)
    return name, ds
