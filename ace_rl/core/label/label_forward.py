"""Forward-looking labels computed from candle data."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import xarray as xr

from ace_rl.core.cube_feature import make_name

EPS = 1e-12


def _inherit_chunks_like(base_cube: xr.Dataset, da: xr.DataArray) -> xr.DataArray:
    """Make label array respect the chunking scheme of the base cube."""
    base_values = base_cube["values"]
    base_chunks = getattr(base_values.data, "chunks", None)
    if not base_chunks:
        return da

    chunk_map = dict(zip(base_values.dims, base_chunks))
    spec: Dict[str, int] = {}
    if "time" in chunk_map:
        spec["time"] = int(chunk_map["time"][0])
    if "symbol" in chunk_map:
        spec["symbol"] = int(chunk_map["symbol"][0])
    if "feature" in da.dims:
        spec["feature"] = da.sizes.get("feature", 1)
    return da.chunk(spec)


def _finalize_dataset(
    base_cube: xr.Dataset,
    data_arrays: Dict[str, xr.DataArray],
    params: Dict[str, object],
) -> xr.Dataset:
    """Expand feature axis, inherit chunks, attach metadata."""
    if not data_arrays:
        raise ValueError("No data arrays supplied for label construction")

    slices: List[xr.DataArray] = []
    for label_name, arr in data_arrays.items():
        if "time" not in arr.dims or "symbol" not in arr.dims:
            raise ValueError(f"Label '{label_name}' missing required dims time/symbol")
        da = arr.transpose("time", "symbol")
        da = da.astype(np.float32)
        da = da.expand_dims(feature=[label_name], axis=-1)
        slices.append(da)

    merged = xr.concat(slices, dim="feature")
    merged = _inherit_chunks_like(base_cube, merged)
    merged = merged.assign_coords(
        symbol=merged["symbol"].astype(str),
        feature=merged["feature"].astype(str),
    )

    ds = merged.to_dataset(name="values")
    ds.attrs["params"] = params
    ds.attrs["kind"] = "label"
    return ds


def _compute_forward_prices(
    close: xr.DataArray,
    horizon: int,
    entry_lag: int,
) -> Tuple[xr.DataArray, xr.DataArray]:
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if entry_lag <= 0:
        raise ValueError("entry_lag must be >= 1")
    entry = close.shift(time=-entry_lag)
    exit_price = close.shift(time=-(entry_lag + horizon))
    return entry, exit_price


def _compute_forward_return(
    close: xr.DataArray,
    horizon: int,
    entry_lag: int,
    log_return: bool,
) -> xr.DataArray:
    entry, exit_price = _compute_forward_prices(close, horizon, entry_lag)
    if log_return:
        raw = np.log((exit_price + EPS) / (entry + EPS))
    else:
        raw = (exit_price - entry) / (entry + EPS)
    return raw


def make_forward_return(
    cube: xr.Dataset,
    horizon: int = 1,
    entry_lag: int = 1,
    price_feature: str = "Close",
    log_return: bool = True,
    clip: float | None = None,
) -> Tuple[str, xr.Dataset]:
    """Forward return measured from t+entry_lag to t+entry_lag+horizon."""
    close = cube["values"].sel(feature=price_feature)
    ret = _compute_forward_return(close, horizon, entry_lag, log_return)
    if clip is not None and clip > 0:
        ret = ret.clip(min=-clip, max=clip)

    params = {
        "type": "forward_return",
        "price_feature": price_feature,
        "horizon": horizon,
        "entry_lag": entry_lag,
        "log_return": log_return,
        "clip": clip,
    }
    name = make_name("label", f"fwdret_h{horizon}", params)
    ds = _finalize_dataset(cube, {f"forward_return_{horizon}": ret}, params)
    return name, ds


def make_forward_direction(
    cube: xr.Dataset,
    horizon: int = 1,
    entry_lag: int = 1,
    price_feature: str = "Close",
    neutral_threshold: float = 0.0,
    log_return: bool = True,
) -> Tuple[str, xr.Dataset]:
    """Directional label derived from forward return."""
    ret = _compute_forward_return(
        cube["values"].sel(feature=price_feature),
        horizon=horizon,
        entry_lag=entry_lag,
        log_return=log_return,
    )
    direction = xr.apply_ufunc(np.sign, ret)
    if neutral_threshold > 0:
        mask = ret.abs() < neutral_threshold
        direction = direction.where(~mask, 0.0)

    params = {
        "type": "forward_direction",
        "price_feature": price_feature,
        "horizon": horizon,
        "entry_lag": entry_lag,
        "neutral_threshold": neutral_threshold,
        "log_return": log_return,
    }
    name = make_name("label", f"fwddir_h{horizon}", params)
    ds = _finalize_dataset(cube, {f"forward_direction_{horizon}": direction}, params)
    return name, ds


def make_multi_horizon_forward_return(
    cube: xr.Dataset,
    horizons: Iterable[int],
    entry_lag: int = 1,
    price_feature: str = "Close",
    log_return: bool = True,
) -> Tuple[str, xr.Dataset]:
    """Pack multiple forward returns (different horizons) in one dataset."""
    horizons = list(sorted({int(h) for h in horizons if h > 0}))
    if not horizons:
        raise ValueError("At least one positive horizon required")

    close = cube["values"].sel(feature=price_feature)
    arrays: Dict[str, xr.DataArray] = {}
    for horizon in horizons:
        ret = _compute_forward_return(close, horizon, entry_lag, log_return)
        arrays[f"forward_return_{horizon}"] = ret

    params = {
        "type": "forward_return_multi",
        "price_feature": price_feature,
        "horizons": horizons,
        "entry_lag": entry_lag,
        "log_return": log_return,
    }
    name = make_name("label", "fwdret_multi", params)
    ds = _finalize_dataset(cube, arrays, params)
    return name, ds
