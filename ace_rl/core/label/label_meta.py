"""Meta labeling utilities for combining base signals with barrier outcomes."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import xarray as xr

from ace_rl.core.cube_feature import make_name
from ace_rl.core.label.label_forward import _finalize_dataset


def _as_direction_array(obj: xr.Dataset | xr.DataArray, feature: str | None = None) -> xr.DataArray:
    """Extract directional signal as DataArray with dims (time, symbol)."""
    if isinstance(obj, xr.Dataset):
        if feature is None:
            raise ValueError("feature must be provided when passing a Dataset")
        data = obj["values"].sel(feature=feature)
    else:
        data = obj
    if "time" not in data.dims or "symbol" not in data.dims:
        raise ValueError("direction array must include time/symbol dims")
    return data.astype(np.float32)


def make_meta_label(
    cube: xr.Dataset,
    direction: xr.Dataset | xr.DataArray,
    barrier_ds: xr.Dataset,
    direction_feature: str | None = None,
    barrier_return_feature: str = "triple_barrier_return",
    min_edge: float = 0.0,
    neutral_as_nan: bool = False,
) -> Tuple[str, xr.Dataset]:
    """Binary meta label indicating whether to act on a base direction signal."""
    dir_da = _as_direction_array(direction, feature=direction_feature)
    realized = barrier_ds["values"].sel(feature=barrier_return_feature)

    dir_da, realized = xr.align(dir_da, realized, join="inner")
    sign = xr.apply_ufunc(np.sign, dir_da)
    if neutral_as_nan:
        sign = sign.where(sign != 0)

    edge = realized * sign
    profit_mask = (sign != 0) & (edge > min_edge)
    meta = xr.where(profit_mask, 1.0, 0.0)
    edge = xr.where(meta > 0, edge, 0.0)
    weight = xr.where(meta > 0, realized.abs(), 0.0)

    params: Dict[str, object] = {
        "type": "meta_label",
        "direction_feature": direction_feature,
        "barrier_return_feature": barrier_return_feature,
        "min_edge": float(min_edge),
        "neutral_as_nan": neutral_as_nan,
    }
    name = make_name("label", "meta", params)
    ds = _finalize_dataset(
        cube,
        {
            "meta_label": meta,
            "meta_edge": edge,
            "meta_weight": weight,
        },
        params,
    )
    return name, ds


def make_meta_label_from_barrier_sign(
    cube: xr.Dataset,
    direction: xr.Dataset | xr.DataArray,
    barrier_ds: xr.Dataset,
    direction_feature: str | None = None,
    barrier_label_feature: str = "triple_barrier_label",
    min_confidence: float = 0.0,
) -> Tuple[str, xr.Dataset]:
    """Meta label using barrier sign agreement with base direction."""
    dir_da = _as_direction_array(direction, feature=direction_feature)
    barrier_sign = barrier_ds["values"].sel(feature=barrier_label_feature)
    dir_da, barrier_sign = xr.align(dir_da, barrier_sign, join="inner")
    dir_sign = xr.apply_ufunc(np.sign, dir_da)

    match = (dir_sign != 0) & (barrier_sign * dir_sign > min_confidence)
    meta = xr.where(match, 1.0, 0.0)

    params: Dict[str, object] = {
        "type": "meta_label_sign",
        "direction_feature": direction_feature,
        "barrier_label_feature": barrier_label_feature,
        "min_confidence": float(min_confidence),
    }
    name = make_name("label", "meta_sign", params)
    ds = _finalize_dataset(
        cube,
        {
            "meta_label": meta,
        },
        params,
    )
    return name, ds


def stack_labels(
    cube: xr.Dataset,
    datasets: Iterable[Tuple[str, xr.Dataset]],
) -> Tuple[str, xr.Dataset]:
    """Combine several label datasets into a single multi-feature label cube."""
    arrays: Dict[str, xr.DataArray] = {}
    params: Dict[str, object] = {"type": "meta_stack", "sources": []}
    for short_name, ds in datasets:
        params["sources"].append(str(short_name))
        for feat in ds["values"].coords["feature"].values.tolist():
            if feat in arrays:
                raise ValueError(f"Duplicate feature '{feat}' encountered while stacking labels")
            arrays[feat] = ds["values"].sel(feature=feat)
    name = make_name("label", "meta_stack", params)
    merged = _finalize_dataset(cube, arrays, params)
    return name, merged
