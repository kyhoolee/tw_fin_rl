from pathlib import Path
import xarray as xr
import yaml
import hashlib
import json
import numpy as np
from typing import Tuple

# ============================================================
# 1. Utils helpers: naming, save, load
# ============================================================

def make_name(prefix: str, shortparam: str, params: dict) -> str:
    """
    Sinh tên folder: <prefix>_<shortparam>_<hash4>
    Hash4 lấy từ JSON dump của params.
    """
    params_str = json.dumps(params, sort_keys=True)
    h = hashlib.sha1(params_str.encode()).hexdigest()[:4]
    return f"{prefix}_{shortparam}_{h}"


def save_cube(ds: xr.Dataset, root: Path, subdir: str, name: str):
    """
    Save dataset xuống root/subdir/name/{data.zarr, meta.yaml}.
    subdir = "features" hoặc "labels"
    """
    path = root / subdir / name
    path.mkdir(parents=True, exist_ok=True)

    ds.to_zarr(path / "data.zarr", mode="w", consolidated=True)
    with open(path / "meta.yaml", "w") as f:
        yaml.safe_dump(dict(ds.attrs), f)


def load_cube(root: Path, subdir: str, name: str) -> xr.Dataset:
    ds = xr.open_zarr(root / subdir / name / "data.zarr", consolidated=True)
    # load thêm meta.yaml nếu cần
    meta_file = root / subdir / name / "meta.yaml"
    if meta_file.exists():
        with open(meta_file, "r") as f:
            attrs = yaml.safe_load(f)
        ds.attrs.update(attrs)
    return ds


# ============================================================
# 2. Feature functions
# ============================================================

def _wrap_single_feature(
    base_cube: xr.Dataset,
    array: xr.DataArray,
    feature_name: str,
    params: dict,
) -> xr.Dataset:
    feature_da = array.transpose("time", "symbol").expand_dims(feature=[feature_name], axis=-1)

    base_values = base_cube["values"]
    chunk_spec: dict[str, int] = {}
    base_chunks = getattr(base_values.data, "chunks", None)
    if base_chunks:
        chunk_map = dict(zip(base_values.dims, base_chunks))
        time_chunks = chunk_map.get("time")
        symbol_chunks = chunk_map.get("symbol")
        if time_chunks:
            chunk_spec["time"] = int(time_chunks[0])
        if symbol_chunks:
            chunk_spec["symbol"] = int(symbol_chunks[0])
    chunk_spec["feature"] = 1
    if chunk_spec:
        feature_da = feature_da.chunk(chunk_spec)

    ds = feature_da.to_dataset(name="values")
    ds.attrs["params"] = params
    return ds


def make_return(cube: xr.Dataset, lookback: int = 1) -> Tuple[str, xr.Dataset]:
    close = cube["values"].sel(feature="Close")
    ret = np.log(close / close.shift(time=lookback))

    params = {"lookback": lookback, "method": "log_return"}
    name = make_name("ret", f"lag{lookback}", params)

    ds = _wrap_single_feature(cube, ret, f"return_lag{lookback}", params)
    return name, ds


def make_volatility(cube: xr.Dataset, window: int = 30) -> Tuple[str, xr.Dataset]:
    """
    rolling std dev của log return.
    """
    close = cube["values"].sel(feature="Close")
    logret = np.log(close / close.shift(time=1))
    vol = logret.rolling(time=window).std()

    params = {"window": window, "method": "rolling_std"}
    name = make_name("vol", f"win{window}", params)

    ds = _wrap_single_feature(cube, vol, f"volatility_{window}", params)
    return name, ds


# ============================================================
# 3. Label functions
# ============================================================

def make_forward_return(cube: xr.Dataset, horizon: int = 5) -> Tuple[str, xr.Dataset]:
    """
    forward log return sau horizon step.
    """
    close = cube["values"].sel(feature="Close")
    fwd = np.log(close.shift(time=-horizon) / close)

    params = {"horizon": horizon, "method": "forward_log_return"}
    name = make_name("label", f"fwd{horizon}", params)

    ds = _wrap_single_feature(cube, fwd, f"forward_return_{horizon}", params)
    return name, ds
