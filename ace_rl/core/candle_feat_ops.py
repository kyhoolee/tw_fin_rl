# features_ops.py
from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
import xarray as xr

EPS = 1e-12

# -------------------------
# Core helpers (không đụng tới chunking logic của bạn)
# -------------------------
def _inherit_chunks_like(base_cube: xr.Dataset, da: xr.DataArray) -> xr.DataArray:
    """Set chunk spec của feature mới giống cube gốc (time,symbol) + feature:1."""
    base_values = base_cube["values"]
    base_chunks = getattr(base_values.data, "chunks", None)
    print("[inherit] incoming array dims:", da.dims, "shape:", da.shape, "dtype:", da.dtype)
    if base_chunks:
        chunk_map = dict(zip(base_values.dims, base_chunks))
        spec: Dict[str, int] = {}
        if "time" in chunk_map:
            spec["time"] = int(chunk_map["time"][0])
        if "symbol" in chunk_map:
            spec["symbol"] = int(chunk_map["symbol"][0])
        spec["feature"] = 1
        printable_chunk_map = {
            dim: tuple(int(c) for c in chunk_map[dim]) for dim in chunk_map
        }
        print("[inherit] base dims:", base_values.dims)
        print("[inherit] base chunk map:", printable_chunk_map)
        print("[inherit] requested chunk spec:", spec)
        da = da.chunk(spec)
        print("[inherit] resulting chunks:", da.chunks)
    else:
        print("[inherit] base cube not chunked; skip chunk inheritance")
    return da

def _wrap_single_feature(
    base_cube: xr.Dataset, array: xr.DataArray, feature_name: str, params: Dict[str, Any]
) -> xr.Dataset:
    print("[wrap] feature:", feature_name)
    print("[wrap] source dims:", array.dims, "shape:", array.shape, "dtype:", array.dtype)
    coord_summary = {
        dim: {
            "dtype": str(array.coords[dim].dtype),
            "type": type(array.coords[dim].data).__name__
        }
        for dim in array.dims if dim in array.coords
    }
    print("[wrap] source coord summary:", coord_summary)

    da = array.transpose("time", "symbol")
    print("[wrap] after transpose dims:", da.dims, "shape:", da.shape)
    da = da.astype("float32")
    print("[wrap] after astype dtype:", da.dtype)
    da = da.expand_dims(feature=[feature_name], axis=-1)
    print("[wrap] after expand dims:", da.dims, "shape:", da.shape)

    da = _inherit_chunks_like(base_cube, da)

    # Ép coord về chuỗi thuần, bỏ encoding object để tránh lỗi VLenUTF8
    if "symbol" in da.coords:
        da = da.assign_coords(symbol=da.coords["symbol"].astype(str))
        da.coords["symbol"].encoding = {}
    if "feature" in da.coords:
        da = da.assign_coords(feature=da.coords["feature"].astype(str))
        da.coords["feature"].encoding = {}

    ds = da.to_dataset(name="values")
    ds.attrs["params"] = params

    if "symbol" in ds.coords:
        symbol_coord = ds.coords["symbol"]
        symbol_preview = symbol_coord.values.tolist()[:5]
        print("[wrap] symbol coord dtype:", symbol_coord.dtype)
        print("[wrap] symbol coord python type:", type(symbol_coord.values).__name__)
        print("[wrap] symbol coord preview:", symbol_preview)
    if "feature" in ds.coords:
        feature_data = ds.coords["feature"].values.tolist()
        print("[wrap] feature coord:", feature_data)
    if "time" in ds.coords:
        time_coord = ds.coords["time"]
        print("[wrap] time coord dtype:", time_coord.dtype, "len:", time_coord.size)

    return ds

def _make_name(prefix: str, short: str, params: Dict[str, Any]) -> str:
    import json as _json
    import hashlib as _hashlib
    dumped = _json.dumps(params, sort_keys=True)
    suffix = _hashlib.sha1(dumped.encode()).hexdigest()[:4]
    name = f"{prefix}_{short}_{suffix}"
    print("[_make_name] params:", dumped)
    print("[_make_name] generated:", name)
    return name

# -------------------------
# Elementwise / simple rolling
# -------------------------
def make_return(cube: xr.Dataset, lookback: int = 1) -> Tuple[str, xr.Dataset]:
    """Log-return k-bar: log(C_t) - log(C_{t-k})."""
    close = cube["values"].sel(feature="Close")
    print(close)
    ret = np.log(close / close.shift(time=lookback))
    print(ret)
    params = {"lookback": lookback, "method": "log_return"}
    name = _make_name("ret", f"lag{lookback}", params)
    print(name)
    return name, _wrap_single_feature(cube, ret, f"return_lag{lookback}", params)



def make_volatility(cube: xr.Dataset, window: int = 30, strict_tminus1: bool = True) -> Tuple[str, xr.Dataset]:
    """Rolling std of log-return; optional strict t-1 stats to tránh leak."""
    close = cube["values"].sel(feature="Close")
    r1 = np.log(close / close.shift(time=1))
    x = r1.shift(time=1) if strict_tminus1 else r1
    vol = x.rolling(time=window, min_periods=window).std()
    params = {"window": window, "method": "rolling_std", "strict_tminus1": strict_tminus1}
    name = _make_name("vol", f"win{window}", params)
    return name, _wrap_single_feature(cube, vol, f"volatility_{window}", params)

def make_zscore(cube: xr.Dataset, source_feature: str, window: int, strict_tminus1: bool = True):
    """Z-score rolling: (x - mean)/(std+eps), stats theo t-1 nếu strict."""
    x = cube["values"].sel(feature=source_feature)
    x_stat = x.shift(time=1) if strict_tminus1 else x
    mu = x_stat.rolling(time=window, min_periods=window).mean()
    sd = x_stat.rolling(time=window, min_periods=window).std()
    z = (x - mu) / (sd + EPS)
    params = {"src": source_feature, "window": window, "strict_tminus1": strict_tminus1}
    name = _make_name("z", f"{source_feature}_w{window}", params)
    return name, _wrap_single_feature(cube, z, f"z_{source_feature}_{window}", params)

def make_volume_ratio(cube: xr.Dataset, window: int = 60):
    """Volume / EMA/MA(window) — dùng MA mặc định cho đơn giản & ổn định."""
    v = cube["values"].sel(feature="Volume")
    ma = v.rolling(time=window, min_periods=window).mean()
    ratio = v / (ma + EPS)
    params = {"window": window, "method": "vol/ma"}
    name = _make_name("liq", f"volratio_w{window}", params)
    return name, _wrap_single_feature(cube, ratio, f"vol_ratio_{window}", params)

def make_taker_buy_ratio(cube: xr.Dataset):
    """TB_base / Volume, bound [0,1] (khi dữ liệu hợp lệ)."""
    tb = cube["values"].sel(feature="TakerBuyBase")
    v  = cube["values"].sel(feature="Volume")
    r = tb / (v + EPS)
    params = {"method": "taker_buy_ratio"}
    name = _make_name("of", "taker_buy_ratio", params)
    return name, _wrap_single_feature(cube, r, "taker_buy_ratio", params)

# -------------------------
# Stateful ops via apply_ufunc (EMA, ATR, RSI)
# -------------------------
def _ema_1d(x: np.ndarray, alpha: float, debug=True, label="") -> np.ndarray:
    y = np.empty_like(x, dtype=np.float64)
    s = np.nan
    for i, v in enumerate(x):
        if np.isnan(v):
            y[i] = s
            continue
        s = v if np.isnan(s) else alpha * v + (1.0 - alpha) * s
        y[i] = s
        if debug and (i < 5 or i == len(x) - 1):
            print(f"{label}[{i}]: x={v:.6f}, ema={s:.6f}")
   
    return y

def _ema_da(x: xr.DataArray, span: int) -> xr.DataArray:
    alpha = 2.0 / (span + 1.0)
    # return xr.apply_ufunc(
    #     _ema_1d, x, alpha,
    #     input_core_dims=[["time"], []],
    #     output_core_dims=[["time"]],
    #     vectorize=True,
    #     dask="parallelized",
    #     output_dtypes=[np.float64],
    # )
    return xr.apply_ufunc(
        _ema_1d, x, alpha,
        input_core_dims=[["time"], []],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.float64],
    )


def make_ema_close(cube: xr.Dataset, span: int = 20):
    """EMA(Close, span) với apply_ufunc (chạy được cả dask)."""
    c = cube["values"].sel(feature="Close")
    ema = _ema_da(c, span)
    params = {"span": span, "method": "ema"}
    name = _make_name("ema", f"C_{span}", params)
    return name, _wrap_single_feature(cube, ema, f"ema_close_{span}", params)

def make_ema_ratio(cube: xr.Dataset, span: int = 20):
    """C/EMA(C)-1."""
    c = cube["values"].sel(feature="Close")
    ema = _ema_da(c, span)
    ratio = c / (ema + EPS) - 1.0
    params = {"span": span, "method": "ema_ratio"}
    name = _make_name("trend", f"emar_{span}", params)
    return name, _wrap_single_feature(cube, ratio, f"ema_ratio_{span}", params)

def make_atr(cube: xr.Dataset, span: int = 14):
    """ATR = EMA(TR, span)."""
    H = cube["values"].sel(feature="High")
    L = cube["values"].sel(feature="Low")
    C = cube["values"].sel(feature="Close")
    Cprev = C.shift(time=1)
    tr = xr.ufuncs.maximum(H - L, xr.ufuncs.maximum(abs(H - Cprev), abs(L - Cprev)))
    atr = _ema_da(tr, span)
    params = {"span": span, "method": "atr=ema(TR)"}
    name = _make_name("risk", f"atr_{span}", params)
    return name, _wrap_single_feature(cube, atr, f"atr_{span}", params)

def make_rsi(cube: xr.Dataset, span: int = 14):
    """RSI = 100 * EMA(gain) / EMA(|delta|)."""
    C = cube["values"].sel(feature="Close")
    d = C - C.shift(time=1)
    gain = xr.where(d > 0, d, 0.0)
    mag  = abs(d)
    up = _ema_da(gain, span)
    dm = _ema_da(mag,  span)
    rsi = 100.0 * (up / (dm + EPS))
    params = {"span": span, "method": "rsi=ema(gain)/ema(|delta|)"}
    name = _make_name("mom", f"rsi_{span}", params)
    return name, _wrap_single_feature(cube, rsi, f"rsi_{span}", params)
