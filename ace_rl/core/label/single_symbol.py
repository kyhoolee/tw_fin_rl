#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Sequence, Optional, Tuple, Dict, List
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import random
import json

try:
    from numba import njit
    NUMBA = True
except Exception:
    NUMBA = False


# ============ Logging ============

def log(s: str):  # flush ngay để xem realtime
    print(s, flush=True)


# ============ Chọn 1 symbol (Close/High/Low) ============

# def select_symbol(ds: xr.Dataset, symbol: str) -> xr.Dataset:
#     """
#     Trả về Dataset time-indexed gồm Close/High/Low (float32).
#     Không fallback: thiếu feature -> NaN (để strict NaN kích hoạt).
#     """
#     sub = ds.sel(symbol=symbol)
#     need = ["Close", "High", "Low"]
#     out = {}
#     for f in need:
#         if f in ds.coords["feature"].values:
#             out[f] = sub["values"].sel(feature=f).astype("float32")
#         else:
#             shape = (sub.sizes["time"],)
#             out[f] = xr.DataArray(
#                 np.full(shape, np.nan, dtype=np.float32),
#                 coords=dict(time=sub.coords["time"]),
#                 dims=("time",),
#                 name=f,
#             )
#     return xr.Dataset(out)

def select_symbol(ds: xr.Dataset, symbol: str) -> xr.Dataset:
    sub = ds.sel(symbol=symbol)
    need = ["Close", "High", "Low"]
    out = {}
    for f in need:
        if f in ds.coords["feature"].values:
            da = sub["values"].sel(feature=f)
            # Loại bỏ scalar coord 'feature' để tránh MergeError:
            if "feature" in da.coords:
                da = da.reset_coords("feature", drop=True)
            # Hoặc: da = da.squeeze(drop=True)  # cũng bỏ coord thừa nếu còn
            out[f] = da.astype("float32")
        else:
            shape = (sub.sizes["time"],)
            out[f] = xr.DataArray(
                np.full(shape, np.nan, dtype=np.float32),
                coords=dict(time=sub.coords["time"]),
                dims=("time",),
                name=f,
            )
    return xr.Dataset(out)



# ============ Returns & Volatility ============

def logret_from_close(close: xr.DataArray, H: int) -> xr.DataArray:
    p_now = close
    p_fut = close.shift(time=-H)
    r = (np.log(p_fut) - np.log(p_now)).astype("float32")
    r.name = f"ret_log__H{H}"
    return r

def multi_horizon_returns(close: xr.DataArray, horizons: Sequence[int]) -> xr.Dataset:
    data = {f"ret_log__H{H}": logret_from_close(close, H) for H in horizons}
    return xr.Dataset(data)

def rolling_vol(close: xr.DataArray, win: int = 60) -> xr.DataArray:
    r1 = (np.log(close.shift(time=-1)) - np.log(close)).astype("float32")
    vol = r1.rolling(time=win, min_periods=win).std().astype("float32")
    vol.name = f"vol__W{win}"
    return vol


# ============ Triple-Barrier (strict NaN) ============

@dataclass
class TBConfig:
    horizon: int               # vertical barrier (bars)
    up_mult: float             # +% (pct-mode) hoặc hệ số * vol (vol-mode)
    dn_mult: float             # -% (pct-mode) hoặc hệ số * vol (vol-mode)
    use_vol: bool = False
    vol_window: int = 60
    min_ret_neutral: float = 0.0
    strict_nan_window: bool = True  # bất kỳ NaN trong [t..t+H] => output NaN

if NUMBA:
    @njit(cache=True, fastmath=True)
    def _tb_scan_numba(
        close, high, low, vol, H, up_mult, dn_mult, use_vol, min_ret_neutral, strict_nan_window
    ):
        n = close.shape[0]
        label = np.full(n, np.nan, dtype=np.float32)      # +1/0/-1/NaN
        exit_ret = np.full(n, np.nan, dtype=np.float32)   # log-return tại lúc thoát
        t_hit = np.full(n, np.nan, dtype=np.float64)      # index thoát (float để có NaN)

        for t in range(n):
            p0 = close[t]
            if np.isnan(p0):
                continue

            t_end = min(n-1, t + H)

            if use_vol:
                v = vol[t]
                if np.isnan(v):
                    continue
                up = p0 * (1.0 + up_mult * v)
                dn = p0 * (1.0 - dn_mult * v)
            else:
                up = p0 * (1.0 + up_mult)
                dn = p0 * (1.0 - dn_mult)

            has_nan_window = False
            hit = 0
            hit_idx = -1

            for k in range(t+1, t_end+1):
                ck = close[k]; hk = high[k]; lk = low[k]
                if np.isnan(ck) or np.isnan(hk) or np.isnan(lk):
                    has_nan_window = True
                    break
                if hk >= up:
                    hit = 1; hit_idx = k
                    break
                if lk <= dn:
                    hit = -1; hit_idx = k
                    break

            if strict_nan_window and has_nan_window:
                continue

            if hit != 0:
                label[t] = 1.0 if hit == 1 else -1.0
                exit_ret[t] = np.float32(np.log(close[hit_idx]) - np.log(p0))
                t_hit[t] = np.float64(hit_idx)
            else:
                if strict_nan_window and np.isnan(close[t_end]):
                    continue
                r = np.log(close[t_end]) - np.log(p0)
                if r > min_ret_neutral:
                    label[t] = 1.0
                elif r < -min_ret_neutral:
                    label[t] = -1.0
                else:
                    label[t] = 0.0
                exit_ret[t] = np.float32(r)
                t_hit[t] = np.float64(t_end)

        return label, exit_ret, t_hit
else:
    def _tb_scan_numba(*args, **kwargs):
        raise RuntimeError("Vui lòng `pip install numba` để chạy triple-barrier nhanh.")


def triple_barrier(
    ds_sym: xr.Dataset, cfg: TBConfig, precomputed_vol: Optional[xr.DataArray] = None
) -> xr.Dataset:
    """
    Output:
      - tb_label__H{H}__{pct|vol} : float32 in {-1,0,1,NaN}
      - tb_exit_ret__H{H}         : float32
      - tb_t_hit__H{H}            : float32 (index, NaN nếu invalid)
    """
    close = ds_sym["Close"].astype("float32").values
    high  = ds_sym["High"].astype("float32").values
    low   = ds_sym["Low"].astype("float32").values

    if cfg.use_vol:
        vol_da = precomputed_vol
        if vol_da is None:
            r1 = (np.log(ds_sym["Close"].shift(time=-1)) - np.log(ds_sym["Close"])).astype("float32")
            vol_da = r1.rolling(time=cfg.vol_window, min_periods=cfg.vol_window).std().astype("float32")
        vol = vol_da.values
    else:
        vol = np.full_like(close, np.nan, dtype=np.float32)

    lab, exret, thit = _tb_scan_numba(
        close, high, low, vol,
        cfg.horizon, cfg.up_mult, cfg.dn_mult, cfg.use_vol, cfg.min_ret_neutral, cfg.strict_nan_window
    )

    coords = dict(time=ds_sym.coords["time"])
    name_suffix = "vol" if cfg.use_vol else "pct"
    return xr.Dataset(
        {
            f"tb_label__H{cfg.horizon}__{name_suffix}": (("time",), lab.astype("float32")),
            f"tb_exit_ret__H{cfg.horizon}": (("time",), exret.astype("float32")),
            f"tb_t_hit__H{cfg.horizon}": (("time",), thit.astype("float32")),
        },
        coords=coords
    )


# ============ Build labels (1 symbol) ============

def build_single_symbol_labels(
    ds: xr.Dataset,
    symbol: str,
    horizons: Sequence[int] = (1, 5, 15, 60),
    tb_cfg_pct: Optional[Tuple[float, float, int, float]] = (0.01, 0.005, 60, 0.0),
    tb_cfg_vol: Optional[Tuple[float, float, int, int, float]] = None,
) -> xr.Dataset:
    """
    Trả về Dataset gồm ret_log__H*, tb_* (pct), tb_* (vol) (tuỳ chọn).
    """
    ds_sym = select_symbol(ds, symbol)
    out = []

    # returns
    out.append(multi_horizon_returns(ds_sym["Close"], horizons))

    # triple-barrier (pct)
    if tb_cfg_pct is not None:
        up, dn, H, neutral = tb_cfg_pct
        cfg = TBConfig(
            horizon=H, up_mult=up, dn_mult=dn, use_vol=False,
            min_ret_neutral=neutral, strict_nan_window=True
        )
        out.append(triple_barrier(ds_sym, cfg))

    # triple-barrier (vol)
    if tb_cfg_vol is not None:
        upm, dnm, H, vw, neutral = tb_cfg_vol
        vol_da = rolling_vol(ds_sym["Close"], win=vw)
        cfgv = TBConfig(
            horizon=H, up_mult=upm, dn_mult=dnm, use_vol=True,
            vol_window=vw, min_ret_neutral=neutral, strict_nan_window=True
        )
        out.append(triple_barrier(ds_sym, cfgv, precomputed_vol=vol_da))

    ds_out = xr.merge(out, compat="override", combine_attrs="drop_conflicts")
    return ds_out


# ============ Assemble cube time × symbol × label ============

def assemble_labels_cube(
    ds: xr.Dataset,
    symbols: List[str],
    horizons: Sequence[int],
    tb_cfg_pct: Optional[Tuple[float, float, int, float]],
    tb_cfg_vol: Optional[Tuple[float, float, int, int, float]],
) -> xr.Dataset:
    """
    Chạy qua tất cả symbol, chuẩn hóa danh sách biến label, ghép thành 1 DataArray:
      values(time, symbol, label) float32
    Coord:
      - time: copy từ cube
      - symbol: như cube
      - label: tên biến label (string)
    """
    label_names: Optional[List[str]] = None
    per_symbol_2d: List[np.ndarray] = []

    for i, sym in enumerate(symbols, 1):
        log(f"[{i}/{len(symbols)}] Build labels for {sym} ...")
        ds_lbl = build_single_symbol_labels(
            ds, sym,
            horizons=horizons,
            tb_cfg_pct=tb_cfg_pct,
            tb_cfg_vol=tb_cfg_vol
        )

        # lần đầu: chốt danh sách label theo thứ tự ổn định
        if label_names is None:
            label_names = sorted(list(ds_lbl.data_vars.keys()))
            log(f"[INFO] label variables: {label_names}")

        # đảm bảo đủ biến; thiếu -> NaN
        col_arrays = []
        T = ds_lbl.sizes["time"]
        for v in label_names:
            if v in ds_lbl.data_vars:
                a = ds_lbl[v].values.astype("float32", copy=False)
            else:
                a = np.full((T,), np.nan, dtype=np.float32)
            col_arrays.append(a)
        # ghép về (time, label)
        mat = np.stack(col_arrays, axis=1)  # (T, L)
        per_symbol_2d.append(mat)

    # Stack symbol: (time, S, L)
    values = np.stack(per_symbol_2d, axis=1).astype("float32")  # (T, S, L)

    # Wrap xarray
    time_coord = ds.coords["time"]
    label_coord = np.array(label_names, dtype=object)
    symbol_coord = np.array(symbols, dtype=object)
    da = xr.DataArray(
        values,
        dims=("time", "symbol", "label"),
        coords=dict(time=time_coord, symbol=symbol_coord, label=label_coord),
        name="values"
    )
    # (tuỳ chọn) mask hợp lệ (!= NaN)
    mask = xr.DataArray(
        ~np.isnan(values),
        dims=("time", "symbol", "label"),
        coords=dict(time=time_coord, symbol=symbol_coord, label=label_coord),
        name="mask"
    )
    ds_out = xr.Dataset(dict(values=da, mask=mask))
    ds_out.attrs["label_builder"] = "returns + triple_barrier(strict_nan)"
    return ds_out


# ============ Stats printer ============

def print_symbol_stats(symbol: str, ds_lbl: xr.Dataset):
    rows = []
    tb_label_vars = [v for v in ds_lbl.data_vars if v.startswith("tb_label__H")]
    for v in sorted(tb_label_vars):
        arr = ds_lbl[v].values
        total = arr.size
        n_nan = int(np.isnan(arr).sum())
        mask_valid = ~np.isnan(arr)
        vals = arr[mask_valid]
        c_pos = int((vals == 1.0).sum())
        c_zero = int((vals == 0.0).sum())
        c_neg = int((vals == -1.0).sum())
        cov = 100.0 * (total - n_nan) / total if total else 0.0
        rows.append(dict(
            symbol=symbol, var=v, total=total, nan=n_nan, coverage_pct=round(cov,2),
            pos=c_pos, zero=c_zero, neg=c_neg
        ))
    ret_vars = [v for v in ds_lbl.data_vars if v.startswith("ret_log__H")]
    for v in sorted(ret_vars):
        arr = ds_lbl[v].values
        n_nan = int(np.isnan(arr).sum())
        cov = 100.0 * (arr.size - n_nan) / arr.size if arr.size else 0.0
        rows.append(dict(
            symbol=symbol, var=v, total=arr.size, nan=n_nan, coverage_pct=round(cov,2),
            pos=None, zero=None, neg=None
        ))
    df = pd.DataFrame(rows)
    if len(df):
        log(f"\n[STATS] {symbol}")
        with pd.option_context("display.max_rows", 200, "display.width", 140):
            log(df.to_string(index=False))
    return df


# ============ Debug: random explain (step-by-step) ============

def explain_tb_once(ds_sym: xr.Dataset, t_idx: int, cfg: TBConfig) -> Dict:
    """
    Giải thích chi tiết quyết định TB tại time-index t_idx.
    Trả về dict gồm p0, barriers, steps[], result.
    """
    time_vals = ds_sym["time"].values
    T = ds_sym.sizes["time"]
    t_end = min(T-1, t_idx + cfg.horizon)

    close = ds_sym["Close"].values
    high  = ds_sym["High"].values
    low   = ds_sym["Low"].values

    if np.isnan(close[t_idx]):
        return {"t": int(t_idx), "time": str(time_vals[t_idx]), "reason": "p0 is NaN", "label": np.nan}

    if cfg.use_vol:
        vol_da = rolling_vol(ds_sym["Close"], win=cfg.vol_window)
        v = float(vol_da.isel(time=t_idx).values)
        if np.isnan(v):
            return {"t": int(t_idx), "time": str(time_vals[t_idx]), "reason": "vol[t] is NaN", "label": np.nan}
        up = close[t_idx] * (1.0 + cfg.up_mult * v)
        dn = close[t_idx] * (1.0 - cfg.dn_mult * v)
        mode = "vol"
        updn_info = dict(vol=v, up_mult=cfg.up_mult, dn_mult=cfg.dn_mult)
    else:
        up = close[t_idx] * (1.0 + cfg.up_mult)
        dn = close[t_idx] * (1.0 - cfg.dn_mult)
        mode = "pct"
        updn_info = dict(up_pct=cfg.up_mult, dn_pct=cfg.dn_mult)

    steps = []
    # Quét, đồng thời kiểm tra NaN trong cửa sổ
    for k in range(t_idx+1, t_end+1):
        ck = close[k]; hk = high[k]; lk = low[k]
        if np.isnan(ck) or np.isnan(hk) or np.isnan(lk):
            return {
                "t": int(t_idx),
                "time": str(time_vals[t_idx]),
                "p0": float(close[t_idx]),
                "mode": mode,
                "barriers": {"up": float(up), "dn": float(dn)},
                "updn_info": updn_info,
                "horizon": cfg.horizon,
                "strict_nan_window": True,
                "steps": steps,
                "reason": f"NaN in window at k={k}",
                "label": np.nan,
            }
        steps.append({
            "k": int(k),
            "time": str(time_vals[k]),
            "close": float(ck),
            "high": float(hk),
            "low": float(lk),
            "hit": "UP" if hk >= up else ("DN" if lk <= dn else "")
        })
        if hk >= up:
            r = float(np.log(close[k]) - np.log(close[t_idx]))
            return {
                "t": int(t_idx),
                "time": str(time_vals[t_idx]),
                "p0": float(close[t_idx]),
                "mode": mode,
                "barriers": {"up": float(up), "dn": float(dn)},
                "updn_info": updn_info,
                "horizon": cfg.horizon,
                "strict_nan_window": True,
                "steps": steps,
                "result": {"type": "TP", "k": int(k), "time": str(time_vals[k]), "exit_ret": r},
                "label": 1.0
            }
        if lk <= dn:
            r = float(np.log(close[k]) - np.log(close[t_idx]))
            return {
                "t": int(t_idx),
                "time": str(time_vals[t_idx]),
                "p0": float(close[t_idx]),
                "mode": mode,
                "barriers": {"up": float(up), "dn": float(dn)},
                "updn_info": updn_info,
                "horizon": cfg.horizon,
                "strict_nan_window": True,
                "steps": steps,
                "result": {"type": "SL", "k": int(k), "time": str(time_vals[k]), "exit_ret": r},
                "label": -1.0
            }

    # Vertical barrier
    if np.isnan(close[t_end]):
        return {
            "t": int(t_idx),
            "time": str(time_vals[t_idx]),
            "p0": float(close[t_idx]),
            "mode": mode,
            "barriers": {"up": float(up), "dn": float(dn)},
            "updn_info": updn_info,
            "horizon": cfg.horizon,
            "strict_nan_window": True,
            "steps": steps,
            "reason": f"close[t_end] is NaN at k={t_end}",
            "label": np.nan
        }
    r = float(np.log(close[t_end]) - np.log(close[t_idx]))
    if r > cfg.min_ret_neutral:
        lab = 1.0; res = "VERT_UP"
    elif r < -cfg.min_ret_neutral:
        lab = -1.0; res = "VERT_DN"
    else:
        lab = 0.0; res = "VERT_NEU"
    return {
        "t": int(t_idx),
        "time": str(time_vals[t_idx]),
        "p0": float(close[t_idx]),
        "mode": mode,
        "barriers": {"up": float(up), "dn": float(dn)},
        "updn_info": updn_info,
        "horizon": cfg.horizon,
        "strict_nan_window": True,
        "steps": steps,
        "result": {"type": res, "k": int(t_end), "time": str(time_vals[t_end]), "exit_ret": r},
        "label": lab
    }

def print_random_debug(
    ds: xr.Dataset,
    symbol: str,
    horizons: Sequence[int],
    tb_cfg_pct: Optional[Tuple[float, float, int, float]],
    n_samples: int = 3,
    seed: Optional[int] = 123
):
    """
    Random pick vài time-index hợp lệ và in giải thích:
      - forward returns cho mọi H
      - triple-barrier (pct) step-by-step
    """
    random.seed(seed)
    ds_sym = select_symbol(ds, symbol)

    # build nhanh để lấy masks lựa chọn
    ds_lbl = build_single_symbol_labels(ds, symbol, horizons=horizons, tb_cfg_pct=tb_cfg_pct, tb_cfg_vol=None)
    tb_vars = [v for v in ds_lbl.data_vars if v.startswith("tb_label__H")]
    assert len(tb_vars) >= 1, "Không thấy biến tb_label__H*"
    tb_name = tb_vars[0]
    arr = ds_lbl[tb_name].values
    valid_idxs = np.where(~np.isnan(arr))[0]
    if len(valid_idxs) == 0:
        log(f"[DEBUG] {symbol}: không tìm thấy điểm TB hợp lệ để debug.")
        return

    sample_idxs = random.sample(list(valid_idxs), min(n_samples, len(valid_idxs)))
    log(f"\n[DEBUG RANDOM] {symbol} | tb={tb_name} | samples={len(sample_idxs)}")
    up, dn, H, neutral = tb_cfg_pct
    cfg = TBConfig(horizon=H, up_mult=up, dn_mult=dn, use_vol=False, min_ret_neutral=neutral, strict_nan_window=True)

    for t in sample_idxs:
        # explain TB
        expl = explain_tb_once(ds_sym, t, cfg)
        log("—" * 80)
        log(json.dumps({
            "t_idx": expl.get("t"), "time": expl.get("time"),
            "tb_label_explained": expl.get("label"),
            "barriers": expl.get("barriers"),
            "result": expl.get("result", expl.get("reason")),
        }, ensure_ascii=False, indent=2))

        # In vài bước đầu (tránh quá dài)
        steps = expl.get("steps", [])
        if steps:
            max_show = min(12, len(steps))
            log(f"[steps 1..{max_show}/{len(steps)}]")
            for s in steps[:max_show]:
                log(f" k={s['k']} time={s['time']} close={s['close']:.6f} "
                    f"high={s['high']:.6f} low={s['low']:.6f} hit={s['hit']}")

        # returns cho tất cả H
        for Hh in horizons:
            rvar = f"ret_log__H{Hh}"
            rv = float(ds_lbl[rvar].isel(time=t).values)
            log(f"  ret_log__H{Hh} @t  = {rv:.8f}" if not np.isnan(rv) else f"  ret_log__H{Hh} @t  = NaN")

        # Giá trị label/t_exit trong ds_lbl để so
        for v in [tb_name, tb_name.replace("tb_label", "tb_exit_ret"), tb_name.replace("tb_label", "tb_t_hit")]:
            if v in ds_lbl:
                vv = ds_lbl[v].isel(time=t).values.item()
                log(f"  {v} @t = {vv}")
    log("—" * 80)


# ============ Main & Zarr writer ============

def main():
    ap = argparse.ArgumentParser(description="Build single-symbol labels → assemble cube (time × symbol × label) and write Zarr.")
    ap.add_argument("--cube-zarr", required=True, help="Path to cube.zarr (time × symbol × feature)")
    ap.add_argument("--out-zarr", required=True, help="Output Zarr path for labels cube")
    ap.add_argument("--group", default="labels", help="Zarr group name to write (default: labels)")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1,5,15,60], help="Forward horizons (bars)")
    # triple-barrier (pct)
    ap.add_argument("--tb-up-pct", type=float, default=0.01, help="Upper barrier +pct (e.g., 0.01 = +1%)")
    ap.add_argument("--tb-dn-pct", type=float, default=0.005, help="Lower barrier -pct (e.g., 0.005 = -0.5%)")
    ap.add_argument("--tb-H", type=int, default=60, help="Vertical barrier horizon (bars)")
    ap.add_argument("--tb-neutral", type=float, default=0.0, help="Neutral band for vertical decision")
    # vol-based (optional) — có thể bật thêm (sẽ tạo thêm biến trong label-dim)
    ap.add_argument("--tb-vol", action="store_true", help="Enable vol-based TB (in addition to pct)")
    ap.add_argument("--tb-vol-up", type=float, default=2.0, help="UP = p0*(1 + up_mult*vol)")
    ap.add_argument("--tb-vol-dn", type=float, default=2.0, help="DN = p0*(1 - dn_mult*vol)")
    ap.add_argument("--tb-vol-window", type=int, default=60, help="Vol rolling window")
    ap.add_argument("--tb-vol-neutral", type=float, default=0.0, help="Neutral band for vertical in vol-mode")
    # debug
    ap.add_argument("--print-stats", action="store_true", help="Print per-symbol stats to stdout")
    ap.add_argument("--print-random-debug", action="store_true", help="Print step-by-step debug for a random few timestamps (first symbol)")
    ap.add_argument("--debug-samples", type=int, default=3, help="Number of random samples for debug")
    ap.add_argument("--seed", type=int, default=123, help="Random seed for debug sampling")
    args = ap.parse_args()

    if not NUMBA:
        log("[WARN] numba chưa cài — hãy `pip install numba` để tăng tốc triple-barrier.")
        # vẫn chạy được returns; TB yêu cầu numba theo code này

    log(f"[LOAD] Opening cube: {args.cube_zarr}")
    ds = xr.open_zarr(str(Path(args.cube_zarr)), chunks={"time": 16384, "symbol": 4, "feature": 9})
    symbols = list(map(str, ds.coords["symbol"].values))
    log(f"[INFO] Symbols: {symbols}")

    tb_cfg_pct = (args.tb_up_pct, args.tb_dn_pct, args.tb_H, args.tb_neutral)
    tb_cfg_vol = None
    if args.tb_vol:
        tb_cfg_vol = (args.tb_vol_up, args.tb_vol_dn, args.tb_H, args.tb_vol_window, args.tb_vol_neutral)

    # (optional) in-place stats per symbol trước khi ghép
    if args.print_stats:
        all_stats = []
        for sym in symbols:
            ds_lbl_sym = build_single_symbol_labels(ds, sym, horizons=args.horizons, tb_cfg_pct=tb_cfg_pct, tb_cfg_vol=tb_cfg_vol)
            df = print_symbol_stats(sym, ds_lbl_sym)
            all_stats.append(df)
        if len(all_stats):
            df_all = pd.concat(all_stats, ignore_index=True)
            out_csv = Path(args.out_zarr).with_suffix(".stats.csv")
            df_all.to_csv(out_csv, index=False)
            log(f"[SUMMARY] wrote per-symbol stats → {out_csv}")

    # assemble full cube (time × symbol × label)
    ds_labels_cube = assemble_labels_cube(ds, symbols, args.horizons, tb_cfg_pct, tb_cfg_vol)
    # metadata
    ds_labels_cube.attrs.update({
        "horizons": json.dumps(args.horizons),
        "tb_cfg_pct": json.dumps(dict(up_pct=args.tb_up_pct, dn_pct=args.tb_dn_pct, H=args.tb_H, neutral=args.tb_neutral)),
        "tb_cfg_vol": json.dumps(dict(enabled=bool(args.tb_vol), up=args.tb_vol_up, dn=args.tb_vol_dn,
                                      H=args.tb_H, window=args.tb_vol_window, neutral=args.tb_vol_neutral)),
        "strict_nan_window": True,
    })

    # write Zarr (group=labels), giống cube: values + mask
    out_store = str(Path(args.out_zarr))
    enc = {
        "values": {"chunks": (16384, len(symbols), len(ds_labels_cube.coords["label"]))},
        "mask":   {"chunks": (16384, len(symbols), len(ds_labels_cube.coords["label"]))},
    }
    log(f"[WRITE] to Zarr: {out_store} (group={args.group})")
    ds_labels_cube.to_zarr(out_store, group=args.group, mode="w", encoding=enc)

    # optional: print random debug (first symbol)
    if args.print_random_debug and len(symbols):
        print_random_debug(
            ds, symbols[0],
            horizons=args.horizons,
            tb_cfg_pct=tb_cfg_pct,
            n_samples=args.debug_samples,
            seed=args.seed
        )

    log("[DONE]")


if __name__ == "__main__":
    main()
