# -*- coding: utf-8 -*-
"""Portfolio baselines for 4 synchronized tokens (no-leak):
- EW equal-weight (periodic rebalance)
- Momentum top-k (cross-sectional)
- Mean-reversion pair spread (delta-neutral)

Assumptions:
- Signals at t are computed using data up to t-1.
- Execution occurs at close[t].
- Fees model: Binance taker per-side fee (default 7.5 bps = 0.075%).
- This version adds verbose progress logging and faster rolling via NumPy.

Usage (unchanged public API):
    syms, out, metrics = run_all_baselines(split_parquet_path, fee_bps=7.5, slippage_bps=0.0)

Extras (optional):
    run_all_baselines(..., verbose=True, progress_every=10000)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional, Sequence

# try tqdm for pretty progress if available
try:
    from tqdm import trange
except Exception:  # pragma: no cover
    trange = None

# ---------- Loader ----------

def load_tensor_safe(
    split_path: str,
    features: Sequence[str] = ("Open","High","Low","Close","Volume"),
    dtype=np.float32,
    strict: bool = True,          # True: yêu cầu đồng bộ tuyệt đối (không NaN)
    return_mask: bool = False,    # True: trả thêm mask (True = hợp lệ)
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Optional[np.ndarray]]:
    """
    Đọc parquet OHLCV → pivot → xếp block theo symbol rồi feature → tensor (T, N, F)
    - Không dùng reshape dựa vào thứ tự cột phẳng (tránh lệch âm thầm).
    - Có kiểm tra strict đồng bộ (timestamp × symbol).
    - Có mask khi cho phép thiếu (strict=False).

    Returns:
      x:        (T, N, F) ndarray
      times:    (T,) mảng datetime (python datetime)
      symbols:  list[str] độ dài N
      feats:    list[str] độ dài F
      mask:     (T, N, F) bool, True = non-NaN (nếu return_mask=True) else None
    """
    # 1) Đọc parquet
    df = pd.read_parquet(split_path)
    # Yêu cầu cột bắt buộc
    required_cols = {"open_time", "symbol", *features}
    missing_cols = required_cols.difference(df.columns)
    assert not missing_cols, f"Thiếu cột bắt buộc: {sorted(missing_cols)}"
    # Sắp theo time/symbol để có trật tự ổn định
    df = df.sort_values(["open_time", "symbol"]).reset_index(drop=True)

    print(f"[load_tensor_safe] Loaded {df.shape} from {split_path}")

    # 2) Danh sách symbol & feature ổn định
    symbols: List[str] = sorted(df["symbol"].unique())  # N symbols (sorted)
    feats: List[str] = list(features)                    # F features (giữ nguyên thứ tự đầu vào)
    N = len(symbols)
    F = len(feats)
    assert N > 0 and F > 0, "Số symbol/feature phải > 0"

    print(f"[load_tensor_safe] N={N} symbols, F={F} features, strict={strict}, return_mask={return_mask}")

    # 3) Kiểm tra đồng bộ (strict)
    T_unique = df["open_time"].nunique()
    if strict:
        # Mọi timestamp phải có đủ N symbol
        assert T_unique * N == len(df), (
            "Split is not per-timestamp synchronized. "
            "Dùng strict=False để chấp nhận thiếu và tạo mask."
        )

    # 4) Pivot: index = open_time (T), columns = (feature, symbol)
    #    Kết quả là DataFrame 2D với MultiIndex cột (level 0: feature, level 1: symbol).
    piv = df.pivot(index="open_time", columns="symbol", values=feats).sort_index()
    # Kiểm tra trục thời gian
    T = len(piv.index)
    assert T == T_unique, "Số timestamp sau pivot phải bằng số open_time duy nhất"

    # 5) Kiểm tra symbol trong pivot (strict) hoặc ghi nhận thiếu
    syms_in_piv = set(piv.columns.levels[1]) if isinstance(piv.columns, pd.MultiIndex) else set()
    missing_syms = [s for s in symbols if s not in syms_in_piv]
    if strict:
        assert not missing_syms, f"Thiếu symbol trong pivot (strict=True): {missing_syms}"

    # 6) Tạo block theo symbol → mỗi block shape (T, F)
    #    - Chọn sub DataFrame cho 1 symbol: (T, F)
    #    - Reindex theo feats để đảm bảo thứ tự cột đúng (feature dimension).
    #    - Nếu symbol vắng, tạo (T, F) toàn NaN.
    blocks: List[np.ndarray] = []  # sẽ có N phần tử, mỗi phần tử (T, F)
    for sym in symbols:
        if sym in syms_in_piv:
            # sub: DataFrame (T, F)
            sub = piv.xs(sym, level=1, axis=1)
            # Đảm bảo đúng thứ tự feature; cột thiếu => NaN
            sub = sub.reindex(columns=feats)
        else:
            # Symbol vắng hoàn toàn: tạo DataFrame NaN (T, F)
            sub = pd.DataFrame(index=piv.index, columns=feats, dtype=float)
        # arr: ndarray (T, F) — T = số timestamp, F = số feature
        arr = sub.to_numpy()
        # Kiểm tra shape từng block
        assert arr.shape == (T, F), f"Block cho {sym} phải có shape (T, F), nhận {arr.shape}"
        blocks.append(arr)

    # Kiểm tra tổng số block
    assert len(blocks) == N, f"Số block phải bằng số symbol N={N}, nhận {len(blocks)}"
    print(f"[load_tensor_safe] Data blocks ready: {[(b.shape) for b in blocks]}")

    # 7) Stack theo trục symbol (axis=1) → (T, N, F)
    #   - axis=0: thời gian T (giữ nguyên)
    #   - axis=1: symbol N (block index)
    #   - axis=2: feature F
    x = np.stack(blocks, axis=1).astype(dtype)
    assert x.shape == (T, N, F), f"x phải có shape (T, N, F) = {(T, N, F)}, nhận {x.shape}"
    print(f"[load_tensor_safe] Tensor x ready: {x.shape}, dtype={x.dtype}")

    # 8) Tạo mask non-NaN
    mask = ~np.isnan(x)
    assert mask.shape == (T, N, F), "mask phải trùng shape với x"
    print(f"[load_tensor_safe] Mask ready: {mask.shape}, dtype={mask.dtype}, n_valid={int(mask.sum())}")

    # 9) Nếu strict, đảm bảo không có NaN
    if strict:
        n_nan = int(np.isnan(x).sum())
        assert n_nan == 0, f"strict=True nhưng còn {n_nan} ô NaN trong dữ liệu"

    # 10) times
    times = piv.index.to_pydatetime()  # (T,)
    print(f"[load_tensor_safe] Times ready: {len(times)} timestamps from {times[0]} to {times[-1]}")

    return (x, times, symbols, feats, mask if return_mask else None)




# ---------- Costs & helpers ----------

def safe_div(a, b, eps=1e-12):
    return a / (b + eps)

def price_to_ret(close: np.ndarray) -> np.ndarray:
    r = np.zeros_like(close, dtype=np.float64)
    r[1:] = safe_div(close[1:], close[:-1]) - 1.0
    return r

def apply_txn_costs(weight_prev: np.ndarray, weight_new: np.ndarray, fee_bps=7.5, slippage_bps=0.0) -> float:
    """
    Binance-style per-side fee model.
    - delta L1 = sum |w_new - w_prev| (buys + sells as fraction of NAV)
    - cost = delta * (fee_rate + slippage_rate)
    """
    delta = float(np.abs(weight_new - weight_prev).sum())
    fee_rate = fee_bps / 1e4
    slip_rate = slippage_bps / 1e4
    return delta * (fee_rate + slip_rate)

# ---------- Metrics ----------

def perf_metrics(nav, dt_index=None) -> Dict[str, float]:

    nav = np.asarray(nav, dtype=np.float64)
    rets = np.diff(nav) / (nav[:-1] + 1e-12)

    if dt_index is None:
        ann = 525600.0  # minute bars per ~year
    else:
        dt = pd.to_datetime(dt_index)
        minutes = (dt[-1] - dt[0]).total_seconds() / 60.0
        ann = max(minutes, 1.0)

    mean = rets.mean()
    std  = rets.std(ddof=0) + 1e-12
    sharpe = mean / std * np.sqrt(ann)
    cagr = nav[-1] - 1.0

    dd = 1.0 - safe_div(nav, np.maximum.accumulate(nav))
    maxdd = float(dd.max())

    return {"CAGR≈": float(cagr), "Sharpe≈": float(sharpe), "MaxDD": maxdd}

# ---------- Baselines ----------

def ew_baseline(close: np.ndarray, fee_bps=7.5, rebalance_every=1440, slippage_bps=0.0,
                verbose: bool=False, progress_every: int=10000) -> np.ndarray:
    """
    Equal-weight portfolio with periodic rebalance.
    """
    T, N = close.shape
    nav = np.ones(T, dtype=np.float64)
    w_prev = np.ones(N) / N
    last_rb = 0

    it = range(1, T)
    if verbose and trange is not None:
        it = trange(1, T, desc="EW_daily", miniters=max(1, progress_every))
    for t in it:
        if verbose and trange is None and (t % progress_every == 0):
            print(f"[EW] t={t}/{T} nav={nav[t-1]:.6f}")

        if (t - last_rb) % rebalance_every == 0:
            w_tgt = np.ones(N) / N
            cost = apply_txn_costs(w_prev, w_tgt, fee_bps, slippage_bps)
            nav[t-1] *= (1 - cost)
            w_prev = w_tgt
            last_rb = t
        r_t = close[t] / close[t-1] - 1.0
        nav[t] = nav[t-1] * (1.0 + (w_prev * r_t).sum())
    return nav

def momentum_topk(close: np.ndarray, lookback=360, topk=2, fee_bps=7.5, rebalance_every=60,
                  slippage_bps=0.0, verbose: bool=False, progress_every: int=10000) -> np.ndarray:
    """
    Cross-sectional momentum: equal-weight top-k by trailing return.
    """
    T, N = close.shape
    nav = np.ones(T, dtype=np.float64)
    w_prev = np.zeros(N, dtype=np.float64)

    it = range(1, T)
    if verbose and trange is not None:
        it = trange(1, T, desc="MOM_top2_look360_rb60", miniters=max(1, progress_every))
    for t in it:
        if verbose and trange is None and (t % progress_every == 0):
            print(f"[MOM] t={t}/{T} nav={nav[t-1]:.6f}")

        if (t % rebalance_every == 0) and (t - 1 - lookback >= 0):
            past = safe_div(close[t-1], close[t-1-lookback])
            ranks = np.argsort(-past)
            sel = ranks[:max(1, min(topk, N))]
            w_tgt = np.zeros(N, dtype=np.float64)
            w_tgt[sel] = 1.0 / len(sel)
            cost = apply_txn_costs(w_prev, w_tgt, fee_bps, slippage_bps)
            nav[t-1] *= (1 - cost)
            w_prev = w_tgt
        r_t = close[t] / close[t-1] - 1.0
        nav[t] = nav[t-1] * (1.0 + (w_prev * r_t).sum())
    return nav

def _rolling_last_mean_std(arr: np.ndarray, window: int) -> Tuple[float, float]:
    """
    Compute mean/std of the *last* window slice quickly using NumPy.
    Assumes len(arr) >= window, returns (mean, std_ddof0).
    """
    win = arr[-window:]
    m = float(win.mean())
    # ddof=0 for population-like use (consistent with original code)
    v = float(win.std(ddof=0))
    # guard against near-zero
    return m, (v + 1e-12)

def pair_meanrev_spread(close: np.ndarray, pair=(0,1), z_window=180, entry_z=1.5, exit_z=0.5,
                        fee_bps=7.5, slippage_bps=0.0, max_gross=1.0,
                        verbose: bool=False, progress_every: int=10000) -> np.ndarray:
    """
    Simple pair mean-reversion on log-price spread(i - j).
    Positions are +/- max_gross/2 per leg (delta-neutral) when |z| > entry;
    flatten when |z| < exit. Signals at t use data up to t-1; execute at close[t].
    """
    i, j = pair
    logp = np.log(close)
    T = close.shape[0]
    nav = np.ones(T, dtype=np.float64)
    pos_i = 0.0; pos_j = 0.0

    it = range(1, T)
    if verbose and trange is not None:
        it = trange(1, T, desc="PAIR_MR_BTC_ETH", miniters=max(1, progress_every))
    for t in it:
        if verbose and trange is None and (t % progress_every == 0):
            print(f"[PAIR] t={t}/{T} nav={nav[t-1]:.6f} pos=({pos_i:.2f},{pos_j:.2f})")

        if t - z_window >= 1:
            # spread up to t-1
            s = logp[:t, i] - logp[:t, j]
            # fast last-window stats
            m, std = _rolling_last_mean_std(s, z_window)
            z = (s[-1] - m) / std

            w_i_new = w_j_new = 0.0
            az = abs(z)
            if az > entry_z:
                if z > 0:  # i rich vs j → short i, long j
                    w_i_new, w_j_new = -max_gross/2, +max_gross/2
                else:
                    w_i_new, w_j_new = +max_gross/2, -max_gross/2
            elif az < exit_z:
                w_i_new = w_j_new = 0.0
            else:
                w_i_new, w_j_new = pos_i, pos_j

            cost = apply_txn_costs(np.array([pos_i, pos_j]),
                                   np.array([w_i_new, w_j_new]),
                                   fee_bps, slippage_bps)
            nav[t-1] *= (1 - cost)
            pos_i, pos_j = w_i_new, w_j_new

        # evolve NAV
        r_i = close[t, i] / close[t-1, i] - 1.0
        r_j = close[t, j] / close[t-1, j] - 1.0
        nav[t] = nav[t-1] * (1.0 + pos_i * r_i + pos_j * r_j)
    return nav

# ---------- Convenience ----------

def run_all_baselines(
        split_parquet_path: str, 
        fee_bps=7.5, slippage_bps=0.0,
        verbose: bool=True, progress_every: int=10000
    ) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """
    Load Close tensor from split parquet, run a small family of baselines, and return:
        symbols, {name -> NAV curve}, {name -> metrics dict}
    """
    if verbose:
        print(f"[run_all_baselines] split={split_parquet_path}")
        print(f"[run_all_baselines] fees: fee_bps={fee_bps}, slippage_bps={slippage_bps if 'slippage_bps' in locals() else 0.0}")

    print(">> loading data ...")
    x, ts, syms, feats, masks = load_tensor_safe(split_parquet_path, features=("Close",))
    close = x[:, :, 0].astype(np.float64)
    print(">> data loaded: T,N,F =", x.shape)

    out: Dict[str, np.ndarray] = {}

    if verbose: print(">> running EW_daily ...")
    out["EW_daily"] = ew_baseline(
        close, 
        fee_bps=fee_bps, 
        rebalance_every=1440, 
        slippage_bps=slippage_bps,
        verbose=verbose, 
        progress_every=progress_every
    )

    if verbose: print(">> running MOM_top2_look360_rb60 ...")
    out["MOM_top2_look360_rb60"] = momentum_topk(
        close, 
        lookback=360, 
        topk=2, 
        fee_bps=fee_bps, 
        rebalance_every=60, 
        slippage_bps=slippage_bps,
        verbose=verbose, 
        progress_every=progress_every
    )

    # Guard for symbol names (BTCUSDT & ETHUSDT may not exist in exotic bundles)
    pair_name = "PAIR_MR_BTC_ETH"
    if "BTCUSDT" in syms and "ETHUSDT" in syms:
        pair = (syms.index("BTCUSDT"), syms.index("ETHUSDT"))
    else:
        # fallback to first two symbols
        pair = (0, 1)
        pair_name = f"PAIR_MR_{syms[pair[0]]}_{syms[pair[1]]}"
        if verbose:
            print(f">> [warn] default pair BTC/ETH not found. Using ({syms[pair[0]]}, {syms[pair[1]]})")

    if verbose: print(f">> running {pair_name} ...")
    out[pair_name] = pair_meanrev_spread(
        close, pair=pair, z_window=180, entry_z=1.5, exit_z=0.5,
        fee_bps=fee_bps, slippage_bps=slippage_bps, max_gross=1.0,
        verbose=verbose, progress_every=progress_every
    )

    metrics = {k: perf_metrics(nav, dt_index=ts) for k, nav in out.items()}
    return syms, out, metrics
