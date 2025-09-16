# === baselines_minute_core.py ===
import numpy as np
import pandas as pd
from dataclasses import dataclass
from prettytable import PrettyTable

# ---------- CONFIG ----------
FEE_BPS = 7.5          # Binance taker ~0.075%
SLIPPAGE_BPS = 0.0     # có thể set >0 nếu muốn
PERIODS_PER_YEAR = 365 * 24 * 60  # annualize theo phút (crypto 24/7)
EPS = 1e-12

# ---------- INPUT CONTRACT ----------
# Yêu cầu sẵn có các biến:
#   x: np.ndarray shape [T, N, F] dtype float32/float64
#   ts: np.ndarray (datetime64[ns] hoặc pandas DatetimeIndex) len T
#   syms: list[str] len N
#   feats: list[str] len F (phải chứa "Close")
#   masks: np.ndarray bool shape [T, N, F] (True = valid)
#
# Ví dụ: đã có từ load_tensor_safe như log của bạn.

def _ensure_inputs(x, ts, syms, feats, masks):
    # Shape / dtype guard
    assert isinstance(x, np.ndarray) and x.ndim == 3, "x must be [T,N,F]"
    T, N, F = x.shape
    assert len(syms) == N, "len(syms) != N"
    assert len(feats) == F, "len(feats) != F"
    assert isinstance(masks, np.ndarray) and masks.shape == x.shape and masks.dtype == bool, "masks must be bool [T,N,F]"
    # ts → pandas.DatetimeIndex (timezone-aware cũng ok)
    if isinstance(ts, np.ndarray) and np.issubdtype(ts.dtype, np.datetime64):
        ts = pd.to_datetime(ts)
    elif not isinstance(ts, (pd.DatetimeIndex, pd.Series)):
        ts = pd.to_datetime(np.asarray(ts))
    ts = pd.DatetimeIndex(ts)
    # Close index
    try:
        ci = feats.index("Close")
    except ValueError:
        raise ValueError("feats must contain 'Close'")
    return T, N, F, ci, ts

def _nan_masked(a, valid_mask):
    """Trả về mảng với NaN tại các vị trí invalid."""
    out = a.astype(float).copy()
    out[~valid_mask] = np.nan
    return out

def _returns_from_close(close, valid_mask):
    """
    close: [T,N] (NaN tại invalid)
    returns: [T,N] với r[0]=0.0 (hoặc NaN → thay 0), 
    r[t]=close[t]/close[t-1]-1 nếu cả 2 valid, else 0.
    """
    T, N = close.shape
    r = np.zeros((T, N), dtype=float)
    # điều kiện hợp lệ cho step t: close[t] và close[t-1] không NaN
    valid_step = (~np.isnan(close)) & (~np.isnan(np.vstack([close[0:1], close[:-1]])))
    # tính tỉ số, tránh chia 0
    prev = np.vstack([close[0:1], close[:-1]])
    ratio = np.divide(close, prev, out=np.ones_like(close), where=(prev > 0))
    r_calc = ratio - 1.0
    r[valid_step] = r_calc[valid_step]
    # ở nơi invalid → r = 0 (neutral)
    return r

def _daily_rebal_mask(ts):
    """
    True tại những index là 'đầu ngày' UTC (thanh 00:00). 
    Nếu phút dữ liệu không chuẩn bắt đầu 00:00, dùng điều kiện đổi ngày.
    """
    date = ts.tz_convert("UTC").date if ts.tz is not None else ts.date
    # True tại t=0 hoặc khi ngày thay đổi so với t-1
    flag = np.zeros(len(ts), dtype=bool)
    flag[0] = True
    flag[1:] = (pd.Series(date[1:]).values != pd.Series(date[:-1]).values)
    return flag

def _topk_weights(scores, valid, k):
    """
    scores: [N] điểm số; valid: [N] bool; chọn top-k (>=1 valid).
    Trả về w: [N] nonnegative, sum=1 (nếu có >=1 valid), else zero.
    """
    w = np.zeros_like(scores, dtype=float)
    vv = np.where(valid)[0]
    if len(vv) == 0:
        return w
    k = min(k, len(vv))
    # argsort giảm dần theo scores[vv]
    sub = vv[np.argsort(scores[vv])[::-1][:k]]
    w[sub] = 1.0 / k
    return w

@dataclass
class BacktestResult:
    equity: np.ndarray        # [T] đường vốn
    port_ret: np.ndarray      # [T] lợi nhuận mỗi phút (sau phí)
    turnover: np.ndarray      # [T] sum |Δw|
    fees_paid: np.ndarray     # [T] phí thực trả (tỷ lệ)
    weights: np.ndarray       # [T,N] trọng số sau khi trade xong
    summary: dict             # metrics tổng hợp

def _compute_metrics(port_ret, equity):
    T = len(port_ret)
    
    # Annualized
    mu = np.nanmean(port_ret)
    sigma = np.nanstd(port_ret)
    ann_return = (1.0 + np.nanmean(port_ret)) ** PERIODS_PER_YEAR - 1.0  # gần đúng
    ann_vol = sigma * np.sqrt(PERIODS_PER_YEAR)
    sharpe = (mu / (sigma + EPS)) * np.sqrt(PERIODS_PER_YEAR)
    
    # MaxDD
    peak = np.maximum.accumulate(equity)
    dd = (equity / (peak + EPS)) - 1.0
    maxdd = np.min(dd)
    
    # Calmar (dùng |maxdd|)
    calmar = ann_return / (abs(maxdd) + EPS)
    total_return = equity[-1] - 1.0
    return {
        "total_return": float(total_return),
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(maxdd),
        "calmar": float(calmar),
    }

def _run_engine(
        returns, 
        valid_asset, 
        target_weight_fn, 
        rebal_mask, 
        fee_rate, 
        verbose=False
    ):
    """
    returns: [T,N] minute returns
    valid_asset: [T,N] bool (asset usable tại thời điểm đó)
    target_weight_fn(t) -> [N] (mục tiêu tại time t khi rebal_mask[t] True). 
                           Khi rebal_mask[t] False, trả None để hold.
    rebal_mask: [T] bool (khi nào rebalance)
    fee_rate: float (tỷ lệ phí theo turnover: sum|Δw|)
    """
    T, N = returns.shape
    equity = np.ones(T, dtype=float)
    port_ret = np.zeros(T, dtype=float)
    turnover = np.zeros(T, dtype=float)
    fees_paid = np.zeros(T, dtype=float)
    w = np.zeros((T, N), dtype=float)

    # t=0: khởi tạo mục tiêu nếu có
    tgt0 = target_weight_fn(0) if rebal_mask[0] else None
    if tgt0 is not None:
        # chuẩn hóa với valid
        valid0 = valid_asset[0]
        if tgt0 is None:
            cur = np.zeros(N)
        else:
            # ép về valid
            tw = np.where(valid0, np.maximum(tgt0, 0.0), 0.0)
            s = tw.sum()
            cur = tw / s if s > 0 else np.zeros(N)
        # phí mở vị thế
        tv = np.sum(np.abs(cur))  # từ 0 → cur, turnover = sum|cur|
        fee = fee_rate * tv
        fees_paid[0] = fee
        w[0] = cur
        port_ret[0] = (cur * returns[0]).sum() - fee
        equity[0] = 1.0 * (1.0 + port_ret[0])
        turnover[0] = tv
        if verbose:
            print(f"[t=0] init turnover={tv:.6f}, fee={fee:.8f}, port_ret={port_ret[0]:.8f}")
    else:
        # không vào lệnh
        w[0] = 0.0
        port_ret[0] = 0.0
        equity[0] = 1.0

    for t in range(1, T):
        # drift trọng số theo return t-1
        prev_w = w[t-1]
        prev_ret = returns[t-1]
        gross = prev_w * (1.0 + prev_ret)
        s = gross.sum()
        pre_w = gross / (s + EPS) if s > EPS else np.zeros(N)

        # quyết định rebalance?
        if rebal_mask[t]:
            tgt = target_weight_fn(t)
        else:
            tgt = None

        if tgt is not None:
            # ép về valid tại t
            valid_t = valid_asset[t]
            tw = np.where(valid_t, np.maximum(tgt, 0.0), 0.0)
            st = tw.sum()
            tgt = tw / st if st > 0 else np.zeros(N)
            dv = tgt - pre_w
            tv = np.sum(np.abs(dv))
            fee = fee_rate * tv
            cur_w = tgt
        else:
            tv = 0.0
            fee = 0.0
            cur_w = pre_w

        w[t] = cur_w
        # lợi nhuận bước t dựa trên returns[t] (sau khi đã trade xong)
        step_ret = (cur_w * returns[t]).sum() - fee
        port_ret[t] = step_ret
        equity[t] = equity[t-1] * (1.0 + step_ret)
        turnover[t] = tv
        fees_paid[t] = fee

        if verbose and rebal_mask[t]:
            print(f"[t={t}] rebal turnover={tv:.6f}, fee={fee:.8f}, step_ret={step_ret:.8f}")

    metrics = _compute_metrics(port_ret, equity)
    return BacktestResult(equity=equity, port_ret=port_ret, turnover=turnover,
                          fees_paid=fees_paid, weights=w, summary=metrics)



# ========================================
# ---------- STRATEGIES ----------
# ========================================

def strat_buyhold_eq(close, valid):
    """
    Buy & Hold Equal Weight: 
        - vào lệnh tại t=0 (trên các asset valid), 
        - không rebalance về sau.
    """
    T, N = close.shape
    rebal_mask = np.zeros(T, dtype=bool)
    rebal_mask[0] = True

    def tgt_fn(t):
        # equal weight trên các asset valid lúc t
        v = valid[t]
        if v.sum() == 0: return np.zeros(N)
        w = np.zeros(N)
        w[v] = 1.0 / v.sum()
        return w

    return rebal_mask, tgt_fn


def strat_equal_weight_daily(close, valid):
    """
    Equal Weight: 
        - rebalance đầu mỗi ngày (UTC) trên tập asset valid tại thời điểm đó.
    """
    T, N = close.shape
    # rebal theo ngày
    # ts global (đặt xong sau) truyền vào scope bên ngoài → dùng closure tốt hơn; ở đây
    # mình assume sẽ truyền rebal_mask từ ngoài vào để tránh lệ thuộc ts ở đây.
    # Tuy nhiên để tiện, ta sẽ trả None và để caller set rebal_mask_daily.
    def tgt_fn(t):
        v = valid[t]
        if v.sum() == 0: return np.zeros(N)
        w = np.zeros(N); w[v] = 1.0 / v.sum()
        return w
    return tgt_fn  # rebal_mask sẽ do caller quyết định (daily)


def strat_xsect_momentum(
        close, valid, 
        lookback=60, topk=2, use_logret=False
    ):
    """
    Cross-sectional momentum: 
        - mỗi phiên rebalance, tính performance L phút qua,
        - chọn top-K để phân bổ đều. 
        - Nếu dữ liệu thiếu → loại khỏi ranking.
    """
    T, N = close.shape

    # precompute score[t,n] = return(Lookback) kết thúc tại t-1 (để tránh lookahead)
    if use_logret:
        logc = np.log(np.maximum(close, EPS))
        # sum log-ret trên cửa sổ → log(C_{t-1})-log(C_{t-1-L})
        score = np.full((T, N), np.nan)
        idx0 = np.arange(T)
        idx1 = idx0 - lookback
        valid_row = idx1 >= 0
        # chỉ tính từ t>=lookback
        for t in np.where(valid_row)[0]:
            a = logc[t-1]     # logC_{t-1}
            b = logc[t-1-lookback]
            s = a - b
            score[t] = s
    else:
        # simple return: C_{t-1}/C_{t-1-L} - 1
        score = np.full((T, N), np.nan)
        for t in range(lookback, T):
            prev = close[t-1]
            base = close[t-1-lookback]
            ok = (~np.isnan(prev)) & (~np.isnan(base)) & (base > 0)
            s = np.full(N, np.nan)
            s[ok] = (prev[ok] / base[ok]) - 1.0
            score[t] = s

    def tgt_fn(t):
        # chỉ dùng score tại t (đã tính bằng dữ liệu đến t-1)
        sc = score[t]
        v = valid[t] & (~np.isnan(sc))
        return _topk_weights(sc, v, topk)

    return tgt_fn


# ========================================
# ---------- HIGH-LEVEL RUNNERS ----------
# ========================================

def run_all_baselines(
        x, ts, syms, feats, masks, 
        fee_bps=FEE_BPS, slippage_bps=SLIPPAGE_BPS, 
        verbose=False
    ):
    T, N, F, ci, ts = _ensure_inputs(x, ts, syms, feats, masks)
    fee_rate = (fee_bps + slippage_bps) / 1e4

    # Close + valid
    close_raw = x[:, :, ci]
    valid_close = masks[:, :, ci]
    close = _nan_masked(close_raw, valid_close)

    # Returns [T,N]
    rets = _returns_from_close(close, valid_close)
    assert rets.shape == (T, N)

    # Valid asset per time (chỉ yêu cầu giá close tại t hợp lệ)
    valid_asset = ~np.isnan(close)

    # Rebal mask theo ngày
    rebal_daily = _daily_rebal_mask(ts)

    results = {}

    # 1. Buy & Hold equal weight
    rb_mask, tgt_fn = strat_buyhold_eq(close, valid_asset)
    res = _run_engine(rets, valid_asset, tgt_fn, rb_mask, fee_rate, verbose=verbose)
    results["buyhold_eq"] = res

    # 2. Equal Weight, rebalance daily
    tgt_fn2 = strat_equal_weight_daily(close, valid_asset)
    res2 = _run_engine(rets, valid_asset, tgt_fn2, rebal_daily, fee_rate, verbose=verbose)
    results["ew_daily"] = res2

    # 3. X-Section Momentum (L=60 phút, topK=2), rebalance daily
    tgt_fn3 = strat_xsect_momentum(close, valid_asset, lookback=60, topk=min(2, N))
    res3 = _run_engine(rets, valid_asset, tgt_fn3, rebal_daily, fee_rate, verbose=verbose)
    results["xmom_L60_top2_daily"] = res3

    return {
        "ts": ts,
        "syms": syms,
        "results": {
            k: v.summary | {
                "avg_turnover_per_rebal": float(np.nanmean(v.turnover[rebal_daily])) if k != "buyhold_eq" else float(np.nanmean(v.turnover)),
                "total_fees_paid": float(np.nansum(v.fees_paid)),
            }
            for k, v in results.items()
        },
        "raw": results  # có equity/weights/turnover để trace chi tiết
    }


# ========================================
# ---------- NICE TRACE ----------
# ========================================

def pretty_print_summary(report):
    table = PrettyTable()
    table.field_names = [
        "Strategy", "TotalRet", "AnnRet", "AnnVol", "Sharpe", 
        "MaxDD", "Calmar", "AvgTurnover", "TotalFees"
    ]
    table.align = "r"
    table.float_format = ".4f"

    for name, s in report["results"].items():
        table.add_row([
            name,
            s['total_return'],
            s['ann_return'],
            s['ann_vol'],
            s['sharpe'],
            s['max_drawdown'],
            s['calmar'],
            s['avg_turnover_per_rebal'],
            s['total_fees_paid']
        ])
    print("== SUMMARY (after fees) ==")
    print(table)

def trace_first_rebals(ts, res: BacktestResult, n_events=5):
    idx = np.where(np.abs(res.turnover) > 1e-9)[0]
    print(f"\n== TRACE first {n_events} rebalances ==")
    for i in idx[:n_events]:
        print(f"{ts[i]}  turnover={res.turnover[i]:.4f}  fee={res.fees_paid[i]:.6f}  step_ret={res.port_ret[i]:+.6f}")
