import math, hashlib
import pandas as pd
from pathlib import Path

# fin_rl/data/env/fragment/utils.py

from ...io import candle_io
import pandas as pd
from typing import List, Dict

import pandas as pd
from typing import Dict, List
from ...io import candle_io


def load_per_symbol(data_dir: str, candle_level: str, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Load từng symbol, ép UTC, sort theo thời gian."""
    store = {}
    for sym in symbols:
        df, _fp = candle_io.load_symbol_csv(sym, data_dir, candle_level)
        df["symbol"] = sym
        if df["open_time"].dt.tz is None:
            df["open_time"] = df["open_time"].dt.tz_localize("UTC")
        df = df.sort_values("open_time").reset_index(drop=True)
        store[sym] = df
    return store


def calc_intersection_range(per_symbol: Dict[str, pd.DataFrame], ts_col: str = "open_time"):
    """Tính khoảng giao nhau (start, end) giữa tất cả symbols."""
    starts, ends = [], []
    for df in per_symbol.values():
        if df.empty:
            continue
        starts.append(df[ts_col].min())
        ends.append(df[ts_col].max())
    return max(starts), min(ends)


def slice_time(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, ts_col: str = "open_time"):
    """Cắt DataFrame theo khoảng [start, end]."""
    return df[(df[ts_col] >= start) & (df[ts_col] <= end)].copy()


def inner_align(df: pd.DataFrame, symbols: List[str], ts_col: str = "open_time"):
    """Giữ lại chỉ những timestamp có đủ tất cả symbols."""
    counts = df.groupby(ts_col)["symbol"].nunique()
    valid_ts = counts[counts == len(symbols)].index
    return df[df[ts_col].isin(valid_ts)].copy()


def load_aligned_dataset(data_dir: str, candle_level: str, symbols: List[str], ts_col: str = "open_time") -> pd.DataFrame:
    """
    Load toàn bộ symbols → cắt intersection time → align timestamps.
    Trả về DataFrame đã align đủ symbols.
    """
    # 1) Load per symbol
    per = load_per_symbol(data_dir, candle_level, symbols)

    # 2) Intersection range
    inter_start, inter_end = calc_intersection_range(per, ts_col)

    # 3) Concatenate & slice
    full = pd.concat(list(per.values()), ignore_index=True)
    full = slice_time(full, inter_start, inter_end, ts_col)

    # 4) Inner align
    before, after = len(full), None
    full = inner_align(full, symbols, ts_col)
    after = len(full)
    print(f"[load_aligned_dataset] {before} -> {after} rows after align across {len(symbols)} symbols")

    return full






def load_per_symbol(data_dir: str, candle_level: str, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Load từng symbol với candle_io, ép UTC, sort theo thời gian."""
    store = {}
    for sym in symbols:
        df, _fp = candle_io.load_symbol_csv(sym, data_dir, candle_level)
        df["symbol"] = sym
        if df["open_time"].dt.tz is None:
            df["open_time"] = df["open_time"].dt.tz_localize("UTC")
        df = df.sort_values("open_time").reset_index(drop=True)
        store[sym] = df
    return store


def parse_duration(s: str) -> pd.Timedelta:
    if not s:
        return pd.Timedelta(0)
    s = s.strip().lower()
    val = float(s[:-1]); unit = s[-1]
    if unit == "m": return pd.Timedelta(minutes=val)
    if unit == "h": return pd.Timedelta(hours=val)
    if unit == "d": return pd.Timedelta(days=val)
    if unit == "s": return pd.Timedelta(seconds=val)
    raise ValueError(f"Unsupported duration: {s}")

def bar_duration_from_level(level: str) -> pd.Timedelta:
    x = level.strip().lower().replace("min","m")
    val = float(x[:-1]); unit = x[-1]
    if unit == "m": return pd.Timedelta(minutes=val)
    if unit == "h": return pd.Timedelta(hours=val)
    if unit == "d": return pd.Timedelta(days=val)
    if unit == "s": return pd.Timedelta(seconds=val)
    raise ValueError(f"Unsupported candle-level: {level}")

def embargo_bars_count(embargo_str: str, candle_level: str) -> int:
    E = parse_duration(embargo_str)
    if E <= pd.Timedelta(0): return 0
    bar_dt = bar_duration_from_level(candle_level)
    return int(math.ceil(E / bar_dt))

def ensure_utc(df: pd.DataFrame, ts_col="open_time") -> pd.DataFrame:
    if df[ts_col].dt.tz is None:
        df[ts_col] = df[ts_col].dt.tz_localize("UTC")
    return df

def md5_file(path: Path) -> str:
    md5 = hashlib.md5()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            md5.update(chunk)
    return md5.hexdigest()



# ===============================================
# DEBUG_UTILS 

# utils_debug.py

import pandas as pd
from typing import List, Optional

def print_all_columns(df: pd.DataFrame, max_colwidth: int = 50):
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", max_colwidth,
    ):
        print(df.head(1).iloc[0:0])  # in header gọn, không in dữ liệu
        print("n_cols =", len(df.columns))
        print(list(df.columns))

def preview_fragment(
    dataset,
    frag_idx: int,
    n: int = 5,
    fields: Optional[List[str]] = None,    # ví dụ ["Close","QuoteVolume"]
    symbols: Optional[List[str]] = None,   # ví dụ ["BTCUSDT","ETHUSDT"]
    tail: bool = False,
):
    frag = dataset.fragments[frag_idx]
    cols = ["open_time"]
    if fields is None and symbols is None:
        # mặc định: toàn bộ cột
        cols = [c for c in frag.df.columns]
    else:
        # build danh sách cột theo {field}_{symbol}
        if fields is None:
            fields = []
        if symbols is None:
            # suy ra từ dataset
            symbols = dataset.symbols
        for f in fields:
            for s in symbols:
                col = f"{f}_{s}"
                if col in frag.df.columns:
                    cols.append(col)
        # đảm bảo không trùng & giữ open_time ở đầu
        cols = list(dict.fromkeys(["open_time"] + [c for c in cols if c != "open_time"]))

    print(f"== FRAG {frag_idx} {frag.start} -> {frag.end} rows={len(frag.df)} ==")
    print("META:", frag.meta)
    if tail:
        print(frag.df[cols].tail(n))
    else:
        print(frag.df[cols].head(n))


def schema_summary(dataset):
    cols = set(dataset.fragments[0].df.columns) if dataset.fragments else set()
    base_fields = ["Open","High","Low","Close","Volume","QuoteVolume","Trades","TakerBuyBase","TakerBuyQuote"]
    print("== SCHEMA SUMMARY ==")
    for s in dataset.symbols:
        have = [f for f in base_fields if f"{f}_{s}" in cols]
        miss = [f for f in base_fields if f"{f}_{s}" not in cols]
        print(f" {s}: have={have} | miss={miss}")
    only_close = all(
        (f"{'Close'}_{s}" in cols) and not any(f"{f}_{s}" in cols for f in ["Open","High","Low","Volume","QuoteVolume","Trades","TakerBuyBase","TakerBuyQuote"])
        for s in dataset.symbols
    )
    if only_close:
        print("[WARN] Có vẻ dataset hiện chỉ có cột Close_*; cân nhắc build lại wide với đủ fields.")



import numpy as np

def stats_overview(dataset):
    print("== DATASET OVERVIEW ==")
    print(f"name={dataset.name} | n_frags={len(dataset.fragments)} | symbols={dataset.symbols} | candle_level={dataset.candle_level}")

    # kiểm tra time monotonic + trùng timestamp
    total_rows = 0
    dup_count = 0
    gap_count = 0
    for i, frag in enumerate(dataset.fragments):
        ts = pd.to_datetime(frag.df["open_time"], utc=True)
        total_rows += len(ts)
        dup_count += int(ts.duplicated().sum())
        gap_count += int((ts.diff().dropna() <= pd.Timedelta(0)).sum())
    print(f"rows_total={total_rows:,} | dup_ts={dup_count} | non_increasing_ts={gap_count}")

    # thống kê per-symbol: return & tổng QuoteVolume (nếu có)
    base_fields = ["Close","QuoteVolume"]
    for s in dataset.symbols:
        col_close = f"Close_{s}"
        col_qv    = f"QuoteVolume_{s}"
        if all(col in dataset.fragments[0].df.columns for col in [col_close]):
            close_firsts, close_lasts = [], []
            qv_sums = []
            for frag in dataset.fragments:
                c = frag.df[col_close].dropna()
                if len(c) >= 2:
                    close_firsts.append(c.iloc[0])
                    close_lasts.append(c.iloc[-1])
                if col_qv in frag.df.columns:
                    qv_sums.append(float(frag.df[col_qv].sum()))
            if close_firsts and close_lasts:
                r = (np.sum(close_lasts) / np.sum(close_firsts)) - 1
            else:
                r = np.nan
            qv_total = np.sum(qv_sums) if qv_sums else np.nan
            print(f" {s}: approx_return_over_all_frags={r:.2%} | total_QuoteVolume={qv_total:.3e}")
        else:
            print(f" {s}: missing {col_close}")

def nan_report(dataset, topk: int = 10):
    print("== NAN REPORT (concat all frags) ==")
    full = dataset.concat()
    na = full.isna().mean().sort_values(ascending=False)
    print(na.head(topk))



from collections import Counter

def regime_summary(dataset):
    regs = [frag.meta.get("regime", None) for frag in dataset.fragments]
    cnt = Counter([r if r is not None else "NA" for r in regs])
    print("== REGIME SUMMARY ==", dict(cnt))
    for i, frag in enumerate(dataset.fragments[:5]):
        print(f"[HEAD {i}] {frag.start.date()}→{frag.end.date()} | regime={frag.meta.get('regime')} | return_weighted={frag.meta.get('return_weighted')}")



def to_long(df_wide: pd.DataFrame, symbols: List[str], fields: List[str]):
    # open_time, symbol, field, value
    long_frames = []
    for f in fields:
        cols = [f"{f}_{s}" for s in symbols if f"{f}_{s}" in df_wide.columns]
        if not cols: 
            continue
        tmp = df_wide[["open_time"] + cols].melt(id_vars="open_time", var_name="feat_sym", value_name="value")
        tmp["field"] = f
        tmp["symbol"] = tmp["feat_sym"].str.split("_", n=1).str[1]
        long_frames.append(tmp[["open_time","symbol","field","value"]])
    return pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame()

# Nếu muốn panel 3D: (time x symbol x field) bằng MultiIndex columns — thuận tiện cho slicing
def to_panel(df_wide: pd.DataFrame, symbols: List[str], fields: List[str]):
    cols = {}
    for f in fields:
        for s in symbols:
            col = f"{f}_{s}"
            if col in df_wide.columns:
                cols[col] = (f, s)
    panel = df_wide.rename(columns=cols)
    # cột MultiIndex: (field, symbol)
    panel = panel.set_index("open_time")
    panel.columns = pd.MultiIndex.from_tuples(panel.columns, names=["field","symbol"])
    return panel.sort_index()



# =============================================== 