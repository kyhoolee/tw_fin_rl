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


# fin_rl/data/rl_env/fragment/utils.py

def load_aligned_dataset(
        data_dir: str, candle_level: str, symbols: List[str],
        fields = ["Open","High","Low","Close","Volume","QuoteVolume","Trades","TakerBuyBase","TakerBuyQuote"]
    ) -> pd.DataFrame:
    """
    Trả về wide frame: mỗi field x symbol là 1 cột, ví dụ: Open_BTCUSDT, High_BTCUSDT, ...
    """
    per = load_per_symbol(data_dir, candle_level, symbols)  # đã UTC + sort

    # cắt theo khoảng giao nhau + inner align để mọi timestamp đều đủ symbols
    inter_start, inter_end = calc_intersection_range(per, ts_col="open_time")
    long = pd.concat(list(per.values()), ignore_index=True)
    long = slice_time(long, inter_start, inter_end, "open_time")
    long = inner_align(long, symbols, "open_time")

    # pivot: (time, symbol) -> fields
    wide = (
        long.pivot(index="open_time", columns="symbol", values=fields)
            .sort_index()
    )
    # wide.columns là MultiIndex (field, symbol) -> flatten: f"{field}_{symbol}"
    wide.columns = [f"{f}_{s}" for f, s in wide.columns.to_flat_index()]
    wide = wide.reset_index()

    # đảm bảo có đủ cột cho mọi (field,symbol) kể cả nếu NA (điền NaN giữ nguyên)
    return wide



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
