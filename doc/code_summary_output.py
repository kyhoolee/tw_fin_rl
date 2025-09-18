

# ===== File: fin_rl/data/env/fragment/__init__.py =====




# ===== File: fin_rl/data/env/fragment/base_splitter.py =====

from typing import Dict
from .trading_dataset import TradingDataset

class BaseSplitter:
    """Interface cho tất cả splitter."""

    def split(self, dataset: TradingDataset) -> Dict[str, TradingDataset]:
        """
        Input: TradingDataset
        Output: {"train": TradingDataset, "test": ..., "unseen": ...}
        """
        raise NotImplementedError



# ===== File: fin_rl/data/env/fragment/cli.py =====

import argparse, yaml, json
from .registry import get_splitter
from .trading_dataset import TradingDatasetFactory

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config cho splitter")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--candle-level", required=True)
    ap.add_argument("--symbols", nargs="+", required=True)
    args = ap.parse_args()

    # Load dataset gốc
    dataset = TradingDatasetFactory.from_raw(
        data_dir=args.data_dir,
        candle_level=args.candle_level,
        symbols=args.symbols
    )

    # Đọc config YAML
    cfg = yaml.safe_load(open(args.config))
    splitter = get_splitter(cfg["mode"])(**cfg["params"])
    result = splitter.split(dataset)

    # Xuất report nhỏ
    print(json.dumps({k: v.time_ranges() for k,v in result.items()}, indent=2, default=str))



# ===== File: fin_rl/data/env/fragment/registry.py =====

from .splitter_continuous import ContinuousSplitter
from .splitter_fragmented import FragmentedSplitter
from .splitter_regime import RegimeSplitter

SPLITTERS = {
    "continuous": ContinuousSplitter,
    "fragmented": FragmentedSplitter,
    "regime": RegimeSplitter,
}

def get_splitter(name: str):
    if name not in SPLITTERS:
        raise ValueError(f"Unknown splitter {name}")
    return SPLITTERS[name]



# ===== File: fin_rl/data/env/fragment/splitter_blockshuffle.py =====




# ===== File: fin_rl/data/env/fragment/splitter_continuous.py =====

import pandas as pd
from .base_splitter import BaseSplitter
from .trading_dataset import TradingDataset, TradingFragment

class ContinuousSplitter(BaseSplitter):
    def __init__(self, ranges: dict):
        """
        ranges = {
            "train": (start,end),
            "test": (start,end),
            "unseen": (start,end)
        }
        """
        self.ranges = ranges
    
    def split(self, dataset: TradingDataset):
        df = dataset.concat()
        out = {}
        for name, (s,e) in self.ranges.items():
            part = df[(df["open_time"]>=pd.Timestamp(s,tz="UTC")) &
                      (df["open_time"]<=pd.Timestamp(e,tz="UTC"))].copy()
            frag = TradingFragment(df=part, start=part["open_time"].iloc[0], end=part["open_time"].iloc[-1])
            out[name] = TradingDataset(name=f"{dataset.name}_{name}",
                                       fragments=[frag],
                                       symbols=dataset.symbols,
                                       candle_level=dataset.candle_level)
        return out



# ===== File: fin_rl/data/env/fragment/splitter_fragmented.py =====

from .base_splitter import BaseSplitter
from .trading_dataset import TradingDataset

class FragmentedSplitter(BaseSplitter):
    def __init__(self, assign: dict):
        """
        assign = {"train":[0,1], "test":[2], "unseen":[3]}
        index dựa theo dataset.fragments
        """
        self.assign = assign
    
    def split(self, dataset: TradingDataset):
        out = {}
        for name, idxs in self.assign.items():
            frags = [dataset.fragments[i] for i in idxs]
            out[name] = TradingDataset(name=f"{dataset.name}_{name}",
                                       fragments=frags,
                                       symbols=dataset.symbols,
                                       candle_level=dataset.candle_level)
        return out


class FragmentGenerator:
    @staticmethod
    def slice_fragments(dataset: TradingDataset, ranges: List[tuple]) -> List[TradingFragment]:
        """Cắt 1 dataset continuous thành list fragment theo ranges (start,end)."""
        df = dataset.concat()
        frags = []
        for (s,e) in ranges:
            part = df[(df["open_time"]>=pd.Timestamp(s,tz="UTC")) &
                      (df["open_time"]<=pd.Timestamp(e,tz="UTC"))].copy()
            if not part.empty:
                frags.append(TradingFragment(
                    df=part,
                    start=part["open_time"].iloc[0],
                    end=part["open_time"].iloc[-1]
                ))
        return frags



# ===== File: fin_rl/data/env/fragment/splitter_regime.py =====

import numpy as np
from .base_splitter import BaseSplitter
from .trading_dataset import TradingDataset

class RegimeSplitter(BaseSplitter):
    def __init__(self, regime_map: dict):
        """
        regime_map = {
            "train": ["bull","sideways"],
            "test":  ["bear"],
            "unseen":["mixed"]
        }
        """
        self.regime_map = regime_map
    
    def split(self, dataset: TradingDataset):
        out = {}
        for name, regimes in self.regime_map.items():
            out[name] = dataset.filter_by_regime(regimes)
        return out

def tag_regimes(df, price_col="Close", window=50):
    """Tag regime đơn giản: bull/bear/sideways dựa trên MA trend."""
    ma = df[price_col].rolling(window).mean()
    trend = df[price_col] - ma
    conds = [(trend > 0.01*df[price_col]),
             (trend < -0.01*df[price_col])]
    choices = ["bull","bear"]
    df["regime"] = np.select(conds, choices, default="sideways")
    return df



# ===== File: fin_rl/data/env/fragment/trading_dataset.py =====

from dataclasses import dataclass, field
from typing import List, Dict, Any
import pandas as pd

@dataclass
class TradingFragment:
    """Một đoạn dữ liệu liên tục."""
    df: pd.DataFrame
    start: pd.Timestamp
    end: pd.Timestamp
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TradingDataset:
    """Dataset gồm nhiều fragment (train/test/unseen đều dùng format này)."""
    name: str
    fragments: List[TradingFragment]
    symbols: List[str]
    candle_level: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def concat(self) -> pd.DataFrame:
        """Nối toàn bộ fragment thành 1 DataFrame."""
        return pd.concat([f.df for f in self.fragments], ignore_index=True)

    def time_ranges(self) -> List[tuple]:
        return [(f.start, f.end) for f in self.fragments]

    def filter_by_regime(self, regimes: List[str]) -> "TradingDataset":
        """Tạo dataset mới chỉ chứa rows có regime trong list."""
        new_frags = []
        for frag in self.fragments:
            if "regime" not in frag.df.columns: 
                continue
            mask = frag.df["regime"].isin(regimes)
            df = frag.df[mask].copy()
            if not df.empty:
                new_frags.append(
                    TradingFragment(df=df, start=df["open_time"].iloc[0], end=df["open_time"].iloc[-1])
                )
        return TradingDataset(
            name=f"{self.name}_regime_{'_'.join(regimes)}",
            fragments=new_frags,
            symbols=self.symbols,
            candle_level=self.candle_level,
            meta=self.meta.copy()
        )



# ===== File: fin_rl/data/env/fragment/utils.py =====

import math, hashlib
import pandas as pd
from pathlib import Path

# fin_rl/data/env/fragment/utils.py

from ...io import candle_io
import pandas as pd
from typing import List, Dict

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

