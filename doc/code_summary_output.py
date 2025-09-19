

# ===== File: fin_rl/data/rl_env/fragment/__init__.py =====




# ===== File: fin_rl/data/rl_env/fragment/base_splitter.py =====

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



# ===== File: fin_rl/data/rl_env/fragment/cli.py =====

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



# ===== File: fin_rl/data/rl_env/fragment/dataset_utils.py =====

"""
dataset_utils.py
----------------
Tiện ích cho TradingDataset:
- save/load ra JSONL + Parquet
- print debug summary
"""

import json
from pathlib import Path
import pandas as pd
from .trading_dataset import TradingDataset, TradingFragment

def save_dataset(ds: TradingDataset, out_dir: str):
    """
    Lưu TradingDataset:
    - meta.json: metadata chung
    - fragments/<i>.parquet: dữ liệu từng fragment
    """
    out_path = Path(out_dir)
    (out_path / "fragments").mkdir(parents=True, exist_ok=True)

    meta = {
        "name": ds.name,
        "symbols": ds.symbols,
        "candle_level": ds.candle_level,
        "meta": ds.meta,
        "fragments": []
    }
    for i, frag in enumerate(ds.fragments):
        frag_path = out_path / "fragments" / f"frag_{i}.parquet"
        frag.df.to_parquet(frag_path, index=False)
        meta["fragments"].append({
            "i": i,
            "start": str(frag.start),
            "end": str(frag.end),
            "path": str(frag_path),
            "meta": frag.meta,
        })
    with open(out_path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_dataset(in_dir: str) -> TradingDataset:
    """
    Load TradingDataset từ meta.json + fragments/*.parquet
    """
    in_path = Path(in_dir)
    meta = json.load(open(in_path / "meta.json", encoding="utf-8"))
    frags = []
    for frag_info in meta["fragments"]:
        df = pd.read_parquet(frag_info["path"])
        start = pd.to_datetime(frag_info["start"])
        end = pd.to_datetime(frag_info["end"])
        frags.append(TradingFragment(df=df, start=start, end=end, meta=frag_info.get("meta", {})))
    return TradingDataset(
        name=meta["name"],
        fragments=frags,
        symbols=meta["symbols"],
        candle_level=meta["candle_level"],
        meta=meta.get("meta", {})
    )


def print_dataset_debug(ds: TradingDataset, max_frag: int = 5):
    """
    In ra thông tin dataset để debug
    """
    print(f"== TradingDataset '{ds.name}' ==")
    print(f"Symbols: {ds.symbols}, candle_level: {ds.candle_level}, n_frag={len(ds.fragments)}")
    for i, frag in enumerate(ds.fragments[:max_frag]):
        meta_keys = list(frag.meta.keys())
        print(f"  Frag {i}: {frag.start} -> {frag.end}, rows={len(frag.df)}, meta={frag.meta}")
        print(f"       df cols : {list(frag.df.columns)[:8]}{' ...' if len(frag.df.columns)>8 else ''}")
    if len(ds.fragments) > max_frag:
        print(f"  ... {len(ds.fragments) - max_frag} more fragments ...")



# ===== File: fin_rl/data/rl_env/fragment/pipeline_monthly_fragments.py =====

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline: load -> fragment by month -> datasets -> regime splits
"""

import pandas as pd
from pathlib import Path
from typing import List
import time
import numpy as np

from fin_rl.data.rl_env.fragment.trading_dataset import TradingDataset, TradingFragment
from fin_rl.data.rl_env.fragment.utils import load_aligned_dataset
from fin_rl.data.rl_env.fragment.splitter_fragmented import FragmentedSplitter
from fin_rl.data.rl_env.fragment.splitter_regime import FlexibleRegimeSplitter


def log(msg: str):
    print(f"[LOG] {msg}")


def log_step(step_id: int, name: str):
    def wrap(func):
        def inner(*args, **kwargs):
            log(f"Step {step_id}: {name} START")
            start = time.time()
            out = func(*args, **kwargs)
            dur = time.time() - start
            log(f"Step {step_id}: {name} DONE in {dur:.2f}s")
            return out
        return inner
    return wrap


def make_monthly_fragments(df: pd.DataFrame, symbol: str) -> List[TradingFragment]:
    # print df meta for debug
    print(f"DataFrame for symbol {symbol}:")
    print(f"  Rows: {len(df)}")
    print(f"  Date range: {df['open_time'].min()} to {df['open_time'].max()}")
    print(f"  Columns: {df.columns.tolist()}")
    # print head and tail
    print("  Head:")
    print(df.head(3))
    print("  Tail:")
    print(df.tail(3))

    frags = []
    grouped = df.groupby(df["open_time"].dt.to_period("M"))
    for _, sub in grouped:
        if sub.empty:
            continue
        sub = sub.sort_values("open_time")
        frags.append(TradingFragment(
            df=sub,
            start=sub["open_time"].iloc[0],
            end=sub["open_time"].iloc[-1],
            meta={"symbol": symbol}
        ))
    return frags


@log_step(1, "Build monthly dataset (aligned + fragments)")
def build_datasets(data_dir: str, candle_level: str, symbols: List[str]) -> TradingDataset:
    full = load_aligned_dataset(data_dir, candle_level, symbols)

    all_frags = []
    grouped = full.groupby(full["open_time"].dt.to_period("M"))
    for _, sub in grouped:
        if sub.empty:
            continue
        sub = sub.sort_values("open_time")
        all_frags.append(TradingFragment(
            df=sub,
            start=sub["open_time"].iloc[0],
            end=sub["open_time"].iloc[-1],
            meta={"symbols": symbols}
        ))

    log(f"Generated {len(all_frags)} monthly fragments")
    n_rows = sum(len(f.df) for f in all_frags)
    log(f"Total rows: {n_rows:,}")
    return TradingDataset(
        name="monthly_fragments",
        fragments=all_frags,
        symbols=symbols,
        candle_level=candle_level
    )


@log_step(2, "Fragmented split")
def example_fragmented_split(dataset: TradingDataset):
    assign = {
        "train": list(range(0, 24)),
        "valid": list(range(24, 30)),
        "test":  list(range(30, 36)),
        "unseen": list(range(36, len(dataset.fragments)))
    }
    log(f"Assign rule: { {k: len(v) for k,v in assign.items()} }")
    splitter = FragmentedSplitter(assign)
    return splitter.split(dataset)

@log_step(3, "Flexible regime split with thresholds")
def example_flexible_split(dataset: TradingDataset):
    regime_ratios = {
        "train":  {"bull": 0.4, "bear": 0.4, "sideways": 0.4},
        "valid":  {"bull": 0.4, "bear": 0.4, "sideways": 0.4},
        "test":   {"bull": 0.4, "bear": 0.4, "sideways": 0.4},
        "unseen": {"bull": 0.4, "bear": 0.4, "sideways": 0.4},
    }
    splitter = FlexibleRegimeSplitter(
        regime_ratios,
        mode="fraction",
        bull_thr=0.02, bear_thr=-0.02, side_thr=0.01,
        strategy="deterministic",
        verbose=True
    )
    splits = splitter.split(dataset)

    # In thêm summary (dùng weighted return trong meta)
    for name, ds in splits.items():
        rets = [f.meta.get("return_weighted", 0.0) for f in ds.fragments]
        if rets:
            print(f"[{name}] n_frag={len(ds.fragments)}, "
                  f"mean_return={np.mean(rets):.2%}, "
                  f"min={np.min(rets):.2%}, max={np.max(rets):.2%}")
            # In thêm head/tail meta để check
            for frag in (ds.fragments[:2] + ds.fragments[-2:]):
                print(f"    Frag {frag.start.date()}–{frag.end.date()}, "
                      f"regime={frag.meta.get('regime')}, "
                      f"weighted_return={frag.meta.get('return_weighted'):.2%}, "
                      f"meta_keys={list(frag.meta.keys())}")
    return splits


if __name__ == "__main__":
    data_dir = "./work/data/binance/spot/1m"
    candle_level = "1m"
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]

    dataset = build_datasets(data_dir, candle_level, symbols)
    dataset.print_debug(max_frag=5)

    frag_splits = example_fragmented_split(dataset)
    print("\n== Fragmented splits ==")
    for k, v in frag_splits.items():
        log(f"Processing split: {k}, {len(v.fragments)} fragments")
        v.print_debug(max_frag=3)
        out_dir = Path(f"./work/datasets/{dataset.name}_frag_{k}")
        v.save(out_dir)
        log(f"Saved split '{k}' to {out_dir}")
        TradingDataset.load(out_dir).print_debug(max_frag=2)

    regime_splits = example_flexible_split(dataset)
    print("\n== Regime splits ==")
    for k, v in regime_splits.items():
        log(f"Processing regime: {k}, {len(v.fragments)} fragments")
        v.print_debug(max_frag=3)
        out_dir = Path(f"./work/datasets/{dataset.name}_regime_{k}")
        v.save(out_dir)
        log(f"Saved regime '{k}' to {out_dir}")
        TradingDataset.load(out_dir).print_debug(max_frag=2)



# ===== File: fin_rl/data/rl_env/fragment/registry.py =====

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



# ===== File: fin_rl/data/rl_env/fragment/splitter_blockshuffle.py =====




# ===== File: fin_rl/data/rl_env/fragment/splitter_continuous.py =====

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



# ===== File: fin_rl/data/rl_env/fragment/splitter_fragmented.py =====

from .base_splitter import BaseSplitter
from .trading_dataset import TradingDataset, TradingFragment
import pandas as pd
from typing import List

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



# ===== File: fin_rl/data/rl_env/fragment/splitter_regime.py =====

import numpy as np
from typing import Dict
from .base_splitter import BaseSplitter
from .trading_dataset import TradingDataset


class FlexibleRegimeSplitter(BaseSplitter):
    def __init__(self, regime_ratios: Dict, mode: str = "fraction",
                 bull_thr=0.02, bear_thr=-0.02, side_thr=0.01,
                 strategy="deterministic", seed=None, verbose=True):
        """
        regime_ratios: cấu hình số lượng/tỉ lệ fragment cho từng split
        mode: "fraction" (tỉ lệ) hoặc "count" (số lượng)
        bull_thr: return_weighted > ngưỡng => bull
        bear_thr: return_weighted < ngưỡng => bear
        side_thr: |return_weighted| < ngưỡng => sideway
        strategy: ["deterministic", "random", "time_stratified"]
        """
        self.regime_ratios = regime_ratios
        self.mode = mode
        self.bull_thr = bull_thr
        self.bear_thr = bear_thr
        self.side_thr = side_thr
        self.strategy = strategy
        self.seed = seed
        self.verbose = verbose

    def _label_fragments(self, dataset: TradingDataset):
        bull_ids, bear_ids, side_ids = [], [], []

        for i, frag in enumerate(dataset.fragments):
            df = frag.df

            returns_per_symbol = {}
            weights_per_symbol = {}

            rets, weights = [], []
            for sym in dataset.symbols:
                col_close = f"Close_{sym}" if f"Close_{sym}" in df.columns else "Close"
                col_qv = f"QuoteVolume_{sym}" if f"QuoteVolume_{sym}" in df.columns else "QuoteVolume"

                if col_close not in df or col_qv not in df:
                    continue

                start_p, end_p = df[col_close].iloc[0], df[col_close].iloc[-1]
                r = (end_p / start_p) - 1
                w = df[col_qv].sum()

                returns_per_symbol[sym] = {
                    "first": float(start_p),
                    "last": float(end_p),
                    "return": float(r)
                }
                weights_per_symbol[sym] = float(w)

                if w > 0:
                    rets.append(r * w)
                    weights.append(w)

            weighted_return = sum(rets) / sum(weights) if weights else 0.0

            frag.meta["returns_per_symbol"] = returns_per_symbol
            frag.meta["weights_per_symbol"] = weights_per_symbol
            frag.meta["return_weighted"] = weighted_return

            # regime classification by weighted return
            if weighted_return > self.bull_thr:
                frag.meta["regime"] = "bull"
                bull_ids.append(i)
            elif weighted_return < self.bear_thr:
                frag.meta["regime"] = "bear"
                bear_ids.append(i)
            elif abs(weighted_return) < self.side_thr:
                frag.meta["regime"] = "sideways"
                side_ids.append(i)
            else:
                frag.meta["regime"] = "sideways"
                side_ids.append(i)

            if self.verbose:
                print(f"[Label] Frag {i} ({frag.start.date()}→{frag.end.date()}): "
                      f"weighted_return={weighted_return:.2%}, regime={frag.meta['regime']}")
                print(f"    df cols = {list(df.columns)[:10]}{' ...' if len(df.columns)>10 else ''}")
                print(f"    meta keys = {list(frag.meta.keys())}")
                for sym, info in returns_per_symbol.items():
                    w = weights_per_symbol[sym]
                    print(f"        {sym}: first={info['first']:.4f}, last={info['last']:.4f}, "
                          f"return={info['return']:.2%}, weight={w:.2e}")

        if self.verbose:
            print(f"[Summary] bull={len(bull_ids)}, bear={len(bear_ids)}, side={len(side_ids)}")

        return {"bull": bull_ids, "bear": bear_ids, "sideways": side_ids}

    def _choose(self, pool, k):
        if k <= 0 or not pool:
            return []
        if self.strategy == "deterministic":
            return pool[:k]
        elif self.strategy == "random":
            rng = np.random.default_rng(self.seed)
            return list(rng.choice(pool, size=min(k, len(pool)), replace=False))
        elif self.strategy == "time_stratified":
            step = max(1, len(pool) // k)
            return pool[::step][:k]
        else:
            raise ValueError(f"Unknown strategy={self.strategy}")

    def split(self, dataset: TradingDataset):
        regime_ids = self._label_fragments(dataset)
        out = {}

        pools = {reg: list(ids) for reg, ids in regime_ids.items()}

        for split_name, ratios in self.regime_ratios.items():
            total = sum(ratios.values())
            norm_ratios = {k: v / total for k, v in ratios.items()} if total > 0 else ratios

            idxs = []
            for regime, val in norm_ratios.items():
                pool = pools[regime]
                if not pool:
                    continue
                if self.mode == "fraction":
                    k = int(val * len(pool))
                elif self.mode == "count":
                    k = min(int(val), len(pool))
                else:
                    raise ValueError(f"Unknown mode={self.mode}")

                chosen = self._choose(pool, k)
                idxs.extend(chosen)
                pools[regime] = [i for i in pool if i not in chosen]

            idxs = sorted(set(idxs))
            frags = [dataset.fragments[i] for i in idxs]
            out[split_name] = TradingDataset(
                name=f"{dataset.name}_{split_name}",
                fragments=frags,
                symbols=dataset.symbols,
                candle_level=dataset.candle_level
            )

            if self.verbose:
                print(f"[FlexibleRegimeSplitter] {split_name}: n_frag={len(frags)}")
                for i in idxs:
                    frag = dataset.fragments[i]
                    print(f"    Frag {i}: regime={frag.meta['regime']}, "
                          f"weighted_return={frag.meta['return_weighted']:.2%}")
                    for sym, info in frag.meta["returns_per_symbol"].items():
                        w = frag.meta["weights_per_symbol"][sym]
                        print(f"        {sym}: first={info['first']:.4f}, last={info['last']:.4f}, "
                              f"return={info['return']:.2%}, weight={w:.2e}")

        return out



# ===== File: fin_rl/data/rl_env/fragment/trading_dataset.py =====

from dataclasses import dataclass, field
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path

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
    
    def save(self, out_dir: str):
        """Lưu TradingDataset ra thư mục (meta.json + fragments/*.parquet)."""
        out_path = Path(out_dir)
        (out_path / "fragments").mkdir(parents=True, exist_ok=True)

        meta = {
            "name": self.name,
            "symbols": self.symbols,
            "candle_level": self.candle_level,
            "meta": self.meta,
            "fragments": []
        }
        for i, frag in enumerate(self.fragments):
            frag_path = out_path / "fragments" / f"frag_{i}.parquet"
            frag.df.to_parquet(frag_path, index=False)
            meta["fragments"].append({
                "i": i,
                "start": str(frag.start),
                "end": str(frag.end),
                "path": str(frag_path),
                "meta": frag.meta,
            })
        with open(out_path / "meta.json", "w", encoding="utf-8") as f:
            import json
            json.dump(meta, f, indent=2)

    @staticmethod
    def load(in_dir: str) -> "TradingDataset":
        """Load TradingDataset từ thư mục (meta.json + fragments)."""
        in_path = Path(in_dir)
        import json
        meta = json.load(open(in_path / "meta.json", encoding="utf-8"))
        frags = []
        for frag_info in meta["fragments"]:
            df = pd.read_parquet(frag_info["path"])
            start = pd.to_datetime(frag_info["start"])
            end = pd.to_datetime(frag_info["end"])
            frags.append(TradingFragment(df=df, start=start, end=end, meta=frag_info.get("meta", {})))
        return TradingDataset(
            name=meta["name"],
            fragments=frags,
            symbols=meta["symbols"],
            candle_level=meta["candle_level"],
            meta=meta.get("meta", {})
        )

    def print_debug(self, max_frag: int = 5):
        print(f"== TradingDataset '{self.name}' ==")
        print(f"Symbols: {self.symbols}, candle_level: {self.candle_level}, n_frag={len(self.fragments)}")

        n = len(self.fragments)
        if n == 0:
            print("  (empty dataset)")
            return

        def _meta_str(frag):
            if not frag.meta:
                return ""
            items = []
            for k, v in frag.meta.items():
                if isinstance(v, float):
                    items.append(f"{k}={v:.4f}")
                else:
                    items.append(f"{k}={v}")
            return ", " + ", ".join(items) if items else ""

        # head
        for i, frag in enumerate(self.fragments[:max_frag]):
            print(f"  [HEAD {i}] {frag.start} -> {frag.end}, rows={len(frag.df)}{_meta_str(frag)}")
            print(f"       meta keys: {list(frag.meta.keys())}")
            print(f"       df cols : {list(frag.df.columns)[:8]}{' ...' if len(frag.df.columns)>8 else ''}")

        # tail
        if n > 2 * max_frag:
            print("  ...")
        for i, frag in enumerate(self.fragments[-max_frag:], start=n - max_frag):
            print(f"  [TAIL {i}] {frag.start} -> {frag.end}, rows={len(frag.df)}{_meta_str(frag)}")
            print(f"       meta keys: {list(frag.meta.keys())}")
            print(f"       df cols : {list(frag.df.columns)[:8]}{' ...' if len(frag.df.columns)>8 else ''}")



# ===== File: fin_rl/data/rl_env/fragment/utils.py =====

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

