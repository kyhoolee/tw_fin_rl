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



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataset_utils.py
----------------
Tiện ích cho TradingDataset:
- save/load ra JSONL + Parquet
- print debug summary
"""

import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd


def save_dataset(ds: TradingDataset, out_dir: str):
    """
    Lưu TradingDataset:
    - meta.json: metadata chung
    - fragments/<i>.parquet: dữ liệu từng fragment
    """
    out_path = Path(out_dir)
    (out_path / "fragments").mkdir(parents=True, exist_ok=True)

    # Lưu meta
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
        print(f"  Frag {i}: {frag.start} -> {frag.end}, rows={len(frag.df)}")
    if len(ds.fragments) > max_frag:
        print(f"  ... {len(ds.fragments) - max_frag} more fragments ...")
