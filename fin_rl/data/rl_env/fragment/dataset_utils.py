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
