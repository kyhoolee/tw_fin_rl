# fin_rl/data/rl_env/fragment/trading_dataset.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence
import json
import pandas as pd

@dataclass
class TradingFragment:
    df: pd.DataFrame
    start: pd.Timestamp
    end: pd.Timestamp
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TradingDataset:
    """
    Dataset gồm nhiều fragment.
    - Mỗi fragment df ở dạng WIDE với cột 'open_time' và các cột {field}_{symbol}.
    """
    name: str
    fragments: List[TradingFragment]
    symbols: List[str]
    candle_level: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def concat(self) -> pd.DataFrame:
        return pd.concat([f.df for f in self.fragments], ignore_index=True) if self.fragments else pd.DataFrame()

    def time_ranges(self) -> List[tuple]:
        return [(f.start, f.end) for f in self.fragments]

    def filter_by_regime(self, regimes: List[str]) -> "TradingDataset":
        new_frags = []
        for frag in self.fragments:
            if frag.meta.get("regime") in regimes:
                new_frags.append(
                    TradingFragment(df=frag.df.copy(), start=frag.start, end=frag.end, meta=frag.meta.copy())
                )
        return TradingDataset(
            name=f"{self.name}_regime_{'_'.join(regimes)}",
            fragments=new_frags,
            symbols=self.symbols,
            candle_level=self.candle_level,
            meta=self.meta.copy()
        )

    # ---------- I/O ----------
    def save(self, out_dir: str):
        out = Path(out_dir)
        (out / "fragments").mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "symbols": self.symbols,
            "candle_level": self.candle_level,
            "meta": self.meta,
            "fragments": [],
        }
        for i, frag in enumerate(self.fragments):
            p = out / "fragments" / f"frag_{i}.parquet"
            frag.df.to_parquet(p, index=False)
            meta["fragments"].append({
                "i": i,
                "start": str(frag.start),
                "end": str(frag.end),
                "path": str(p),
                "meta": frag.meta,
            })
        with open(out / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load(in_dir: str) -> "TradingDataset":
        p = Path(in_dir)
        meta = json.load(open(p / "meta.json", encoding="utf-8"))
        frags = []
        for info in meta["fragments"]:
            df = pd.read_parquet(info["path"])
            start = pd.to_datetime(info["start"])
            end = pd.to_datetime(info["end"])
            frags.append(TradingFragment(df=df, start=start, end=end, meta=info.get("meta", {})))
        return TradingDataset(
            name=meta["name"],
            fragments=frags,
            symbols=meta["symbols"],
            candle_level=meta["candle_level"],
            meta=meta.get("meta", {}),
        )

    # ---------- Helpers ----------
    def per_symbol_missing(self, require: Sequence[str] = ("Close","QuoteVolume")) -> Dict[str, List[str]]:
        """Map symbol -> list cột thiếu (ví dụ Close_{sym}, QuoteVolume_{sym})."""
        miss = {}
        if not self.fragments:
            return miss
        cols = set(self.fragments[0].df.columns)
        for sym in self.symbols:
            need = [f"{f}_{sym}" for f in require]
            lack = [c for c in need if c not in cols]
            if lack:
                miss[sym] = lack
        return miss

    def print_debug(self, max_frag: int = 5):
        print(f"== TradingDataset '{self.name}' ==")
        print(f"Symbols: {self.symbols}, candle_level: {self.candle_level}, n_frag={len(self.fragments)}")

        miss = self.per_symbol_missing()
        if miss:
            print("[WARN] Missing per-symbol columns in fragments (showing first frag schema):", miss)

        n = len(self.fragments)
        if n == 0:
            print("  (empty dataset)")
            return

        def _meta_str(frag):
            if not frag.meta: return ""
            items = []
            for k, v in frag.meta.items():
                items.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
            return ", " + ", ".join(items) if items else ""

        # head
        for i, frag in enumerate(self.fragments[:max_frag]):
            print(f"  [HEAD {i}] {frag.start} -> {frag.end}, rows={len(frag.df)}{_meta_str(frag)}")
            print(f"       meta keys: {list(frag.meta.keys())}")
            show_cols = ["open_time"] + [c for c in frag.df.columns if c.startswith("Close_")][:6]
            if len(show_cols) == 1: show_cols = list(frag.df.columns)[:8]
            print(f"       df cols : {show_cols}{' ...' if len(frag.df.columns)>len(show_cols) else ''}")

        if n > 2 * max_frag:
            print("  ...")
        # tail
        for i, frag in enumerate(self.fragments[-max_frag:], start=n - max_frag):
            print(f"  [TAIL {i}] {frag.start} -> {frag.end}, rows={len(frag.df)}{_meta_str(frag)}")
            print(f"       meta keys: {list(frag.meta.keys())}")
            show_cols = ["open_time"] + [c for c in frag.df.columns if c.startswith("Close_")][:6]
            if len(show_cols) == 1: show_cols = list(frag.df.columns)[:8]
            print(f"       df cols : {show_cols}{' ...' if len(frag.df.columns)>len(show_cols) else ''}")
