from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import json
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
    """
    Dataset gồm nhiều fragment (train/test/unseen đều dùng format này).

    YÊU CẦU MỚI:
    - Mỗi fragment df ở dạng WIDE, với các cột per-symbol gắn hậu tố:
        Close_{SYM}, QuoteVolume_{SYM}, ... (Open/High/Low/Volume/TakerBuy... tương tự nếu có)
    - Cột chung thời gian: open_time
    """
    name: str
    fragments: List[TradingFragment]
    symbols: List[str]
    candle_level: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def concat(self) -> pd.DataFrame:
        """Nối toàn bộ fragment thành 1 DataFrame."""
        if not self.fragments:
            return pd.DataFrame()
        return pd.concat([f.df for f in self.fragments], ignore_index=True)

    def time_ranges(self) -> List[tuple]:
        return [(f.start, f.end) for f in self.fragments]

    def filter_by_regime(self, regimes: List[str]) -> "TradingDataset":
        """Tạo dataset mới chỉ chứa rows có regime trong list."""
        new_frags = []
        for frag in self.fragments:
            if "regime" not in frag.meta and "regime" not in frag.df.columns:
                # Không có thông tin regime -> bỏ qua
                continue
            # Ưu tiên meta.regime theo fragment (toàn khối). Nếu muốn row-wise theo df['regime'] có thể mở rộng sau.
            if frag.meta.get("regime") in regimes:
                new_frags.append(
                    TradingFragment(df=frag.df.copy(),
                                    start=frag.start,
                                    end=frag.end,
                                    meta=frag.meta.copy())
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
            json.dump(meta, f, indent=2)

    @staticmethod
    def load(in_dir: str) -> "TradingDataset":
        """Load TradingDataset từ thư mục (meta.json + fragments)."""
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

    # ======= Helpers cho debug & sanity =======

    def _per_symbol_missing(self) -> Dict[str, List[str]]:
        """Trả về map symbol -> list cột thiếu (Close_{sym}, QuoteVolume_{sym})."""
        missing = {}
        if not self.fragments:
            return missing
        cols = set(self.fragments[0].df.columns)
        for sym in self.symbols:
            need = [f"Close_{sym}", f"QuoteVolume_{sym}"]
            miss = [c for c in need if c not in cols]
            if miss:
                missing[sym] = miss
        return missing

    def print_debug(self, max_frag: int = 5):
        print(f"== TradingDataset '{self.name}' ==")
        print(f"Symbols: {self.symbols}, candle_level: {self.candle_level}, n_frag={len(self.fragments)}")

        # Cảnh báo nhanh nếu thiếu cột per-symbol
        miss = self._per_symbol_missing()
        if miss:
            print("[WARN] Missing per-symbol columns in fragments (showing first frag schema):", miss)

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
            # gợi ý: show một số cột Close_* để đảm bảo wide đúng
            show_cols = ["open_time"] + [c for c in frag.df.columns if c.startswith("Close_")][:6]
            if len(show_cols) == 1:  # không có Close_*
                show_cols = list(frag.df.columns)[:8]
            print(f"       df cols : {show_cols}{' ...' if len(frag.df.columns)>len(show_cols) else ''}")

        # tail
        if n > 2 * max_frag:
            print("  ...")
        for i, frag in enumerate(self.fragments[-max_frag:], start=n - max_frag):
            print(f"  [TAIL {i}] {frag.start} -> {frag.end}, rows={len(frag.df)}{_meta_str(frag)}")
            print(f"       meta keys: {list(frag.meta.keys())}")
            show_cols = ["open_time"] + [c for c in frag.df.columns if c.startswith("Close_")][:6]
            if len(show_cols) == 1:
                show_cols = list(frag.df.columns)[:8]
            print(f"       df cols : {show_cols}{' ...' if len(frag.df.columns)>len(show_cols) else ''}")
