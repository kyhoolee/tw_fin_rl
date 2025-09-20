#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline: load -> fragment by month -> datasets -> regime splits

Bản vá:
- Xây dựng dataframe "WIDE" theo chiều symbol bằng cách gọi
  load_aligned_dataset(..., symbols=[sym]) cho từng symbol rồi merge on open_time.
- Mỗi feature gắn hậu tố _{SYMBOL} (Close_BTCUSDT, QuoteVolume_ETHUSDT, ...).
- Cắt monthly fragments theo chiều thời gian; mỗi fragment chứa đủ chuỗi giá cho tất cả symbol.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict
import time
import numpy as np

from .trading_dataset import TradingDataset, TradingFragment
from .utils import load_aligned_dataset  # dùng lại hàm có sẵn
from .splitter_fragmented import FragmentedSplitter
from .splitter_regime import FlexibleRegimeSplitter

BASE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "QuoteVolume", "Trades", "TakerBuyBase", "TakerBuyQuote"
]


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


def _build_aligned_wide(data_dir: str, candle_level: str, symbols: List[str]) -> pd.DataFrame:
    """
    Tạo DataFrame WIDE:
      - Một cột thời gian chung: open_time
      - Các cột feature gắn hậu tố theo symbol: {FEAT}_{SYMBOL}
    Cách làm an toàn: gọi load_aligned_dataset cho từng symbol riêng rồi merge inner theo open_time.
    Điều này né được lỗi "mọi symbol dùng chung 1 cột" đã thấy trong log hiện tại. :contentReference[oaicite:2]{index=2}
    """
    def _one(sym: str) -> pd.DataFrame:
        # load cho đúng 1 symbol để tránh "gộp phẳng" mất chiều symbol
        df = load_aligned_dataset(data_dir, candle_level, [sym])
        # yêu cầu open_time + các cột chuẩn
        assert "open_time" in df.columns, f"missing open_time for {sym}"
        # chỉ giữ open_time + các cột tồn tại trong BASE_COLS
        keep = ["open_time"] + [c for c in BASE_COLS if c in df.columns]
        df = df[keep].sort_values("open_time").drop_duplicates(subset=["open_time"])
        # gắn hậu tố
        rename = {c: f"{c}_{sym}" for c in df.columns if c != "open_time"}
        df = df.rename(columns=rename)
        return df

    dfs = [_one(sym) for sym in symbols]
    out = dfs[0]
    for i in range(1, len(dfs)):
        out = out.merge(dfs[i], on="open_time", how="inner")

    out = out.sort_values("open_time").reset_index(drop=True)
    # sanity: yêu cầu Close_{sym} tồn tại
    for sym in symbols:
        assert f"Close_{sym}" in out.columns, f"Missing Close_{sym}; dataset must be wide per-symbol"
    return out


def make_monthly_fragments(df_wide: pd.DataFrame, symbols: List[str]) -> List[TradingFragment]:
    """
    Cắt theo tháng trên dataframe WIDE (đã có các cột _{SYMBOL}).
    """
    frags: List[TradingFragment] = []
    grouped = df_wide.groupby(df_wide["open_time"].dt.to_period("M"))
    for _, sub in grouped:
        if sub.empty:
            continue
        sub = sub.sort_values("open_time")
        frags.append(TradingFragment(
            df=sub,
            start=sub["open_time"].iloc[0],
            end=sub["open_time"].iloc[-1],
            meta={"symbols": symbols}
        ))
    return frags


@log_step(1, "Build monthly dataset (wide aligned + fragments)")
def build_datasets(data_dir: str, candle_level: str, symbols: List[str]) -> TradingDataset:
    # 1) Xây dataframe dạng WIDE đúng chuẩn chiều symbol
    full_wide = _build_aligned_wide(data_dir, candle_level, symbols)

    # 2) Cắt monthly fragments
    all_frags = make_monthly_fragments(full_wide, symbols)

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

    # summary nhỏ theo weighted return trong meta
    for name, ds in splits.items():
        rets = [f.meta.get("return_weighted", 0.0) for f in ds.fragments]
        if rets:
            print(f"[{name}] n_frag={len(ds.fragments)}, "
                  f"mean_return={np.mean(rets):.2%}, "
                  f"min={np.min(rets):.2%}, max={np.max(rets):.2%}")
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
