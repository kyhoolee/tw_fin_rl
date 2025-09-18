#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline: load -> fragment by month -> datasets -> regime splits
"""

import pandas as pd
from pathlib import Path
from typing import List

from fin_rl.data.env.fragment.trading_dataset import TradingDataset, TradingFragment
from fin_rl.data.env.fragment.utils import load_aligned_dataset
from fin_rl.data.env.fragment.splitter_fragmented import FragmentedSplitter
from fin_rl.data.env.fragment.splitter_regime import tag_regimes, RegimeSplitter


def make_monthly_fragments(df: pd.DataFrame, symbol: str) -> List[TradingFragment]:
    """Cắt DataFrame thành fragments theo tháng."""
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


def build_datasets(data_dir: str, candle_level: str, symbols: List[str]) -> TradingDataset:
    # Load & align đủ symbols
    full = load_aligned_dataset(data_dir, candle_level, symbols)

    # Tách fragments theo tháng
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

    return TradingDataset(
        name="monthly_fragments",
        fragments=all_frags,
        symbols=symbols,
        candle_level=candle_level
    )


def example_fragmented_split(dataset: TradingDataset):
    """Ví dụ: chia train/test/unseen bằng index fragments."""
    assign = {
        "train": list(range(0, 24)),   # 2 năm đầu
        "valid": list(range(24, 30)),  # 6 tháng tiếp
        "test":  list(range(30, 36)),  # 6 tháng tiếp
        "unseen": list(range(36, len(dataset.fragments)))  # phần còn lại
    }
    splitter = FragmentedSplitter(assign)
    return splitter.split(dataset)


def example_regime_split(dataset: TradingDataset):
    """Ví dụ: tag regime và chia theo bull/bear/sideways."""
    for frag in dataset.fragments:
        frag.df = tag_regimes(frag.df, price_col="Close", window=50)

    regime_map = {
        "train": ["bull", "sideways"],
        "valid": ["bear"],
        "test":  ["bull"],
        "unseen": ["sideways"]
    }
    splitter = RegimeSplitter(regime_map)
    return splitter.split(dataset)


if __name__ == "__main__":
    data_dir = "./work/data/binance/spot/1m"
    candle_level = "1m"
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]

    # Step 1–2: load & monthly fragments
    dataset = build_datasets(data_dir, candle_level, symbols)
    print(f"Dataset '{dataset.name}' with {len(dataset.fragments)} fragments")

    # Step 3: split by fragment index
    frag_splits = example_fragmented_split(dataset)
    print("\n== Fragmented splits ==")
    for k, v in frag_splits.items():
        print(f"--- {k} ---")
        v.print_debug()
        out_dir = Path(f"./work/datasets/{dataset.name}_frag_{k}")
        v.save(out_dir)
        print(f"Saved split '{k}' to {out_dir}")
        # Load lại để test
        loaded = TradingDataset.load(out_dir)
        loaded.print_debug(max_frag=2)

    # Step 4: regime split
    regime_splits = example_regime_split(dataset)
    print("\n== Regime splits ==")
    for k, v in regime_splits.items():
        print(f"--- {k} ---")
        v.print_debug()
        out_dir = Path(f"./work/datasets/{dataset.name}_regime_{k}")
        v.save(out_dir)
        print(f"Saved regime '{k}' to {out_dir}")
        # Load lại để test
        loaded = TradingDataset.load(out_dir)
        loaded.print_debug(max_frag=2)
