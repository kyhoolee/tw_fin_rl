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
