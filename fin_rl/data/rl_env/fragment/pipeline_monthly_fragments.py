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

# ========================================================================

# fin_rl/data/rl_env/fragment/pipeline_monthly_fragments.py
from typing import List, Sequence
import pandas as pd

# ---- cấu hình mặc định cho các field
DEFAULT_FIELDS: Sequence[str] = (
    "Open","High","Low","Close",
    "Volume","QuoteVolume","Trades",
    "TakerBuyBase","TakerBuyQuote"
)

def _build_aligned_wide(
    data_dir: str,
    candle_level: str,
    symbols: List[str],
    fields: Sequence[str] = DEFAULT_FIELDS,
) -> pd.DataFrame:
    """
    Trả về WIDE frame:
      - 1 cột thời gian 'open_time'
      - Các cột {field}_{symbol} cho mọi (field, symbol) tồn tại.
    Cách làm: load align từng symbol riêng (tránh mất chiều symbol), rồi merge inner theo open_time.
    """
    def _one(sym: str) -> pd.DataFrame:
        df = load_aligned_dataset(data_dir, candle_level, [sym])  # <- API cũ, đã align/UTC
        assert "open_time" in df.columns, f"missing open_time for {sym}"

        # chỉ giữ các field có thật, + open_time
        keep = ["open_time"] + [f for f in fields if f in df.columns]
        if keep == ["open_time"]:
            raise KeyError(f"No requested fields present for {sym}. Requested={list(fields)}; df cols={list(df.columns)}")

        df = (
            df[keep]
            .sort_values("open_time")
            .drop_duplicates(subset=["open_time"])
        )
        # gắn hậu tố symbol
        rename = {c: f"{c}_{sym}" for c in df.columns if c != "open_time"}
        return df.rename(columns=rename)

    # merge inner theo open_time để đảm bảo mọi timestamp đủ symbols
    dfs = [_one(sym) for sym in symbols]
    out = dfs[0]
    for i in range(1, len(dfs)):
        out = out.merge(dfs[i], on="open_time", how="inner")

    out = out.sort_values("open_time").reset_index(drop=True)

    # sanity: với regime splitter mặc định dùng Close & QuoteVolume
    for sym in symbols:
        need_close = f"Close_{sym}"
        need_qv = f"QuoteVolume_{sym}"
        if need_close not in out.columns:
            raise KeyError(f"Missing {need_close} in wide frame.")
        if need_qv not in out.columns:
            # không bắt buộc tuyệt đối nếu bạn set weight_field khác; chỉ cảnh báo
            print(f"[WARN] {need_qv} not found. If you weight by another field, set weight_field trong splitter.")

    return out


@log_step(1, "Build monthly dataset (wide aligned + fragments)")
def build_datasets(
    data_dir: str,
    candle_level: str,
    symbols: List[str],
    fields: Sequence[str] = DEFAULT_FIELDS,   # <-- thêm tham số
) -> TradingDataset:
    df_wide = _build_aligned_wide(data_dir, candle_level, symbols, fields=fields)

    # group theo tháng (lưu ý warning: Period drop tz là ok vì chỉ để group)
    grouped = df_wide.groupby(df_wide["open_time"].dt.to_period("M"))
    all_frags = []
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


# @log_step(1, "Build monthly dataset (wide aligned + fragments)")
# def build_datasets(data_dir: str, candle_level: str, symbols: List[str]) -> TradingDataset:
#     # 1) Xây dataframe dạng WIDE đúng chuẩn chiều symbol
#     full_wide = _build_aligned_wide(data_dir, candle_level, symbols)

#     # 2) Cắt monthly fragments
#     all_frags = make_monthly_fragments(full_wide, symbols)

#     log(f"Generated {len(all_frags)} monthly fragments")
#     n_rows = sum(len(f.df) for f in all_frags)
#     log(f"Total rows: {n_rows:,}")
#     return TradingDataset(
#         name="monthly_fragments",
#         fragments=all_frags,
#         symbols=symbols,
#         candle_level=candle_level
#     )


#==============================================================




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


from .utils import (
    schema_summary, 
    stats_overview,
    nan_report,
    preview_fragment,
)

def print_debug_ds(dataset: TradingDataset):
    dataset.print_debug(max_frag=3)  # cái hiện có
    schema_summary(dataset)
    stats_overview(dataset)
    nan_report(dataset, topk=20)
    preview_fragment(dataset, 0, n=10, fields=["Open","High","Low","Close","QuoteVolume"], symbols=dataset.symbols)
#==============================================================


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
        print_debug_ds(v)

    regime_splits = example_flexible_split(dataset)
    print("\n== Regime splits ==")
    for k, v in regime_splits.items():
        log(f"Processing regime: {k}, {len(v.fragments)} fragments")
        v.print_debug(max_frag=3)
        out_dir = Path(f"./work/datasets/{dataset.name}_regime_{k}")
        v.save(out_dir)
        log(f"Saved regime '{k}' to {out_dir}")
        TradingDataset.load(out_dir).print_debug(max_frag=2)
        print_debug_ds(v)
