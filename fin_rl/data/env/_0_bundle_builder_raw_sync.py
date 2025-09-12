#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bundle_builder_raw_sync.py
--------------------------
Chia dữ liệu Binance OHLCV thành train/test/unseen (raw, không feature/label),
đồng bộ nhiều symbol để dùng cho mô hình portfolio.

Điểm nhấn:
- Tự động tính K (số candle) tương ứng với EMBARGO dựa trên candle-level (1m/5m/1h/1d...).
- Ghi rõ vào META.yaml: bar_duration, embargo_timedelta, embargo_bars (K).

Quy trình:
1) Load từng symbol (CSV → DataFrame).
2) Tìm đoạn giao nhau thời gian giữa tất cả symbols.
3) Snap 3 khoảng train/test/unseen vào đoạn giao nhau.
4) Giữ CHỈ các timestamp có đủ tất cả symbols (inner align).
5) Áp dụng EMBARGO: loại bỏ vùng train nằm trong [test_start - E, test_end + E].
6) Xuất parquet (gộp + shards), kèm META/checksums/report.
"""

import sys, os, json, argparse, hashlib, math
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# ---- import user's candle_io.py ----
sys.path.append(".")
import candle_io  # exposes load_symbol_csv

# -------------------------------------------------------
# Utilities
# -------------------------------------------------------

def parse_duration(s: str) -> pd.Timedelta:
    """
    Parse '30m', '12h', '2d', '15s' → pandas.Timedelta.
    Dùng cho embargo/buffer giữa train và test.
    """
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
    """
    Chuyển '1m','5m','15m','1h','4h','1d' → timedelta cho 1 bar.
    Hỗ trợ số + đơn vị (m, h, d, s).
    """
    x = level.strip().lower()
    # cho phép dạng '1min' '5min' nhanh gọn:
    x = x.replace("min", "m")
    # bắt buộc kết thúc bằng 1 ký tự đơn vị
    val = float(x[:-1]); unit = x[-1]
    if unit == "m": return pd.Timedelta(minutes=val)
    if unit == "h": return pd.Timedelta(hours=val)
    if unit == "d": return pd.Timedelta(days=val)
    if unit == "s": return pd.Timedelta(seconds=val)
    raise ValueError(f"Unsupported candle-level: {level}")

def embargo_bars_count(embargo_str: str, candle_level: str) -> int:
    """
    Tính K = số candle tối thiểu tương ứng với EMBARGO.
    K = ceil(embargo_timedelta / bar_duration).
    Ví dụ: embargo='12h', candle='1m' → K=ceil(12*60)=720.
    """
    E = parse_duration(embargo_str)
    if E <= pd.Timedelta(0):
        return 0
    bar_dt = bar_duration_from_level(candle_level)
    # division giữa Timedelta → float số lượng bars, làm tròn lên để an toàn.
    return int(math.ceil(E / bar_dt))

def md5_file(path: Path) -> str:
    """MD5 để track reproducibility."""
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
    return md5.hexdigest()

def ensure_utc(df: pd.DataFrame, ts_col: str = "open_time") -> pd.DataFrame:
    """Ép timestamp về UTC nếu thiếu tz."""
    if df[ts_col].dt.tz is None:
        df[ts_col] = df[ts_col].dt.tz_localize("UTC")
    return df

def calc_intersection_range(per_sym: Dict[str, pd.DataFrame], ts_col="open_time"):
    """Giao nhau thời gian giữa các symbol: [max(starts) .. min(ends)]."""
    starts, ends = [], []
    for sym, g in per_sym.items():
        g = g.sort_values(ts_col)
        starts.append(g[ts_col].iloc[0])
        ends.append(g[ts_col].iloc[-1])
    return max(starts), min(ends)

def inner_align(full: pd.DataFrame, symbols: List[str], ts_col="open_time") -> pd.DataFrame:
    """
    Chỉ giữ các timestamp có đủ ALL symbols.
    Trả về df đã sort, unique theo (symbol, ts).
    """
    cnt = full.groupby(ts_col)["symbol"].nunique()
    good_ts = cnt[cnt == len(symbols)].index
    aligned = full[full[ts_col].isin(good_ts)].copy()
    return aligned.sort_values([ts_col, "symbol"]).drop_duplicates(["symbol", ts_col])

def slice_time(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, ts_col="open_time") -> pd.DataFrame:
    """Cắt df theo [start, end] (đã UTC)."""
    m = (df[ts_col] >= start) & (df[ts_col] <= end)
    return df.loc[m].copy()

def apply_embargo_to_train(train: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp,
                           embargo: str, ts_col="open_time") -> pd.DataFrame:
    """
    Áp dụng EMBARGO cho TRAIN: loại bỏ các bar trong khoảng [test_start - E, test_end + E].

    Ý nghĩa K-candle:
      - Với candle-level có bar_duration = Δ, và embargo=E,
        K = ceil(E / Δ) là số candle tối thiểu cần cách ly train khỏi test.
      - Ví dụ: level=1m (Δ=1 phút), embargo=12h → K=720 bar.
      - Điều này đặc biệt quan trọng khi tầng model/feature dùng rolling dài:
        train gần ranh giới sẽ không "nhìn thấy" thông tin gây rò rỉ từ test.
    """
    if not embargo:
        return train
    E = parse_duration(embargo)
    lo = test_start - E
    hi = test_end + E
    return train[(train[ts_col] < lo) | (train[ts_col] > hi)].copy()

# -------------------------------------------------------
# Core builder
# -------------------------------------------------------

def load_per_symbol(data_dir: str, candle_level: str, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Load từng symbol với candle_io, ép UTC, sort theo thời gian."""
    store = {}
    for sym in symbols:
        df, _fp = candle_io.load_symbol_csv(sym, data_dir, candle_level)
        df["symbol"] = sym
        df = ensure_utc(df, "open_time").sort_values("open_time").reset_index(drop=True)
        store[sym] = df
    return store

def build_bundle_raw_sync(
    data_dir: str,
    candle_level: str,
    symbols: List[str],
    out_dir: str,
    train_start: str, train_end: str,
    test_start: str,  test_end: str,
    unseen_start: str, unseen_end: str,
    embargo: str = "",
    write_shards: bool = True
) -> Dict[str, Any]:

    out_dir = Path(out_dir)
    (out_dir / "splits").mkdir(parents=True, exist_ok=True)
    if write_shards:
        for split in ["train", "test", "unseen"]:
            (out_dir / "shards" / split).mkdir(parents=True, exist_ok=True)

    # 1) Load per symbol
    per = load_per_symbol(data_dir, candle_level, symbols)

    # 2) Intersection range across symbols
    inter_start, inter_end = calc_intersection_range(per, "open_time")

    # 3) Concatenate and cut to intersection
    full = pd.concat(list(per.values()), ignore_index=True)
    full = slice_time(full, inter_start, inter_end, "open_time")

    # 4) Inner align: chỉ giữ timestamp có đủ ALL symbols
    before_align = len(full)
    full = inner_align(full, symbols, "open_time")
    after_align = len(full)

    # 5) Clamp user ranges to intersection
    def clamp(a, b):
        lo = max(pd.Timestamp(a, tz="UTC"), inter_start)
        hi = min(pd.Timestamp(b, tz="UTC"), inter_end)
        return lo, hi
    t0, t1 = clamp(train_start, train_end)
    s0, s1 = clamp(test_start,  test_end)
    u0, u1 = clamp(unseen_start, unseen_end)

    # 6) Split
    train = slice_time(full, t0, t1, "open_time")
    test  = slice_time(full, s0, s1, "open_time")
    unseen= slice_time(full, u0, u1, "open_time")

    # 7) Embargo train quanh test
    train = apply_embargo_to_train(train, s0, s1, embargo, "open_time")

    # 8) Save splits
    p_train  = out_dir / "splits" / "train.parquet"
    p_test   = out_dir / "splits" / "test.parquet"
    p_unseen = out_dir / "splits" / "unseen.parquet"
    train.to_parquet(p_train, index=False)
    test.to_parquet(p_test, index=False)
    unseen.to_parquet(p_unseen, index=False)

    # 9) Per-symbol shards (tùy chọn)
    if write_shards:
        for name, df in [("train", train), ("test", test), ("unseen", unseen)]:
            for sym, g in df.groupby("symbol"):
                g.to_parquet(out_dir / "shards" / name / f"{sym}.parquet", index=False)

    # 10) META / checksums / report
    import yaml
    bar_dt = bar_duration_from_level(candle_level)
    K = embargo_bars_count(embargo, candle_level)
    meta = {
        "symbols": symbols,
        "freq": candle_level,
        "bar_duration": str(bar_dt),          # ví dụ: '0 days 00:01:00' cho 1m
        "intersection_range": {"start": inter_start.isoformat(), "end": inter_end.isoformat()},
        "effective_ranges": {
            "train":  {"start": t0.isoformat(), "end": t1.isoformat()},
            "test":   {"start": s0.isoformat(), "end": s1.isoformat()},
            "unseen": {"start": u0.isoformat(), "end": u1.isoformat()},
        },
        # Embargo phần ghi chú rõ ràng
        "embargo": {
            "input": embargo or None,             # chuỗi đầu vào, ví dụ '12h'
            "timedelta": str(parse_duration(embargo)),  # iso-like string của timedelta
            "bars": K,                             # K = ceil(E / bar_duration)
            "note": "Train rows in [test_start - E, test_end + E] are removed. K is the minimum bar-gap."
        },
        "alignment_info": {
            "before_align_rows": int(before_align),
            "after_align_rows": int(after_align),
            "dropped_rows_due_to_alignment": int(before_align - after_align),
        }
    }
    with open(out_dir / "META.yaml", "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)

    checksums = {
        "splits/train.parquet":  md5_file(p_train),
        "splits/test.parquet":   md5_file(p_test),
        "splits/unseen.parquet": md5_file(p_unseen),
    }
    if write_shards:
        for split in ["train", "test", "unseen"]:
            for sym in symbols:
                fp = out_dir / "shards" / split / f"{sym}.parquet"
                if fp.exists():
                    checksums[str(fp.relative_to(out_dir))] = md5_file(fp)
    with open(out_dir / "checksums.json", "w") as f:
        json.dump(checksums, f, indent=2)

    report = {
        "rows": {"train": len(train), "test": len(test), "unseen": len(unseen)},
        "paths": {"train": str(p_train), "test": str(p_test), "unseen": str(p_unseen)},
        "embargo_bars": K
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    return {"meta": meta, "report": report}

# -------------------------------------------------------
# CLI
# -------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Split raw, synchronized OHLCV bundles (train/test/unseen)")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--candle-level", default="1m")
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--train-start", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--test-start", required=True)
    ap.add_argument("--test-end", required=True)
    ap.add_argument("--unseen-start", required=True)
    ap.add_argument("--unseen-end", required=True)
    ap.add_argument("--embargo", default="", help="Buffer zone (e.g. '12h') between train and test")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-shards", action="store_true", help="Disable per-symbol shard output")
    args = ap.parse_args()

    res = build_bundle_raw_sync(
        data_dir=args.data_dir,
        candle_level=args.candle_level,
        symbols=args.symbols,
        out_dir=args.out_dir,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start,   test_end=args.test_end,
        unseen_start=args.unseen_start, unseen_end=args.unseen_end,
        embargo=args.embargo,
        write_shards=(not args.no_shards)
    )
    print(json.dumps(res["report"], indent=2))

if __name__ == "__main__":
    main()
