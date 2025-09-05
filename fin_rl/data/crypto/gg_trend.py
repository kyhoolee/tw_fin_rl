# fin_rl/data/proxy/google_trends.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Trends collector (daily, 5y+) with window stitching & cross-window scaling.

- Keywords fixed for 4 tokens (clean terms):
  Bitcoin, Ethereum, Binance Coin, Solana
- Queries per "group" (pairs + all-4) over 90-day windows for daily granularity
- Overlap-based scaling between consecutive windows to remove 0–100-per-window bias
- Saves one parquet per group under out-dir

CLI:
  python -m fin_rl.data.proxy.google_trends \
    --start 2019-01-01 --end 2025-09-01 \
    --out-dir ./work/data/proxy/google_trends \
    --sleep-min 5 --sleep-max 10 \
    --window-days 90 \
    --log DEBUG
"""

from __future__ import annotations
import os
import sys
import time
import math
import json
import random
import logging
import argparse
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional

import pandas as pd
from dateutil import parser as dtp
from tqdm import tqdm
from pytrends.request import TrendReq

UTC = timezone.utc

# ----- Default clean keywords for the 4 tokens -----
KW_MAP_DEFAULT = {
    "BTCUSDT": "Bitcoin",
    "ETHUSDT": "Ethereum",
    "BNBUSDT": "Binance Coin",
    "SOLUSDT": "Solana",
}

# ----- Utility -----
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _daterange_windows(start: datetime, end: datetime, window_days: int) -> List[Tuple[datetime, datetime]]:
    """Create (inclusive) 90-day like windows with 1 day overlap for smoother stitching."""
    assert end > start
    win = []
    cur = start
    step = timedelta(days=window_days)
    # Add small overlap (e.g., 5 days) to improve scaling stability
    overlap = timedelta(days=5)

    while cur < end:
        w_start = cur
        w_end = min(end, cur + step)
        win.append((w_start, w_end))
        # next window begins slightly before previous ended (overlap)
        cur = w_end - overlap
    return win

def _sleep(smin: float, smax: float):
    t = random.uniform(smin, smax)
    time.sleep(t)

def _safe_mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else 1.0

def _stitch_windows(dfs: List[pd.DataFrame], columns: List[str]) -> pd.DataFrame:
    """
    Stitch list of daily dataframes with same columns using overlap-based scaling.

    Approach:
      - Keep first window as base (no scaling).
      - For each subsequent window, compute overlap with existing index.
      - For each column, scale the new window so that overlap means align:
            scale = (mean_existing_overlap / mean_new_overlap)
        If insufficient overlap data -> fallback scale = 1.
      - Append non-overlapping tail from new window.

    Returns a single daily dataframe (index=date, columns=columns).
    """
    if not dfs:
        return pd.DataFrame(columns=columns)

    base = dfs[0].copy()
    base = base[columns]
    for df in dfs[1:]:
        cur = df[columns].copy()
        # align on overlapping dates
        overlap_idx = base.index.intersection(cur.index)
        if len(overlap_idx) >= 3:  # need at least a few points to be stable
            for col in columns:
                ex_mean = _safe_mean(base.loc[overlap_idx, col])
                nw_mean = _safe_mean(cur.loc[overlap_idx, col])
                scale = ex_mean / nw_mean if nw_mean not in (0, None) else 1.0
                if not math.isfinite(scale) or scale <= 0:
                    scale = 1.0
                cur[col] = cur[col] * scale
        else:
            logging.debug("[stitch] insufficient overlap; skip scaling this window")

        # append non-overlap part
        add_idx = cur.index.difference(base.index)
        base = pd.concat([base, cur.loc[add_idx]], axis=0).sort_index()

    # Ensure numeric & fill missing with 0 (Trends may have NaN holidays)
    for c in columns:
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0)
    return base

def _fetch_window(pytrends: TrendReq, kw_list: List[str], start: datetime, end: datetime,
                  cat: Optional[int]) -> pd.DataFrame:
    """
    Fetch one window [start, end] (inclusive) as daily if window_days <= ~270.
    Returns dataframe indexed by date, columns = kw_list (int), dropped 'isPartial'.
    """
    tf = f"{start.strftime('%Y-%m-%d')} {end.strftime('%Y-%m-%d')}"
    pytrends.build_payload(
            kw_list, 
            timeframe=tf, 
            cat=cat,
            geo=""
        )
    df = pytrends.interest_over_time()
    if df is None or df.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
    # Drop isPartial; keep keywords only
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    df.index.name = "date"
    # Ensure daily index (some rare cases weekly if Google decides so)
    # If freq is weekly, upsample to daily via forward-fill
    if len(df.index) > 1:
        day_diff = (df.index[1] - df.index[0]).days
        if day_diff > 1:
            # Upsample to daily; ffill within window
            idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
            df = df.reindex(idx).ffill()
            df.index.name = "date"
    return df

def fetch_group_daily(kw_list: List[str],
                      start: datetime,
                      end: datetime,
                      window_days: int = 90,
                      sleep_min: float = 5.0,
                      sleep_max: float = 10.0,
                      retries: int = 3,
                      category: Optional[int] = 7) -> pd.DataFrame:
    """
    Fetch full-range daily Google Trends for a given group of keywords,
    by stitching 90-day windows with overlap-based scaling.

    Args:
        kw_list: up to 5 keywords (we use ≤4).
        start, end: datetime (UTC ok).
        window_days: 90 recommended for daily granularity.
        sleep_min, sleep_max: seconds to sleep between calls.
        retries: per-window retries on transient errors.
        category: Google Trends category id (7=Finance). None => all categories.

    Returns:
        DataFrame indexed by date with columns = normalized group keywords.
    """
    assert 1 <= len(kw_list) <= 5, "kw_list must have 1..5 keywords"
    # Normalize column names (snake case) for consistency
    colmap = {k: k.lower().replace(" ", "_") for k in kw_list}
    columns = list(colmap.values())

    # tr = TrendReq(hl="en-US", tz=0)  # tz=0 => UTC-like

    print("START TrendRequest here =========================")

    tr = TrendReq(
        hl="en-US",
        tz=0,
        geo="",  # << quan trọng: để rỗng (hoặc "VN") để tránh hit /?geo=US 404
        timeout=(10, 30),
        retries=2,
        backoff_factor=0.5,
        requests_args={
            "headers": {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            }
        },
    )

    print("=== TrendReq attrs ===")
    print("hl:", tr.hl, "tz:", tr.tz, "geo:", tr.geo)
    print("proxies:", tr.proxies)
    print("requests_args:", tr.requests_args)
    print("session headers:", dict(tr.s.headers))
    print("cookies:", tr.s.cookies.get_dict())
    print("======================")

    print("END TrendRequest here =========================")

    windows = _daterange_windows(start, end, window_days)
    logging.info(f"[group] keywords={kw_list} | windows={len(windows)} | range={start.date()}..{end.date()}")

    dfs: List[pd.DataFrame] = []
    for (ws, we) in tqdm(windows, desc=f"{'/'.join(columns)}"):
        # Retry logic per window
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                logging.info(f"[fetch] {kw_list} | {ws.date()}..{we.date()} | attempt={attempt}")
                df = _fetch_window(tr, kw_list, ws, we, cat=category)
                _sleep(sleep_min, sleep_max)
                if df is None or df.empty:
                    logging.warning(f"[fetch] empty window {ws.date()}..{we.date()}")
                    break
                df = df.rename(columns=colmap)
                dfs.append(df)
                logging.debug(f"[fetch] rows={len(df)} head={df.head(1).to_dict()}")
                break
            except Exception as e:
                last_err = e
                logging.warning(f"[retry] error={e} | sleeping before retry")
                _sleep(sleep_min + 2.0, sleep_max + 4.0)
        else:
            # after retries
            logging.error(f"[skip] window failed after {retries} attempts: {ws}..{we} | err={last_err}")

    stitched = _stitch_windows(dfs, columns=columns)
    # Clip negative (just in case scaling numeric issues)
    stitched = stitched.clip(lower=0.0)
    return stitched

def save_parquet(df: pd.DataFrame, out_path: str):
    _ensure_dir(os.path.dirname(out_path))
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "date"}, inplace=True)
    df.to_parquet(out_path, index=False)
    logging.info(f"[save] {out_path} rows={len(df)} cols={list(df.columns)}")

def _group_title(cols: List[str]) -> str:
    return "pair_" + "_".join(cols) if len(cols) == 2 else "group_" + "_".join(cols)




def quick_test():
    """
    Run a quick test: fetch ~30 days window for Bitcoin & Ethereum,
    print the head of dataframe.
    """
    from datetime import datetime, timedelta, timezone
    tr_start = datetime.now(timezone.utc) - timedelta(days=30)
    tr_end = datetime.now(timezone.utc)

    kws = ["Bitcoin", "Ethereum"]
    print(f"[quick_test] Fetching {kws} from {tr_start.date()} to {tr_end.date()}")

    df = fetch_group_daily(
        kw_list=kws,
        start=tr_start,
        end=tr_end,
        window_days=30,      # shorter window to force daily
        sleep_min=1,         # faster for test
        sleep_max=2,
        retries=1,
        category=7
    )
    print(df.head(10))
    print(f"[quick_test] rows={len(df)} cols={df.columns.tolist()}")





def main():
    ap = argparse.ArgumentParser(description="Google Trends daily collector (pairs + all-4)")
    ap.add_argument("--start", type=str, help="YYYY-MM-DD")
    ap.add_argument("--end", type=str, help="YYYY-MM-DD (exclusive or inclusive; we'll include this day)")
    ap.add_argument("--out-dir", type=str, default="./work/data/proxy/google_trends")
    ap.add_argument("--window-days", type=int, default=90)
    ap.add_argument("--sleep-min", type=float, default=5.0)
    ap.add_argument("--sleep-max", type=float, default=10.0)
    ap.add_argument("--log", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    ap.add_argument("--test", action="store_true", help="Run quick test instead of full job")
    
    
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )


    if args.test:
        quick_test()
        return

    start = dtp.parse(args.start).replace(tzinfo=UTC)
    # include end date by adding 1 day (Google Trends timeframe end is inclusive-ish)
    end = (dtp.parse(args.end) + timedelta(days=1)).replace(tzinfo=UTC)

    # Keywords (clean) for our 4 tokens
    kw_map = KW_MAP_DEFAULT.copy()
    kw_all = [
        kw_map["BTCUSDT"],        # "Bitcoin"
        kw_map["ETHUSDT"],        # "Ethereum"
        kw_map["BNBUSDT"],        # "Binance Coin"
        kw_map["SOLUSDT"],        # "Solana"
    ]
    # Build all pairs + all-4 group
    groups: List[List[str]] = [
        [kw_map["BTCUSDT"], kw_map["ETHUSDT"]],
        [kw_map["BTCUSDT"], kw_map["BNBUSDT"]],
        [kw_map["BTCUSDT"], kw_map["SOLUSDT"]],
        [kw_map["ETHUSDT"], kw_map["BNBUSDT"]],
        [kw_map["ETHUSDT"], kw_map["SOLUSDT"]],
        [kw_map["BNBUSDT"], kw_map["SOLUSDT"]],
        kw_all,  # all four together
    ]

    for kws in groups:
        cols = [k.lower().replace(" ", "_") for k in kws]
        title = _group_title(cols)
        logging.info(f"[run] group={title} | keywords={kws}")
        df = fetch_group_daily(
            kw_list=kws,
            start=start,
            end=end,
            window_days=args.window_days,
            sleep_min=args.sleep_min,
            sleep_max=args.sleep_max,
            retries=3,
            category=7,  # Finance
        )
        out_fp = os.path.join(args.out_dir, f"{title}.parquet")
        save_parquet(df, out_fp)

    logging.info("[done] all groups fetched & saved.")


if __name__ == "__main__":
    main()
