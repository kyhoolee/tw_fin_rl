#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe-fetch Binance OHLCV (Spot) with append + resume.
- Interval: 1m..1d..1M (api /api/v3/klines)
- Writes each fetched batch directly to CSV (append mode)
- Saves a sidecar .state (JSON) for resume without re-reading big CSV
"""

import argparse, json, os, time
from datetime import datetime, timezone, timedelta
from typing import Optional
import pandas as pd
import requests

BASE_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000
INTERVALS_ALLOWED = {
    "1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h",
    "1d","3d","1w","1M"
}

# rough millis per bar (for estimation/progress only)
EST_MS = {
    "1m":60_000, "3m":180_000, "5m":300_000, "15m":900_000, "30m":1_800_000,
    "1h":3_600_000, "2h":7_200_000, "4h":14_400_000, "6h":21_600_000, "8h":28_800_000, "12h":43_200_000,
    "1d":86_400_000, "3d":259_200_000, "1w":604_800_000, "1M":30*86_400_000
}

COLS = [
    "open_time","open","high","low","close","volume","close_time",
    "quote_asset_volume","number_of_trades",
    "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
]

def to_ms(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

def ms_to_iso(ms: int) -> str:
    return datetime.utcfromtimestamp(ms/1000).replace(tzinfo=timezone.utc).isoformat()

def estimate_bars(start_ms: int, end_ms: int, interval: str) -> int:
    return max(1, (end_ms - start_ms) // EST_MS[interval])

def fetch_once(symbol: str, interval: str, start_ms: int, end_ms: int,
               session: requests.Session, max_retries: int = 3, timeout: int = 15):
    params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": MAX_LIMIT}
    for a in range(max_retries):
        try:
            r = session.get(BASE_URL, params=params, timeout=timeout)
            if r.status_code == 429:
                time.sleep(1.0 + a)  # backoff on rate limit
                continue
            r.raise_for_status()
            rows = r.json()
            return rows
        except Exception as e:
            if a == max_retries - 1:
                raise
            time.sleep(0.5 * (a + 1))
    return []

def write_chunk_csv(fp: str, rows):
    df = pd.DataFrame(rows, columns=COLS)
    # cast types
    num_cols = ["open","high","low","close","volume","quote_asset_volume",
                "taker_buy_base_asset_volume","taker_buy_quote_asset_volume"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    # append mode; header only if new file
    header = not os.path.exists(fp) or os.path.getsize(fp) == 0
    df.to_csv(fp, mode="a", header=header, index=False)

def run(symbols, start_date, end_date, out_dir, interval="1d", pause=0.2, overwrite=False, resume=True):
    assert interval in INTERVALS_ALLOWED, f"interval must be one of {sorted(INTERVALS_ALLOWED)}"
    start_ms_raw = to_ms(start_date)
    end_ms = to_ms(end_date) + EST_MS["1d"]  # include end date fully for daily; ok for others as well

    os.makedirs(out_dir, exist_ok=True)
    out_subdir = os.path.join(out_dir, "spot", interval)
    os.makedirs(out_subdir, exist_ok=True)

    with requests.Session() as sess:
        for sym in symbols:
            base_name = f"{sym}_{interval}_{start_date}_{end_date}"
            csv_path = os.path.join(out_subdir, base_name + ".csv")
            state_path = csv_path + ".state"

            # determine start_ms (resume or fresh)
            if overwrite:
                for p in (csv_path, state_path):
                    if os.path.exists(p): os.remove(p)

            start_ms = start_ms_raw
            total_rows = 0
            if resume and os.path.exists(state_path):
                try:
                    st = json.load(open(state_path, "r"))
                    last_ms = int(st.get("last_open_time_ms", start_ms_raw))
                    start_ms = max(start_ms_raw, last_ms + 1)
                    total_rows = int(st.get("total_rows", 0))
                    print(f"[RESUME] {sym}: resume from {ms_to_iso(start_ms)}; rows_already={total_rows}")
                except Exception:
                    pass

            est = estimate_bars(start_ms, end_ms, interval)
            est_reqs = (est // MAX_LIMIT) + 1
            print(f"[START] {sym} interval={interval}  range={ms_to_iso(start_ms)} → {ms_to_iso(end_ms)}  "
                  f"~est_bars={est:,}  ~est_reqs={est_reqs:,}")
            cur = start_ms
            req_count = 0
            t0 = time.time()

            while cur < end_ms:
                rows = fetch_once(sym, interval, cur, end_ms, session=sess)
                if not rows:
                    break
                write_chunk_csv(csv_path, rows)
                last_open = rows[-1][0]
                cur = last_open + 1
                req_count += 1
                total_rows += len(rows)

                # update state (checkpoint)
                with open(state_path, "w") as f:
                    json.dump({"symbol": sym,
                               "interval": interval,
                               "last_open_time_ms": last_open,
                               "total_rows": total_rows,
                               "csv_path": csv_path}, f)

                # progress
                last_dt = pd.to_datetime(last_open, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M UTC")
                done_frac = min(1.0, (cur - start_ms) / max(1, end_ms - start_ms))
                eta = (time.time() - t0) * (1 - done_frac) / max(1e-6, done_frac) if done_frac > 0 else 0
                print(f"[PROGRESS] {sym} req={req_count:,}/{est_reqs:,}  rows+={len(rows)}  total={total_rows:,}  "
                      f"up_to={last_dt}  {done_frac*100:5.1f}%  ETA~{eta:,.1f}s")
                if cur >= end_ms:
                    break
                time.sleep(pause)

            print(f"[DONE] {sym} saved={total_rows:,} rows  file={csv_path}")
            # optional: cleanup state if fully done
            # os.remove(state_path)
            # (giữ lại state để biết đã lấy được bao nhiêu; xoá nếu bạn muốn)
    print("[ALL DONE]")

def parse_args():
    ap = argparse.ArgumentParser(description="Safe-fetch Binance OHLCV (append + resume)")
    ap.add_argument("--symbols", required=True, help="Comma list, e.g. BTCUSDT,ETHUSDT,SOLUSDT")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--interval", default="1d", help=f"Binance interval, one of {sorted(INTERVALS_ALLOWED)}")
    ap.add_argument("--out-dir", default="./data/binance")
    ap.add_argument("--pause", type=float, default=0.2)
    ap.add_argument("--overwrite", action="store_true", help="Delete existing CSV/state and start fresh")
    ap.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume even if .state exists")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    run(symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        out_dir=args.out_dir,
        interval=args.interval,
        pause=args.pause,
        overwrite=args.overwrite,
        resume=args.resume)
