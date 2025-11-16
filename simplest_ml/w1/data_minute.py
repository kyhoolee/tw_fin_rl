"""
Minute data ingestion utilities (Binance klines).

Goals
- Fetch 1m candles robustly (chunked, retry/backoff) for a date range.
- Save Parquet shards by month: data/minute/binance_{symbol}_1m_YYYYMM.parquet
- Keep UTC timestamps, consistent columns, and no duplicates.

Usage (from notebook or CLI):

    from data_minute import download_minute_range, download_minute_for_symbols
    download_minute_range("BTCUSDT", start="2023-10-30", end="2025-10-30")

CLI (from repo root):

    python simplest_ml/w1/data_minute.py BTCUSDT 2023-10-30 2025-10-30 \
        --out-dir simplest_ml/w1/data/minute --force 0 --sleep 0.2

Notes
- Requires: requests, pandas, pyarrow (or fastparquet) to write Parquet.
- Default out_dir is "data/minute" relative to CWD; override as needed.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import requests

BINANCE = "https://api.binance.com/api/v3/klines"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_dt(x: str | datetime) -> datetime:
    if isinstance(x, datetime):
        return x.astimezone(timezone.utc)
    # Allow YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS
    try:
        dt = datetime.fromisoformat(x)
    except Exception:
        dt = datetime.strptime(x, "%Y-%m-%d")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _request_klines(params: dict, retries: int = 5, backoff: float = 0.5) -> List[list]:
    last_exc = None
    for i in range(retries):
        try:
            r = requests.get(BINANCE, params=params, timeout=30)
            if r.status_code == 429:  # rate limit
                time.sleep(backoff * (2 ** i))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            time.sleep(backoff * (2 ** i))
    raise RuntimeError(f"Failed to fetch klines after retries: {last_exc}")


def fetch_klines_chunked(
    symbol: str,
    interval: str = "1m",
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    limit: int = 1000,
    sleep: float = 0.2,
) -> pd.DataFrame:
    """Fetch klines in chunks between [start, end).

    - start/end: datetime (UTC) or ISO strings; defaults to last ~7 days if not provided.
    - Returns a DataFrame with standard Binance kline columns, UTC timestamps, sorted by close_time.
    """
    if start is None:
        start = datetime.now(timezone.utc) - timedelta(days=7)
    if end is None:
        end = datetime.now(timezone.utc)
    start = _parse_dt(start) if not isinstance(start, datetime) else start
    end = _parse_dt(end) if not isinstance(end, datetime) else end

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    params_base = {"symbol": symbol.upper(), "interval": interval, "limit": limit}

    out: List[list] = []
    cur = start_ms
    while True:
        if cur >= end_ms:
            break
        params = {**params_base, "startTime": cur, "endTime": end_ms}
        batch = _request_klines(params)
        if not batch:
            break
        out.extend(batch)
        # Next window start: last candle open time + interval
        next_open = batch[-1][0] + 1
        if next_open <= cur:  # safety against stuck loops
            next_open = cur + 1
        cur = next_open
        time.sleep(sleep)

    if not out:
        return pd.DataFrame(columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_vol","num_trades","taker_base","taker_quote","ignore"
        ])

    cols = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","num_trades","taker_base","taker_quote","ignore"
    ]
    df = pd.DataFrame(out, columns=cols)
    for c in ["open","high","low","close","volume","quote_vol","taker_base","taker_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["close_time"]).set_index("close_time").sort_index()
    return df


def _month_range(start: datetime, end: datetime) -> List[Tuple[datetime, datetime, str]]:
    """Yield (start_dt, end_dt, yyyymm) for each month fully/partially within [start, end]."""
    s = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    e = datetime(end.year, end.month, 1, tzinfo=timezone.utc)
    out = []
    cur = s
    while cur <= e:
        # month end = first day next month
        if cur.month == 12:
            nxt = datetime(cur.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            nxt = datetime(cur.year, cur.month + 1, 1, tzinfo=timezone.utc)
        seg_start = max(cur, start)
        seg_end = min(nxt, end)
        out.append((seg_start, seg_end, f"{cur.year:04d}{cur.month:02d}"))
        cur = nxt
    return out


def download_minute_range(
    symbol: str,
    start: str | datetime,
    end: str | datetime,
    *,
    out_dir: str = "data/minute",
    interval: str = "1m",
    overwrite: bool = False,
    sleep: float = 0.2,
    verbose: bool = True,
) -> List[str]:
    """Download 1m klines for [start, end) and save monthly Parquet shards.

    Returns list of file paths written/skipped.
    """
    start_dt = _parse_dt(start)
    end_dt = _parse_dt(end)
    _ensure_dir(out_dir)
    written: List[str] = []
    for seg_start, seg_end, yyyymm in _month_range(start_dt, end_dt):
        fn = os.path.join(out_dir, f"binance_{symbol.lower()}_{interval}_{yyyymm}.parquet")
        if os.path.exists(fn) and not overwrite:
            if verbose:
                print(f"[skip] {symbol} {interval} {yyyymm}: exists -> {fn}")
            written.append(fn)
            continue
        if verbose:
            print(f"[fetch] {symbol} {interval} {yyyymm}: {seg_start.date()} → {seg_end.date()}")
        df = fetch_klines_chunked(symbol, interval=interval, start=seg_start, end=seg_end, sleep=sleep)
        # Guard: only save if we have data in this month
        if not df.empty:
            df.to_parquet(fn)
            if verbose:
                print(f"  -> rows={len(df):,} saved: {fn}")
            written.append(fn)
        else:
            # touch an empty file to mark processed month (optional)
            # pd.DataFrame().to_parquet(fn)  # uncomment if you prefer creating empties
            if verbose:
                print(f"  -> no data for {yyyymm} (empty month)")
    return written


def download_minute_for_symbols(
    symbols: Iterable[str],
    start: str | datetime,
    end: str | datetime,
    *,
    out_dir: str = "data/minute",
    interval: str = "1m",
    overwrite: bool = False,
    sleep: float = 0.2,
    verbose: bool = True,
) -> None:
    for sym in symbols:
        if verbose:
            print(f"[minute] {sym}: downloading {interval} from {start} to {end} ...")
        files = download_minute_range(
            sym,
            start,
            end,
            out_dir=out_dir,
            interval=interval,
            overwrite=overwrite,
            sleep=sleep,
            verbose=verbose,
        )
        if verbose:
            print(f"  -> {len(files)} shard(s) saved/kept in {out_dir}")


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download Binance 1m klines and save monthly Parquet shards")
    p.add_argument("symbol", help="e.g., BTCUSDT")
    p.add_argument("start", help="ISO date/time, e.g., 2023-10-30")
    p.add_argument("end", help="ISO date/time, e.g., 2025-10-30")
    p.add_argument("--out-dir", default="data/minute", help="Output directory for Parquet files")
    p.add_argument("--interval", default="1m", help="Kline interval (default 1m)")
    p.add_argument("--overwrite", type=int, default=0, help="1 to overwrite existing shard files")
    p.add_argument("--sleep", type=float, default=0.2, help="Sleep between requests to avoid 429")
    p.add_argument("--verbose", type=int, default=1, help="1 to print progress logs")
    return p


if __name__ == "__main__":
    args = _build_cli().parse_args()
    download_minute_range(
        args.symbol,
        start=args.start,
        end=args.end,
        out_dir=args.out_dir,
        interval=args.interval,
        overwrite=bool(args.overwrite),
        sleep=args.sleep,
        verbose=bool(args.verbose),
    )
