#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick checker for crawled GDELT data.

- Scans `<base-dir>/filtered/*.parquet` (preferred). If empty, falls back to `raw/*.parquet`.
- Prints:
  * Total rows, symbols present
  * Date range (min/max) from `seendate`/`date`
  * Per-symbol counts
  * Per-day counts (top/bottom few)
  * Error/OK day manifests (if `_manifest/*.json` exists)
  * Samples for the FIRST and LAST date (configurable number)

Usage:
  python -m fin_rl.data.crypto.gdelt_check \
    --base-dir ./work/data/gdelt_align_hardfix \
    --show-samples 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


# --- replace _load_parquets, _daily_counts, main blocks in your checker ---

def _load_parquets(folder: Path, prefer_filtered: bool = True) -> Tuple[pd.DataFrame, str]:
    flt_dir = folder / "filtered"
    raw_dir = folder / "raw"

    def _safe_read_all(dir_path: Path) -> Tuple[pd.DataFrame, int]:
        dfs, used = [], 0
        for p in sorted(dir_path.glob("*.parquet")):
            try:
                d = pd.read_parquet(p)
                # skip truly empty frames (no rows OR no columns)
                if d is None or d.empty or len(d.columns) == 0:
                    continue
                dfs.append(d)
                used += 1
            except Exception as e:
                print(f"[WARN] cannot read {p}: {e}")
        if not dfs:
            return pd.DataFrame(), used
        df = pd.concat(dfs, ignore_index=True)

        # normalize seendate
        if "seendate" in df.columns:
            df["seendate"] = pd.to_datetime(df["seendate"], errors="coerce", utc=True)

        # robust date normalization:
        # 1) if no `date` or it's all NaN -> derive from seendate
        # 2) else coerce to datetime then take .dt.date
        if "date" not in df.columns or df["date"].isna().all():
            if "seendate" in df.columns:
                df["date"] = df["seendate"].dt.date
            else:
                df["date"] = pd.NaT
        else:
            # coerce everything to datetime then cast to date
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

        # standardize symbol
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str)
        else:
            df["symbol"] = "UNKNOWN"

        return df, used

    if prefer_filtered and flt_dir.exists():
        df, nfiles = _safe_read_all(flt_dir)
        if not df.empty:
            print(f"[INFO] loaded {nfiles} parquet(s) from {flt_dir}")
            return df, "filtered"

    if raw_dir.exists():
        df, nfiles = _safe_read_all(raw_dir)
        if not df.empty:
            print(f"[INFO] loaded {nfiles} parquet(s) from {raw_dir}")
            return df, "raw"

    return pd.DataFrame(), "none"


def _daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date","symbol","count"])
    # date is already normalized in _load_parquets
    d = (df.dropna(subset=["date","symbol"])
           .groupby(["date","symbol"])
           .size()
           .reset_index(name="count")
           .sort_values(["date","symbol"]))
    return d



def _print_samples_for_date(df: pd.DataFrame, target_date, n: int = 5):
    sub = df.copy()
    # derive date
    if "date" in sub.columns:
        sub = sub[sub["date"] == target_date]
    elif "seendate" in sub.columns:
        sub = sub[sub["seendate"].dt.date == target_date]
    else:
        print("[INFO] No date/seendate column to slice samples.")
        return
    if sub.empty:
        print(f"[SAMPLES] {target_date}: (no rows)")
        return

    cols = [c for c in ["seendate", "symbol", "domain", "title", "url"] if c in sub.columns]
    sub = sub.sort_values(by=[c for c in ["seendate"] if c in sub.columns])
    print(f"[SAMPLES] {target_date} — showing up to {n} rows:")
    for i, (_, row) in enumerate(sub.head(n).iterrows(), 1):
        vals = []
        for c in cols:
            v = row.get(c)
            if isinstance(v, str) and c == "title":
                v = (v[:160] + "…") if len(v) > 160 else v
            vals.append(f"{c}={v!r}")
        print("  - " + " | ".join(vals))


def _scan_manifests(base_dir: Path):
    mf_dir = base_dir / "_manifest"
    if not mf_dir.exists():
        print("[MANIFEST] No _manifest folder.")
        return
    oks, errs = 0, 0
    err_days = []
    for p in sorted(mf_dir.glob("*.json")):
        try:
            js = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        st = js.get("status")
        if st == "ok":
            oks += 1
        elif st == "error":
            errs += 1
            err_days.append((p.name, js.get("date"), js.get("symbol"), js.get("error")))
    print(f"[MANIFEST] OK={oks} | ERROR={errs}")
    if err_days:
        print("  Some errors (up to 10 shown):")
        for name, d, sym, msg in err_days[:10]:
            print(f"   - {name}: date={d} symbol={sym} error={msg}")


def main():
    ap = argparse.ArgumentParser(description="Check crawled GDELT data and print quick stats + samples.")
    ap.add_argument("--base-dir", type=str, required=True, help="Base output dir (contains filtered/, raw/, _manifest/)")
    ap.add_argument("--show-samples", type=int, default=3, help="Number of samples to show for first/last date")
    ap.add_argument("--limit-rows", type=int, default=None, help="Optional cap rows to speed up (debug)")
    args = ap.parse_args()

    base = Path(args.base_dir)
    if not base.exists():
        print(f"[ERR] base-dir not found: {base}")
        sys.exit(1)

    df, source = _load_parquets(base, prefer_filtered=True)
    if df.empty:
        print(f"[INFO] No parquet data found under {base}/filtered or {base}/raw")
        _scan_manifests(base)
        sys.exit(0)

    if args.limit_rows and len(df) > args.limit_rows:
        df = df.head(args.limit_rows).copy()

    total = len(df)
    syms = sorted(df["symbol"].dropna().astype(str).unique().tolist())

    dmin_seendate = df["seendate"].dropna().min() if "seendate" in df.columns else None
    dmax_seendate = df["seendate"].dropna().max() if "seendate" in df.columns else None
    dmin_date = df["date"].dropna().min() if "date" in df.columns else None
    dmax_date = df["date"].dropna().max() if "date" in df.columns else None

    print(f"[INFO] Source folder: {source}")
    print(f"[INFO] Rows: {total}")
    print(f"[INFO] Symbols present: {syms}")
    print(f"[INFO] Date range(seendate): {dmin_seendate} → {dmax_seendate}")
    print(f"[INFO] Date range(date):     {dmin_date} → {dmax_date}")

    # Per-symbol counts
    per_sym = (df.groupby("symbol").size().reset_index(name="count").sort_values("count", ascending=False))
    print("\n[STATS] Per-symbol counts:")
    for _, row in per_sym.iterrows():
        print(f"  - {row['symbol']}: {int(row['count'])}")

    # Per-day counts
    daily = _daily_counts(df)
    if not daily.empty:
        print("\n[STATS] Daily counts (head 5):")
        print(daily.head(5).to_string(index=False))
        print("\n[STATS] Daily counts (tail 5):")
        print(daily.tail(5).to_string(index=False))
    else:
        print("\n[STATS] Daily counts: (empty)")

    # Samples for first/last date (use normalized 'date')
    def _print_samples_for_date(df_in: pd.DataFrame, target_date, n: int):
        sub = df_in[df_in["date"] == target_date].sort_values(by=[c for c in ["seendate"] if c in df_in.columns])
        if sub.empty:
            print(f"[SAMPLES] {target_date}: (no rows)")
            return
        cols = [c for c in ["seendate", "symbol", "domain", "title", "url"] if c in sub.columns]
        print(f"[SAMPLES] {target_date} — showing up to {n} rows:")
        for _, row in sub.head(n).iterrows():
            parts = []
            for c in cols:
                v = row.get(c)
                if isinstance(v, str) and c == "title":
                    v = (v[:160] + "…") if len(v) > 160 else v
                parts.append(f"{c}={v!r}")
            print("  - " + " | ".join(parts))

    if dmin_date is not None:
        print("\n----- SAMPLES: FIRST DAY -----")
        _print_samples_for_date(df, dmin_date, n=args.show_samples)
    if dmax_date is not None and dmax_date != dmin_date:
        print("\n----- SAMPLES: LAST DAY ------")
        _print_samples_for_date(df, dmax_date, n=args.show_samples)

    print("\n----- MANIFEST SUMMARY -------")
    _scan_manifests(base)


if __name__ == "__main__":
    main()
