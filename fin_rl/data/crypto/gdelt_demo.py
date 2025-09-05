#!/usr/bin/env python3
import argparse
import datetime as dt
import re
import time
import urllib.parse
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import requests
from tqdm import tqdm


GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# ===== Heuristics: domain whitelist & keyword blacklists =====
CRYPTO_WHITELIST = {
    "coindesk.com", "theblock.co", "decrypt.co", "bitcoinmagazine.com",
    "cointelegraph.com", "beincrypto.com", "cryptoslate.com",
    "ambcrypto.com", "u.today", "blockworks.co", "cryptobriefing.com",
    "finbold.com", "newsbtc.com", "coingeek.com", "coincu.com",
    "cryptonews.com", "cryptopotato.com", "cryptonewsz.com",
}

BLACKLIST_PATTERNS = {
    "SOL": [
        r"\bCosta del Sol\b", r"\bdel Sol\b", r"\bSol de\b",
        r"\bsolar\b", r"\bsolstice\b"
    ],
    "ETH": [
        r"\bETH Zurich\b", r"\bEthiopia(n)?\b", r"\bethanol\b",
        r"\bethane\b", r"\bethylene\b"
    ],
    "BNB": [
        r"\bAirbnb\b", r"\bB&B\b",
        r"\bbed(?:\s|-)and(?:\s|-)breakfast\b"
    ],
    # BTC: hiếm khi cần blacklist
}

CRYPTO_CONTEXT = [
    r"\bcrypto(currenc(y|ies))?\b", r"\bblockchain\b", r"\btoken(s)?\b", r"\bweb3\b",
    r"\bdefi\b", r"\bstaking\b", r"\bon-?chain\b", r"\bexchange\b", r"\bDEX\b", r"\bCEX\b",
    r"\bETF\b", r"\bspot ETF\b", r"\bmin(ing|er)\b", r"\bL2\b", r"\bsmart contract\b",
    r"\bgas fees?\b", r"\bstablecoin\b",
]

SYMBOL_ANCHORS = {
    "BTC": [r"\bBitcoin\b", r"\bBTC token\b", r"\bBTC price\b", r"\bBTC\b"],
    "ETH": [r"\bEthereum\b", r"\bETH token\b", r"\bETH price\b", r"\bETH\b"],
    "BNB": [r"\bBNB Chain\b", r"\bBNB token\b", r"\bBNB price\b", r"\bBinance\b", r"\bBNB\b"],
    "SOL": [r"\bSolana\b", r"\bSOL token\b", r"\bSOL price\b", r"\bSOL\b"],
}

def _ymd_hms(dtobj: dt.datetime) -> str:
    return dtobj.strftime("%Y%m%d%H%M%S")

def _match_any(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)

def build_query(sym: str, langs: List[str], mode: str) -> str:
    """
    Balanced = full name + ticker khi có ngữ cảnh 'token/price'
    Strict   = chỉ full name (và BNB Chain/Binance)
    Loose    = thêm cả ticker trần
    """
    s = sym.upper()
    if s == "BTC":
        if mode == "strict":
            base = '("Bitcoin")'
        elif mode == "balanced":
            base = '("Bitcoin" OR "BTC token" OR "BTC price")'
        else:
            base = '("Bitcoin" OR "BTC token" OR "BTC price" OR BTC)'
    elif s == "ETH":
        if mode == "strict":
            base = '("Ethereum")'
        elif mode == "balanced":
            base = '("Ethereum" OR "ETH token" OR "ETH price")'
        else:
            base = '("Ethereum" OR "ETH token" OR "ETH price" OR ETH)'
    elif s == "BNB":
        if mode == "strict":
            base = '("BNB Chain" OR "Binance")'
        elif mode == "balanced":
            base = '("BNB Chain" OR "BNB token" OR "BNB price" OR "Binance")'
        else:
            base = '("BNB Chain" OR "BNB token" OR "BNB price" OR "Binance" OR BNB)'
    elif s == "SOL":
        if mode == "strict":
            base = '("Solana")'
        elif mode == "balanced":
            base = '("Solana" OR "SOL token" OR "SOL price")'
        else:
            base = '("Solana" OR "SOL token" OR "SOL price" OR SOL)'
    else:
        base = s

    lang_clause = " ".join([f"sourcelang:{l}" for l in langs]) if langs else ""
    return f"{base} {lang_clause}".strip()

def fetch_window(query: str,
                 start: dt.datetime,
                 end: dt.datetime,
                 maxrecords: int = 250,
                 sort: str = "DateDesc") -> List[Dict[str, Any]]:
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": maxrecords,
        "sort": sort,
        "format": "json",
        "startdatetime": _ymd_hms(start),
        "enddatetime": _ymd_hms(end),
    }
    url = f"{GDELT_DOC_API}?{urllib.parse.urlencode(params)}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    try:
        js = r.json()
    except Exception:
        return []
    return js.get("articles", [])


import sys
from time import time as _time

class _Progress:
    """Progress that works well in tmux & non-TTY.

    - kind='tqdm' && isatty && not no_progress  -> real tqdm bar (single line)
    - otherwise                                   -> periodic log lines with ETA
    """
    def __init__(self, total:int, desc:str, kind:str="tqdm",
                 no_progress:bool=False,
                 every:int=10,
                 min_interval:float=2.0):
        self.total = total
        self.desc = desc
        self.kind = kind
        self.no = no_progress
        self.every = max(1, int(every))
        self.min_interval = float(min_interval)
        self.count = 0
        self.start_ts = _time()
        self._last_print = 0.0

        self.use_tqdm = False
        if not self.no and self.kind == "tqdm" and sys.stdout.isatty():
            try:
                self._pbar = tqdm(
                    total=total,
                    desc=desc,
                    ncols=100,
                    dynamic_ncols=True,
                    leave=False,
                    mininterval=0.5,
                    position=0,
                    file=sys.stdout,
                    ascii=True  # tránh kí tự unicode trong tmux
                )
                self.use_tqdm = True
            except Exception:
                self.use_tqdm = False
                self._pbar = None
        else:
            self._pbar = None

    def _fmt_hms(self, secs: float) -> str:
        if secs < 0: secs = 0
        m, s = divmod(int(secs), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def update(self, n:int=1):
        if self.no:
            return
        self.count += n
        if self.use_tqdm:
            self._pbar.update(n)
            return

        # periodic log line
        now = _time()
        need_print = (self.count % self.every == 0) or (now - self._last_print >= self.min_interval) or (self.count == self.total)
        if not need_print:
            return
        elapsed = now - self.start_ts
        rate = self.count / elapsed if elapsed > 0 else 0.0
        remain = (self.total - self.count) / rate if rate > 0 else float("inf")
        pct = (self.count / self.total * 100.0) if self.total > 0 else 0.0
        msg = (f"{self.desc} {self.count}/{self.total} ({pct:5.1f}%)  "
               f"elapsed={self._fmt_hms(elapsed)}  eta={self._fmt_hms(remain)}  "
               f"win/s={rate:.2f}")
        # ensure single line in tmux logs (no carriage return games)
        print(msg, flush=True)
        self._last_print = now

    def close(self):
        if self.use_tqdm and self._pbar is not None:
            self._pbar.close()


def gdelt_collect_streaming(query: str,
                            sym: str,
                            start: dt.datetime,
                            end: dt.datetime,
                            out_raw_dir: Path,
                            step_hours: int = 12,
                            sleep_sec: float = 0.15,
                            maxrecords: int = 250,
                            resume: bool = True,
                            # NEW:
                            no_progress: bool = False,
                            progress_kind: str = "tqdm",
                            progress_every: int = 10,
                            progress_min_interval: float = 2.0) -> List[Path]:
    out_raw_dir.mkdir(parents=True, exist_ok=True)
    files = []
    cursor = start
    step = dt.timedelta(hours=step_hours)

    # tính tổng window
    total_windows = 0
    tmp = start
    while tmp < end:
        tmp = min(tmp + step, end)
        total_windows += 1

    p = _Progress(
        total=total_windows,
        desc=f"[{sym}] crawl",
        kind=progress_kind,
        no_progress=no_progress,
        every=progress_every,
        min_interval=progress_min_interval,
    )

    try:
        while cursor < end:
            w_start = cursor
            w_end = min(cursor + step, end)
            fname = f"{sym}_{w_start.strftime('%Y%m%d%H%M%S')}_{w_end.strftime('%Y%m%d%H%M%S')}.parquet"
            fpath = out_raw_dir / fname

            if resume and fpath.exists():
                files.append(fpath)
                p.update(1)
                cursor = w_end
                continue

            arts = fetch_window(query, w_start, w_end, maxrecords=maxrecords)
            if arts:
                rows = [{
                    "title": a.get("title"),
                    "url": a.get("url"),
                    "domain": a.get("domain"),
                    "language": a.get("language"),
                    "seendate": a.get("seendate"),
                    "socialimage": a.get("socialimage"),
                    "sourcecountry": a.get("sourcecountry"),
                    "isduplicate": a.get("isduplicate"),
                    "symbol": sym,
                } for a in arts]
                df = pd.DataFrame(rows).drop_duplicates(subset=["url"])
                df["seendate"] = pd.to_datetime(df["seendate"], errors="coerce", utc=True)
                df["date"] = df["seendate"].dt.date
                df.to_parquet(fpath, index=False)
                files.append(fpath)
            else:
                # ghi file rỗng để đánh dấu đã xử lý window này
                pd.DataFrame(columns=["title","url","domain","language","seendate","socialimage","sourcecountry","isduplicate","symbol","date"]).to_parquet(fpath, index=False)
                files.append(fpath)

            p.update(1)
            cursor = w_end
            time.sleep(sleep_sec)
    finally:
        p.close()

    return files



def load_concat(paths: List[Path]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        try:
            d = pd.read_parquet(p)
            dfs.append(d)
        except Exception as e:
            print(f"[WARN] cannot read {p.name}: {e}")
    if not dfs:
        return pd.DataFrame(columns=["title","url","domain","language","seendate","socialimage","sourcecountry","isduplicate","symbol","date"])
    df = pd.concat(dfs, ignore_index=True)
    # drop final dups
    if "url" in df.columns:
        df = df.drop_duplicates(subset=["url","symbol"], keep="first")
    return df

def filter_rows(df: pd.DataFrame, mode: str = "balanced") -> pd.DataFrame:
    if df.empty:
        return df.assign(keep=False, keep_reason="no_data")
    df = df.copy()
    df["title_lc"] = df["title"].fillna("").astype(str)
    df["domain_lc"] = df.get("domain", "").fillna("").astype(str)
    df["keep"] = False
    df["keep_reason"] = "no_match"

    for i, row in df.iterrows():
        title = row["title_lc"]
        domain = row["domain_lc"]
        sym = str(row.get("symbol", "")).upper()

        # blacklist theo symbol
        blist = BLACKLIST_PATTERNS.get(sym, [])
        if blist and _match_any(blist, title):
            df.at[i, "keep"] = False
            df.at[i, "keep_reason"] = "blacklist_symbol_noise"
            continue

        anchors = SYMBOL_ANCHORS.get(sym, [])

        if mode == "strict":
            # whitelist
            if domain in CRYPTO_WHITELIST and len(title) > 3:
                df.at[i, "keep"] = True; df.at[i, "keep_reason"] = "whitelist_domain"; continue
            # full-name preferred
            full_only = [p for p in anchors if (" " in p) or any(k in p for k in ["Solana","Ethereum","Bitcoin","BNB Chain","Binance"])]
            if full_only and _match_any(full_only, title):
                df.at[i, "keep"] = True; df.at[i, "keep_reason"] = "symbol_full_anchor"; continue
            if _match_any(CRYPTO_CONTEXT, title) and anchors and _match_any(anchors, title):
                df.at[i, "keep"] = True; df.at[i, "keep_reason"] = "context_plus_anchor"; continue

        elif mode == "balanced":
            if domain in CRYPTO_WHITELIST and len(title) > 3:
                df.at[i, "keep"] = True; df.at[i, "keep_reason"] = "whitelist_domain"; continue
            if anchors and _match_any(anchors, title):
                df.at[i, "keep"] = True; df.at[i, "keep_reason"] = "symbol_anchor"; continue
            if _match_any(CRYPTO_CONTEXT, title) and anchors and _match_any(anchors, title):
                df.at[i, "keep"] = True; df.at[i, "keep_reason"] = "context_plus_anchor"; continue

        else:  # loose
            if domain in CRYPTO_WHITELIST and len(title) > 3:
                df.at[i, "keep"] = True; df.at[i, "keep_reason"] = "whitelist_domain"; continue
            if (anchors and _match_any(anchors, title)) or _match_any(CRYPTO_CONTEXT, title):
                df.at[i, "keep"] = True; df.at[i, "keep_reason"] = "anchor_or_context"; continue

        df.at[i, "keep"] = False
        df.at[i, "keep_reason"] = "no_match"

    return df

def daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date","symbol","count"])
    d = (df.dropna(subset=["date","symbol"])
           .groupby(["date","symbol"])
           .size()
           .reset_index(name="count")
           .sort_values(["date","symbol"]))
    return d

def main():
    ap = argparse.ArgumentParser(description="GDELT DOC aligned to OHLCV (streaming + progress)")
    ap.add_argument("--symbols", type=str, default="BTC,ETH,BNB,SOL", help="Comma-separated symbols")
    ap.add_argument("--start", type=str, help="Start date (YYYY-MM-DD) UTC")
    ap.add_argument("--end", type=str, help="End date (YYYY-MM-DD, exclusive) UTC")
    ap.add_argument("--step-hours", type=int, default=12, help="Window step (hours)")
    ap.add_argument("--langs", type=str, default="english", help="Comma-separated sourcelang filters")
    ap.add_argument("--out", type=str, default="./work/data/gdelt_align", help="Output dir")
    ap.add_argument("--mode", type=str, default="balanced", choices=["strict","balanced","loose"], help="Filter mode")
    ap.add_argument("--sleep-sec", type=float, default=0.15, help="Sleep between requests")
    ap.add_argument("--maxrecords", type=int, default=250, help="Max records per GDELT window")
    ap.add_argument("--no-resume", action="store_true", help="Disable resume (re-fetch all windows)")
    # thêm flag hardcoded
    ap.add_argument("--hardcoded", action="store_true", help="Run fixed ranges aligned to OHLCV (BTC/ETH/BNB: 2019→2025, SOL: 2020→2025)")
    # NEW: progress options
    ap.add_argument("--no-progress", action="store_true", help="Disable progress display")
    ap.add_argument("--progress-kind", type=str, default="tqdm", choices=["tqdm","log"],
                    help="Progress display style: tqdm (bar) or log (periodic lines)")
    ap.add_argument("--progress-every", type=int, default=10,
                    help="In log mode: print a line every N windows")
    ap.add_argument("--progress-min-interval", type=float, default=2.0,
                    help="In log mode: minimal seconds between progress prints")
    args = ap.parse_args()

    if args.hardcoded:
        run_hardcoded_crypto_news(
            out_dir=args.out,
            step_hours=args.step_hours,
            mode=args.mode,
            langs=args.langs,
            sleep_sec=args.sleep_sec,
            maxrecords=args.maxrecords,
            resume=(not args.no_resume),
            # progress
            no_progress=args.no_progress,
            progress_kind=args.progress_kind,
            progress_every=args.progress_every,
            progress_min_interval=args.progress_min_interval,
        )
        return


    # --------- normal custom mode ----------
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    if not args.start or not args.end:
        raise ValueError("Must provide --start and --end unless using --hardcoded")

    start = dt.datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    end   = dt.datetime.strptime(args.end,   "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)

    # (phần crawl cũ giữ nguyên ...)

    outdir = Path(args.out)
    raw_dir = outdir / "raw"
    flt_dir = outdir / "filtered"
    agg_dir = outdir / "agg"
    raw_dir.mkdir(parents=True, exist_ok=True)
    flt_dir.mkdir(parents=True, exist_ok=True)
    agg_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Crawl symbols={symbols} | {start} → {end} (UTC) | mode={args.mode}")
    merged_raw_frames, merged_flt_frames = [], []

    for sym in symbols:
        query = build_query(sym, langs, args.mode)
        print(f"[INFO] Query[{sym}]: {query}")

        # 1) streaming crawl -> per-window parquet
        sym_raw_dir = raw_dir / sym
        files = gdelt_collect_streaming(
            query=query,
            sym=sym,
            start=start,
            end=end,
            out_raw_dir=sym_raw_dir,
            step_hours=args.step_hours,
            sleep_sec=args.sleep_sec,
            maxrecords=args.maxrecords,
            resume=(not args.no_resume),
            # NEW
            no_progress=args.no_progress,
            progress_kind=args.progress_kind,
            progress_every=args.progress_every,
            progress_min_interval=args.progress_min_interval,
        )


        # 2) concat raw
        df_raw = load_concat(files)
        if not df_raw.empty:
            df_raw["symbol"] = sym
            df_rawp = flt_dir / f"gdelt_raw_{sym}_{start.date()}_{(end-dt.timedelta(days=1)).date()}.parquet"
            df_raw.to_parquet(df_rawp, index=False)

        # 3) filter -> save
        df_flt = filter_rows(df_raw, mode=args.mode)
        df_kept = df_flt[df_flt["keep"]].drop(columns=["title_lc","domain_lc"], errors="ignore")
        df_fltp = flt_dir / f"gdelt_filtered_{sym}_{start.date()}_{(end-dt.timedelta(days=1)).date()}.parquet"
        df_kept.to_parquet(df_fltp, index=False)

        # 4) daily counts align với OHLC daily
        daily = daily_counts(df_kept)
        daily_p = agg_dir / f"gdelt_daily_counts_{sym}_{start.date()}_{(end-dt.timedelta(days=1)).date()}.parquet"
        daily.to_parquet(daily_p, index=False)

        print(f"[OK] {sym}: windows={len(files)} | raw={len(df_raw)} | kept={len(df_kept)}  → {df_fltp.name}")

        merged_raw_frames.append(df_raw)
        merged_flt_frames.append(df_kept)

    # 5) merged outputs
    if merged_raw_frames:
        mraw = pd.concat(merged_raw_frames, ignore_index=True)
        mraw_fp = flt_dir / f"gdelt_raw_all_{start.date()}_{(end-dt.timedelta(days=1)).date()}.parquet"
        mraw.to_parquet(mraw_fp, index=False)
        print(f"[OK] merged raw: {len(mraw)} → {mraw_fp.name}")

    if merged_flt_frames:
        mflt = pd.concat(merged_flt_frames, ignore_index=True)
        mflt_fp = flt_dir / f"gdelt_filtered_all_{start.date()}_{(end-dt.timedelta(days=1)).date()}.parquet"
        mflt.to_parquet(mflt_fp, index=False)
        print(f"[OK] merged filtered: {len(mflt)} → {mflt_fp.name}")

        # merged daily counts
        mdaily = daily_counts(mflt)
        mdaily_fp = agg_dir / f"gdelt_daily_counts_all_{start.date()}_{(end-dt.timedelta(days=1)).date()}.parquet"
        mdaily.to_parquet(mdaily_fp, index=False)
        print(f"[OK] merged daily counts: {len(mdaily)} → {mdaily_fp.name}")
    else:
        print("[WARN] No filtered data across symbols.")

def run_hardcoded_crypto_news(
    out_dir: str = "./work/data/gdelt_align_hard",
    step_hours: int = 6,
    mode: str = "balanced",
    langs: str = "english",
    sleep_sec: float = 0.15,
    maxrecords: int = 250,
    resume: bool = True,
    # progress options (tmux-friendly)
    no_progress: bool = False,
    progress_kind: str = "tqdm",         # "tqdm" | "log"
    progress_every: int = 10,            # log mode: print every N windows
    progress_min_interval: float = 2.0,  # log mode: min seconds between prints
):
    """
    Crawl GDELT cho các khoảng thời gian align với OHLCV bạn đang có:
      - BTC/ETH/BNB: 2019-01-01 → 2025-09-05 (end exclusive => cover tới 2025-09-04)
      - SOL:         2020-08-11 → 2025-09-05
    Ghi per-window ngay, có progress thân thiện tmux, xuất daily-counts.
    """
    symbols_plan = [
        ("BTC", "2019-01-01", "2025-09-05"),
        ("ETH", "2019-01-01", "2025-09-05"),
        ("BNB", "2019-01-01", "2025-09-05"),
        ("SOL", "2020-08-11", "2025-09-05"),
    ]

    outdir = Path(out_dir)
    raw_dir = outdir / "raw"
    flt_dir = outdir / "filtered"
    agg_dir = outdir / "agg"
    raw_dir.mkdir(parents=True, exist_ok=True)
    flt_dir.mkdir(parents=True, exist_ok=True)
    agg_dir.mkdir(parents=True, exist_ok=True)

    langs_list = [s.strip() for s in langs.split(",") if s.strip()]

    merged_raw_frames, merged_flt_frames = [], []

    print(f"[INFO] run_hardcoded_crypto_news | out={outdir} | step_hours={step_hours} | mode={mode} | langs={langs}")
    for sym, start_s, end_s in symbols_plan:
        start = dt.datetime.strptime(start_s, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
        end   = dt.datetime.strptime(end_s,   "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)

        query = build_query(sym, langs_list, mode)
        print(f"[INFO] Query[{sym}]: {query} | {start} → {end} (UTC)")

        # 1) crawl per-window -> write immediately
        sym_raw_dir = raw_dir / sym
        files = gdelt_collect_streaming(
            query=query,
            sym=sym,
            start=start,
            end=end,
            out_raw_dir=sym_raw_dir,
            step_hours=step_hours,
            sleep_sec=sleep_sec,
            maxrecords=maxrecords,
            resume=resume,
            # progress (tmux-friendly)
            no_progress=no_progress,
            progress_kind=progress_kind,
            progress_every=progress_every,
            progress_min_interval=progress_min_interval,
        )

        # 2) concat raw
        df_raw = load_concat(files)
        if not df_raw.empty:
            df_raw["symbol"] = sym
            df_rawp = flt_dir / f"gdelt_raw_{sym}_{start.date()}_{(end-dt.timedelta(days=1)).date()}.parquet"
            df_raw.to_parquet(df_rawp, index=False)

        # 3) filter & save
        df_flt = filter_rows(df_raw, mode=mode)
        df_kept = df_flt[df_flt["keep"]].drop(columns=["title_lc","domain_lc"], errors="ignore")
        df_fltp = flt_dir / f"gdelt_filtered_{sym}_{start.date()}_{(end-dt.timedelta(days=1)).date()}.parquet"
        df_kept.to_parquet(df_fltp, index=False)

        # 4) daily counts
        daily = daily_counts(df_kept)
        daily_p = agg_dir / f"gdelt_daily_counts_{sym}_{start.date()}_{(end-dt.timedelta(days=1)).date()}.parquet"
        daily.to_parquet(daily_p, index=False)

        print(f"[OK] {sym}: windows={len(files)} | raw={len(df_raw)} | kept={len(df_kept)}  → {df_fltp.name}")

        merged_raw_frames.append(df_raw)
        merged_flt_frames.append(df_kept)

    # 5) merged outputs
    if merged_raw_frames:
        mraw = pd.concat(merged_raw_frames, ignore_index=True)
        mraw_fp = flt_dir / f"gdelt_raw_all_2019-01-01_2025-09-04.parquet"
        mraw.to_parquet(mraw_fp, index=False)
        print(f"[OK] merged raw: {len(mraw)} → {mraw_fp.name}")

    if merged_flt_frames:
        mflt = pd.concat(merged_flt_frames, ignore_index=True)
        mflt_fp = flt_dir / f"gdelt_filtered_all_2019-01-01_2025-09-04.parquet"
        mflt.to_parquet(mflt_fp, index=False)
        print(f"[OK] merged filtered: {len(mflt)} → {mflt_fp.name}")

        mdaily = daily_counts(mflt)
        mdaily_fp = agg_dir / f"gdelt_daily_counts_all_2019-01-01_2025-09-04.parquet"
        mdaily.to_parquet(mdaily_fp, index=False)
        print(f"[OK] merged daily counts: {len(mdaily)} → {mdaily_fp.name}")
    else:
        print("[WARN] No filtered data across symbols.")


if __name__ == "__main__":
    main()
