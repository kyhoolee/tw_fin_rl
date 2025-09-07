#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GDELT hard-fix (split by symbol) — 4 req/day (BTC, ETH, BNB, SOL), verbose console debug,
fallback to strict on “too common”, per-day+window+symbol caching.

Example:
  nohup python -m fin_rl.data.crypto.gdelt_demo_hardfix_split \
    --start 2024-01-01 --end 2024-01-05 \
    --window 09:00-12:00 \
    --langs english \
    --sleep-sec 0.5 \
    --sleep-between-req 0.5 \
    --verbose \
    --out ./work/data/gdelt_align_hardfix_split \
    > gdelt_split.log 2>&1 &
"""

import argparse
import datetime as dt
import hashlib
import json
import re
import time
import traceback
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Hard ranges (end exclusive)
RANGE_BTC = (dt.date(2019, 1, 1), dt.date(2025, 9, 5))
RANGE_ETH = (dt.date(2019, 1, 1), dt.date(2025, 9, 5))
RANGE_BNB = (dt.date(2019, 1, 1), dt.date(2025, 9, 5))
RANGE_SOL = (dt.date(2020, 8, 11), dt.date(2025, 9, 5))

COMMON_START = max(RANGE_BTC[0], RANGE_ETH[0], RANGE_BNB[0], RANGE_SOL[0])
COMMON_END   = min(RANGE_BTC[1], RANGE_ETH[1], RANGE_BNB[1], RANGE_SOL[1])  # exclusive

TOKENS = ["BTC", "ETH", "BNB", "SOL"]

# ===== Heuristics =====
CRYPTO_WHITELIST = {
    "coindesk.com", "theblock.co", "decrypt.co", "bitcoinmagazine.com",
    "cointelegraph.com", "beincrypto.com", "cryptoslate.com",
    "ambcrypto.com", "u.today", "blockworks.co", "cryptobriefing.com",
    "finbold.com", "newsbtc.com", "coingeek.com", "coincu.com",
    "cryptonews.com", "cryptopotato.com", "cryptonewsz.com",
}

BLACKLIST_PATTERNS = {
    "SOL": [r"\bCosta del Sol\b", r"\bdel Sol\b", r"\bSol de\b", r"\bsolar\b", r"\bsolstice\b"],
    "ETH": [r"\bETH Zurich\b", r"\bEthiopia(n)?\b", r"\bethanol\b", r"\bethane\b", r"\bethylene\b"],
    "BNB": [r"\bAirbnb\b", r"\bB&B\b", r"\bbed(?:\s|-)and(?:\s|-)breakfast\b"],
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

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/javascript,*/*;q=0.1",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

# ---------- Utils ----------
def _ymd_hms(dtobj: dt.datetime) -> str:
    return dtobj.strftime("%Y%m%d%H%M%S")

def _match_any(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)

def _parse_window(s: str) -> Tuple[dt.time, dt.time]:
    a, b = s.split("-")
    ts = dt.datetime.strptime(a.strip(), "%H:%M").time()
    te = dt.datetime.strptime(b.strip(), "%H:%M").time()
    if dt.datetime.combine(dt.date.today(), te) <= dt.datetime.combine(dt.date.today(), ts):
        raise ValueError("window end must be after start")
    return ts, te

def _build_balanced_clause(symbol: str) -> str:
    s = symbol.upper()
    if s == "BTC": return '"Bitcoin" OR "BTC token" OR "BTC price"'
    if s == "ETH": return '"Ethereum" OR "ETH token" OR "ETH price"'
    if s == "BNB": return '"BNB Chain" OR "BNB token" OR "BNB price" OR "Binance"'
    if s == "SOL": return '"Solana" OR "SOL token" OR "SOL price"'
    return s

def _build_strict_clause(symbol: str) -> str:
    s = symbol.upper()
    if s == "BTC": return '"Bitcoin"'
    if s == "ETH": return '"Ethereum"'
    if s == "BNB": return '"BNB Chain" OR "Binance"'
    if s == "SOL": return '"Solana"'
    return s

def build_query_for_symbol(symbol: str, langs: List[str], mode: str = "balanced") -> str:
    clause = _build_balanced_clause(symbol) if mode == "balanced" else _build_strict_clause(symbol)
    q = "(" + clause + ")"
    if langs:
        q += " " + " ".join([f"sourcelang:{l}" for l in langs])
    return q

def day_cache_key(date_obj: dt.date, start_ts: dt.time, end_ts: dt.time, symbol: str, query: str) -> str:
    raw = f"{date_obj.isoformat()}|{start_ts.isoformat()}|{end_ts.isoformat()}|{symbol}|{query}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

def short_headers(h: Dict[str, str], keys: List[str] = None) -> Dict[str, str]:
    if not h: return {}
    if keys:
        return {k: h.get(k) for k in keys}
    out = {}
    for i, (k, v) in enumerate(h.items()):
        if i >= 10: break
        sv = str(v)
        out[k] = sv if len(sv) <= 200 else (sv[:200] + "...(+)")
    return out

# ---------- Networking with verbose console & fallback ----------
def gdelt_fetch_once(query: str,
                     start_dt: dt.datetime,
                     end_dt: dt.datetime,
                     headers: Optional[Dict[str, str]] = None,
                     maxrecords: int = 250,
                     verbose: bool = False,
                     max_retries: int = 6,
                     base_sleep: float = 2.0) -> List[Dict[str, Any]]:

    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": maxrecords,
        "sort": "DateDesc",
        "format": "json",
        "startdatetime": _ymd_hms(start_dt),
        "enddatetime": _ymd_hms(end_dt),
    }
    url = f"{GDELT_DOC_API}?{urllib.parse.urlencode(params)}"

    sess = requests.Session()
    req_headers = {**DEFAULT_HEADERS, **(headers or {})}

    if verbose:
        print(f"[HTTP] GET {url}")

    for attempt in range(max_retries + 1):
        try:
            r = sess.get(url, headers=req_headers, timeout=30)
            ct = r.headers.get("Content-Type", "")
            clen = r.headers.get("Content-Length", "")
            ra  = r.headers.get("Retry-After")
            if verbose:
                print(f"[HTTP] status={r.status_code} ct={ct} len={clen} retry-after={ra} hdrs={short_headers(r.headers, ['Date','Content-Type','Content-Length','Retry-After'])}")

            if r.status_code == 429 or 500 <= r.status_code < 600:
                raise requests.HTTPError(f"{r.status_code} Server/Rate limit: {r.text[:200]}")

            r.raise_for_status()

            try:
                js = r.json()
            except Exception as je:
                snippet = (r.text or "")[:300].replace("\n", " ")
                print(f"[HTTP] JSON parse error: {je} | snippet='{snippet}'")
                raise

            return js.get("articles", []) or []

        except Exception as e:
            if attempt >= max_retries:
                print(f"[HTTP] final failure after {attempt} retries: {e}")
                traceback.print_exc()
                raise
            sleep_s = base_sleep * (2 ** attempt) + (0.3 * attempt)
            print(f"[WARN] fetch retry {attempt+1}/{max_retries} after error: {e} — sleep {sleep_s:.1f}s")
            time.sleep(sleep_s)

    return []

def fetch_symbol_with_fallback(symbol: str,
                               langs: List[str],
                               start_dt: dt.datetime,
                               end_dt: dt.datetime,
                               maxrecords: int,
                               verbose: bool) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Try balanced query first; if 400 'too common' or related parse error occurs,
    fallback to strict clause and retry once (with backoff inside gdelt_fetch_once).
    Returns (mode_used, articles).
    """
    # 1) balanced
    q_bal = build_query_for_symbol(symbol, langs, mode="balanced")
    try:
        arts = gdelt_fetch_once(q_bal, start_dt, end_dt, maxrecords=maxrecords, verbose=verbose)
        return ("balanced", arts)
    except Exception as e:
        # Detect “too common/too short/too long” from message or body snippet
        msg = str(e).lower()
        too_common = ("too common" in msg) or ("too short" in msg) or ("too long" in msg) or ("orclauseid" in msg)
        if verbose:
            print(f"[INFO] {symbol} balanced failed, too_common={too_common}. Falling back to strict...")
        # 2) strict fallback
        q_strict = build_query_for_symbol(symbol, langs, mode="strict")
        arts = gdelt_fetch_once(q_strict, start_dt, end_dt, maxrecords=maxrecords, verbose=verbose)
        return ("strict", arts)

# ---------- Filter & assign ----------
def classify_symbols_for_title(title: str, domain: str) -> List[str]:
    title = (title or "").strip()
    domain = (domain or "").strip().lower()
    if not title:
        return []

    matched_syms = set()

    if domain in CRYPTO_WHITELIST and len(title) > 3:
        for sym, anchors in SYMBOL_ANCHORS.items():
            blist = BLACKLIST_PATTERNS.get(sym, [])
            if blist and _match_any(blist, title): continue
            if _match_any(anchors, title): matched_syms.add(sym)
        if not matched_syms:
            for sym, anchors in SYMBOL_ANCHORS.items():
                blist = BLACKLIST_PATTERNS.get(sym, [])
                if blist and _match_any(blist, title): continue
                if _match_any(CRYPTO_CONTEXT, title) and _match_any(anchors, title):
                    matched_syms.add(sym)
        return sorted(matched_syms)

    for sym, anchors in SYMBOL_ANCHORS.items():
        blist = BLACKLIST_PATTERNS.get(sym, [])
        if blist and _match_any(blist, title): continue
        if _match_any(anchors, title): matched_syms.add(sym)

    if not matched_syms and _match_any(CRYPTO_CONTEXT, title):
        for sym, anchors in SYMBOL_ANCHORS.items():
            blist = BLACKLIST_PATTERNS.get(sym, [])
            if blist and _match_any(blist, title): continue
            if _match_any(anchors, title): matched_syms.add(sym)

    return sorted(matched_syms)

def filter_and_assign(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame(columns=[
            "title", "url", "domain", "language", "seendate",
            "socialimage", "sourcecountry", "isduplicate",
            "symbol", "date"
        ])

    df = df_raw.copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["domain"] = df.get("domain", "").fillna("").astype(str)
    df["seendate"] = pd.to_datetime(df["seendate"], errors="coerce", utc=True)
    df["date"] = df["seendate"].dt.date

    # assign symbols by anchors/context
    sym_lists = []
    for _, row in df.iterrows():
        syms = classify_symbols_for_title(row["title"], row["domain"])
        sym_lists.append(syms)
    df["__symbols__"] = sym_lists

    df = df.loc[df["__symbols__"].map(len) > 0].explode("__symbols__").rename(columns={"__symbols__": "symbol"})
    df = df.drop_duplicates(subset=["url", "symbol"], keep="first")
    keep_cols = [
        "title", "url", "domain", "language", "seendate",
        "socialimage", "sourcecountry", "isduplicate", "symbol", "date"
    ]
    return df[keep_cols]

def daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "symbol", "count"])
    d = (df.dropna(subset=["date", "symbol"])
           .groupby(["date", "symbol"])
           .size()
           .reset_index(name="count")
           .sort_values(["date", "symbol"]))
    return d

# ---------- Runner ----------
def run_hardfix_split(out_dir: str,
                      langs: str = "english",
                      window: str = "09:00-12:00",
                      maxrecords: int = 250,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      sleep_sec: float = 0.0,            # between days
                      sleep_between_req: float = 0.0,     # between symbols in a day
                      verbose: bool = False,
                      force: bool = False):

    outdir = Path(out_dir)
    raw_dir = outdir / "raw"
    flt_dir = outdir / "filtered"
    agg_dir = outdir / "agg"
    manifest_dir = outdir / "_manifest"
    for d in [raw_dir, flt_dir, agg_dir, manifest_dir]:
        d.mkdir(parents=True, exist_ok=True)

    langs_list = [s.strip() for s in langs.split(",") if s.strip()]
    t0, t1 = _parse_window(window)  # UTC window

    s_date = dt.date.fromisoformat(start_date) if start_date else COMMON_START
    e_date = dt.date.fromisoformat(end_date) if end_date else COMMON_END
    if e_date <= s_date:
        raise ValueError("end_date must be after start_date")

    print(f"[INFO] Hardfix split | {s_date} → {e_date} (end exclusive) | window={window} UTC | langs={langs} | sleep_day={sleep_sec:.1f}s | sleep_req={sleep_between_req:.1f}s")

    merged_kept_frames = []
    merged_raw_frames = []

    cur = s_date
    while cur < e_date:
        start_dt = dt.datetime.combine(cur, t0, tzinfo=dt.timezone.utc)
        end_dt   = dt.datetime.combine(cur, t1, tzinfo=dt.timezone.utc)

        day_raw_frames = []
        day_kept_frames = []

        for i, sym in enumerate(TOKENS):
            q = build_query_for_symbol(sym, langs_list, mode="balanced")
            key = day_cache_key(cur, t0, t1, sym, q)
            raw_fp = raw_dir / f"gdelt_raw_{sym}_{cur.isoformat()}_{t0.strftime('%H%M')}_{t1.strftime('%H%M')}_{key}.parquet"
            flt_fp = flt_dir / f"gdelt_filtered_{sym}_{cur.isoformat()}_{t0.strftime('%H%M')}_{t1.strftime('%H%M')}_{key}.parquet"
            mft_fp = manifest_dir / f"request_{sym}_{cur.isoformat()}_{key}.json"

            if raw_fp.exists() and flt_fp.exists() and not force:
                print(f"[SKIP] {cur} {sym} (cached)")
                try:    df_raw = pd.read_parquet(raw_fp)
                except: df_raw = pd.DataFrame()
                try:    df_flt = pd.read_parquet(flt_fp)
                except: df_flt = pd.DataFrame()
            else:
                try:
                    mode_used, arts = fetch_symbol_with_fallback(sym, langs_list, start_dt, end_dt, maxrecords, verbose)
                    rows = [{
                        "title": a.get("title"),
                        "url": a.get("url"),
                        "domain": a.get("domain"),
                        "language": a.get("language"),
                        "seendate": a.get("seendate"),
                        "socialimage": a.get("socialimage"),
                        "sourcecountry": a.get("sourcecountry"),
                        "isduplicate": a.get("isduplicate"),
                    } for a in arts]
                    df_raw = pd.DataFrame(rows).drop_duplicates(subset=["url"])
                    df_raw["seendate"] = pd.to_datetime(df_raw["seendate"], errors="coerce", utc=True)
                    df_raw.to_parquet(raw_fp, index=False)

                    df_flt = filter_and_assign(df_raw)
                    # Khi chạy per-symbol, có thể giữ đúng symbol (nếu muốn “ép nhãn”):
                    # df_flt["symbol"] = sym
                    df_flt.to_parquet(flt_fp, index=False)

                    with open(mft_fp, "w", encoding="utf-8") as f:
                        json.dump({
                            "date": cur.isoformat(),
                            "window": window,
                            "symbol": sym,
                            "query": q,
                            "langs": langs_list,
                            "maxrecords": maxrecords,
                            "mode_used": mode_used,
                            "status": "ok",
                            "raw_count": int(len(df_raw)),
                            "kept_count": int(len(df_flt)),
                            "ts": dt.datetime.utcnow().isoformat() + "Z",
                            "cache_key": key,
                            "files": {"raw": str(raw_fp), "filtered": str(flt_fp)},
                        }, f, ensure_ascii=False, indent=2)

                except Exception as e:
                    traceback.print_exc()
                    # placeholders
                    pd.DataFrame(columns=["title","url","domain","language","seendate","socialimage","sourcecountry","isduplicate"]).to_parquet(raw_fp, index=False)
                    pd.DataFrame(columns=["title","url","domain","language","seendate","socialimage","sourcecountry","isduplicate","symbol","date"]).to_parquet(flt_fp, index=False)
                    with open(mft_fp, "w", encoding="utf-8") as f:
                        json.dump({
                            "date": cur.isoformat(),
                            "window": window,
                            "symbol": sym,
                            "query": q,
                            "langs": langs_list,
                            "maxrecords": maxrecords,
                            "status": "error",
                            "error": str(e),
                            "ts": dt.datetime.utcnow().isoformat() + "Z",
                            "cache_key": key,
                            "files": {"raw": str(raw_fp), "filtered": str(flt_fp)},
                        }, f, ensure_ascii=False, indent=2)

                    # tiếp symbol kế
                    if sleep_between_req > 0:
                        print(f"[SLEEP] between-req {sleep_between_req:.1f}s …")
                        time.sleep(sleep_between_req)
                    continue

            print(f"[OK] {cur} {sym} | raw={len(df_raw)} | kept={len(df_flt)} → {flt_fp.name}")
            if not df_raw.empty:
                day_raw_frames.append(df_raw.assign(day=cur, symbol_req=sym))
            if not df_flt.empty:
                day_kept_frames.append(df_flt.assign(day=cur, symbol_req=sym))

            if i < len(TOKENS)-1 and sleep_between_req > 0:
                print(f"[SLEEP] between-req {sleep_between_req:.1f}s …")
                time.sleep(sleep_between_req)

        # merge trong ngày
        if day_raw_frames:
            merged_raw_frames.append(pd.concat(day_raw_frames, ignore_index=True))
        if day_kept_frames:
            merged_kept_frames.append(pd.concat(day_kept_frames, ignore_index=True))

        if sleep_sec > 0:
            print(f"[SLEEP] between-day {sleep_sec:.1f}s …")
            time.sleep(sleep_sec)

        cur += dt.timedelta(days=1)

    # merged outputs (for this run)
    if merged_raw_frames:
        mraw = pd.concat(merged_raw_frames, ignore_index=True)
        mraw_fp = flt_dir / f"gdelt_raw_merged_{s_date}_{(e_date - dt.timedelta(days=1))}.parquet"
        mraw.to_parquet(mraw_fp, index=False)
        print(f"[OK] merged raw: {len(mraw)} → {mraw_fp.name}")

    if merged_kept_frames:
        mflt = pd.concat(merged_kept_frames, ignore_index=True)
        # chống trùng nếu bạn sẽ chạy nhiều khung giờ/ngày sau này
        mflt = mflt.drop_duplicates(subset=["url","symbol"], keep="first")
        mflt_fp = flt_dir / f"gdelt_filtered_merged_{s_date}_{(e_date - dt.timedelta(days=1))}.parquet"
        mflt.to_parquet(mflt_fp, index=False)
        print(f"[OK] merged filtered: {len(mflt)} → {mflt_fp.name}")

        mdaily = daily_counts(mflt)
        mdaily_fp = agg_dir / f"gdelt_daily_counts_{s_date}_{(e_date - dt.timedelta(days=1))}.parquet"
        mdaily.to_parquet(mdaily_fp, index=False)
        print(f"[OK] merged daily counts: {len(mdaily)} → {mdaily_fp.name}")
    else:
        print("[WARN] No filtered data at all.")

def main():
    ap = argparse.ArgumentParser(description="GDELT hard-fix split-by-symbol (4 req/day) with console debug")
    ap.add_argument("--out", type=str, default="./work/data/gdelt_align_hardfix_split", help="Output base directory")
    ap.add_argument("--langs", type=str, default="english", help="Comma-separated sourcelang filters")
    ap.add_argument("--window", type=str, default="09:00-12:00", help="UTC window 'HH:MM-HH:MM'")
    ap.add_argument("--maxrecords", type=int, default=250, help="Max records per GDELT request")
    ap.add_argument("--start", type=str, default=None, help=f"Override start date (YYYY-MM-DD). Default={COMMON_START}")
    ap.add_argument("--end", type=str, default=None, help=f"Override end date EXCLUSIVE (YYYY-MM-DD). Default={COMMON_END}")
    ap.add_argument("--sleep-sec", type=float, default=0.0, help="Fixed sleep between days")
    ap.add_argument("--sleep-between-req", type=float, default=0.0, help="Fixed sleep between 4 requests in a day")
    ap.add_argument("--verbose", action="store_true", help="Verbose HTTP logging (console)")
    ap.add_argument("--force", action="store_true", help="Re-fetch even if cached files exist for a day/symbol")
    args = ap.parse_args()

    run_hardfix_split(
        out_dir=args.out,
        langs=args.langs,
        window=args.window,
        maxrecords=args.maxrecords,
        start_date=args.start,
        end_date=args.end,
        sleep_sec=args.sleep_sec,
        sleep_between_req=args.sleep_between_req,
        verbose=args.verbose,
        force=args.force,
    )

if __name__ == "__main__":
    main()
