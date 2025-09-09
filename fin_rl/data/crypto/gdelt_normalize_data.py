#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalize GDELT-like crypto news:
- Canonicalize URL
- Build unique articles table (one row per canonical URL)
- Build many-to-many relation table (url_canon <-> symbol_req) with stats
- Optional per-symbol unique view
- Optional main_symbol assignment by priority list

Usage:
  python gdelt_unique_normalize.py \
    --input /path/to/gdelt_filtered_merged_2020-08-11_2025-09-03.parquet \
    --out-dir /path/to/normalized_unique \
    --priority-tokens BTC,ETH,BNB,SOL \
    --write-csv
"""
import argparse
import os
import re
from typing import List, Set
import pandas as pd
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode


DEFAULT_TRACKING_PARAMS: Set[str] = {
    "utm_source","utm_medium","utm_campaign","utm_term","utm_content",
    "gclid","fbclid","mc_cid","mc_eid","yclid","igshid","utm_name","utm_id","utm_reader"
}


def canonicalize_url(u: str, tracking_params: Set[str]) -> str:
    """Lower-case host, drop fragment, strip common tracking params, keep order."""
    try:
        u = (u or "").strip()
        p = urlparse(u)
        netloc = p.netloc.lower()
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if k.lower() not in tracking_params]
        query = urlencode(q, doseq=True)
        p2 = p._replace(netloc=netloc, query=query, fragment="")
        return urlunparse(p2)
    except Exception:
        return (u or "").strip()


def normalize_title(t: str) -> str:
    """Light normalization for dedup-near comparisons."""
    t = (t or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]+", "", t)
    return t


def parse_priority_list(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def build_relationship(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build url<->token relation with:
      first_seen, last_seen, n_hits, days_list
    """
    g = df.groupby(["url_canon","symbol_req"], dropna=False)
    rel = g.agg(
        first_seen=("seendate", "min"),
        last_seen =("seendate", "max"),
        n_hits    =("seendate", "size"),
        days_list =("day", lambda s: sorted(pd.unique(s)))
    ).reset_index()
    return rel


def build_articles(df: pd.DataFrame, rel: pd.DataFrame) -> pd.DataFrame:
    """
    Representative row per url_canon (latest seendate if available),
    then enrich with tokens, languages, sourcecountries, days, and counts.
    """
    # choose representative row: latest seendate per url
    df_sorted = df.sort_values(by=["url_canon", "seendate"], ascending=[True, True])
    rep_idx = df_sorted.groupby("url_canon", as_index=False).tail(1).index
    rep = df_sorted.loc[rep_idx].copy()

    # tokens list
    tokens_by_url = rel.groupby("url_canon")["symbol_req"] \
                       .apply(lambda s: sorted(pd.unique(s))) \
                       .rename("tokens")

    # all days that URL appeared
    days_by_url = rel.groupby("url_canon")["days_list"] \
                     .apply(lambda lists: sorted({d for L in lists for d in L})) \
                     .rename("all_days")

    # languages, countries (sets)
    lang_by_url = df.groupby("url_canon")["language"] \
                    .apply(lambda s: sorted(pd.unique(s.dropna()))) \
                    .rename("languages")

    country_by_url = df.groupby("url_canon")["sourcecountry"] \
                       .apply(lambda s: sorted(pd.unique(s.dropna()))) \
                       .rename("sourcecountries")

    # first/last seen per url, and total records
    first_seen_url = rel.groupby("url_canon")["first_seen"].min().rename("first_seen_any")
    last_seen_url  = rel.groupby("url_canon")["last_seen"].max().rename("last_seen_any")
    n_records_url  = df.groupby("url_canon").size().rename("n_records")

    articles = (rep
        .set_index("url_canon")
        .join([tokens_by_url, days_by_url, lang_by_url, country_by_url,
               first_seen_url, last_seen_url, n_records_url],
              how="left")
        .reset_index()
    )

    # keep tidy columns when present
    keep_cols = [
        "url_canon","url","title","title_norm","domain","language","sourcecountry",
        "socialimage","seendate","date","day",
        "tokens","all_days","languages","sourcecountries",
        "first_seen_any","last_seen_any","n_records"
    ]
    return articles[[c for c in keep_cols if c in articles.columns]]


def assign_main_symbol(articles: pd.DataFrame, rel: pd.DataFrame, priority: List[str]) -> pd.DataFrame:
    """
    Optional: add a 'main_symbol' to articles by priority order.
    If none of the article's tokens are in priority list, pick the first token (if any).
    """
    if not priority or "tokens" not in articles.columns:
        return articles

    pri_map = {sym: i for i, sym in enumerate(priority)}
    def pick_main(tokens: List[str]) -> str:
        if not tokens:
            return None
        ranked = sorted(tokens, key=lambda s: pri_map.get(s, 10_000))
        return ranked[0]

    articles = articles.copy()
    articles["main_symbol"] = articles["tokens"].apply(pick_main)
    return articles


def main():
    ap = argparse.ArgumentParser(description="Normalize GDELT-like crypto news to unique articles and url↔token map.")
    ap.add_argument("--input", required=True, help="Input parquet file (merged/filtered).")
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--priority-tokens", default="", help="Comma-separated token priority (e.g., BTC,ETH,BNB,SOL).")
    ap.add_argument("--write-csv", action="store_true", help="Also export CSV (besides Parquet).")
    ap.add_argument("--no-per-symbol-view", action="store_true", help="Skip generating per_symbol_unique view.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    priority = parse_priority_list(args.priority_tokens)

    # --- Load
    df = pd.read_parquet(args.input)
    n0 = len(df)

    # Ensure columns exist / types
    if "seendate" in df.columns:
        df["seendate"] = pd.to_datetime(df["seendate"], errors="coerce", utc=True)
    else:
        df["seendate"] = pd.NaT

    if "day" not in df.columns or df["day"].isna().all():
        df["day"] = pd.to_datetime(df["seendate"]).dt.date.astype(str)

    # Canonical/normalized fields
    df["url_canon"]  = df["url"].astype(str).apply(lambda u: canonicalize_url(u, DEFAULT_TRACKING_PARAMS))
    df["title_norm"] = df["title"].astype(str).apply(normalize_title)
    df["symbol_req"] = df["symbol_req"].astype(str)

    # --- Build relation and articles
    rel = build_relationship(df)
    articles = build_articles(df, rel)
    articles = assign_main_symbol(articles, rel, priority=priority)

    # Optional per-symbol unique (1 row per (url_canon, symbol_req))
    if not args.no_per_symbol_view:
        per_symbol_unique = rel[["url_canon","symbol_req","first_seen","last_seen","n_hits"]].copy()
    else:
        per_symbol_unique = None

    # --- Save
    articles_path = os.path.join(args.out_dir, "articles.parquet")
    rel_path      = os.path.join(args.out_dir, "article_token_map.parquet")
    articles.to_parquet(articles_path, index=False)
    rel.to_parquet(rel_path, index=False)

    if per_symbol_unique is not None:
        per_sym_path = os.path.join(args.out_dir, "per_symbol_unique.parquet")
        per_symbol_unique.to_parquet(per_sym_path, index=False)
    else:
        per_sym_path = None

    if args.write_csv:
        articles.to_csv(os.path.splitext(articles_path)[0] + ".csv", index=False)
        rel.to_csv(os.path.splitext(rel_path)[0] + ".csv", index=False)
        if per_sym_path:
            per_symbol_unique.to_csv(os.path.splitext(per_sym_path)[0] + ".csv", index=False)

    # --- Summary
    print("=== SUMMARY ===")
    print(f"Input rows              : {n0}")
    print(f"Unique URLs (articles)  : {len(articles)}")
    print(f"URL↔Token relations     : {len(rel)}")
    print(f"- Saved: {articles_path}")
    print(f"- Saved: {rel_path}")
    if per_sym_path:
        print(f"- Saved: {per_sym_path}")

    # Sanity checks
    dups_articles = articles.duplicated(subset=["url_canon"]).sum()
    dups_rel = rel.duplicated(subset=["url_canon","symbol_req"]).sum()
    print("\nChecks:")
    print(f"- articles duplicate url_canon  : {dups_articles}")
    print(f"- rel duplicate (url,symbol)    : {dups_rel}")

    # Peek
    with pd.option_context("display.max_colwidth", 120):
        print("\n=== Sample: articles ===")
        print(articles.head(5)[["url_canon","title","domain","tokens","first_seen_any","last_seen_any","n_records"]])
        print("\n=== Sample: article_token_map ===")
        print(rel.head(8))


if __name__ == "__main__":
    main()
