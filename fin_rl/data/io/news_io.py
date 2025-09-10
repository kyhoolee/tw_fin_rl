#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
news_io.py — Reader cho dữ liệu news đã normalize
=================================================

Dữ liệu được sinh ra bởi script gdelt_unique_normalize.py và lưu trong thư mục
`normalized_unique/`. Cấu trúc chuẩn:

normalized_unique/
├── articles.parquet
├── article_token_map.parquet
└── per_symbol_unique.parquet

1. articles.parquet
   - Granularity: 1 row / 1 unique canonical URL
   - Columns chính:
       url_canon       : URL đã canonicalize
       url             : URL gốc
       title           : tiêu đề
       tokens          : list các symbol liên quan (VD: ['BTC','ETH'])
       all_days        : list ngày xuất hiện
       first_seen_any  : timestamp đầu tiên thấy news
       last_seen_any   : timestamp cuối cùng thấy news
       n_records       : số record gốc match vào URL này
       main_symbol     : (optional) token chính theo priority

2. article_token_map.parquet
   - Granularity: 1 row / (url_canon, symbol_req)
   - Columns chính:
       url_canon, symbol_req
       first_seen, last_seen
       n_hits          : số lần gặp
       days_list       : list ngày

3. per_symbol_unique.parquet
   - Granularity: 1 row / (url_canon, symbol_req) (giản lược)
   - Columns chính:
       url_canon, symbol_req, first_seen, last_seen, n_hits
"""

import argparse
import pandas as pd
from pathlib import Path
from prettytable import PrettyTable
import ast
import numpy as np


# -------------------------------------------------------------------
# Đọc và summarize bảng ARTICLES (unique URLs)
# -------------------------------------------------------------------
def summarize_articles(articles_path: Path) -> pd.DataFrame:
    """
    Đọc articles.parquet:
    - 1 row / unique URL
    - chứa tokens (list symbols), first_seen_any, last_seen_any
    Trả về DataFrame articles.
    """
    df = pd.read_parquet(articles_path)

    # Table tổng quan
    t = PrettyTable()
    t.title = "ARTICLES (unique URLs)"
    t.field_names = ["Rows", "Start", "End"]
    t.add_row([len(df), str(df["first_seen_any"].min()), str(df["last_seen_any"].max())])
    print(t)

    # Table sample
    t2 = PrettyTable()
    t2.title = "Sample articles"
    t2.field_names = ["url_canon", "tokens", "first_seen_any", "last_seen_any", "n_records"]
    for _, r in df.head(5).iterrows():
        t2.add_row([
            str(r["url_canon"])[:60] + ("..." if len(str(r["url_canon"])) > 60 else ""),
            ",".join(r["tokens"]) if isinstance(r["tokens"], list) else str(r["tokens"]),
            r["first_seen_any"], r["last_seen_any"], r["n_records"]
        ])
    print(t2)
    return df


# -------------------------------------------------------------------
# Đọc và summarize bảng ARTICLE ↔ TOKEN MAP
# -------------------------------------------------------------------
def summarize_rel(rel_path: Path):
    """
    Đọc article_token_map.parquet:
    - 1 row / (url_canon, symbol_req)
    - chứa first_seen, last_seen, n_hits
    Trả về DataFrame rel và list symbols.
    """
    df = pd.read_parquet(rel_path)
    g = df.groupby("symbol_req").agg(
        urls=("url_canon", "nunique"),
        first_seen=("first_seen", "min"),
        last_seen=("last_seen", "max"),
        hits=("n_hits", "sum")
    ).reset_index()

    t = PrettyTable()
    t.title = "ARTICLE ↔ TOKEN MAP"
    t.field_names = ["Symbol", "URLs", "First seen", "Last seen", "Hits"]
    for _, r in g.iterrows():
        t.add_row([r["symbol_req"], r["urls"], r["first_seen"], r["last_seen"], r["hits"]])
    print(t)
    return df, g["symbol_req"].tolist()


# -------------------------------------------------------------------
# Đọc và summarize bảng PER-SYMBOL UNIQUE
# -------------------------------------------------------------------
def summarize_per_symbol(per_symbol_path: Path):
    """
    Đọc per_symbol_unique.parquet:
    - 1 row / (url_canon, symbol_req)
    - giản lược, chỉ giữ count
    """
    df = pd.read_parquet(per_symbol_path)
    g = df.groupby("symbol_req").size().reset_index(name="count")

    t = PrettyTable()
    t.title = "PER-SYMBOL UNIQUE"
    t.field_names = ["Symbol", "Count"]
    for _, r in g.iterrows():
        t.add_row([r["symbol_req"], r["count"]])
    print(t)
    return df


# -------------------------------------------------------------------
# In sample news cho từng token
# -------------------------------------------------------------------
def print_samples_by_token(articles_df: pd.DataFrame, symbols: list, n: int = 3):
    """
    Với mỗi symbol:
    - Lọc articles có tokens chứa symbol đó
    - In ra n sample (head)
    """
    # Chuẩn hóa tokens về list thực
    if "tokens" in articles_df.columns:
        articles_df = articles_df.copy()
        articles_df["tokens"] = articles_df["tokens"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
        )

    print(f"Sample articles_df:: {articles_df.columns.tolist()}")

    # In dtype chung và thêm type thực tế của 1–2 sample
    for c in articles_df.columns:
        sample_val = articles_df[c].dropna().iloc[0] if not articles_df[c].dropna().empty else None
        print(f"  {c}: dtype={articles_df[c].dtype}, sample_type={type(sample_val)}")

    
    print(articles_df.head(3))
    print(articles_df[["tokens", "title_norm", "main_symbol"]].head(3))



    

    for sym in symbols:
        sub = articles_df[articles_df["tokens"].apply(
            lambda toks: sym in toks if isinstance(toks, (list, np.ndarray)) else False
        )]
        if sub.empty:
            continue
        print(sym, sub.head())


        t = PrettyTable()
        t.title = f"Sample news for {sym}"
        t.field_names = ["url_canon", "title", "first_seen", "last_seen"]
        for _, r in sub.head(n).iterrows():
            t.add_row([
                str(r["url_canon"])[:60] + ("..." if len(str(r["url_canon"])) > 60 else ""),
                (str(r["title"])[:40] + "...") if len(str(r["title"])) > 40 else str(r["title"]),
                r["first_seen_any"], r["last_seen_any"]
            ])
        print(t)


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news-dir", required=True,
                    help="Path to normalized_unique/ (contains articles.parquet, article_token_map.parquet, per_symbol_unique.parquet)")
    ap.add_argument("--samples-per-token", type=int, default=3,
                    help="Number of sample articles per token to show")
    args = ap.parse_args()
    news_dir = Path(args.news_dir)

    articles_df = summarize_articles(news_dir / "articles.parquet")
    rel_df, symbols = summarize_rel(news_dir / "article_token_map.parquet")
    per_symbol = news_dir / "per_symbol_unique.parquet"
    if per_symbol.exists():
        summarize_per_symbol(per_symbol)

    print_samples_by_token(articles_df, symbols, n=args.samples_per_token)


if __name__ == "__main__":
    main()
