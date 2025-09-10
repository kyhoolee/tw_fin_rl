#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate notebook: news_examine.ipynb

This script creates a Jupyter Notebook that uses your helpers in
`fin_rl.data.io.news_io` to examine the normalized GDELT news dataset.

Usage:
    python make_news_notebook.py

The notebook will be saved as: ./news_examine.ipynb
"""

import os
import nbformat as nbf

def main():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title + intro
    cells.append(nbf.v4.new_markdown_cell(r"""
# Crypto News (GDELT normalized) — Quick Examine Notebook

This notebook **uses the helper functions from** `fin_rl.data.io.news_io` to explore your
normalized GDELT news dataset (under `normalized_unique/`).

What it does:
- Call module helpers:
  - `summarize_articles(articles.parquet)`
  - `summarize_rel(article_token_map.parquet)`
  - `summarize_per_symbol(per_symbol_unique.parquet)` *(if present)*
  - `print_samples_by_token(...)`
- Then some quick extra explorations (no resampling, just groupby):
  - Daily article counts (overall)
  - Daily article counts per symbol (subset)
  - Top domains / languages overview

> Assumes you have already installed your repo as a package (e.g. `pip install -e .`)
> so that `fin_rl` can be imported without manual `sys.path` edits.
"""))

    # Config cell
    cells.append(nbf.v4.new_code_cell(r"""
# ==== CONFIG ====
NEWS_DIR = "work/data/gdelt_0024_hardfix_split/normalized_unique"  # change if needed
SAMPLES_PER_TOKEN = 3
# Optional subset to plot per-symbol daily counts. Leave empty to auto-pick top 4.
PLOT_SYMBOLS = ["BTC", "ETH", "BNB", "SOL"]
SAVE_PLOTS = True
OUT_DIR = "./_out_news"
"""))

    # Imports
    cells.append(nbf.v4.new_code_cell(r"""
# ==== IMPORTS ====
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fin_rl.data.io import news_io as nio
from importlib import reload
reload(nio)

print("Loaded helpers from:", nio.__file__)
"""))

    # Call helpers
    cells.append(nbf.v4.new_markdown_cell("## 1) Load + Summaries via module helpers"))
    cells.append(nbf.v4.new_code_cell(r"""
from pathlib import Path
news_dir = Path(NEWS_DIR)

# 1) Articles (unique URLs)
articles_df = nio.summarize_articles(news_dir / "articles.parquet")

# 2) Article <-> Token map
rel_df, all_symbols = nio.summarize_rel(news_dir / "article_token_map.parquet")

# 3) Per-symbol unique (if exists)
per_symbol_path = news_dir / "per_symbol_unique.parquet"
if per_symbol_path.exists():
    per_symbol_df = nio.summarize_per_symbol(per_symbol_path)
else:
    per_symbol_df = None

# 4) Print token samples
nio.print_samples_by_token(articles_df, all_symbols, n=SAMPLES_PER_TOKEN)
"""))

    # Quick peeks (dataframes)
    cells.append(nbf.v4.new_markdown_cell("## 2) Quick Peeks (dataframes)"))
    cells.append(nbf.v4.new_code_cell(r"""
print("\n=== articles_df.head() ===")
display(articles_df.head())

print("\n=== rel_df.head() ===")
display(rel_df.head())

if per_symbol_df is not None:
    print("\n=== per_symbol_df.head() ===")
    display(per_symbol_df.head())
"""))

    # Basic Quality Checks
    cells.append(nbf.v4.new_markdown_cell("## 3) Basic Quality/Info Checks"))
    cells.append(nbf.v4.new_code_cell(r"""
def print_df_overview(name, df):
    print(f"\n== {name} | shape={df.shape} ==")
    print("dtypes:")
    print(df.dtypes)
    for col in ["first_seen_any", "last_seen_any", "first_seen", "last_seen", "seendate"]:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            print(f"  {col}: [{df[col].min()} .. {df[col].max()}]")
    n_dups = df.duplicated().sum()
    print("duplicates (full-row):", n_dups)
    n_na = df.isna().sum().sum()
    print("NA cells:", int(n_na))

print_df_overview("articles_df", articles_df)
print_df_overview("rel_df", rel_df)
if per_symbol_df is not None:
    print_df_overview("per_symbol_df", per_symbol_df)
"""))

    # Daily counts
    cells.append(nbf.v4.new_markdown_cell("## 4) Daily Counts (overall and by symbol)"))
    cells.append(nbf.v4.new_code_cell(r"""
# Overall daily counts from articles_df (based on first_seen_any)
if "first_seen_any" in articles_df.columns:
    adf = articles_df.copy()
    adf["day"] = pd.to_datetime(adf["first_seen_any"], utc=True).dt.date
    daily_all = adf.groupby("day").size().rename("n_articles").to_frame()
else:
    daily_all = pd.DataFrame()

# Per-symbol daily counts from rel_df (based on first_seen)
if {"symbol_req", "first_seen"}.issubset(rel_df.columns):
    rdf = rel_df.copy()
    rdf["day"] = pd.to_datetime(rdf["first_seen"], utc=True).dt.date
    daily_sym = rdf.groupby(["day", "symbol_req"]).size().rename("n_articles").reset_index()
else:
    daily_sym = pd.DataFrame()

display(daily_all.head())
display(daily_sym.head())
"""))

    # Plot overall daily
    cells.append(nbf.v4.new_markdown_cell("### 4.a) Plot — Overall daily article counts"))
    cells.append(nbf.v4.new_code_cell(r"""
if not daily_all.empty:
    s = daily_all["n_articles"]
    plt.figure(figsize=(12, 3.5))
    plt.plot(s.index, s.values)
    plt.title("Daily article counts (overall)")
    plt.xlabel("Day")
    plt.ylabel("#Articles")
    plt.tight_layout()
    if SAVE_PLOTS:
        os.makedirs(OUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUT_DIR, "daily_articles_overall.png"), dpi=150)
    plt.show()
"""))

    # Plot per-symbol daily
    cells.append(nbf.v4.new_markdown_cell("### 4.b) Plot — Daily article counts by symbol (subset)"))
    cells.append(nbf.v4.new_code_cell(r"""
if not daily_sym.empty:
    # Choose symbols to plot
    if not PLOT_SYMBOLS:
        top = (
            daily_sym.groupby("symbol_req")["n_articles"].sum()
            .sort_values(ascending=False)
            .head(4).index.tolist()
        )
    else:
        top = PLOT_SYMBOLS

    for sym in top:
        sub = daily_sym[daily_sym["symbol_req"] == sym]
        if sub.empty:
            continue
        s = sub.set_index("day")["n_articles"].sort_index()
        plt.figure(figsize=(12, 3))
        plt.plot(s.index, s.values)
        plt.title(f"Daily article counts — {sym}")
        plt.xlabel("Day")
        plt.ylabel("#Articles")
        plt.tight_layout()
        if SAVE_PLOTS:
            os.makedirs(OUT_DIR, exist_ok=True)
            plt.savefig(os.path.join(OUT_DIR, f"daily_articles_{sym}.png"), dpi=150)
        plt.show()
"""))

    # Top domains / languages
    cells.append(nbf.v4.new_markdown_cell("## 5) Top Domains / Languages"))
    cells.append(nbf.v4.new_code_cell(r"""
top_n = 20

# Top domains
if "domain" in articles_df.columns:
    dom = (
        articles_df["domain"].dropna().astype(str).str.strip().value_counts().head(top_n)
    )
    plt.figure(figsize=(10, max(3, 0.3*len(dom))))
    plt.barh(dom.index[::-1], dom.values[::-1])
    plt.title("Top domains (articles)")
    plt.xlabel("#Articles")
    plt.tight_layout()
    if SAVE_PLOTS:
        os.makedirs(OUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUT_DIR, "top_domains.png"), dpi=150)
    plt.show()

# Languages
if "language" in articles_df.columns:
    lang = (
        articles_df["language"].dropna().astype(str).str.strip().value_counts().head(top_n)
    )
    plt.figure(figsize=(8, max(3, 0.3*len(lang))))
    plt.barh(lang.index[::-1], lang.values[::-1])
    plt.title("Top languages (articles)")
    plt.xlabel("#Articles")
    plt.tight_layout()
    if SAVE_PLOTS:
        os.makedirs(OUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUT_DIR, "top_languages.png"), dpi=150)
    plt.show()
"""))

    # Optional exports
    cells.append(nbf.v4.new_markdown_cell("## 6) (Optional) Export compact CSVs"))
    cells.append(nbf.v4.new_code_cell(r"""
# Save daily count tables for quick comparison outside notebook
if not daily_all.empty:
    os.makedirs(OUT_DIR, exist_ok=True)
    daily_all.to_csv(os.path.join(OUT_DIR, "daily_articles_overall.csv"))
if not daily_sym.empty:
    os.makedirs(OUT_DIR, exist_ok=True)
    daily_sym.to_csv(os.path.join(OUT_DIR, "daily_articles_by_symbol.csv"), index=False)
print("Done.")
"""))

    nb["cells"] = cells
    out_path = os.path.abspath("./news_examine.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[OK] Wrote notebook to: {out_path}")

if __name__ == "__main__":
    main()
