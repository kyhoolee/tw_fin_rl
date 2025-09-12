#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run baseline families on a split parquet and export:
- curves.csv (NAV per strategy, aligned by timestamp)
- metrics.csv (CAGR, Sharpe, MaxDD, Sortino, Calmar, Turnover approx if available)

Usage:
  python -m fin_rl.portfolio.run_baselines \
    --split ./work/.../splits/test.parquet \
    --fee-cfg fin_rl/portfolio/fees.yaml --fee-name binance_taker_0075 \
    --out ./work/bundle_out/reports/baseline_v1
"""
import argparse, os
import numpy as np, pandas as pd
from pathlib import Path
from ..utils import load_fee_cfg, perf_metrics
from ._0_portfolio_baseline import run_all_baselines as run_core
from ._0_portfolio_baseline_ext import run_ext_baselines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--fee-cfg", required=True)
    ap.add_argument("--fee-name", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("== Running core baselines ==")

    fee_bps, slip_bps = load_fee_cfg(args.fee_cfg, args.fee_name)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[main] out_dir: {out_dir}")
    print(f"[main] fee_bps={fee_bps}, slippage_bps={slip_bps}")

    # core
    print("== Running core baselines ==")
    syms, curves_core, metrics_core = run_core(args.split, fee_bps=fee_bps, slippage_bps=slip_bps)
    print("== Core baselines done ==")
    
    # ext
    print("== Running extended baselines ==")
    syms2, ts, curves_ext = run_ext_baselines(args.split, fee_bps=fee_bps, slippage_bps=slip_bps)
    print("== Extended baselines done ==")
    # merge curves
    curves = {**curves_core, **curves_ext}
    df_curves = pd.DataFrame({k: v for k, v in curves.items()}, index=pd.to_datetime(ts))
    df_curves.index.name = "open_time"
    df_curves.to_csv(out_dir / "curves.csv")

    # metrics
    metrics = {}
    for k, nav in curves.items():
        metrics[k] = perf_metrics(nav, dt_index=ts)
    df_metrics = pd.DataFrame(metrics).T
    df_metrics.to_csv(out_dir / "metrics.csv")

    # save a small README
    with open(out_dir / "README.txt", "w") as f:
        f.write(f"symbols: {','.join(syms)}\n")
        f.write(f"fees: fee_bps={fee_bps}, slippage_bps={slip_bps}\n")
        f.write("curves.csv: NAV per baseline\nmetrics.csv: summary metrics\n")

    print("== Done ==")
    print("symbols:", syms)
    print(df_metrics.sort_values("Sharpe≈", ascending=False).round(4).to_string())

if __name__ == "__main__":
    main()
