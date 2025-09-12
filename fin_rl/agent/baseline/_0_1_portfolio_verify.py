from ._0_portfolio_baseline import run_all_baselines
from ..utils import load_fee_cfg

fee_bps, slip_bps = load_fee_cfg("fin_rl/portfolio/fees.yaml", "binance_taker_0075")
syms, curves, metrics = run_all_baselines(
    "./work/bundle_out/bundles/v1_minutes/bundle_01/splits/test.parquet",
    fee_bps=fee_bps, slippage_bps=slip_bps,
)
print("symbols:", syms)
for name, m in metrics.items():
    print(name, m)