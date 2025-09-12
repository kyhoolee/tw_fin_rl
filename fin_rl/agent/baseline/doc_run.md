# 1) (đã có) build bundle & verify theo hướng dẫn ở trên

# 2) Chạy baseline core + ext với preset phí
python -m fin_rl.agent.baseline._0_1_portfolio_verify \
  --split ./work/bundle_out/bundles/v1_minutes/bundle_01/splits/test.parquet \
  --fee-cfg fin_rl/agent/baseline/_0_config.yaml --fee-name binance_taker_0075 \
  --out ./work/bundle_out/reports/baseline_v1

# 3) (tuỳ chọn) sweep theo heuristics.rebalance_sweep_bars trong fees.yaml
# gợi ý: tạo script nhỏ loop rebalance_every ∈ {60,120,240,720,1440} cho EW/MOM
