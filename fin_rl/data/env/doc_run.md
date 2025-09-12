## Split backtest data

python -m fin_rl.data.env._0_bundle_builder_raw_sync.py \
  --data-dir "../../work/data/binance/spot/1m" \
  --candle-level 1m \
  --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT \
  --train-start "2020-01-01" --train-end "2023-12-31" \
  --test-start  "2024-01-01" --test-end  "2024-12-31" \
  --unseen-start "2025-01-01" --unseen-end "2025-06-30" \
  --embargo "12h" \
  --out-dir "./bundle_out/bundles/v1_minutes/bundle_01"
