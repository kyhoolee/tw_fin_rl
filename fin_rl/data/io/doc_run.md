## NEWS

python -m fin_rl.data.io.news_io \
  --news-dir work/data/gdelt_0024_hardfix_split/normalized_unique


## CANDLE

python -m fin_rl.data.io.candle_io \
  --data-dir ./work/data/binance/spot/1m \
  --candle-level 1m \
  --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT \
  --save-plots
