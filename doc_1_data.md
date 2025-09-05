python -m  fin_rl.data.crypto.binance \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT \
  --start 2019-01-01 --end 2025-09-04 \
  --interval 1d \
  --out-dir ./work/data/binance 


python -m  fin_rl.data.crypto.binance \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT \
  --start 2019-01-01 --end 2025-09-04 \
  --interval 1m \
  --out-dir ./work/data/binance 

python -m  fin_rl.data.crypto.binance \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT \
  --start 2019-01-01 --end 2025-09-04 \
  --interval 1h \
  --out-dir ./work/data/binance 

tmux new -s binance_fetch
tmux attach -t binance_fetch



===============

## Trend signal 

python -m fin_rl.data.crypto.gg_trend \
  --start 2019-01-01 \
  --end   2025-09-01 \
  --out-dir ./work/data/proxy/google_trends \
  --sleep-min 5 --sleep-max 10 \
  --window-days 90 \
  --log INFO


python -m fin_rl.data.crypto.gg_trend --test --log DEBUG


python -m fin_rl.data.crypto.gdelt_demo \
  --symbols BTC,ETH,BNB,SOL \
  --days 7 \
  --step-hours 24 \
  --langs english \
  --out ./work/data/gdelt


python -m fin_rl.data.crypto.gdelt_demo \
  --symbols BTC,ETH,BNB,SOL \
  --days 14 \
  --step-hours 12 \
  --langs english \
  --out ./work/data/gdelt_v2 \
  --mode balanced

python -m fin_rl.data.crypto.gdelt_demo \
  --out ./work/data/gdelt_align_hard \
  --mode balanced \
  --step-hours 6 \
  --hardcoded

python -m fin_rl.data.crypto.gdelt_demo \
  --hardcoded \
  --mode balanced \
  --step-hours 6 \
  --progress-kind log \
  --progress-every 10 \
  --progress-min-interval 2



