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

tmux new -s binance_fetch
tmux attach -t binance_fetch
