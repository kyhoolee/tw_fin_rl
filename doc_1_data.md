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



python -m fin_rl.data.crypto.gdelt_hardfix \
  --out ./work/data/gdelt_fast_hardfix \
  --langs english \
  --window 09:00-12:00 \
  --sleep-sec 1.0

nohup python -u -m fin_rl.data.crypto.gdelt_hardfix \
  --out ./work/data/gdelt_fast_hardfix \
  --langs english \
  --window 09:00-12:00 \
  --sleep-sec 1.0 \
  > gdelt_hardfix.log 2>&1 &


nohup python -m fin_rl.data.crypto.gdelt_hardfix \
  --start 2024-01-01 --end 2024-01-10 \
  --window 09:00-12:00 \
  --langs english \
  --sleep-sec 1.0 \
  --verbose \
  --debug-dump-dir ./work/data/gdelt_fast_hardfix/_debug_raw \
  --out ./work/data/gdelt_fast_hardfix \
  > gdelt_hardfix.log 2>&1 &


  First date : 2020-08-11
  Last date  : 2025-09-04

nohup python -m fin_rl.data.crypto.gdelt_hardfix \
  --start 2020-08-11 --end 2025-09-04 \
  --window 09:00-12:00 \
  --langs english \
  --sleep-sec 0.5 \
  --sleep-between-req 0.5 \
  --verbose \
  --out ./work/data/gdelt_full_hardfix_split \
  > gdelt_full_split.log 2>&1 &

python -m fin_rl.data.crypto.gdelt_check --base-dir ./work/data/gdelt_fast_hardfix_split --show-samples 5



nohup python -m fin_rl.data.crypto.gdelt_hardfix \
  --start 2020-08-11 --end 2025-09-04 \
  --window 09:00-21:00 \
  --langs english \
  --sleep-sec 0.5 \
  --sleep-between-req 0.5 \
  --verbose \
  --out ./work/data/gdelt_0921_hardfix_split \
  > gdelt_0921_split.log 2>&1 &



nohup python -m fin_rl.data.crypto.gdelt_hardfix \
  --start 2025-09-01 --end 2025-09-04 \
  --window 06:00-23:00 \
  --langs english \
  --sleep-sec 0.5 \
  --sleep-between-req 0.5 \
  --verbose \
  --out ./work/data/gdelt_debug_hardfix_split \
  > gdelt_debug_split.log 2>&1 &


nohup python -m fin_rl.data.crypto.gdelt_hardfix \
  --start 2020-08-11 --end 2025-09-04 \
  --window 00:00-23:59 \
  --langs english \
  --sleep-sec 0.5 \
  --sleep-between-req 0.5 \
  --verbose \
  --out ./work/data/gdelt_0024_hardfix_split \
  > gdelt_0024_split.log 2>&1 &