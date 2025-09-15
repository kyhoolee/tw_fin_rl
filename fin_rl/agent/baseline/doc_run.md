# 1) (đã có) build bundle & verify theo hướng dẫn ở trên

# 2) Chạy baseline core + ext với preset phí
python -m fin_rl.agent.baseline._0_1_portfolio_verify \
  --split ./work/bundle_out/bundles/v1_minutes/bundle_01/splits/test.parquet \
  --fee-cfg fin_rl/agent/baseline/_0_config.yaml --fee-name binance_taker_0075 \
  --out ./work/bundle_out/reports/baseline_v1


```
(cleanrl) kylh@educa:~/phd/tw_fin_rl$ python -m fin_rl.agent.baseline._0_1_portfolio_verify   --split ./work/bundle_out/bundles/v1_minutes/bundle_01/splits/test.parquet   --fee-cfg fin_rl/agent/baseline/_0_config.yaml --fee-name binance_taker_0075   --out ./work/bundle_out/reports/baseline_v1
== Running core baselines ==
[main] out_dir: work/bundle_out/reports/baseline_v1
[main] fee_bps=7.5, slippage_bps=0.0
== Running core baselines ==
[run_all_baselines] split=./work/bundle_out/bundles/v1_minutes/bundle_01/splits/test.parquet
[run_all_baselines] fees: fee_bps=7.5, slippage_bps=0.0
>> loading data ...
>> data loaded: T,N,F = (525601, 4, 1)
>> running EW_daily ...
EW_daily: 100%|█████████████████████████████████████████████████████████████████████████████████| 525600/525600 [00:02<00:00, 253221.07it/s]
>> running MOM_top2_look360_rb60 ...
MOM_top2_look360_rb60: 100%|████████████████████████████████████████████████████████████████████| 525600/525600 [00:02<00:00, 240673.11it/s]
>> running PAIR_MR_BTC_ETH ...
PAIR_MR_BTC_ETH: 100%|████████████████████████████████████████████████████████████████████████████| 525600/525600 [02:24<00:00, 3637.73it/s]
== Core baselines done ==
== Running extended baselines ==
[run_ext_baselines] split=./work/bundle_out/bundles/v1_minutes/bundle_01/splits/test.parquet
[run_ext_baselines] fees: fee_bps=7.5, slippage_bps=0.0
>> loading data ...
== Running extended baselines ==
>> running EW_VolTarget_hl1440 ...
EW_VolTarget: 100%|█████████████████████████████████████████████████████████████████████████████| 525600/525600 [00:02<00:00, 238324.73it/s]
>> running TS_Breakout_240 ...
TS_Breakout: 100%|██████████████████████████████████████████████████████████████████████████████| 525600/525600 [00:02<00:00, 240451.97it/s]
>> running RiskParity_MinVar_720 ...
RiskParity_MinVar: 100%|████████████████████████████████████████████████████████████████████████| 525600/525600 [00:02<00:00, 201257.56it/s]
>> running MOM_Top2_Turnover50%_60 ...
MOM_TurnoverCapped: 100%|███████████████████████████████████████████████████████████████████████| 525600/525600 [00:02<00:00, 236229.13it/s]
== Extended baselines done ==
== Done ==
symbols: ['BNBUSDT', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT']
                          CAGR≈  Sharpe≈  Sortino≈   MaxDD  Calmar≈
EW_daily                 1.0579   1.4739    1.8980  0.3817   2.7714
EW_VolTarget_hl1440      1.8482   1.4648    1.8861  0.6491   2.8473
RiskParity_MinVar_720    0.2759   0.7215    0.9321  0.4910   0.5620
MOM_Top2_Turnover50%_60 -0.6868  -1.4883   -1.9527  0.7755  -0.8856
MOM_top2_look360_rb60   -0.8564  -2.6677   -3.5070  0.8711  -0.9831
TS_Breakout_240         -0.9108 -11.2004   -4.4121  0.9119  -0.9987
PAIR_MR_BTC_ETH         -0.9998 -49.2587  -43.3233  0.9998  -1.0000
(cleanrl) kylh@educa:~/phd/tw_fin_rl$ 

```




# 3) (tuỳ chọn) sweep theo heuristics.rebalance_sweep_bars trong fees.yaml
# gợi ý: tạo script nhỏ loop rebalance_every ∈ {60,120,240,720,1440} cho EW/MOM
