Using data_dir = data

[BTC] ==== PIPELINE START ====
[BTC] Step 1/4 LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[BTC] Step 3/5 FEATURES+LABEL: windows=(3, 6, 12, 24) mode=three_class H=4 eps=0.001
[BTC] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[BTC] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[BTC] add_features: generated 67 features; total NaNs after lag=787
[BTC] make_label(3-class): H=4 eps=0.001 counts={1: 7852, -1: 7292, 0: 2424}
[BTC] prepare_Xy: dropped NaNs -> rows 17568 → 17518; features=67
[BTC] Step 4/5 TRAIN: TSCV n_splits=5, C=0.5, max_iter=2000
[BTC] train: fold 1/5 | train=2923 test=2919
[BTC] train: fold 1 metrics | AUC=0.5464 ACC=0.4460
[BTC] train: fold 2/5 | train=5842 test=2919
[BTC] train: fold 2 metrics | AUC=0.5460 ACC=0.4669
[BTC] train: fold 3/5 | train=8761 test=2919
[BTC] train: fold 3 metrics | AUC=0.5726 ACC=0.4502
[BTC] train: fold 4/5 | train=11680 test=2919
[BTC] train: fold 4 metrics | AUC=0.5579 ACC=0.4457
[BTC] train: fold 5/5 | train=14599 test=2919
[BTC] train: fold 5 metrics | AUC=0.5436 ACC=0.4104
[BTC] train: OOF metrics | AUC=nan ACC=0.3929
[BTC] FINAL METRICS: AUC=nan ACC=0.3929
              precision    recall  f1-score   support

          -1      0.428     0.406     0.417      7268
           0      0.141     0.175     0.156      2422
           1      0.460     0.448     0.454      7828

    accuracy                          0.393     17518
   macro avg      0.343     0.343     0.342     17518
weighted avg      0.403     0.393     0.397     17518

Confusion matrix:
 [[2954 1216 3098]
 [ 980  424 1018]
 [2964 1359 3505]]
[BTC] Step 5/5 BACKTEST: t_long=0.6 t_short=0.6 fee_bps=5.0 min_hold=1
[BTC] BACKTEST: CAGR=0.1266 Sharpe=1.09 HitRate=0.33

[BNB] ==== PIPELINE START ====
[BNB] Step 1/4 LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[BNB] Step 3/5 FEATURES+LABEL: windows=(3, 6, 12, 24) mode=three_class H=4 eps=0.001
[BNB] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[BNB] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[BNB] add_features: generated 67 features; total NaNs after lag=787
[BNB] make_label(3-class): H=4 eps=0.001 counts={1: 8355, -1: 7365, 0: 1848}
[BNB] prepare_Xy: dropped NaNs -> rows 17568 → 17518; features=67
[BNB] Step 4/5 TRAIN: TSCV n_splits=5, C=0.5, max_iter=2000
[BNB] train: fold 1/5 | train=2923 test=2919
[BNB] train: fold 1 metrics | AUC=0.5536 ACC=0.4440
[BNB] train: fold 2/5 | train=5842 test=2919
[BNB] train: fold 2 metrics | AUC=0.5496 ACC=0.4673
[BNB] train: fold 3/5 | train=8761 test=2919
[BNB] train: fold 3 metrics | AUC=0.5573 ACC=0.4721
[BNB] train: fold 4/5 | train=11680 test=2919
[BNB] train: fold 4 metrics | AUC=0.5409 ACC=0.4423
[BNB] train: fold 5/5 | train=14599 test=2919
[BNB] train: fold 5 metrics | AUC=0.5653 ACC=0.4190
[BNB] train: OOF metrics | AUC=nan ACC=0.3914
[BNB] FINAL METRICS: AUC=nan ACC=0.3914
              precision    recall  f1-score   support

          -1      0.429     0.511     0.466      7344
           0      0.104     0.165     0.128      1837
           1      0.479     0.336     0.395      8337

    accuracy                          0.391     17518
   macro avg      0.337     0.337     0.330     17518
weighted avg      0.419     0.391     0.397     17518

Confusion matrix:
 [[3753 1180 2411]
 [ 903  304  630]
 [4099 1439 2799]]
[BNB] Step 5/5 BACKTEST: t_long=0.6 t_short=0.6 fee_bps=5.0 min_hold=1
[BNB] BACKTEST: CAGR=0.1721 Sharpe=0.85 HitRate=0.42

[SOL] ==== PIPELINE START ====
[SOL] Step 1/4 LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[SOL] Step 3/5 FEATURES+LABEL: windows=(3, 6, 12, 24) mode=three_class H=4 eps=0.001
[SOL] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[SOL] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[SOL] add_features: generated 67 features; total NaNs after lag=787
[SOL] make_label(3-class): H=4 eps=0.001 counts={1: 8449, -1: 8106, 0: 1013}
[SOL] prepare_Xy: dropped NaNs -> rows 17568 → 17518; features=67
[SOL] Step 4/5 TRAIN: TSCV n_splits=5, C=0.5, max_iter=2000
[SOL] train: fold 1/5 | train=2923 test=2919
[SOL] train: fold 1 metrics | AUC=0.5375 ACC=0.4734
[SOL] train: fold 2/5 | train=5842 test=2919
[SOL] train: fold 2 metrics | AUC=0.5083 ACC=0.4669
[SOL] train: fold 3/5 | train=8761 test=2919
[SOL] train: fold 3 metrics | AUC=0.5167 ACC=0.4902
[SOL] train: fold 4/5 | train=11680 test=2919
[SOL] train: fold 4 metrics | AUC=0.5469 ACC=0.4913
[SOL] train: fold 5/5 | train=14599 test=2919
[SOL] train: fold 5 metrics | AUC=0.5399 ACC=0.4782
[SOL] train: OOF metrics | AUC=nan ACC=0.4072
[SOL] FINAL METRICS: AUC=nan ACC=0.4072
              precision    recall  f1-score   support

          -1      0.473     0.487     0.479      8092
           0      0.044     0.127     0.065      1008
           1      0.490     0.365     0.418      8418

    accuracy                          0.407     17518
   macro avg      0.336     0.326     0.321     17518
weighted avg      0.456     0.407     0.426     17518

Confusion matrix:
 [[3937 1324 2831]
 [ 521  128  359]
 [3873 1476 3069]]
[SOL] Step 5/5 BACKTEST: t_long=0.6 t_short=0.6 fee_bps=5.0 min_hold=1
[SOL] BACKTEST: CAGR=0.3522 Sharpe=1.30 HitRate=0.33

[ETH] ==== PIPELINE START ====
[ETH] Step 1/4 LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[ETH] Step 3/5 FEATURES+LABEL: windows=(3, 6, 12, 24) mode=three_class H=4 eps=0.001
[ETH] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[ETH] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[ETH] add_features: generated 67 features; total NaNs after lag=787
[ETH] make_label(3-class): H=4 eps=0.001 counts={1: 8182, -1: 7748, 0: 1638}
[ETH] prepare_Xy: dropped NaNs -> rows 17568 → 17518; features=67
[ETH] Step 4/5 TRAIN: TSCV n_splits=5, C=0.5, max_iter=2000
[ETH] train: fold 1/5 | train=2923 test=2919
[ETH] train: fold 1 metrics | AUC=0.5403 ACC=0.4659
[ETH] train: fold 2/5 | train=5842 test=2919
[ETH] train: fold 2 metrics | AUC=0.5532 ACC=0.4772
[ETH] train: fold 3/5 | train=8761 test=2919
[ETH] train: fold 3 metrics | AUC=0.5447 ACC=0.4800
[ETH] train: fold 4/5 | train=11680 test=2919
[ETH] train: fold 4 metrics | AUC=0.5511 ACC=0.4800
[ETH] train: fold 5/5 | train=14599 test=2919
[ETH] train: fold 5 metrics | AUC=0.5604 ACC=0.4419
[ETH] train: OOF metrics | AUC=nan ACC=0.4097
[ETH] FINAL METRICS: AUC=nan ACC=0.4097
              precision    recall  f1-score   support

          -1      0.458     0.520     0.487      7728
           0      0.114     0.204     0.146      1633
           1      0.486     0.346     0.404      8157

    accuracy                          0.410     17518
   macro avg      0.353     0.357     0.346     17518
weighted avg      0.439     0.410     0.417     17518

Confusion matrix:
 [[4020 1207 2501]
 [ 813  333  487]
 [3947 1385 2825]]
[ETH] Step 5/5 BACKTEST: t_long=0.6 t_short=0.6 fee_bps=5.0 min_hold=1
[ETH] BACKTEST: CAGR=0.3947 Sharpe=1.95 HitRate=0.38

[AVAX] ==== PIPELINE START ====
[AVAX] Step 1/4 LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[AVAX] Step 3/5 FEATURES+LABEL: windows=(3, 6, 12, 24) mode=three_class H=4 eps=0.001
[AVAX] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[AVAX] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[AVAX] add_features: generated 67 features; total NaNs after lag=787
[AVAX] make_label(3-class): H=4 eps=0.001 counts={-1: 8449, 1: 8139, 0: 980}
[AVAX] prepare_Xy: dropped NaNs -> rows 17568 → 17518; features=67
[AVAX] Step 4/5 TRAIN: TSCV n_splits=5, C=0.5, max_iter=2000
[AVAX] train: fold 1/5 | train=2923 test=2919
[AVAX] train: fold 1 metrics | AUC=0.5260 ACC=0.4734
[AVAX] train: fold 2/5 | train=5842 test=2919
[AVAX] train: fold 2 metrics | AUC=0.5312 ACC=0.4865
[AVAX] train: fold 3/5 | train=8761 test=2919
[AVAX] train: fold 3 metrics | AUC=0.5263 ACC=0.4967
[AVAX] train: fold 4/5 | train=11680 test=2919
[AVAX] train: fold 4 metrics | AUC=0.5185 ACC=0.4782
[AVAX] train: fold 5/5 | train=14599 test=2919
[AVAX] train: fold 5 metrics | AUC=0.5062 ACC=0.4765
[AVAX] train: OOF metrics | AUC=nan ACC=0.4093
[AVAX] FINAL METRICS: AUC=nan ACC=0.4093
              precision    recall  f1-score   support

          -1      0.502     0.356     0.416      8421
           0      0.045     0.134     0.067       978
           1      0.469     0.498     0.483      8119

    accuracy                          0.409     17518
   macro avg      0.339     0.329     0.322     17518
weighted avg      0.461     0.409     0.428     17518

Confusion matrix:
 [[2996 1360 4065]
 [ 328  131  519]
 [2643 1433 4043]]
[AVAX] Step 5/5 BACKTEST: t_long=0.6 t_short=0.6 fee_bps=5.0 min_hold=1
[AVAX] BACKTEST: CAGR=0.4017 Sharpe=1.42 HitRate=0.38

[TRON] ==== PIPELINE START ====
[TRON] Step 1/4 LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[TRON] Step 3/5 FEATURES+LABEL: windows=(3, 6, 12, 24) mode=three_class H=4 eps=0.001
[TRON] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[TRON] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[TRON] add_features: generated 67 features; total NaNs after lag=787
[TRON] make_label(3-class): H=4 eps=0.001 counts={1: 8152, -1: 6756, 0: 2660}
[TRON] prepare_Xy: dropped NaNs -> rows 17568 → 17518; features=67
[TRON] Step 4/5 TRAIN: TSCV n_splits=5, C=0.5, max_iter=2000
[TRON] train: fold 1/5 | train=2923 test=2919
[TRON] train: fold 1 metrics | AUC=0.5486 ACC=0.4457
[TRON] train: fold 2/5 | train=5842 test=2919
[TRON] train: fold 2 metrics | AUC=0.5429 ACC=0.3813
[TRON] train: fold 3/5 | train=8761 test=2919
[TRON] train: fold 3 metrics | AUC=0.5467 ACC=0.4471
[TRON] train: fold 4/5 | train=11680 test=2919
[TRON] train: fold 4 metrics | AUC=0.5178 ACC=0.4430
[TRON] train: fold 5/5 | train=14599 test=2919
[TRON] train: fold 5 metrics | AUC=0.5400 ACC=0.3991
[TRON] train: OOF metrics | AUC=nan ACC=0.3784
[TRON] FINAL METRICS: AUC=nan ACC=0.3784
              precision    recall  f1-score   support

          -1      0.399     0.447     0.422      6747
           0      0.174     0.243     0.203      2649
           1      0.476     0.365     0.413      8122

    accuracy                          0.378     17518
   macro avg      0.349     0.352     0.346     17518
weighted avg      0.400     0.378     0.385     17518

Confusion matrix:
 [[3017 1390 2340]
 [1074  645  930]
 [3474 1681 2967]]
[TRON] Step 5/5 BACKTEST: t_long=0.6 t_short=0.6 fee_bps=5.0 min_hold=1
[TRON] BACKTEST: CAGR=0.2669 Sharpe=1.13 HitRate=0.41

[ARBtrium] ==== PIPELINE START ====
[ARBtrium] Step 1/4 LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[ARBtrium] Step 3/5 FEATURES+LABEL: windows=(3, 6, 12, 24) mode=three_class H=4 eps=0.001
[ARBtrium] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[ARBtrium] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[ARBtrium] add_features: generated 67 features; total NaNs after lag=787
[ARBtrium] make_label(3-class): H=4 eps=0.001 counts={-1: 8387, 1: 8222, 0: 959}
[ARBtrium] prepare_Xy: dropped NaNs -> rows 17568 → 17518; features=67
[ARBtrium] Step 4/5 TRAIN: TSCV n_splits=5, C=0.5, max_iter=2000
[ARBtrium] train: fold 1/5 | train=2923 test=2919
[ARBtrium] train: fold 1 metrics | AUC=0.5328 ACC=0.4676
[ARBtrium] train: fold 2/5 | train=5842 test=2919
[ARBtrium] train: fold 2 metrics | AUC=0.5368 ACC=0.4961
[ARBtrium] train: fold 3/5 | train=8761 test=2919
[ARBtrium] train: fold 3 metrics | AUC=0.5645 ACC=0.4967
[ARBtrium] train: fold 4/5 | train=11680 test=2919
[ARBtrium] train: fold 4 metrics | AUC=0.5123 ACC=0.4752
[ARBtrium] train: fold 5/5 | train=14599 test=2919
[ARBtrium] train: fold 5 metrics | AUC=0.5087 ACC=0.4841
[ARBtrium] train: OOF metrics | AUC=nan ACC=0.4117
[ARBtrium] FINAL METRICS: AUC=nan ACC=0.4117
              precision    recall  f1-score   support

          -1      0.502     0.339     0.405      8359
           0      0.051     0.156     0.077       956
           1      0.473     0.515     0.493      8203

    accuracy                          0.412     17518
   macro avg      0.342     0.337     0.325     17518
weighted avg      0.463     0.412     0.428     17518

Confusion matrix:
 [[2837 1312 4210]
 [ 304  149  503]
 [2515 1462 4226]]
[ARBtrium] Step 5/5 BACKTEST: t_long=0.6 t_short=0.6 fee_bps=5.0 min_hold=1
[ARBtrium] BACKTEST: CAGR=0.3798 Sharpe=1.22 HitRate=0.39

[Sui] ==== PIPELINE START ====
[Sui] Step 1/4 LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[Sui] Step 3/5 FEATURES+LABEL: windows=(3, 6, 12, 24) mode=three_class H=4 eps=0.001
[Sui] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[Sui] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[Sui] add_features: generated 67 features; total NaNs after lag=787
[Sui] make_label(3-class): H=4 eps=0.001 counts={-1: 8478, 1: 8313, 0: 777}
[Sui] prepare_Xy: dropped NaNs -> rows 17568 → 17518; features=67
[Sui] Step 4/5 TRAIN: TSCV n_splits=5, C=0.5, max_iter=2000
[Sui] train: fold 1/5 | train=2923 test=2919
[Sui] train: fold 1 metrics | AUC=0.5312 ACC=0.5053
[Sui] train: fold 2/5 | train=5842 test=2919
[Sui] train: fold 2 metrics | AUC=0.5365 ACC=0.5067
[Sui] train: fold 3/5 | train=8761 test=2919
[Sui] train: fold 3 metrics | AUC=0.5683 ACC=0.5344
[Sui] train: fold 4/5 | train=11680 test=2919
[Sui] train: fold 4 metrics | AUC=0.5348 ACC=0.4810
[Sui] train: fold 5/5 | train=14599 test=2919
[Sui] train: fold 5 metrics | AUC=0.5272 ACC=0.4776
[Sui] train: OOF metrics | AUC=nan ACC=0.4227
[Sui] FINAL METRICS: AUC=nan ACC=0.4227
              precision    recall  f1-score   support

          -1      0.497     0.661     0.567      8451
           0      0.031     0.119     0.050       774
           1      0.515     0.208     0.297      8293

    accuracy                          0.423     17518
   macro avg      0.348     0.329     0.305     17518
weighted avg      0.485     0.423     0.416     17518

Confusion matrix:
 [[5585 1373 1493]
 [ 547   92  135]
 [5106 1460 1727]]
[Sui] Step 5/5 BACKTEST: t_long=0.6 t_short=0.6 fee_bps=5.0 min_hold=1
[Sui] BACKTEST: CAGR=0.9118 Sharpe=2.09 HitRate=0.44

[Polygon] ==== PIPELINE START ====
[Polygon] Step 1/4 LOAD: rows=7587, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2024-09-10 02:59:59.999000+00:00, cols_req_present=True
[Polygon] Step 3/5 FEATURES+LABEL: windows=(3, 6, 12, 24) mode=three_class H=4 eps=0.001
[Polygon] prepare_Xy: input rows=7587 [2023-10-30 00:59:59.999000+00:00 → 2024-09-10 02:59:59.999000+00:00]
[Polygon] add_features: start, rows=7587, windows=(3, 6, 12, 24)
[Polygon] add_features: generated 67 features; total NaNs after lag=787
[Polygon] make_label(3-class): H=4 eps=0.001 counts={1: 3616, -1: 3456, 0: 515}
[Polygon] prepare_Xy: dropped NaNs -> rows 7587 → 7537; features=67
[Polygon] Step 4/5 TRAIN: TSCV n_splits=5, C=0.5, max_iter=2000
[Polygon] train: fold 1/5 | train=1257 test=1256
[Polygon] train: fold 1 metrics | AUC=0.5472 ACC=0.5032
[Polygon] train: fold 2/5 | train=2513 test=1256
[Polygon] train: fold 2 metrics | AUC=0.5322 ACC=0.4546
[Polygon] train: fold 3/5 | train=3769 test=1256
[Polygon] train: fold 3 metrics | AUC=0.5290 ACC=0.4658
[Polygon] train: fold 4/5 | train=5025 test=1256
[Polygon] train: fold 4 metrics | AUC=0.5175 ACC=0.4323
[Polygon] train: fold 5/5 | train=6281 test=1256
[Polygon] train: fold 5 metrics | AUC=0.5117 ACC=0.4498
[Polygon] train: OOF metrics | AUC=nan ACC=0.3934
[Polygon] FINAL METRICS: AUC=nan ACC=0.3934
              precision    recall  f1-score   support

          -1      0.460     0.361     0.404      3431
           0      0.055     0.136     0.079       515
           1      0.462     0.462     0.462      3591

    accuracy                          0.393      7537
   macro avg      0.326     0.319     0.315      7537
weighted avg      0.434     0.393     0.410      7537

Confusion matrix:
 [[1237  545 1649]
 [ 167   70  278]
 [1283  650 1658]]
[Polygon] Step 5/5 BACKTEST: t_long=0.6 t_short=0.6 fee_bps=5.0 min_hold=1
[Polygon] BACKTEST: CAGR=0.1560 Sharpe=0.73 HitRate=0.35
+----------+-----------+-------+-----+----------+--------------+-----------+-------+--------+---------+----------------------------------+----------------------------------+
|  Token   |   Symbol  |  Rows | AUC | Accuracy | Precision(1) | Recall(1) |  CAGR | Sharpe | HitRate |              Start               |               End                |
+----------+-----------+-------+-----+----------+--------------+-----------+-------+--------+---------+----------------------------------+----------------------------------+
|   BTC    |  BTCUSDT  | 17518 | nan |  0.393   |    0.460     |   0.448   | 0.127 |  1.09  |   0.33  | 2023-11-01 02:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
|   BNB    |  BNBUSDT  | 17518 | nan |  0.391   |    0.479     |   0.336   | 0.172 |  0.85  |   0.42  | 2023-11-01 02:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
|   SOL    |  SOLUSDT  | 17518 | nan |  0.407   |    0.490     |   0.365   | 0.352 |  1.30  |   0.33  | 2023-11-01 02:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
|   ETH    |  ETHUSDT  | 17518 | nan |  0.410   |    0.486     |   0.346   | 0.395 |  1.95  |   0.38  | 2023-11-01 02:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
|   AVAX   |  AVAXUSDT | 17518 | nan |  0.409   |    0.469     |   0.498   | 0.402 |  1.42  |   0.38  | 2023-11-01 02:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
|   TRON   |  TRXUSDT  | 17518 | nan |  0.378   |    0.476     |   0.365   | 0.267 |  1.13  |   0.41  | 2023-11-01 02:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
| ARBtrium |  ARBUSDT  | 17518 | nan |  0.412   |    0.473     |   0.515   | 0.380 |  1.22  |   0.39  | 2023-11-01 02:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
|   Sui    |  SUIUSDT  | 17518 | nan |  0.423   |    0.515     |   0.208   | 0.912 |  2.09  |   0.44  | 2023-11-01 02:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
| Polygon  | MATICUSDT |  7537 | nan |  0.393   |    0.462     |   0.462   | 0.156 |  0.73  |   0.35  | 2023-11-01 02:59:59.999000+00:00 | 2024-09-10 02:59:59.999000+00:00 |
+----------+-----------+-------+-----+----------+--------------+-----------+-------+--------+---------+----------------------------------+----------------------------------+