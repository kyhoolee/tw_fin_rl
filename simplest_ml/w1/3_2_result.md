Using data_dir = data

[BTC] ==== PIPELINE START (v3) ====
[BTC] LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[BTC] FEATURES+LABEL: windows=(3, 6, 12, 24) H=4 eps=0.001
[BTC] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[BTC] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[BTC] add_features: generated 67 features; total NaNs after lag=787
[BTC] make_label(3-class): H=4 eps=0.001 counts={1: 7852, -1: 7292, 0: 2424}
[BTC] prepare_Xy: dropped NaNs -> rows 17568 → 17518; features=67
[BTC] TRAIN: TSCV n_splits=5, C=0.5
[BTC] train: fold 1/5 | train=2923 test=2919
[BTC] train: fold 1 metrics | macroAUC=0.5464 ACC=0.4460
[BTC] train: fold 2/5 | train=5842 test=2919
[BTC] train: fold 2 metrics | macroAUC=0.5460 ACC=0.4669
[BTC] train: fold 3/5 | train=8761 test=2919
[BTC] train: fold 3 metrics | macroAUC=0.5726 ACC=0.4502
[BTC] train: fold 4/5 | train=11680 test=2919
[BTC] train: fold 4 metrics | macroAUC=0.5579 ACC=0.4457
[BTC] train: fold 5/5 | train=14599 test=2919
[BTC] train: fold 5 metrics | macroAUC=0.5436 ACC=0.4104
[BTC] train: OOF metrics | macroAUC=0.5356 ACC=0.3929
[BTC] OOF macroAUC=0.5356 ACC=0.3929
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
[BTC] TUNED THRESHOLDS: t_long=0.55 t_short=0.55 | f1_avg=0.121
[BTC] BACKTEST (gross):  CAGR=0.0778 Sharpe=0.44
[BTC] BACKTEST (VIP9 2.0bps): CAGR=-0.1186 Sharpe=-0.43
[BTC] BACKTEST (VIP4 5.4bps): CAGR=-0.3741 Sharpe=-1.91

[BNB] ==== PIPELINE START (v3) ====
[BNB] LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[BNB] FEATURES+LABEL: windows=(3, 6, 12, 24) H=4 eps=0.001
[BNB] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[BNB] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[BNB] add_features: generated 67 features; total NaNs after lag=787
[BNB] make_label(3-class): H=4 eps=0.001 counts={1: 8355, -1: 7365, 0: 1848}
[BNB] prepare_Xy: dropped NaNs -> rows 17568 → 17518; features=67
[BNB] TRAIN: TSCV n_splits=5, C=0.5
[BNB] train: fold 1/5 | train=2923 test=2919
[BNB] train: fold 1 metrics | macroAUC=0.5536 ACC=0.4440
[BNB] train: fold 2/5 | train=5842 test=2919
[BNB] train: fold 2 metrics | macroAUC=0.5496 ACC=0.4673
[BNB] train: fold 3/5 | train=8761 test=2919
[BNB] train: fold 3 metrics | macroAUC=0.5573 ACC=0.4721
[BNB] train: fold 4/5 | train=11680 test=2919
[BNB] train: fold 4 metrics | macroAUC=0.5409 ACC=0.4423
[BNB] train: fold 5/5 | train=14599 test=2919
[BNB] train: fold 5 metrics | macroAUC=0.5653 ACC=0.4190
[BNB] train: OOF metrics | macroAUC=0.5284 ACC=0.3914
[BNB] OOF macroAUC=0.5284 ACC=0.3914
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
[BNB] TUNED THRESHOLDS: t_long=0.55 t_short=0.55 | f1_avg=0.201
[BNB] BACKTEST (gross):  CAGR=0.0985 Sharpe=0.44
[BNB] BACKTEST (VIP9 2.0bps): CAGR=-0.0997 Sharpe=-0.07
[BNB] BACKTEST (VIP4 5.4bps): CAGR=-0.3581 Sharpe=-0.94

[SOL] ==== PIPELINE START (v3) ====
[SOL] LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[SOL] FEATURES+LABEL: windows=(3, 6, 12, 24) H=4 eps=0.001
[SOL] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[SOL] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[SOL] add_features: generated 67 features; total NaNs after lag=787
[SOL] make_label(3-class): H=4 eps=0.001 counts={1: 8449, -1: 8106, 0: 1013}
[SOL] prepare_Xy: dropped NaNs -> rows 17568 → 17518; features=67
[SOL] TRAIN: TSCV n_splits=5, C=0.5
[SOL] train: fold 1/5 | train=2923 test=2919
[SOL] train: fold 1 metrics | macroAUC=0.5375 ACC=0.4734
[SOL] train: fold 2/5 | train=5842 test=2919
[SOL] train: fold 2 metrics | macroAUC=0.5083 ACC=0.4669
[SOL] train: fold 3/5 | train=8761 test=2919
[SOL] train: fold 3 metrics | macroAUC=0.5167 ACC=0.4902
[SOL] train: fold 4/5 | train=11680 test=2919
[SOL] train: fold 4 metrics | macroAUC=0.5469 ACC=0.4913
[SOL] train: fold 5/5 | train=14599 test=2919
[SOL] train: fold 5 metrics | macroAUC=0.5399 ACC=0.4782
[SOL] train: OOF metrics | macroAUC=0.5259 ACC=0.4072
[SOL] OOF macroAUC=0.5259 ACC=0.4072
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
[SOL] TUNED THRESHOLDS: t_long=0.55 t_short=0.55 | f1_avg=0.148
[SOL] BACKTEST (gross):  CAGR=0.3053 Sharpe=0.81
[SOL] BACKTEST (VIP9 2.0bps): CAGR=0.0002 Sharpe=0.23
[SOL] BACKTEST (VIP4 5.4bps): CAGR=-0.3640 Sharpe=-0.76

[ETH] ==== PIPELINE START (v3) ====
[ETH] LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[ETH] FEATURES+LABEL: windows=(3, 6, 12, 24) H=4 eps=0.001
[ETH] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[ETH] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[ETH] add_features: generated 67 features; total NaNs after lag=787
[ETH] make_label(3-class): H=4 eps=0.001 counts={1: 8182, -1: 7748, 0: 1638}
[ETH] prepare_Xy: dropped NaNs -> rows 17568 → 17518; features=67
[ETH] TRAIN: TSCV n_splits=5, C=0.5
[ETH] train: fold 1/5 | train=2923 test=2919
[ETH] train: fold 1 metrics | macroAUC=0.5403 ACC=0.4659
[ETH] train: fold 2/5 | train=5842 test=2919
[ETH] train: fold 2 metrics | macroAUC=0.5532 ACC=0.4772
[ETH] train: fold 3/5 | train=8761 test=2919
[ETH] train: fold 3 metrics | macroAUC=0.5447 ACC=0.4800
[ETH] train: fold 4/5 | train=11680 test=2919
[ETH] train: fold 4 metrics | macroAUC=0.5511 ACC=0.4800
[ETH] train: fold 5/5 | train=14599 test=2919
[ETH] train: fold 5 metrics | macroAUC=0.5604 ACC=0.4419
[ETH] train: OOF metrics | macroAUC=0.5237 ACC=0.4097
[ETH] OOF macroAUC=0.5237 ACC=0.4097
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
[ETH] TUNED THRESHOLDS: t_long=0.55 t_short=0.55 | f1_avg=0.199
[ETH] BACKTEST (gross):  CAGR=0.9644 Sharpe=2.05
[ETH] BACKTEST (VIP9 2.0bps): CAGR=0.4986 Sharpe=1.30
[ETH] BACKTEST (VIP4 5.4bps): CAGR=-0.0542 Sharpe=0.03

[AVAX] ==== PIPELINE START (v3) ====
[AVAX] LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[AVAX] FEATURES+LABEL: windows=(3, 6, 12, 24) H=4 eps=0.001
[AVAX] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[AVAX] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[AVAX] add_features: generated 67 features; total NaNs after lag=787
[AVAX] make_label(3-class): H=4 eps=0.001 counts={-1: 8449, 1: 8139, 0: 980}
[AVAX] prepare_Xy: dropped NaNs -> rows 17568 → 17518; features=67
[AVAX] TRAIN: TSCV n_splits=5, C=0.5
[AVAX] train: fold 1/5 | train=2923 test=2919
[AVAX] train: fold 1 metrics | macroAUC=0.5260 ACC=0.4734
[AVAX] train: fold 2/5 | train=5842 test=2919
[AVAX] train: fold 2 metrics | macroAUC=0.5312 ACC=0.4865
[AVAX] train: fold 3/5 | train=8761 test=2919
[AVAX] train: fold 3 metrics | macroAUC=0.5263 ACC=0.4967
[AVAX] train: fold 4/5 | train=11680 test=2919
[AVAX] train: fold 4 metrics | macroAUC=0.5185 ACC=0.4782
[AVAX] train: fold 5/5 | train=14599 test=2919
[AVAX] train: fold 5 metrics | macroAUC=0.5062 ACC=0.4765
[AVAX] train: OOF metrics | macroAUC=0.5184 ACC=0.4093
[AVAX] OOF macroAUC=0.5184 ACC=0.4093
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
[AVAX] TUNED THRESHOLDS: t_long=0.55 t_short=0.55 | f1_avg=0.148
[AVAX] BACKTEST (gross):  CAGR=0.1906 Sharpe=0.60
[AVAX] BACKTEST (VIP9 2.0bps): CAGR=-0.0621 Sharpe=0.13
[AVAX] BACKTEST (VIP4 5.4bps): CAGR=-0.3749 Sharpe=-0.66

[TRON] ==== PIPELINE START (v3) ====
[TRON] LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[TRON] FEATURES+LABEL: windows=(3, 6, 12, 24) H=4 eps=0.001
[TRON] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[TRON] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[TRON] add_features: generated 67 features; total NaNs after lag=787
[TRON] make_label(3-class): H=4 eps=0.001 counts={1: 8152, -1: 6756, 0: 2660}
[TRON] prepare_Xy: dropped NaNs -> rows 17568 → 17518; features=67
[TRON] TRAIN: TSCV n_splits=5, C=0.5
[TRON] train: fold 1/5 | train=2923 test=2919
[TRON] train: fold 1 metrics | macroAUC=0.5486 ACC=0.4457
[TRON] train: fold 2/5 | train=5842 test=2919
[TRON] train: fold 2 metrics | macroAUC=0.5429 ACC=0.3813
[TRON] train: fold 3/5 | train=8761 test=2919
[TRON] train: fold 3 metrics | macroAUC=0.5467 ACC=0.4471
[TRON] train: fold 4/5 | train=11680 test=2919
[TRON] train: fold 4 metrics | macroAUC=0.5178 ACC=0.4430
[TRON] train: fold 5/5 | train=14599 test=2919
[TRON] train: fold 5 metrics | macroAUC=0.5400 ACC=0.3991
[TRON] train: OOF metrics | macroAUC=0.5319 ACC=0.3784
[TRON] OOF macroAUC=0.5319 ACC=0.3784
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
[TRON] TUNED THRESHOLDS: t_long=0.55 t_short=0.55 | f1_avg=0.169
[TRON] BACKTEST (gross):  CAGR=-0.1976 Sharpe=-0.43
[TRON] BACKTEST (VIP9 2.0bps): CAGR=-0.3443 Sharpe=-1.00
[TRON] BACKTEST (VIP4 5.4bps): CAGR=-0.5349 Sharpe=-1.95

[ARBtrium] ==== PIPELINE START (v3) ====
[ARBtrium] LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[ARBtrium] FEATURES+LABEL: windows=(3, 6, 12, 24) H=4 eps=0.001
[ARBtrium] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[ARBtrium] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[ARBtrium] add_features: generated 67 features; total NaNs after lag=787
[ARBtrium] make_label(3-class): H=4 eps=0.001 counts={-1: 8387, 1: 8222, 0: 959}
[ARBtrium] prepare_Xy: dropped NaNs -> rows 17568 → 17518; features=67
[ARBtrium] TRAIN: TSCV n_splits=5, C=0.5
[ARBtrium] train: fold 1/5 | train=2923 test=2919
[ARBtrium] train: fold 1 metrics | macroAUC=0.5328 ACC=0.4676
[ARBtrium] train: fold 2/5 | train=5842 test=2919
[ARBtrium] train: fold 2 metrics | macroAUC=0.5368 ACC=0.4961
[ARBtrium] train: fold 3/5 | train=8761 test=2919
[ARBtrium] train: fold 3 metrics | macroAUC=0.5645 ACC=0.4967
[ARBtrium] train: fold 4/5 | train=11680 test=2919
[ARBtrium] train: fold 4 metrics | macroAUC=0.5123 ACC=0.4752
[ARBtrium] train: fold 5/5 | train=14599 test=2919
[ARBtrium] train: fold 5 metrics | macroAUC=0.5087 ACC=0.4841
[ARBtrium] train: OOF metrics | macroAUC=0.5219 ACC=0.4117
[ARBtrium] OOF macroAUC=0.5219 ACC=0.4117
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
[ARBtrium] TUNED THRESHOLDS: t_long=0.55 t_short=0.55 | f1_avg=0.154
[ARBtrium] BACKTEST (gross):  CAGR=0.4757 Sharpe=1.00
[ARBtrium] BACKTEST (VIP9 2.0bps): CAGR=0.1438 Sharpe=0.52
[ARBtrium] BACKTEST (VIP4 5.4bps): CAGR=-0.2585 Sharpe=-0.29

[Sui] ==== PIPELINE START (v3) ====
[Sui] LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[Sui] FEATURES+LABEL: windows=(3, 6, 12, 24) H=4 eps=0.001
[Sui] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[Sui] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[Sui] add_features: generated 67 features; total NaNs after lag=787
[Sui] make_label(3-class): H=4 eps=0.001 counts={-1: 8478, 1: 8313, 0: 777}
[Sui] prepare_Xy: dropped NaNs -> rows 17568 → 17518; features=67
[Sui] TRAIN: TSCV n_splits=5, C=0.5
[Sui] train: fold 1/5 | train=2923 test=2919
[Sui] train: fold 1 metrics | macroAUC=0.5312 ACC=0.5053
[Sui] train: fold 2/5 | train=5842 test=2919
[Sui] train: fold 2 metrics | macroAUC=0.5365 ACC=0.5067
[Sui] train: fold 3/5 | train=8761 test=2919
[Sui] train: fold 3 metrics | macroAUC=0.5683 ACC=0.5344
[Sui] train: fold 4/5 | train=11680 test=2919
[Sui] train: fold 4 metrics | macroAUC=0.5348 ACC=0.4810
[Sui] train: fold 5/5 | train=14599 test=2919
[Sui] train: fold 5 metrics | macroAUC=0.5272 ACC=0.4776
[Sui] train: OOF metrics | macroAUC=0.5357 ACC=0.4227
[Sui] OOF macroAUC=0.5357 ACC=0.4227
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
[Sui] TUNED THRESHOLDS: t_long=0.55 t_short=0.55 | f1_avg=0.301
[Sui] BACKTEST (gross):  CAGR=0.3153 Sharpe=0.74
[Sui] BACKTEST (VIP9 2.0bps): CAGR=-0.0508 Sharpe=0.33
[Sui] BACKTEST (VIP4 5.4bps): CAGR=-0.4549 Sharpe=-0.38

[Polygon] ==== PIPELINE START (v3) ====
[Polygon] LOAD: rows=7587, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2024-09-10 02:59:59.999000+00:00, cols_req_present=True
[Polygon] FEATURES+LABEL: windows=(3, 6, 12, 24) H=4 eps=0.001
[Polygon] prepare_Xy: input rows=7587 [2023-10-30 00:59:59.999000+00:00 → 2024-09-10 02:59:59.999000+00:00]
[Polygon] add_features: start, rows=7587, windows=(3, 6, 12, 24)
[Polygon] add_features: generated 67 features; total NaNs after lag=787
[Polygon] make_label(3-class): H=4 eps=0.001 counts={1: 3616, -1: 3456, 0: 515}
[Polygon] prepare_Xy: dropped NaNs -> rows 7587 → 7537; features=67
[Polygon] TRAIN: TSCV n_splits=5, C=0.5
[Polygon] train: fold 1/5 | train=1257 test=1256
[Polygon] train: fold 1 metrics | macroAUC=0.5472 ACC=0.5032
[Polygon] train: fold 2/5 | train=2513 test=1256
[Polygon] train: fold 2 metrics | macroAUC=0.5322 ACC=0.4546
[Polygon] train: fold 3/5 | train=3769 test=1256
[Polygon] train: fold 3 metrics | macroAUC=0.5290 ACC=0.4658
[Polygon] train: fold 4/5 | train=5025 test=1256
[Polygon] train: fold 4 metrics | macroAUC=0.5175 ACC=0.4323
[Polygon] train: fold 5/5 | train=6281 test=1256
[Polygon] train: fold 5 metrics | macroAUC=0.5117 ACC=0.4498
[Polygon] train: OOF metrics | macroAUC=0.5179 ACC=0.3934
[Polygon] OOF macroAUC=0.5179 ACC=0.3934
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
[Polygon] TUNED THRESHOLDS: t_long=0.55 t_short=0.55 | f1_avg=0.258
[Polygon] BACKTEST (gross):  CAGR=1.2769 Sharpe=1.76
[Polygon] BACKTEST (VIP9 2.0bps): CAGR=0.7227 Sharpe=1.26
[Polygon] BACKTEST (VIP4 5.4bps): CAGR=0.0720 Sharpe=0.40
+----------+-----------+-------+----------+----------+---------+--------+--------+---------+-------------+-----------+------------+------------+----------------------------------+----------------------------------+
|  Token   |   Symbol  |  Rows | macroAUC | Accuracy | Prec(1) | Rec(1) | t_long | t_short | CAGR(gross) | Sharpe(g) | CAGR(VIP9) | CAGR(VIP4) |              Start               |               End                |
+----------+-----------+-------+----------+----------+---------+--------+--------+---------+-------------+-----------+------------+------------+----------------------------------+----------------------------------+
|   BTC    |  BTCUSDT  | 17518 |  0.536   |  0.393   |  0.460  | 0.448  |  0.55  |   0.55  |    0.078    |    0.44   |   -0.119   |   -0.374   | 2023-11-01 02:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
|   BNB    |  BNBUSDT  | 17518 |  0.528   |  0.391   |  0.479  | 0.336  |  0.55  |   0.55  |    0.099    |    0.44   |   -0.100   |   -0.358   | 2023-11-01 02:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
|   SOL    |  SOLUSDT  | 17518 |  0.526   |  0.407   |  0.490  | 0.365  |  0.55  |   0.55  |    0.305    |    0.81   |   0.000    |   -0.364   | 2023-11-01 02:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
|   ETH    |  ETHUSDT  | 17518 |  0.524   |  0.410   |  0.486  | 0.346  |  0.55  |   0.55  |    0.964    |    2.05   |   0.499    |   -0.054   | 2023-11-01 02:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
|   AVAX   |  AVAXUSDT | 17518 |  0.518   |  0.409   |  0.469  | 0.498  |  0.55  |   0.55  |    0.191    |    0.60   |   -0.062   |   -0.375   | 2023-11-01 02:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
|   TRON   |  TRXUSDT  | 17518 |  0.532   |  0.378   |  0.476  | 0.365  |  0.55  |   0.55  |    -0.198   |   -0.43   |   -0.344   |   -0.535   | 2023-11-01 02:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
| ARBtrium |  ARBUSDT  | 17518 |  0.522   |  0.412   |  0.473  | 0.515  |  0.55  |   0.55  |    0.476    |    1.00   |   0.144    |   -0.258   | 2023-11-01 02:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
|   Sui    |  SUIUSDT  | 17518 |  0.536   |  0.423   |  0.515  | 0.208  |  0.55  |   0.55  |    0.315    |    0.74   |   -0.051   |   -0.455   | 2023-11-01 02:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
| Polygon  | MATICUSDT |  7537 |  0.518   |  0.393   |  0.462  | 0.462  |  0.55  |   0.55  |    1.277    |    1.76   |   0.723    |   0.072    | 2023-11-01 02:59:59.999000+00:00 | 2024-09-10 02:59:59.999000+00:00 |
+----------+-----------+-------+----------+----------+---------+--------+--------+---------+-------------+-----------+------------+------------+----------------------------------+----------------------------------+