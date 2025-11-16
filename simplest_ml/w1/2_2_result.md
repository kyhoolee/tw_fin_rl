Using data_dir = data

[BTC] ==== PIPELINE START ====
[BTC] Step 1/4 LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[BTC] Step 3/4 FEATURES: windows=(3, 6, 12, 24)
[BTC] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[BTC] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[BTC] add_features: generated 29 features; total NaNs after lag=288
[BTC] make_label: positives=8939 / 17568
[BTC] prepare_Xy: dropped NaNs -> rows 17568 → 17543; features=29
[BTC] Step 4/4 TRAIN: TSCV n_splits=5, C=0.5, max_iter=2000
[BTC] train: fold 1/5 | train=2928 test=2923
[BTC] train: fold 1 metrics | AUC=0.5235 ACC=0.5180
[BTC] train: fold 2/5 | train=5851 test=2923
[BTC] train: fold 2 metrics | AUC=0.5292 ACC=0.5293
[BTC] train: fold 3/5 | train=8774 test=2923
[BTC] train: fold 3 metrics | AUC=0.5220 ACC=0.5084
[BTC] train: fold 4/5 | train=11697 test=2923
[BTC] train: fold 4 metrics | AUC=0.5174 ACC=0.5214
[BTC] train: fold 5/5 | train=14620 test=2923
[BTC] train: fold 5 metrics | AUC=0.5359 ACC=0.5306
[BTC] train: OOF metrics | AUC=0.5143 ACC=0.5152
[BTC] FINAL METRICS: AUC=0.5143 ACC=0.5152
              precision    recall  f1-score   support

           0      0.505     0.614     0.554      8617
           1      0.530     0.420     0.468      8926

    accuracy                          0.515     17543
   macro avg      0.518     0.517     0.511     17543
weighted avg      0.518     0.515     0.511     17543

Confusion matrix:
 [[5292 3325]
 [5180 3746]]

[BNB] ==== PIPELINE START ====
[BNB] Step 1/4 LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[BNB] Step 3/4 FEATURES: windows=(3, 6, 12, 24)
[BNB] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[BNB] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[BNB] add_features: generated 29 features; total NaNs after lag=288
[BNB] make_label: positives=9079 / 17568
[BNB] prepare_Xy: dropped NaNs -> rows 17568 → 17543; features=29
[BNB] Step 4/4 TRAIN: TSCV n_splits=5, C=0.5, max_iter=2000
[BNB] train: fold 1/5 | train=2928 test=2923
[BNB] train: fold 1 metrics | AUC=0.5111 ACC=0.4967
[BNB] train: fold 2/5 | train=5851 test=2923
[BNB] train: fold 2 metrics | AUC=0.5098 ACC=0.5005
[BNB] train: fold 3/5 | train=8774 test=2923
[BNB] train: fold 3 metrics | AUC=0.5277 ACC=0.5337
[BNB] train: fold 4/5 | train=11697 test=2923
[BNB] train: fold 4 metrics | AUC=0.5195 ACC=0.5142
[BNB] train: fold 5/5 | train=14620 test=2923
[BNB] train: fold 5 metrics | AUC=0.5223 ACC=0.5091
[BNB] train: OOF metrics | AUC=0.5050 ACC=0.5053
[BNB] FINAL METRICS: AUC=0.5050 ACC=0.5053
              precision    recall  f1-score   support

           0      0.490     0.575     0.529      8476
           1      0.526     0.441     0.479      9067

    accuracy                          0.505     17543
   macro avg      0.508     0.508     0.504     17543
weighted avg      0.508     0.505     0.503     17543

Confusion matrix:
 [[4870 3606]
 [5072 3995]]

[SOL] ==== PIPELINE START ====
[SOL] Step 1/4 LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[SOL] Step 3/4 FEATURES: windows=(3, 6, 12, 24)
[SOL] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[SOL] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[SOL] add_features: generated 29 features; total NaNs after lag=288
[SOL] make_label: positives=8921 / 17568
[SOL] prepare_Xy: dropped NaNs -> rows 17568 → 17543; features=29
[SOL] Step 4/4 TRAIN: TSCV n_splits=5, C=0.5, max_iter=2000
[SOL] train: fold 1/5 | train=2928 test=2923
[SOL] train: fold 1 metrics | AUC=0.5283 ACC=0.5115
[SOL] train: fold 2/5 | train=5851 test=2923
[SOL] train: fold 2 metrics | AUC=0.5314 ACC=0.5210
[SOL] train: fold 3/5 | train=8774 test=2923
[SOL] train: fold 3 metrics | AUC=0.5133 ACC=0.5104
[SOL] train: fold 4/5 | train=11697 test=2923
[SOL] train: fold 4 metrics | AUC=0.5080 ACC=0.5111
[SOL] train: fold 5/5 | train=14620 test=2923
[SOL] train: fold 5 metrics | AUC=0.5334 ACC=0.5197
[SOL] train: OOF metrics | AUC=0.5128 ACC=0.5098
[SOL] FINAL METRICS: AUC=0.5128 ACC=0.5098
              precision    recall  f1-score   support

           0      0.502     0.647     0.565      8636
           1      0.524     0.377     0.439      8907

    accuracy                          0.510     17543
   macro avg      0.513     0.512     0.502     17543
weighted avg      0.513     0.510     0.501     17543

Confusion matrix:
 [[5584 3052]
 [5547 3360]]

[ETH] ==== PIPELINE START ====
[ETH] Step 1/4 LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[ETH] Step 3/4 FEATURES: windows=(3, 6, 12, 24)
[ETH] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[ETH] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[ETH] add_features: generated 29 features; total NaNs after lag=288
[ETH] make_label: positives=8961 / 17568
[ETH] prepare_Xy: dropped NaNs -> rows 17568 → 17543; features=29
[ETH] Step 4/4 TRAIN: TSCV n_splits=5, C=0.5, max_iter=2000
[ETH] train: fold 1/5 | train=2928 test=2923
[ETH] train: fold 1 metrics | AUC=0.5192 ACC=0.5115
[ETH] train: fold 2/5 | train=5851 test=2923
[ETH] train: fold 2 metrics | AUC=0.5336 ACC=0.5214
[ETH] train: fold 3/5 | train=8774 test=2923
[ETH] train: fold 3 metrics | AUC=0.5228 ACC=0.5210
[ETH] train: fold 4/5 | train=11697 test=2923
[ETH] train: fold 4 metrics | AUC=0.5341 ACC=0.5327
[ETH] train: fold 5/5 | train=14620 test=2923
[ETH] train: fold 5 metrics | AUC=0.5298 ACC=0.5265
[ETH] train: OOF metrics | AUC=0.5097 ACC=0.5142
[ETH] FINAL METRICS: AUC=0.5097 ACC=0.5142
              precision    recall  f1-score   support

           0      0.505     0.442     0.471      8593
           1      0.521     0.584     0.551      8950

    accuracy                          0.514     17543
   macro avg      0.513     0.513     0.511     17543
weighted avg      0.513     0.514     0.512     17543

Confusion matrix:
 [[3798 4795]
 [3727 5223]]

[AVAX] ==== PIPELINE START ====
[AVAX] Step 1/4 LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[AVAX] Step 3/4 FEATURES: windows=(3, 6, 12, 24)
[AVAX] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[AVAX] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[AVAX] add_features: generated 29 features; total NaNs after lag=288
[AVAX] make_label: positives=8533 / 17568
[AVAX] prepare_Xy: dropped NaNs -> rows 17568 → 17543; features=29
[AVAX] Step 4/4 TRAIN: TSCV n_splits=5, C=0.5, max_iter=2000
[AVAX] train: fold 1/5 | train=2928 test=2923
[AVAX] train: fold 1 metrics | AUC=0.5070 ACC=0.5036
[AVAX] train: fold 2/5 | train=5851 test=2923
[AVAX] train: fold 2 metrics | AUC=0.5098 ACC=0.5125
[AVAX] train: fold 3/5 | train=8774 test=2923
[AVAX] train: fold 3 metrics | AUC=0.5072 ACC=0.4957
[AVAX] train: fold 4/5 | train=11697 test=2923
[AVAX] train: fold 4 metrics | AUC=0.5143 ACC=0.5204
[AVAX] train: fold 5/5 | train=14620 test=2923
[AVAX] train: fold 5 metrics | AUC=0.5096 ACC=0.5132
[AVAX] train: OOF metrics | AUC=0.5023 ACC=0.5086
[AVAX] FINAL METRICS: AUC=0.5023 ACC=0.5086
              precision    recall  f1-score   support

           0      0.514     0.797     0.625      9020
           1      0.486     0.203     0.286      8523

    accuracy                          0.509     17543
   macro avg      0.500     0.500     0.456     17543
weighted avg      0.501     0.509     0.461     17543

Confusion matrix:
 [[7192 1828]
 [6793 1730]]

[TRON] ==== PIPELINE START ====
[TRON] Step 1/4 LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[TRON] Step 3/4 FEATURES: windows=(3, 6, 12, 24)
[TRON] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[TRON] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[TRON] add_features: generated 29 features; total NaNs after lag=288
[TRON] make_label: positives=8913 / 17568
[TRON] prepare_Xy: dropped NaNs -> rows 17568 → 17543; features=29
[TRON] Step 4/4 TRAIN: TSCV n_splits=5, C=0.5, max_iter=2000
[TRON] train: fold 1/5 | train=2928 test=2923
[TRON] train: fold 1 metrics | AUC=0.5230 ACC=0.5399
[TRON] train: fold 2/5 | train=5851 test=2923
[TRON] train: fold 2 metrics | AUC=0.5593 ACC=0.5228
[TRON] train: fold 3/5 | train=8774 test=2923
[TRON] train: fold 3 metrics | AUC=0.5197 ACC=0.4944
[TRON] train: fold 4/5 | train=11697 test=2923
[TRON] train: fold 4 metrics | AUC=0.4951 ACC=0.5142
[TRON] train: fold 5/5 | train=14620 test=2923
[TRON] train: fold 5 metrics | AUC=0.5037 ACC=0.5183
[TRON] train: OOF metrics | AUC=0.4992 ACC=0.5076
[TRON] FINAL METRICS: AUC=0.4992 ACC=0.5076
              precision    recall  f1-score   support

           0      0.500     0.692     0.581      8645
           1      0.523     0.329     0.404      8898

    accuracy                          0.508     17543
   macro avg      0.512     0.510     0.492     17543
weighted avg      0.512     0.508     0.491     17543

Confusion matrix:
 [[5982 2663]
 [5975 2923]]

[ARBtrium] ==== PIPELINE START ====
[ARBtrium] Step 1/4 LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[ARBtrium] Step 3/4 FEATURES: windows=(3, 6, 12, 24)
[ARBtrium] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[ARBtrium] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[ARBtrium] add_features: generated 29 features; total NaNs after lag=288
[ARBtrium] make_label: positives=8695 / 17568
[ARBtrium] prepare_Xy: dropped NaNs -> rows 17568 → 17543; features=29
[ARBtrium] Step 4/4 TRAIN: TSCV n_splits=5, C=0.5, max_iter=2000
[ARBtrium] train: fold 1/5 | train=2928 test=2923
[ARBtrium] train: fold 1 metrics | AUC=0.5306 ACC=0.5046
[ARBtrium] train: fold 2/5 | train=5851 test=2923
[ARBtrium] train: fold 2 metrics | AUC=0.5138 ACC=0.5139
[ARBtrium] train: fold 3/5 | train=8774 test=2923
[ARBtrium] train: fold 3 metrics | AUC=0.5139 ACC=0.5005
[ARBtrium] train: fold 4/5 | train=11697 test=2923
[ARBtrium] train: fold 4 metrics | AUC=0.5107 ACC=0.5070
[ARBtrium] train: fold 5/5 | train=14620 test=2923
[ARBtrium] train: fold 5 metrics | AUC=0.5144 ACC=0.5186
[ARBtrium] train: OOF metrics | AUC=0.5043 ACC=0.5060
[ARBtrium] FINAL METRICS: AUC=0.5043 ACC=0.5060
              precision    recall  f1-score   support

           0      0.510     0.570     0.538      8861
           1      0.501     0.441     0.469      8682

    accuracy                          0.506     17543
   macro avg      0.505     0.505     0.504     17543
weighted avg      0.505     0.506     0.504     17543

Confusion matrix:
 [[5052 3809]
 [4857 3825]]

[Sui] ==== PIPELINE START ====
[Sui] Step 1/4 LOAD: rows=17568, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00, cols_req_present=True
[Sui] Step 3/4 FEATURES: windows=(3, 6, 12, 24)
[Sui] prepare_Xy: input rows=17568 [2023-10-30 00:59:59.999000+00:00 → 2025-10-30 23:59:59.999000+00:00]
[Sui] add_features: start, rows=17568, windows=(3, 6, 12, 24)
[Sui] add_features: generated 29 features; total NaNs after lag=288
[Sui] make_label: positives=8651 / 17568
[Sui] prepare_Xy: dropped NaNs -> rows 17568 → 17543; features=29
[Sui] Step 4/4 TRAIN: TSCV n_splits=5, C=0.5, max_iter=2000
[Sui] train: fold 1/5 | train=2928 test=2923
[Sui] train: fold 1 metrics | AUC=0.5036 ACC=0.4978
[Sui] train: fold 2/5 | train=5851 test=2923
[Sui] train: fold 2 metrics | AUC=0.4863 ACC=0.4913
[Sui] train: fold 3/5 | train=8774 test=2923
[Sui] train: fold 3 metrics | AUC=0.5283 ACC=0.5286
[Sui] train: fold 4/5 | train=11697 test=2923
[Sui] train: fold 4 metrics | AUC=0.5122 ACC=0.5176
[Sui] train: fold 5/5 | train=14620 test=2923
[Sui] train: fold 5 metrics | AUC=0.5033 ACC=0.5026
[Sui] train: OOF metrics | AUC=0.4984 ACC=0.5048
[Sui] FINAL METRICS: AUC=0.4984 ACC=0.5048
              precision    recall  f1-score   support

           0      0.508     0.748     0.605      8905
           1      0.494     0.254     0.335      8638

    accuracy                          0.505     17543
   macro avg      0.501     0.501     0.470     17543
weighted avg      0.501     0.505     0.472     17543

Confusion matrix:
 [[6664 2241]
 [6447 2191]]

[Polygon] ==== PIPELINE START ====
[Polygon] Step 1/4 LOAD: rows=7587, cols=11, span=2023-10-30 00:59:59.999000+00:00 → 2024-09-10 02:59:59.999000+00:00, cols_req_present=True
[Polygon] Step 3/4 FEATURES: windows=(3, 6, 12, 24)
[Polygon] prepare_Xy: input rows=7587 [2023-10-30 00:59:59.999000+00:00 → 2024-09-10 02:59:59.999000+00:00]
[Polygon] add_features: start, rows=7587, windows=(3, 6, 12, 24)
[Polygon] add_features: generated 29 features; total NaNs after lag=288
[Polygon] make_label: positives=3826 / 7587
[Polygon] prepare_Xy: dropped NaNs -> rows 7587 → 7562; features=29
[Polygon] Step 4/4 TRAIN: TSCV n_splits=5, C=0.5, max_iter=2000
[Polygon] train: fold 1/5 | train=1262 test=1260
[Polygon] train: fold 1 metrics | AUC=0.5290 ACC=0.5143
[Polygon] train: fold 2/5 | train=2522 test=1260
[Polygon] train: fold 2 metrics | AUC=0.5275 ACC=0.4921
[Polygon] train: fold 3/5 | train=3782 test=1260
[Polygon] train: fold 3 metrics | AUC=0.4978 ACC=0.5008
[Polygon] train: fold 4/5 | train=5042 test=1260
[Polygon] train: fold 4 metrics | AUC=0.5158 ACC=0.5040
[Polygon] train: fold 5/5 | train=6302 test=1260
[Polygon] train: fold 5 metrics | AUC=0.4786 ACC=0.4817
[Polygon] train: OOF metrics | AUC=0.4991 ACC=0.4975
[Polygon] FINAL METRICS: AUC=0.4991 ACC=0.4975
              precision    recall  f1-score   support

           0      0.494     0.586     0.536      3747
           1      0.502     0.411     0.452      3815

    accuracy                          0.497      7562
   macro avg      0.498     0.498     0.494      7562
weighted avg      0.498     0.497     0.494      7562

Confusion matrix:
 [[2194 1553]
 [2247 1568]]
+----------+-----------+-------+-------+----------+--------------+-----------+----------------------------------+----------------------------------+
|  Token   |   Symbol  |  Rows |  AUC  | Accuracy | Precision(1) | Recall(1) |              Start               |               End                |
+----------+-----------+-------+-------+----------+--------------+-----------+----------------------------------+----------------------------------+
|   BTC    |  BTCUSDT  | 17543 | 0.514 |  0.515   |    0.530     |   0.420   | 2023-10-31 01:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
|   BNB    |  BNBUSDT  | 17543 | 0.505 |  0.505   |    0.526     |   0.441   | 2023-10-31 01:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
|   SOL    |  SOLUSDT  | 17543 | 0.513 |  0.510   |    0.524     |   0.377   | 2023-10-31 01:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
|   ETH    |  ETHUSDT  | 17543 | 0.510 |  0.514   |    0.521     |   0.584   | 2023-10-31 01:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
|   AVAX   |  AVAXUSDT | 17543 | 0.502 |  0.509   |    0.486     |   0.203   | 2023-10-31 01:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
|   TRON   |  TRXUSDT  | 17543 | 0.499 |  0.508   |    0.523     |   0.329   | 2023-10-31 01:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
| ARBtrium |  ARBUSDT  | 17543 | 0.504 |  0.506   |    0.501     |   0.441   | 2023-10-31 01:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
|   Sui    |  SUIUSDT  | 17543 | 0.498 |  0.505   |    0.494     |   0.254   | 2023-10-31 01:59:59.999000+00:00 | 2025-10-30 23:59:59.999000+00:00 |
| Polygon  | MATICUSDT |  7562 | 0.499 |  0.497   |    0.502     |   0.411   | 2023-10-31 01:59:59.999000+00:00 | 2024-09-10 02:59:59.999000+00:00 |
+----------+-----------+-------+-------+----------+--------------+-----------+----------------------------------+----------------------------------+