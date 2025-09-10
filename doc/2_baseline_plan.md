Tuyệt vời 🙌 mình gom lại thành một **roadmap tổng thể** để bạn dễ quản lý. Toàn bộ dựa trên các trao đổi trước: baseline plan, feature/news xử lý, thiết kế splits dataset, HPO, eval.

---

# 📌 Roadmap Khai phá Baseline cho Crypto (candles + GDELT news)

## 1. Dataset & ENV

* **Unit of prediction**

  * Feature tại thời điểm *t*: giá + news ≤ *t*.
  * Label: return(t→t+H) hoặc sign(return).
  * Horizons: {5m, 15m, 60m, 1d}.

* **Resample candles**: từ 1m → K-phút (5/15/60).

* **News alignment**: bin news về K-phút, tạo feature `news_count`, `news_z`, `burst`, `pol_mean`, `topic_*`, cross-spill từ BTC.

* **Anti-leak guard**:

  * Purging (loại overlap do horizon H).
  * Embargo (1–3 ngày sát biên).

---

## 2. Splits Strategy

* **Cố định Holdout**: 2025-04 → 2025-09.
* **Tune-set (Mₜ \~16 splits)**

  * Anchored Walk-Forward (6)
  * Rolling Window (6)
  * Regime-Balanced (4)
* **Eval-set (Mₑ \~8 splits, disjoint với Tune)**

  * Regime-Balanced (4)
  * Event-kFold theo News (4)

👉 Kết quả: M = 24 splits + Holdout → bao quát nhiều regime, giảm cherry-pick.

---

## 3. Feature Engineering

* **Price-only**: returns, vol (ATR, realized), SMA/EMA, MACD, RSI, volume-z.
* **News-only**: count, news\_pressure (EMA), burst, z-score, polarity (lexicon), topic-tags (ETF/hack/reg/fork/listing), cross-spill.
* **Hybrid**: interaction terms (`news_z × ret`, `pol_mean × trend`), regime-conditioned.
* **Candle-value convert**: tạo OHLC giả từ feature (để plug thẳng vào backtest/plot).

---

## 4. Baseline Models

* **Rule-based**

  1. Buy & Hold
  2. SMA-Cross (5/20)
  3. Breakout+ATR
  4. News-Spike Momentum (long/short theo news\_z, polarity)
  5. Confirm (News × Trend)
  6. Pair-trade BTC→ETH/BNB/SOL

* **ML Classical**
  7\. Logistic Regression (ridge/lasso, price+news)
  8\. LightGBM / XGBoost (tabular, lag features, hybrid)

* **Sequence nhẹ**
  9\. GRU/CNN nhỏ (causal, horizon 15m).

---

## 5. Backtest Engine

* **Inputs**: candles K-phút + features + signal/positions.
* **Logic**: update portfolio theo bar → PnL.
* **Costs**: fee/taker (0.04–0.1%/side), slippage model (cao hơn khi news spike).
* **Risk controls**:

  * Vol-target sizing
  * Time-stop, ATR SL/TP
  * Cooldown sau trade
* **Outputs**: equity, PnL, trades log.

---

## 6. Evaluation Metrics

* **Core**: CAGR, Sharpe, Sortino, MDD, Calmar, hit-rate, turnover.
* **Slicing**: event vs non-event, session (US/EU/APAC), regime (vol buckets), topic (ETF/hack/...).
* **Robustness**: multiple splits → median/IQR, p25, p75 KPI.
* **Overfit guard**: White’s Reality Check / SPA test.

---

## 7. Hyper-param Tuning Protocol

1. **HPO trên Tune-set (Mₜ)**

   * Optuna/Bayes search.
   * Objective = median Sharpe across K splits.
   * Constraints: p25 Sharpe ≥ S\_min, MDD ≤ MDD\_max, Turnover ≤ TO\_max.
   * Chọn top-N configs.
2. **Lock & Refit** trên toàn Tune-set.
3. **Eval** trên Eval-set (Mₑ, non-overlap).
4. **Final test** trên Holdout (sealed).

---

## 8. Code Structure (gợi ý)

```
fin_rl/
  data/crypto/...
  exp/
    splits.py        # generate Tune/Eval/Holdout splits
    features.py      # price, news, hybrid feature builders
    backtest/
      engine.py
      report.py
    models/
      rules.py       # SMA, Breakout, NewsSpike, Confirm, PairTrade
      ml_baseline.py # LogReg, LGBM, GRU-tiny
      search_spaces.py
    configs/
      00_bh.yaml
      01_sma_cross.yaml
      ...
    run.py           # main runner: hpo / eval / holdout
tests/
  test_leakage.py
  test_data_quality.py
  test_backtest_math.py
```

---

## 9. Ưu tiên triển khai (2 tuần)

1. **Week 1**

   * Xong `splits.py` (AWF, RW, RB, EkF) + test leakage.
   * `features.py` (price+news align, z-score).
   * `backtest/engine.py` (PnL với fee, slippage).
   * Implement rules baselines (BH, SMA, Breakout, NewsSpike, Confirm, PairTrade).

2. **Week 2**

   * ML baseline (LogReg, LGBM, GRU nhỏ).
   * HPO runner + configs.
   * Report slicing (event/regime/session).
   * Run full Tune/Eval, test Holdout cuối.

---

👉 Với khung này, bạn sẽ có:

* Dataset sạch, nhiều splits → giảm cherry-pick.
* Bộ baseline rule/ML → so sánh công bằng.
* Backtest engine + report → tiêu chuẩn hoá đánh giá.
* Pipeline train/tune/eval/holdout → bền để mở rộng model phức tạp sau này.

---

Bạn có muốn mình viết ngay **template `configs/*.yaml` mẫu** (ví dụ cho SMA-Cross, News-Spike Momentum, LGBM) để bạn chạy thử với `run.py` không?
