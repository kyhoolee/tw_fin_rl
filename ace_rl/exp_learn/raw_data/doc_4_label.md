Ok chuẩn 🚀 mình tổng hợp lại các **kiểu labeling** cho trading dataset (cube time × symbol × candle) theo mức độ từ **đơn giản → phức tạp**, bám theo mấy case mình với bạn vừa bàn:

---

## 1. **Label cơ bản cho 1 asset**

🔹 Mục tiêu: dự báo hướng đi / trạng thái của 1 token riêng lẻ.

* **Binary direction**:

  * 1 nếu giá tại (t+N) > giá tại (t), 0 ngược lại.
* **Multi-class movement**:

  * Up / Down / Flat với ngưỡng (\pm \epsilon).
* **Regression target**:

  * Predict log-return (r = \log(p_{t+N}/p_t)).
* **Risk-adjusted**:

  * Predict return / volatility (Sharpe local window).

---

## 2. **Portfolio-aware label (multi-asset, có constraint)**

🔹 Mục tiêu: coi toàn bộ (N) asset + 1 quote (USDT) như portfolio.

* **Wealth delta**: Label = thay đổi tổng giá trị portfolio sau N step.
* **Best action label**: Ở step (t), trong tập {Buy A, Sell A, Hold}, chọn action đem lại wealth cao nhất sau (t+N).
* **Allocation label**: Vector tỷ trọng tối ưu ((w_1, w_2, ..., w_N)) (giả sử biết tương lai).

---

## 3. **Trade-behavioral label (gần thực tế hơn)**

🔹 Bắt chước trader:

* **Take-profit / Stop-loss triggered**: Label = {TP, SL, Hold}.
* **Partial execution**: Label = “scale in / scale out / flat”.
* **Risk control**: Label = hành động tối ưu nếu muốn giữ drawdown ≤ X%.

---

## 4. **Trajectory-level label (ACE_env style)**

🔹 Đánh giá cả hành trình, không chỉ 1 điểm.

* **Wealth curve labeling**: lưu (W_t) trajectory → so sánh với baseline (hold USDT, hold BTC, v.v.).
* **ACE_score (Anytime Capital Entry)**:

  * Với mỗi possible entry (t_0), tính return đến T.
  * Label = trung bình / quantile của các return.
* **Trajectory classification**: cluster curve thành Good / Neutral / Bad.
* **RL-inspired return-to-go**: tổng reward tương lai với penalty transaction cost, risk.

---

## 5. **Advanced labeling**

🔹 Dùng khi muốn mô hình capture chiến lược phức tạp hơn.

* **Event-based label**: entry trước news, breakout, volume spike.
* **Arbitrage opportunity label**: cycle A → B → … → A có profit > 0.
* **Dynamic scoring**: reward function thay đổi theo regime (bull / bear).
* **Simulation-based**: run một agent policy trên future data → label = outcome trajectory của agent đó.

---

👉 Như vậy, lộ trình implement từ dễ đến khó có thể là:

1. Start: **Binary direction** (Up/Down).
2. Mở rộng: **Portfolio delta / best action**.
3. Tiến lên: **TP/SL partial action**.
4. Cao hơn: **ACE trajectory score**.
5. Cuối cùng: **Event-based / RL-style trajectory**.

---
