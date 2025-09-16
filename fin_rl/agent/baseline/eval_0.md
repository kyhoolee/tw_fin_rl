Rồi 👍 mình tổng hợp lại để bạn có bản **tóm lược ver 1 baseline run**:

---

# 📊 Baseline Portfolio Backtest — Version 1

## 1. Thiết lập

* **Dữ liệu**: minute bars (1 phút) cho 4 coin (BTC, ETH, BNB, SOL).
* **Thời gian**: \~525,600 phút ≈ 1 năm.
* **Phí**: Binance taker 0.075% (7.5 bps).
* **Unit**: tính toán trên returns theo phút, annualization dùng 525,600 phút/năm.

---

## 2. Chiến lược baseline

### Core

* **EW\_daily**: Equal-Weight, rebalance mỗi ngày (1440 phút).
* **MOM\_top2\_look360\_rb60**: Momentum cross-sectional, chọn top-2 theo 360 phút lookback, rebalance mỗi 60 phút.
* **PAIR\_MR\_BTC\_ETH**: Pair trading mean-reversion giữa BTC–ETH.

### Extended

* **EW\_VolTarget\_hl1440**: Equal-Weight + Volatility Targeting (EWMA vol, leverage cap=2).
* **TS\_Breakout\_240**: Time-series breakout Donchian (240 bar).
* **RiskParity\_MinVar\_720**: Min-variance weights dựa rolling cov (720 bar).
* **MOM\_Top2\_Turnover50%\_60**: Momentum top-2 nhưng giới hạn turnover ≤ 50%.

---

## 3. Metrics dùng

* **CAGR**: tăng trưởng kép (ở đây code = NAV\_end – 1, \~ CAGR khi run ≈ 1 năm).
* **Sharpe ratio**: return trên volatility (annualized).
* **Sortino ratio**: return trên downside volatility (annualized).
* **Max Drawdown (MaxDD)**: mức lỗ cực đại từ đỉnh → đáy.
* **Calmar ratio**: CAGR / MaxDD.

---

## 4. Kết quả chính

| Strategy                     | CAGR≈ | Sharpe≈ | Sortino≈ | MaxDD | Calmar≈ | Nhận xét直感                                                |
| ---------------------------- | ----- | ------- | -------- | ----- | ------- | --------------------------------------------------------- |
| **EW\_daily**                | 1.06  | 1.47    | 1.90     | 0.38  | 2.77    | Đơn giản mà hiệu quả; hưởng lợi market beta + rebalance.  |
| **EW\_VolTarget\_hl1440**    | 1.85  | 1.46    | 1.89     | 0.65  | 2.85    | Tốt nhất; risk-control giúp tăng CAGR mạnh, Sharpe cao.   |
| **RiskParity\_MinVar\_720**  | 0.28  | 0.72    | 0.93     | 0.49  | 0.56    | Có lãi nhẹ, risk thấp nhưng hiệu suất không cao.          |
| **MOM\_Top2\_Turnover50%**   | -0.69 | -1.49   | -1.95    | 0.78  | -0.89   | Momentum nhưng phí & đảo chiều thị trường → âm.           |
| **MOM\_top2\_look360\_rb60** | -0.86 | -2.67   | -3.51    | 0.87  | -0.98   | Thua lỗ, nhiều momentum crash.                            |
| **TS\_Breakout\_240**        | -0.91 | -11.20  | -4.41    | 0.91  | -0.99   | Breakout fail vì nhiều whipsaw, phí cao.                  |
| **PAIR\_MR\_BTC\_ETH**       | -1.00 | -49.26  | -43.32   | 1.00  | -1.00   | Gần như cháy tài khoản; spread BTC/ETH không mean-revert. |

---

## 5. Intuition rút ra

* **Hiệu quả nhất**:

  * EW + VolTarget (risk control, leverage hợp lý).
  * EW\_daily (ôm đều, diversify, rebalance định kỳ).

* **An toàn nhưng lãi thấp**:

  * RiskParity (do coin high correlation, diversification hạn chế).

* **Thất bại**:

  * Momentum, breakout: nhiều regime shift, phí cao, whipsaw → âm nặng.
  * Pair mean reversion BTC–ETH: assumption cointegration không đúng → gần như phá sản.

---

## 6. Kết luận ver 1

* **Crypto hợp với broad exposure + risk management**, không hợp alpha-based naive strategies (momentum, breakout, mean-reversion).
* Baseline EW & EW+VolTarget là nền tốt để benchmark.
* Nên **refactor tính CAGR chuẩn (annualized)** để so sánh cross-period.

---

👉 Đây là bản **summary ver 1**.
Bạn có muốn mình viết tiếp một **“ver 2 kế hoạch”** (ví dụ sweep rebalance frequency, thêm benchmark Buy\&Hold BTC, thêm equity curve plots) để chuẩn bị bước tiếp không?
