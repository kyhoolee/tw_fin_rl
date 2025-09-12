tuyệt — mình đề xuất một “spec” gọn – rõ – sạch để bạn có thể chạy từ rule-based → ML → RL mà không sợ leak/cherry-pick.

# 1) Mục tiêu & nguyên tắc

* **Không leak tương lai**: mọi feature đều là quá khứ đến `t-1`; mọi label chỉ nhìn trong tương lai giới hạn.
* **Tách bạch vai trò**:

  * **Train bundle**: bạn muốn làm gì cũng được (chia train/valid, tune HParams…), nhưng **không đụng** vào test.
  * **Test bundle (fixed)**: bất biến, chỉ đánh giá. Có **embargo** (vùng đệm) loại bỏ lân cận thời gian để tránh thông tin rỉ.
  * **Unseen bundle**: giai đoạn/điều kiện thị trường khác, chỉ report thêm để đo **out-of-distribution** (chống cherry-pick).
* **Nhiều tổ hợp (bundles)**: bạn có thể tạo nhiều cặp `train/test/unseen` khác nhau theo “kịch bản thị trường” và đánh giá **overall** bằng cách **macro-average**/**pooled**.

---

# 2) Tổ chức thư mục & manifest

```
data/
  binance/
    1m/ 5m/ 1h/
      BTCUSDT_1m_*.csv  ...
  bundles/
    v1_minutes/
      bundle_01/
        META.yaml               # mô tả bundle
        symbols.txt             # danh sách symbols
        feature_spec.yaml       # định nghĩa feature + lag an toàn
        label_spec.yaml         # định nghĩa nhãn/horizon
        splits/
          train.parquet         # rows = (symbol, ts, features..., label..., meta...)
          test.parquet
          unseen.parquet
        embargo.json            # thông số embargo cho test (e.g. 1d, 3h…)
      bundle_02/
        ...
    v1_hours/
      ...
artifacts/
  reports/
    bundle_01/
      metrics_test.json
      metrics_unseen.json
      curves/                   # equity, drawdown, PR/ROC (nếu có)
```

**META.yaml** (ví dụ):

```yaml
name: bundle_01
freq: 1m
created_at: 2025-09-12T14:00:00+07:00
symbols: [BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT]
time_ranges:
  train: {start: '2020-01-01', end: '2023-12-31'}
  test:  {start: '2024-01-01', end: '2024-12-31'}
  unseen:{start: '2025-01-01', end: '2025-06-30'}
embargo: '12h'            # vùng đệm loại giao cắt thông tin quanh ranh giới
forward_fill_limit: 3     # số bar cho phép ffill các feature chậm
```

---

# 3) Định nghĩa feature (an toàn thời gian)

## 3.1. Quy tắc chung

* Mọi cột feature tại thời điểm **t** chỉ dùng dữ liệu ≤ **t-1**.
* Dùng **lag**/rolling **closed='left'**; không được `.shift(-k)` trong pipeline feature.
* Nếu resample hoặc dùng MA/EMA: “last/mean” phải là **đến t-1**.

## 3.2. Gợi ý `feature_spec.yaml`

```yaml
lookback_bars: 240        # ~4h ở 1m
columns:
  - name: ret_1
    expr: "pct_change(Close, 1, at='t-1')"
  - name: logret_1
    expr: "log(Close).diff(1, at='t-1')"
  - name: zscore_close_60
    expr: "zscore(Close, 60, at='t-1')"
  - name: vwap_30
    expr: "vwap(Price=Close, Vol=Volume, window=30, at='t-1')"
  - name: roll_vol_60
    expr: "std(logret, 60, at='t-1')"
  - name: dollar_vol_30
    expr: "sum(Close*Volume, 30, at='t-1')"
  - name: rsi_14
    expr: "rsi(Close, 14, at='t-1')"
  - name: time_sin_1d
    expr: "sin(2π * minute_of_day / 1440)"
  - name: time_cos_1d
    expr: "cos(2π * minute_of_day / 1440)"
policy:
  missing: drop_row_if_any_nan
  winsorize_pct: 0.001    # cắt đuôi 0.1%
  scale: robust           # median/IQR
```

> Gợi ý: implement parser nhỏ cho `expr` hoặc viết pipeline Python tương ứng; điểm mấu chốt là **mọi aggregator có tham số `at='t-1'`**.

---

# 4) Định nghĩa label (minh bạch, giới hạn tương lai)

## 4.1. Các kiểu label phổ biến

* **Return\@H**: `y = sign( Close[t+H] / Close[t] - 1 )` hoặc giá trị thực (regression).
* **Hit-TP/SL-within-H**:

  * TP nếu max(High\[t+1..t+H]) ≥ `Close[t]*(1+tp)`,
  * SL nếu min(Low\[t+1..t+H]) ≤ `Close[t]*(1-sl)`,
  * Else neutral (ternary).
* **Next-bar movement**: `sign(Close[t+1]-Close[t])` (cho chiến lược tốc độ).

## 4.2. `label_spec.yaml` (ví dụ)

```yaml
type: "tp_sl_horizon"
horizon: 30          # 30 bars (30 phút)
tp: 0.002            # +0.2%
sl: 0.002            # -0.2%
priority: "first-hit"   # nếu TP&SL trong cùng cửa sổ: ưu tiên cái hit sớm hơn
fallback: "neutral"     # nếu không hit gì trong H bars
```

> Với ML: có thể lưu **cả y\_raw** (ret\@H) và **y\_class** (tp/sl/neutral) để linh hoạt.

---

# 5) Quy tắc split & embargo

* **Time-based split**: cứng theo khoảng thời gian trong `META.yaml`.
* **Embargo**: loại bỏ **E** giờ/bars quanh ranh giới train↔test để:

  * Tránh leak do tính năng rolling dài, hoặc do đuôi vị thế mở.
* **Purged K-fold (tuỳ chọn)**: trong **train bundle** khi làm CV, dùng **PurgedGroupTimeSeriesSplit**:

  * Purge vùng lân cận giữa folds.
  * Có thể thêm **group by day/week** để giảm auto-corr.

---

# 6) Cột dữ liệu chuẩn trong parquet

Mỗi file `*.parquet` nên có:

| cột                              | kiểu                 | mô tả                                                        |
| -------------------------------- | -------------------- | ------------------------------------------------------------ |
| `symbol`                         | str                  | mã                                                           |
| `ts`                             | datetime64\[ns, UTC] | mốc thời gian (open\_time/close\_time tùy chuẩn, thống nhất) |
| `Open, High, Low, Close, Volume` | float                | OHLCV gốc                                                    |
| `feat_*`                         | float                | các feature đã lag-safe                                      |
| `y_raw`                          | float                | nhãn thực (vd: ret\@H) nếu có                                |
| `y_class`                        | int/str              | nhãn phân loại (↑/↓ hoặc TP/SL/NEU)                          |
| `meta_regime`                    | category             | (tuỳ chọn) bull/bear/sideways                                |
| `meta_session`                   | category             | (tuỳ chọn) phiên trong ngày                                  |
| `split`                          | category             | {train,test,unseen}                                          |

> Bạn có thể thêm `idx` tăng dần để dễ index; nhưng `ts` là khoá chính tự nhiên.

---

# 7) Quy trình dựng bundle (pseudo-code)

```python
from pathlib import Path
import pandas as pd
from candle_io import load_symbol_csv, enrich_features

def make_bundle(data_dir, candle, symbols, meta, feat_spec, label_spec, out_dir):
    rows = []
    for sym in symbols:
        raw, _ = load_symbol_csv(sym, data_dir, candle)
        df = enrich_features(raw)                        # có LogRet, DollarVolume,...
        df = apply_features(df, feat_spec)               # TẤT CẢ dùng dữ liệu <= t-1
        df = apply_labels(df, label_spec)                # nhìn về tương lai trong H bars
        df["symbol"] = sym
        rows.append(df)

    full = pd.concat(rows, ignore_index=True)
    full = post_clean(full, feat_spec)                   # winsorize/scale/dropna

    train = slice_time(full, meta["time_ranges"]["train"])
    test  = slice_time(full, meta["time_ranges"]["test"])
    unseen= slice_time(full, meta["time_ranges"]["unseen"])

    # áp dụng EMBARGO cho test: cắt bỏ +- E quanh ranh giới
    E = parse_duration(meta["embargo"])
    train = drop_range(train, around=meta["time_ranges"]["test"], embargo=E)

    # set split flag
    train["split"] = "train"
    test["split"]  = "test"
    unseen["split"]= "unseen"

    Path(out_dir, "splits").mkdir(parents=True, exist_ok=True)
    train.to_parquet(Path(out_dir,"splits/train.parquet"), index=False)
    test.to_parquet (Path(out_dir,"splits/test.parquet"),  index=False)
    unseen.to_parquet(Path(out_dir,"splits/unseen.parquet"),index=False)
```

**Ghi chú an toàn**:

* `apply_features`: mọi rolling/EMA/MA đều phải **shift(1)** hoặc dùng cửa sổ **closed='left'**.
* `apply_labels`: duyệt tương lai **\[t+1 … t+H]**, tuyệt đối không đụng dữ liệu > `t+H`.

---

# 8) Nhiều tổ hợp & đánh giá tổng hợp

Bạn có thể tạo các bundle khác nhau để stress-test:

* **Theo chế độ thị trường**: bull (↑), bear (↓), sideways (biên độ thấp).
* **Theo biến động**: vol-low / vol-med / vol-high (phân vị RollVol).
* **Theo mùa vụ**: năm/lễ/tết, halving, ETF events…
* **Theo phiên**: theo UTC giờ, hoặc “US/EU/Asia session”.

**Đánh giá overall**:

* Report **per-bundle**: CAGR, Sharpe, MaxDD, Calmar, HitRate, Turnover, Fee%.
* **Aggregate**:

  * **Macro-average**: trung bình các chỉ tiêu giữa bundles.
  * **Pooled**: ghép equity curves theo thời gian và tính lại (nếu không chồng thời gian).

---

# 9) Rule cứng cho Test & Unseen

1. **Tuyệt đối không tái-fit** bất kỳ tham số nào bằng dữ liệu test/unseen.
2. **Không thay đổi** `feature_spec.yaml`/`label_spec.yaml` sau khi đã sinh test/unseen.
3. **Không regenerate** test/unseen trừ khi:

   * sửa bug **ảnh hưởng leak**; khi đó tăng `bundle` version và giữ bản cũ để đối chiếu.
4. Báo cáo luôn kèm:

   * **checksum** (MD5/SHA) file `test.parquet`, `unseen.parquet`.
   * thông số `embargo`, `horizon`, `fee`, `slippage` đã dùng.

---

# 10) Starter code (skeleton an toàn thời gian)

```python
import numpy as np, pandas as pd

def _roll_zscore(x: pd.Series, w: int):
    m = x.rolling(w, min_periods=w, closed="left").mean()
    s = x.rolling(w, min_periods=w, closed="left").std(ddof=0)
    return (x.shift(1) - m) / (s + 1e-12)

def build_features_safe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["logret"] = np.log(out["Close"]).diff().shift(0)  # logret(t) dùng Close[t] & Close[t-1]
    out["ret_1"]  = out["Close"].pct_change().shift(1)    # quan sát đến t-1
    out["z_close_60"] = _roll_zscore(out["Close"], 60)
    out["vol_roll_60"] = out["logret"].rolling(60, closed="left").std().fillna(0)
    out["dollar_vol_30"] = (out["Close"]*out["Volume"]).rolling(30, closed="left").sum()
    # time encodings
    m = out["open_time"].dt
    out["tod_sin"] = np.sin(2*np.pi*(m.hour*60+m.minute)/1440.0)
    out["tod_cos"] = np.cos(2*np.pi*(m.hour*60+m.minute)/1440.0)
    # drop rows with incomplete windows (optional)
    return out

def make_label_tp_sl(df: pd.DataFrame, H=30, tp=0.002, sl=0.002):
    hi = df["High"].shift(-np.arange(1, H+1)).to_numpy().T  # (H, N)
    lo = df["Low"] .shift(-np.arange(1, H+1)).to_numpy().T
    px = df["Close"].to_numpy()
    # tìm bar đầu tiên chạm TP/SL
    tp_hit = (hi >= px*(1+tp))
    sl_hit = (lo <= px*(1-sl))
    first_tp = tp_hit.argmax(axis=0); tp_any = tp_hit.any(axis=0)
    first_sl = sl_hit.argmax(axis=0); sl_any = sl_hit.any(axis=0)

    y = np.zeros(len(df), dtype=np.int8)  # -1: SL, 0: NEU, +1: TP
    both = tp_any & sl_any
    only_tp = tp_any & ~sl_any
    only_sl = sl_any & ~tp_any
    # ưu tiên "first-hit"
    y[only_tp] = +1
    y[only_sl] = -1
    first_min = np.minimum(first_tp, first_sl)
    y[both & (first_tp < first_sl)] = +1
    y[both & (first_sl < first_tp)] = -1
    y[both & (first_tp == first_sl)] = 0  # hoà: neutral
    df["y_class"] = y
    return df
```

> Lưu ý: vector hoá nhãn ở trên vẫn **chỉ đọc \[t+1..t+H]**; không ảnh hưởng feature vì bạn đã xây feature với **closed='left'** và/hoặc **shift(1)**.

---

# 11) Kiểm tra “fairness” nhanh trước khi đóng gói

* **Leak audit**:

  * Với một vài timestamp `t*` trong test, in ra các feature → xác nhận tất cả đều là hàm của ≤`t*-1`.
  * Tạm **xáo trộn** thứ tự thời gian và xác nhận bạn **không** sử dụng bất kỳ transform “future-aware” (ví dụ `.expanding().mean()` không closed='left').
* **Embargo sanity**: kiểm tra khoảng cách tối thiểu giữa điểm train gần nhất và điểm test đầu tiên ≥ E.
* **Label sanity**: kiểm tra phân phối nhãn (`y_class` tỷ lệ TP/SL/NEU).

---

# 12) Lộ trình dùng

1. Dùng `candle_io.py` load & enrich cơ bản.
2. Áp `build_features_safe` + spec riêng của bạn (hoặc parser `feature_spec.yaml`).
3. Gắn nhãn theo `label_spec.yaml`.
4. Cắt `train/test/unseen` theo `META.yaml` + **embargo**.
5. Lưu parquet + checksum.
6. Chạy chiến lược:

   * **Rule-based**: đọc parquet test/unseen → backtest với fee/slippage cố định.
   * **ML**: fit models **chỉ trên train** (CV bên trong train), freeze → predict test/unseen.
   * **RL/online**: env chỉ cấp **obs tại t**; reward dùng **thực tế tương lai** trong test/unseen; không tái-fit thông số môi trường.

---

nếu bạn muốn, mình có thể viết luôn 2 file “khởi động nhanh”:

* `bundle_builder.py` (đọc CSV → sinh parquet cho train/test/unseen đúng spec),
* `backtest_harness.py` (chạy một vài chiến lược baseline & in metrics).
