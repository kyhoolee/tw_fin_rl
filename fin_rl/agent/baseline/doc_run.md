đẹp quá 😄 dữ liệu “chảy” mượt y như bạn mô tả. Giờ mình gợi ý bước kế tiếp ở **layer model/portfolio** để tận dụng bundle đồng bộ + K-candle:

## Gợi ý tổ chức layer model (no-leak)

* **Loader đồng bộ** (per-split):

  * Index theo `open_time` (unique\_ts) và pivot theo `symbol` → tensor dạng `(T, N, F)` (T = số bar, N = 4 token, F = cột OHLCV bạn dùng).
  * Khi tạo bất kỳ feature/window ở tầng model, luôn dùng quá khứ đến `t-1`.
  * Nếu cần horizon H để tạo label, chỉ lấy nhãn từ `[t+1..t+H]`.
* **Tôn trọng K**:

  * `META.yaml.embargo.bars = K` chính là “khoảng cách tối thiểu bar” giữa train và test.
  * Quy ước rõ trong code: “mọi phép tính ở train không bao giờ đọc sang vùng `[test_start − KΔ .. test_end + KΔ]`”.

## Loader mẫu (PyTorch-friendly)

```python
import pandas as pd
import numpy as np

def make_tensor(split_path, cols=("Open","High","Low","Close","Volume")):
    df = pd.read_parquet(split_path)
    # đảm bảo đồng bộ: mỗi ts có đủ 4 symbol
    assert df["open_time"].nunique() * df["symbol"].nunique() == len(df)
    # pivot: (ts x symbol) -> cột đa mức
    piv = df.pivot(index="open_time", columns="symbol", values=list(cols)).sort_index()
    # reshape thành (T, N, F)
    # hiện piv có MultiIndex columns: (feature, symbol); ta swap để (symbol, feature)
    piv = piv.reindex(sorted(piv.columns, key=lambda x:(x[1], x[0])), axis=1)
    symbols = sorted(df["symbol"].unique())
    features = list(cols)
    T = len(piv)
    N = len(symbols)
    F = len(features)
    # stack theo thứ tự (symbol-major, feature-minor)
    x = piv.to_numpy().reshape(T, N, F, order="C")
    return x.astype(np.float32), np.array(piv.index), symbols, features
```

## Baseline portfolio rất nhanh (để “feel” dữ liệu)

* **EW (equal-weight)**, **rebalance mỗi ngày** (hoặc mỗi H bar).
* Dùng **Close** để tính return đơn giản; giả sử không phí để cảm nhận dòng chảy.

```python
def ew_rebalance_nav(x_close, rebal_every=1440):  # 1440 bar = 1 ngày cho 1m
    # x_close.shape = (T, N)  ; NAV bắt đầu 1.0
    T, N = x_close.shape
    nav = np.ones(T, dtype=np.float64)
    w = np.ones(N) / N
    # normalised price per asset
    rel = x_close / x_close[0:1]  # scale
    last_rb = 0
    for t in range(1, T):
        # re-balance theo lịch
        if (t - last_rb) % rebal_every == 0:
            last_rb = t
            # đặt lại w = 1/N (ở thực tế: khớp theo NAV * w rồi hold)
        # NAV = w · (rel_t / rel_last_rb) kết hợp từng phase rebalance
        phase_rel = x_close[t] / x_close[last_rb]
        nav[t] = (w * phase_rel).sum()
    return nav
```

Gợi ý chạy nhanh:

```python
x_train, ts_train, syms, feats = make_tensor(".../train.parquet", cols=("Close",))
x_test,  ts_test,  _,     _    = make_tensor(".../test.parquet",  cols=("Close",))

nav_test = ew_rebalance_nav(x_test[:, :, 0], rebal_every=1440)
print("EW test nav end:", float(nav_test[-1]))
```

## Nâng dần độ “promising”

* **Rule-based**: mean-reverting / momentum trên vector 4 token (ví dụ signal theo cross-asset spread).
* **Supervised**: model dự báo vector return/vol cho 4 token; layer trên quyết định weight (mean-variance/Black-Litterman “online”).
* **RL**: state = cửa sổ OHLCV đã chuẩn hoá (no-leak), action = weight vector (simplex), reward = increment NAV; ràng buộc turnover/fee.

Nếu bạn muốn, mình gói sẵn:

1. `PortfolioBatcher` (yield minibatch `(B, L, N, F)` cho LSTM/Transformer),
2. 2 baseline: **EW** và **x% vào top-k momentum** (rebalance theo lịch),
3. khung metric: CAGR, Sharpe, MaxDD, Turnover, Fee impact.

Bạn thích baseline nào trước? Momentum top-k hay mean-reversion cặp BTC-ETH?
