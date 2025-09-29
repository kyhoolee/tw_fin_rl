Rất chuẩn hướng rồi 👍

Cấu trúc dữ liệu lớn dạng `cube` mà bạn build xong `raw` thì coi như có một **data lake nền tảng**. Để mở rộng sang feature và label, bạn nên tách riêng thành các *layer* để vừa tái sử dụng vừa tiết kiệm RAM/disk.

### 1. Nguyên tắc tổ chức

* **Raw cube**: dữ liệu gốc chuẩn hóa, immutable, chỉ build một lần từ source (bạn đã có).
* **Feature cube(s)**: mỗi nhóm feature có thể xuất ra một dataset riêng, cùng index `(time, symbol)`. Ví dụ:

  * `feature_price.zarr` → chứa MA, returns, volatility, …
  * `feature_orderflow.zarr` → chứa imbalance, taker ratio, …
  * `feature_micro.zarr` → chứa orderbook-based nếu sau này có.
* **Label cube(s)**: ví dụ `label_future_return.zarr`, `label_volatility_breakout.zarr`, …
* Như vậy khi training, bạn load `raw + feature nhóm X + label Y` rồi join theo `time, symbol`.

### 2. Cách join

Do bạn đã có **calendar_mode = range_intersection** → đảm bảo các cube con đều thẳng hàng theo `(time, symbol)`
👉 khi join chỉ cần:

```python
xr.merge([raw, feature_price, feature_orderflow, label_future_return])
```

hoặc join từng phần tùy cần.

### 3. Quy trình build feature/label

Đề xuất tách thành 3 bước:

1. **Extractor**: module đọc raw cube, tính toán feature/label cụ thể (dùng dask/xarray/numpy).
2. **Builder**: ghi ra zarr (giống `build_raw_cube`), với metadata `calendar_source` & `feature_order_source` = từ raw.
3. **Registry** (optional): bạn có thể duy trì 1 YAML/json list tất cả feature-cube đã build (tên → path, input requirement), để khi cần thì load đúng tổ hợp.

### 4. Lợi ích

* **Modular**: chỉ cần tính lại nhóm feature liên quan khi thay đổi công thức.
* **Efficient**: load đúng subset cần cho training/analysis.
* **Versioning dễ**: feature A_v1, A_v2 có thể sống song song.
* **Streaming pipeline**: sau này dễ update incremental (thêm data mới chỉ tính thêm cho `latest`).

---

Chuẩn rồi 👍. Sang **phase feature/label engineering** thì mình có thể thiết kế một module kiểu “engine” để giữ cấu trúc thống nhất, rồi sau này chỉ việc thêm các feature/label cụ thể vào thôi.

Mình gợi ý chia làm **2 lớp chính**:

---

## 1. Khung chung (Feature/Label Cube Engine)

### Input:

* Raw cube (time × symbol × feature cơ bản: OHLCV, trades, takerbuy,…).
* Meta (calendar chuẩn, feature_order chuẩn).

### Output:

* **Feature cube**: time × symbol × feature_engineered.
* **Label cube**: time × symbol × label(s).
* Lưu ra `.zarr` độc lập, có thể load/join khi cần.

### Cấu trúc code gợi ý:

```python
@dataclass
class FeatureLabelConfig:
    input_cube_path: Path       # zarr raw cube
    output_feature_path: Path   # zarr features
    output_label_path: Path     # zarr labels
    use_dask: bool = True
    overwrite: bool = False

class FeatureLabelEngineer:
    def __init__(self, config: FeatureLabelConfig):
        self.cfg = config
        self.raw_cube = xr.open_zarr(config.input_cube_path, chunks={})  # lazy nếu dùng dask

    def build_features(self, funcs: dict[str, Callable]) -> xr.Dataset:
        """
        funcs: mapping {feature_name: func(raw_cube)->DataArray}
        Trả về Dataset time×symbol×feature.
        """
        feats = {}
        for name, f in funcs.items():
            feats[name] = f(self.raw_cube)
        ds = xr.Dataset(feats)
        if self.cfg.output_feature_path:
            ds.to_zarr(self.cfg.output_feature_path, mode="w")
        return ds

    def build_labels(self, funcs: dict[str, Callable]) -> xr.Dataset:
        """
        funcs: mapping {label_name: func(raw_cube|feature_cube)->DataArray}
        """
        labs = {}
        for name, f in funcs.items():
            labs[name] = f(self.raw_cube)  # hoặc truyền feature_cube nếu cần
        ds = xr.Dataset(labs)
        if self.cfg.output_label_path:
            ds.to_zarr(self.cfg.output_label_path, mode="w")
        return ds
```

Như vậy có framework cố định, sau chỉ cần viết thêm “hàm tính feature/label” là xong.

---

## 2. Detail tính Feature & Label

### Feature ví dụ:

* **Return-based**:

  * log-return, %change, volatility rolling, z-score.
* **Liquidity/Volume**:

  * rolling mean volume, imbalance (taker buy/sell).
* **Microstructure**:

  * VPIN, trade intensity.
* **Technical**:

  * RSI, moving average, Bollinger bands, v.v.

### Label ví dụ:

* **Classification**:

  * `y = sign(return_{t+Δ})` (up/down).
  * `y = 1 if return > threshold else 0`.
* **Regression**:

  * next-Δ return value.
  * realized volatility in horizon.
* **Event-based**:

  * label khi vượt breakout, large move.

---

## 3. Lưu trữ & sử dụng

* Tách riêng Zarr:

  * `zarr_cube_feature_group1/`
  * `zarr_cube_feature_group2/`
  * `zarr_cube_label/`
* Khi training → chọn nhóm nào thì `xr.merge([raw_cube, feature_cube_X, label_cube])`.

---

👉 Như vậy:

* **CubeBuilder**: raw alignment.
* **FeatureLabelEngineer**: framework sinh feature/label.
* **Func library**: tập hợp hàm tính cụ thể.

---

Bạn muốn mình phác thử **API cụ thể** cho một nhóm feature (ví dụ returns/volatility) và một label (ví dụ next-5m return > 0) để thấy rõ cách dùng engine không?
