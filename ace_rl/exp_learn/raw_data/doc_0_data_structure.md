Tuyệt — đây là **sơ đồ text** gọn, mạch lạc cho toàn pipeline: *phase → kiểu dữ liệu → “data-engine” dùng ở mỗi phase → artefact đầu ra*.

```
[PHASE 0] META & HỢP ĐỒNG DỮ LIỆU (Data Contracts)
  ├─ Data: calendar (UTC, candle-level), feature_order, symbols, fees, fragments (non-overlap)
  ├─ Engine: JSON/NPY thuần + kiểm tra bằng Pandas/Polars
  └─ Artefacts:
      meta/calendar.npy
      meta/feature_order.json
      meta/symbols.json
      meta/fees.json
      meta/fragments.jsonl
```

```
[PHASE 1] PREP PER-SYMBOL (ETL gốc, độc lập từng symbol)
  ├─ Data: bảng 2D cho mỗi symbol: [time, features], + mask
  ├─ Engine chính: **Polars** (ETL nhanh, rolling/reset theo fragment), có thể dùng Pandas phụ trợ
  ├─ Lưu trữ:
      - Master: Parquet per-symbol
      - Cache train: Zarr per-symbol (dims: [time, feature], chunk theo time, dtype float32)
  └─ Artefacts:
      data/raw_parquet/{SYMBOL}.parquet
      data/zarr_per_symbol/{SYMBOL}.zarr/
```

```
[PHASE 2] ENV-DATA BUILDER (gom 3D cube + index fragment)
  ├─ Data in: Zarr/Parquet per-symbol
  ├─ Engine: **xarray** (DataArray/Dataset) + (tùy) **dask** cho lazy; concat theo dim="symbol"
  ├─ Kết quả: Cube 3D [time, symbol, feature] (xarray-backed), + chỉ số fragment/valid
  ├─ (Tùy chọn) Cache 3D: Zarr cube (chunk theo time; symbol/feature = -1 nếu nhỏ)
  └─ Artefacts:
      cache/cube/market.zarr/              (optional)
      index/frag_index/{frag_id}.npy       (time indices)
      index/valid_idx/{N}_{K}/{frag}.npy   (vị trí hợp lệ đủ N-lookback & K-forward)
```

```
[PHASE 3] FEATURE PREPROCESSING (no-leak, reset theo fragment)
  ├─ Data: Cube 3D [T,N,F]; fragments & valid_idx
  ├─ Engine:
      - Stateless/rolling per-symbol: **Polars** (trước) hoặc **xarray** (sau), đảm bảo reset theo fragment
      - Cross-symbol (nếu cần): **xarray** (+dask) trên cube
  ├─ Chuẩn hóa: per-symbol rolling (z-score/robust), không dùng toàn chuỗi
  ├─ (Tùy chọn) Prefix-sum/cum-logret để truy hồi nhanh mọi horizon ≤ K
  └─ Artefacts:
      cache/cube_enriched/market.zarr/     (cube đã enrich features)
      cache/returns/cum_logret.zarr/       (optional)
```

```
[PHASE 4] TRAIN/EVAL LOADER (sampler N/K + fee model)
  ├─ Data: cube (hoặc per-symbol zarr) + valid_idx + fees
  ├─ Engine:
      - Cắt cửa sổ ngẫu nhiên: **xarray** → **NumPy** → **PyTorch** tensor (np.ascontiguousarray)
      - Batch sampler dùng valid_idx (không vượt biên fragment)
      - Tính phí: module Python thuần (vectorized NumPy)
  └─ Artefacts runtime:
      - Tensors [B, N_lookback, N_symbol, N_feat] feed vào model
      - Metrics & logs (ACE, PnL, Sharpe, turnover, DD…)
```

```
[PHASE 5] MODELS (Rule-based → Adaptive → RL)
  ├─ Rule-based:
      - Engine: Python/NumPy (+Numpyro/Numba tuỳ ý), dùng cùng loader & fee model
  ├─ Adaptive (ML supervised/unsupervised):
      - Engine: **PyTorch/Lightning**; nhãn tạo từ cum_logret/horizon K (không leak)
  ├─ RL (responsive):
      - Engine: **PyTorch + RLlib/cleanrl/tiện ích riêng**
      - Env: mỗi episode = 1 fragment; reward = per-step PnL sau phí; constraint Spot
  └─ Artefacts:
      models/{exp}/checkpoints/...
      reports/{exp}/metrics_ACE_PnL.json
      plots/{exp}/equity_curves.png
```

---

## Tóm tắt “ai làm gì, ở đâu”

* **Polars** → vua ETL per-symbol & rolling **reset theo fragment** (nhanh, ít RAM).
* **Xarray (+Zarr [+Dask])** → “bộ khung 3D”: gom/ghi cube, slice cửa sổ ngẫu nhiên, random access rẻ nhờ **chunk theo time**.
* **NumPy/PyTorch** → tiêu chuẩn cuối cho training; `np.ascontiguousarray` đảm bảo tensor layout mượt.
* **Parquet** (master, kiểm toán) & **Zarr** (cache train, random window).

---

## Gợi ý cây thư mục “sạch”

```
meta/
  calendar.npy
  feature_order.json
  symbols.json
  fees.json
  fragments.jsonl

data/
  raw_parquet/{SYMBOL}.parquet
  zarr_per_symbol/{SYMBOL}.zarr/

cache/
  cube/market.zarr/                 # optional 3D cache
  cube_enriched/market.zarr/        # optional (sau khi thêm feature)
  returns/cum_logret.zarr/          # optional

index/
  frag_index/{frag_id}.npy
  valid_idx/{N}_{K}/{frag_id}.npy

models/
  {exp}/checkpoints/...

reports/
  {exp}/metrics.json
plots/
  {exp}/equity_curve.png
```

Nếu bạn muốn, mình có thể nhét phần này thành **README.md + Makefile** + vài script CLI (`build-per-symbol`, `build-cube`, `make-valid-idx`, `train-rule`, `train-ml`, `train-rl`) để bạn cắm ngay vào repo `fin_rl`.
