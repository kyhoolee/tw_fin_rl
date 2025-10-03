==============================================

chuẩn rồi 🚀 mình vẽ sơ đồ text để bạn hình dung mối quan hệ **cube gốc (feature)** → **cube nhãn (label)** và các bước sinh ra:

---

## 📐 Sơ đồ pipeline feature → label

```
┌──────────────────────────────┐
│  CUBE.GỐC (time × symbol × feature)  
│  - values[time,symbol,feature]  
│  - feature ∈ {Open,High,Low,Close,Vol,...}  
└─────────────┬────────────────┘
              │  chọn symbol
              ▼
      ┌────────────────────────┐
      │  SELECT SYMBOL         
      │  lấy series Close/High/Low  
      │  -> ds_sym(time, {Close,High,Low})  
      └───────────┬────────────┘
                  │
   ┌──────────────┴────────────────┐
   │                               │
   ▼                               ▼
[Forward Return]             [Triple Barrier]  
ret_log__Hk:                 tb_label__Hk (±1/0)  
 log(C[t+H]/C[t])             tb_exit_ret__Hk  
                             tb_t_hit__Hk  
   │                               │
   └──────────────┬────────────────┘
                  ▼
       ┌────────────────────────┐
       │  DS_LABEL_SYM (per symbol)  
       │  - nhiều biến label (ret, tb_*)  
       └───────────┬────────────┘
                   │ lặp qua tất cả symbol
                   ▼
       ┌────────────────────────┐
       │  ASSEMBLE CUBE LABEL   
       │  values(time, symbol, label)  
       │  mask(time, symbol, label)  
       │  label ∈ {ret_log__H5, tb_label__H60__pct,...}  
       └───────────┬────────────┘
                   ▼
     ┌──────────────────────────────┐
     │   ZARR OUTPUT (group="labels")  
     │   - /labels/values  
     │   - /labels/mask  
     │   attrs: horizons, tb_cfg,...  
     └──────────────────────────────┘
```

---

## 📝 Giải thích dòng chảy

1. **Cube gốc (feature)**: đầu vào time × symbol × feature.
   → Lấy ra `Close`, `High`, `Low`.

2. **Sinh nhãn per-symbol**:

   * Forward log-returns cho mỗi horizon `H`.
   * Triple-barrier cho mỗi `H` (chạm ngưỡng TP/SL hay hết hạn vertical).

3. **Ghép lại cube label**:

   * Trục `label` thay cho `feature`.
   * Mỗi nhãn là một biến → reshape về `values(time,symbol,label)`.
   * Có thêm `mask` bool để đánh dấu valid/NaN.

4. **Ghi ra Zarr** (group `"labels"`):

   * Cùng format như cube feature → dễ join & align khi train.
   * Metadata lưu tham số `horizons`, `tb_cfg`,… để trace.

---

👉 như vậy bạn có **song song 2 cube**:

* `features`: time × symbol × feature
* `labels`: time × symbol × label

=> khi train model: join theo `(time,symbol)` là có đủ input + target.

---


==============================================

okie, mình vẽ thêm sơ đồ **flow debug** để thấy cách `print_random_debug` hoạt động:

---

## 🔎 Sơ đồ debug triple-barrier

```
┌──────────────────────────────┐
│  Input: symbol + horizon H   │
│  (ds_sym với Close/High/Low) │
└─────────────┬────────────────┘
              │ chọn random t_idx
              ▼
      ┌────────────────────────┐
      │   p0 = Close[t]        │
      │   Horizon = H          │
      │   Ngưỡng:              │
      │     up = p0*(1+up_pct) │
      │     dn = p0*(1-dn_pct) │
      └───────────┬────────────┘
                  │
                  ▼
     ┌────────────────────────────┐
     │ QUÉT từ k = t+1 … t+H      │
     │  - Nếu NaN -> toàn bộ NaN  │
     │  - Nếu High[k] ≥ up → TP   │
     │  - Nếu Low[k] ≤ dn → SL    │
     │  - Lưu lại step:           │
     │    (k, time, close, high,  │
     │     low, hit-status)       │
     └───────────┬────────────────┘
                 │ không hit trong H
                 ▼
     ┌────────────────────────────┐
     │ Vertical barrier tại t+H   │
     │ r = log(C[t+H]/p0)         │
     │  - r > neutral → +1        │
     │  - r < -neutral → -1       │
     │  - else → 0                │
     └───────────┬────────────────┘
                 ▼
     ┌────────────────────────────┐
     │  OUTPUT EXPLAIN JSON:      │
     │   {                        │
     │     t, time, p0, barriers, │
     │     steps[],               │
     │     result (TP/SL/VERT),   │
     │     label                  │
     │   }                        │
     └────────────────────────────┘
```

---

## 📝 Ý nghĩa

* **p0**: giá khởi điểm tại thời gian t.
* **barriers**: ngưỡng trên/dưới (TP/SL).
* **steps[]**: log từng bar trong cửa sổ (k, giá close/high/low, có hit không).
* **result**: lý do dừng → TP / SL / Vertical.
* **label**: giá trị cuối cùng {+1,0,-1,NaN}.

---

👉 Debug flow này giúp bạn **đối chiếu label thực tế với dữ liệu giá** để đảm bảo logic triple-barrier chính xác.
Ví dụ chạy `--print-random-debug` sẽ in ra 2 phần:

1. JSON summary (p0, barrier, kết quả cuối).
2. Steps chi tiết (từng bar trong cửa sổ).

---



==============================================
== RUN LABELING ==============================

python -m ace_rl.core.label.single_symbol \
  --cube-zarr ./work/processed/binance/data/zarr_cube_intersection \
  --out-zarr  ./work/processed/binance/data/zarr_cube_label \
  --group labels \
  --horizons 1 5 15 60 \
  --tb-up-pct 0.01 --tb-dn-pct 0.005 --tb-H 60 --tb-neutral 0.0 \
  --tb-vol \
  --tb-vol-up 2.0 --tb-vol-dn 2.0 --tb-vol-window 60 --tb-vol-neutral 0.0 \
  --print-stats \
  --print-random-debug --debug-samples 3 > single_symbol_label.log
