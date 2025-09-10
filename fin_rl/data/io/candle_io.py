#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binance OHLCV Minute Data Explorer
----------------------------------
Một script "all-in-one" để:
1) Load file CSV mới nhất cho mỗi symbol theo chuẩn tên: {SYMBOL}_{candle_level}_*.csv
2) Chuẩn hoá tên cột về schema thống nhất: 
   ["open_time","Open","High","Low","Close","Volume","QuoteVolume","Trades",
    "TakerBuyBase","TakerBuyQuote"]
3) Kiểm tra chất lượng dữ liệu: duplicates, NaN, giá trị âm, thiếu mốc thời gian (gaps)
4) Thống kê cơ bản và ví dụ enrich (returns, log-returns, rolling vol, dollar volume)
5) Resample ví dụ (5m, 1h, 1d)
6) Vẽ các biểu đồ cơ bản: Price + MA, Volume, Histogram log-returns, Corr heatmap giữa các symbol

Cách dùng nhanh:
    python ohlcv_explorer.py \
        --data-dir ../../work/data/binance/spot/1m \
        --candle-level 1m \
        --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT \
        --save-plots

Mẹo:
- Lúc mới khám phá, có thể bỏ --save-plots để chỉ xem on-screen.
- Thư mục lưu hình mặc định: ./_out
"""

import os
import glob
import argparse
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========== CONFIG MẶC ĐỊNH (có thể override bằng argparse) ==========
DEFAULT_CANDLE = "1m"
DEFAULT_DATA_DIR = f"../../work/data/binance/spot/{DEFAULT_CANDLE}"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
OUT_DIR = "./_out"  # nơi lưu hình nếu --save-plots


# ========== HÀM TIỆN ÍCH ==========

def find_latest_file(sym: str, data_dir: str, candle_level: str) -> str:
    """
    Tìm file CSV mới nhất theo pattern: {sym}_{candle_level}_*.csv
    Trả về đường dẫn tuyệt đối.
    """
    pattern = os.path.join(data_dir, f"{sym}_{candle_level}_*.csv")
    files = sorted(glob.glob(pattern))
    assert files, f"[ERROR] Không thấy file cho {sym} trong {data_dir} với pattern {os.path.basename(pattern)}"
    return files[-1]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hoá tên cột về schema thống nhất, nếu cột gốc có tên khác.
    Kỳ vọng dữ liệu Binance tiêu chuẩn (kline export).
    """
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "quote_asset_volume": "QuoteVolume",
        "number_of_trades": "Trades",
        "taker_buy_base_asset_volume": "TakerBuyBase",
        "taker_buy_quote_asset_volume": "TakerBuyQuote",
    }
    # open_time thường đã có, nếu tên khác thì thêm vào đây
    # Ví dụ: "Open time" -> "open_time"
    if "Open time" in df.columns and "open_time" not in df.columns:
        df = df.rename(columns={"Open time": "open_time"})
    df = df.rename(columns=rename_map)
    return df


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Đảm bảo các cột quan trọng tồn tại, kiểu dữ liệu hợp lý, sort theo thời gian.
    """
    required = [
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "QuoteVolume", "Trades", "TakerBuyBase", "TakerBuyQuote"
    ]
    # Không phải file nào cũng có đầy đủ QuoteVolume/TakerBuy..., linh hoạt optional
    # Ta chỉ enforce core cột bắt buộc:
    core_required = ["open_time", "Open", "High", "Low", "Close", "Volume"]
    for c in core_required:
        assert c in df.columns, f"[ERROR] Thiếu cột bắt buộc: {c}"

    # Parse datetime cho open_time nếu chưa phải datetime
    if not np.issubdtype(df["open_time"].dtype, np.datetime64):
        df["open_time"] = pd.to_datetime(df["open_time"], errors="coerce", utc=True)

    # Sort và reset index
    df = df.sort_values("open_time").reset_index(drop=True)

    # Ép kiểu numeric cho các cột numeric nếu cần
    for c in df.columns:
        if c in ["Open", "High", "Low", "Close", "Volume", "QuoteVolume", "Trades", "TakerBuyBase", "TakerBuyQuote"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def load_symbol_csv(sym: str, data_dir: str, candle_level: str) -> Tuple[pd.DataFrame, str]:
    """
    Load CSV mới nhất của symbol, chuẩn hoá cột, sort, trả về (df, filepath).
    """
    fp = find_latest_file(sym, data_dir, candle_level)
    df = pd.read_csv(fp)
    if "open_time" not in df.columns:
        # nếu file đã parse datetime sẵn (như bạn làm), giữ nguyên:
        # Nhưng trong trường hợp khác, parse ở ensure_schema()
        pass
    df = normalize_columns(df)
    df = ensure_schema(df)
    return df, fp


def print_overview(df: pd.DataFrame, sym: str):
    """
    In overview: rows, dtypes, head/tail, date range, tần suất (ước lượng).
    """
    print(f"\n== {sym} | Overview ==")
    print(f"Rows: {len(df):,}")
    print("Dtypes:")
    print(df.dtypes)
    print("\nHead:")
    print(df.head(3))
    print("\nTail:")
    print(df.tail(3))

    if len(df) > 1:
        start, end = df["open_time"].iloc[0], df["open_time"].iloc[-1]
        delta = end - start
        print(f"\nDate range: {start} → {end} | span={delta}")

        # Ước lượng tần suất bằng median diff
        diffs = df["open_time"].diff().dropna().dt.total_seconds()
        if not diffs.empty:
            med = diffs.median()
            print(f"Estimated frequency (median Δt): {med:.0f} sec")


def quality_checks(df: pd.DataFrame, sym: str, expected_freq="1min"):
    """
    Kiểm tra: duplicates theo open_time, NaN, giá trị âm, thiếu timestamp so với grid.
    expected_freq: '1min' cho nến 1 phút.
    """
    print(f"\n== {sym} | Quality checks ==")
    # duplicates
    n_dup = df["open_time"].duplicated().sum()
    print(f"Duplicates(open_time): {n_dup}")

    # NaN thống kê nhanh ở numeric columns
    num_cols = ["Open", "High", "Low", "Close", "Volume", "QuoteVolume", "Trades", "TakerBuyBase", "TakerBuyQuote"]
    num_cols = [c for c in num_cols if c in df.columns]
    n_nan = df[num_cols].isna().sum().sum()
    print(f"NaN in numeric columns: {int(n_nan)}")

    # Giá trị âm (không hợp lệ) ở giá/khối lượng
    neg_cols = []
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns and (df[c] < 0).any():
            neg_cols.append(c)
    print(f"Negative columns: {neg_cols if neg_cols else 'None'}")

    # Missing timestamps so với full grid (nếu dữ liệu khá dày và liên tục)
    if len(df) > 1:
        start, end = df["open_time"].iloc[0], df["open_time"].iloc[-1]
        full = pd.date_range(start=start, end=end, freq=expected_freq, inclusive="both")
        missing = full.difference(df["open_time"])
        print(f"Missing bars vs full {expected_freq} grid: {len(missing)}")


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm một số feature cơ bản để phân tích:
    - returns đơn giản (pct_change)
    - log returns
    - dollar volume ~ Close * Volume (xấp xỉ)
    - rolling volatility trên log-returns (window 60)
    """
    out = df.copy()
    out["Ret"] = out["Close"].pct_change()
    out["LogRet"] = np.log(out["Close"]).diff()
    out["DollarVolume"] = out["Close"] * out["Volume"]
    out["RollVol60"] = out["LogRet"].rolling(60).std() * np.sqrt(60)  # vol phút, annual hoá thì scale thêm
    return out


def resample_ohlcv(df: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    """
    Ví dụ resample sang khung lớn hơn. Đảm bảo open_time làm index trước khi resample.
    Lưu ý: cần chuẩn aggregator cho OHLCV.
    """
    tmp = df.set_index("open_time")
    o = tmp["Open"].resample(rule).first()
    h = tmp["High"].resample(rule).max()
    l = tmp["Low"].resample(rule).min()
    c = tmp["Close"].resample(rule).last()
    v = tmp["Volume"].resample(rule).sum()
    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ["Open", "High", "Low", "Close", "Volume"]
    out = out.dropna(how="any").reset_index()
    return out


def basic_stats(df: pd.DataFrame) -> pd.Series:
    """
    Trả về thống kê tóm tắt hữu ích.
    """
    s = pd.Series(dtype="float64")
    s["rows"] = len(df)
    s["start_ts"] = df["open_time"].iloc[0].value if len(df) else np.nan
    s["end_ts"] = df["open_time"].iloc[-1].value if len(df) else np.nan
    s["close_mean"] = df["Close"].mean()
    s["close_std"] = df["Close"].std()
    s["volume_mean"] = df["Volume"].mean()
    s["trades_mean"] = df["Trades"].mean() if "Trades" in df.columns else np.nan
    return s


# ========== VẼ BIỂU ĐỒ CƠ BẢN ==========

def plot_price_with_ma(df: pd.DataFrame, sym: str, save: bool = False):
    plt.figure(figsize=(10, 4))
    plt.plot(df["open_time"], df["Close"], label="Close")
    if len(df) >= 50:
        plt.plot(df["open_time"], df["Close"].rolling(50).mean(), label="MA50")
    if len(df) >= 200:
        plt.plot(df["open_time"], df["Close"].rolling(200).mean(), label="MA200")
    plt.title(f"{sym} - Close with MA")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUT_DIR, f"{sym}_price_ma.png"), dpi=150)
    plt.show()


def plot_volume(df: pd.DataFrame, sym: str, save: bool = False):
    plt.figure(figsize=(10, 3))
    plt.plot(df["open_time"], df["Volume"], label="Volume")
    plt.title(f"{sym} - Volume")
    plt.xlabel("Time")
    plt.ylabel("Volume")
    plt.tight_layout()
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUT_DIR, f"{sym}_volume.png"), dpi=150)
    plt.show()


def plot_logret_hist(df: pd.DataFrame, sym: str, save: bool = False):
    lr = df["LogRet"].dropna()
    if lr.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(lr, bins=80, alpha=0.8)
    plt.title(f"{sym} - Log-returns histogram")
    plt.xlabel("Log-Return")
    plt.ylabel("Count")
    plt.tight_layout()
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUT_DIR, f"{sym}_logret_hist.png"), dpi=150)
    plt.show()


def plot_corr_heatmap(prices: Dict[str, pd.DataFrame], save: bool = False):
    """
    Vẽ ma trận tương quan (Pearson) giữa log-returns của các symbol.
    """
    # Chuẩn hoá: chỉ lấy cột LogRet và align theo thời gian chung
    lr_dict = {}
    for sym, df in prices.items():
        if "LogRet" in df.columns:
            s = df.set_index("open_time")["LogRet"]
            lr_dict[sym] = s

    if not lr_dict:
        print("[WARN] Không có LogRet để vẽ tương quan.")
        return

    aligned = pd.concat(lr_dict, axis=1).dropna(how="any")
    if aligned.empty:
        print("[WARN] Không có khoảng thời gian chung giữa các series.")
        return

    corr = aligned.corr().values
    labels = list(aligned.columns)

    plt.figure(figsize=(5 + 0.4*len(labels), 4 + 0.4*len(labels)))
    im = plt.imshow(corr, interpolation="nearest")
    plt.title("Correlation (Log-Returns)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.tight_layout()
    if save:
        os.makedirs(OUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUT_DIR, f"corr_heatmap.png"), dpi=150)
    plt.show()


# ========== MAIN ==========

def main(args):
    symbols = args.symbols
    data_dir = args.data_dir
    candle = args.candle_level
    save_plots = args.save_plots

    print(f"[INFO] Loading {len(symbols)} symbols from {data_dir} | candle={candle}")

    store: Dict[str, Dict[str, pd.DataFrame]] = {}

    # 1) Load & Inspect từng symbol
    for sym in symbols:
        df, fp = load_symbol_csv(sym, data_dir, candle)
        print(f"[OK] {sym}: {len(df):,} rows | file={os.path.basename(fp)}")

        print_overview(df, sym)
        quality_checks(df, sym, expected_freq="1min")

        # 2) Enrich features + Stats
        df_en = enrich_features(df)
        stats = basic_stats(df_en)
        print("\nBasic stats:")
        print(stats)

        # 3) Resample ví dụ (5 phút & 1 giờ)
        df_5m = resample_ohlcv(df_en, "5min")
        df_1h = resample_ohlcv(df_en, "1H")
        print(f"\nResample examples: {sym} | 5m rows={len(df_5m):,} | 1h rows={len(df_1h):,}")

        # 4) Plot
        plot_price_with_ma(df_en, sym, save=save_plots)
        plot_volume(df_en, sym, save=save_plots)
        plot_logret_hist(df_en, sym, save=save_plots)

        # Lưu vào store để sau cùng vẽ corr heatmap giữa các symbol
        store[sym] = {"df": df_en, "path": fp, "df5m": df_5m, "df1h": df_1h}

    # 5) Corr heatmap giữa các symbol (log-returns)
    plot_corr_heatmap({k: v["df"] for k, v in store.items()}, save=save_plots)

    # 6) Tổng hợp bảng tóm tắt cho tất cả symbol
    print("\n== Summary table across symbols ==")
    rows = []
    for sym, obj in store.items():
        s = basic_stats(obj["df"])
        s["symbol"] = sym
        rows.append(s)
    if rows:
        summary = pd.DataFrame(rows).set_index("symbol")
        # Chuyển start_ts/end_ts từ ns → datetime chỉ để hiển thị đẹp
        for col in ["start_ts", "end_ts"]:
            summary[col] = pd.to_datetime(summary[col])
        print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binance OHLCV Minute Data Explorer")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Thư mục chứa CSV theo symbol")
    parser.add_argument("--candle-level", type=str, default=DEFAULT_CANDLE, help="Khung nến, ví dụ: 1m, 5m, 1h, 1d")
    parser.add_argument("--symbols", type=str, nargs="+", default=DEFAULT_SYMBOLS, help="Danh sách symbol")
    parser.add_argument("--save-plots", action="store_true", help="Lưu hình vào ./_out thay vì chỉ hiển thị")
    args = parser.parse_args()
    main(args)
