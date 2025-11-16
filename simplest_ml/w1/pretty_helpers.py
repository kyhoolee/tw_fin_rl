import os
import re
import glob
from typing import Optional, Sequence

import pandas as pd

# Default mapping from human token names to Binance spot symbols
DEFAULT_TOKEN_TO_SYMBOL = {
    "BTC": "BTCUSDT",
    "BNB": "BNBUSDT",
    "SOL": "SOLUSDT",
    "ETH": "ETHUSDT",
    "AVAX": "AVAXUSDT",
    "TRON": "TRXUSDT",
    "ARBtrium": "ARBUSDT",
    "Sui": "SUIUSDT",
    "Polygon": "MATICUSDT",
}


def resolve_symbol(token_or_symbol: str, mapping: Optional[dict] = None) -> str:
    """
    Resolve a human-readable token name (e.g., "BTC", "Polygon") to a Binance
    symbol (e.g., "BTCUSDT", "MATICUSDT"). If `token_or_symbol` already looks
    like a symbol (endswith "USDT"), returns the uppercased input.
    """
    if token_or_symbol is None:
        raise ValueError("token_or_symbol must be provided")
    s = str(token_or_symbol).strip()
    if not s:
        raise ValueError("token_or_symbol must not be empty")
    # If already a symbol
    if s.upper().endswith("USDT"):
        return s.upper()
    mapping = mapping or DEFAULT_TOKEN_TO_SYMBOL
    if s in mapping:
        return mapping[s]
    # Try case-insensitive lookup
    for k, v in mapping.items():
        if k.lower() == s.lower():
            return v
    # As a last resort, assume it's a base asset and append USDT
    return f"{s.upper()}USDT"


def _select_best_parquet_for_symbol(
    symbol: str,
    data_dir: str,
    interval: str,
    start: Optional[str],
    end: Optional[str],
):
    """
    Internal helper to find the most relevant parquet file for `symbol`.
    Preference order:
      1) Exact match with given start/end if provided
      2) Any file for the symbol+interval, pick the one with the latest end date
    Returns absolute path or None.
    """
    sym_l = symbol.lower()
    # 1) Exact match
    if start and end:
        exact = os.path.join(
            data_dir, f"binance_{sym_l}_{interval}_{start}_{end}.parquet"
        )
        if os.path.exists(exact):
            return exact
    # 2) Any match for symbol+interval
    pattern = os.path.join(data_dir, f"binance_{sym_l}_{interval}_*.parquet")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return None
    # Parse end date to choose the latest
    best = None
    best_end = None
    rx = re.compile(rf"binance_{re.escape(sym_l)}_{re.escape(interval)}_([0-9]{{8}})_([0-9]{{8}})\.parquet$")
    for p in candidates:
        m = rx.search(os.path.basename(p))
        if not m:
            # fallback: keep last
            best = p
            continue
        end_str = m.group(2)
        if (best_end is None) or (end_str > best_end):
            best_end = end_str
            best = p
    return best


def load_token_df(
    token_or_symbol: str,
    data_dir: str = "data",
    interval: str = "1h",
    start: Optional[str] = "20231030",
    end: Optional[str] = "20251030",
) -> pd.DataFrame:
    """
    Load a single token's DataFrame from previously saved Parquet files.

    Parameters:
      - token_or_symbol: e.g., "BTC", "Polygon", "ETHUSDT".
      - data_dir: directory containing parquet files.
      - interval: e.g., "1h".
      - start/end: date strings yyyymmdd. If the exact span is not found,
        the function falls back to the latest available file for the symbol+interval.

    Returns: pandas DataFrame.
    Raises: FileNotFoundError if no matching file is found.
    """
    symbol = resolve_symbol(token_or_symbol)
    p = _select_best_parquet_for_symbol(symbol, data_dir, interval, start, end)
    if not p:
        raise FileNotFoundError(
            f"No parquet found for {symbol} in {data_dir} (interval={interval}, start={start}, end={end})."
        )
    return pd.read_parquet(p)



def pretty_print_sample(
    df: pd.DataFrame,
    head: int = 3,
    tail: int = 3,
    time_preference: Sequence[str] = ("open_time", "index", "close_time"),
) -> None:
    """
    Pretty print first `head` and last `tail` rows of a DataFrame of klines.

    - Expects columns: open, high, low, close, volume; and optionally open_time / close_time.
    - `time_preference` controls which time field to display (open_time, index, close_time).

    Requires `prettytable` to be installed.
    """
    try:
        from prettytable import PrettyTable
    except Exception as e:
        raise ImportError(
            "prettytable is required. Install with: pip install prettytable"
        ) from e

    table = PrettyTable()
    table.field_names = [
        "Index",
        "Open Time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    n = len(df)

    def pick_time(row):
        for key in time_preference:
            if key.lower() in ("index", "idx"):
                try:
                    return row.Index
                except Exception:
                    continue
            val = getattr(row, key, None)
            if val is not None:
                return val
        # Fallback
        return getattr(row, "Index", None)

    for i, row in enumerate(df.itertuples()):
        if i < head or i >= max(0, n - tail):
            table.add_row([
                i,
                pick_time(row),
                row.open,
                row.high,
                row.low,
                row.close,
                row.volume,
            ])

    print(table)


def find_downloaded_parquet_files(
    data_dir: str = "data",
    interval: str = "1h",
    start: Optional[str] = "20231030",
    end: Optional[str] = "20251030",
):
    """
    Return list of Parquet files matching naming scheme:
      data/binance_{symbol}_{interval}_{start}_{end}.parquet

    If `start`/`end` are None or no exact matches found, falls back to
    any files matching: data/binance_*_{interval}_*.parquet
    """
    pattern = (
        f"binance_*_{interval}_{start}_{end}.parquet"
        if start and end
        else f"binance_*_{interval}_*.parquet"
    )
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not paths:
        paths = sorted(glob.glob(os.path.join(data_dir, f"binance_*_{interval}_*.parquet")))
    return paths


def read_and_pretty_print_all_downloaded(
    data_dir: str = "data",
    interval: str = "1h",
    start: Optional[str] = "20231030",
    end: Optional[str] = "20251030",
    head: int = 3,
    tail: int = 3,
) -> None:
    """
    Read all downloaded Parquet files for the given `interval` and optional
    `start`/`end` span, then pretty print a sample for each symbol.
    """
    paths = find_downloaded_parquet_files(
        data_dir=data_dir, interval=interval, start=start, end=end
    )

    if not paths:
        print(
            f"No Parquet files found in {data_dir} for interval={interval} and start/end criteria."
        )
        return

    for p in paths:
        base = os.path.basename(p)
        # Expected: binance_{symbol}_{interval}_{start}_{end}.parquet
        m = re.match(
            r"binance_(.+?)_([^_]+)_([0-9]{8})_([0-9]{8})\.parquet$", base
        )
        if m:
            symbol = m.group(1).upper()
            interval_file = m.group(2)
            start_file = m.group(3)
            end_file = m.group(4)
        else:
            # Fallback parsing
            m2 = re.match(r"binance_(.+?)_", base)
            symbol = m2.group(1).upper() if m2 else base
            interval_file = interval
            start_file = start or ""
            end_file = end or ""

        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print(f"Failed to read {base}: {e}")
            continue

        print(
            f"\n=== {symbol} | {interval_file} | {start_file}->{end_file} | rows={len(df)} ==="
        )
        pretty_print_sample(df, head=head, tail=tail)
