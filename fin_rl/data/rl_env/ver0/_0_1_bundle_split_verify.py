import pandas as pd
from pathlib import Path
import yaml, json
import numpy as np
import prettytable as pt


def show_head_tail(df, n=3):
    print(f"\n== head/tail per symbol (n={n}) ==")
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("open_time")
        print(f"\n[{sym}] head:")
        # print(g.head(n)[["open_time","Open","High","Low","Close","Volume"]])
        # print(g.head(n))
        # print by prettytable
        # print(f"[{sym}] tail:")
        # print(g.tail(n))
        # print(g.tail(n)[["open_time","Open","High","Low","Close","Volume"]])

        tab = pt.PrettyTable()
        tab.field_names = g.columns.tolist()
        for _, row in g.head(n).iterrows():
            tab.add_row(row.tolist())

        print(tab)



def show_ts(df, ts, cols=("Open","High","Low","Close","Volume")):
    g = df[df["open_time"]==ts].sort_values("symbol")
    if g.empty:
        print(f"[{ts}] not found.")
        return
    print(f"\n== snapshot @ {ts} ==")
    print(g[["symbol","open_time",*cols]])

def show_random_ts(df, k=3, seed=0):
    rng = np.random.default_rng(seed)
    cand = df["open_time"].unique()
    for ts in rng.choice(cand, size=min(k, len(cand)), replace=False):
        show_ts(df, ts)



def around_ts(df, ts, n=2, cols=("Open","High","Low","Close")):
    # lấy window thời gian theo MultiIndex (open_time, symbol)
    idx = df.set_index(["open_time","symbol"]).sort_index()
    # list mốc xung quanh ts
    uniq = idx.index.get_level_values(0).unique().sort_values()
    if ts not in uniq:
        print(f"{ts} not in split timestamps"); return
    pos = uniq.get_loc(ts)
    lo, hi = max(0, pos-n), min(len(uniq)-1, pos+n)
    window_ts = uniq[lo:hi+1]
    snap = idx.loc[window_ts].reset_index().sort_values(["open_time","symbol"])
    print(f"\n== around {ts} (±{n} bars) ==")
    print(snap[["open_time","symbol",*cols]])



def check_bar_spacing(df, expected_freq="1min", per_symbol=True, show=5):
    print(f"\n== spacing check (expected={expected_freq}) ==")
    if per_symbol:
        for sym, g in df.groupby("symbol"):
            g = g.sort_values("open_time")
            deltas = g["open_time"].diff().dropna()
            bad = deltas[deltas != pd.Timedelta(expected_freq)]
            print(f"[{sym}] rows={len(g)} | bad_deltas={len(bad)}")
            if not bad.empty:
                print("  sample bad (first):")
                print(g.loc[bad.index[:show], ["open_time"]])
    else:
        g = df.sort_values(["open_time","symbol"]).drop_duplicates("open_time")
        deltas = g["open_time"].diff().dropna()
        bad = deltas[deltas != pd.Timedelta(expected_freq)]
        print(f"[ALL] unique_ts={g['open_time'].nunique()} | bad_deltas={len(bad)}")
        if not bad.empty:
            print(g.loc[bad.index[:show], ["open_time"]])



def show_boundaries(train, test, unseen, meta, k_neighbors=2):
    t0 = pd.to_datetime(meta["effective_ranges"]["train"]["start"])
    t1 = pd.to_datetime(meta["effective_ranges"]["train"]["end"])
    s0 = pd.to_datetime(meta["effective_ranges"]["test"]["start"])
    s1 = pd.to_datetime(meta["effective_ranges"]["test"]["end"])
    u0 = pd.to_datetime(meta["effective_ranges"]["unseen"]["start"])
    u1 = pd.to_datetime(meta["effective_ranges"]["unseen"]["end"])

    # in vài bar quanh các mốc
    for name, df, ts in [
        ("train_start", train, t0),
        ("train_end",   train, t1),
        ("test_start",  test,  s0),
        ("test_end",    test,  s1),
        ("unseen_start",unseen,u0),
        ("unseen_end",  unseen,u1),
    ]:
        print(f"\n--- {name}: {ts} ---")
        around_ts(df, ts, n=k_neighbors)

    # embargo window
    emb = meta.get("embargo", {})
    if isinstance(emb, dict) and emb.get("timedelta"):
        E = pd.to_timedelta(emb["timedelta"])
        lo, hi = s0 - E, s1 + E
        print(f"\nEmbargo window: [{lo} .. {hi}]  | K bars = {emb.get('bars')}")
        viol = train[(train["open_time"] >= lo) & (train["open_time"] <= hi)]
        print("train rows in embargo window:", len(viol))




root = Path("./work/bundle_out/bundles/v1_minutes/bundle_01")
p_train, p_test, p_unseen = root/"splits/train.parquet", root/"splits/test.parquet", root/"splits/unseen.parquet"

train = pd.read_parquet(p_train)
test  = pd.read_parquet(p_test)
unseen= pd.read_parquet(p_unseen)


with open(root/"META.yaml") as f: meta = yaml.safe_load(f)

# 1) xem head/tail
show_head_tail(train, n=2)
show_head_tail(test,  n=2)

# 2) soi vài timestamp ngẫu nhiên trong test
show_random_ts(test, k=3, seed=42)

# 3) soi quanh mốc ranh giới
show_boundaries(train, test, unseen, meta, k_neighbors=2)

# 4) spacing check theo symbol
#    NOTE: nếu candle-level là "1m" → expected_freq="1min"; "5m"→"5min"; "1h"→"1H"
check_bar_spacing(test, expected_freq="1min", per_symbol=True)


def check_split(name, df, symbols):
    print(f"\n== {name.upper()} ==")
    # 1) duplicates
    dups = df.duplicated(subset=["symbol","open_time"]).sum()
    print("dups(symbol,open_time):", int(dups))

    # 2) unique ts & per-symbol sizes
    uts = df["open_time"].nunique()
    print("unique_ts:", uts)
    for sym, g in df.groupby("symbol"):
        print(f"  {sym}: rows={len(g)} | ts[{g['open_time'].min()} .. {g['open_time'].max()}]")
        assert len(g) == uts, f"{name}:{sym} not aligned (len != unique_ts)"

    # 3) ensure all symbols present
    miss = set(symbols) - set(df["symbol"].unique())
    print("missing_symbols:", miss)

with open(root/"META.yaml","r") as f:
    meta = yaml.safe_load(f)
symbols = meta["symbols"]
print("symbols:", symbols)
print("embargo:", meta["embargo"])

check_split("train", train, symbols)
check_split("test",  test, symbols)
check_split("unseen", unseen, symbols)

# 4) Verify embargo window actually removed from train
s0 = pd.to_datetime(meta["effective_ranges"]["test"]["start"])
s1 = pd.to_datetime(meta["effective_ranges"]["test"]["end"])
E  = pd.to_timedelta(meta["embargo"]["timedelta"]) if isinstance(meta["embargo"], dict) else pd.to_timedelta("0s")
lo, hi = s0 - E, s1 + E

viol = train[(train["open_time"] >= lo) & (train["open_time"] <= hi)]
print(f"\nembargo window: [{lo} .. {hi}]")
print("train rows in embargo window:", len(viol))  # expect 0
