from ace_rl.core.cube_feature import (
    save_cube, load_cube
)
from ace_rl.core.candle_feat_ops import (
    make_return, make_volatility, make_ema_ratio,
    make_atr, make_rsi, make_zscore,
    make_volume_ratio, make_taker_buy_ratio,
)
from pathlib import Path
import xarray as xr

root = Path("../../../work/processed/binance").expanduser()
cached_cube = xr.open_zarr(
    (root / "data" / "zarr_cube_intersection").as_posix(),
    consolidated=True,  # flip to False if you ever disable consolidation
)
print(cached_cube)

cube = cached_cube 


# ví dụ build và save một vài feature
root = Path("../../../work/processed/binance")
base = cube

print("base dims:", base.dims)
print("base data variables:", list(base.data_vars))

# xem toạ độ symbol
sym = base.coords["symbol"]
print("symbol dtype:", sym.dtype, "encoding:", sym.encoding)
print("symbol type:", type(sym.values).__name__)
print("symbol values (full):", sym.values.tolist())

# nếu muốn, soi cả feature/time
feat = base.coords["feature"]
print("feature dtype:", feat.dtype, "encoding:", feat.encoding)
print("feature values:", feat.values.tolist())
time = base.coords["time"]
print("time dtype:", time.dtype, "len:", time.size)

# ======== Tạo và lưu các feature ========

name, ds_ret5   = make_return(base, lookback=5)
save_cube(ds_ret5, root, subdir="features", name=name)

name, ds_vol60  = make_volatility(base, window=60)
save_cube(ds_vol60, root, subdir="features", name=name)

name, ds_emar20 = make_ema_ratio(base, span=20)
save_cube(ds_emar20, root, subdir="features", name=name)

name, ds_atr14  = make_atr(base, span=14)
save_cube(ds_atr14, root, subdir="features", name=name)

name, ds_rsi14  = make_rsi(base, span=14)
save_cube(ds_rsi14, root, subdir="features", name=name)

name, ds_zC240  = make_zscore(base, source_feature="Close", window=240, strict_tminus1=True)
save_cube(ds_zC240, root, subdir="features", name=name)

name, ds_vr60   = make_volume_ratio(base, window=60)
save_cube(ds_vr60, root, subdir="features", name=name)

name, ds_tbr    = make_taker_buy_ratio(base)
save_cube(ds_tbr, root, subdir="features", name=name)


