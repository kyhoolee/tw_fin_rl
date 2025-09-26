"""Utilities for building the raw time×symbol×feature cube from per-symbol Zarr caches.

Mục tiêu (Phase 2):
[STEP 1] Load các Zarr per-symbol (đã chuẩn hoá ở Phase 1).
[STEP 2] Căn chỉnh về cùng lịch thời gian (calendar) và thứ tự feature (feature order).
[STEP 3] Tạo xarray.Dataset dạng 3D (time × symbol × feature), kèm mask tuỳ chọn, và (tuỳ chọn) lưu Zarr.

Ghi chú:
- calendar_mode='range_intersection' sẽ "cắt block" theo [max(start_time), min(end_time)] của tất cả symbol
  để giảm NaN và có một vùng fully-aligned lớn nhất (phù hợp nhu cầu của bạn).
- Nếu đã có meta cố định (calendar.npy, feature_order.json), nên chỉ định để đảm bảo reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Literal

import numpy as np
import xarray as xr
import zarr


__all__ = [
    "CubeBuilderConfig",
    "SymbolZarrSlice",
    "build_raw_cube",
    "infer_and_dump_meta",
]


@dataclass(slots=True)
class CubeBuilderConfig:
    """Cấu hình cho :func:`build_raw_cube`.

    Thuộc tính chính
    ----------------
    per_symbol_root : Thư mục chứa các store dạng {symbol}.zarr (Phase 1 output).
    calendar_path   : (Tuỳ chọn) file .npy chứa calendar chuẩn (UTC ns). Nếu None: sẽ suy luận.
    feature_order_path : (Tuỳ chọn) JSON list tên feature theo thứ tự chuẩn. Nếu None: sẽ suy luận.
    symbols         : (Tuỳ chọn) danh sách symbol. Nếu None: tự quét *.zarr trong per_symbol_root.
    dtype           : Kiểu số cho mảng values (mặc định float32 cho nhẹ RAM/IO).
    chunk_time, chunk_symbol, chunk_feature : Chunk size theo từng trục cho xarray/dask.
    compute_mask    : Có sinh biến 'mask' (True nếu giá trị là số hữu hạn) hay không.
    output_path     : (Tuỳ chọn) nếu cung cấp, ghi dataset ra .zarr và đồng thời trả về dataset.
    consolidate_metadata : Consolidate metadata khi ghi Zarr (đọc lại nhanh hơn).
    use_dask        : Dùng dask.array để lazy-load từ Zarr thay vì nạp hết (hữu ích với dữ liệu lớn).

    calendar_mode   : Cách suy luận calendar khi không có calendar_path:
        - 'range_intersection' (mặc định): Lấy khoảng giao [max(min(times_i)), min(max(times_i))] rồi
          lọc calendar trong khoảng đó. Thực tế thường cho block liên tục lớn nhất có mặt ở mọi symbol.
        - 'union': Hợp tất cả mốc thời gian của mọi symbol (nhiều NaN hơn).
        - 'exact_intersection': Giao tập mốc xuất hiện ở MỌI symbol (khắt khe nhất, ít NaN nhất).
    """

    per_symbol_root: Path
    calendar_path: Path | None = None
    feature_order_path: Path | None = None
    symbols: Sequence[str] | None = None
    dtype: np.dtype | type = np.float32
    chunk_time: int | None = 16384
    chunk_symbol: int | None = None
    chunk_feature: int | None = None
    compute_mask: bool = True
    output_path: Path | None = None
    consolidate_metadata: bool = True
    use_dask: bool = False
    calendar_mode: Literal["range_intersection", "union", "exact_intersection"] = "range_intersection"

    def iter_symbols(self) -> Iterable[str]:
        """Liệt kê symbol: ưu tiên danh sách cung cấp; nếu không, quét *.zarr."""
        if self.symbols:
            return list(self.symbols)
        stores = sorted(Path(self.per_symbol_root).glob("*.zarr"))
        return [path.stem for path in stores]

    def chunk_dict(self) -> dict[str, int]:
        """Đóng gói dict chunk theo từng trục cho xarray."""
        chunks: dict[str, int] = {}
        if self.chunk_time:
            chunks["time"] = int(self.chunk_time)
        if self.chunk_symbol:
            chunks["symbol"] = int(self.chunk_symbol)
        if self.chunk_feature:
            chunks["feature"] = int(self.chunk_feature)
        return chunks


@dataclass(slots=True)
class SymbolZarrSlice:
    """Đại diện cho một lát dữ liệu của 1 symbol từ Zarr store."""

    symbol: str
    times: np.ndarray          # datetime64[ns]
    features: np.ndarray       # tên feature (str)
    values: object             # np.ndarray hoặc dask.array tuỳ use_dask
    path: Path

    @classmethod
    def from_store(
        cls,
        symbol: str,
        store_path: Path,
        dtype: np.dtype | type = np.float32,
        use_dask: bool = False,
    ) -> "SymbolZarrSlice":
        """Mở Zarr store, đọc time/features/values. Hỗ trợ dask lazy khi use_dask=True."""
        if not store_path.exists():
            raise FileNotFoundError(f"Missing Zarr store for {symbol}: {store_path}")

        group = zarr.open(store_path.as_posix(), mode="r")
        raw_times = np.asarray(group["time"][:], dtype="datetime64[ns]")
        raw_features = np.asarray(group["features"][:], dtype=str)

        if use_dask:
            try:
                import dask.array as da  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - optional path
                raise RuntimeError(
                    "CubeBuilderConfig.use_dask=True requires dask[array] to be installed"
                ) from exc
            data = da.from_zarr(store_path.as_posix(), component="values").astype(dtype)
        else:
            data = np.asarray(group["values"][:], dtype=dtype)

        return cls(
            symbol=symbol,
            times=raw_times,
            features=raw_features,
            values=data,
            path=store_path,
        )

    def to_xarray(
        self,
        calendar: np.ndarray,
        feature_order: Sequence[str],
        dtype: np.dtype | type = np.float32,
    ) -> xr.DataArray:
        """Chuyển lát dữ liệu 1 symbol sang DataArray, rồi reindex về (calendar, feature_order)."""
        da = xr.DataArray(
            self.values,
            coords={"time": self.times, "feature": self.features},
            dims=("time", "feature"),
            name="values",
        ).astype(dtype)

        # Căn lại theo calendar/feature chuẩn -> đảm bảo cùng shape cho concat nhiều symbol.
        aligned = da.reindex({"time": calendar, "feature": list(feature_order)})
        return aligned.expand_dims(symbol=[self.symbol])


def _resolve_calendar(
    calendar_path: Path | None,
    slices: Sequence[SymbolZarrSlice],
    mode: Literal["range_intersection", "union", "exact_intersection"],
) -> np.ndarray:
    """Xác định calendar (trục thời gian) dùng chung.

    Ưu tiên đọc từ calendar_path nếu có; nếu không, suy luận theo 'mode':
      - range_intersection: cắt theo [max(start_i), min(end_i)] trên calendar hợp (union)
      - union: hợp tất cả mốc thời gian
      - exact_intersection: giao tập mốc xuất hiện ở mọi symbol (khắt khe, ít NaN nhất)
    """
    if calendar_path and calendar_path.exists():
        calendar = np.load(calendar_path)
        return np.asarray(calendar, dtype="datetime64[ns]")

    if not slices:
        raise ValueError("No symbol slices supplied to resolve the calendar")

    # Hợp tất cả mốc time trước (tiện để lọc/rút gọn theo mode)
    union_calendar = np.sort(np.unique(np.concatenate([sl.times for sl in slices])))

    if mode == "union":
        return union_calendar

    # Tính start/end mỗi symbol
    starts = [np.asarray(sl.times)[0] for sl in slices]
    ends = [np.asarray(sl.times)[-1] for sl in slices]
    max_start = np.max(starts)
    min_end = np.min(ends)

    if max_start > min_end:
        raise ValueError(
            "Empty range for 'range_intersection': max(start) > min(end). "
            "Dữ liệu các symbol không giao nhau theo thời gian."
        )

    if mode == "range_intersection":
        # Lọc mốc nằm trong [max_start, min_end] từ lịch hợp.
        mask = (union_calendar >= max_start) & (union_calendar <= min_end)
        return union_calendar[mask]

    # mode == "exact_intersection": giao tập mốc xuất hiện ở MỌI symbol
    intersect = None
    for sl in slices:
        times_i = np.unique(sl.times)
        intersect = times_i if intersect is None else np.intersect1d(intersect, times_i, assume_unique=False)
        if intersect.size == 0:
            break
    return np.sort(intersect) if intersect is not None else np.array([], dtype="datetime64[ns]")


def _resolve_feature_order(
    feature_order_path: Path | None,
    slices: Sequence[SymbolZarrSlice],
) -> List[str]:
    """Xác định thứ tự feature chuẩn.

    - Nếu có file JSON: dùng trực tiếp (ổn định & reproducible).
    - Nếu không: lấy thứ tự từ symbol đầu tiên (giả định Phase 1 đã đồng nhất schema).
    """
    if feature_order_path and feature_order_path.exists():
        raw = json.loads(feature_order_path.read_text())
        if not isinstance(raw, list):
            raise ValueError("feature_order.json must contain a JSON list")
        return [str(item) for item in raw]

    if not slices:
        raise ValueError("No symbol slices supplied to resolve feature order")

    base = list(slices[0].features)
    return base


def build_raw_cube(config: CubeBuilderConfig) -> xr.Dataset:
    """Tạo dataset 3D raw cube (time × symbol × feature) từ các Zarr per-symbol.

    Quy trình:
    [STEP 1] Load tất cả symbol -> SymbolZarrSlice
    [STEP 2] Chuẩn hoá trục: calendar (theo calendar_mode hoặc file .npy) & feature_order (file .json hoặc từ store)
    [STEP 3] Reindex, concat theo symbol -> tạo 'values' (+ 'mask' tuỳ chọn), chunk, và (tuỳ chọn) ghi ra Zarr.
    """
    # [STEP 1] Load slices
    symbols = list(config.iter_symbols())
    if not symbols:
        raise ValueError(f"No symbols discovered under {config.per_symbol_root}")

    slices = [
        SymbolZarrSlice.from_store(
            symbol=symbol,
            store_path=Path(config.per_symbol_root) / f"{symbol}.zarr",
            dtype=config.dtype,
            use_dask=config.use_dask,
        )
        for symbol in symbols
    ]

    # [STEP 2] Chuẩn hoá calendar & feature order
    calendar = _resolve_calendar(config.calendar_path, slices, config.calendar_mode)
    if calendar.size == 0:
        raise ValueError("Resolved calendar is empty. Hãy kiểm tra calendar_mode hoặc dữ liệu đầu vào.")
    feature_order = _resolve_feature_order(config.feature_order_path, slices)

    # [STEP 3] Reindex từng symbol -> concat -> tạo dataset
    data_arrays = [sl.to_xarray(calendar, feature_order, dtype=config.dtype) for sl in slices]
    values = xr.concat(data_arrays, dim="symbol").transpose("time", "symbol", "feature")

    data_vars = {"values": values}

    if config.compute_mask:
        # mask = True nếu giá trị hữu hạn (không NaN, không inf)
        mask = xr.apply_ufunc(
            np.isfinite,
            values,
            dask="allowed",
            keep_attrs=False,
        )
        data_vars["mask"] = mask

    cube = xr.Dataset(data_vars)

    # Chunk theo cấu hình (tối ưu đọc/ghi và training)
    chunk_map = config.chunk_dict()
    if chunk_map:
        cube = cube.chunk(chunk_map)

    # Ghi nguồn meta để trace về sau
    cube.attrs["calendar_source"] = str(config.calendar_path) if config.calendar_path else f"inferred:{config.calendar_mode}"
    cube.attrs["feature_order_source"] = str(config.feature_order_path) if config.feature_order_path else "inferred"

    # (Tuỳ chọn) Ghi ra Zarr
    if config.output_path:
        target = config.output_path
        target.parent.mkdir(parents=True, exist_ok=True)
        cube.to_zarr(
            target.as_posix(),
            mode="w",
            consolidated=config.consolidate_metadata,
        )

    return cube


def infer_and_dump_meta(
    config: CubeBuilderConfig,
    calendar_out: Path,
    feature_order_out: Path,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Suy luận nhanh meta khi CHƯA có calendar.npy & feature_order.json.

    - Lấy 1 symbol đại diện để cố định thứ tự feature (giả định Phase 1 đồng nhất).
    - Calendar mặc định = times của symbol mẫu (ổn định & đơn giản).
      Nếu bạn muốn dùng khoảng giao của TẤT CẢ symbol, có thể đọc tất cả symbol rồi
      tự tính theo logic _resolve_calendar(..., mode='range_intersection') và np.save.
    """
    symbols = list(config.iter_symbols())
    if not symbols:
        raise ValueError("Cannot infer meta without any symbols")

    sample = SymbolZarrSlice.from_store(
        symbol=symbols[0],
        store_path=Path(config.per_symbol_root) / f"{symbols[0]}.zarr",
        dtype=config.dtype,
        use_dask=False,
    )

    calendar = sample.times
    features = sample.features.tolist()

    if calendar_out.exists() and not overwrite:
        raise FileExistsError(f"calendar file already exists: {calendar_out}")
    if feature_order_out.exists() and not overwrite:
        raise FileExistsError(f"feature order file already exists: {feature_order_out}")

    calendar_out.parent.mkdir(parents=True, exist_ok=True)
    feature_order_out.parent.mkdir(parents=True, exist_ok=True)

    np.save(calendar_out, calendar)
    feature_order_out.write_text(json.dumps(features, indent=2))

    return calendar_out, feature_order_out
