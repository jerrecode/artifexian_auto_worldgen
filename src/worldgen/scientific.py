from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from .field_schema import build_field_catalog


def arrays_to_xarray(arrays: Mapping[str, np.ndarray]):
    """Convert canonical world arrays to an xarray Dataset on demand.

    xarray is optional so the core generator remains lightweight. Variable
    dimensions are taken from the same field catalog written with every world.
    """
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "Scientific export requires xarray. Install with: pip install 'artifexian-auto-worldgen[scientific]'"
        ) from exc

    arrs = {k: np.asarray(v) for k, v in arrays.items()}
    catalog = build_field_catalog(arrs)
    coords = {
        "lat": arrs["lat"],
        "lon": arrs["lon"],
        "month": np.arange(1, 13, dtype=np.int16),
        "channel": np.arange(3, dtype=np.int8),
        "xyz": np.asarray(["x", "y", "z"]),
    }
    data_vars = {}
    dim_sizes: dict[str, int] = {k: int(np.asarray(v).size) for k, v in coords.items()}
    for name, a in arrs.items():
        if name in {"lat", "lon"}:
            continue
        dims = tuple(catalog[name]["dimensions"])
        normalized_dims: list[str] = []
        for axis, dim in enumerate(dims):
            size = int(a.shape[axis])
            existing = dim_sizes.get(dim)
            if existing is not None and existing != size:
                dim = f"{name}_{dim}"
            dim_sizes[dim] = size
            normalized_dims.append(dim)
        attrs = {
            "role": catalog[name]["role"],
            "description": catalog[name]["description"],
        }
        if catalog[name]["units"] is not None:
            attrs["units"] = catalog[name]["units"]
        data_vars[name] = (tuple(normalized_dims), a, attrs)
    ds = xr.Dataset(data_vars=data_vars, coords={k: v for k, v in coords.items() if k in dim_sizes})
    ds.attrs["generator"] = "artifexian-auto-worldgen"
    ds.attrs["conventions"] = "CF-inspired; equirectangular storage with spherical topology"
    return ds


def export_zarr(arrays: Mapping[str, np.ndarray], path: str | Path, *, mode: str = "w") -> Path:
    ds = arrays_to_xarray(arrays)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Chunk monthly/global rasters along time and moderate spatial tiles when possible.
    chunks = {}
    if "month" in ds.dims:
        chunks["month"] = 1
    if "lat" in ds.dims:
        chunks["lat"] = min(256, int(ds.sizes["lat"]))
    if "lon" in ds.dims:
        chunks["lon"] = min(256, int(ds.sizes["lon"]))
    if chunks:
        ds = ds.chunk(chunks)
    ds.to_zarr(target, mode=mode)
    return target


def export_netcdf(arrays: Mapping[str, np.ndarray], path: str | Path) -> Path:
    ds = arrays_to_xarray(arrays)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(target)
    return target


def export_geotiff_rasters(arrays: Mapping[str, np.ndarray], root: str | Path) -> tuple[Path, ...]:
    """Export numeric 2-D global rasters as EPSG:4326 GeoTIFF files."""
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError as exc:
        raise RuntimeError(
            "GeoTIFF export requires rasterio. Install with: pip install 'artifexian-auto-worldgen[gis]'"
        ) from exc

    arrs = {k: np.asarray(v) for k, v in arrays.items()}
    h, w = len(arrs["lat"]), len(arrs["lon"])
    transform = from_bounds(-180.0, -90.0, 180.0, 90.0, w, h)
    out_root = Path(root)
    out_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, a in arrs.items():
        if a.shape != (h, w) or a.dtype.kind in {"U", "S", "O"}:
            continue
        target = out_root / f"{name}.tif"
        data = a.astype(np.uint8 if a.dtype == bool else a.dtype, copy=False)
        with rasterio.open(
            target,
            "w",
            driver="GTiff",
            width=w,
            height=h,
            count=1,
            dtype=data.dtype,
            crs="EPSG:4326",
            transform=transform,
            compress="deflate",
            tiled=True,
        ) as dst:
            dst.write(data, 1)
        written.append(target)
    return tuple(written)


def load_npz_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}
