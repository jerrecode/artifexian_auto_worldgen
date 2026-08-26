from __future__ import annotations

"""Bounded-memory 2-D tiling for spherical equirectangular fields.

Unlike a clipped rectangular raster tile, a spherical tile must preserve two
non-Cartesian boundary rules: longitude is periodic and crossing either pole
reflects latitude while rotating longitude by 180 degrees.  ``SphericalTile``
therefore stores virtual bounds and maps them through the canonical topology at
extraction time instead of pretending every halo is a contiguous NumPy slice.
"""

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np

from .topology import _map_spherical_lattice_indices


@dataclass(slots=True, frozen=True)
class SphericalTile:
    """One deterministic core tile plus a spherical halo.

    ``y0:y1`` and ``x0:x1`` are core coordinates in the global raster.  Halo
    coordinates are virtual and may be negative, beyond the southern edge, or
    beyond the longitude seam.  :meth:`extract` resolves them with spherical
    pole/seam topology.
    """

    index: int
    tile_row: int
    tile_col: int
    shape: tuple[int, int]
    y0: int
    y1: int
    x0: int
    x1: int
    halo_y: int = 0
    halo_x: int = 0

    @property
    def core(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)

    @property
    def core_shape(self) -> tuple[int, int]:
        return self.y1 - self.y0, self.x1 - self.x0

    @property
    def expanded_shape(self) -> tuple[int, int]:
        h, w = self.core_shape
        return h + 2 * self.halo_y, w + 2 * self.halo_x

    @property
    def crop(self) -> tuple[slice, slice]:
        h, w = self.core_shape
        return (
            slice(self.halo_y, self.halo_y + h),
            slice(self.halo_x, self.halo_x + w),
        )

    def mapped_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return global y/x indices for every expanded-halo sample."""
        vy = np.arange(self.y0 - self.halo_y, self.y1 + self.halo_y, dtype=np.int64)
        vx = np.arange(self.x0 - self.halo_x, self.x1 + self.halo_x, dtype=np.int64)
        yy, xx = np.meshgrid(vy, vx, indexing="ij")
        return _map_spherical_lattice_indices(yy, xx, self.shape)

    def extract(self, values: np.ndarray) -> np.ndarray:
        """Extract a spherical-halo tile from the final two array dimensions."""
        a = np.asarray(values)
        if a.shape[-2:] != self.shape:
            raise ValueError(f"last two dimensions must be {self.shape}, got {a.shape}")
        yy, xx = self.mapped_indices()
        return a[..., yy, xx]

    def crop_core(self, expanded_values: np.ndarray) -> np.ndarray:
        """Crop an expanded kernel result back to this tile's core."""
        a = np.asarray(expanded_values)
        if a.shape[-2:] != self.expanded_shape:
            raise ValueError(
                f"expanded trailing shape must be {self.expanded_shape}, got {a.shape[-2:]}"
            )
        cy, cx = self.crop
        return a[..., cy, cx]


def auto_tile_shape(
    shape: tuple[int, int],
    dtype=np.float32,
    *,
    target_mb: float = 32.0,
    arrays_in_flight: int = 6,
    minimum_edge: int = 16,
    maximum_edge: int = 512,
) -> tuple[int, int]:
    """Choose a genuine 2-D chunk from an approximate working-set budget.

    Equirectangular world rasters use equal angular spacing in latitude/longitude,
    so near-square cell chunks are a useful default.  The calculation budgets for
    all arrays expected to be simultaneously resident in the kernel.
    """
    h, w = map(int, shape)
    if h <= 0 or w <= 0:
        raise ValueError("shape dimensions must be positive")
    itemsize = np.dtype(dtype).itemsize
    budget = max(1, int(float(target_mb) * 1024**2))
    denom = max(1, itemsize * max(1, int(arrays_in_flight)))
    cells = max(1, budget // denom)
    edge = max(1, int(np.sqrt(cells)))
    edge = max(1, min(int(maximum_edge), edge))
    lo = max(1, int(minimum_edge))
    ch = min(h, max(min(h, lo), edge))
    cw = min(w, max(min(w, lo), edge))
    return int(ch), int(cw)


def _halo_cells_for_km(grid, y0: int, y1: int, radius_km: float) -> tuple[int, int]:
    """Return a conservative rectangular cell halo for a physical radius."""
    radius = max(0.0, float(radius_km))
    if radius <= 0.0:
        return 0, 0
    h, w = int(grid.height), int(grid.width)
    hy = int(np.ceil(radius / max(float(grid.dy_km), 1e-12)))

    # Include latitude rows reachable by the north/south halo, with polar reflection,
    # then use the smallest east-west cell width among those rows.  This may produce
    # a wide halo near a pole, but it is conservative and deterministic.
    virtual_y = np.arange(y0 - hy, y1 + hy, dtype=np.int64)
    mapped_y, _ = _map_spherical_lattice_indices(
        virtual_y, np.zeros_like(virtual_y), (h, w)
    )
    row_dx = np.asarray(grid.dx_km, dtype=np.float64)[mapped_y, 0]
    min_dx = float(np.min(row_dx)) if row_dx.size else float(grid.dy_km)
    hx = int(np.ceil(radius / max(min_dx, 1e-12)))
    # More than half a periodic circumference repeats cells and cannot add coverage.
    hx = min(hx, max(0, w // 2))
    return hy, hx


class SphericalTiler:
    """Deterministic y-major/x-major tile planner with spherical halos."""

    def __init__(
        self,
        grid,
        *,
        chunk_shape: tuple[int, int] | None = None,
        target_mb: float = 32.0,
        dtype=np.float32,
        arrays_in_flight: int = 6,
        halo_cells: int | tuple[int, int] = 0,
        halo_km: float | None = None,
    ) -> None:
        self.grid = grid
        self.shape = (int(grid.height), int(grid.width))
        self.chunk_shape = chunk_shape or auto_tile_shape(
            self.shape,
            dtype,
            target_mb=target_mb,
            arrays_in_flight=arrays_in_flight,
        )
        ch, cw = map(int, self.chunk_shape)
        if ch <= 0 or cw <= 0:
            raise ValueError("chunk_shape dimensions must be positive")
        self.chunk_shape = min(ch, self.shape[0]), min(cw, self.shape[1])
        if isinstance(halo_cells, tuple):
            hy, hx = halo_cells
        else:
            hy = hx = halo_cells
        self.halo_cells = max(0, int(hy)), max(0, int(hx))
        self.halo_km = None if halo_km is None else max(0.0, float(halo_km))

    def __iter__(self) -> Iterator[SphericalTile]:
        h, w = self.shape
        ch, cw = self.chunk_shape
        index = 0
        tile_row = 0
        for y0 in range(0, h, ch):
            y1 = min(h, y0 + ch)
            tile_col = 0
            for x0 in range(0, w, cw):
                x1 = min(w, x0 + cw)
                hy, hx = self.halo_cells
                if self.halo_km is not None:
                    phy, phx = _halo_cells_for_km(self.grid, y0, y1, self.halo_km)
                    hy = max(hy, phy)
                    hx = max(hx, phx)
                yield SphericalTile(
                    index=index,
                    tile_row=tile_row,
                    tile_col=tile_col,
                    shape=self.shape,
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                    halo_y=hy,
                    halo_x=hx,
                )
                index += 1
                tile_col += 1
            tile_row += 1

    def count(self) -> int:
        h, w = self.shape
        ch, cw = self.chunk_shape
        return int(np.ceil(h / ch) * np.ceil(w / cw))


def apply_tiled(
    values: np.ndarray,
    tiler: SphericalTiler,
    kernel: Callable[[np.ndarray], np.ndarray],
    *,
    out_dtype=None,
) -> np.ndarray:
    """Apply a halo-local kernel tile by tile with bounded intermediate memory.

    ``kernel`` must preserve the expanded tile's trailing spatial shape.  This helper
    is intentionally serial: execution policy belongs to the runtime scheduler, while
    tile topology and deterministic assembly stay independent of worker count.
    """
    a = np.asarray(values)
    if a.shape[-2:] != tiler.shape:
        raise ValueError(f"last two dimensions must be {tiler.shape}, got {a.shape}")
    dtype = np.dtype(out_dtype) if out_dtype is not None else a.dtype
    out = np.empty(a.shape, dtype=dtype)
    for tile in tiler:
        expanded = tile.extract(a)
        transformed = np.asarray(kernel(expanded))
        if transformed.shape != expanded.shape:
            raise ValueError(
                f"tile kernel changed shape from {expanded.shape} to {transformed.shape}"
            )
        out[..., tile.core[0], tile.core[1]] = tile.crop_core(transformed)
    return out


__all__ = [
    "SphericalTile",
    "SphericalTiler",
    "auto_tile_shape",
    "apply_tiled",
]
