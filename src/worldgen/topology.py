from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from .grid import SphereGrid


@dataclass(slots=True)
class SphericalRasterOps:
    """Topology- and metric-aware operations for an equirectangular spherical raster.

    Longitude is periodic. Crossing either pole reflects the latitude index and
    rotates longitude by 180 degrees. Keeping those rules in one place prevents
    the climate, hydrology, geology and society layers from silently disagreeing
    about what cells are neighbors.
    """

    grid: "SphereGrid"

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.height, self.grid.width

    def neighbor_indices(self, dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
        h, w = self.shape
        yy, xx = np.indices((h, w), dtype=np.int32)
        ny = yy + int(dy)
        nx = xx + int(dx)

        # General reflection loop also supports halo offsets larger than one row.
        # A pole crossing rotates the longitude by half a world.
        while np.any((ny < 0) | (ny >= h)):
            north = ny < 0
            south = ny >= h
            if np.any(north):
                ny = np.where(north, -ny - 1, ny)
                nx = np.where(north, nx + w // 2, nx)
            if np.any(south):
                ny = np.where(south, 2 * h - ny - 1, ny)
                nx = np.where(south, nx + w // 2, nx)
        return ny.astype(np.int32, copy=False), (nx % w).astype(np.int32, copy=False)

    def shift(self, array: np.ndarray, dy: int, dx: int) -> np.ndarray:
        a = np.asarray(array)
        if a.shape[-2:] != self.shape:
            raise ValueError(f"last two dimensions must be {self.shape}, got {a.shape}")
        ny, nx = self.neighbor_indices(dy, dx)
        if a.ndim == 2:
            return a[ny, nx]
        return a[..., ny, nx]

    def neighbors8(self) -> Iterator[tuple[int, int]]:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    yield dy, dx

    def binary_dilation(self, mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        out = np.asarray(mask, dtype=bool).copy()
        for _ in range(max(0, int(iterations))):
            src = out
            expanded = src.copy()
            for dy, dx in self.neighbors8():
                expanded |= self.shift(src, dy, dx)
            out = expanded
        return out

    def binary_erosion(self, mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        out = np.asarray(mask, dtype=bool).copy()
        for _ in range(max(0, int(iterations))):
            src = out
            shrunk = src.copy()
            for dy, dx in self.neighbors8():
                shrunk &= self.shift(src, dy, dx)
            out = shrunk
        return out

    def boundary(self, mask: np.ndarray) -> np.ndarray:
        m = np.asarray(mask, dtype=bool)
        if not np.any(m):
            return np.zeros_like(m)
        interior = self.binary_erosion(m, 1)
        return m & ~interior

    def metric_gradient(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (d/dy, d/dx) in units per km.

        Raster y increases southward; x increases eastward. The east-west metric
        varies with latitude and is already represented by ``grid.dx_km``.
        """
        a = np.asarray(values, dtype=np.float64)
        north = self.shift(a, -1, 0)
        south = self.shift(a, 1, 0)
        west = self.shift(a, 0, -1)
        east = self.shift(a, 0, 1)
        gy = (south - north) / max(2.0 * float(self.grid.dy_km), 1e-12)
        gx = (east - west) / np.maximum(2.0 * self.grid.dx_km, 1e-12)
        return gy, gx

    def divergence(self, u_east: np.ndarray, v_south: np.ndarray) -> np.ndarray:
        _, du_dx = self.metric_gradient(u_east)
        dv_dy, _ = self.metric_gradient(v_south)
        return du_dx + dv_dy

    def curl(self, u_east: np.ndarray, v_south: np.ndarray) -> np.ndarray:
        du_dy, _ = self.metric_gradient(u_east)
        _, dv_dx = self.metric_gradient(v_south)
        return dv_dx - du_dy

    def connected_components(self, mask: np.ndarray) -> tuple[np.ndarray, int]:
        """8-connected components with seam and pole adjacency resolved."""
        m = np.asarray(mask, dtype=bool)
        labels, n = ndimage.label(m, structure=np.ones((3, 3), dtype=np.int8))
        if n <= 1:
            return labels.astype(np.int32, copy=False), int(n)

        parent = np.arange(n + 1, dtype=np.int32)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = int(parent[x])
            return x

        def union(a: int, b: int) -> None:
            if a == 0 or b == 0:
                return
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # ndimage already handles interior adjacency. These pair scans additionally
        # reconcile the longitude seam and the reflected polar topology. Running all
        # eight directions is cheap relative to the physical solvers and robust.
        for dy, dx in self.neighbors8():
            ny, nx = self.neighbor_indices(dy, dx)
            other = labels[ny, nx]
            good = m & (other > 0) & (labels != other)
            if not np.any(good):
                continue
            pairs = np.stack((labels[good], other[good]), axis=1)
            pairs.sort(axis=1)
            for a, b in np.unique(pairs, axis=0):
                union(int(a), int(b))

        roots = np.array([find(i) for i in range(n + 1)], dtype=np.int32)
        merged = roots[labels]
        vals = np.unique(merged)
        vals = vals[vals != 0]
        lut = np.zeros(int(merged.max()) + 1 if merged.size else 1, dtype=np.int32)
        for new, old in enumerate(vals, start=1):
            lut[int(old)] = new
        out = lut[merged]
        return out.astype(np.int32, copy=False), int(len(vals))


def geodesic_distance_to(mask: np.ndarray, grid: "SphereGrid", *, chunk: int = 262_144) -> np.ndarray:
    """Great-circle distance in km to the nearest True region boundary.

    The old implementation used an isotropic Euclidean distance transform in
    equirectangular pixel coordinates and multiplied by the north-south cell size.
    That increasingly overestimated east-west distance toward the poles. Here the
    boundary is indexed in 3-D unit-vector space and queried with a cKDTree; chord
    distance is then converted exactly to great-circle arc distance.
    """
    m = np.asarray(mask, dtype=bool)
    if m.shape != (grid.height, grid.width):
        raise ValueError("mask shape must match grid")
    if not np.any(m):
        return np.full(m.shape, np.inf, dtype=np.float64)
    if np.all(m):
        return np.zeros(m.shape, dtype=np.float64)

    ops = SphericalRasterOps(grid)
    boundary = ops.boundary(m)
    points = np.asarray(grid.xyz[boundary], dtype=np.float64)
    if points.size == 0:
        points = np.asarray(grid.xyz[m], dtype=np.float64)
    tree = cKDTree(points)
    flat_xyz = np.asarray(grid.xyz, dtype=np.float64).reshape(-1, 3)
    out = np.empty(len(flat_xyz), dtype=np.float64)
    for start in range(0, len(flat_xyz), max(1, int(chunk))):
        stop = min(len(flat_xyz), start + max(1, int(chunk)))
        chord, _ = tree.query(flat_xyz[start:stop], k=1, workers=1)
        chord = np.clip(chord, 0.0, 2.0)
        out[start:stop] = 2.0 * float(grid.radius_km) * np.arcsin(0.5 * chord)
    out = out.reshape(m.shape)
    out[m] = 0.0
    return out
