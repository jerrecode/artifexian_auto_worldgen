from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from .grid import SphereGrid


def _map_spherical_lattice_indices(
    y: np.ndarray, x: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Map arbitrary integer raster indices onto the spherical equirectangular grid.

    Longitude wraps periodically. Each latitude crossing reflects about the pole and
    advances longitude by half a world, which is the discrete equivalent of passing
    continuously over a pole on a sphere.
    """
    h, w = map(int, shape)
    yy = np.asarray(y, dtype=np.int64).copy()
    xx = np.asarray(x, dtype=np.int64).copy()
    if h <= 0 or w <= 0:
        raise ValueError("shape dimensions must be positive")
    while np.any((yy < 0) | (yy >= h)):
        north = yy < 0
        south = yy >= h
        if np.any(north):
            yy = np.where(north, -yy - 1, yy)
            xx = np.where(north, xx + w // 2, xx)
        if np.any(south):
            yy = np.where(south, 2 * h - yy - 1, yy)
            xx = np.where(south, xx + w // 2, xx)
    return yy.astype(np.int32, copy=False), (xx % w).astype(np.int32, copy=False)


def prepare_spherical_bilinear_sampler(
    src_y: np.ndarray, src_x: np.ndarray, shape: tuple[int, int]
):
    """Precompute bilinear indices/weights with true seam and pole topology.

    Coordinates may lie outside the latitude raster. Rather than clamping them at
    the first/last row, each interpolation corner is reflected across the pole and
    shifted by 180 degrees in longitude. This keeps semi-Lagrangian transport
    continuous across both poles as well as across the longitude seam.
    """
    h, w = map(int, shape)
    sy, sx = np.broadcast_arrays(
        np.asarray(src_y, dtype=np.float64), np.asarray(src_x, dtype=np.float64)
    )
    y0 = np.floor(sy).astype(np.int64)
    x0 = np.floor(sx).astype(np.int64)
    y1 = y0 + 1
    x1 = x0 + 1
    fy = (sy - y0).astype(np.float32)
    fx = (sx - x0).astype(np.float32)

    y00, x00 = _map_spherical_lattice_indices(y0, x0, (h, w))
    y01, x01 = _map_spherical_lattice_indices(y0, x1, (h, w))
    y10, x10 = _map_spherical_lattice_indices(y1, x0, (h, w))
    y11, x11 = _map_spherical_lattice_indices(y1, x1, (h, w))

    return (
        (y00 * w + x00).ravel(),
        (y01 * w + x01).ravel(),
        (y10 * w + x10).ravel(),
        (y11 * w + x11).ravel(),
        ((1.0 - fy) * (1.0 - fx)).ravel(),
        ((1.0 - fy) * fx).ravel(),
        (fy * (1.0 - fx)).ravel(),
        (fy * fx).ravel(),
        sy.shape,
    )


def apply_bilinear_sampler(values: np.ndarray, sampler) -> np.ndarray:
    """Apply a sampler created by :func:`prepare_spherical_bilinear_sampler`."""
    i00, i01, i10, i11, w00, w01, w10, w11, shape = sampler
    f = np.asarray(values).ravel()
    return (
        f[i00] * w00
        + f[i01] * w01
        + f[i10] * w10
        + f[i11] * w11
    ).reshape(shape)


def spherical_gaussian_filter(
    values: np.ndarray,
    sigma: float | tuple[float, float],
    *,
    truncate: float = 4.0,
) -> np.ndarray:
    """Gaussian-filter the last two raster axes using spherical pole topology.

    Longitude is wrapped by SciPy. Latitude receives an explicit reflected halo;
    every reflected halo row is rotated by 180 degrees in longitude. This avoids
    the artificial zero-normal-flow boundary implicit in ``mode='nearest'`` at the
    poles while preserving the same Gaussian kernel and tuning elsewhere.
    """
    a = np.asarray(values)
    if a.ndim < 2:
        raise ValueError("spherical Gaussian filtering requires at least two dimensions")
    if np.isscalar(sigma):
        sy = sx = float(sigma)
    else:
        seq = tuple(float(s) for s in sigma)
        if len(seq) != 2:
            raise ValueError("sigma must be a scalar or a (latitude, longitude) pair")
        sy, sx = seq
    sy = max(0.0, sy)
    sx = max(0.0, sx)
    if sy <= 0.0 and sx <= 0.0:
        return a.copy()

    h, w = a.shape[-2:]
    if h <= 0 or w <= 0:
        return a.copy()
    halo = int(np.ceil(max(float(truncate), 0.0) * sy)) if sy > 0.0 else 0

    if halo > 0:
        virtual_y = np.arange(-halo, h + halo, dtype=np.int64)
        mapped_y = virtual_y.copy()
        half_turns = np.zeros_like(mapped_y)
        while np.any((mapped_y < 0) | (mapped_y >= h)):
            north = mapped_y < 0
            south = mapped_y >= h
            if np.any(north):
                mapped_y = np.where(north, -mapped_y - 1, mapped_y)
                half_turns = np.where(north, half_turns + 1, half_turns)
            if np.any(south):
                mapped_y = np.where(south, 2 * h - mapped_y - 1, mapped_y)
                half_turns = np.where(south, half_turns + 1, half_turns)
        padded = np.take(a, mapped_y.astype(np.intp), axis=-2).copy()
        if w > 1:
            for row, turns in enumerate(half_turns):
                if int(turns) & 1:
                    padded[..., row, :] = np.roll(padded[..., row, :], -(w // 2), axis=-1)
    else:
        padded = a

    sigmas = (0.0,) * (padded.ndim - 2) + (sy, sx)
    modes = ("nearest",) * (padded.ndim - 1) + ("wrap",)
    filtered = ndimage.gaussian_filter(
        padded, sigma=sigmas, mode=modes, truncate=float(truncate)
    )
    if halo > 0:
        filtered = filtered[..., halo : halo + h, :]
    return filtered


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
        return _map_spherical_lattice_indices(yy + int(dy), xx + int(dx), (h, w))

    def shift(self, array: np.ndarray, dy: int, dx: int) -> np.ndarray:
        a = np.asarray(array)
        if a.shape[-2:] != self.shape:
            raise ValueError(f"last two dimensions must be {self.shape}, got {a.shape}")
        ny, nx = self.neighbor_indices(dy, dx)
        if a.ndim == 2:
            return a[ny, nx]
        return a[..., ny, nx]

    def bilinear_sampler(self, src_y: np.ndarray, src_x: np.ndarray):
        return prepare_spherical_bilinear_sampler(src_y, src_x, self.shape)

    def bilinear_sample(self, values: np.ndarray, sampler) -> np.ndarray:
        return apply_bilinear_sampler(values, sampler)

    def gaussian_filter(
        self, values: np.ndarray, sigma: float | tuple[float, float], *, truncate: float = 4.0
    ) -> np.ndarray:
        a = np.asarray(values)
        if a.shape[-2:] != self.shape:
            raise ValueError(f"last two dimensions must be {self.shape}, got {a.shape}")
        return spherical_gaussian_filter(a, sigma, truncate=truncate)

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

    def grey_dilation(self, values: np.ndarray, iterations: int = 1) -> np.ndarray:
        """Maximum filter over spherical 8-neighborhoods.

        Each iteration expands the Chebyshev neighborhood radius by one cell while
        preserving longitude wrap and the 180-degree rotation required when a
        neighborhood crosses either pole. This is the spherical counterpart of a
        square ``scipy.ndimage.maximum_filter`` / grey dilation.
        """
        out = np.asarray(values).copy()
        if out.shape[-2:] != self.shape:
            raise ValueError(f"last two dimensions must be {self.shape}, got {out.shape}")
        for _ in range(max(0, int(iterations))):
            src = out
            expanded = src.copy()
            for dy, dx in self.neighbors8():
                expanded = np.maximum(expanded, self.shift(src, dy, dx))
            out = expanded
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
