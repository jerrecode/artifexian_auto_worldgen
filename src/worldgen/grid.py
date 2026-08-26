from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .cache import ByteBoundLRUCache

EARTH_RADIUS_KM = 6371.0088


@dataclass
class SphereGrid:
    """Regular 2:1 raster whose metric/topology is a sphere.

    The storage layout remains equirectangular for fast NumPy access and familiar
    exports, but geometry-sensitive helpers use spherical metrics.  Longitude is
    periodic and motion across either pole is reflected with a 180-degree
    longitude rotation.
    """

    width: int
    height: int
    radius_km: float = EARTH_RADIUS_KM
    distance_cache_bytes: int = 256 * 1024**2

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.width != 2 * self.height:
            raise ValueError("SphereGrid requires a positive 2:1 equirectangular shape")
        self.lon_1d = np.linspace(-180.0, 180.0, self.width, endpoint=False, dtype=np.float64)
        dlat = 180.0 / self.height
        self.lat_1d = np.linspace(90.0 - dlat / 2, -90.0 + dlat / 2, self.height, dtype=np.float64)
        self.lon, self.lat = np.meshgrid(self.lon_1d, self.lat_1d)
        lat_r = np.deg2rad(self.lat)
        lon_r = np.deg2rad(self.lon)
        c = np.cos(lat_r)
        self.xyz = np.stack((c * np.cos(lon_r), c * np.sin(lon_r), np.sin(lat_r)), axis=-1)
        self.cell_area_weights = np.cos(lat_r)
        self.cell_area_weights /= self.cell_area_weights.sum()
        self.dlat_rad = np.pi / self.height
        self.dlon_rad = 2 * np.pi / self.width
        self.dy_km = self.radius_km * self.dlat_rad
        self.dx_km = self.radius_km * self.dlon_rad * np.maximum(np.cos(lat_r), 1e-6)
        self.cell_area_km2 = self.cell_area_weights * (4.0 * np.pi * self.radius_km**2)
        self._distance_cache: ByteBoundLRUCache[str, np.ndarray] = ByteBoundLRUCache(
            max_bytes=max(0, int(self.distance_cache_bytes))
        )

    def weighted_fraction(self, mask: np.ndarray) -> float:
        return float(np.sum(self.cell_area_weights * np.asarray(mask, dtype=float)))

    def weighted_mean(self, values: np.ndarray, mask: np.ndarray | None = None) -> float:
        a = np.asarray(values, dtype=np.float64)
        if a.shape != (self.height, self.width):
            raise ValueError("weighted_mean expects one global raster")
        if mask is None:
            return float(np.sum(a * self.cell_area_weights))
        m = np.asarray(mask, dtype=bool)
        if not np.any(m):
            return float("nan")
        return float(np.average(a[m], weights=self.cell_area_weights[m]))

    def weighted_quantile(self, data: np.ndarray, q: float) -> float:
        a = np.asarray(data).ravel()
        w = self.cell_area_weights.ravel()
        idx = np.argsort(a)
        aa = a[idx]
        ww = w[idx]
        c = np.cumsum(ww)
        return float(np.interp(q, c, aa))

    def great_circle_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        p1, p2 = np.deg2rad([lat1, lat2])
        dl = np.deg2rad(lon2 - lon1)
        dp = p2 - p1
        a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        return float(2 * self.radius_km * np.arcsin(np.sqrt(np.clip(a, 0, 1))))

    @staticmethod
    def wrap_lon_index(x: np.ndarray | int, width: int) -> np.ndarray:
        return np.asarray(x) % width

    def clear_geometry_cache(self) -> None:
        self._distance_cache.clear()

    def geometry_cache_stats(self):
        return self._distance_cache.stats()


def spherical_voronoi_ids(grid: SphereGrid, seeds_xyz: np.ndarray, chunk: int = 65536) -> np.ndarray:
    """Assign pixels to the nearest seed by maximum unit-vector dot product."""
    pts = grid.xyz.reshape(-1, 3)
    out = np.empty(len(pts), dtype=np.int16 if len(seeds_xyz) < 32767 else np.int32)
    for i in range(0, len(pts), chunk):
        dot = pts[i:i + chunk] @ seeds_xyz.T
        out[i:i + chunk] = np.argmax(dot, axis=1)
    return out.reshape(grid.height, grid.width)


def _neighbor_geometry(shape: tuple[int, int], dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return neighbor indices with periodic longitude and pole reflection.

    Crossing a pole reflects latitude and rotates longitude by 180 degrees.  The
    implementation supports offsets larger than one by repeatedly reflecting
    until every latitude index lies in range.
    """
    h, w = map(int, shape)
    yy, xx = np.indices((h, w), dtype=np.int64)
    ny = yy + int(dy)
    nx = xx + int(dx)
    while np.any((ny < 0) | (ny >= h)):
        north = ny < 0
        if np.any(north):
            ny = np.where(north, -ny - 1, ny)
            nx = np.where(north, nx + w // 2, nx)
        south = ny >= h
        if np.any(south):
            ny = np.where(south, 2 * h - ny - 1, ny)
            nx = np.where(south, nx + w // 2, nx)
    return ny.astype(np.int32), (nx % w).astype(np.int32)


def spherical_shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Sample a 2-D raster at a spherical neighbor offset."""
    arr = np.asarray(a)
    if arr.ndim != 2:
        raise ValueError("spherical_shift currently expects a 2-D raster")
    ny, nx = _neighbor_geometry(arr.shape, dy, dx)
    return arr[ny, nx]


def connected_components_spherical(mask: np.ndarray, *, diagonal: bool = True) -> tuple[np.ndarray, int]:
    """Connected components on the equirectangular storage raster with spherical seams.

    SciPy performs the interior labeling. A compact union-find then merges labels
    across the longitude seam and across both reflected poles.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError("mask must be 2-D")
    structure = np.ones((3, 3), dtype=np.uint8) if diagonal else ndimage.generate_binary_structure(2, 1)
    labels, n = ndimage.label(m, structure=structure)
    if n <= 1:
        return labels.astype(np.int32), int(n)
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

    h, w = m.shape
    y_offsets = (-1, 0, 1) if diagonal else (0,)
    for y in range(h):
        if not m[y, 0]:
            continue
        for oy in y_offsets:
            yy = y + oy
            if 0 <= yy < h and m[yy, w - 1]:
                union(int(labels[y, 0]), int(labels[yy, w - 1]))

    half = w // 2
    for x in range(w):
        xp = (x + half) % w
        if m[0, x] and m[0, xp]:
            union(int(labels[0, x]), int(labels[0, xp]))
        if m[h - 1, x] and m[h - 1, xp]:
            union(int(labels[h - 1, x]), int(labels[h - 1, xp]))

    root = np.asarray([find(i) for i in range(n + 1)], dtype=np.int32)
    merged = root[labels]
    unique = [int(v) for v in np.unique(merged) if v != 0]
    remap = {v: i + 1 for i, v in enumerate(unique)}
    out = np.zeros_like(merged, dtype=np.int32)
    for old, new in remap.items():
        out[merged == old] = new
    return out, len(unique)


def smooth_periodic(a: np.ndarray, sigma: float | tuple[float, float]) -> np.ndarray:
    """Legacy equirectangular smoothing: latitude clamped, longitude periodic."""
    return ndimage.gaussian_filter(a, sigma=sigma, mode=("nearest", "wrap"))


def smooth_spherical(a: np.ndarray, sigma: float | tuple[float, float]) -> np.ndarray:
    """Gaussian-like smoothing whose latitude padding follows spherical pole topology."""
    arr = np.asarray(a)
    if arr.ndim != 2:
        raise ValueError("smooth_spherical currently expects a 2-D raster")
    sy, sx = (float(sigma), float(sigma)) if np.isscalar(sigma) else map(float, sigma)
    pad = min(arr.shape[0] - 1, max(1, int(np.ceil(4.0 * max(sy, 0.0)))))
    if pad <= 0:
        return arr.copy()
    half = arr.shape[1] // 2
    top = np.roll(arr[:pad][::-1], half, axis=1)
    bottom = np.roll(arr[-pad:][::-1], half, axis=1)
    ext = np.concatenate((top, arr, bottom), axis=0)
    sm = ndimage.gaussian_filter(ext, sigma=(sy, sx), mode=("nearest", "wrap"))
    return sm[pad:pad + arr.shape[0]]


def _mask_cache_key(mask: np.ndarray, grid: SphereGrid) -> str:
    packed = np.packbits(np.asarray(mask, dtype=np.uint8), axis=None)
    digest = hashlib.blake2b(packed.tobytes(), digest_size=16).hexdigest()
    return f"gc:{grid.height}x{grid.width}:{grid.radius_km:.9f}:{digest}"


def distance_to(mask: np.ndarray, grid: SphereGrid) -> np.ndarray:
    """Exact nearest-cell great-circle distance to True pixels, in kilometres.

    Nearest neighbors are found in 3-D chord space with ``cKDTree``. Chord
    distance is monotonic with central angle on the unit sphere, so the returned
    nearest point is also the nearest point geodesically. Results are cached by
    mask content because coast/orogen/river distance fields are reused heavily
    across Earth-system stages.
    """
    m = np.asarray(mask, dtype=bool)
    if m.shape != (grid.height, grid.width):
        raise ValueError("mask shape does not match grid")
    if not np.any(m):
        return np.full(m.shape, np.inf, dtype=np.float32)
    if np.all(m):
        return np.zeros(m.shape, dtype=np.float32)
    key = _mask_cache_key(m, grid)
    cached = grid._distance_cache.get(key)
    if cached is not None:
        return cached
    sources = grid.xyz[m]
    tree = cKDTree(sources, compact_nodes=True, balanced_tree=True)
    chord, _ = tree.query(grid.xyz.reshape(-1, 3), k=1, workers=1)
    angle = 2.0 * np.arcsin(np.clip(chord * 0.5, 0.0, 1.0))
    result = (angle.reshape(m.shape) * grid.radius_km).astype(np.float32)
    result[m] = 0.0
    grid._distance_cache.put(key, result)
    return result


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("labels must be 2-D")
    b = np.zeros_like(labels, dtype=bool)
    b |= labels != spherical_shift(labels, 0, -1)
    b |= labels != spherical_shift(labels, 0, 1)
    b |= labels != spherical_shift(labels, -1, 0)
    b |= labels != spherical_shift(labels, 1, 0)
    return b


def local_slope(elevation_km: np.ndarray, grid: SphereGrid) -> np.ndarray:
    z = np.asarray(elevation_km, dtype=np.float64)
    north = spherical_shift(z, -1, 0)
    south = spherical_shift(z, 1, 0)
    west = spherical_shift(z, 0, -1)
    east = spherical_shift(z, 0, 1)
    sy = (south - north) / (2.0 * max(grid.dy_km, 1e-6))
    sx = (east - west) / np.maximum(2.0 * grid.dx_km, 1e-6)
    return np.hypot(sx, sy)


def normalize01(a: np.ndarray, robust: bool = True) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if robust:
        lo, hi = np.nanpercentile(arr, [1.0, 99.0])
    else:
        lo, hi = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
