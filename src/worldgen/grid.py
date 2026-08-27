from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import numpy as np

from .cache import ByteBoundLRUCache, CacheStats

EARTH_RADIUS_KM = 6371.0088


@dataclass
class SphereGrid:
    width: int
    height: int
    radius_km: float = EARTH_RADIUS_KM
    distance_cache_max_bytes: int | None = None

    def __post_init__(self) -> None:
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
        self.dx_km = self.radius_km * self.dlon_rad * np.maximum(np.cos(lat_r), 1e-3)

        if self.distance_cache_max_bytes is None:
            try:
                mb = float(os.environ.get("WORLDGEN_DISTANCE_CACHE_MB", "192"))
            except ValueError:
                mb = 192.0
            self.distance_cache_max_bytes = max(0, int(mb * 1024**2))
        self._distance_cache: ByteBoundLRUCache[str, np.ndarray] = ByteBoundLRUCache(
            max_bytes=int(self.distance_cache_max_bytes)
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Canonical raster shape in NumPy order: ``(height, width)``."""
        return self.height, self.width

    @property
    def ops(self):
        from .topology import SphericalRasterOps
        return SphericalRasterOps(self)

    def clear_spatial_cache(self) -> None:
        self._distance_cache.clear()

    def spatial_cache_stats(self) -> CacheStats:
        return self._distance_cache.stats()

    def weighted_fraction(self, mask: np.ndarray) -> float:
        return float(np.sum(self.cell_area_weights * np.asarray(mask, dtype=float)))

    def weighted_quantile(self, data: np.ndarray, q: float) -> float:
        a = np.asarray(data, dtype=np.float64).ravel()
        w = np.asarray(self.cell_area_weights, dtype=np.float64).ravel()
        finite = np.isfinite(a) & np.isfinite(w) & (w > 0)
        if not np.any(finite):
            return float("nan")
        aa = a[finite]
        ww = w[finite]
        idx = np.argsort(aa)
        aa = aa[idx]
        ww = ww[idx]
        c = np.cumsum(ww)
        c /= max(float(c[-1]), 1e-30)
        return float(np.interp(float(np.clip(q, 0.0, 1.0)), c, aa))

    def great_circle_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        p1, p2 = np.deg2rad([lat1, lat2])
        dl = np.deg2rad(lon2 - lon1)
        dp = p2 - p1
        a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        return float(2 * self.radius_km * np.arcsin(np.sqrt(np.clip(a, 0, 1))))

    @staticmethod
    def wrap_lon_index(x: np.ndarray | int, width: int) -> np.ndarray:
        return np.asarray(x) % width


def spherical_voronoi_ids(grid: SphereGrid, seeds_xyz: np.ndarray, chunk: int = 65536) -> np.ndarray:
    pts = grid.xyz.reshape(-1, 3)
    out = np.empty(len(pts), dtype=np.int16 if len(seeds_xyz) < 32767 else np.int32)
    for i in range(0, len(pts), chunk):
        dot = pts[i:i + chunk] @ seeds_xyz.T
        out[i:i + chunk] = np.argmax(dot, axis=1)
    return out.reshape(grid.height, grid.width)


def smooth_periodic(a: np.ndarray, sigma: float | tuple[float, float]) -> np.ndarray:
    """Gaussian smoothing on a spherical equirectangular raster.

    Longitude wraps and latitude is reflected through each pole with the required
    antipodal longitude rotation. The public name is retained for compatibility.
    """
    from .topology import spherical_gaussian_filter
    return spherical_gaussian_filter(a, sigma)


def _mask_digest(mask: np.ndarray) -> str:
    m = np.asarray(mask, dtype=bool)
    packed = np.packbits(m.ravel(order="C"))
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray(m.shape, dtype=np.int64).tobytes())
    h.update(packed)
    return h.hexdigest()


def distance_to(mask: np.ndarray, grid: SphereGrid) -> np.ndarray:
    """Cached great-circle distance in km to the nearest True region.

    Distance fields are among the most frequently repeated expensive spatial
    primitives in climate, terrain, resources and society. The cache is LRU- and
    byte-bounded; ``WORLDGEN_DISTANCE_CACHE_MB`` controls its default resident cap.
    """
    m = np.asarray(mask, dtype=bool)
    key = _mask_digest(m)
    cached = grid._distance_cache.get(key)
    if cached is not None:
        return cached
    from .topology import geodesic_distance_to
    result = geodesic_distance_to(m, grid).astype(np.float32)
    grid._distance_cache.put(key, result, size_bytes=int(result.nbytes))
    return result


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    b = np.zeros_like(labels, dtype=bool)
    b |= labels != np.roll(labels, 1, axis=1)
    b |= labels != np.roll(labels, -1, axis=1)
    b[:-1] |= labels[:-1] != labels[1:]
    b[1:] |= labels[1:] != labels[:-1]
    return b


def local_slope(elevation_km: np.ndarray, grid: SphereGrid) -> np.ndarray:
    gy, gx = grid.ops.metric_gradient(elevation_km)
    return np.hypot(gx, gy)


def normalize01(a: np.ndarray, robust: bool = True) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr)
    if robust:
        lo, hi = np.nanpercentile(arr[finite], [1.0, 99.0])
    else:
        lo, hi = np.nanmin(arr[finite]), np.nanmax(arr[finite])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr)
    out = np.zeros_like(arr)
    out[finite] = np.clip((arr[finite] - lo) / (hi - lo), 0.0, 1.0)
    return out
