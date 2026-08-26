from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import ndimage

EARTH_RADIUS_KM = 6371.0088


@dataclass
class SphereGrid:
    width: int
    height: int
    radius_km: float = EARTH_RADIUS_KM

    def __post_init__(self) -> None:
        self.lon_1d = np.linspace(-180.0, 180.0, self.width, endpoint=False, dtype=np.float64)
        # Pixel centers, not poles, prevent zero-area rows.
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

    def weighted_fraction(self, mask: np.ndarray) -> float:
        return float(np.sum(self.cell_area_weights * np.asarray(mask, dtype=float)))

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


def spherical_voronoi_ids(grid: SphereGrid, seeds_xyz: np.ndarray, chunk: int = 65536) -> np.ndarray:
    """Assign pixels to the nearest seed by maximum unit-vector dot product."""
    pts = grid.xyz.reshape(-1, 3)
    out = np.empty(len(pts), dtype=np.int16 if len(seeds_xyz) < 32767 else np.int32)
    for i in range(0, len(pts), chunk):
        dot = pts[i:i + chunk] @ seeds_xyz.T
        out[i:i + chunk] = np.argmax(dot, axis=1)
    return out.reshape(grid.height, grid.width)


def smooth_periodic(a: np.ndarray, sigma: float | tuple[float, float]) -> np.ndarray:
    return ndimage.gaussian_filter(a, sigma=sigma, mode=("nearest", "wrap"))


def distance_to(mask: np.ndarray, grid: SphereGrid) -> np.ndarray:
    """Fast equirectangular approximation to distance to True pixels in km."""
    if not np.any(mask):
        return np.full(mask.shape, np.inf, dtype=np.float64)
    # EDT does not wrap longitude. Triplication makes seam distances correct.
    trip = np.concatenate([mask, mask, mask], axis=1)
    dist_px = ndimage.distance_transform_edt(~trip)
    dist_px = dist_px[:, grid.width:2 * grid.width]
    # latitude-varying east-west pixel scale; EDT isotropic in px, use conservative mean.
    return dist_px * grid.dy_km


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    b = np.zeros_like(labels, dtype=bool)
    b |= labels != np.roll(labels, 1, axis=1)
    b |= labels != np.roll(labels, -1, axis=1)
    b[:-1] |= labels[:-1] != labels[1:]
    b[1:] |= labels[1:] != labels[:-1]
    return b


def local_slope(elevation_km: np.ndarray, grid: SphereGrid) -> np.ndarray:
    gy, gx = np.gradient(elevation_km)
    dx = np.maximum(grid.dx_km, 1e-3)
    sy = gy / grid.dy_km
    sx = gx / dx
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
