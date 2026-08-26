from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .config import WeatherConfig
from .grid import SphereGrid, normalize01, smooth_periodic, local_slope
from .terrain import TerrainResult
from .ocean import OceanResult
from .climate import ClimateResult
from .hydrology import HydrologyResult


@dataclass(slots=True)
class WeatherResult:
    fog: np.ndarray
    thunderstorm_level: np.ndarray      # 0 none, 1 basic, 2 moderate, 3 severe, 4 tornado-prone
    lightning_flashes_km2_year: np.ndarray
    tornado_potential: np.ndarray
    hurricane_genesis: np.ndarray
    hurricane_tracks: list[dict]
    blizzard: np.ndarray
    sandstorm: np.ndarray
    duststorm: np.ndarray
    aurora: np.ndarray
    sea_ice_max: np.ndarray
    sea_ice_min: np.ndarray
    coral_reef: np.ndarray
    metadata: dict


def _large_land_mask(grid: SphereGrid, land: np.ndarray) -> np.ndarray:
    """Return components large enough to support continental weather regimes.

    Component connectivity and size are spherical: pieces meeting across the
    longitude seam or across a reflected pole are one landmass, and high-latitude
    cells contribute their smaller physical area rather than one full pixel each.
    """
    labels, n = grid.ops.connected_components(land)
    if n == 0:
        return np.zeros_like(land, dtype=bool)
    areas = np.bincount(
        labels.ravel(),
        weights=np.asarray(grid.cell_area_weights, float).ravel(),
        minlength=n + 1,
    )
    # Preserve the former max(30 pixels, 0.3% of raster) intent, but measure it
    # in spherical surface area so polar pixels are not over-weighted.
    cutoff = max(30.0 / max(int(land.size), 1), 0.003)
    good = areas >= cutoff
    good[0] = False
    return good[labels]


def _sample_track(
    grid: SphereGrid,
    seed_yx: tuple[int, int],
    month: int,
    terrain: TerrainResult,
    climate: ClimateResult,
    cfg: WeatherConfig,
) -> list[tuple[float, float]]:
    h, w = terrain.land.shape
    y, x = map(float, seed_yx)
    pts: list[tuple[float, float]] = []
    for _ in range(cfg.hurricane_max_steps):
        iy = int(np.clip(round(y), 0, h - 1)); ix = int(round(x)) % w
        if terrain.land[iy, ix] or climate.temperature_c[month, iy, ix] < cfg.hurricane_sst_c - 2.0:
            break
        pts.append((float(grid.lat[iy, ix]), float(grid.lon[iy, ix])))
        u = float(climate.wind_u[month, iy, ix])
        v = float(climate.wind_v[month, iy, ix])
        # Synoptic steering plus a westward tropical drift. Pixel-space integration.
        x = (x + 1.3 * u - 0.45) % w
        y = np.clip(y + 1.15 * v, 0, h - 1)
        if abs(grid.lat[int(round(y)), int(round(x)) % w]) > 55:
            break
    return pts


def build_weather(
    grid: SphereGrid,
    terrain: TerrainResult,
    ocean: OceanResult,
    climate: ClimateResult,
    hydro: HydrologyResult,
    cfg: WeatherConfig,
    rng: np.random.Generator,
) -> WeatherResult:
    h, w = terrain.land.shape
    arid = np.char.startswith(climate.koppen, "B")
    polar = np.char.startswith(climate.koppen, "E")
    large_land = _large_land_mask(grid, terrain.land)
    slope = local_slope(ocean.elevation_km, grid)
    flat = slope < np.quantile(slope[terrain.land], 0.70) if np.any(terrain.land) else np.zeros_like(terrain.land)

    influence_sum = np.zeros((h, w), dtype=np.int8)
    tornado = np.zeros((h, w), float)
    lightning = np.zeros((h, w), float)
    terrain_gy, terrain_gx = grid.ops.metric_gradient(ocean.elevation_km)

    # Combine summer and winter maps by retaining the annual maximum at each location.
    for m in range(12):
        t = climate.temperature_c[m]
        p = climate.pressure_anomaly[m]
        u, v = climate.wind_u[m], climate.wind_v[m]
        warm = (t >= 18.0) & ~arid & ~polar
        substantial = warm & large_land
        lowp = (p <= np.percentile(p, 42)) & ~arid & ~polar
        sp = np.hypot(u, v)
        upslope = (u * terrain_gx + v * terrain_gy) / np.maximum(sp, 0.2)
        windward = (upslope > np.percentile(upslope[terrain.land], 75) if np.any(terrain.land) else False) & ~arid & ~polar
        s = warm.astype(np.int8) + substantial.astype(np.int8) + lowp.astype(np.int8) + windward.astype(np.int8)
        influence_sum = np.maximum(influence_sum, s)

        # Tornado proxy: warm moist air + strong temperature gradient/shear in 20–55° belts.
        gy, gx = grid.ops.metric_gradient(t)
        tempgrad = normalize01(np.hypot(gx, gy))
        if m > 0:
            shear = np.hypot(u - climate.wind_u[m - 1], v - climate.wind_v[m - 1])
        else:
            shear = np.hypot(u - climate.wind_u[-1], v - climate.wind_v[-1])
        moist = normalize01(climate.precipitation_mm[m])
        midlat = np.clip((np.abs(grid.lat) - 18) / 12, 0, 1) * np.clip((60 - np.abs(grid.lat)) / 12, 0, 1)
        tornado = np.maximum(tornado, tempgrad * normalize01(shear) * moist * midlat * warm)

    level = np.where(influence_sum == 0, 0,
                     np.where(influence_sum == 1, 1,
                              np.where(influence_sum == 2, 2, 3))).astype(np.int8)
    tornado_norm = normalize01(smooth_periodic(tornado, (1.5, 2.0)))
    tor_mask = tornado_norm > 0.78
    level[tor_mask] = 4
    # Transcript anchors: <1, 1–5, 5–15, 15+ flashes/km²/year.
    lightning = np.select(
        [level == 0, level == 1, level == 2, level >= 3],
        [0.4, 3.0, 10.0, 22.0], default=0.4
    ).astype(np.float32)
    lightning *= (0.65 + 0.7 * normalize01(climate.annual_precipitation_mm))

    # Fog: humid coasts with cold-current/upwelling support + mountain/river fog inland.
    humidity_proxy = normalize01(climate.annual_precipitation_mm)
    lowwind = 1.0 - normalize01(np.mean(np.hypot(climate.wind_u, climate.wind_v), axis=0))
    marine_fog = normalize01(ocean.upwelling + np.clip(-ocean.sst_anomaly_c / 5, 0, 1))
    marine_fog = grid.ops.grey_dilation(marine_fog, iterations=3)
    valley = normalize01(humidity_proxy * (1.0 - normalize01(slope))) * (hydro.rivers | hydro.lakes)
    fog = normalize01(0.55 * marine_fog * humidity_proxy + 0.30 * lowwind * humidity_proxy + 0.35 * valley)

    # Hurricane genesis: warm tropical ocean, low pressure, moisture, modest wind shear.
    warmest_month = np.argmax(climate.temperature_c, axis=0)
    yy, xx = np.indices((h, w))
    sst_warmest = climate.temperature_c[warmest_month, yy, xx]
    p_warmest = climate.pressure_anomaly[warmest_month, yy, xx]
    u_w = climate.wind_u[warmest_month, yy, xx]; v_w = climate.wind_v[warmest_month, yy, xx]
    wind_speed_w = np.hypot(u_w, v_w)
    shear_gy, shear_gx = grid.ops.metric_gradient(wind_speed_w)
    shear_proxy = normalize01(np.hypot(shear_gx, shear_gy))
    lat_ok = (np.abs(grid.lat) >= cfg.hurricane_lat_min_deg) & (np.abs(grid.lat) <= cfg.hurricane_lat_max_deg)
    genesis = terrain.ocean * lat_ok * np.clip((sst_warmest - cfg.hurricane_sst_c) / 5.0, 0, 1)
    genesis *= normalize01(-p_warmest) * (1.0 - 0.65 * shear_proxy)
    genesis = normalize01(smooth_periodic(genesis, (2, 3)))

    candidates = np.flatnonzero(genesis.ravel() > 0.35)
    tracks: list[dict] = []
    if len(candidates):
        weights = genesis.ravel()[candidates].astype(float)
        weights /= weights.sum()
        count = min(cfg.hurricane_seed_count, len(candidates))
        picks = rng.choice(candidates, size=count, replace=False, p=weights)
        for idx in picks:
            y, x = divmod(int(idx), w)
            month = int(warmest_month[y, x])
            pts = _sample_track(grid, (y, x), month, terrain, climate, cfg)
            if len(pts) >= 4:
                tracks.append({"month": month + 1, "points_lat_lon": pts, "steps": len(pts)})

    wind_speed = np.mean(np.hypot(climate.wind_u, climate.wind_v), axis=0)
    cold_precip = np.sum(climate.precipitation_mm * (climate.temperature_c <= 1.0), axis=0)
    blizzard = normalize01(cold_precip) * normalize01(wind_speed) * (climate.annual_temperature_c < 8)

    # Transcript weather rules: sandstorms in summer arid/semi-arid large flat land;
    # duststorms in dry large flat plains, excluding polar regions.
    sand = arid & terrain.land & flat & (np.abs(grid.lat) < 55)
    very_dry = climate.annual_precipitation_mm < np.percentile(climate.annual_precipitation_mm[terrain.land], 25) if np.any(terrain.land) else False
    dust = terrain.land & flat & very_dry & ~polar & ~sand

    # Aurora oval around a seeded geomagnetic axis.
    tilt = rng.uniform(4, 20)
    lon0 = rng.uniform(-180, 180)
    pole = np.array([np.sin(np.deg2rad(tilt)) * np.cos(np.deg2rad(lon0)),
                     np.sin(np.deg2rad(tilt)) * np.sin(np.deg2rad(lon0)),
                     np.cos(np.deg2rad(tilt))])
    dot = np.clip(grid.xyz @ pole, -1, 1)
    mag_lat = 90 - np.rad2deg(np.arccos(np.abs(dot)))
    aurora = np.exp(-0.5 * ((mag_lat - 68.0) / 6.5) ** 2)

    # Sea ice from monthly sea-surface thermal state. `max` is seasonal maximum extent; `min` is perennial.
    ocean_freeze = (climate.temperature_c <= -1.8) & terrain.ocean[None, :, :]
    sea_ice_max = np.any(ocean_freeze, axis=0)
    sea_ice_min = np.all(ocean_freeze, axis=0)

    # Warm, shallow, low-upwelling water proxy for reef-building coral; river mouths and polar/arid edge
    # effects are excluded. This automates the upwelling/coral cross-reference step.
    river_mouth_influence = grid.ops.binary_dilation(hydro.rivers, iterations=5)
    coral = (terrain.shelf & (climate.temperature_c.min(0) >= 18.0) &
             (climate.temperature_c.max(0) <= 32.5) & (ocean.upwelling < 0.50) &
             ~river_mouth_influence)

    meta = {
        "hurricane_track_count": len(tracks),
        "magnetic_axis_tilt_deg": float(tilt),
        "magnetic_axis_lon_deg": float(lon0),
        "thunderstorm_levels": {"0": "low-to-none", "1": "basic", "2": "moderate", "3": "severe", "4": "tornado-prone"},
        "sea_ice_max_fraction_ocean": float(np.average(sea_ice_max[terrain.ocean], weights=grid.cell_area_weights[terrain.ocean])) if np.any(terrain.ocean) else 0.0,
        "sea_ice_min_fraction_ocean": float(np.average(sea_ice_min[terrain.ocean], weights=grid.cell_area_weights[terrain.ocean])) if np.any(terrain.ocean) else 0.0,
        "coral_reef_area_fraction_ocean": grid.weighted_fraction(coral) / max(grid.weighted_fraction(terrain.ocean), 1e-12),
        "coral_reef_pixel_count": int(coral.sum()),
    }
    return WeatherResult(fog.astype(np.float32), level, lightning.astype(np.float32), tornado_norm.astype(np.float32),
                         genesis.astype(np.float32), tracks, blizzard.astype(np.float32), sand, dust,
                         aurora.astype(np.float32), sea_ice_max, sea_ice_min, coral, meta)
