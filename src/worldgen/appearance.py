from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage

from .config import AppearanceConfig
from .grid import SphereGrid, normalize01, smooth_periodic, distance_to
from .terrain import TerrainResult
from .ocean import OceanResult
from .climate import ClimateResult
from .hydrology import HydrologyResult
from .geology import GeologyResult
from .weather import WeatherResult


@dataclass(slots=True)
class SurfaceAppearanceResult:
    vegetation_fraction: np.ndarray
    forest_fraction: np.ndarray
    grass_fraction: np.ndarray
    bare_ground_fraction: np.ndarray
    soil_moisture_index: np.ndarray
    snow_persistence: np.ndarray
    surface_albedo: np.ndarray
    water_turbidity: np.ndarray
    cloud_fraction_monthly: np.ndarray
    cloud_fraction_annual: np.ndarray
    true_color_rgb: np.ndarray
    true_color_january_rgb: np.ndarray
    true_color_july_rgb: np.ndarray
    true_color_with_clouds_rgb: np.ndarray
    true_color_january_with_clouds_rgb: np.ndarray
    true_color_july_with_clouds_rgb: np.ndarray
    metadata: dict


_ROCK_RGB = np.asarray([
    [0.63, 0.55, 0.42],  # unconsolidated sediment
    [0.67, 0.50, 0.34],  # clastic/sandstone
    [0.68, 0.68, 0.61],  # carbonate
    [0.56, 0.52, 0.50],  # granite
    [0.46, 0.44, 0.44],  # metamorphic
    [0.31, 0.32, 0.31],  # basalt/mafic
    [0.49, 0.45, 0.42],  # andesite
    [0.63, 0.56, 0.54],  # rhyolite
    [0.28, 0.34, 0.30],  # ultramafic/greenstone
], dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -18.0, 18.0)
    return 1.0 / (1.0 + np.exp(-x))


def _hillshade(grid: SphereGrid, elevation_km: np.ndarray) -> np.ndarray:
    z = np.asarray(elevation_km, float)
    gy, gx = np.gradient(z)
    sx = gx / np.maximum(grid.dx_km, 1e-3)
    sy = gy / max(grid.dy_km, 1e-3)
    slope = np.arctan(np.hypot(sx, sy) * 65.0)
    aspect = np.arctan2(-sx, sy)
    az = np.deg2rad(315.0); alt = np.deg2rad(42.0)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return normalize01(hs, robust=False).astype(np.float32)


def _monthly_vegetation(climate: ClimateResult, terrain: TerrainResult, cfg: AppearanceConfig) -> np.ndarray:
    t = climate.temperature_c.astype(float)
    p = climate.precipitation_mm.astype(float)
    warmth = _sigmoid((t - cfg.vegetation_temp_mid_c) / max(cfg.vegetation_temp_scale_c, 0.5))
    heat_stress = 1.0 - 0.62 * _sigmoid((t - 35.0) / 3.5)
    pet = np.maximum(12.0, 18.0 * np.maximum(t + 5.0, 0.0))
    moisture = p / np.maximum(p + pet, 1e-6)
    veg = warmth * heat_stress * np.sqrt(np.clip(moisture * 2.15, 0.0, 1.0))
    elev = np.maximum(terrain.elevation_km, 0.0)
    alpine = np.clip((elev - cfg.alpine_bare_start_km) /
                     max(cfg.alpine_bare_full_km - cfg.alpine_bare_start_km, 0.25), 0.0, 1.0)
    veg *= (1.0 - 0.88 * alpine)[None, ...]
    veg *= terrain.land[None, ...]
    return np.clip(veg, 0.0, 1.0).astype(np.float32)


def _render_rgb(
    grid: SphereGrid, terrain: TerrainResult, ocean: OceanResult, geology: GeologyResult,
    weather: WeatherResult, vegetation: np.ndarray, forest: np.ndarray, grass: np.ndarray,
    bare: np.ndarray, snow: np.ndarray, turbidity: np.ndarray, cfg: AppearanceConfig,
) -> np.ndarray:
    land = terrain.land; water = ~land
    h, w = land.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Water: deep ocean is dark blue; shelves become optically lighter/turquoise.
    depth = np.clip(ocean.depth_m.astype(float), 0.0, 11000.0)
    shallow = np.exp(-depth / 850.0)
    deep = np.clip(depth / 8000.0, 0.0, 1.0)
    rgb[..., 0] = np.where(water, 0.025 + 0.055 * shallow, 0.0)
    rgb[..., 1] = np.where(water, 0.15 + 0.43 * shallow, 0.0)
    rgb[..., 2] = np.where(water, 0.31 + 0.58 * shallow - 0.08 * deep, 0.0)
    # Sediment-rich water shifts toward green/brown and lightens where suspended load is high.
    turbid_rgb = np.asarray([0.30, 0.49, 0.39], np.float32)
    tt = np.clip(turbidity * cfg.turbidity_strength, 0.0, 0.85)[..., None]
    rgb[water] = rgb[water] * (1.0 - tt[water]) + turbid_rgb * tt[water]
    coral = weather.coral_reef & water
    if np.any(coral):
        rgb[coral] = 0.76 * rgb[coral] + 0.24 * np.asarray([0.08, 0.63, 0.60], np.float32)

    # Land substrate starts from coherent geology, then biological cover replaces the optical surface.
    rock = _ROCK_RGB[np.clip(geology.rock_code, 0, len(_ROCK_RGB) - 1)]
    rgb[land] = rock[land]
    forest_rgb = np.asarray([0.075, 0.29, 0.095], np.float32)
    grass_rgb = np.asarray([0.34, 0.49, 0.20], np.float32)
    fv = np.clip(forest, 0.0, 0.95)[..., None]
    gv = np.clip(grass, 0.0, 0.90)[..., None]
    rgb[land] = rgb[land] * (1.0 - fv[land]) + forest_rgb * fv[land]
    rgb[land] = rgb[land] * (1.0 - 0.72 * gv[land]) + grass_rgb * (0.72 * gv[land])
    # Bare dry soil warms toward ochre rather than merely desaturating the geology.
    dry_rgb = np.asarray([0.67, 0.57, 0.39], np.float32)
    bv = np.clip(bare * 0.56, 0.0, 0.56)[..., None]
    rgb[land] = rgb[land] * (1.0 - bv[land]) + dry_rgb * bv[land]

    # Beaches/deltaic lowlands are subtle—avoid the previous neon cyan coastline outline.
    coast_land = land & ndimage.binary_dilation(water, iterations=1)
    beach = coast_land & (terrain.elevation_km < 0.10) & (vegetation < 0.65)
    if np.any(beach):
        rgb[beach] = 0.72 * rgb[beach] + 0.28 * np.asarray([0.78, 0.72, 0.56], np.float32)

    sv = np.clip(snow, 0.0, 1.0)[..., None]
    rgb[land] = rgb[land] * (1.0 - sv[land]) + np.asarray([0.94, 0.96, 0.98], np.float32) * sv[land]

    # Sea ice is treated separately from land snow.
    ice = weather.sea_ice_max & water
    if np.any(ice):
        rgb[ice] = 0.20 * rgb[ice] + 0.80 * np.asarray([0.88, 0.94, 0.97], np.float32)

    hs = _hillshade(grid, ocean.elevation_km)
    strength = float(np.clip(cfg.hillshade_strength, 0.0, 0.65))
    shade = (1.0 - strength) + strength * (0.55 + 0.75 * hs)
    rgb[land] *= shade[land, None]
    rgb[water] *= (0.93 + 0.10 * hs[water, None])
    return np.clip(rgb, 0.0, 1.0)


def _composite_clouds(rgb: np.ndarray, cloud: np.ndarray, cfg: AppearanceConfig) -> np.ndarray:
    """Simple top-of-atmosphere visible composite over the cloud-free surface proxy."""
    c = np.clip(np.asarray(cloud, float) * float(cfg.cloud_max_optical_opacity), 0.0, 0.92)[..., None]
    # Thick cloud is nearly neutral-white; thinner cloud retains a subtle blue cast from Rayleigh scattering.
    cloud_rgb = np.asarray([0.94, 0.96, 0.98], np.float32)
    return np.clip(np.asarray(rgb, float) * (1.0 - c) + cloud_rgb * c, 0.0, 1.0).astype(np.float32)


def build_surface_appearance(
    grid: SphereGrid, terrain: TerrainResult, ocean: OceanResult, climate: ClimateResult,
    hydro: HydrologyResult, geology: GeologyResult, weather: WeatherResult, cfg: AppearanceConfig,
) -> SurfaceAppearanceResult:
    land = terrain.land; water = ~land
    veg_m = _monthly_vegetation(climate, terrain, cfg)
    vegetation = np.mean(veg_m, axis=0)
    p_ann = climate.annual_precipitation_mm.astype(float)
    forest_climate = _sigmoid((p_ann - cfg.forest_precip_mid_mm_year) /
                              max(cfg.forest_precip_scale_mm_year, 50.0))
    forest = np.clip(vegetation * forest_climate * (1.0 - 0.55 * terrain.ruggedness), 0.0, 1.0) * land
    grass = np.clip(vegetation - 0.72 * forest, 0.0, 1.0) * land
    bare = np.clip(1.0 - vegetation, 0.0, 1.0) * land

    t = climate.temperature_c.astype(float); p = climate.precipitation_mm.astype(float)
    pet = np.maximum(20.0, 18.0 * np.maximum(t + 5.0, 0.0))
    soil_month = p / np.maximum(p + pet, 1e-6)
    soil = np.clip(0.72 * np.mean(soil_month, axis=0) + 0.28 * normalize01(hydro.runoff), 0.0, 1.0) * land
    snow_month = ((t < 0.0) * np.clip(p / 45.0, 0.0, 1.0)).astype(float) * land[None, ...]
    snow_persistence = np.clip(np.mean(snow_month, axis=0) + 0.30 * climate.snow_fraction, 0.0, 1.0) * land

    # Coastal turbidity comes from river sediment export, delta aggradation and shallow shelf resuspension.
    sediment_source = np.clip(hydro.sediment_flux_index + normalize01(hydro.delta_deposition_m), 0.0, 2.0) * land
    sig = max(0.7, float(cfg.turbidity_spread_sigma_px))
    plume = smooth_periodic(sediment_source, (sig, sig * 1.45))
    coast_dist = distance_to(land, grid)
    shelf = water * np.exp(-coast_dist / 210.0) * np.exp(-ocean.depth_m / 220.0)
    turbidity = normalize01(plume * water + 0.28 * shelf) * water

    # Cloud fraction is a diagnostic optical field rather than a separate microphysics model. It couples
    # atmospheric humidity, precipitation and low-pressure ascent, then receives mild mesoscale smoothing.
    hum = climate.humidity_proxy.astype(float)
    pr = climate.precipitation_mm.astype(float)
    pres = climate.pressure_anomaly.astype(float)
    hc = _sigmoid((hum - cfg.cloud_humidity_mid) / 0.055)
    pc = _sigmoid((pr - cfg.cloud_precip_mid_mm_month) / 24.0)
    asc = _sigmoid((-pres - 0.25) / 1.9)
    cloud_m = np.clip(0.48 * hc + 0.38 * pc + 0.14 * asc, 0.0, 1.0)
    cs = max(0.0, float(cfg.cloud_smoothing_sigma_px))
    if cs > 0.05:
        for m in range(12):
            cloud_m[m] = smooth_periodic(cloud_m[m], (cs, cs * 1.35))
    cloud_m = np.clip(cloud_m, 0.0, 1.0).astype(np.float32)
    cloud_ann = np.mean(cloud_m, axis=0).astype(np.float32)

    albedo = np.full(land.shape, float(cfg.ocean_albedo), dtype=np.float32)
    land_albedo = (cfg.vegetation_albedo * vegetation + cfg.desert_albedo * bare + 0.20 * (1.0 - vegetation - bare))
    albedo[land] = np.clip(land_albedo[land], 0.08, 0.45)
    albedo[land] = albedo[land] * (1.0 - snow_persistence[land]) + cfg.snow_albedo * snow_persistence[land]
    albedo[weather.sea_ice_max & water] = 0.62

    # Seasonal render inputs use that month's vegetation and snow; annual uses persistent fractions.
    jan_snow = np.clip(snow_month[0], 0.0, 1.0)
    jul_snow = np.clip(snow_month[6], 0.0, 1.0)
    forest_ratio = forest / np.maximum(vegetation, 1e-6)
    jan_forest = veg_m[0] * np.clip(forest_ratio, 0.0, 1.0)
    jul_forest = veg_m[6] * np.clip(forest_ratio, 0.0, 1.0)
    jan_grass = np.clip(veg_m[0] - 0.72 * jan_forest, 0.0, 1.0)
    jul_grass = np.clip(veg_m[6] - 0.72 * jul_forest, 0.0, 1.0)

    annual_rgb = _render_rgb(grid, terrain, ocean, geology, weather, vegetation, forest, grass, bare, snow_persistence, turbidity, cfg)
    jan_rgb = _render_rgb(grid, terrain, ocean, geology, weather, veg_m[0], jan_forest, jan_grass,
                          np.clip(1.0 - veg_m[0], 0.0, 1.0) * land, jan_snow, turbidity, cfg)
    jul_rgb = _render_rgb(grid, terrain, ocean, geology, weather, veg_m[6], jul_forest, jul_grass,
                          np.clip(1.0 - veg_m[6], 0.0, 1.0) * land, jul_snow, turbidity, cfg)
    annual_cloudy = _composite_clouds(annual_rgb, cloud_ann, cfg)
    jan_cloudy = _composite_clouds(jan_rgb, cloud_m[0], cfg)
    jul_cloudy = _composite_clouds(jul_rgb, cloud_m[6], cfg)

    meta = {
        "model": "cloud-free surface reflectance proxy from climate, vegetation, geology, snow, bathymetry, sediment turbidity and hillshade",
        "mean_land_vegetation_fraction": float(np.average(vegetation[land], weights=grid.cell_area_weights[land])) if np.any(land) else 0.0,
        "mean_land_forest_fraction": float(np.average(forest[land], weights=grid.cell_area_weights[land])) if np.any(land) else 0.0,
        "mean_land_soil_moisture_index": float(np.average(soil[land], weights=grid.cell_area_weights[land])) if np.any(land) else 0.0,
        "mean_ocean_turbidity_index": float(np.average(turbidity[water], weights=grid.cell_area_weights[water])) if np.any(water) else 0.0,
        "mean_cloud_fraction": float(np.average(cloud_ann, weights=grid.cell_area_weights)),
    }
    return SurfaceAppearanceResult(
        vegetation.astype(np.float32), forest.astype(np.float32), grass.astype(np.float32), bare.astype(np.float32),
        soil.astype(np.float32), snow_persistence.astype(np.float32), albedo.astype(np.float32), turbidity.astype(np.float32),
        cloud_m.astype(np.float32), cloud_ann.astype(np.float32),
        (annual_rgb * 255).astype(np.uint8), (jan_rgb * 255).astype(np.uint8), (jul_rgb * 255).astype(np.uint8),
        (annual_cloudy * 255).astype(np.uint8), (jan_cloudy * 255).astype(np.uint8), (jul_cloudy * 255).astype(np.uint8), meta,
    )
