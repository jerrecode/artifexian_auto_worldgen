from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import TerrainConfig, TectonicsConfig, NoiseConfig
from .grid import SphereGrid, distance_to, normalize01, smooth_periodic
from .tectonics import TectonicResult
from .noise import hybrid_multifractal, hybrid_noise01, noise_kwargs, TERRAIN_BLEND, TECTONIC_BLEND, NoiseBlend


@dataclass(slots=True)
class TerrainResult:
    elevation_km: np.ndarray
    land: np.ndarray
    ocean: np.ndarray
    coast: np.ndarray
    shelf: np.ndarray
    sea_level_offset_km: float
    mountain_strength: np.ndarray
    lowland_strength: np.ndarray
    ruggedness: np.ndarray
    metadata: dict


def _fractal_noise(shape: tuple[int, int], rng: np.random.Generator, octaves: int = 7, noise_cfg: NoiseConfig | None = None) -> np.ndarray:
    return hybrid_multifractal(shape, rng, base_scale_px=max(shape[0] / 7.5, 3.0),
                               **noise_kwargs(noise_cfg, profile=TERRAIN_BLEND, octaves=octaves))


def _ridge_noise(shape: tuple[int, int], rng: np.random.Generator, octaves: int = 6, noise_cfg: NoiseConfig | None = None) -> np.ndarray:
    profile = NoiseBlend(0.26, 0.50, 0.08, 0.16)
    return hybrid_noise01(shape, rng, base_scale_px=max(shape[0] / 8.0, 3.0),
                          **noise_kwargs(noise_cfg, profile=profile, octaves=octaves))


def _rough_gradient(elev: np.ndarray, grid: SphereGrid) -> np.ndarray:
    gy, gx = grid.ops.metric_gradient(np.asarray(elev, float))
    return normalize01(np.hypot(gx, gy))


def rebuild_terrain_from_elevation(
    grid: SphereGrid,
    tect: TectonicResult,
    cfg: TerrainConfig,
    elevation_km: np.ndarray,
    sea_level_offset_km: float = 0.0,
    metadata_extra: dict | None = None,
) -> TerrainResult:
    elev = np.asarray(elevation_km, float).copy()
    land = elev > 0.0
    ocean = ~land
    coast_land = land & grid.ops.binary_dilation(ocean, iterations=1)
    coast_ocean = ocean & grid.ops.binary_dilation(land, iterations=1)
    coast = coast_land | coast_ocean
    active_margin = distance_to(tect.convergent, grid) < 180.0
    dist_land = distance_to(land, grid)
    shelf_width = np.where(active_margin, cfg.shelf_width_km_active, cfg.shelf_width_km_passive)
    shelf = ocean & (dist_land <= shelf_width)

    relief = np.maximum(elev, 0.0)
    rough = _rough_gradient(elev, grid)
    mountain = normalize01(relief + 1.2 * tect.convergence_strength + 0.55 * tect.strain_field + 0.30 * rough)
    # Continuous lowland score: flat, low-elevation land is strongest.
    low = land.astype(float) * np.exp(-np.maximum(elev, 0.0) / 0.85) * (1.0 - 0.70 * rough)
    low = normalize01(low) * land
    # Geometry-derived metadata is canonical and must never be overwritten by a
    # carried-forward metadata dictionary from an older shoreline.  Build from the
    # historical/provenance payload first, then stamp fields that describe this exact
    # raster last.
    meta = dict(metadata_extra or {})
    meta.update({
        "actual_land_fraction": grid.weighted_fraction(land),
        "actual_ocean_fraction": grid.weighted_fraction(ocean),
        "sea_level_offset_km": float(sea_level_offset_km),
        "max_elevation_km_pre_bathymetry": float(np.max(elev)),
        "geometry_metadata_reconciled": True,
    })
    return TerrainResult(elev.astype(np.float32), land, ocean, coast, shelf, float(sea_level_offset_km),
                         mountain.astype(np.float32), low.astype(np.float32), rough.astype(np.float32), meta)


def _coastal_rework(grid: SphereGrid, elev: np.ndarray, cfg: TerrainConfig, target_land: float) -> tuple[np.ndarray, dict]:
    """Apply conservative marine reworking near sea level and remove sub-resolution island specks."""
    z = np.asarray(elev, float).copy()
    strength = float(np.clip(cfg.coastal_reworking_strength, 0.0, 0.75))
    if strength > 0:
        band = np.abs(z) <= max(float(cfg.coastal_reworking_band_km), 0.02)
        sm = smooth_periodic(z, max(0.35, float(cfg.coastal_reworking_sigma_px)))
        z[band] = (1.0 - strength) * z[band] + strength * sm[band]
        # Restore requested spherical land fraction after local coastal diffusion.
        shift = grid.weighted_quantile(z, 1.0 - target_land)
        z -= shift
    removed_area = 0.0
    min_area = max(0.0, float(cfg.min_island_area_km2))
    if min_area > 0:
        land = z > 0
        labs, n = grid.ops.connected_components(land)
        surface_km2 = 4.0 * np.pi * grid.radius_km ** 2
        for lab in range(1, n + 1):
            m = labs == lab
            area = float(np.sum(grid.cell_area_weights[m]) * surface_km2)
            if area < min_area:
                # Sink only slightly below local sea level; later shelf/bathymetry logic takes over.
                z[m] = np.minimum(z[m], -0.015)
                removed_area += area
        if removed_area > 0:
            shift = grid.weighted_quantile(z, 1.0 - target_land)
            z -= shift
    return z, {
        "coastal_reworking_strength": strength,
        "min_island_area_km2": min_area,
        "removed_micro_island_area_km2": float(removed_area),
    }


def build_terrain(
    grid: SphereGrid,
    tect: TectonicResult,
    tcfg: TectonicsConfig,
    cfg: TerrainConfig,
    rng: np.random.Generator,
    noise_cfg: NoiseConfig | None = None,
) -> TerrainResult:
    cont = tect.continental_crust
    oceanic = ~cont
    dconv = distance_to(tect.convergent, grid)
    ddiv = distance_to(tect.divergent, grid)

    # Active convergence is variable in intensity and heavily modulated by accumulated strain.
    conv_core = np.exp(-dconv / 260.0) * (0.28 + 0.72 * smooth_periodic(tect.convergence_strength, (1.0, 1.4)))
    collision = conv_core * cont
    ocean_arc = conv_core * oceanic
    conv_uplift = tcfg.mountain_uplift_km * (1.05 * collision + 0.22 * ocean_arc)

    # Current ridges and continental rifts are separate morphologies. Rifts have depressed axes and
    # uplifted shoulders rather than being represented by one broad smooth low.
    ridge = np.exp(-ddiv / 150.0) * (0.30 + 0.70 * smooth_periodic(tect.divergence_strength, (0.8, 1.2)))
    ridge_uplift = ridge * tcfg.ridge_uplift_km
    rift_axis = tect.paleo_divergence * np.exp(-tect.rift_age_myr / 130.0)
    rift_axis += np.exp(-ddiv / 115.0) * cont * 0.8
    rift_axis = np.clip(rift_axis, 0, 1)
    rift_shoulder = np.exp(-ddiv / 300.0) - np.exp(-ddiv / 95.0)
    rift_shoulder = np.maximum(rift_shoulder, 0) * cont * cfg.rift_shoulder_uplift_km

    # Ancient orogens decay but preserve structural grain. Strain remembers multiple cycles of
    # compression/shear, producing broken mountain belts instead of uniformly blurred arcs.
    erosion_scale_myr = max(150.0, 3000.0 / max(cfg.erosion_m_per_myr, 0.1))
    ancient = tect.paleo_convergence * np.exp(-tect.orogen_age_myr / erosion_scale_myr)
    ancient_uplift = ancient * (0.72 * tcfg.mountain_uplift_km)

    # Multi-scale detail fields. Ridge noise is preferentially injected into active deformation belts,
    # while ordinary fractal noise supplies drainage-scale roughness across stable crust.
    fractal = _fractal_noise(cont.shape, rng, cfg.fractal_octaves, noise_cfg)
    ridged = _ridge_noise(cont.shape, rng, max(4, cfg.fractal_octaves - 1), noise_cfg)
    broad = hybrid_multifractal(
        cont.shape, rng, base_scale_px=max(grid.height / 9.0, 5.0),
        **noise_kwargs(noise_cfg, profile=NoiseBlend(0.56, 0.10, 0.18, 0.16), octaves=max(3, cfg.fractal_octaves - 3)),
    )
    deformation = normalize01(conv_core + rift_axis + 0.75 * tect.strain_field)
    detail = tcfg.terrain_noise_km * (0.58 * fractal + cfg.relief_detail_strength * (ridged - 0.42) * (0.30 + 0.90 * deformation))

    # Fault-block relief creates local alternating highs/lows around subplate boundaries and transform
    # systems. The sign field is correlated, so this resembles horst/graben and broken ranges rather
    # than salt-and-pepper pixel noise.
    block_sign = hybrid_multifractal(
        cont.shape, rng, base_scale_px=max(3.5, grid.height / 48.0),
        **noise_kwargs(noise_cfg, profile=TECTONIC_BLEND, octaves=max(4, cfg.fractal_octaves - 2)),
    )
    fault_relief = cfg.fault_block_relief_km * tect.stress_field * np.tanh(block_sign)

    plume = 1.35 * tect.lip_strength + 0.75 * tect.hotspot_strength
    elev = np.where(cont, 0.72 + 0.78 * broad, -3.65 + 0.40 * broad)
    elev += detail
    elev += conv_uplift
    elev += ancient_uplift * np.where(cont, 1.0, 0.30)
    elev += ridge_uplift * oceanic
    elev += rift_shoulder
    elev -= 1.25 * rift_axis * cont
    elev += plume * np.where(cont, 0.60, 0.42)
    elev += fault_relief * np.where(cont, 1.0, 0.32)

    # Subduction trenches lie primarily on oceanic crust and scale with active convergence strength.
    trench = np.exp(-dconv / 72.0) * tcfg.trench_depth_km * (0.35 + 0.65 * tect.convergence_strength)
    elev -= trench * oceanic

    # Area-weighted sea level targeting is preserved; jagged continental crust and multiscale relief now
    # make the resulting coastlines much less circular/smooth.
    target_land = tcfg.continental_fraction_target
    threshold = grid.weighted_quantile(elev, 1.0 - target_land)
    elev -= threshold
    elev, coast_meta = _coastal_rework(grid, elev, cfg, target_land)
    result = rebuild_terrain_from_elevation(
        grid, tect, cfg, elev, float(threshold),
        {
            "target_land_fraction": target_land,
            "erosion_m_per_myr": cfg.erosion_m_per_myr,
            "terrain_fractal_octaves": cfg.fractal_octaves,
            "fault_block_relief_km": cfg.fault_block_relief_km,
            "noise_model": "hybrid multi-type multifractal with decreasing octave amplitude and domain warping",
            **coast_meta,
        },
    )
    return result
