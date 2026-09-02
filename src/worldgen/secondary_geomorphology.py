from __future__ import annotations

"""Reduced-order secondary geomorphic processes that actively modify topography.

The purpose of this layer is not to fake centimeter-scale precision on a global grid.
It supplies physically motivated, conservation-aware terrain tendencies for processes
missing from pure stream-power erosion: mass wasting, glaciers, groundwater/karst,
floodplains/fans, wetlands, channel instability, coast/estuaries, submarine canyons,
river capture and flexural/isostatic response.  The canonical hydrology is rebuilt
afterward, so capture/avulsion breaches can actually change drainage topology.
"""

from dataclasses import dataclass
import numpy as np

from . import hydrology_base as _base
from .grid import SphereGrid, normalize01, smooth_periodic
from .hydrology_advanced import transport_sediment_topological


@dataclass(slots=True)
class SecondaryGeomorphologyResult:
    elevation_km: np.ndarray
    regolith_thickness_m: np.ndarray
    landslide_erosion_m: np.ndarray
    glacial_erosion_m: np.ndarray
    glacial_deposition_m: np.ndarray
    spring_erosion_m: np.ndarray
    karst_erosion_m: np.ndarray
    floodplain_deposition_m: np.ndarray
    alluvial_fan_deposition_m: np.ndarray
    wetland_index: np.ndarray
    braided_channel_index: np.ndarray
    avulsion_potential: np.ndarray
    estuary_index: np.ndarray
    submarine_canyon_incision_m: np.ndarray
    coastal_erosion_m: np.ndarray
    river_capture_susceptibility: np.ndarray
    isostatic_adjustment_m: np.ndarray
    metadata: dict


def _different_basin_divides(grid: SphereGrid, basin: np.ndarray, land: np.ndarray) -> np.ndarray:
    b = np.asarray(basin, np.int64)
    divide = np.zeros_like(land, dtype=bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = grid.ops.neighbor_indices(dy, dx)
        other = b[ny, nx]
        divide |= land & (b > 0) & (other > 0) & (other != b)
    return divide


def _mass_scaled_deposition(
    source_m: np.ndarray,
    preference: np.ndarray,
    cell_area_km2: np.ndarray,
    retention: float,
) -> np.ndarray:
    src_volume = float(np.sum(np.maximum(source_m, 0.0) * cell_area_km2)) * float(np.clip(retention, 0.0, 1.0))
    pref = np.maximum(np.asarray(preference, float), 0.0) * cell_area_km2
    total = float(np.sum(pref))
    if src_volume <= 0.0 or total <= 1e-20:
        return np.zeros_like(source_m, dtype=np.float64)
    # depth * area = retained eroded volume, by construction.
    return (src_volume * pref / total) / np.maximum(cell_area_km2, 1e-20)


def evolve_secondary_geomorphology(
    grid: SphereGrid,
    terrain,
    ocean,
    climate,
    hydrology,
    geology,
    weather,
    appearance,
    tectonics,
    hydrology_cfg,
    tides=None,
) -> SecondaryGeomorphologyResult:
    z = np.asarray(terrain.elevation_km, float)
    land = np.asarray(terrain.land, bool)
    ocean_mask = np.asarray(terrain.ocean, bool)
    cell_area = _base._cell_area_km2(grid)
    gy, gx = grid.ops.metric_gradient(z)
    slope = np.hypot(gx, gy)
    slope_n = normalize01(slope, robust=True)
    temp = np.asarray(climate.annual_temperature_c, float)
    precip = np.maximum(np.asarray(climate.annual_precipitation_mm, float), 0.0)
    runoff = np.maximum(np.asarray(hydrology.runoff, float), 0.0)
    q = normalize01(np.asarray(hydrology.discharge_index, float), robust=True)
    storm = np.asarray(getattr(hydrology, "storminess_index", np.zeros_like(z)), float)
    baseflow = np.asarray(getattr(hydrology, "baseflow_mm_year", np.zeros_like(z)), float)
    groundwater = np.asarray(getattr(hydrology, "groundwater_storage_mm", np.zeros_like(z)), float)
    twi = np.asarray(getattr(hydrology, "topographic_wetness_index", np.zeros_like(z)), float)
    hand = np.asarray(getattr(hydrology, "height_above_nearest_drainage_m", np.full_like(z, 1e6)), float)
    channels = np.asarray(getattr(hydrology, "channel_class", hydrology.rivers.astype(np.uint8) * 3), np.uint8)
    vegetation = np.asarray(getattr(appearance, "vegetation_fraction", np.zeros_like(z)), float)
    snow = np.asarray(getattr(appearance, "snow_persistence", np.zeros_like(z)), float)
    rock = np.asarray(geology.rock_code, int)

    # Chemical/weathering production of movable regolith. Carbonate and unconsolidated
    # material weather faster; cold/dry and bare crystalline surfaces slower.
    lith_weather = np.asarray([1.35, 1.10, 1.55, 0.62, 0.52, 0.78, 0.73, 0.60, 0.45])[np.clip(rock, 0, 8)]
    thermal = np.exp(-((temp - 18.0) / 27.0) ** 2)
    moisture = np.clip(precip / 1400.0, 0.0, 2.0)
    regolith = np.clip((0.8 + 18.0 * thermal * np.sqrt(moisture)) * lith_weather * land, 0.0, 45.0)

    # Coarse-grid landslide susceptibility uses resolved relief, storm forcing,
    # vegetation/root cohesion and available regolith rather than a fixed slope only.
    critical = 0.60 + 0.18 * vegetation
    landslide_index = np.clip((slope_n - critical) / np.maximum(1.0 - critical, 0.05), 0.0, 1.0)
    landslide_index *= (0.35 + 0.65 * storm) * (0.45 + 0.55 * normalize01(regolith, robust=True)) * land
    landslide = np.clip(120.0 * landslide_index ** 1.4, 0.0, 120.0)

    # Glacier erosion/deposition uses persistent cold snow, slope and discharge-like ice
    # convergence. It remains an annual/long-term screening model, not Stokes ice flow.
    cold = np.clip((3.0 - temp) / 18.0, 0.0, 1.0)
    glacier = np.clip(cold * (0.25 + 0.75 * snow) * (0.25 + 0.75 * slope_n), 0.0, 1.0) * land
    glacial_erosion = np.clip(95.0 * glacier ** 1.25, 0.0, 95.0)
    glacier_margin = np.clip(smooth_periodic(glacier, (1.0, 1.25)) - glacier, 0.0, 1.0) * land
    glacial_deposition = _mass_scaled_deposition(glacial_erosion, glacier_margin, cell_area, 0.72)

    spring = normalize01(baseflow * (0.25 + slope_n) * (0.3 + normalize01(groundwater, robust=True)), robust=True) * land
    spring_erosion = np.clip(24.0 * spring, 0.0, 24.0)
    carbonate = (rock == 2).astype(float)
    karst_index = carbonate * np.clip(precip / 1100.0, 0.0, 1.8) * np.clip((temp + 5.0) / 30.0, 0.0, 1.0) * land
    karst_index *= 0.45 + 0.55 * normalize01(twi, robust=True)
    karst_erosion = np.clip(38.0 * karst_index, 0.0, 38.0)

    near_channel = grid.ops.binary_dilation(channels >= 2, iterations=2) & land
    low_hand = np.exp(-np.maximum(hand, 0.0) / 38.0)
    low_slope = np.exp(-slope / 0.0028)
    floodplain_pref = near_channel * low_hand * low_slope * (0.25 + 0.75 * q)
    erosion_sources = landslide + spring_erosion + karst_erosion
    routed_dep, _, exported = transport_sediment_topological(
        np.asarray(hydrology.filled_elevation_km, float),
        hydrology.flow_to,
        erosion_sources,
        q,
        _base._receiver_slope(hydrology.filled_elevation_km, hydrology.flow_to, grid),
        cell_area,
        land,
        hydrology_cfg,
    )
    floodplain_deposition = routed_dep * np.clip(0.45 + 0.85 * normalize01(floodplain_pref, robust=True), 0.0, 1.6)

    # Fan deposition where high-gradient channels enter flatter terrain.
    flow = np.asarray(hydrology.flow_to, np.int64).ravel()
    safe = np.where(flow >= 0, flow, 0)
    sf = slope.ravel()
    drop = np.zeros(flow.size, float)
    good = flow >= 0
    drop[good] = np.maximum(sf[good] - sf[safe[good]], 0.0)
    fan_pref = normalize01(drop.reshape(z.shape), robust=True) * q * (channels >= 2) * low_slope
    fan_pref = smooth_periodic(fan_pref, (0.7, 1.0)) * land
    fan_deposition = _mass_scaled_deposition(erosion_sources, fan_pref, cell_area, 0.12)

    sediment_supply = normalize01(
        np.asarray(hydrology.sediment_flux_index, float) + routed_dep + glacial_deposition,
        robust=True,
    )
    braided = np.clip(
        sediment_supply * (0.35 + 0.65 * storm) * (0.45 + 0.55 * q)
        * np.exp(-np.maximum(hand, 0.0) / 70.0) * (1.0 - 0.55 * vegetation),
        0.0, 1.0,
    ) * (channels >= 2)
    wetland = normalize01((0.6 + normalize01(twi, robust=True)) * low_hand * (0.35 + 0.65 * baseflow), robust=True) * land
    avulsion = normalize01((floodplain_deposition + fan_deposition) * (0.3 + 0.7 * storm) * low_hand, robust=True) * near_channel

    tidal = np.zeros_like(z)
    tidal_range = np.zeros_like(z)
    if tides is not None:
        tidal = np.asarray(tides.tidal_current_index, float)
        tidal_range = np.asarray(tides.tidal_range_m, float)
    mouth = land & grid.ops.binary_dilation(ocean_mask, iterations=1) & (channels >= 3)
    estuary = normalize01(grid.ops.grey_dilation(q * mouth, iterations=2) * (0.25 + tidal + 0.08 * tidal_range), robust=True) * ocean_mask

    shelf = np.asarray(terrain.shelf, bool) & ocean_mask
    mouth_ocean = grid.ops.binary_dilation(mouth, iterations=1) & ocean_mask
    canyon_seed = normalize01(grid.ops.grey_dilation(q * mouth, iterations=4), robust=True)
    canyon = shelf * canyon_seed * (0.25 + 0.75 * slope_n) * (0.55 + 0.45 * (1.0 - tidal))
    submarine_canyon = np.clip(85.0 * canyon, 0.0, 85.0)

    coast_land = land & grid.ops.binary_dilation(ocean_mask, iterations=1)
    wave = normalize01(np.asarray(ocean.current_speed, float) + 0.85 * tidal, robust=True)
    coastal_erosion = np.clip(32.0 * coast_land * wave * (0.45 + 0.55 * storm), 0.0, 32.0)

    # River capture: low/susceptible divides between distinct outlet basins receive a
    # small deterministic breach. The final hydrology rebuild decides whether flow
    # actually switches basin; no graph edge is manually forced.
    basin = np.asarray(getattr(hydrology, "basin_id", np.zeros_like(rock)), int)
    divides = _different_basin_divides(grid, basin, land) if np.any(basin > 0) else np.zeros_like(land)
    capture = normalize01(divides * q * (1.0 - slope_n) * (0.35 + 0.65 * storm), robust=True) * divides
    capture_breach = np.clip(18.0 * capture, 0.0, 18.0)

    total_erosion = landslide + spring_erosion + karst_erosion + glacial_erosion + coastal_erosion + capture_breach
    total_deposition = floodplain_deposition + fan_deposition + glacial_deposition

    # Airy-like broad rebound/subsidence proxy. Surface mass is still tracked
    # separately; this term represents mantle/lithosphere response, not sediment gain.
    unloading = smooth_periodic(total_erosion, (2.1, 2.8))
    loading = smooth_periodic(total_deposition, (2.1, 2.8))
    isostasy = np.clip(0.62 * unloading - 0.42 * loading, -85.0, 85.0) * land

    final = z - total_erosion / 1000.0 + total_deposition / 1000.0 + isostasy / 1000.0
    final -= submarine_canyon / 1000.0

    metadata = {
        "model": "coupled reduced-order weathering/mass-wasting/glacial/groundwater/karst/alluvial/coastal/capture/isostatic geomorphology",
        "max_landslide_erosion_m": float(np.max(landslide)),
        "max_glacial_erosion_m": float(np.max(glacial_erosion)),
        "max_karst_erosion_m": float(np.max(karst_erosion)),
        "max_coastal_erosion_m": float(np.max(coastal_erosion)),
        "max_submarine_canyon_incision_m": float(np.max(submarine_canyon)),
        "max_isostatic_adjustment_m": float(np.max(np.abs(isostasy))),
        "wetland_area_fraction_land": float(grid.weighted_fraction(wetland > 0.45) / max(grid.weighted_fraction(land), 1e-12)),
        "braided_channel_fraction_resolved_channels": float(grid.weighted_fraction(braided > 0.45) / max(grid.weighted_fraction(channels >= 2), 1e-12)),
        "river_capture_breach_cells": int(np.count_nonzero(capture_breach > 1.0)),
        "limitations": "global reduced-order tendencies; refined backends should solve glacier dynamics, aquifer heads, coastal waves and lithospheric flexure explicitly",
    }
    return SecondaryGeomorphologyResult(
        final.astype(np.float32), regolith.astype(np.float32), landslide.astype(np.float32),
        glacial_erosion.astype(np.float32), glacial_deposition.astype(np.float32),
        spring_erosion.astype(np.float32), karst_erosion.astype(np.float32),
        floodplain_deposition.astype(np.float32), fan_deposition.astype(np.float32),
        wetland.astype(np.float32), braided.astype(np.float32), avulsion.astype(np.float32),
        estuary.astype(np.float32), submarine_canyon.astype(np.float32), coastal_erosion.astype(np.float32),
        capture.astype(np.float32), isostasy.astype(np.float32), metadata,
    )


__all__ = ["SecondaryGeomorphologyResult", "evolve_secondary_geomorphology"]
