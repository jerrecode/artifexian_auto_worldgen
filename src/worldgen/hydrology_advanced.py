from __future__ import annotations

"""Advanced conservative hydrology layered over the established spherical solver.

This module deliberately reuses the canonical Priority-Flood/receiver graph.  It
adds the missing physical hierarchy around that graph: monthly soil/snow/groundwater
water balance, bankfull flow, scale-aware channel initiation, true outlet watersheds,
sub-grid drainage density and one-pass conservative sediment routing.
"""

from collections import OrderedDict
from dataclasses import dataclass
import copy
import hashlib
import math
from typing import Any

import numpy as np

from . import hydrology_base as _base
from .drainage import DrainageGraph
from .grid import SphereGrid, normalize01
from .surface_evolution import evolve_surface as _policy_evolve_surface
from .watersheds import WatershedHierarchy, build_watershed_hierarchy


_INFILTRATION = np.asarray([0.62, 0.54, 0.80, 0.47, 0.40, 0.52, 0.49, 0.58, 0.44], dtype=float)
_SOIL_CAPACITY = np.asarray([190., 145., 245., 115., 105., 135., 125., 155., 110.], dtype=float)
_WATER_CACHE: OrderedDict[tuple, "WaterBalanceResult"] = OrderedDict()
_LAST_SEDIMENT_LEDGER: dict[str, float] = {}


@dataclass(slots=True)
class WaterBalanceResult:
    total_runoff_mm_year: np.ndarray
    surface_runoff_mm_year: np.ndarray
    baseflow_mm_year: np.ndarray
    groundwater_recharge_mm_year: np.ndarray
    actual_evapotranspiration_mm_year: np.ndarray
    soil_water_storage_mm: np.ndarray
    groundwater_storage_mm: np.ndarray
    snowpack_mm: np.ndarray
    storminess_index: np.ndarray
    closure_residual_mm_year: np.ndarray
    metadata: dict


@dataclass(slots=True)
class AdvancedHydrologyResult:
    base: Any
    runoff: np.ndarray
    rivers: np.ndarray
    stream_order: np.ndarray
    river_width_proxy: np.ndarray
    basin_id: np.ndarray
    channel_class: np.ndarray
    subbasin_level_1: np.ndarray
    subbasin_level_2: np.ndarray
    subbasin_level_3: np.ndarray
    exorheic: np.ndarray
    distance_to_outlet_km: np.ndarray
    topographic_wetness_index: np.ndarray
    height_above_nearest_drainage_m: np.ndarray
    surface_runoff_mm_year: np.ndarray
    baseflow_mm_year: np.ndarray
    groundwater_recharge_mm_year: np.ndarray
    actual_evapotranspiration_mm_year: np.ndarray
    soil_water_storage_mm: np.ndarray
    groundwater_storage_mm: np.ndarray
    snowpack_mm: np.ndarray
    storminess_index: np.ndarray
    bankfull_discharge_index: np.ndarray
    subgrid_drainage_density_km_per_km2: np.ndarray
    river_centerlines: list[dict]
    metadata: dict

    def __getattr__(self, name: str):
        return getattr(self.base, name)



def _land_digest(land: np.ndarray) -> bytes:
    packed = np.packbits(np.asarray(land, dtype=np.uint8).ravel())
    return hashlib.blake2b(packed, digest_size=8).digest()


def _water_cache_key(climate: Any, geology: Any | None, cfg: Any, land: np.ndarray) -> tuple:
    return (
        id(climate),
        id(geology),
        _land_digest(land),
        float(getattr(cfg, "runoff_base_fraction", 0.24)),
        float(getattr(cfg, "soil_storage_multiplier", 1.0)),
        float(getattr(cfg, "groundwater_recession_fraction_month", 0.065)),
        float(getattr(cfg, "storm_runoff_strength", 1.0)),
    )


def build_water_balance(climate: Any, land: np.ndarray, geology: Any | None, cfg: Any) -> WaterBalanceResult:
    """Reduced-order monthly water/energy-compatible land water balance.

    The bucket model explicitly conserves water between precipitation, snow storage,
    soil storage, groundwater, evapotranspiration, fast runoff and baseflow.  It is
    not a Richards-equation groundwater solver; it is designed as the conservative
    global backend on which local/refined groundwater solvers can later sit.
    """
    p = np.maximum(np.asarray(climate.precipitation_mm, dtype=np.float64), 0.0)
    t = np.asarray(climate.temperature_c, dtype=np.float64)
    if p.ndim != 3 or t.shape != p.shape:
        raise ValueError("monthly precipitation and temperature must have shape (month, y, x)")
    lf = np.asarray(land, dtype=bool)
    shape = lf.shape

    if geology is None:
        infiltration = np.full(shape, 0.58, dtype=np.float64)
        soil_capacity = np.full(shape, 165.0, dtype=np.float64)
    else:
        rock = np.clip(np.asarray(geology.rock_code, dtype=int), 0, len(_INFILTRATION) - 1)
        infiltration = _INFILTRATION[rock]
        soil_capacity = _SOIL_CAPACITY[rock]
    soil_capacity *= max(float(getattr(cfg, "soil_storage_multiplier", 1.0)), 0.05)
    infiltration = np.clip(infiltration, 0.08, 0.92)

    soil = 0.52 * soil_capacity * lf
    groundwater = 28.0 * infiltration * lf
    snow = np.zeros(shape, dtype=np.float64)
    recession = float(np.clip(getattr(cfg, "groundwater_recession_fraction_month", 0.065), 0.005, 0.45))
    storm_strength = float(np.clip(getattr(cfg, "storm_runoff_strength", 1.0), 0.1, 4.0))
    spinup_years = max(1, int(getattr(cfg, "water_balance_spinup_years", 3)))

    annual_fast = np.zeros(shape, dtype=np.float64)
    annual_base = np.zeros(shape, dtype=np.float64)
    annual_recharge = np.zeros(shape, dtype=np.float64)
    annual_et = np.zeros(shape, dtype=np.float64)
    start_storage = None

    mean_monthly_p = np.mean(p, axis=0)
    p_cv = np.std(p, axis=0) / np.maximum(mean_monthly_p, 1.0)
    thermal_convective = np.mean(np.clip((t - 8.0) / 27.0, 0.0, 1.0), axis=0)

    for year in range(spinup_years):
        if year == spinup_years - 1:
            annual_fast.fill(0.0); annual_base.fill(0.0); annual_recharge.fill(0.0); annual_et.fill(0.0)
            start_storage = soil + groundwater + snow
        for month in range(p.shape[0]):
            pm = p[month] * lf
            tm = t[month]
            snowfall = np.where(tm < 0.0, pm, 0.0)
            rainfall = pm - snowfall
            snow += snowfall
            melt = np.minimum(snow, np.maximum(tm, 0.0) * 11.0)
            snow -= melt
            liquid = rainfall + melt

            convective = np.clip((tm - 8.0) / 25.0, 0.0, 1.0)
            relative_eventiness = np.clip(pm / np.maximum(mean_monthly_p, 8.0) - 0.65, 0.0, 2.5)
            storm_fraction = np.clip(
                storm_strength * (0.08 + 0.28 * convective + 0.12 * relative_eventiness) * (1.10 - 0.62 * infiltration),
                0.015,
                0.72,
            )
            direct_storm = liquid * storm_fraction
            infiltrable = liquid - direct_storm

            deficit = np.clip(1.0 - soil / np.maximum(soil_capacity, 1.0), 0.0, 1.0)
            infiltration_capacity = (34.0 + 105.0 * infiltration) * (0.35 + 0.65 * deficit)
            infiltrated = np.minimum(infiltrable, infiltration_capacity)
            horton = infiltrable - infiltrated
            soil += infiltrated
            saturation = np.maximum(soil - soil_capacity, 0.0)
            soil = np.minimum(soil, soil_capacity)

            # Screening-grade monthly PET.  The energy-limited climate backend can
            # replace this without changing the conserved reservoir topology.
            pet = np.maximum(0.0, 2.55 * (tm + 5.0)) * lf
            et = np.minimum(soil, pet)
            soil -= et

            recharge = np.maximum(soil - 0.70 * soil_capacity, 0.0) * (0.24 + 0.28 * infiltration)
            soil -= recharge
            groundwater += recharge
            baseflow = groundwater * recession
            groundwater -= baseflow

            fast = direct_storm + horton + saturation
            if year == spinup_years - 1:
                annual_fast += fast
                annual_base += baseflow
                annual_recharge += recharge
                annual_et += et

    if start_storage is None:
        start_storage = soil + groundwater + snow
    end_storage = soil + groundwater + snow
    annual_p = np.sum(p, axis=0) * lf
    total_runoff = annual_fast + annual_base
    residual = annual_p + start_storage - (total_runoff + annual_et + end_storage)

    storminess = np.clip(
        0.48 * (annual_fast / np.maximum(total_runoff, 1.0))
        + 0.30 * np.clip(p_cv / 1.5, 0.0, 1.0)
        + 0.22 * thermal_convective,
        0.0,
        1.0,
    ) * lf
    max_abs_resid = float(np.max(np.abs(residual[lf]))) if np.any(lf) else 0.0
    meta = {
        "model": "monthly conservative snow + soil bucket + infiltration-excess/saturation-excess runoff + groundwater recharge/baseflow",
        "spinup_years": spinup_years,
        "max_absolute_water_balance_residual_mm_year": max_abs_resid,
        "mean_land_runoff_mm_year": float(np.mean(total_runoff[lf])) if np.any(lf) else 0.0,
        "mean_land_baseflow_fraction": float(np.mean(annual_base[lf] / np.maximum(total_runoff[lf], 1.0))) if np.any(lf) else 0.0,
        "limitations": "global reduced-order bucket; no local Richards equation, aquifer head PDE or river-groundwater exchange solve yet",
    }
    return WaterBalanceResult(
        total_runoff.astype(np.float32),
        annual_fast.astype(np.float32),
        annual_base.astype(np.float32),
        annual_recharge.astype(np.float32),
        annual_et.astype(np.float32),
        soil.astype(np.float32),
        groundwater.astype(np.float32),
        snow.astype(np.float32),
        storminess.astype(np.float32),
        residual.astype(np.float32),
        meta,
    )


def cached_water_balance(climate: Any, land: np.ndarray, geology: Any | None, cfg: Any) -> WaterBalanceResult:
    key = _water_cache_key(climate, geology, cfg, land)
    found = _WATER_CACHE.get(key)
    if found is not None:
        _WATER_CACHE.move_to_end(key)
        return found
    result = build_water_balance(climate, land, geology, cfg)
    _WATER_CACHE[key] = result
    _WATER_CACHE.move_to_end(key)
    while len(_WATER_CACHE) > 3:
        _WATER_CACHE.popitem(last=False)
    return result


def runoff_mm_advanced(climate: Any, land: np.ndarray, geology: Any | None, cfg: Any) -> np.ndarray:
    """Drop-in replacement for hydrology_base._runoff_mm."""
    return np.asarray(cached_water_balance(climate, land, geology, cfg).total_runoff_mm_year, dtype=np.float64)


def transport_sediment_topological(
    routing_z: np.ndarray,
    flow: np.ndarray,
    erosion_m: np.ndarray,
    discharge_norm: np.ndarray,
    slope: np.ndarray,
    cell_area: np.ndarray,
    land: np.ndarray,
    cfg: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conservatively route all eroded sediment once in topological order.

    Unlike the historical fixed-pass algorithm, transport distance is not limited by
    an arbitrary number of receiver hops.  Every parcel is either deposited on land,
    retained at an internal sink, or exported through a shoreline receiver in one
    O(N) traversal.
    """
    shape = np.asarray(routing_z).shape
    graph = DrainageGraph.from_receiver(flow, shape)
    recv = graph.receiver
    lf = np.asarray(land, dtype=bool).ravel()
    area = np.asarray(cell_area, dtype=np.float64).ravel()
    q = np.asarray(discharge_norm, dtype=np.float64).ravel()
    s = np.asarray(slope, dtype=np.float64).ravel()
    flatness = np.exp(-s / max(float(getattr(cfg, "sediment_deposition_slope_scale", 0.0028)), 1.0e-6))
    depfrac = np.clip(
        float(cfg.deposition_strength) * flatness * (0.66 + 0.34 * (1.0 - np.clip(q, 0.0, 1.0))),
        0.0,
        0.92,
    ) * lf

    load = (np.maximum(np.asarray(erosion_m, dtype=np.float64), 0.0).ravel() * area).astype(np.float64)
    source_total = float(np.sum(load))
    deposited = np.zeros(recv.size, dtype=np.float64)
    transit = np.zeros(recv.size, dtype=np.float64)
    exported = np.zeros(recv.size, dtype=np.float64)

    for node in graph.order:
        node = int(node)
        if not lf[node]:
            continue
        cur = load[node]
        if cur <= 0.0:
            continue
        transit[node] += cur
        local_dep = cur * depfrac[node]
        deposited[node] += local_dep
        remain = cur - local_dep
        nxt = int(recv[node])
        if nxt < 0:
            # Closed/internal terminal: sediment is conserved in the terminal basin.
            deposited[node] += remain
        elif lf[nxt]:
            load[nxt] += remain
        else:
            exported[nxt] += remain

    deposited_total = float(np.sum(deposited))
    exported_total = float(np.sum(exported))
    residual = source_total - deposited_total - exported_total
    global _LAST_SEDIMENT_LEDGER
    _LAST_SEDIMENT_LEDGER = {
        "eroded_volume_m_km2": source_total,
        "land_deposited_volume_m_km2": deposited_total,
        "shoreline_export_volume_m_km2": exported_total,
        "routing_residual_m_km2": residual,
        "relative_residual": residual / max(source_total, 1.0e-30),
        "algorithm": "single upstream-to-downstream O(N) conservative traversal",
    }

    dep_depth = deposited / np.maximum(area, 1.0e-12)
    return (
        dep_depth.reshape(shape),
        normalize01(np.log1p(transit.reshape(shape))),
        exported.reshape(shape),
    )


def _effective_surface_config(cfg: Any) -> Any:
    dt = float(getattr(cfg, "landscape_timestep_years", 0.0))
    if dt <= 0.0:
        return cfg
    out = copy.deepcopy(cfg)
    max_step = max(float(getattr(cfg, "max_geomorphic_step_m", 120.0)), 0.1)
    erosion_rate = max(float(getattr(cfg, "max_fluvial_erosion_rate_mm_year", 5.0)), 0.0)
    uplift_rate = max(float(getattr(cfg, "tectonic_uplift_rate_mm_year", 2.0)), 0.0)
    subsidence_rate = max(float(getattr(cfg, "rift_subsidence_rate_mm_year", 0.8)), 0.0)
    out.max_fluvial_erosion_m_per_iteration = min(max_step, erosion_rate * dt / 1000.0)
    out.tectonic_uplift_m_per_iteration = min(max_step, uplift_rate * dt / 1000.0)
    out.rift_subsidence_m_per_iteration = min(max_step, subsidence_rate * dt / 1000.0)
    # Diffusion is scaled sub-linearly with elapsed time to remain stable on the
    # explicit raster smoother while longer-term fluvial incision uses a hard step cap.
    out.hillslope_diffusion_strength = float(out.hillslope_diffusion_strength) * math.sqrt(max(dt, 1.0) / 25000.0)
    return out


def evolve_surface_advanced(*args, **kwargs):
    cfg = args[5] if len(args) > 5 else kwargs.get("cfg")
    effective = _effective_surface_config(cfg)
    if len(args) > 5:
        mutable = list(args); mutable[5] = effective; args = tuple(mutable)
    else:
        kwargs["cfg"] = effective
    result = _policy_evolve_surface(*args, **kwargs)
    dt = float(getattr(cfg, "landscape_timestep_years", 0.0))
    result.metadata = {
        **result.metadata,
        "landscape_timestep_years": dt,
        "dimensionful_landscape_time_enabled": bool(dt > 0.0),
        "effective_max_fluvial_erosion_m_per_iteration": float(effective.max_fluvial_erosion_m_per_iteration),
        "effective_tectonic_uplift_m_per_iteration": float(effective.tectonic_uplift_m_per_iteration),
        "sediment_mass_ledger": dict(_LAST_SEDIMENT_LEDGER),
    }
    return result


def _channel_hierarchy(
    grid: SphereGrid,
    base: Any,
    water: WaterBalanceResult,
    cfg: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    land = np.asarray(base.runoff > 0.0, dtype=bool)
    graph = DrainageGraph.from_receiver(base.flow_to, land.shape)
    cell_area = _base._cell_area_km2(grid)
    storm_multiplier = 1.0 + float(getattr(cfg, "bankfull_storm_multiplier", 3.0)) * np.asarray(water.storminess_index, float)
    bankfull_source = np.asarray(water.total_runoff_mm_year, float) * storm_multiplier / 1000.0 * cell_area
    bankfull = graph.accumulate(bankfull_source)
    bankfull_index = normalize01(np.log1p(bankfull)).astype(np.float32)

    drainage = np.asarray(base.drainage_area_km2, dtype=float)
    slope = _base._receiver_slope(base.filled_elevation_km, base.flow_to, grid)
    median_cell = float(np.median(cell_area[land])) if np.any(land) else float(np.median(cell_area))
    configured = float(getattr(cfg, "channel_min_catchment_km2", 0.0))
    initiation_area = max(configured, 0.58 * median_cell, 1.0)
    wet = np.clip(np.asarray(water.total_runoff_mm_year, float) / 420.0, 0.0, 3.0)
    baseflow = np.clip(np.asarray(water.baseflow_mm_year, float) / 120.0, 0.0, 2.0)
    slope_term = np.clip((np.maximum(slope, 1.0e-6) / 0.0012) ** 0.16, 0.55, 2.2)
    metric = (
        (np.maximum(drainage, 1.0) / initiation_area) ** 0.55
        * (0.30 + 0.52 * np.sqrt(wet) + 0.18 * baseflow)
        * slope_term
        * (0.82 + 0.35 * np.asarray(water.storminess_index, float))
    )
    channel = land & (drainage >= initiation_area) & (metric >= 1.0)

    cls = np.zeros(land.shape, dtype=np.uint8)
    cls[channel] = 1  # rill / ephemeral resolved channel
    cls[channel & ((drainage >= 3.0 * initiation_area) | (baseflow >= 0.20))] = 2
    cls[channel & (drainage >= 12.0 * initiation_area) & ((baseflow >= 0.35) | (wet >= 0.55))] = 3
    cls[channel & (drainage >= 65.0 * initiation_area)] = 4
    cls[channel & (drainage >= 320.0 * initiation_area)] = 5
    rivers = cls >= 2
    stream_order = graph.strahler_order(channel).astype(np.uint8)
    width = (bankfull_index * (0.55 + 0.45 * stream_order / max(float(stream_order.max()), 1.0)) * rivers).astype(np.float32)

    # Expected unresolved channel length per unit area.  This records the headwater
    # network that a 10-50 km global cell physically cannot rasterize explicitly.
    relief = np.clip(np.sqrt(np.maximum(slope, 0.0) / 0.002), 0.0, 2.0)
    density = np.clip(
        (0.10 + 1.55 * np.sqrt(np.clip(wet, 0.0, 1.8)))
        * (0.75 + 0.35 * relief)
        * (0.62 + 0.38 * (1.0 - np.asarray(water.storminess_index, float) * 0.25)),
        0.0,
        float(getattr(cfg, "max_subgrid_drainage_density_km_per_km2", 3.2)),
    ) * land
    return channel, cls, rivers, stream_order, width, bankfull_index, density


def _trim_centerlines_to_land(centerlines: list[dict], land: np.ndarray) -> list[dict]:
    h, w = land.shape
    out: list[dict] = []
    for item in centerlines:
        points = item.get("points_lat_lon", [])
        kept = []
        for lat, lon in points:
            yy = int(np.clip(round((90.0 - float(lat)) / 180.0 * h - 0.5), 0, h - 1))
            xx = int(round((float(lon) + 180.0) / 360.0 * w)) % w
            if not land[yy, xx] and len(kept) >= 2:
                break
            if land[yy, xx] or len(kept) < 2:
                kept.append([float(lat), float(lon)])
        if len(kept) >= 2:
            copy_item = dict(item)
            copy_item["points_lat_lon"] = kept
            copy_item["terminates_at_shoreline"] = True
            out.append(copy_item)
    return out


def build_hydrology_advanced(
    grid: SphereGrid,
    terrain: Any,
    ocean: Any,
    climate: Any,
    cfg: Any,
    geology: Any | None = None,
    surface: Any | None = None,
) -> AdvancedHydrologyResult:
    base = _base.build_hydrology(grid, terrain, ocean, climate, cfg, geology, surface)
    water = cached_water_balance(climate, terrain.land, geology, cfg)
    graph = DrainageGraph.from_receiver(base.flow_to, terrain.land.shape)
    slope = _base._receiver_slope(base.filled_elevation_km, base.flow_to, grid)
    channel, cls, rivers, stream_order, width, bankfull_index, density = _channel_hierarchy(grid, base, water, cfg)

    thresholds = tuple(getattr(cfg, "subbasin_thresholds_km2", (1.0e6, 1.0e5, 1.0e4)))
    if len(thresholds) != 3:
        thresholds = (1.0e6, 1.0e5, 1.0e4)
    watershed: WatershedHierarchy = build_watershed_hierarchy(
        grid,
        graph,
        terrain.land,
        ocean.elevation_km,
        base.drainage_area_km2,
        slope,
        channel,
        subbasin_thresholds_km2=tuple(map(float, thresholds)),
    )

    meander = np.asarray(base.meander_potential, float) * rivers
    centerlines = _base._build_river_centerlines(
        grid,
        base.flow_to,
        rivers,
        base.accumulation,
        meander,
        geology,
        max_paths=max(20, int(cfg.max_river_centerlines)),
    )
    centerlines = _trim_centerlines_to_land(centerlines, terrain.land)

    meta = {
        **base.metadata,
        "water_balance": water.metadata,
        "watersheds": watershed.metadata,
        "channel_model": "area-slope-runoff-bankfull threshold with explicit rill/intermittent/perennial/river/major-river hierarchy",
        "channel_class_codes": {
            "0": "hillslope/no resolved channel",
            "1": "rill or ephemeral resolved channel",
            "2": "intermittent stream",
            "3": "perennial stream or small river",
            "4": "river",
            "5": "major river",
        },
        "resolved_channel_area_fraction_land": grid.weighted_fraction(channel) / max(grid.weighted_fraction(terrain.land), 1.0e-12),
        "river_area_fraction_of_land": grid.weighted_fraction(rivers) / max(grid.weighted_fraction(terrain.land), 1.0e-12),
        "mean_subgrid_drainage_density_km_per_km2_land": float(np.average(density[terrain.land], weights=grid.cell_area_weights[terrain.land])) if np.any(terrain.land) else 0.0,
        "max_strahler_order_all_resolved_channels": int(stream_order.max()) if np.any(channel) else 0,
        "river_centerline_count": len(centerlines),
        "centerline_shoreline_policy": "river geometry terminates on land; estuary/delta/submarine continuation is a separate process class",
    }
    return AdvancedHydrologyResult(
        base=base,
        runoff=np.asarray(water.total_runoff_mm_year, np.float32),
        rivers=rivers,
        stream_order=stream_order,
        river_width_proxy=width,
        basin_id=watershed.basin_id,
        channel_class=cls,
        subbasin_level_1=watershed.subbasin_level_1,
        subbasin_level_2=watershed.subbasin_level_2,
        subbasin_level_3=watershed.subbasin_level_3,
        exorheic=watershed.exorheic,
        distance_to_outlet_km=watershed.distance_to_outlet_km,
        topographic_wetness_index=watershed.topographic_wetness_index,
        height_above_nearest_drainage_m=watershed.height_above_nearest_drainage_m,
        surface_runoff_mm_year=water.surface_runoff_mm_year,
        baseflow_mm_year=water.baseflow_mm_year,
        groundwater_recharge_mm_year=water.groundwater_recharge_mm_year,
        actual_evapotranspiration_mm_year=water.actual_evapotranspiration_mm_year,
        soil_water_storage_mm=water.soil_water_storage_mm,
        groundwater_storage_mm=water.groundwater_storage_mm,
        snowpack_mm=water.snowpack_mm,
        storminess_index=water.storminess_index,
        bankfull_discharge_index=bankfull_index,
        subgrid_drainage_density_km_per_km2=density.astype(np.float32),
        river_centerlines=centerlines,
        metadata=meta,
    )


__all__ = [
    "WaterBalanceResult",
    "AdvancedHydrologyResult",
    "build_water_balance",
    "cached_water_balance",
    "runoff_mm_advanced",
    "transport_sediment_topological",
    "evolve_surface_advanced",
    "build_hydrology_advanced",
]
