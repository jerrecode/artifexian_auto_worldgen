from __future__ import annotations

"""Global hydrology recomputation for each completed recursive refinement level.

Refinement may interpolate *forcing* such as runoff, baseflow and climate fields, but
it must never interpolate a receiver graph or watershed labels.  After all section
cores are composed into one seamless refined sphere this module rebuilds drainage on
that complete raster.  Consequently section boundaries cannot become artificial
watershed boundaries and new sub-grid relief can create/capture tributaries naturally.

The kernel is intentionally reduced to fields available in ``world_arrays.npz``.  It
does not rerun atmosphere, ocean or tectonics at every refinement level; those remain
inherited global forcings.  The globally coupled topographic derivatives are solved
from scratch at the new resolution.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from .drainage import DrainageGraph
from .grid import SphereGrid, normalize01
from . import hydrology_base as _base
from .priority_flood import priority_flood
from .watersheds import build_watershed_hierarchy


@dataclass(slots=True)
class RefinedHydrologyResult:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]


_DEFAULTS = {
    "channel_min_catchment_km2": 0.0,
    "bankfull_storm_multiplier": 3.0,
    "max_subgrid_drainage_density_km_per_km2": 3.2,
    "subbasin_thresholds_km2": (1.0e6, 1.0e5, 1.0e4),
    "lake_min_depth_m": 5.0,
    "lake_min_catchment_km2": 350.0,
    "lake_area_soft_cap_fraction_land": 0.022,
    "max_river_centerlines": 180,
}


DERIVED_HYDROLOGY_FIELDS = frozenset({
    "flow_to",
    "filled_elevation_km",
    "flow_accumulation",
    "drainage_area_km2",
    "discharge_index",
    "rivers",
    "stream_order",
    "river_width_proxy",
    "lakes",
    "basin_id",
    "channel_class",
    "subbasin_level_1",
    "subbasin_level_2",
    "subbasin_level_3",
    "exorheic",
    "distance_to_outlet_km",
    "topographic_wetness_index",
    "height_above_nearest_drainage_m",
    "bankfull_discharge_index",
    "subgrid_drainage_density_km_per_km2",
    "meander_potential",
    "river_sinuosity_proxy",
})


def load_refinement_hydrology_context(world_root: str | Path) -> tuple[float, SimpleNamespace]:
    """Read radius and hydrology controls from the original world's JSON metadata."""
    root = Path(world_root)
    radius_km = 6371.0088
    values: dict[str, Any] = dict(_DEFAULTS)
    path = root / "world.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        astronomy = payload.get("astronomy", {}) if isinstance(payload, dict) else {}
        planet = astronomy.get("planet", {}) if isinstance(astronomy, dict) else {}
        try:
            radius_km = 6371.0 * float(planet.get("radius_earth", 1.0))
        except (TypeError, ValueError):
            radius_km = 6371.0088
        config = payload.get("config", {}) if isinstance(payload, dict) else {}
        hydrology = config.get("hydrology", {}) if isinstance(config, dict) else {}
        if isinstance(hydrology, dict):
            for key in values:
                if key in hydrology:
                    values[key] = hydrology[key]
    if not math.isfinite(radius_km) or radius_km <= 0.0:
        radius_km = 6371.0088
    thresholds = values.get("subbasin_thresholds_km2", _DEFAULTS["subbasin_thresholds_km2"])
    if not isinstance(thresholds, (list, tuple)) or len(thresholds) != 3:
        thresholds = _DEFAULTS["subbasin_thresholds_km2"]
    values["subbasin_thresholds_km2"] = tuple(float(x) for x in thresholds)
    return float(radius_km), SimpleNamespace(**values)


def _existing(arrays: Mapping[str, np.ndarray], name: str, shape: tuple[int, int], default: float = 0.0) -> np.ndarray:
    value = arrays.get(name)
    if value is None:
        return np.full(shape, default, dtype=np.float64)
    out = np.asarray(value, dtype=np.float64)
    if out.shape != shape:
        raise ValueError(f"refinement hydrology forcing {name!r} has shape {out.shape}, expected {shape}")
    return out


def _lake_mask_refined(
    grid: SphereGrid,
    elevation: np.ndarray,
    filled: np.ndarray,
    land: np.ndarray,
    drainage_area: np.ndarray,
    runoff: np.ndarray,
    cfg: Any,
) -> np.ndarray:
    """Retain physically substantial filled depressions without climate re-solving."""
    depth_m = np.maximum((filled - elevation) * 1000.0, 0.0)
    candidate = (
        land
        & (depth_m >= float(cfg.lake_min_depth_m))
        & (drainage_area >= float(cfg.lake_min_catchment_km2))
        & (runoff > 1.0)
    )
    labels, count = grid.ops.connected_components(candidate)
    if count <= 0:
        return np.zeros_like(land)
    ids = np.arange(1, count + 1, dtype=np.int32)
    area = _base._cell_area_km2(grid)
    component_area = np.asarray(ndimage.sum(area, labels=labels, index=ids), dtype=float)
    max_depth = np.asarray(ndimage.maximum(depth_m, labels=labels, index=ids), dtype=float)
    max_drain = np.asarray(ndimage.maximum(drainage_area, labels=labels, index=ids), dtype=float)
    score = max_depth * np.log1p(np.maximum(max_drain, 0.0))
    order = np.argsort(score)[::-1]
    land_area = float(np.sum(area[land]))
    soft_cap = land_area * float(np.clip(cfg.lake_area_soft_cap_fraction_land, 0.001, 0.20))
    keep = np.zeros(count + 1, dtype=bool)
    used = 0.0
    for index in order:
        a = float(component_area[index])
        if a <= 0.0:
            continue
        if used > 0.0 and used + a > soft_cap:
            continue
        keep[index + 1] = True
        used += a
    return keep[labels]


def _channel_hierarchy_refined(
    grid: SphereGrid,
    graph: DrainageGraph,
    drainage: np.ndarray,
    runoff: np.ndarray,
    baseflow: np.ndarray,
    storminess: np.ndarray,
    slope: np.ndarray,
    cfg: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    land = runoff >= 0.0
    # Explicit land is applied by the caller after the metric is formed.  Keeping the
    # metric continuous here makes a dry headwater cell eligible if its accumulated
    # upstream drainage is sufficiently large.
    cell_area = _base._cell_area_km2(grid)
    median_cell = float(np.median(cell_area))
    configured = float(getattr(cfg, "channel_min_catchment_km2", 0.0))
    initiation_area = max(configured, 0.58 * median_cell, 0.25)
    wet = np.clip(runoff / 420.0, 0.0, 3.0)
    base = np.clip(baseflow / 120.0, 0.0, 2.0)
    slope_term = np.clip((np.maximum(slope, 1.0e-7) / 0.0012) ** 0.16, 0.55, 2.2)
    metric = (
        (np.maximum(drainage, 1.0) / initiation_area) ** 0.55
        * (0.30 + 0.52 * np.sqrt(wet) + 0.18 * base)
        * slope_term
        * (0.82 + 0.35 * storminess)
    )
    channel = (drainage >= initiation_area) & (metric >= 1.0)
    cls = np.zeros(drainage.shape, dtype=np.uint8)
    cls[channel] = 1
    cls[channel & ((drainage >= 3.0 * initiation_area) | (base >= 0.20))] = 2
    cls[channel & (drainage >= 12.0 * initiation_area) & ((base >= 0.35) | (wet >= 0.55))] = 3
    cls[channel & (drainage >= 65.0 * initiation_area)] = 4
    cls[channel & (drainage >= 320.0 * initiation_area)] = 5
    rivers = cls >= 2
    stream_order = graph.strahler_order(channel).astype(np.uint8)

    cell_source = runoff / 1000.0 * cell_area
    mean_q = graph.accumulate(cell_source)
    storm_source = runoff * (1.0 + float(cfg.bankfull_storm_multiplier) * storminess) / 1000.0 * cell_area
    bankfull = graph.accumulate(storm_source)
    discharge_index = normalize01(np.log1p(mean_q)).astype(np.float32)
    bankfull_index = normalize01(np.log1p(bankfull)).astype(np.float32)
    order_norm = stream_order.astype(float) / max(float(stream_order.max()), 1.0)
    width = (bankfull_index * (0.55 + 0.45 * order_norm) * rivers).astype(np.float32)

    relief = np.clip(np.sqrt(np.maximum(slope, 0.0) / 0.002), 0.0, 2.0)
    density = np.clip(
        (0.10 + 1.55 * np.sqrt(np.clip(wet, 0.0, 1.8)))
        * (0.75 + 0.35 * relief)
        * (0.62 + 0.38 * (1.0 - storminess * 0.25)),
        0.0,
        float(cfg.max_subgrid_drainage_density_km_per_km2),
    )
    return channel, cls, rivers, stream_order, width, discharge_index, bankfull_index, density, mean_q


def recompute_refined_hydrology(
    elevation_km: np.ndarray,
    forcing_arrays: Mapping[str, np.ndarray],
    *,
    radius_km: float = 6371.0088,
    cfg: Any | None = None,
) -> RefinedHydrologyResult:
    """Rebuild globally coupled drainage topology on one complete refined sphere."""
    elevation = np.asarray(elevation_km, dtype=np.float64)
    if elevation.ndim != 2:
        raise ValueError("refined elevation must be a 2-D global raster")
    h, w = elevation.shape
    if w != 2 * h:
        raise ValueError("refined hydrology requires the canonical 2:1 global raster")
    if cfg is None:
        cfg = SimpleNamespace(**_DEFAULTS)
    grid = SphereGrid(w, h, float(radius_km))
    land = elevation > 0.0
    ocean = ~land

    filled = priority_flood(elevation, ocean, grid)
    flow = _base._flow_directions(filled, ocean, grid)
    graph = DrainageGraph.from_receiver(flow, elevation.shape)
    cell_area = _base._cell_area_km2(grid)

    runoff = np.maximum(_existing(forcing_arrays, "runoff_mm_year", elevation.shape, 0.0), 0.0) * land
    # If runoff was unavailable in an old dataset, annual precipitation is a safer
    # forcing proxy than inventing a spatially uniform discharge field.
    if not np.any(runoff > 0.0) and "annual_precipitation_mm" in forcing_arrays:
        runoff = 0.42 * np.maximum(
            _existing(forcing_arrays, "annual_precipitation_mm", elevation.shape, 0.0), 0.0
        ) * land
    baseflow = np.maximum(_existing(forcing_arrays, "baseflow_mm_year", elevation.shape, 0.0), 0.0) * land
    storminess = np.clip(_existing(forcing_arrays, "storminess_index", elevation.shape, 0.35), 0.0, 1.0) * land

    drainage = graph.accumulate(cell_area * land)
    slope = _base._receiver_slope(filled, flow, grid)
    channel, cls, rivers, stream_order, width, discharge_index, bankfull_index, density, accumulation = _channel_hierarchy_refined(
        grid, graph, drainage, runoff, baseflow, storminess, slope, cfg
    )
    channel &= land
    cls = np.where(land, cls, 0).astype(np.uint8)
    rivers &= land
    stream_order = np.where(land, stream_order, 0).astype(np.uint8)
    width *= land
    density *= land

    watershed = build_watershed_hierarchy(
        grid,
        graph,
        land,
        elevation,
        drainage,
        slope,
        channel,
        subbasin_thresholds_km2=tuple(cfg.subbasin_thresholds_km2),
    )
    lakes = _lake_mask_refined(grid, elevation, filled, land, drainage, runoff, cfg)

    qnorm = np.clip(discharge_index.astype(float), 0.0, 1.0)
    lowgrad = np.exp(-slope / 0.0019)
    meander = np.clip((qnorm ** 0.62) * lowgrad * land, 0.0, 1.0).astype(np.float32)
    sinuosity = (1.0 + 2.35 * meander).astype(np.float32)

    arrays: dict[str, np.ndarray] = {
        "flow_to": np.asarray(flow, dtype=np.int32),
        "filled_elevation_km": filled.astype(np.float32),
        "flow_accumulation": np.asarray(accumulation, dtype=np.float32),
        "drainage_area_km2": np.asarray(drainage, dtype=np.float32),
        "discharge_index": discharge_index,
        "rivers": rivers.astype(bool),
        "stream_order": stream_order,
        "river_width_proxy": width,
        "lakes": lakes.astype(bool),
        "basin_id": watershed.basin_id,
        "channel_class": cls,
        "subbasin_level_1": watershed.subbasin_level_1,
        "subbasin_level_2": watershed.subbasin_level_2,
        "subbasin_level_3": watershed.subbasin_level_3,
        "exorheic": watershed.exorheic.astype(bool),
        "distance_to_outlet_km": watershed.distance_to_outlet_km,
        "topographic_wetness_index": watershed.topographic_wetness_index,
        "height_above_nearest_drainage_m": watershed.height_above_nearest_drainage_m,
        "bankfull_discharge_index": bankfull_index,
        "subgrid_drainage_density_km_per_km2": density.astype(np.float32),
        "meander_potential": meander,
        "river_sinuosity_proxy": sinuosity,
    }
    unassigned = int(np.count_nonzero(land & (watershed.basin_id <= 0)))
    metadata = {
        "algorithm": "full-sphere Priority-Flood + refined receiver graph + O(N) accumulation + outlet watershed hierarchy",
        "resolution": [w, h],
        "radius_km": float(radius_km),
        "land_fraction": float(grid.weighted_fraction(land)),
        "resolved_river_fraction_land": float(
            grid.weighted_fraction(rivers) / max(grid.weighted_fraction(land), 1.0e-30)
        ),
        "terminal_watershed_count": int(watershed.metadata.get("outlet_basin_count", 0)),
        "max_strahler_order": int(stream_order.max()) if np.any(channel) else 0,
        "all_land_assigned": unassigned == 0,
        "unassigned_land_cells": unassigned,
        "forcing": "interpolated globally coupled runoff/baseflow/storminess; topology recomputed from refined elevation",
        "tile_boundary_policy": "none: solve runs once after complete level composition, never independently per section",
    }
    return RefinedHydrologyResult(arrays=arrays, metadata=metadata)


__all__ = [
    "DERIVED_HYDROLOGY_FIELDS",
    "RefinedHydrologyResult",
    "load_refinement_hydrology_context",
    "recompute_refined_hydrology",
]
