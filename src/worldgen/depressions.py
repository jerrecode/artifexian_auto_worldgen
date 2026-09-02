from __future__ import annotations

"""Depression hierarchy and endorheic-storage diagnostics.

Priority-Flood remains the robust routing surface, but its fill depth contains useful
physical information that must not be discarded.  This module turns that information
into explicit depression storage, spill-capacity and climate-balance state so real
closed basins can be distinguished from numerical pits.
"""

from dataclasses import dataclass
import math
import numpy as np
from scipy import ndimage

from .grid import SphereGrid


@dataclass(slots=True)
class DepressionResult:
    depression_id: np.ndarray
    depression_depth_m: np.ndarray
    depression_storage_capacity_m3: np.ndarray
    endorheic_depression: np.ndarray
    seasonally_inundated: np.ndarray
    records: list[dict]
    metadata: dict

    def to_dict(self) -> dict:
        return {"records": self.records, "metadata": self.metadata}


def build_depressions(
    grid: SphereGrid,
    terrain,
    climate,
    hydrology,
    *,
    minimum_depth_m: float = 1.0,
    minimum_area_km2: float = 5.0,
) -> DepressionResult:
    z = np.asarray(terrain.elevation_km, dtype=np.float64)
    filled = np.asarray(hydrology.filled_elevation_km, dtype=np.float64)
    land = np.asarray(terrain.land, dtype=bool)
    depth = np.maximum((filled - z) * 1000.0, 0.0) * land
    candidate = land & (depth >= float(minimum_depth_m))
    labels, n = grid.ops.connected_components(candidate)
    if n == 0:
        zero_i = np.zeros(grid.shape, np.int32)
        zero_f = np.zeros(grid.shape, np.float32)
        zero_b = np.zeros(grid.shape, bool)
        return DepressionResult(zero_i, zero_f, zero_f, zero_b, zero_b, [], {
            "depression_count": 0,
            "model": "Priority-Flood storage hierarchy with climatic endorheic screening",
        })

    ids = np.arange(1, n + 1, dtype=np.int32)
    area_km2 = np.asarray(grid.cell_area_weights, dtype=np.float64) * (4.0 * math.pi * grid.radius_km**2)
    areas = np.asarray(ndimage.sum(area_km2, labels=labels, index=ids), float)
    capacity_m3 = np.asarray(ndimage.sum(depth * area_km2 * 1.0e6, labels=labels, index=ids), float)
    max_depth = np.asarray(ndimage.maximum(depth, labels=labels, index=ids), float)
    mean_depth = np.asarray(ndimage.mean(depth, labels=labels, index=ids), float)
    max_drainage = np.asarray(ndimage.maximum(np.asarray(hydrology.drainage_area_km2, float), labels=labels, index=ids), float)

    precip = np.asarray(climate.annual_precipitation_mm, dtype=float)
    temp = np.asarray(climate.annual_temperature_c, dtype=float)
    # Reduced-order PET used only for closed-basin screening.  The conservative water
    # balance provides actual ET on advanced hydrology objects when available.
    if hasattr(hydrology, "actual_evapotranspiration_mm_year"):
        aet = np.asarray(hydrology.actual_evapotranspiration_mm_year, float)
        pet_proxy = np.maximum(aet, 22.0 * np.maximum(temp + 5.0, 0.0))
    else:
        pet_proxy = np.maximum(0.0, 24.0 * np.maximum(temp + 5.0, 0.0))
    p_mean = np.asarray(ndimage.mean(precip, labels=labels, index=ids), float)
    pet_mean = np.asarray(ndimage.mean(pet_proxy, labels=labels, index=ids), float)
    runoff_mean = np.asarray(ndimage.mean(np.asarray(hydrology.runoff, float), labels=labels, index=ids), float)

    valid = areas >= float(minimum_area_km2)
    # Arid basins with meaningful storage are allowed to remain internally drained.
    # Humid depressions are more likely to fill and spill through their Priority-Flood
    # sill; seasonal basins occupy the transition.
    aridity = pet_mean / np.maximum(p_mean, 1.0)
    closed = valid & (aridity >= 1.15) & (capacity_m3 > 0.0)
    seasonal = valid & ~closed & (aridity >= 0.78) & (aridity < 1.15)

    keep = np.zeros(n + 1, dtype=bool)
    keep[ids[valid]] = True
    clean_labels = np.where(keep[labels], labels, 0).astype(np.int32)
    closed_lookup = np.zeros(n + 1, dtype=bool); closed_lookup[ids[closed]] = True
    seasonal_lookup = np.zeros(n + 1, dtype=bool); seasonal_lookup[ids[seasonal]] = True
    closed_map = closed_lookup[clean_labels]
    seasonal_map = seasonal_lookup[clean_labels]

    capacity_raster = np.zeros(grid.shape, dtype=np.float32)
    if np.any(valid):
        cap_lookup = np.zeros(n + 1, dtype=np.float64)
        cap_lookup[ids[valid]] = capacity_m3[valid]
        capacity_raster = cap_lookup[clean_labels].astype(np.float32)

    records: list[dict] = []
    valid_indices = np.flatnonzero(valid)
    order = valid_indices[np.argsort(capacity_m3[valid_indices])[::-1]]
    for j in order[:10000]:
        records.append({
            "depression_id": int(ids[j]),
            "area_km2": float(areas[j]),
            "storage_capacity_m3": float(capacity_m3[j]),
            "max_depth_m": float(max_depth[j]),
            "mean_depth_m": float(mean_depth[j]),
            "upstream_drainage_area_km2": float(max_drainage[j]),
            "mean_precipitation_mm_year": float(p_mean[j]),
            "mean_pet_proxy_mm_year": float(pet_mean[j]),
            "mean_runoff_mm_year": float(runoff_mean[j]),
            "aridity_index": float(aridity[j]),
            "endorheic": bool(closed[j]),
            "seasonally_inundated": bool(seasonal[j]),
        })

    metadata = {
        "depression_count": int(np.sum(valid)),
        "endorheic_depression_count": int(np.sum(closed)),
        "seasonal_depression_count": int(np.sum(seasonal)),
        "total_storage_capacity_m3": float(np.sum(capacity_m3[valid])),
        "endorheic_storage_capacity_m3": float(np.sum(capacity_m3[closed])),
        "minimum_depth_m": float(minimum_depth_m),
        "minimum_area_km2": float(minimum_area_km2),
        "model": "Priority-Flood depression geometry + climate water-deficit screening",
        "limitations": "spill timing and dynamic lake level are reduced-order; local groundwater head and explicit lake evaporation remain future high-resolution solves",
    }
    return DepressionResult(
        clean_labels,
        depth.astype(np.float32),
        capacity_raster,
        closed_map,
        seasonal_map,
        records,
        metadata,
    )


__all__ = ["DepressionResult", "build_depressions"]
