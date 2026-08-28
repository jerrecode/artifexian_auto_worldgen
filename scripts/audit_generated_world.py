#!/usr/bin/env python3
from __future__ import annotations

"""Numerically audit a generated world directory without loading the pipeline state.

This intentionally operates on public saved artifacts.  It is suitable for CI
showcases where the generation process has already exited and therefore catches
serialization/schema inconsistencies as well as physical/numerical problems.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _corr(a: np.ndarray | None, b: np.ndarray | None, mask: np.ndarray) -> float | None:
    if a is None or b is None:
        return None
    x = np.asarray(a, dtype=np.float64)[mask]
    y = np.asarray(b, dtype=np.float64)[mask]
    good = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(good)) < 30 or np.std(x[good]) <= 1.0e-15 or np.std(y[good]) <= 1.0e-15:
        return None
    return float(np.corrcoef(x[good], y[good])[0, 1])


def _field(data: Any, name: str, dtype=None):
    if name not in data.files:
        return None
    out = np.asarray(data[name])
    return out.astype(dtype, copy=False) if dtype is not None else out


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _weighted_mean(a: np.ndarray, mask: np.ndarray, weight: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.average(np.asarray(a, float)[mask], weights=weight[mask]))


def audit(root: Path) -> dict[str, Any]:
    arrays_path = root / "world_arrays.npz"
    if not arrays_path.exists():
        raise FileNotFoundError(arrays_path)
    data = np.load(arrays_path, allow_pickle=False)
    elevation = _field(data, "elevation_km", np.float64)
    if elevation is None or elevation.ndim != 2:
        raise ValueError("world_arrays.npz must contain 2-D elevation_km")
    h, w = elevation.shape
    if w != 2 * h:
        raise ValueError(f"expected canonical 2:1 raster, got {w}x{h}")

    world = _json(root / "world.json")
    diagnostics = _json(root / "diagnostics.json")
    radius_km = 6371.0088
    try:
        radius_km = 6371.0 * float(world["astronomy"]["planet"]["radius_earth"])
    except (KeyError, TypeError, ValueError):
        pass

    lat = np.pi / 2.0 - (np.arange(h, dtype=np.float64) + 0.5) * np.pi / h
    weight = np.broadcast_to(np.cos(lat)[:, None], (h, w)).copy()
    weight /= float(np.sum(weight))
    surface_area_km2 = 4.0 * math.pi * radius_km**2
    cell_area_km2 = weight * surface_area_km2
    land = elevation > 0.0
    ocean = ~land

    runoff = _field(data, "runoff_mm_year", np.float64)
    precip = _field(data, "annual_precipitation_mm", np.float64)
    erosion = _field(data, "cumulative_erosion_m", np.float64)
    deposition = _field(data, "cumulative_deposition_m", np.float64)
    discharge = _field(data, "discharge_index", np.float64)
    drainage = _field(data, "drainage_area_km2", np.float64)
    rivers = _field(data, "rivers")
    stream_order = _field(data, "stream_order")
    basin = _field(data, "basin_id")
    channel_class = _field(data, "channel_class")
    subgrid_density = _field(data, "subgrid_drainage_density_km_per_km2", np.float64)
    baseflow = _field(data, "baseflow_mm_year", np.float64)
    bankfull = _field(data, "bankfull_discharge_index", np.float64)
    depression_id = _field(data, "depression_id")
    endorheic_depression = _field(data, "endorheic_depression")
    tidal_range = _field(data, "tidal_range_m", np.float64)

    river = np.asarray(rivers, bool) if rivers is not None else np.zeros_like(land)
    river_land = river & land

    dlat = np.pi / h
    dlon = 2.0 * np.pi / w
    z_m = elevation * 1000.0
    dzlat = np.gradient(z_m, dlat, axis=0, edge_order=1)
    dzlon = (np.roll(z_m, -1, axis=1) - np.roll(z_m, 1, axis=1)) / (2.0 * dlon)
    slope = np.hypot(
        dzlat / (radius_km * 1000.0),
        dzlon / (radius_km * 1000.0 * np.maximum(np.cos(lat)[:, None], 1.0e-5)),
    )

    land_elev_m = z_m[land]
    p1, p99 = np.percentile(land_elev_m, [1, 99]) if land_elev_m.size else (0.0, 0.0)
    relief_m = max(float(p99 - p1), 1.0e-12)
    mean_erosion = _weighted_mean(erosion, land, weight) if erosion is not None else 0.0

    basin_metrics: dict[str, Any] = {}
    if basin is not None and basin.shape == land.shape:
        labels = np.asarray(basin, dtype=np.int64)[land]
        labels = labels[labels > 0]
        if labels.size:
            ids, counts = np.unique(labels, return_counts=True)
            max_id = int(ids.max())
            areas = np.bincount(
                np.asarray(basin, dtype=np.int64)[land],
                weights=cell_area_km2[land],
                minlength=max_id + 1,
            )[ids]
            basin_metrics = {
                "terminal_watershed_count": int(ids.size),
                "unassigned_land_cells": int(np.count_nonzero(land & (np.asarray(basin) <= 0))),
                "largest_basin_km2": float(np.max(areas)),
                "median_basin_km2": float(np.median(areas)),
                "p90_basin_km2": float(np.percentile(areas, 90)),
                "largest_basin_fraction_land": float(np.max(areas) / max(np.sum(cell_area_km2[land]), 1.0e-30)),
                "largest_basin_cells": int(np.max(counts)),
            }

    resolved_length_km = float(np.sum(np.sqrt(cell_area_km2[river_land])))
    resolved_density = resolved_length_km / max(float(np.sum(cell_area_km2[land])), 1.0e-30)
    subgrid_mean = _weighted_mean(subgrid_density, land, weight) if subgrid_density is not None else 0.0

    metadata = world.get("metadata", {}) if isinstance(world, dict) else {}
    hydrology_meta = metadata.get("hydrology", {}) if isinstance(metadata, dict) else {}
    surface_meta = metadata.get("surface_evolution", {}) if isinstance(metadata, dict) else {}
    condensate_meta = world.get("condensate_hydrology", {}) if isinstance(world, dict) else {}
    coupling = world.get("coupling_summary", {}) if isinstance(world, dict) else {}

    audit_result: dict[str, Any] = {
        "schema_version": 1,
        "grid": {
            "width": w,
            "height": h,
            "radius_km": radius_km,
            "surface_area_km2": surface_area_km2,
            "land_area_fraction": float(np.sum(weight[land])),
            "ocean_area_fraction": float(np.sum(weight[ocean])),
        },
        "topography": {
            "min_elevation_m": float(np.min(z_m)),
            "max_elevation_m": float(np.max(z_m)),
            "land_p1_elevation_m": float(p1),
            "land_p99_elevation_m": float(p99),
            "land_p1_p99_relief_m": relief_m,
            "land_median_slope_gradient": float(np.median(slope[land])) if np.any(land) else 0.0,
            "land_p90_slope_gradient": float(np.percentile(slope[land], 90)) if np.any(land) else 0.0,
            "land_p99_slope_gradient": float(np.percentile(slope[land], 99)) if np.any(land) else 0.0,
        },
        "hydrology": {
            "river_cells": int(np.count_nonzero(river)),
            "river_cells_on_ocean": int(np.count_nonzero(river & ocean)),
            "river_area_fraction_land": float(np.sum(weight[river_land]) / max(np.sum(weight[land]), 1.0e-30)),
            "max_strahler_order": int(np.max(stream_order)) if stream_order is not None and stream_order.size else 0,
            "resolved_river_length_proxy_km": resolved_length_km,
            "resolved_drainage_density_km_per_km2": resolved_density,
            "mean_subgrid_drainage_density_km_per_km2": subgrid_mean,
            "mean_land_precipitation_mm_year": _weighted_mean(precip, land, weight) if precip is not None else None,
            "mean_land_runoff_mm_year": _weighted_mean(runoff, land, weight) if runoff is not None else None,
            "mean_land_baseflow_mm_year": _weighted_mean(baseflow, land, weight) if baseflow is not None else None,
            "mean_land_bankfull_index": _weighted_mean(bankfull, land, weight) if bankfull is not None else None,
            **basin_metrics,
        },
        "landscape": {
            "mean_land_cumulative_erosion_m": mean_erosion,
            "median_land_cumulative_erosion_m": float(np.median(erosion[land])) if erosion is not None and np.any(land) else None,
            "p90_land_cumulative_erosion_m": float(np.percentile(erosion[land], 90)) if erosion is not None and np.any(land) else None,
            "p99_land_cumulative_erosion_m": float(np.percentile(erosion[land], 99)) if erosion is not None and np.any(land) else None,
            "max_land_cumulative_erosion_m": float(np.max(erosion[land])) if erosion is not None and np.any(land) else None,
            "mean_erosion_to_relief_ratio": mean_erosion / relief_m,
            "mean_land_cumulative_deposition_m": _weighted_mean(deposition, land, weight) if deposition is not None else None,
            "precipitation_erosion_correlation": _corr(precip, erosion, land),
            "runoff_erosion_correlation": _corr(runoff, erosion, land),
            "discharge_erosion_correlation": _corr(discharge, erosion, land),
            "drainage_area_erosion_correlation": _corr(drainage, erosion, land),
        },
        "processes": {
            "resolved_channel_class_max": int(np.max(channel_class)) if channel_class is not None and channel_class.size else 0,
            "depression_count": int(np.max(depression_id)) if depression_id is not None and depression_id.size else 0,
            "endorheic_depression_cells": int(np.count_nonzero(endorheic_depression)) if endorheic_depression is not None else 0,
            "max_tidal_range_m": float(np.max(tidal_range)) if tidal_range is not None and tidal_range.size else 0.0,
        },
        "conservation": {
            "water_balance_max_abs_residual_mm_year": hydrology_meta.get("water_balance", {}).get("max_absolute_water_balance_residual_mm_year") if isinstance(hydrology_meta, dict) else None,
            "sediment_routing_relative_residual": surface_meta.get("sediment_mass_ledger", {}).get("relative_residual") if isinstance(surface_meta, dict) else None,
            "condensate_mass_partition_relative_l1_residual": condensate_meta.get("mass_conservation_relative_l1_residual") if isinstance(condensate_meta, dict) else None,
        },
        "multicondensate": {
            "reference_species": condensate_meta.get("reference_species") if isinstance(condensate_meta, dict) else None,
            "active_hydrologic_species": condensate_meta.get("active_hydrologic_species", []) if isinstance(condensate_meta, dict) else [],
            "enabled": bool(condensate_meta),
        },
        "coupling_summary": coupling,
        "saved_diagnostics": {
            "all_invariants_passed": diagnostics.get("all_invariants_passed"),
            "invariant_failures": [
                item for item in diagnostics.get("invariants", [])
                if isinstance(item, dict) and not item.get("passed", False)
            ],
        },
    }
    return audit_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("world_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit(args.world_dir)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
