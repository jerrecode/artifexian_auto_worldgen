from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(slots=True, frozen=True)
class InvariantResult:
    name: str
    passed: bool
    value: float | int | str | None = None
    tolerance: float | None = None
    detail: str = ""


def array_digest(array: np.ndarray, *, digest_size: int = 16) -> str:
    a = np.asarray(array)
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(a.dtype.str.encode("ascii"))
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()


def receiver_graph_is_acyclic(flow_to: np.ndarray) -> bool:
    recv = np.asarray(flow_to, dtype=np.int64).ravel()
    n = recv.size
    indeg = np.zeros(n, dtype=np.int32)
    src = np.flatnonzero(recv >= 0)
    if src.size:
        tgt = recv[src]
        valid = (tgt >= 0) & (tgt < n)
        if not np.all(valid):
            return False
        np.add.at(indeg, tgt, 1)
    queue = list(map(int, np.flatnonzero(indeg == 0)))
    head = 0
    visited = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        visited += 1
        nxt = int(recv[cur])
        if nxt >= 0:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    return visited == n


def _finite_fraction(a: np.ndarray) -> float:
    arr = np.asarray(a)
    if arr.size == 0 or arr.dtype.kind not in "fc":
        return 1.0
    return float(np.mean(np.isfinite(arr)))


def _safe_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
    x = np.asarray(a, float)[mask]
    y = np.asarray(b, float)[mask]
    good = np.isfinite(x) & np.isfinite(y)
    if np.sum(good) < 20 or np.std(x[good]) <= 1e-12 or np.std(y[good]) <= 1e-12:
        return None
    return float(np.corrcoef(x[good], y[good])[0, 1])


def world_diagnostics(world: Mapping[str, Any]) -> dict[str, Any]:
    """Compute numerical, topological and conservation-oriented world diagnostics."""
    grid = world["grid"]
    terrain = world["terrain"]
    climate = world["climate"]
    ocean = world["ocean"]
    hydro = world["hydrology"]
    surface = world.get("surface_evolution")
    appearance = world.get("appearance")

    arrays = {
        "elevation_km": terrain.elevation_km,
        "annual_temperature_c": climate.annual_temperature_c,
        "annual_precipitation_mm": climate.annual_precipitation_mm,
        "runoff_mm_year": hydro.runoff,
        "ocean_current_u": ocean.current_u,
        "ocean_current_v": ocean.current_v,
    }
    for name in (
        "basin_id", "channel_class", "baseflow_mm_year", "groundwater_recharge_mm_year",
        "bankfull_discharge_index", "subgrid_drainage_density_km_per_km2",
    ):
        if hasattr(hydro, name):
            arrays[name] = getattr(hydro, name)
    finite = {name: _finite_fraction(value) for name, value in arrays.items()}

    invariants: list[InvariantResult] = [
        InvariantResult(f"finite:{name}", frac == 1.0, frac, 1.0)
        for name, frac in finite.items()
    ]
    invariants.append(InvariantResult(
        "hydrology:receiver_graph_acyclic", receiver_graph_is_acyclic(hydro.flow_to),
        detail="filled single-receiver drainage graph must be acyclic",
    ))
    runoff_min = float(np.nanmin(hydro.runoff)) if np.size(hydro.runoff) else 0.0
    invariants.append(InvariantResult("hydrology:nonnegative_runoff", runoff_min >= -1e-8, runoff_min, 1e-8))
    precip_min = float(np.nanmin(climate.precipitation_mm)) if np.size(climate.precipitation_mm) else 0.0
    invariants.append(InvariantResult("climate:nonnegative_precipitation", precip_min >= -1e-8, precip_min, 1e-8))

    land_fraction = float(grid.weighted_fraction(terrain.land))
    liquids = world.get("surface_liquids")
    if liquids is None:
        land_fraction_ok = 0.0 < land_fraction < 1.0
        land_fraction_detail = (
            "legacy target-land-fraction worlds must retain both emergent land and ocean"
        )
    else:
        liquid_volume_m3 = max(float(getattr(liquids, "total_liquid_volume_m3", 0.0)), 0.0)
        liquid_mask = np.asarray(
            getattr(liquids, "liquid_mask", np.zeros(grid.shape, dtype=bool)),
            dtype=bool,
        )
        wet_fraction = float(grid.weighted_fraction(liquid_mask))
        terrain_ocean = np.asarray(terrain.ocean, dtype=bool)
        mask_consistent = bool(np.array_equal(liquid_mask, terrain_ocean))
        invariants.append(InvariantResult(
            "surface_liquid:wet_mask_matches_terrain_ocean",
            mask_consistent,
            int(np.count_nonzero(liquid_mask ^ terrain_ocean)),
            0.0,
            "the conserved mobile-liquid mask must be the canonical terrain ocean mask",
        ))
        if liquid_volume_m3 > 0.0:
            land_fraction_ok = bool(np.any(liquid_mask)) and land_fraction < 1.0
            land_fraction_detail = (
                "positive mobile-liquid volume requires at least one wet cell; fully oceanic "
                "worlds are valid when the conserved fill leaves no emergent land"
            )
        else:
            land_fraction_ok = (not np.any(liquid_mask)) and abs(land_fraction - 1.0) <= 1.0e-12
            land_fraction_detail = (
                "zero mobile-liquid volume is valid only when no wet cells remain; configured "
                "volatile mass may be entirely vapor or thermodynamically sequestered as solid"
            )
    invariants.append(InvariantResult(
        "terrain:land_fraction_valid", land_fraction_ok, land_fraction,
        detail=land_fraction_detail,
    ))
    reported_land = terrain.metadata.get("actual_land_fraction")
    if reported_land is not None:
        land_error = abs(float(reported_land) - land_fraction)
        invariants.append(InvariantResult(
            "terrain:land_fraction_metadata_reconciled", land_error <= 1e-6, land_error, 1e-6,
            "derived metadata must describe the final canonical shoreline raster",
        ))

    river_ocean_count = int(np.count_nonzero(np.asarray(hydro.rivers, bool) & np.asarray(terrain.ocean, bool)))
    invariants.append(InvariantResult(
        "hydrology:rivers_on_land", river_ocean_count == 0, river_ocean_count, 0.0,
        "river channel class terminates at shoreline; estuary/submarine processes are separate",
    ))
    if hasattr(hydro, "basin_id"):
        basin = np.asarray(hydro.basin_id)
        unassigned = int(np.count_nonzero(np.asarray(terrain.land, bool) & (basin <= 0)))
        invariants.append(InvariantResult(
            "hydrology:all_land_assigned_to_terminal_basin", unassigned == 0, unassigned, 0.0,
            "every routed land cell must belong to an exorheic or internal terminal catchment",
        ))

    water_meta = hydro.metadata.get("water_balance", {}) if isinstance(hydro.metadata, dict) else {}
    water_residual = water_meta.get("max_absolute_water_balance_residual_mm_year")
    if water_residual is not None:
        water_residual = abs(float(water_residual))
        invariants.append(InvariantResult(
            "hydrology:annual_water_balance_closed", water_residual <= 0.02,
            water_residual, 0.02,
            "P + initial storage = runoff + AET + final storage within numerical bucket tolerance",
        ))

    condensate = world.get("condensate_hydrology")
    condensate_residual = None
    active_condensates: list[str] = []
    if condensate is not None:
        meta = getattr(condensate, "metadata", {}) or {}
        condensate_residual = abs(float(meta.get("mass_conservation_relative_l1_residual", 0.0)))
        active_condensates = list(meta.get("active_hydrologic_species", []))
        invariants.append(InvariantResult(
            "hydrology:multicondensate_mass_partition_closed",
            condensate_residual <= 1e-6,
            condensate_residual,
            1e-6,
            "sum of all species condensate mass fluxes must equal the transported reference condensate mass",
        ))
        forcing_shape_ok = (
            np.asarray(condensate.monthly_total_precipitation_depth_mm).shape
            == np.asarray(climate.precipitation_mm).shape
        )
        invariants.append(InvariantResult(
            "hydrology:multicondensate_forcing_shape_matches_climate",
            forcing_shape_ok,
            str(np.asarray(condensate.monthly_total_precipitation_depth_mm).shape),
            detail="species-aware hydrologic forcing must retain the monthly climate raster shape",
        ))

    sediment = {}
    if surface is not None:
        eroded = np.maximum(np.asarray(surface.cumulative_erosion_m, float), 0.0)
        deposited = np.maximum(np.asarray(surface.cumulative_deposition_m, float), 0.0)
        area = np.asarray(grid.cell_area_weights, float) * (4.0 * np.pi * grid.radius_km ** 2)
        eroded_volume_km3_proxy = float(np.sum(eroded / 1000.0 * area))
        deposited_volume_km3_proxy = float(np.sum(deposited / 1000.0 * area))
        sediment = {
            "eroded_volume_km3_proxy": eroded_volume_km3_proxy,
            "deposited_volume_km3_proxy": deposited_volume_km3_proxy,
            "deposition_to_erosion_ratio": deposited_volume_km3_proxy / max(eroded_volume_km3_proxy, 1e-12),
        }
        ledger = surface.metadata.get("sediment_mass_ledger", {}) if isinstance(surface.metadata, dict) else {}
        rel = ledger.get("relative_residual")
        if rel is not None:
            rel = abs(float(rel))
            invariants.append(InvariantResult(
                "sediment:topological_routing_mass_closed", rel <= 1e-10, rel, 1e-10,
                "each routed sediment parcel must be deposited, retained at a sink, or exported",
            ))

    if liquids is not None:
        total = max(abs(float(liquids.total_liquid_volume_m3)), 1.0)
        relative_volume_residual = abs(float(liquids.volume_residual_m3)) / total
        invariants.append(InvariantResult(
            "surface_liquid:volume_closed", relative_volume_residual <= 5e-7,
            relative_volume_residual, 5e-7,
            "integrated filled raster volume must match available mobile liquid volume",
        ))

    if appearance is not None:
        rgb_dtype_ok = np.asarray(appearance.true_color_rgb).dtype == np.uint8
        invariants.append(InvariantResult(
            "appearance:true_color_uint8_contract", rgb_dtype_ok,
            str(np.asarray(appearance.true_color_rgb).dtype), detail="true-color raster API is uint8 RGB",
        ))

    div = grid.ops.divergence(ocean.current_u, ocean.current_v)
    ocean_mask = terrain.ocean
    div_rms = float(np.sqrt(np.average(div[ocean_mask] ** 2, weights=grid.cell_area_weights[ocean_mask]))) if np.any(ocean_mask) else 0.0

    relief_p1 = relief_p99 = erosion_relief = 0.0
    precip_erosion_corr = runoff_erosion_corr = discharge_erosion_corr = None
    if np.any(terrain.land):
        land_elev_m = np.asarray(terrain.elevation_km, float)[terrain.land] * 1000.0
        relief_p1, relief_p99 = np.percentile(land_elev_m, [1, 99])
        relief_span = max(float(relief_p99 - relief_p1), 1e-9)
        if surface is not None:
            er = np.asarray(surface.cumulative_erosion_m, float)
            mean_er = float(np.average(er[terrain.land], weights=grid.cell_area_weights[terrain.land]))
            erosion_relief = mean_er / relief_span
            precip_erosion_corr = _safe_corr(climate.annual_precipitation_mm, er, terrain.land)
            runoff_erosion_corr = _safe_corr(hydro.runoff, er, terrain.land)
            discharge_erosion_corr = _safe_corr(hydro.discharge_index, er, terrain.land)

    hashes = {name: array_digest(value) for name, value in arrays.items()}
    watershed_meta = hydro.metadata.get("watersheds", {}) if isinstance(hydro.metadata, dict) else {}
    result = {
        "schema_version": 3,
        "invariants": [asdict(x) for x in invariants],
        "all_invariants_passed": bool(all(x.passed for x in invariants)),
        "metrics": {
            "land_fraction": land_fraction,
            "global_mean_temperature_c": float(np.sum(climate.annual_temperature_c * grid.cell_area_weights)),
            "land_mean_precipitation_mm_year": float(np.average(
                climate.annual_precipitation_mm[terrain.land], weights=grid.cell_area_weights[terrain.land]
            )) if np.any(terrain.land) else 0.0,
            "land_mean_runoff_mm_year": float(np.average(
                hydro.runoff[terrain.land], weights=grid.cell_area_weights[terrain.land]
            )) if np.any(terrain.land) else 0.0,
            "river_fraction_land": float(grid.weighted_fraction(hydro.rivers) / max(grid.weighted_fraction(terrain.land), 1e-12)),
            "lake_fraction_land": float(grid.weighted_fraction(hydro.lakes) / max(grid.weighted_fraction(terrain.land), 1e-12)),
            "max_strahler_order": int(np.max(hydro.stream_order)) if np.size(hydro.stream_order) else 0,
            "terminal_watershed_count": int(watershed_meta.get("outlet_basin_count", 0)),
            "mean_subgrid_drainage_density_km_per_km2": float(hydro.metadata.get("mean_subgrid_drainage_density_km_per_km2_land", 0.0)),
            "ocean_current_divergence_rms_per_km": div_rms,
            "land_relief_p1_m": float(relief_p1),
            "land_relief_p99_m": float(relief_p99),
            "mean_erosion_to_p1_p99_relief_ratio": float(erosion_relief),
            "precipitation_erosion_correlation": precip_erosion_corr,
            "runoff_erosion_correlation": runoff_erosion_corr,
            "discharge_erosion_correlation": discharge_erosion_corr,
            "multicondensate_mass_partition_relative_l1_residual": condensate_residual,
            "active_hydrologic_condensates": active_condensates,
            **sediment,
        },
        "field_hashes": hashes,
    }
    return result


def write_world_diagnostics(world: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    result = world_diagnostics(world)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
