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
    """Deterministic content digest including dtype and shape."""
    a = np.asarray(array)
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(a.dtype.str.encode("ascii"))
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    # C-order canonicalization prevents equivalent views from hashing differently.
    h.update(np.ascontiguousarray(a).view(np.uint8))
    return h.hexdigest()


def receiver_graph_is_acyclic(flow_to: np.ndarray) -> bool:
    """Return True when a single-receiver drainage graph contains no directed cycle."""
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
    if arr.size == 0:
        return 1.0
    if arr.dtype.kind not in "fc":
        return 1.0
    return float(np.mean(np.isfinite(arr)))


def world_diagnostics(world: Mapping[str, Any]) -> dict[str, Any]:
    """Compute cross-system numerical and scientific diagnostics for one generated world.

    The diagnostics intentionally avoid hard-coding Earth-only target values. They
    verify conservation-adjacent, topological and numerical properties that should
    hold for any configured world, then expose calibration metrics separately.
    """
    grid = world["grid"]
    terrain = world["terrain"]
    climate = world["climate"]
    ocean = world["ocean"]
    hydro = world["hydrology"]
    surface = world.get("surface_evolution")

    arrays = {
        "elevation_km": terrain.elevation_km,
        "annual_temperature_c": climate.annual_temperature_c,
        "annual_precipitation_mm": climate.annual_precipitation_mm,
        "runoff_mm_year": hydro.runoff,
        "ocean_current_u": ocean.current_u,
        "ocean_current_v": ocean.current_v,
    }
    finite = {name: _finite_fraction(value) for name, value in arrays.items()}

    invariants: list[InvariantResult] = []
    for name, frac in finite.items():
        invariants.append(InvariantResult(f"finite:{name}", frac == 1.0, frac, 1.0))

    invariants.append(InvariantResult(
        "hydrology:receiver_graph_acyclic",
        receiver_graph_is_acyclic(hydro.flow_to),
        detail="filled single-receiver drainage graph must be acyclic",
    ))
    runoff_min = float(np.nanmin(hydro.runoff)) if np.size(hydro.runoff) else 0.0
    invariants.append(InvariantResult("hydrology:nonnegative_runoff", runoff_min >= -1e-8, runoff_min, 1e-8))
    precip_min = float(np.nanmin(climate.precipitation_mm)) if np.size(climate.precipitation_mm) else 0.0
    invariants.append(InvariantResult("climate:nonnegative_precipitation", precip_min >= -1e-8, precip_min, 1e-8))

    land_fraction = float(grid.weighted_fraction(terrain.land))
    invariants.append(InvariantResult(
        "terrain:land_fraction_valid", 0.0 < land_fraction < 1.0, land_fraction,
        detail="planet must retain both land and ocean for the Earth-system configuration",
    ))

    # A diagnostic divergence norm is useful before the future mass-conserving
    # streamfunction ocean solver lands. It is reported, not treated as a pass/fail
    # invariant because the current reduced-order circulation is not divergence-free.
    div = grid.ops.divergence(ocean.current_u, ocean.current_v)
    ocean_mask = terrain.ocean
    div_rms = float(np.sqrt(np.average(div[ocean_mask] ** 2, weights=grid.cell_area_weights[ocean_mask]))) if np.any(ocean_mask) else 0.0

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

    hashes = {name: array_digest(value) for name, value in arrays.items()}
    result = {
        "schema_version": 1,
        "invariants": [asdict(x) for x in invariants],
        "all_invariants_passed": bool(all(x.passed for x in invariants)),
        "metrics": {
            "land_fraction": land_fraction,
            "global_mean_temperature_c": float(np.sum(climate.annual_temperature_c * grid.cell_area_weights)),
            "land_mean_precipitation_mm_year": float(np.average(
                climate.annual_precipitation_mm[terrain.land], weights=grid.cell_area_weights[terrain.land]
            )) if np.any(terrain.land) else 0.0,
            "river_fraction_land": float(grid.weighted_fraction(hydro.rivers) / max(grid.weighted_fraction(terrain.land), 1e-12)),
            "lake_fraction_land": float(grid.weighted_fraction(hydro.lakes) / max(grid.weighted_fraction(terrain.land), 1e-12)),
            "max_strahler_order": int(np.max(hydro.stream_order)) if np.size(hydro.stream_order) else 0,
            "ocean_current_divergence_rms_per_km": div_rms,
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
