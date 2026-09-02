from __future__ import annotations

"""Hydrology reliability kernels for spherical global rasters.

These functions address three failure modes that become obvious at high resolution:
artificial polar outlets on oceanless bodies, long lattice-aligned receiver paths,
and channel/lake masks whose drainage-area term can overwhelm the available liquid
water budget.  They preserve the canonical single-receiver DrainageGraph API while
making the receiver stencil denser and enforcing physically meaningful discharge
and occupied-cell guardrails.
"""

import math
from math import gcd
from typing import Any

import numpy as np
from scipy import ndimage

from .drainage import DrainageGraph
from .grid import normalize01
from .priority_flood import priority_flood as _ocean_priority_flood

_SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0


def _primitive_offsets(radius: int = 4) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    r = max(1, int(radius))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            if max(abs(dy), abs(dx)) > r:
                continue
            if gcd(abs(dy), abs(dx)) != 1:
                continue
            out.append((dy, dx))
    out.sort(key=lambda item: (math.hypot(*item), math.atan2(item[0], item[1])))
    return tuple(out)


_ROUTING_OFFSETS = _primitive_offsets(4)


def priority_flood_closed_aware(
    elev: np.ndarray,
    ocean: np.ndarray,
    grid,
    *,
    epsilon_km: float = 1.0e-7,
) -> np.ndarray:
    """Priority-Flood only when a genuine open ocean boundary exists."""
    z = np.asarray(elev, dtype=np.float64)
    oc = np.asarray(ocean, dtype=bool)
    if z.ndim != 2 or oc.shape != z.shape:
        raise ValueError("elevation and ocean must be equal-shaped 2-D arrays")
    if not np.any(oc):
        return z.copy()
    return _ocean_priority_flood(z, oc, grid, epsilon_km=epsilon_km)


def _direction_texture(grid, dy: int, dx: int) -> np.ndarray:
    """Smooth deterministic near-tie breaker with no seed/state dependence."""
    angle = math.atan2(float(dy), float(dx))
    phase = (
        np.deg2rad(np.asarray(grid.lon, dtype=np.float64)) * (0.83 + 0.17 * abs(dy))
        + np.deg2rad(np.asarray(grid.lat, dtype=np.float64)) * (1.11 + 0.13 * abs(dx))
        + 2.61803398875 * angle
    )
    return np.sin(phase) + 0.45 * np.sin(2.37 * phase + 0.91)


def flow_directions_multidirection(
    z: np.ndarray,
    ocean: np.ndarray,
    grid,
    *,
    near_tie_fraction: float = 0.035,
) -> np.ndarray:
    """Single-receiver descent over a dense angular stencil with near-tie steering.

    The graph stays acyclic because every accepted receiver is strictly lower than
    its source. Compared with the historical 20-direction stencil, primitive vectors
    through radius four provide many more flow angles. A weak, smooth deterministic
    steering term is considered only for slopes within 3.5% of physical steepest
    descent, removing planet-scale ruler-straight artifacts while retaining
    topographic control.
    """
    elev = np.asarray(z, dtype=np.float64)
    oc = np.asarray(ocean, dtype=bool)
    if elev.ndim != 2 or oc.shape != elev.shape:
        raise ValueError("elevation and ocean must be equal-shaped 2-D arrays")
    h, w = elev.shape
    best_slope = np.zeros_like(elev)

    for dy, dx in _ROUTING_OFFSETS:
        ny, nx = grid.ops.neighbor_indices(dy, dx)
        nb = elev[ny, nx]
        if dx and dy:
            dist = np.hypot(abs(dy) * float(grid.dy_km), abs(dx) * grid.dx_km)
        elif dx:
            dist = abs(dx) * grid.dx_km
        else:
            dist = np.full_like(grid.dx_km, abs(dy) * float(grid.dy_km))
        slope = (elev - nb) / np.maximum(dist, 1.0e-9)
        best_slope = np.maximum(best_slope, slope)

    receiver = np.full((h, w), -1, dtype=np.int32)
    best_score = np.full((h, w), -np.inf, dtype=np.float64)
    tolerance = float(np.clip(near_tie_fraction, 0.0, 0.20))
    active = best_slope > 0.0

    for dy, dx in _ROUTING_OFFSETS:
        ny, nx = grid.ops.neighbor_indices(dy, dx)
        nb = elev[ny, nx]
        if dx and dy:
            dist = np.hypot(abs(dy) * float(grid.dy_km), abs(dx) * grid.dx_km)
        elif dx:
            dist = abs(dx) * grid.dx_km
        else:
            dist = np.full_like(grid.dx_km, abs(dy) * float(grid.dy_km))
        slope = (elev - nb) / np.maximum(dist, 1.0e-9)
        close = active & (slope > 0.0) & (slope >= best_slope * (1.0 - tolerance))
        if not np.any(close):
            continue
        normalized = slope / np.maximum(best_slope, 1.0e-30)
        texture = _direction_texture(grid, dy, dx)
        hop_penalty = 0.0015 * max(math.hypot(dy, dx) - 1.0, 0.0)
        score = normalized + 0.018 * texture - hop_penalty
        better = close & (score > best_score)
        if np.any(better):
            target = (ny * w + nx).astype(np.int32, copy=False)
            receiver[better] = target[better]
            best_score[better] = score[better]

    receiver[oc] = -1
    return receiver.ravel()


def lake_mask_volume_guarded(
    grid,
    z: np.ndarray,
    filled: np.ndarray,
    land: np.ndarray,
    drainage_area: np.ndarray,
    runoff_acc: np.ndarray,
    climate,
    cfg,
) -> np.ndarray:
    """Depression/lake mask with a strict area budget and deep-core clipping."""
    elev = np.asarray(z, dtype=np.float64)
    fill = np.asarray(filled, dtype=np.float64)
    lf = np.asarray(land, dtype=bool)
    depth_m = np.maximum(fill - elev, 0.0) * 1000.0
    if not np.any(depth_m >= float(cfg.lake_min_depth_m)):
        return np.zeros_like(lf)

    pet = np.maximum(0.0, 26.0 * (np.asarray(climate.annual_temperature_c, float) + 5.0))
    moisture = np.asarray(climate.annual_precipitation_mm, float) / np.maximum(pet, 250.0)
    rv = np.asarray(runoff_acc, float)[lf & (np.asarray(runoff_acc, float) > 0)]
    rref = float(np.quantile(rv, 0.60)) if rv.size else float("inf")
    candidate = (
        lf
        & (depth_m >= float(cfg.lake_min_depth_m))
        & (np.asarray(drainage_area, float) >= float(cfg.lake_min_catchment_km2))
        & ((np.asarray(runoff_acc, float) >= rref) | (moisture > 0.85))
    )
    labs, n = grid.ops.connected_components(candidate)
    if n == 0:
        return np.zeros_like(candidate)

    ids = np.arange(1, n + 1, dtype=np.int32)
    cell_area = np.asarray(grid.cell_area_weights, float) * (
        4.0 * math.pi * float(grid.radius_km) ** 2
    )
    areas = np.asarray(ndimage.sum(cell_area, labels=labs, index=ids), dtype=float)
    max_depth = np.asarray(ndimage.maximum(depth_m, labels=labs, index=ids), dtype=float)
    max_drain = np.asarray(
        ndimage.maximum(np.asarray(drainage_area, float), labels=labs, index=ids),
        dtype=float,
    )
    mean_moist = np.asarray(ndimage.mean(moisture, labels=labs, index=ids), dtype=float)
    min_area = max(20.0, 0.08 * float(cfg.lake_min_catchment_km2))
    valid = (areas >= min_area) & ~((areas > 850000.0) & (mean_moist < 1.15))
    scores = max_depth * np.log1p(np.maximum(max_drain, 0.0)) * np.clip(mean_moist, 0.15, 3.0)
    order = np.argsort(scores)[::-1]

    max_area = float(cell_area[lf].sum()) * float(
        np.clip(cfg.lake_area_soft_cap_fraction_land, 0.001, 0.20)
    )
    result = np.zeros_like(lf)
    used = 0.0
    for j in order:
        if not valid[j] or used >= max_area:
            continue
        component = labs == (j + 1)
        area = float(areas[j])
        remaining = max_area - used
        if area <= remaining:
            result |= component
            used += area
            continue

        flat_idx = np.flatnonzero(component)
        if flat_idx.size == 0 or remaining < min_area:
            continue
        priority = depth_m.ravel()[flat_idx] * np.clip(
            moisture.ravel()[flat_idx], 0.15, 3.0
        )
        ranked = flat_idx[np.argsort(priority)[::-1]]
        cumulative = np.cumsum(cell_area.ravel()[ranked])
        take = int(np.searchsorted(cumulative, remaining, side="right"))
        if take > 0:
            result.ravel()[ranked[:take]] = True
            used += float(cumulative[take - 1])
    return result


def channel_hierarchy_discharge_guarded(
    grid,
    base: Any,
    water: Any,
    cfg: Any,
):
    """Resolved-channel hierarchy requiring physically meaningful accumulated flow."""
    land = np.asarray(base.runoff > 0.0, dtype=bool)
    graph = DrainageGraph.from_receiver(base.flow_to, land.shape)
    cell_area = np.asarray(grid.cell_area_weights, float) * (
        4.0 * math.pi * float(grid.radius_km) ** 2
    )
    storm_multiplier = 1.0 + float(getattr(cfg, "bankfull_storm_multiplier", 3.0)) * np.asarray(
        water.storminess_index, float
    )
    bankfull_source = (
        np.asarray(water.total_runoff_mm_year, float)
        * storm_multiplier
        / 1000.0
        * cell_area
    )
    bankfull = graph.accumulate(bankfull_source)
    bankfull_m3_s = bankfull * 1.0e6 / _SECONDS_PER_YEAR
    bankfull_index = normalize01(np.log1p(bankfull_m3_s)).astype(np.float32)

    from . import hydrology_base as _base

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

    min_channel_q = float(getattr(cfg, "min_resolved_channel_discharge_m3_s", 0.02))
    min_stream_q = float(getattr(cfg, "min_resolved_stream_discharge_m3_s", 0.10))
    min_perennial_q = float(getattr(cfg, "min_perennial_stream_discharge_m3_s", 1.0))
    min_river_q = float(getattr(cfg, "min_river_discharge_m3_s", 10.0))
    min_major_q = float(getattr(cfg, "min_major_river_discharge_m3_s", 100.0))

    channel = (
        land
        & (drainage >= initiation_area)
        & (metric >= 1.0)
        & (bankfull_m3_s >= min_channel_q)
    )
    stream_order = graph.strahler_order(channel).astype(np.uint8)

    cls = np.zeros(land.shape, dtype=np.uint8)
    cls[channel] = 1
    cls[channel & (stream_order >= 2) & (bankfull_m3_s >= min_stream_q)] = 2
    cls[
        channel
        & (stream_order >= 2)
        & (bankfull_m3_s >= min_perennial_q)
        & ((baseflow >= 0.15) | (wet >= 0.25))
    ] = 3
    cls[channel & (stream_order >= 2) & (bankfull_m3_s >= min_river_q)] = 4
    cls[channel & (stream_order >= 3) & (bankfull_m3_s >= min_major_q)] = 5
    rivers = cls >= 2

    land_count = int(np.count_nonzero(land))
    max_fraction = float(getattr(cfg, "max_resolved_river_cell_fraction_land", 0.20))
    max_cells = max(1, int(max_fraction * land_count)) if land_count else 0
    river_idx = np.flatnonzero(rivers)
    if max_cells and river_idx.size > max_cells:
        scores = bankfull_m3_s.ravel()[river_idx]
        keep_idx = river_idx[np.argpartition(scores, -max_cells)[-max_cells:]]
        keep = np.zeros(rivers.size, dtype=bool)
        keep[keep_idx] = True
        keep = keep.reshape(rivers.shape)
        rivers &= keep
        cls[~rivers & (cls >= 2)] = 1

    stream_order = np.where(rivers, stream_order, 0).astype(np.uint8)
    width = (
        bankfull_index
        * (0.55 + 0.45 * stream_order / max(float(stream_order.max()), 1.0))
        * rivers
    ).astype(np.float32)

    relief = np.clip(np.sqrt(np.maximum(slope, 0.0) / 0.002), 0.0, 2.0)
    density = np.clip(
        (0.10 + 1.55 * np.sqrt(np.clip(wet, 0.0, 1.8)))
        * (0.75 + 0.35 * relief)
        * (0.62 + 0.38 * (1.0 - np.asarray(water.storminess_index, float) * 0.25)),
        0.0,
        float(getattr(cfg, "max_subgrid_drainage_density_km_per_km2", 3.2)),
    ) * land
    return channel, cls, rivers, stream_order, width, bankfull_index, density


def enforce_hydrology_guardrails(result: Any, terrain: Any, cfg: Any) -> None:
    """Fail obviously pathological global hydrology instead of silently rendering it."""
    land_cells = max(int(np.count_nonzero(np.asarray(terrain.land, bool))), 1)
    river_fraction = float(result.metadata.get("river_area_fraction_of_land", 0.0))
    lake_fraction = float(result.base.metadata.get("lake_area_fraction_of_land", 0.0))
    largest_cells = int(result.metadata.get("watersheds", {}).get("largest_basin_cells", 0))
    largest_fraction = largest_cells / land_cells

    limits = {
        "river_fraction_land": float(getattr(cfg, "hydrology_fail_river_fraction_land", 0.35)),
        "lake_fraction_land": float(getattr(cfg, "hydrology_fail_lake_fraction_land", 0.10)),
        "largest_basin_fraction_land": float(
            getattr(cfg, "hydrology_fail_largest_basin_fraction_land", 0.95)
        ),
    }
    result.metadata["reliability_guardrails"] = {
        "river_fraction_land": river_fraction,
        "lake_fraction_land": lake_fraction,
        "largest_basin_fraction_land": largest_fraction,
        "limits": limits,
    }

    failures: list[str] = []
    if river_fraction > limits["river_fraction_land"]:
        failures.append(
            f"resolved river-cell fraction {river_fraction:.3f} exceeds {limits['river_fraction_land']:.3f}"
        )
    if lake_fraction > limits["lake_fraction_land"]:
        failures.append(
            f"lake-cell fraction {lake_fraction:.3f} exceeds {limits['lake_fraction_land']:.3f}"
        )
    if largest_fraction > limits["largest_basin_fraction_land"]:
        failures.append(
            f"largest terminal basin contains {largest_fraction:.3f} of land cells, exceeding {limits['largest_basin_fraction_land']:.3f}"
        )
    if failures:
        raise RuntimeError("hydrology reliability guardrail failure: " + "; ".join(failures))


__all__ = [
    "priority_flood_closed_aware",
    "flow_directions_multidirection",
    "lake_mask_volume_guarded",
    "channel_hierarchy_discharge_guarded",
    "enforce_hydrology_guardrails",
]
