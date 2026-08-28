from __future__ import annotations

"""Explicit horizontal-climate to vertical-atmogen coupling.

Worldgen owns geographic forcing and clustering.  atmogen owns every local vertical
column solve.  The coupling therefore scales with a bounded number of representative
states rather than one expensive chemistry/radiation solve per raster cell.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .atmogen_adapter import AtmogenAdapter, result_summary
from .climate import _daily_insolation_factor, _orbital_flux_factors


@dataclass(slots=True)
class RepresentativeColumnCouplingResult:
    temperature_correction_c: np.ndarray
    cluster_index: np.ndarray
    representative_temperature_k: np.ndarray
    representative_stellar_flux_scale: np.ndarray
    representative_area_fraction: np.ndarray
    solved_surface_temperature_k: np.ndarray
    summaries: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]


def annual_stellar_flux_scale(grid, astronomy_result) -> np.ndarray:
    """Annual local-mean irradiation divided by the spherical global mean."""
    tilt = float(astronomy_result.planet["axial_tilt_deg"])
    eccentricity = float(astronomy_result.planet.get("eccentricity", 0.0))
    periapsis = float(astronomy_result.planet.get("longitude_periapsis_deg", 103.0))
    declinations = [
        tilt * np.sin(2.0 * np.pi * (month + 0.5) / 12.0 - np.pi / 2.0)
        for month in range(12)
    ]
    orbital_flux = _orbital_flux_factors(eccentricity, periapsis)
    annual = np.mean(
        np.stack([
            _daily_insolation_factor(grid.lat, declination) * orbital_flux[month]
            for month, declination in enumerate(declinations)
        ], axis=0),
        axis=0,
    )
    weights = np.asarray(grid.cell_area_weights, dtype=float)
    mean = float(np.sum(annual * weights))
    if not np.isfinite(mean) or mean <= 0:
        raise RuntimeError("annual stellar forcing has a non-positive spherical mean")
    scale = np.asarray(annual / mean, dtype=np.float64)
    if np.any(~np.isfinite(scale)) or np.any(scale < 0):
        raise RuntimeError("annual stellar forcing scale is not finite/non-negative")
    return scale


def _weighted_quantile_indices(values: np.ndarray, weights: np.ndarray, count: int) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    total = float(cumulative[-1])
    targets = (np.arange(count, dtype=float) + 0.5) / count * total
    positions = np.searchsorted(cumulative, targets, side="left")
    return order[np.clip(positions, 0, order.size - 1)]


def cluster_representative_states(
    *,
    temperature_c: np.ndarray,
    stellar_flux_scale: np.ndarray,
    cell_area_weights: np.ndarray,
    count: int,
    iterations: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic weighted Lloyd clustering in temperature/irradiation state space."""
    temperature = np.asarray(temperature_c, dtype=float)
    forcing = np.asarray(stellar_flux_scale, dtype=float)
    weights = np.asarray(cell_area_weights, dtype=float)
    if temperature.shape != forcing.shape or temperature.shape != weights.shape:
        raise ValueError("temperature, stellar forcing and area weights must share a shape")
    if np.any(~np.isfinite(temperature)) or np.any(~np.isfinite(forcing)) or np.any(~np.isfinite(weights)):
        raise ValueError("representative-column inputs must be finite")
    if np.any(forcing < 0) or np.any(weights < 0) or float(np.sum(weights)) <= 0:
        raise ValueError("forcing/weights must be non-negative with positive total area")

    flat_t = temperature.reshape(-1)
    flat_f = forcing.reshape(-1)
    flat_w = weights.reshape(-1)
    active = flat_w > 0
    t = flat_t[active]
    f = flat_f[active]
    w = flat_w[active]
    k = min(max(1, int(count)), t.size)

    t_scale = max(float(np.sqrt(np.average((t - np.average(t, weights=w)) ** 2, weights=w))), 2.0)
    f_scale = max(float(np.sqrt(np.average((f - np.average(f, weights=w)) ** 2, weights=w))), 0.05)
    t_norm = (t - np.average(t, weights=w)) / t_scale
    f_norm = (f - np.average(f, weights=w)) / f_scale
    key = 0.55 * f_norm + 0.45 * t_norm
    init = _weighted_quantile_indices(key, w, k)
    centers_t = t_norm[init].copy()
    centers_f = f_norm[init].copy()
    assignment = np.zeros(t.size, dtype=np.int32)

    for _ in range(max(1, int(iterations))):
        best = np.full(t.size, np.inf, dtype=float)
        for idx in range(k):
            distance = (t_norm - centers_t[idx]) ** 2 + (f_norm - centers_f[idx]) ** 2
            take = distance < best
            assignment[take] = idx
            best[take] = distance[take]
        for idx in range(k):
            selected = assignment == idx
            if not np.any(selected):
                continue
            local_w = w[selected]
            centers_t[idx] = float(np.average(t_norm[selected], weights=local_w))
            centers_f[idx] = float(np.average(f_norm[selected], weights=local_w))

    rep_t = np.empty(k, dtype=float)
    rep_f = np.empty(k, dtype=float)
    rep_w = np.empty(k, dtype=float)
    for idx in range(k):
        selected = assignment == idx
        if not np.any(selected):
            raise RuntimeError("deterministic representative-state clustering produced an empty cluster")
        local_w = w[selected]
        rep_t[idx] = float(np.average(t[selected], weights=local_w))
        rep_f[idx] = float(np.average(f[selected], weights=local_w))
        rep_w[idx] = float(np.sum(local_w))
    rep_w /= float(np.sum(rep_w))

    full_assignment = np.full(flat_t.size, -1, dtype=np.int32)
    full_assignment[np.flatnonzero(active)] = assignment
    return full_assignment.reshape(temperature.shape), rep_t + 273.15, rep_f, rep_w


def solve_representative_columns(
    *,
    grid,
    astronomy_result,
    climate_result,
    world_config,
) -> RepresentativeColumnCouplingResult:
    """Solve clustered vertical columns and return a bounded next-pass temperature correction."""
    cfg = world_config.atmogen
    forcing = annual_stellar_flux_scale(grid, astronomy_result)
    cluster, representative_t, representative_f, area_fraction = cluster_representative_states(
        temperature_c=climate_result.annual_temperature_c,
        stellar_flux_scale=forcing,
        cell_area_weights=grid.cell_area_weights,
        count=int(cfg.representative_column_count),
    )
    # Avoid mathematically zero annual forcing at degenerate pole/grid combinations;
    # this is only an input floor to the column solver and is recorded below.
    representative_f = np.maximum(representative_f, 1.0e-6)
    adapter = AtmogenAdapter(world_config)
    results = adapter.solve_columns(
        astronomy_result,
        initial_surface_temperature_k=representative_t,
        stellar_flux_scale=representative_f,
    )
    solved_t = np.asarray([float(item.atmosphere.temperature_k[0]) for item in results], dtype=float)
    raw_delta = solved_t - representative_t
    limit = float(cfg.representative_feedback_max_abs_k)
    relaxation = float(cfg.representative_feedback_relaxation)
    relaxed_delta = np.clip(raw_delta * relaxation, -limit, limit)
    correction = relaxed_delta[cluster]
    correction = np.asarray(correction, dtype=np.float32)

    summaries: list[dict[str, Any]] = []
    for idx, result in enumerate(results):
        summary = result_summary(result)
        summary["representative_index"] = idx
        summary["area_fraction"] = float(area_fraction[idx])
        summary["input_surface_temperature_k"] = float(representative_t[idx])
        summary["stellar_flux_scale"] = float(representative_f[idx])
        summary["raw_temperature_delta_k"] = float(raw_delta[idx])
        summary["applied_temperature_delta_k"] = float(relaxed_delta[idx])
        summaries.append(summary)

    weights = np.asarray(grid.cell_area_weights, dtype=float)
    diagnostics = {
        "column_count": int(len(results)),
        "forcing_scale_area_mean": float(np.sum(forcing * weights)),
        "temperature_correction_area_mean_c": float(np.sum(correction * weights)),
        "temperature_correction_max_abs_c": float(np.max(np.abs(correction))),
        "all_columns_converged": bool(all(result.convergence.converged for result in results)),
        "max_column_energy_imbalance_w_m2": float(max(abs(result.energy_budget.imbalance_w_m2) for result in results)),
        "max_vertical_mass_closure_relative": float(max(result.vertical.mass_closure_relative for result in results)),
        "microphysics_surface_precipitation_semantics": "kg m^-2 per configured atmogen microphysics operator step; not an annual climatology",
        "feedback_semantics": "operator-split correction applied to the next horizontal climate solve",
    }
    return RepresentativeColumnCouplingResult(
        temperature_correction_c=correction,
        cluster_index=cluster,
        representative_temperature_k=representative_t,
        representative_stellar_flux_scale=representative_f,
        representative_area_fraction=area_fraction,
        solved_surface_temperature_k=solved_t,
        summaries=tuple(summaries),
        diagnostics=diagnostics,
    )
