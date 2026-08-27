from __future__ import annotations

"""Reduced-order mass-conserving barotropic ocean circulation.

This backend deliberately remains lightweight: it does not solve the full primitive
equations. Instead it builds a basin-aware streamfunction from wind-stress curl,
latitude-dependent Coriolis/beta structure, bathymetry, and boundary-current
shaping. Horizontal velocity is then derived as the rotated spherical gradient of
the streamfunction, which makes the interior transport non-divergent by
construction up to discrete polar/coastal masking error.
"""

from dataclasses import dataclass

import numpy as np

from .config import OceanConfig
from .grid import SphereGrid, normalize01, smooth_periodic


@dataclass(slots=True, frozen=True)
class BarotropicDiagnostics:
    divergence_rms_per_km: float
    interior_divergence_rms_per_km: float
    kinetic_energy_index: float


def _masked_scale(values: np.ndarray, mask: np.ndarray, percentile: float = 95.0) -> float:
    vals = np.abs(np.asarray(values, dtype=float)[np.asarray(mask, dtype=bool)])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    return max(float(np.percentile(vals, percentile)), 1e-12)


def _normalize_monthly_vectors(
    u: np.ndarray, v: np.ndarray, ocean: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    uu = np.asarray(u, dtype=float).copy()
    vv = np.asarray(v, dtype=float).copy()
    for month in range(uu.shape[0]):
        ref = _masked_scale(np.hypot(uu[month], vv[month]), ocean)
        uu[month] /= ref
        vv[month] /= ref
    return uu, vv


def velocity_from_streamfunction(
    grid: SphereGrid,
    streamfunction: np.ndarray,
    ocean: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return east/south velocity from a scalar streamfunction.

    With raster-y/south positive, ``u_east = dpsi/dsouth`` and
    ``v_south = -dpsi/deast``. The public spherical metric-gradient operator is
    used so the construction has the same pole/seam convention as diagnostics.
    """
    psi = np.asarray(streamfunction, dtype=float)
    gy, gx = grid.ops.metric_gradient(psi)
    u = gy
    v = -gx
    if ocean is not None:
        wet = np.asarray(ocean, dtype=bool)
        u = np.where(wet, u, 0.0)
        v = np.where(wet, v, 0.0)
    return u, v


def _basin_envelope(
    coast_distance_km: np.ndarray,
    ocean: np.ndarray,
    boundary_width_km: float,
) -> np.ndarray:
    """Smoothly force the streamfunction toward a common coastal boundary value."""
    wet = np.asarray(ocean, dtype=bool)
    d = np.asarray(coast_distance_km, dtype=float)
    width = max(float(boundary_width_km), 1.0)
    envelope = np.tanh(np.maximum(d, 0.0) / max(0.30 * width, 1.0))
    return np.where(wet, envelope, 0.0)


def _monthly_signed_normalize(values: np.ndarray, ocean: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    wet = np.asarray(ocean, dtype=bool)
    for month in range(out.shape[0]):
        ref = _masked_scale(out[month], wet)
        out[month] = np.clip(out[month] / ref, -4.0, 4.0)
    return out


def build_barotropic_currents(
    grid: SphereGrid,
    ocean: np.ndarray,
    depth_m: np.ndarray,
    coast_distance_km: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    cfg: OceanConfig,
) -> tuple[np.ndarray, np.ndarray, BarotropicDiagnostics]:
    """Construct monthly divergence-minimizing horizontal circulation.

    The forcing is Sverdrup/Stommel-inspired rather than a full PDE solve. Wind
    curl is scaled by a normalized beta factor, relaxed laterally, modulated by
    basin geometry and bathymetry, and converted to velocity solely through a
    streamfunction. Therefore wind forcing never injects a separate divergent
    velocity component as it does in the fast heuristic backend.
    """
    wet = np.asarray(ocean, dtype=bool)
    depth = np.asarray(depth_m, dtype=float)
    wu = np.asarray(wind_u, dtype=float)
    wv = np.asarray(wind_v, dtype=float)
    if wu.shape != wv.shape or wu.ndim != 3 or wu.shape[1:] != wet.shape:
        raise ValueError("wind_u and wind_v must have shape [month,height,width]")

    lat_rad = np.deg2rad(grid.lat)
    cosphi = np.abs(np.cos(lat_rad))
    # Dimensionless beta structure. A floor avoids amplifying polar discretization
    # noise; amplitude is normalized later, so the omitted common 2*Omega/R factor
    # does not affect this reduced-order direction/shape solution.
    beta_shape = np.maximum(cosphi, 0.12)
    wind_curl = grid.ops.curl(wu, wv)
    forcing = _monthly_signed_normalize(wind_curl / beta_shape[None, ...], wet)

    envelope = _basin_envelope(
        coast_distance_km, wet, float(cfg.boundary_current_width_km)
    )
    depth_norm = normalize01(np.where(wet, depth, 0.0), robust=True) * wet
    depth_anomaly = depth_norm - (
        float(np.mean(depth_norm[wet])) if np.any(wet) else 0.0
    )

    coast_dist = np.asarray(coast_distance_km, dtype=float)
    cgy, cgx = grid.ops.metric_gradient(coast_dist)
    cgn = np.hypot(cgx, cgy) + 1e-12
    east_from_coast = cgx / cgn
    near_coast = np.exp(
        -coast_dist / max(float(cfg.boundary_current_width_km), 50.0)
    ) * wet
    western = np.clip(east_from_coast, 0.0, 1.0) * near_coast
    eastern = np.clip(-east_from_coast, 0.0, 1.0) * near_coast
    boundary_shape = (
        1.0
        + float(cfg.western_boundary_strength) * 0.55 * western
        - float(cfg.eastern_boundary_strength) * 0.16 * eastern
    )

    # A background planetary gyre mode makes weak-wind months and enclosed basins
    # well behaved while retaining the same broad circulation family as fast mode.
    base_gyre = np.sin(2.0 * lat_rad) - 0.28 * np.sin(4.0 * lat_rad)
    base_gyre *= envelope

    psi = np.empty_like(wu, dtype=float)
    relax_reps = max(1, int(cfg.current_iterations) // 12)
    for month in range(wu.shape[0]):
        seasonal_forcing = smooth_periodic(forcing[month], (1.25, 1.8))
        bathy_term = (
            float(cfg.bathymetric_steering_strength)
            * 0.22
            * depth_anomaly
            * np.sign(np.sin(lat_rad) + 1e-12)
        )
        field = envelope * (
            0.76 * seasonal_forcing
            + 0.24 * float(cfg.gyre_strength) * base_gyre
            + bathy_term
        )
        for _ in range(relax_reps):
            field = smooth_periodic(field, (1.1, 1.55)) * envelope
        psi[month] = field * boundary_shape

    u = np.empty_like(psi)
    v = np.empty_like(psi)
    for month in range(psi.shape[0]):
        u[month], v[month] = velocity_from_streamfunction(grid, psi[month], wet)
    u, v = _normalize_monthly_vectors(u, v, wet)
    u *= wet[None, ...]
    v *= wet[None, ...]

    u_ann = u.mean(axis=0)
    v_ann = v.mean(axis=0)
    divergence = grid.ops.divergence(u_ann, v_ann)
    if np.any(wet):
        weights = grid.cell_area_weights[wet]
        div_rms = float(
            np.sqrt(np.average(divergence[wet] ** 2, weights=weights))
        )
        interior = wet & (
            coast_dist
            > max(2.0 * float(grid.dy_km), 0.5 * float(cfg.boundary_current_width_km))
        )
        if np.any(interior):
            int_rms = float(
                np.sqrt(
                    np.average(
                        divergence[interior] ** 2,
                        weights=grid.cell_area_weights[interior],
                    )
                )
            )
        else:
            int_rms = div_rms
        ke = float(
            np.average(
                0.5 * (u_ann[wet] ** 2 + v_ann[wet] ** 2),
                weights=weights,
            )
        )
    else:
        div_rms = int_rms = ke = 0.0

    diagnostics = BarotropicDiagnostics(div_rms, int_rms, ke)
    return u.astype(np.float32), v.astype(np.float32), diagnostics


__all__ = [
    "BarotropicDiagnostics",
    "build_barotropic_currents",
    "velocity_from_streamfunction",
]
