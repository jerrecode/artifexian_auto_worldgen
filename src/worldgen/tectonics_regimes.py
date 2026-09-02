from __future__ import annotations

"""Regime-safe tectonic motion initialization.

The historical tectonic generator used a fixed 0.8 cm/yr lower bound for macro-plate
speed.  That is appropriate for its active terrestrial calibration but is invalid for
stagnant/inactive shells whose configured maximum may be orders of magnitude lower.

This module replaces only the initial-motion sampler.  For configurations with
``max_plate_speed_cm_yr >= 0.8`` the historical random draw is exactly preserved.
For slower configurations, speeds span 15--100% of the requested maximum, retaining
motion heterogeneity without silently turning an inactive world into Earth.
"""

import numpy as np

from . import tectonics as _tect


_ORIGINAL_INITIAL_SUBPLATES = _tect._initial_subplates


def _macro_plate_speeds_cm_yr(
    rng: np.random.Generator,
    max_speed_cm_yr: float,
    count: int,
) -> np.ndarray:
    vmax = max(float(max_speed_cm_yr), 0.0)
    if count <= 0:
        return np.empty(0, dtype=np.float64)
    if vmax <= 0.0:
        return np.zeros(count, dtype=np.float64)
    # Preserve historical active-world RNG behavior exactly whenever its original
    # 0.8 cm/yr floor is valid.  Only low-activity regimes take the new branch.
    lower = 0.8 if vmax >= 0.8 else 0.15 * vmax
    return rng.uniform(lower, vmax, count)


def _initial_subplates_regime_safe(
    macro_centers,
    cfg,
    grid,
    rng: np.random.Generator,
):
    counts = _tect._subplate_counts(len(macro_centers), cfg, rng)
    parent: list[int] = []
    centers: list[np.ndarray] = []
    omega_des: list[np.ndarray] = []
    sub_cont: list[bool] = []

    macro_axes = _tect.random_unit_vectors(rng, len(macro_centers))
    speeds = _macro_plate_speeds_cm_yr(rng, cfg.max_plate_speed_cm_yr, len(macro_centers))
    macro_rates = speeds * 10.0 / grid.radius_km  # rad/Myr proxy
    macro_sign = rng.choice([-1.0, 1.0], len(macro_centers))
    macro_omega = macro_axes * (macro_rates * macro_sign)[:, None]
    macro_cont = rng.random(len(macro_centers)) < cfg.continental_plate_fraction
    if not np.any(macro_cont):
        macro_cont[int(rng.integers(0, len(macro_cont)))] = True

    for p, (center, k) in enumerate(zip(macro_centers, counts)):
        spread = np.deg2rad(rng.uniform(8.0, 23.0))
        for j in range(int(k)):
            angle = 0.0 if j == 0 else abs(float(rng.normal(spread * 0.62, spread * 0.26)))
            angle = min(angle, spread * 1.45)
            subcenter = _tect._offset_on_sphere(
                center, angle, float(rng.uniform(0, 2 * np.pi))
            )
            centers.append(subcenter)
            parent.append(p)
            sub_cont.append(bool(macro_cont[p]))

            perturb = rng.normal(size=3)
            parent_omega = macro_omega[p]
            denom = max(float(np.dot(parent_omega, parent_omega)), 1.0e-12)
            perturb -= np.dot(perturb, parent_omega) * parent_omega / denom
            pnorm = float(np.linalg.norm(perturb))
            if pnorm > 0.0:
                perturb /= pnorm
            # Scale perturbations to the actual slow macro motion.  The tiny numerical
            # floor prevents undefined orientation without imposing a physical speed.
            mag = max(float(np.linalg.norm(parent_omega)), 1.0e-12)
            omega = parent_omega + perturb * mag * rng.normal(
                0.0, cfg.subplate_motion_dispersion
            )
            omega *= rng.uniform(0.82, 1.18)
            omega_des.append(omega)

    return (
        np.asarray(centers, float),
        np.asarray(parent, np.int32),
        np.asarray(omega_des, float),
        np.asarray(sub_cont, bool),
    )


def install_regime_safe_tectonic_initializer() -> None:
    """Install the low-speed-safe initializer into the canonical tectonics module."""
    _tect._initial_subplates = _initial_subplates_regime_safe


__all__ = [
    "_macro_plate_speeds_cm_yr",
    "_initial_subplates_regime_safe",
    "install_regime_safe_tectonic_initializer",
]
