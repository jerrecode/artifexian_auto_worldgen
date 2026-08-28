from __future__ import annotations

"""Low-speed tectonic initialization compatibility layer.

The historical tectonic initializer sampled macro-plate speeds from ``[0.8,
max_plate_speed_cm_yr]``.  That interval is invalid for deliberately sluggish
stagnant-lid/icy worlds such as the Titan preset, whose configured maximum speed is
well below 0.8 cm/yr.  More importantly, the old subplate perturbation used a hard
angular-rate floor that could inject motion orders of magnitude above a very small
configured maximum.

This module installs a corrected initializer while the tectonic core remains on its
legacy monolithic module boundary.  It preserves the historical Earth-like lower
speed floor when the configured maximum is large enough, scales the lower bound down
for slow worlds, and preserves an exactly stationary solution when the configured
maximum is zero.
"""

import numpy as np

from .config import TectonicsConfig
from .grid import SphereGrid
from . import tectonics as _tectonics


def _initial_subplates_low_speed(
    macro_centers: np.ndarray,
    cfg: TectonicsConfig,
    grid: SphereGrid,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = _tectonics._subplate_counts(len(macro_centers), cfg, rng)
    parent: list[int] = []
    centers: list[np.ndarray] = []
    omega_des: list[np.ndarray] = []
    sub_cont: list[bool] = []

    macro_axes = _tectonics.random_unit_vectors(rng, len(macro_centers))
    max_speed = max(float(cfg.max_plate_speed_cm_yr), 0.0)
    if max_speed > 0.0:
        # Keep the historical 0.8 cm/yr lower floor for normally active worlds,
        # but never let the lower bound exceed the configured physical maximum.
        min_speed = min(0.8, 0.25 * max_speed)
        speeds = rng.uniform(min_speed, max_speed, len(macro_centers))
    else:
        speeds = np.zeros(len(macro_centers), dtype=np.float64)

    macro_rates = speeds * 10.0 / grid.radius_km  # 1 cm/yr = 10 km/Myr
    macro_sign = rng.choice([-1.0, 1.0], len(macro_centers))
    macro_omega = macro_axes * (macro_rates * macro_sign)[:, None]
    macro_cont = rng.random(len(macro_centers)) < cfg.continental_plate_fraction
    if not np.any(macro_cont):
        macro_cont[int(rng.integers(0, len(macro_cont)))] = True

    for p, (c, k) in enumerate(zip(macro_centers, counts)):
        spread = np.deg2rad(rng.uniform(8.0, 23.0))
        for j in range(int(k)):
            angle = 0.0 if j == 0 else abs(float(rng.normal(spread * 0.62, spread * 0.26)))
            angle = min(angle, spread * 1.45)
            sc = _tectonics._offset_on_sphere(c, angle, float(rng.uniform(0, 2 * np.pi)))
            centers.append(sc)
            parent.append(p)
            sub_cont.append(bool(macro_cont[p]))

            base_mag = float(np.linalg.norm(macro_omega[p]))
            if base_mag <= 0.0:
                # A configured zero maximum must remain exactly stationary instead
                # of acquiring motion from a numerical perturbation floor.
                omega_des.append(np.zeros(3, dtype=np.float64))
                continue

            perturb = rng.normal(size=3)
            perturb -= (
                np.dot(perturb, macro_omega[p])
                * macro_omega[p]
                / max(np.dot(macro_omega[p], macro_omega[p]), 1e-30)
            )
            pnorm = float(np.linalg.norm(perturb))
            if pnorm > 0.0:
                perturb /= pnorm
            om = macro_omega[p] + perturb * base_mag * rng.normal(
                0.0, cfg.subplate_motion_dispersion
            )
            om *= rng.uniform(0.82, 1.18)
            omega_des.append(om)

    return (
        np.asarray(centers, dtype=np.float64),
        np.asarray(parent, dtype=np.int32),
        np.asarray(omega_des, dtype=np.float64),
        np.asarray(sub_cont, dtype=bool),
    )


def install_low_speed_tectonics_fix() -> None:
    """Install the corrected initializer idempotently."""
    if _tectonics._initial_subplates is not _initial_subplates_low_speed:
        _tectonics._initial_subplates = _initial_subplates_low_speed


__all__ = ["install_low_speed_tectonics_fix"]
