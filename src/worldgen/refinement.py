from __future__ import annotations

"""Public recursive-refinement API.

The implementation lives in :mod:`worldgen.refinement_core`. This thin module also
installs the partition-invariant spherical detail evaluator: normalization must be
global-by-construction, never computed from one tile's local statistics, otherwise
changing the section layout changes the generated relief and can create section
signatures.
"""

import numpy as np

from . import refinement_core as _core
from .refinement_core import *  # noqa: F401,F403


def _partition_invariant_spherical_detail(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    *,
    seed: int,
    frequency: float,
) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    c = np.cos(lat)
    xyz = np.stack((c * np.cos(lon), c * np.sin(lon), np.sin(lat)), axis=-1)
    rng = np.random.default_rng(seed)
    detail = np.zeros(lat.shape, dtype=np.float64)
    total_weight = 0.0
    for octave, weight in enumerate((1.0, 0.55, 0.30)):
        freq = float(frequency) * (2.0**octave)
        for _ in range(4):
            axis = rng.normal(size=3)
            axis /= max(float(np.linalg.norm(axis)), 1e-12)
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            detail += weight * np.sin(
                freq * np.tensordot(xyz, axis, axes=([-1], [0])) + phase
            )
            total_weight += weight
    # The deterministic divisor depends only on the configured octave weights,
    # not on which geographic subsection happened to evaluate these coordinates.
    return detail / max(total_weight, 1e-12)


_core._spherical_detail = _partition_invariant_spherical_detail
__all__ = _core.__all__
