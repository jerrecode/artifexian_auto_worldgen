from __future__ import annotations

"""Naturalized single-receiver routing for spherical rasters.

The previous reliability stencil allowed receivers several raster cells away.  That
reduced obvious D8 alignment, but it also let the drainage DAG jump over intervening
cells.  Basin labels propagated over those long hops can form striping and block-like
boundaries.  This module keeps the existing acyclic single-receiver API while using
multi-scale terrain look-ahead only to estimate a continuous preferred direction;
the actual receiver is always one immediately adjacent spherical cell.
"""

import math
from math import gcd

import numpy as np


def _primitive_offsets(radius: int = 4) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            if gcd(abs(dy), abs(dx)) != 1:
                continue
            out.append((dy, dx))
    out.sort(key=lambda d: (math.hypot(*d), math.atan2(d[0], d[1])))
    return tuple(out)


_LOOKAHEAD = _primitive_offsets(4)
_LOCAL = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _distance_km(grid, dy: int, dx: int) -> np.ndarray:
    if dx and dy:
        return np.hypot(abs(dy) * float(grid.dy_km), abs(dx) * np.asarray(grid.dx_km, float))
    if dx:
        return abs(dx) * np.asarray(grid.dx_km, float)
    return np.full(grid.shape, abs(dy) * float(grid.dy_km), dtype=np.float64)


def _texture(grid, dy: int, dx: int) -> np.ndarray:
    """Very weak coherent tie breaker; never overrules a materially steeper path."""
    a = math.atan2(float(dy), float(dx))
    phase = (
        np.deg2rad(np.asarray(grid.lon, float)) * (0.47 + 0.11 * abs(dy))
        + np.deg2rad(np.asarray(grid.lat, float)) * (0.73 + 0.09 * abs(dx))
        + 1.61803398875 * a
    )
    return np.sin(phase) + 0.35 * np.sin(1.91 * phase + 0.63)


def flow_directions_continuous_local(
    z: np.ndarray,
    ocean: np.ndarray,
    grid,
    *,
    near_tie_fraction: float = 0.08,
) -> np.ndarray:
    """Route to an adjacent cell using a multi-scale continuous steering vector.

    Long-range samples estimate the terrain's preferred downslope direction, while
    the final edge remains local.  This removes the several-cell receiver jumps that
    can imprint rectangular watershed structures while retaining substantially more
    angular information than a plain D8 steepest-descent choice.
    """
    elev = np.asarray(z, dtype=np.float64)
    oc = np.asarray(ocean, dtype=bool)
    if elev.ndim != 2 or oc.shape != elev.shape:
        raise ValueError("elevation and ocean must be equal-shaped 2-D arrays")
    if elev.shape != grid.shape:
        raise ValueError(f"elevation and ocean shape must match grid shape {grid.shape}")
    if np.any(~np.isfinite(elev)):
        raise ValueError("elevation must be finite")
    tie_value = float(near_tie_fraction)
    if not np.isfinite(tie_value):
        raise ValueError("near_tie_fraction must be finite")
    h, w = elev.shape

    best_far = np.zeros_like(elev)
    for dy, dx in _LOOKAHEAD:
        ny, nx = grid.ops.neighbor_indices(dy, dx)
        slope = (elev - elev[ny, nx]) / np.maximum(_distance_km(grid, dy, dx), 1.0e-12)
        best_far = np.maximum(best_far, slope)

    pref_x = np.zeros_like(elev)
    pref_y = np.zeros_like(elev)
    weight_sum = np.zeros_like(elev)
    for dy, dx in _LOOKAHEAD:
        ny, nx = grid.ops.neighbor_indices(dy, dx)
        dist = np.maximum(_distance_km(grid, dy, dx), 1.0e-12)
        slope = (elev - elev[ny, nx]) / dist
        useful = (slope > 0.0) & (slope >= 0.62 * best_far)
        if not np.any(useful):
            continue
        dx_phys = dx * np.asarray(grid.dx_km, float)
        dy_phys = np.full_like(elev, dy * float(grid.dy_km))
        norm = np.maximum(np.hypot(dx_phys, dy_phys), 1.0e-12)
        normalized_slope = np.maximum(slope, 0.0) / np.maximum(best_far, 1.0e-30)
        ww = np.where(useful, normalized_slope ** 2.2, 0.0)
        pref_x += ww * dx_phys / norm
        pref_y += ww * dy_phys / norm
        weight_sum += ww

    pref_mag = np.hypot(pref_x, pref_y)
    has_pref = pref_mag > 1.0e-12

    best_local = np.zeros_like(elev)
    for dy, dx in _LOCAL:
        ny, nx = grid.ops.neighbor_indices(dy, dx)
        slope = (elev - elev[ny, nx]) / np.maximum(_distance_km(grid, dy, dx), 1.0e-12)
        best_local = np.maximum(best_local, slope)

    receiver = np.full((h, w), -1, dtype=np.int32)
    best_score = np.full_like(elev, -np.inf)
    tie = float(np.clip(tie_value, 0.0, 0.20))
    active = best_local > 0.0

    for dy, dx in _LOCAL:
        ny, nx = grid.ops.neighbor_indices(dy, dx)
        dist = np.maximum(_distance_km(grid, dy, dx), 1.0e-12)
        slope = (elev - elev[ny, nx]) / dist
        close = active & (slope > 0.0) & (slope >= best_local * (1.0 - tie))
        if not np.any(close):
            continue

        dx_phys = dx * np.asarray(grid.dx_km, float)
        dy_phys = np.full_like(elev, dy * float(grid.dy_km))
        align = np.zeros_like(elev)
        edge_norm = np.maximum(np.hypot(dx_phys, dy_phys), 1.0e-12)
        align[has_pref] = (
            dx_phys[has_pref] * pref_x[has_pref]
            + dy_phys[has_pref] * pref_y[has_pref]
        ) / (edge_norm[has_pref] * pref_mag[has_pref])

        normalized = slope / np.maximum(best_local, 1.0e-30)
        # Physical slope remains dominant.  Steering and texture are only near-tie
        # terms and cannot select an uphill receiver.
        score = normalized + 0.085 * align + 0.010 * _texture(grid, dy, dx)
        better = close & (score > best_score)
        if np.any(better):
            target = (ny * w + nx).astype(np.int32, copy=False)
            receiver[better] = target[better]
            best_score[better] = score[better]

    receiver[oc] = -1
    return receiver.ravel()


__all__ = ["flow_directions_continuous_local"]
