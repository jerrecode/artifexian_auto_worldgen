from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Iterator

import numpy as np


@dataclass(slots=True, frozen=True)
class Tile2D:
    core: tuple[slice, slice]
    expanded: tuple[slice, slice]
    crop: tuple[slice, slice]


def working_set_bytes(*arrays: object) -> int:
    total = 0
    for value in arrays:
        nbytes = getattr(value, "nbytes", None)
        if isinstance(nbytes, int):
            total += max(0, nbytes)
    return total


def auto_chunk_shape(
    shape: tuple[int, int],
    dtype=np.float32,
    *,
    target_mb: float = 32.0,
    arrays_in_flight: int = 6,
    minimum_rows: int = 8,
) -> tuple[int, int]:
    """Choose a row-major 2-D chunk that respects an approximate working-set cap."""
    h, w = map(int, shape)
    if h <= 0 or w <= 0:
        raise ValueError("shape must be positive")
    itemsize = np.dtype(dtype).itemsize
    budget = max(1, int(target_mb * 1024**2))
    bytes_per_row = max(1, w * itemsize * max(1, int(arrays_in_flight)))
    rows = max(1, min(h, budget // bytes_per_row))
    rows = min(h, max(min(h, int(minimum_rows)), rows))
    return rows, w


def iter_tiles_2d(
    shape: tuple[int, int],
    *,
    chunk_shape: tuple[int, int] | None = None,
    halo: int = 0,
) -> Iterator[Tile2D]:
    """Yield deterministic tiles with clipped halo and crop slices."""
    h, w = map(int, shape)
    ch, cw = chunk_shape or (h, w)
    ch = max(1, int(ch)); cw = max(1, int(cw)); halo = max(0, int(halo))
    for y0 in range(0, h, ch):
        y1 = min(h, y0 + ch)
        for x0 in range(0, w, cw):
            x1 = min(w, x0 + cw)
            ey0 = max(0, y0 - halo); ey1 = min(h, y1 + halo)
            ex0 = max(0, x0 - halo); ex1 = min(w, x1 + halo)
            core = (slice(y0, y1), slice(x0, x1))
            expanded = (slice(ey0, ey1), slice(ex0, ex1))
            crop = (slice(y0 - ey0, y1 - ey0), slice(x0 - ex0, x1 - ex0))
            yield Tile2D(core=core, expanded=expanded, crop=crop)


def fast_hypot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Use NumExpr for large arrays when available, otherwise NumPy."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("a and b must have equal shapes")
    if a.size >= 200_000 and importlib.util.find_spec("numexpr") is not None:
        import numexpr as ne
        return ne.evaluate("sqrt(a*a + b*b)", local_dict={"a": a, "b": b})
    return np.hypot(a, b)


def finite_normalize(values: np.ndarray, *, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Normalize finite values to [low, high] without NaN/Inf propagation."""
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    out = np.full(arr.shape, low, dtype=np.float64)
    if not finite.any():
        return out
    lo = float(arr[finite].min()); hi = float(arr[finite].max())
    if hi <= lo:
        out[finite] = (low + high) * 0.5
        return out
    out[finite] = low + (arr[finite] - lo) * ((high - low) / (hi - lo))
    return out


def compensated_sum(values: np.ndarray) -> float:
    """Kahan-Neumaier style scalar reduction for numerically sensitive totals."""
    total = 0.0
    correction = 0.0
    for x in np.asarray(values, dtype=np.float64).ravel():
        t = total + float(x)
        if abs(total) >= abs(x):
            correction += (total - t) + float(x)
        else:
            correction += (float(x) - t) + total
        total = t
    return total + correction


def weighted_mean_stable(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape:
        raise ValueError("values and weights must have equal shapes")
    finite = np.isfinite(values) & np.isfinite(weights) & (weights != 0)
    if not finite.any():
        return float("nan")
    numerator = compensated_sum(values[finite] * weights[finite])
    denominator = compensated_sum(weights[finite])
    return numerator / denominator if denominator != 0 else float("nan")


def optional_njit(*jit_args, **jit_kwargs):
    """Return ``numba.njit`` when installed, otherwise an identity decorator."""
    if importlib.util.find_spec("numba") is not None:
        from numba import njit
        return njit(*jit_args, **jit_kwargs)

    def decorate(fn):
        return fn
    return decorate
