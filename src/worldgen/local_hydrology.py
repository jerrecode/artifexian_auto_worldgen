from __future__ import annotations

"""Open-boundary local hydrology for sparse high-resolution terrain tiles.

Continental drainage topology remains a global-world responsibility.  This module
uses a deterministic halo around one requested cube-sphere tile, inherits global
runoff and major-river constraints, rebuilds sub-grid D8 drainage with open patch
boundaries, and crops the result back to the tile core.

It intentionally does not reuse global ``flow_to`` indices: those identify cells in
the equirectangular simulation raster and are meaningless in the sparse cube-sphere
address space.
"""

from dataclasses import asdict, dataclass
import heapq
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np
from scipy import ndimage

from .procedural_erosion import phase_cell_octave_xyz
from .planet_tiles import (
    PlanetTilePyramid,
    TileGeometry,
    TileKey,
    _cube_direction,
    tile_geometry,
)


_D8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


_D16 = _D8 + (
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
)

_D16_ANGLES = np.asarray(
    [math.atan2(float(dy), float(dx)) for dy, dx in _D16],
    dtype=np.float64,
)

_D16_TO_D8 = np.asarray(
    [
        int(
            np.argmin(
                np.abs(
                    np.angle(
                        np.exp(
                            1j
                            * (
                                math.atan2(float(dy), float(dx))
                                - np.asarray(
                                    [math.atan2(float(ddy), float(ddx)) for ddy, ddx in _D8]
                                )
                            )
                        )
                    )
                )
            )
        )
        for dy, dx in _D16
    ],
    dtype=np.int8,
)


@dataclass(slots=True, frozen=True)
class LocalHydrologySpec:
    halo_cells: int = 16
    priority_flood_epsilon_m: float = 0.01
    stream_quantile: float = 0.985
    fallback_runoff_base_fraction: float = 0.24
    meander_strength: float = 0.78
    meander_max_turn_deg: float = 55.0
    meander_slope_scale: float = 0.012
    meander_discharge_power: float = 0.65
    major_river_corridor_cells: int = 7
    major_river_guide_depth_m: float = 4.0

    def validate(self) -> "LocalHydrologySpec":
        if not 2 <= int(self.halo_cells) <= 256:
            raise ValueError("halo_cells must be in [2, 256]")
        if not math.isfinite(float(self.priority_flood_epsilon_m)) or float(
            self.priority_flood_epsilon_m
        ) < 0.0:
            raise ValueError("priority_flood_epsilon_m must be finite and non-negative")
        if not 0.5 <= float(self.stream_quantile) < 1.0:
            raise ValueError("stream_quantile must be in [0.5, 1)")
        if not 0.0 <= float(self.fallback_runoff_base_fraction) <= 1.0:
            raise ValueError("fallback_runoff_base_fraction must be in [0, 1]")
        if not 0.0 <= float(self.meander_strength) <= 1.0:
            raise ValueError("meander_strength must be in [0,1]")
        if not 0.0 <= float(self.meander_max_turn_deg) <= 80.0:
            raise ValueError("meander_max_turn_deg must be in [0,80]")
        if not math.isfinite(float(self.meander_slope_scale)) or self.meander_slope_scale <= 0.0:
            raise ValueError("meander_slope_scale must be positive")
        if not 0.0 < float(self.meander_discharge_power) <= 2.0:
            raise ValueError("meander_discharge_power must be in (0,2]")
        if not 1 <= int(self.major_river_corridor_cells) <= 64:
            raise ValueError("major_river_corridor_cells must be in [1,64]")
        if not 0.0 <= float(self.major_river_guide_depth_m) <= 50.0:
            raise ValueError("major_river_guide_depth_m must be in [0,50]")
        return self


@dataclass(slots=True, frozen=True)
class LocalHydrologyResult:
    filled_elevation_m: np.ndarray
    flow_direction_d8: np.ndarray
    flow_direction_d16: np.ndarray
    flow_angle_rad: np.ndarray
    meander_potential: np.ndarray
    runoff_mm_year: np.ndarray
    drainage_area_km2: np.ndarray
    discharge_index: np.ndarray
    streams: np.ndarray
    inherited_major_river: np.ndarray
    metadata: Mapping[str, object]


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, np.asarray(values), allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _patch_geometry(key: TileKey, tile_size: int, halo: int) -> TileGeometry:
    """Extend a tile's cube-face parameterization by ``halo`` sample intervals."""
    key.validate()
    n = int(tile_size)
    side = key.side
    offsets = np.arange(-halo, n + halo + 1, dtype=np.float64) / float(n)
    qx = int(key.x) + offsets
    qy = int(key.y) + offsets
    s = -1.0 + 2.0 * qx[None, :] / float(side)
    t = -1.0 + 2.0 * qy[:, None] / float(side)
    s, t = np.broadcast_arrays(s, t)
    xyz = _cube_direction(key.face, s, t)
    lat = np.rad2deg(np.arcsin(np.clip(xyz[..., 2], -1.0, 1.0)))
    lon = np.rad2deg(np.arctan2(xyz[..., 1], xyz[..., 0]))
    return TileGeometry(xyz=xyz, latitude_deg=lat, longitude_deg=lon)


def _resolved_elevation_patch(
    pyramid: PlanetTilePyramid,
    key: TileKey,
    geom: TileGeometry,
) -> np.ndarray:
    base = np.asarray(pyramid._sample_source_field("elevation_m", geom), dtype=np.float64)
    detail = pyramid._spectral_detail(geom.xyz, key.level)
    modulation = 0.65 + 0.35 * np.tanh(np.abs(base) / 1500.0)
    return base + modulation * detail


def _coastal_land(ocean: np.ndarray) -> np.ndarray:
    ocean = np.asarray(ocean, dtype=bool)
    land = ~ocean
    result = np.zeros_like(ocean)
    h, w = ocean.shape
    for dy, dx in _D8:
        sy0 = max(0, -dy)
        sy1 = min(h, h - dy)
        sx0 = max(0, -dx)
        sx1 = min(w, w - dx)
        ty0, ty1 = sy0 + dy, sy1 + dy
        tx0, tx1 = sx0 + dx, sx1 + dx
        result[sy0:sy1, sx0:sx1] |= ocean[ty0:ty1, tx0:tx1]
    return result & land


def _priority_flood_open(
    elevation_m: np.ndarray,
    ocean: np.ndarray,
    *,
    epsilon_m: float,
) -> np.ndarray:
    """Priority-Flood with ocean and patch perimeter as explicit open boundaries."""
    z = np.asarray(elevation_m, dtype=np.float64).copy()
    oc = np.asarray(ocean, dtype=bool)
    if z.ndim != 2 or oc.shape != z.shape:
        raise ValueError("elevation and ocean must be equal-shaped 2-D arrays")
    h, w = z.shape
    if h < 3 or w < 3:
        raise ValueError("open local priority flood requires at least 3x3 samples")
    visited = oc.copy()
    seed = _coastal_land(oc)
    seed[0, :] |= ~oc[0, :]
    seed[-1, :] |= ~oc[-1, :]
    seed[:, 0] |= ~oc[:, 0]
    seed[:, -1] |= ~oc[:, -1]
    heap: list[tuple[float, int, int]] = []
    ys, xs = np.where(seed & ~visited)
    for y, x in zip(ys.tolist(), xs.tolist()):
        visited[y, x] = True
        heapq.heappush(heap, (float(z[y, x]), y, x))
    if not heap and not np.all(oc):
        raise RuntimeError("local priority flood could not establish an open boundary")
    eps = max(float(epsilon_m), 0.0)
    while heap:
        cur, y, x = heapq.heappop(heap)
        for dy, dx in _D8:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w or visited[ny, nx]:
                continue
            visited[ny, nx] = True
            nz = float(z[ny, nx])
            if nz <= cur:
                nz = cur + eps
                z[ny, nx] = nz
            heapq.heappush(heap, (nz, ny, nx))
    return z


def _great_circle_distance_m(a: np.ndarray, b: np.ndarray, radius_m: float) -> np.ndarray:
    dot = np.sum(a * b, axis=-1)
    return float(radius_m) * np.arccos(np.clip(dot, -1.0, 1.0))


def _flow_d8_open(
    filled_elevation_m: np.ndarray,
    ocean: np.ndarray,
    xyz: np.ndarray,
    radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Steepest-descent D8 flow with no wrapping and perimeter cells as outlets."""
    z = np.asarray(filled_elevation_m, dtype=np.float64)
    oc = np.asarray(ocean, dtype=bool)
    h, w = z.shape
    best = np.zeros((h, w), dtype=np.float64)
    receiver = np.full((h, w), -1, dtype=np.int64)
    code = np.full((h, w), -1, dtype=np.int8)
    for direction, (dy, dx) in enumerate(_D8):
        sy0 = max(0, -dy)
        sy1 = min(h, h - dy)
        sx0 = max(0, -dx)
        sx1 = min(w, w - dx)
        ty0, ty1 = sy0 + dy, sy1 + dy
        tx0, tx1 = sx0 + dx, sx1 + dx
        source = z[sy0:sy1, sx0:sx1]
        target = z[ty0:ty1, tx0:tx1]
        distance = _great_circle_distance_m(
            xyz[sy0:sy1, sx0:sx1],
            xyz[ty0:ty1, tx0:tx1],
            radius_m,
        )
        slope = (source - target) / np.maximum(distance, 1.0e-6)
        view_best = best[sy0:sy1, sx0:sx1]
        better = slope > view_best
        if np.any(better):
            target_y = np.arange(ty0, ty1, dtype=np.int64)[:, None]
            target_x = np.arange(tx0, tx1, dtype=np.int64)[None, :]
            target_flat = target_y * w + target_x
            receiver_view = receiver[sy0:sy1, sx0:sx1]
            code_view = code[sy0:sy1, sx0:sx1]
            receiver_view[better] = np.broadcast_to(target_flat, better.shape)[better]
            code_view[better] = direction
            view_best[better] = slope[better]
    receiver[oc] = -1
    code[oc] = -1
    receiver[0, :] = -1
    receiver[-1, :] = -1
    receiver[:, 0] = -1
    receiver[:, -1] = -1
    code[0, :] = -1
    code[-1, :] = -1
    code[:, 0] = -1
    code[:, -1] = -1
    return receiver.ravel(), code


def _flow_d16_open(
    filled_elevation_m: np.ndarray,
    ocean: np.ndarray,
    xyz: np.ndarray,
    radius_m: float,
    *,
    preferred_angle_rad: np.ndarray | None = None,
    steering_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steepest-descent routing on a 16-direction queen+knight stencil.

    Knight moves supply intermediate directions missing from D8.  They are accepted
    only when at least one raster corridor between the source and target remains
    below the source elevation, preventing the router from jumping across a ridge.
    Optional steering rotates the preferred direction on low-gradient, high-
    discharge reaches while every receiver remains strictly downhill.
    """
    z = np.asarray(filled_elevation_m, dtype=np.float64)
    oc = np.asarray(ocean, dtype=bool)
    unit = np.asarray(xyz, dtype=np.float64)
    h, w = z.shape
    if oc.shape != z.shape or unit.shape != (h, w, 3):
        raise ValueError("D16 routing inputs have incompatible shapes")
    preferred = (
        None
        if preferred_angle_rad is None
        else np.asarray(preferred_angle_rad, dtype=np.float64)
    )
    steer = (
        np.zeros(z.shape, dtype=np.float64)
        if steering_weight is None
        else np.clip(np.asarray(steering_weight, dtype=np.float64), 0.0, 1.0)
    )
    if preferred is not None and preferred.shape != z.shape:
        raise ValueError("preferred_angle_rad must match elevation")
    if steer.shape != z.shape:
        raise ValueError("steering_weight must match elevation")

    best_score = np.zeros((h, w), dtype=np.float64)
    best_slope = np.zeros((h, w), dtype=np.float64)
    receiver = np.full((h, w), -1, dtype=np.int64)
    code = np.full((h, w), -1, dtype=np.int8)

    for direction, (dy, dx) in enumerate(_D16):
        sy0 = max(0, -dy)
        sy1 = min(h, h - dy)
        sx0 = max(0, -dx)
        sx1 = min(w, w - dx)
        ty0, ty1 = sy0 + dy, sy1 + dy
        tx0, tx1 = sx0 + dx, sx1 + dx
        source = z[sy0:sy1, sx0:sx1]
        target = z[ty0:ty1, tx0:tx1]
        distance = _great_circle_distance_m(
            unit[sy0:sy1, sx0:sx1],
            unit[ty0:ty1, tx0:tx1],
            radius_m,
        )
        slope = (source - target) / np.maximum(distance, 1.0e-6)
        valid = slope > 0.0

        if max(abs(dy), abs(dx)) > 1:
            ys = np.arange(sy0, sy1, dtype=np.int64)[:, None]
            xs = np.arange(sx0, sx1, dtype=np.int64)[None, :]
            sdy = int(np.sign(dy))
            sdx = int(np.sign(dx))
            if abs(dy) == 2:
                mid_a = z[ys + sdy, xs]
                mid_b = z[ys + sdy, xs + sdx]
            else:
                mid_a = z[ys, xs + sdx]
                mid_b = z[ys + sdy, xs + sdx]
            corridor = np.minimum(mid_a, mid_b)
            # At least one plausible raster corridor must descend away from the
            # source.  A very small tolerance avoids floating-point rejection on
            # Priority-Flood flats.
            corridor_limit = source - 0.015 * np.maximum(source - target, 0.0)
            valid &= corridor <= corridor_limit + 1.0e-8

        score = np.where(valid, slope, 0.0)
        if preferred is not None:
            delta = np.angle(
                np.exp(
                    1j
                    * (
                        float(_D16_ANGLES[direction])
                        - preferred[sy0:sy1, sx0:sx1]
                    )
                )
            )
            alignment = np.square(0.5 + 0.5 * np.cos(delta))
            sw = steer[sy0:sy1, sx0:sx1]
            directional = (1.0 - sw) + sw * (0.10 + 0.90 * alignment)
            score *= directional

        view_best = best_score[sy0:sy1, sx0:sx1]
        better = score > view_best
        if np.any(better):
            target_y = np.arange(ty0, ty1, dtype=np.int64)[:, None]
            target_x = np.arange(tx0, tx1, dtype=np.int64)[None, :]
            target_flat = target_y * w + target_x
            receiver_view = receiver[sy0:sy1, sx0:sx1]
            code_view = code[sy0:sy1, sx0:sx1]
            slope_view = best_slope[sy0:sy1, sx0:sx1]
            receiver_view[better] = np.broadcast_to(target_flat, better.shape)[better]
            code_view[better] = direction
            slope_view[better] = slope[better]
            view_best[better] = score[better]

    receiver[oc] = -1
    code[oc] = -1
    best_slope[oc] = 0.0
    for edge in (
        (0, slice(None)),
        (-1, slice(None)),
        (slice(None), 0),
        (slice(None), -1),
    ):
        receiver[edge] = -1
        code[edge] = -1
        best_slope[edge] = 0.0
    return receiver.ravel(), code, best_slope


def _compat_d8_codes(code16: np.ndarray) -> np.ndarray:
    code = np.asarray(code16, dtype=np.int16)
    out = np.full(code.shape, -1, dtype=np.int8)
    active = code >= 0
    if np.any(active):
        out[active] = _D16_TO_D8[code[active]]
    return out


def _flow_angles(code16: np.ndarray) -> np.ndarray:
    code = np.asarray(code16, dtype=np.int16)
    out = np.full(code.shape, np.nan, dtype=np.float64)
    active = code >= 0
    if np.any(active):
        out[active] = _D16_ANGLES[code[active]]
    return out


def _meander_phase(
    xyz: np.ndarray,
    radius_m: float,
    discharge_index: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Absolute-coordinate meander phase with discharge-scaled wavelength."""
    unit = np.asarray(xyz, dtype=np.float64)
    q = np.clip(np.asarray(discharge_index, dtype=np.float64), 0.0, 1.0)
    rng = np.random.default_rng(int(seed) ^ 0x4D45414E44455232)
    axis = rng.normal(size=3)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-15)
    tangent = axis - np.sum(unit * axis, axis=-1, keepdims=True) * unit
    norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    fallback_axis = np.array([0.0, 0.0, 1.0])
    fallback = fallback_axis - (
        np.sum(unit * fallback_axis, axis=-1, keepdims=True) * unit
    )
    tangent = np.where(norm > 1.0e-10, tangent, fallback)
    tangent /= np.maximum(
        np.linalg.norm(tangent, axis=-1, keepdims=True), 1.0e-12
    )
    wavelength_km = 18.0 + 165.0 * np.power(q, 0.72)
    _cosine, sine, coherence = phase_cell_octave_xyz(
        unit,
        float(radius_m) / 1000.0,
        wavelength_km,
        tangent,
        cell_scale=0.82,
        seed=int(seed) ^ 0x6D65616E,
        octave=31,
    )
    return np.asarray(sine * coherence, dtype=np.float64)


def _major_river_guide(
    inherited_major_river: np.ndarray,
    *,
    corridor_cells: int,
) -> np.ndarray:
    raw = np.asarray(inherited_major_river, dtype=bool)
    if not np.any(raw):
        return np.zeros(raw.shape, dtype=np.float64)
    distance = ndimage.distance_transform_edt(~raw)
    sigma = max(float(corridor_cells) * 0.52, 1.0)
    guide = np.exp(-0.5 * np.square(distance / sigma))
    guide[distance > float(corridor_cells) * 1.75] = 0.0
    return np.asarray(guide, dtype=np.float64)


def _connect_stream_knight_moves(
    streams: np.ndarray,
    receiver_flat: np.ndarray,
    code16: np.ndarray,
) -> np.ndarray:
    """Rasterize the intermediate cell of D16 knight-move channel segments."""
    out = np.asarray(streams, dtype=bool).copy()
    code = np.asarray(code16, dtype=np.int16)
    h, w = out.shape
    for direction, (dy, dx) in enumerate(_D16):
        if max(abs(dy), abs(dx)) <= 1:
            continue
        ys, xs = np.where(out & (code == direction))
        if ys.size == 0:
            continue
        sdy = int(np.sign(dy))
        sdx = int(np.sign(dx))
        parity = (ys + xs + direction) & 1
        if abs(dy) == 2:
            my = ys + sdy
            mx = xs + parity * sdx
        else:
            my = ys + parity * sdy
            mx = xs + sdx
        valid = (my >= 0) & (my < h) & (mx >= 0) & (mx < w)
        out[my[valid], mx[valid]] = True
    return out


def _routing_metrics(
    receiver_flat: np.ndarray,
    code16: np.ndarray,
    streams: np.ndarray,
    discharge_index: np.ndarray,
    xyz: np.ndarray,
    radius_m: float,
    *,
    active_mask: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    receiver = np.asarray(receiver_flat, dtype=np.int64)
    code = np.asarray(code16, dtype=np.int16).ravel()
    stream = np.asarray(streams, dtype=bool).ravel()
    q = np.asarray(discharge_index, dtype=np.float64).ravel()
    unit = np.asarray(xyz, dtype=np.float64).reshape((-1, 3))
    active = stream & (code >= 0)
    if active_mask is not None:
        mask = np.asarray(active_mask, dtype=bool)
        if mask.shape != np.asarray(streams).shape:
            raise ValueError("active_mask must match streams")
        active &= mask.ravel()
    if not np.any(active):
        return {
            "directional_fourfold_anisotropy": 0.0,
            "directional_fourfold_moment_real": 0.0,
            "directional_fourfold_moment_imag": 0.0,
            "directional_eighth_anisotropy": 0.0,
            "directional_eighth_moment_real": 0.0,
            "directional_eighth_moment_imag": 0.0,
            "stream_direction_count": 0,
            "stream_transition_count": 0,
            "stream_turn_fraction_gt10deg": 0.0,
            "max_straight_run_cells": 0,
            "median_sampled_sinuosity": None,
        }

    angles = _D16_ANGLES[code[active]]
    fourth_sum = np.sum(np.exp(4j * angles))
    eighth_sum = np.sum(np.exp(8j * angles))
    anisotropy = float(np.abs(fourth_sum / max(len(angles), 1)))
    eighth_anisotropy = float(np.abs(eighth_sum / max(len(angles), 1)))

    nodes = np.flatnonzero(active)
    targets = receiver[nodes]
    valid_target = (
        (targets >= 0)
        & (targets < stream.size)
        & stream[np.clip(targets, 0, stream.size - 1)]
        & (code[np.clip(targets, 0, code.size - 1)] >= 0)
    )
    transition_count = int(np.count_nonzero(valid_target))
    if transition_count:
        a0 = _D16_ANGLES[code[nodes[valid_target]]]
        a1 = _D16_ANGLES[code[targets[valid_target]]]
        turn = np.abs(np.angle(np.exp(1j * (a1 - a0))))
        turn_fraction = float(np.mean(turn >= np.deg2rad(10.0)))
    else:
        turn_fraction = 0.0

    max_straight = 0
    for start in nodes.tolist():
        direction = int(code[start])
        cur = int(start)
        run = 0
        while run < 512:
            target = int(receiver[cur])
            if (
                target < 0
                or target >= stream.size
                or not stream[target]
                or int(code[cur]) != direction
            ):
                break
            run += 1
            cur = target
            if int(code[cur]) != direction:
                break
        max_straight = max(max_straight, run)

    candidates = nodes[np.argsort(q[nodes], kind="stable")[-min(160, len(nodes)):]]
    sinuosity: list[float] = []
    for start in candidates.tolist():
        cur = int(start)
        distance = 0.0
        segments = 0
        for _ in range(160):
            target = int(receiver[cur])
            if (
                target < 0
                or target >= stream.size
                or not stream[target]
            ):
                break
            distance += float(
                _great_circle_distance_m(
                    unit[cur][None, :],
                    unit[target][None, :],
                    radius_m,
                )[0]
            )
            segments += 1
            cur = target
        if segments >= 12 and distance > 0.0:
            direct = float(
                _great_circle_distance_m(
                    unit[start][None, :],
                    unit[cur][None, :],
                    radius_m,
                )[0]
            )
            if direct > 1.0:
                sinuosity.append(distance / direct)
    return {
        "directional_fourfold_anisotropy": anisotropy,
        "directional_fourfold_moment_real": float(np.real(fourth_sum)),
        "directional_fourfold_moment_imag": float(np.imag(fourth_sum)),
        "directional_eighth_anisotropy": eighth_anisotropy,
        "directional_eighth_moment_real": float(np.real(eighth_sum)),
        "directional_eighth_moment_imag": float(np.imag(eighth_sum)),
        "stream_direction_count": int(len(angles)),
        "stream_transition_count": transition_count,
        "stream_turn_fraction_gt10deg": turn_fraction,
        "max_straight_run_cells": int(max_straight),
        "median_sampled_sinuosity": (
            float(np.median(sinuosity)) if sinuosity else None
        ),
    }


def _sample_area_km2(xyz: np.ndarray, radius_m: float) -> np.ndarray:
    """Approximate vertex support area from local great-circle neighbour spacing."""
    h, w, _ = xyz.shape
    dx = np.empty((h, w), dtype=np.float64)
    dy = np.empty((h, w), dtype=np.float64)
    dx[:, 1:-1] = 0.5 * _great_circle_distance_m(
        xyz[:, :-2], xyz[:, 2:], radius_m
    )
    dx[:, 0] = _great_circle_distance_m(xyz[:, 0], xyz[:, 1], radius_m)
    dx[:, -1] = _great_circle_distance_m(xyz[:, -2], xyz[:, -1], radius_m)
    dy[1:-1, :] = 0.5 * _great_circle_distance_m(
        xyz[:-2, :], xyz[2:, :], radius_m
    )
    dy[0, :] = _great_circle_distance_m(xyz[0, :], xyz[1, :], radius_m)
    dy[-1, :] = _great_circle_distance_m(xyz[-2, :], xyz[-1, :], radius_m)
    return np.maximum(dx * dy / 1.0e6, 1.0e-12)


def _accumulate_open(
    filled_elevation_m: np.ndarray,
    receiver_flat: np.ndarray,
    runoff_mm_year: np.ndarray,
    area_km2: np.ndarray,
    ocean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(filled_elevation_m, dtype=np.float64).ravel()
    receiver = np.asarray(receiver_flat, dtype=np.int64)
    area = np.asarray(area_km2, dtype=np.float64).ravel()
    runoff = np.asarray(runoff_mm_year, dtype=np.float64).ravel()
    land = ~np.asarray(ocean, dtype=bool).ravel()
    drainage = area * land
    discharge = np.maximum(runoff, 0.0) * area * land
    # Priority-Flood epsilon makes interior receiver heights strictly lower.  A
    # descending elevation pass is therefore a deterministic topological order.
    for node in np.argsort(z, kind="stable")[::-1]:
        target = int(receiver[node])
        if target >= 0:
            drainage[target] += drainage[node]
            discharge[target] += discharge[node]
    return drainage.reshape(filled_elevation_m.shape), discharge.reshape(
        filled_elevation_m.shape
    )


def _normalize_log(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    active = np.asarray(mask, dtype=bool) & np.isfinite(values) & (values > 0)
    if not np.any(active):
        return out
    logged = np.log1p(np.asarray(values, dtype=np.float64))
    lo = float(np.min(logged[active]))
    hi = float(np.max(logged[active]))
    if hi <= lo:
        out[active] = 1.0
    else:
        out[active] = (logged[active] - lo) / (hi - lo)
    return out


class LocalHydrologySolver:
    """Compute/cache local high-resolution drainage for individual terrain tiles."""

    def __init__(
        self,
        pyramid: PlanetTilePyramid,
        *,
        spec: LocalHydrologySpec | None = None,
    ) -> None:
        self.pyramid = pyramid
        self.spec = (spec or LocalHydrologySpec()).validate()
        self.root = pyramid.root / "derived" / "local_hydrology_v1"

    def _path(self, key: TileKey, field: str) -> Path:
        key.validate()
        return (
            self.root
            / field
            / f"z{key.level:02d}"
            / key.face
            / f"x{key.x:08d}"
            / f"y{key.y:08d}.npy"
        )

    def _metadata_path(self, key: TileKey) -> Path:
        return (
            self.root
            / "metadata"
            / f"z{key.level:02d}"
            / key.face
            / f"x{key.x:08d}"
            / f"y{key.y:08d}.json"
        )

    def _load_cached(self, key: TileKey) -> LocalHydrologyResult | None:
        fields = {
            "filled_elevation_m": np.float32,
            "flow_direction_d8": np.int8,
            "flow_direction_d16": np.int8,
            "flow_angle_rad": np.float32,
            "meander_potential": np.float32,
            "runoff_mm_year": np.float32,
            "drainage_area_km2": np.float32,
            "discharge_index": np.float32,
            "streams": np.bool_,
            "inherited_major_river": np.bool_,
        }
        paths = {name: self._path(key, name) for name in fields}
        meta_path = self._metadata_path(key)
        if not meta_path.exists() or not all(path.exists() for path in paths.values()):
            return None
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        arrays = {
            name: np.load(path, mmap_mode="r", allow_pickle=False)
            for name, path in paths.items()
        }
        return LocalHydrologyResult(metadata=metadata, **arrays)

    def _runoff_patch(
        self, geom: TileGeometry, elevation_m: np.ndarray
    ) -> tuple[np.ndarray, str]:
        _shape, fields = self.pyramid._source_metadata()
        if "runoff_mm_year" in fields:
            runoff = np.asarray(
                self.pyramid._sample_source_field("runoff_mm_year", geom),
                dtype=np.float64,
            )
            return np.maximum(runoff, 0.0), "inherited global runoff_mm_year"
        if "annual_precipitation_mm" not in fields:
            return np.zeros(elevation_m.shape, dtype=np.float64), (
                "no global runoff_mm_year or annual_precipitation_mm; local runoff set to zero"
            )
        precipitation = np.maximum(
            np.asarray(
                self.pyramid._sample_source_field("annual_precipitation_mm", geom),
                dtype=np.float64,
            ),
            0.0,
        )
        frac = float(self.spec.fallback_runoff_base_fraction) + 0.46 * (
            1.0 - np.exp(-precipitation / 1050.0)
        )
        if "annual_temperature_c" in fields:
            temperature = np.asarray(
                self.pyramid._sample_source_field("annual_temperature_c", geom),
                dtype=np.float64,
            )
            # Apply the same resolved-relief lapse correction used by local climate.
            inherited_elevation = np.asarray(
                self.pyramid._sample_source_field("elevation_m", geom),
                dtype=np.float64,
            )
            temperature = temperature - 6.5 * (
                elevation_m - inherited_elevation
            ) / 1000.0
            frac -= np.clip((temperature - 8.0) / 38.0, 0.0, 0.28)
        return precipitation * np.clip(frac, 0.05, 0.92), (
            "fallback runoff derived from inherited precipitation and temperature; "
            "lithology/snow terms unavailable at this local boundary"
        )

    def _major_river_patch(self, geom: TileGeometry) -> np.ndarray:
        _shape, fields = self.pyramid._source_metadata()
        if "rivers" not in fields:
            return np.zeros(geom.latitude_deg.shape, dtype=bool)
        inherited = np.asarray(
            self.pyramid._sample_source_field("rivers", geom), dtype=np.float64
        )
        return inherited >= 0.5

    def _parent_discharge_patch(self, geom: TileGeometry) -> np.ndarray:
        _shape, fields = self.pyramid._source_metadata()
        if "discharge_index" not in fields:
            return np.zeros(geom.latitude_deg.shape, dtype=np.float64)
        return np.clip(
            np.asarray(
                self.pyramid._sample_source_field("discharge_index", geom),
                dtype=np.float64,
            ),
            0.0,
            1.0,
        )

    def _route_elevation(
        self,
        key: TileKey,
        geom: TileGeometry,
        elevation_m: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, object], np.ndarray]:
        cfg = self.spec
        elevation = np.asarray(elevation_m, dtype=np.float64)
        if elevation.shape != geom.latitude_deg.shape:
            raise ValueError("elevation_m and tile geometry shapes differ")
        ocean = elevation < 0.0
        inherited_river = self._major_river_patch(geom) & ~ocean
        guide = _major_river_guide(
            inherited_river,
            corridor_cells=int(cfg.major_river_corridor_cells),
        )
        # The low-resolution global river is a topology corridor, not a raster
        # centreline. Lower only the routing potential within the corridor so the
        # high-resolution network can find its own valley-conforming path.
        routing_surface = elevation - (
            float(cfg.major_river_guide_depth_m) * guide * (~ocean)
        )
        filled = _priority_flood_open(
            routing_surface,
            ocean,
            epsilon_m=float(cfg.priority_flood_epsilon_m),
        )

        runoff, runoff_semantics = self._runoff_patch(geom, elevation)
        runoff = np.asarray(runoff, dtype=np.float64)
        runoff[ocean] = 0.0
        area = _sample_area_km2(geom.xyz, self.pyramid.planet_radius_m)

        receiver0, code0, best_slope0 = _flow_d16_open(
            filled,
            ocean,
            geom.xyz,
            self.pyramid.planet_radius_m,
        )
        drainage0, discharge0 = _accumulate_open(
            filled, receiver0, runoff, area, ocean
        )
        local0 = _normalize_log(discharge0, ~ocean)
        base_angle = _flow_angles(code0)
        phase = _meander_phase(
            geom.xyz,
            self.pyramid.planet_radius_m,
            local0,
            seed=int(self.pyramid._read_seed()),
        )
        meander = (
            float(cfg.meander_strength)
            * np.power(np.clip(local0, 0.0, 1.0), float(cfg.meander_discharge_power))
            * np.exp(
                -np.maximum(best_slope0, 0.0)
                / max(float(cfg.meander_slope_scale), 1.0e-12)
            )
            * (~ocean)
        )
        # Major-river corridors permit a little more lateral migration because
        # kilometre-scale alluvial channels are not expected to occupy the exact
        # coarse parent-raster centreline.
        meander *= 0.82 + 0.18 * guide
        preferred = np.where(
            np.isfinite(base_angle),
            base_angle
            + np.deg2rad(float(cfg.meander_max_turn_deg)) * phase * meander,
            0.0,
        )

        receiver, code16, _best_slope = _flow_d16_open(
            filled,
            ocean,
            geom.xyz,
            self.pyramid.planet_radius_m,
            preferred_angle_rad=preferred,
            steering_weight=meander,
        )
        drainage, discharge = _accumulate_open(
            filled, receiver, runoff, area, ocean
        )
        local_discharge = _normalize_log(discharge, ~ocean)

        parent_discharge = self._parent_discharge_patch(geom)
        # Preserve continental topology as a soft discharge prior, but only where
        # the locally routed terrain already carries water. This avoids stamping
        # the straight low-resolution parent mask into the refined stream raster.
        parent_support = (
            parent_discharge
            * guide
            * (0.28 + 0.72 * np.sqrt(np.clip(local_discharge, 0.0, 1.0)))
        )
        channel_score = np.maximum(local_discharge, parent_support)
        land_values = channel_score[~ocean]
        if land_values.size:
            threshold = float(np.quantile(land_values, cfg.stream_quantile))
            streams = (~ocean) & (
                channel_score >= max(threshold, 1.0e-12)
            )
        else:
            threshold = 1.0
            streams = np.zeros_like(ocean)

        corridor = (guide >= 0.28) & (~ocean) & (parent_discharge >= 0.12)
        if np.any(corridor):
            routed = local_discharge[corridor]
            if routed.size:
                corridor_threshold = float(np.quantile(routed, 0.76))
                streams |= corridor & (
                    local_discharge >= max(corridor_threshold, 1.0e-12)
                )

        streams = _connect_stream_knight_moves(streams, receiver, code16)
        code8 = _compat_d8_codes(code16)
        angle = _flow_angles(code16)
        metrics = _routing_metrics(
            receiver,
            code16,
            streams,
            local_discharge,
            geom.xyz,
            self.pyramid.planet_radius_m,
        )
        arrays = {
            "filled_elevation_m": np.asarray(filled, dtype=np.float32),
            "flow_direction_d8": np.asarray(code8, dtype=np.int8),
            "flow_direction_d16": np.asarray(code16, dtype=np.int8),
            "flow_angle_rad": np.asarray(angle, dtype=np.float32),
            "meander_potential": np.asarray(meander, dtype=np.float32),
            "runoff_mm_year": np.asarray(runoff, dtype=np.float32),
            "drainage_area_km2": np.asarray(drainage, dtype=np.float32),
            "discharge_index": np.asarray(local_discharge, dtype=np.float32),
            "streams": np.asarray(streams, dtype=np.bool_),
            "inherited_major_river": np.asarray(inherited_river, dtype=np.bool_),
        }
        route_meta: dict[str, object] = {
            "runoff_semantics": runoff_semantics,
            "stream_threshold_discharge_index": threshold,
            "major_river_guide_cells": int(np.count_nonzero(guide > 0.05)),
            "inherited_major_river_cells": int(np.count_nonzero(inherited_river)),
            "local_stream_cells": int(np.count_nonzero(streams)),
            "routing_metrics": metrics,
            "flow_direction_semantics": {
                "type": "D16 queen+knight direction code with low-gradient meander steering",
                "codes": {
                    str(i): [dy, dx] for i, (dy, dx) in enumerate(_D16)
                },
                "compatibility_d8_field": "flow_direction_d8",
                "outlet": -1,
                "strictly_downhill_receivers": True,
                "long_move_ridge_jump_guard": True,
                "not_global_flow_to": True,
            },
        }
        return arrays, route_meta, receiver

    def solve_elevation(
        self,
        key: TileKey,
        elevation_m: np.ndarray,
    ) -> LocalHydrologyResult:
        """Route over final terrain with a deterministic halo and no core-edge outlets.

        The supplied tile is inserted into the same halo geometry used by the
        pre-erosion local solve.  Outside the core, the halo is reconstructed from
        the globally continuous inherited terrain plus native absolute-XYZ
        microrelief.  Because geomorphic perturbations are explicitly anchored to
        zero at every tile edge, this gives a continuous boundary neighbourhood
        without depending on neighbour generation order.

        Only the halo perimeter is open.  The exported tile perimeter therefore
        behaves as ordinary interior routing context rather than as an artificial
        row of outlets, substantially reducing river terminations and grid seams.
        """
        key.validate()
        n = int(self.pyramid.spec.tile_size)
        halo = int(self.spec.halo_cells)
        elevation = np.asarray(elevation_m, dtype=np.float64)
        if elevation.shape != (n + 1, n + 1):
            raise ValueError(
                f"final terrain shape must be {(n + 1, n + 1)}, got {elevation.shape}"
            )

        geom = _patch_geometry(key, n, halo)
        patch_elevation = _resolved_elevation_patch(self.pyramid, key, geom)
        core = (slice(halo, halo + n + 1), slice(halo, halo + n + 1))
        patch_elevation = np.asarray(patch_elevation, dtype=np.float64).copy()
        patch_elevation[core] = elevation

        arrays_patch, route_meta, receiver = self._route_elevation(
            key, geom, patch_elevation
        )

        core_mask = np.zeros(patch_elevation.shape, dtype=bool)
        core_mask[core] = True
        route_meta = dict(route_meta)
        route_meta["routing_metrics"] = _routing_metrics(
            receiver,
            np.asarray(arrays_patch["flow_direction_d16"]),
            np.asarray(arrays_patch["streams"]),
            np.asarray(arrays_patch["discharge_index"]),
            geom.xyz,
            self.pyramid.planet_radius_m,
            active_mask=core_mask,
        )

        arrays = {
            name: np.asarray(values[core], dtype=values.dtype)
            for name, values in arrays_patch.items()
        }
        metadata = {
            "schema_version": 3,
            "key": asdict(key),
            "spec": asdict(self.spec),
            "source_sha256": self.pyramid._source_hash(),
            "patch_shape": [
                int(patch_elevation.shape[0]),
                int(patch_elevation.shape[1]),
            ],
            "core_shape": [n + 1, n + 1],
            **route_meta,
            "boundary_semantics": (
                "final-terrain reroute uses a deterministic inherited-terrain halo; "
                "only the halo perimeter is open, so the exported tile perimeter "
                "is not an artificial outlet"
            ),
            "terrain_semantics": (
                "core routing is evaluated after all local terrain detail and erosion; "
                "halo context uses globally continuous inherited terrain plus native "
                "absolute-XYZ microrelief"
            ),
        }
        return LocalHydrologyResult(metadata=metadata, **arrays)

    def solve(self, key: TileKey) -> LocalHydrologyResult:
        key.validate()
        cached = self._load_cached(key)
        if cached is not None:
            return cached
        n = int(self.pyramid.spec.tile_size)
        halo = int(self.spec.halo_cells)
        geom = _patch_geometry(key, n, halo)
        elevation = _resolved_elevation_patch(self.pyramid, key, geom)
        arrays_patch, route_meta, _receiver = self._route_elevation(
            key, geom, elevation
        )

        core = (slice(halo, halo + n + 1), slice(halo, halo + n + 1))
        arrays = {
            name: np.asarray(values[core], dtype=values.dtype)
            for name, values in arrays_patch.items()
        }
        metadata = {
            "schema_version": 2,
            "key": asdict(key),
            "spec": asdict(self.spec),
            "source_sha256": self.pyramid._source_hash(),
            "patch_shape": [int(elevation.shape[0]), int(elevation.shape[1])],
            "core_shape": [n + 1, n + 1],
            **route_meta,
            "boundary_semantics": (
                "halo patch has open non-periodic perimeter; global basin topology "
                "is a soft corridor rather than a stamped centreline"
            ),
            "limitations": [
                "local stream accumulation can terminate at halo perimeter and is not a substitute for continental drainage area",
                "cube-face-edge halos use normalized extension of the tile face parameterization and are boundary context only",
            ],
        }
        for name, values in arrays.items():
            _atomic_save_npy(self._path(key, name), values)
        _atomic_json(self._metadata_path(key), metadata)
        return LocalHydrologyResult(metadata=metadata, **arrays)


__all__ = [
    "LocalHydrologyResult",
    "LocalHydrologySolver",
    "LocalHydrologySpec",
]
