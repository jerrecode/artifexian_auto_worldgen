from __future__ import annotations

"""Terrain-aware local wind and precipitation downscaling for sparse planet tiles.

The global climate remains the horizontal/seasonal authority. This module derives a
resolved terrain normal/gradient, reduces cross-barrier wind on locally resolved
slopes, and redistributes inherited precipitation between windward and lee terrain.
The precipitation perturbation is mass-neutral under cube-face area weights and is
anchored to zero on the tile perimeter, so independently generated same-LOD tiles
retain identical inherited boundary values.
"""

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np

from .planet_tiles import PlanetTilePyramid, TileKey, tile_geometry


@dataclass(slots=True, frozen=True)
class OrographicDownscalingSpec:
    wind_blocking_strength: float = 0.55
    precipitation_lift_strength: float = 0.65
    rain_shadow_strength: float = 0.45
    maximum_fractional_redistribution: float = 0.30
    edge_anchor_cells: int = 3
    minimum_wind_speed_m_s: float = 0.25

    def validate(self) -> "OrographicDownscalingSpec":
        for name in (
            "wind_blocking_strength",
            "precipitation_lift_strength",
            "rain_shadow_strength",
            "maximum_fractional_redistribution",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.maximum_fractional_redistribution > 0.45:
            raise ValueError("maximum_fractional_redistribution must be <= 0.45")
        if int(self.edge_anchor_cells) < 1:
            raise ValueError("edge_anchor_cells must be >= 1")
        if not math.isfinite(float(self.minimum_wind_speed_m_s)) or self.minimum_wind_speed_m_s <= 0:
            raise ValueError("minimum_wind_speed_m_s must be finite and positive")
        return self


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


def edge_anchor_taper(shape: tuple[int, int], cells: int) -> np.ndarray:
    """Return a smooth 0-at-edge/1-interior blending field."""
    h, w = map(int, shape)
    if h < 2 or w < 2:
        return np.zeros((h, w), dtype=np.float64)
    c = max(1, int(cells))
    y = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float64)
    x = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float64)
    ry = np.clip(y / float(c), 0.0, 1.0)
    rx = np.clip(x / float(c), 0.0, 1.0)
    # Half-cosine ramps have zero first derivative at the fully anchored/interior ends.
    ry = 0.5 - 0.5 * np.cos(np.pi * ry)
    rx = 0.5 - 0.5 * np.cos(np.pi * rx)
    return np.minimum(ry[:, None], rx[None, :])


def cube_vertex_area_weights(xyz: np.ndarray) -> np.ndarray:
    """Relative solid-angle weights for an equal-step cube-face lattice."""
    p = np.asarray(xyz, dtype=np.float64)
    if p.ndim != 3 or p.shape[-1] != 3:
        raise ValueError("xyz must have shape (y, x, 3)")
    # Cubemap Jacobian dOmega/dsdt=(1+s^2+t^2)^(-3/2). For normalized cube
    # directions the dominant absolute component is 1/sqrt(1+s^2+t^2).
    dominant = np.max(np.abs(p), axis=-1)
    weights = np.maximum(dominant, 1e-12) ** 3
    weights /= max(float(np.sum(weights)), 1e-300)
    return weights


def terrain_frame(
    xyz: np.ndarray,
    elevation_m: np.ndarray,
    planet_radius_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return `(normal_xyz, slope_deg, grad_east, grad_south)` on one tile."""
    direction = np.asarray(xyz, dtype=np.float64)
    elevation = np.asarray(elevation_m, dtype=np.float64)
    if direction.shape[:-1] != elevation.shape or direction.shape[-1] != 3:
        raise ValueError("xyz and elevation shapes are incompatible")
    radius = float(planet_radius_m)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("planet_radius_m must be finite and positive")
    position = direction * (radius + elevation)[..., None]
    edge_order = 2 if min(elevation.shape) >= 3 else 1
    drow = np.gradient(position, axis=0, edge_order=edge_order)
    dcol = np.gradient(position, axis=1, edge_order=edge_order)
    normal = np.cross(dcol, drow)
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.maximum(length, 1e-300)
    orientation = np.sum(normal * direction, axis=-1) < 0.0
    normal[orientation] *= -1.0

    radial_component = np.clip(np.sum(normal * direction, axis=-1), 1e-8, 1.0)
    tangent_normal = normal - radial_component[..., None] * direction
    gradient_xyz = -tangent_normal / radial_component[..., None]

    z_axis = np.zeros_like(direction)
    z_axis[..., 2] = 1.0
    east = np.cross(z_axis, direction)
    east_norm = np.linalg.norm(east, axis=-1, keepdims=True)
    polar = east_norm[..., 0] < 1e-8
    if np.any(polar):
        fallback = np.zeros_like(direction)
        fallback[..., 1] = 1.0
        east[polar] = np.cross(fallback[polar], direction[polar])
        east_norm = np.linalg.norm(east, axis=-1, keepdims=True)
    east /= np.maximum(east_norm, 1e-300)
    north = np.cross(direction, east)
    south = -north

    grad_east = np.sum(gradient_xyz * east, axis=-1)
    grad_south = np.sum(gradient_xyz * south, axis=-1)
    slope_deg = np.rad2deg(np.arctan(np.hypot(grad_east, grad_south)))
    return normal, slope_deg, grad_east, grad_south


def downscale_wind(
    base_u: np.ndarray,
    base_v: np.ndarray,
    grad_east: np.ndarray,
    grad_south: np.ndarray,
    taper: np.ndarray,
    *,
    spec: OrographicDownscalingSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce the upslope component of inherited east/south wind."""
    cfg = spec.validate()
    u = np.asarray(base_u, dtype=np.float64)
    v = np.asarray(base_v, dtype=np.float64)
    if u.shape != v.shape or u.shape[-2:] != grad_east.shape:
        raise ValueError("wind and terrain-gradient shapes disagree")
    g_e = np.asarray(grad_east, dtype=np.float64)
    g_s = np.asarray(grad_south, dtype=np.float64)
    gmag = np.hypot(g_e, g_s)
    unit_e = np.divide(g_e, gmag, out=np.zeros_like(g_e), where=gmag > 1e-12)
    unit_s = np.divide(g_s, gmag, out=np.zeros_like(g_s), where=gmag > 1e-12)
    proj = u * unit_e + v * unit_s
    speed = np.hypot(u, v)
    directional_slope = np.divide(
        u * g_e + v * g_s,
        np.maximum(speed, cfg.minimum_wind_speed_m_s),
    )
    reduction = np.clip(
        cfg.wind_blocking_strength * np.maximum(directional_slope, 0.0),
        0.0,
        0.85,
    )
    blocked_u = u - reduction * np.maximum(proj, 0.0) * unit_e
    blocked_v = v - reduction * np.maximum(proj, 0.0) * unit_s
    blend = np.asarray(taper, dtype=np.float64)
    while blend.ndim < u.ndim:
        blend = blend[None, ...]
    return (
        (u + blend * (blocked_u - u)).astype(np.float32),
        (v + blend * (blocked_v - v)).astype(np.float32),
    )


def redistribute_precipitation(
    base_precipitation: np.ndarray,
    wind_u: np.ndarray,
    wind_v: np.ndarray,
    grad_east: np.ndarray,
    grad_south: np.ndarray,
    taper: np.ndarray,
    area_weights: np.ndarray,
    *,
    spec: OrographicDownscalingSpec,
) -> np.ndarray:
    """Mass-neutral windward/lee redistribution anchored to inherited tile edges."""
    cfg = spec.validate()
    base = np.maximum(np.asarray(base_precipitation, dtype=np.float64), 0.0)
    u = np.asarray(wind_u, dtype=np.float64)
    v = np.asarray(wind_v, dtype=np.float64)
    if base.shape != u.shape or base.shape != v.shape:
        raise ValueError("precipitation and wind arrays must have identical shapes")
    if base.shape[-2:] != grad_east.shape:
        raise ValueError("precipitation and terrain-gradient shapes disagree")
    speed = np.hypot(u, v)
    directional_slope = (
        u * np.asarray(grad_east, dtype=np.float64)
        + v * np.asarray(grad_south, dtype=np.float64)
    ) / np.maximum(speed, cfg.minimum_wind_speed_m_s)
    raw = (
        cfg.precipitation_lift_strength * np.maximum(directional_slope, 0.0)
        - cfg.rain_shadow_strength * np.maximum(-directional_slope, 0.0)
    )
    limit = float(cfg.maximum_fractional_redistribution)
    raw = np.clip(raw, -limit, limit)
    blend = np.asarray(taper, dtype=np.float64)
    weights = np.asarray(area_weights, dtype=np.float64)
    if blend.shape != grad_east.shape or weights.shape != grad_east.shape:
        raise ValueError("taper/area weight shapes disagree")
    while blend.ndim < base.ndim:
        blend = blend[None, ...]
        weights = weights[None, ...]
    # Subtract the base-precipitation-weighted mean anomaly. This makes the
    # integrated precipitation perturbation exactly zero before float32 rounding,
    # while taper=0 leaves every perimeter vertex equal to its inherited value.
    numerator = np.sum(weights * base * blend * raw, axis=(-2, -1), keepdims=True)
    denominator = np.sum(weights * base * blend, axis=(-2, -1), keepdims=True)
    mean_raw = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-20,
    )
    anomaly = blend * (raw - mean_raw)
    # Since both raw and its weighted mean lie in [-limit, limit], |anomaly| <=
    # 2*limit <= 0.9, preserving non-negative precipitation without clipping.
    result = base * (1.0 + anomaly)
    return np.maximum(result, 0.0).astype(np.float32)


class LocalOrographicDownscaler:
    def __init__(
        self,
        pyramid: PlanetTilePyramid,
        *,
        spec: OrographicDownscalingSpec | None = None,
    ) -> None:
        self.pyramid = pyramid
        self.spec = (spec or OrographicDownscalingSpec()).validate()
        self.root = pyramid.root / "derived" / "local_orography_v1"

    def _path(self, key: TileKey, field: str) -> Path:
        return (
            self.root / field / f"z{key.level:02d}" / key.face
            / f"x{key.x:08d}" / f"y{key.y:08d}.npy"
        )

    def _metadata_path(self, key: TileKey) -> Path:
        return (
            self.root / "metadata" / f"z{key.level:02d}" / key.face
            / f"x{key.x:08d}" / f"y{key.y:08d}.json"
        )

    def generate(self, key: TileKey) -> dict[str, Path]:
        key.validate()
        names = (
            "terrain_normal_xyz",
            "slope_deg",
            "wind_u_monthly",
            "wind_v_monthly",
            "precipitation_mm_monthly",
            "annual_precipitation_mm",
        )
        paths = {name: self._path(key, name) for name in names}
        if all(path.exists() for path in paths.values()) and self._metadata_path(key).exists():
            return paths

        geom = tile_geometry(key, self.pyramid.spec.tile_size)
        elevation = np.asarray(self.pyramid.load_field(key, "elevation_m"), dtype=np.float64)
        normal, slope, grad_e, grad_s = terrain_frame(
            geom.xyz, elevation, self.pyramid.planet_radius_m
        )
        taper = edge_anchor_taper(elevation.shape, self.spec.edge_anchor_cells)
        weights = cube_vertex_area_weights(geom.xyz)
        base_u = np.asarray(
            self.pyramid._sample_source_field("wind_u_monthly", geom), dtype=np.float64
        )
        base_v = np.asarray(
            self.pyramid._sample_source_field("wind_v_monthly", geom), dtype=np.float64
        )
        base_p = np.asarray(
            self.pyramid._sample_source_field("precipitation_mm_monthly", geom), dtype=np.float64
        )
        if base_u.ndim != 3 or base_v.shape != base_u.shape or base_p.shape != base_u.shape:
            raise RuntimeError("monthly wind/precipitation must sample to equal (month,y,x) arrays")
        local_u, local_v = downscale_wind(
            base_u, base_v, grad_e, grad_s, taper, spec=self.spec
        )
        local_p = redistribute_precipitation(
            base_p, local_u, local_v, grad_e, grad_s, taper, weights, spec=self.spec
        )
        values = {
            "terrain_normal_xyz": normal.astype(np.float32),
            "slope_deg": slope.astype(np.float32),
            "wind_u_monthly": local_u,
            "wind_v_monthly": local_v,
            "precipitation_mm_monthly": local_p,
            "annual_precipitation_mm": np.sum(local_p, axis=0, dtype=np.float64).astype(np.float32),
        }
        for name, value in values.items():
            _atomic_save_npy(paths[name], value)
        metadata = {
            "schema_version": 1,
            "key": asdict(key),
            "source_sha256": self.pyramid._source_hash(),
            "spec": asdict(self.spec),
            "generated_fields": list(names),
            "semantics": {
                "wind": "global inherited wind with terrain cross-barrier blocking blended to zero at the tile perimeter",
                "precipitation": "global inherited monthly precipitation redistributed between windward/lee terrain with zero integrated tile perturbation and anchored perimeter",
                "boundary": "all local climate perturbations vanish at the output perimeter for independent same-LOD seam continuity",
            },
        }
        _atomic_json(self._metadata_path(key), metadata)
        return paths


__all__ = [
    "LocalOrographicDownscaler",
    "OrographicDownscalingSpec",
    "cube_vertex_area_weights",
    "downscale_wind",
    "edge_anchor_taper",
    "redistribute_precipitation",
    "terrain_frame",
]
