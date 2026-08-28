from __future__ import annotations

"""Camera-driven screen-space-error selection for sparse planetary tiles."""

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .planet_tiles import (
    CUBE_FACES,
    TileKey,
    _angular_distance,
    _tile_bounding_cap,
    approximate_meters_per_sample,
    latlon_to_unit,
)


@dataclass(slots=True, frozen=True)
class CameraLODRequest:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    viewport_width_px: int = 1920
    viewport_height_px: int = 1080
    vertical_fov_deg: float = 60.0
    maximum_screen_error_px: float = 1.5
    maximum_level: int = 24
    maximum_tiles: int = 4096

    def validate(self) -> "CameraLODRequest":
        if not -90.0 <= float(self.latitude_deg) <= 90.0:
            raise ValueError("latitude_deg must be in [-90, 90]")
        if not -180.0 <= float(self.longitude_deg) <= 180.0:
            raise ValueError("longitude_deg must be in [-180, 180]")
        if not math.isfinite(float(self.altitude_m)) or float(self.altitude_m) <= 0:
            raise ValueError("altitude_m must be finite and positive")
        if int(self.viewport_width_px) < 1 or int(self.viewport_height_px) < 1:
            raise ValueError("viewport dimensions must be positive")
        if not 1.0 <= float(self.vertical_fov_deg) < 179.0:
            raise ValueError("vertical_fov_deg must be in [1, 179)")
        if not math.isfinite(float(self.maximum_screen_error_px)) or float(
            self.maximum_screen_error_px
        ) <= 0:
            raise ValueError("maximum_screen_error_px must be finite and positive")
        if not 0 <= int(self.maximum_level) <= 30:
            raise ValueError("maximum_level must be in [0, 30]")
        if int(self.maximum_tiles) < 1:
            raise ValueError("maximum_tiles must be positive")
        return self


@dataclass(slots=True, frozen=True)
class SelectedTile:
    key: TileKey
    distance_m: float
    screen_error_px: float
    meters_per_sample_approx: float


@dataclass(slots=True, frozen=True)
class LODSelection:
    tiles: tuple[SelectedTile, ...]
    visible_angular_radius_deg: float
    minimum_level: int
    maximum_level: int
    estimated_resident_height_bytes: int

    @property
    def keys(self) -> tuple[TileKey, ...]:
        return tuple(item.key for item in self.tiles)


def camera_footprint_angular_radius_deg(
    *,
    planet_radius_m: float,
    altitude_m: float,
    vertical_fov_deg: float,
    viewport_width_px: int,
    viewport_height_px: int,
) -> float:
    """Conservative circular cap containing a nadir-facing perspective viewport.

    The camera is assumed to point at the planet centre.  We use the larger of the
    horizontal/vertical half-FOV and intersect that cone with the spherical surface.
    If the cone reaches beyond the geometric horizon, the horizon cap is returned.
    """
    radius = float(planet_radius_m)
    altitude = float(altitude_m)
    if radius <= 0 or altitude <= 0:
        raise ValueError("planet radius and camera altitude must be positive")
    aspect = float(viewport_width_px) / float(viewport_height_px)
    half_v = math.radians(float(vertical_fov_deg)) * 0.5
    half_h = math.atan(math.tan(half_v) * aspect)
    theta = max(half_v, half_h)
    camera_radius = radius + altitude
    horizon_theta = math.asin(min(1.0, radius / camera_radius))
    horizon_phi = math.acos(min(1.0, radius / camera_radius))
    if theta >= horizon_theta:
        return math.degrees(horizon_phi)
    argument = np.clip((camera_radius / radius) * math.sin(theta), -1.0, 1.0)
    phi = math.asin(float(argument)) - theta
    return math.degrees(max(phi, 0.0))


def _intersects_cap(key: TileKey, view: np.ndarray, view_radius_rad: float) -> bool:
    center, radius = _tile_bounding_cap(key)
    return _angular_distance(view, center) <= view_radius_rad + radius


def _tile_error(
    *,
    key: TileKey,
    view_direction: np.ndarray,
    camera_position_m: np.ndarray,
    planet_radius_m: float,
    tile_size: int,
    focal_length_px: float,
) -> tuple[float, float, float]:
    center, _radius = _tile_bounding_cap(key)
    surface = center * float(planet_radius_m)
    distance = max(float(np.linalg.norm(camera_position_m - surface)), 1.0)
    mps = approximate_meters_per_sample(
        planet_radius_m, key.level, tile_size
    )
    # Conservative geometric-error proxy.  A later precomputed per-tile error
    # metric can replace this without changing the traversal contract.
    geometric_error_m = 0.5 * mps
    screen_error = geometric_error_m / distance * float(focal_length_px)
    return distance, screen_error, mps


def select_camera_lod(
    *,
    planet_radius_m: float,
    tile_size: int,
    request: CameraLODRequest,
) -> LODSelection:
    """Select a variable-resolution quadtree leaf set for a nadir-facing camera.

    A tile refines only when it intersects the camera footprint and its conservative
    projected geometric error exceeds ``maximum_screen_error_px``.  Consequently
    nearby terrain reaches much deeper levels than distant terrain in the same view.
    """
    req = request.validate()
    radius = float(planet_radius_m)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("planet_radius_m must be finite and positive")
    if int(tile_size) <= 0:
        raise ValueError("tile_size must be positive")
    view = latlon_to_unit(req.latitude_deg, req.longitude_deg)
    camera = view * (radius + float(req.altitude_m))
    view_radius_deg = camera_footprint_angular_radius_deg(
        planet_radius_m=radius,
        altitude_m=req.altitude_m,
        vertical_fov_deg=req.vertical_fov_deg,
        viewport_width_px=req.viewport_width_px,
        viewport_height_px=req.viewport_height_px,
    )
    view_radius = math.radians(view_radius_deg)
    focal = float(req.viewport_height_px) / (
        2.0 * math.tan(math.radians(req.vertical_fov_deg) * 0.5)
    )

    pending = [TileKey(face, 0, 0, 0) for face in reversed(CUBE_FACES)]
    selected: list[SelectedTile] = []
    while pending:
        key = pending.pop()
        if not _intersects_cap(key, view, view_radius):
            continue
        distance, error, mps = _tile_error(
            key=key,
            view_direction=view,
            camera_position_m=camera,
            planet_radius_m=radius,
            tile_size=tile_size,
            focal_length_px=focal,
        )
        should_refine = (
            error > float(req.maximum_screen_error_px)
            and key.level < int(req.maximum_level)
        )
        if should_refine:
            # Only intersecting children are pushed, so deep zoom remains sparse.
            for child in reversed(key.children()):
                if _intersects_cap(child, view, view_radius):
                    pending.append(child)
        else:
            selected.append(
                SelectedTile(
                    key=key,
                    distance_m=distance,
                    screen_error_px=error,
                    meters_per_sample_approx=mps,
                )
            )
            if len(selected) + len(pending) > int(req.maximum_tiles) * 8:
                # Defensive early bound for pathological settings.  The precise
                # selected limit is checked below.
                raise RuntimeError(
                    "LOD traversal exceeded safety budget; increase screen error or reduce maximum level"
                )
    if len(selected) > int(req.maximum_tiles):
        raise RuntimeError(
            f"LOD selection produced {len(selected)} tiles, exceeding maximum_tiles={req.maximum_tiles}"
        )
    selected.sort(key=lambda item: item.key)
    levels = [item.key.level for item in selected]
    # Height-only resident estimate; textures/meshes have separate budgets.
    per_tile_height_bytes = (int(tile_size) + 1) ** 2 * 4
    return LODSelection(
        tiles=tuple(selected),
        visible_angular_radius_deg=float(view_radius_deg),
        minimum_level=min(levels) if levels else 0,
        maximum_level=max(levels) if levels else 0,
        estimated_resident_height_bytes=len(selected) * per_tile_height_bytes,
    )


def parent_chain(key: TileKey) -> tuple[TileKey, ...]:
    """Return root-to-parent fallback tiles useful while children are loading."""
    key.validate()
    parents: list[TileKey] = []
    z, x, y = key.level, key.x, key.y
    while z > 0:
        z -= 1
        x //= 2
        y //= 2
        parents.append(TileKey(key.face, z, x, y))
    parents.reverse()
    return tuple(parents)


def required_fallback_tiles(keys: Iterable[TileKey]) -> tuple[TileKey, ...]:
    """Unique ancestors that can remain visible until selected descendants load."""
    values = {parent for key in keys for parent in parent_chain(key)}
    return tuple(sorted(values))


__all__ = [
    "CameraLODRequest",
    "LODSelection",
    "SelectedTile",
    "camera_footprint_angular_radius_deg",
    "parent_chain",
    "required_fallback_tiles",
    "select_camera_lod",
]
