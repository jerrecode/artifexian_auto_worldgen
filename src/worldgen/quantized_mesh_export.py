from __future__ import annotations

"""Cesium quantized-mesh-1.0 export from lazy EPSG:4326/TMS terrain.

This module deliberately exports through :mod:`worldgen.geodetic_tiles`; the
internal cube-sphere address space is not a valid quantized-mesh layer projection.
The on-disk `.terrain` payload is the uncompressed little-endian representation.
HTTP delivery may gzip it, as expected by typical terrain services.

For non-Earth generated planets, consumers must use an ellipsoid/globe whose radii
match the exported `ellipsoid_radii_m`. The quantized-mesh format's standard
projection/addressing is geodetic, but a stock Earth globe will of course interpret
ECEF values using its own Earth ellipsoid.
"""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np

from .geodetic_tiles import (
    GeodeticTileKey,
    GeodeticTilePyramid,
    geodetic_tile_geometry,
)
from .terrain_mesh import _surface_triangles


@dataclass(slots=True, frozen=True)
class QuantizedMeshMetadata:
    path: Path
    vertex_count: int
    triangle_count: int
    minimum_height_m: float
    maximum_height_m: float
    bounding_sphere: tuple[float, float, float, float]
    horizon_occlusion_point_scaled: tuple[float, float, float]


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_bytes(
        path,
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
    )


def _zigzag_delta_encode(values: np.ndarray) -> np.ndarray:
    values_i = np.asarray(values, dtype=np.int64).reshape(-1)
    delta = np.empty_like(values_i)
    if values_i.size:
        delta[0] = values_i[0]
        delta[1:] = values_i[1:] - values_i[:-1]
    encoded = (delta << 1) ^ (delta >> 63)
    if np.any(encoded < 0) or np.any(encoded > np.iinfo(np.uint16).max):
        raise ValueError("zig-zag delta value does not fit quantized-mesh uint16")
    return encoded.astype("<u2")


def _first_occurrence_remap(triangles: np.ndarray, vertex_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remap vertices so high-water-mark decoding never references unseen indices."""
    flat = np.asarray(triangles, dtype=np.int64).reshape(-1)
    old_to_new = np.full(int(vertex_count), -1, dtype=np.int64)
    new_to_old: list[int] = []
    mapped = np.empty_like(flat)
    for i, old_value in enumerate(flat.tolist()):
        old = int(old_value)
        value = int(old_to_new[old])
        if value < 0:
            value = len(new_to_old)
            old_to_new[old] = value
            new_to_old.append(old)
        mapped[i] = value
    if len(new_to_old) != int(vertex_count):
        missing = np.where(old_to_new < 0)[0]
        for old in missing.tolist():
            old_to_new[old] = len(new_to_old)
            new_to_old.append(int(old))
    return mapped.reshape(np.asarray(triangles).shape), old_to_new, np.asarray(new_to_old, dtype=np.int64)


def _high_water_encode(indices: np.ndarray, *, use_uint32: bool) -> np.ndarray:
    flat = np.asarray(indices, dtype=np.int64).reshape(-1)
    codes = np.empty_like(flat)
    highest = 0
    for i, value in enumerate(flat.tolist()):
        index = int(value)
        if index > highest:
            raise ValueError("indices are not ordered for high-water-mark encoding")
        code = highest - index
        codes[i] = code
        if code == 0:
            highest += 1
    dtype = "<u4" if use_uint32 else "<u2"
    limit = np.iinfo(np.uint32 if use_uint32 else np.uint16).max
    if np.any(codes < 0) or np.any(codes > limit):
        raise ValueError("high-water-mark code exceeds index component range")
    return codes.astype(dtype)


def _bounding_sphere(world_vertices: np.ndarray) -> tuple[float, float, float, float]:
    points = np.asarray(world_vertices, dtype=np.float64)
    centre = np.mean(points, axis=0)
    radius = float(np.max(np.linalg.norm(points - centre[None, :], axis=1)))
    # Mean-centred sphere is conservative for the supplied points; tiny margin
    # protects against client-side reconstruction/rounding differences.
    radius += max(1e-3, radius * 1e-9)
    return float(centre[0]), float(centre[1]), float(centre[2]), radius


def compute_horizon_occlusion_point(
    world_vertices_m: np.ndarray,
    ellipsoid_radii_m: Sequence[float],
) -> tuple[float, float, float]:
    """Compute Cesium's conservative ellipsoid-scaled horizon occlusion point.

    This follows the published `computeMagnitude` construction: positions are
    transformed to ellipsoid-scaled space; below-ellipsoid magnitudes are clamped to
    one; the maximum required magnitude along a representative tile direction is
    returned. Near a 90-degree root-tile horizon the exact magnitude tends toward
    infinity, so a very large finite conservative magnitude is used to disable
    aggressive culling rather than risk false occlusion.
    """
    points = np.asarray(world_vertices_m, dtype=np.float64).reshape(-1, 3)
    radii = np.asarray(tuple(ellipsoid_radii_m), dtype=np.float64)
    if radii.shape != (3,) or np.any(~np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError("ellipsoid_radii_m must contain three finite positive radii")
    scaled = points / radii[None, :]
    directions = scaled / np.maximum(np.linalg.norm(scaled, axis=1)[:, None], 1e-300)
    representative = np.sum(directions, axis=0)
    rep_norm = float(np.linalg.norm(representative))
    if rep_norm < 1e-12:
        representative = directions[len(directions) // 2]
    else:
        representative /= rep_norm

    maximum = 1.0
    for scaled_position, direction in zip(scaled, directions, strict=True):
        mag2_raw = float(np.dot(scaled_position, scaled_position))
        mag2 = max(1.0, mag2_raw)
        magnitude = math.sqrt(mag2)
        cos_alpha = float(np.clip(np.dot(direction, representative), -1.0, 1.0))
        sin_alpha = float(np.linalg.norm(np.cross(direction, representative)))
        cos_beta = 1.0 / magnitude
        sin_beta = math.sqrt(max(mag2 - 1.0, 0.0)) * cos_beta
        denominator = cos_alpha * cos_beta - sin_alpha * sin_beta
        if denominator <= 1e-12:
            candidate = 1.0e12
        else:
            candidate = 1.0 / denominator
        maximum = max(maximum, candidate)
    point = representative * maximum
    return float(point[0]), float(point[1]), float(point[2])


def _edge_original_indices(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stride = n + 1
    west = np.arange(0, (n + 1) * stride, stride, dtype=np.int64)  # south -> north
    south = np.arange(0, n + 1, dtype=np.int64)  # west -> east
    east = west + n  # south -> north
    north = np.arange(n * stride, n * stride + n + 1, dtype=np.int64)  # west -> east
    return west, south, east, north


def _quantized_mesh_payload(
    geodetic: GeodeticTilePyramid,
    key: GeodeticTileKey,
    *,
    ellipsoid_radii_m: Sequence[float],
) -> tuple[bytes, QuantizedMeshMetadata]:
    n = int(geodetic.spec.tile_size)
    geom = geodetic_tile_geometry(key, n)
    height = np.asarray(geodetic.load_field(key, "elevation_m"), dtype=np.float64)
    if height.shape != (n + 1, n + 1):
        raise RuntimeError("geodetic height tile has unexpected shape")
    radii = np.asarray(tuple(ellipsoid_radii_m), dtype=np.float64)
    # For a spherical generated world xyz*(R+h) is authoritative. Quantized-mesh's
    # ECEF/horizon fields are evaluated against the caller-supplied matching sphere
    # or ellipsoid radii. Non-spherical custom ellipsoid export uses the geodetic
    # surface direction with per-axis radii as an interoperability approximation.
    base_surface = geom.xyz * radii[None, None, :]
    radial = base_surface / np.maximum(np.linalg.norm(base_surface, axis=-1, keepdims=True), 1e-300)
    world = base_surface + radial * height[..., None]
    world_flat = world.reshape(-1, 3)

    hmin = float(np.min(height))
    hmax = float(np.max(height))
    q_axis = np.rint(np.linspace(0.0, 32767.0, n + 1)).astype(np.int64)
    u_grid = np.broadcast_to(q_axis[None, :], (n + 1, n + 1))
    v_grid = np.broadcast_to(q_axis[:, None], (n + 1, n + 1))
    if hmax > hmin:
        h_grid = np.rint((height - hmin) / (hmax - hmin) * 32767.0).astype(np.int64)
    else:
        h_grid = np.zeros_like(height, dtype=np.int64)

    triangles_old = _surface_triangles(world_flat, n)
    triangles, old_to_new, new_to_old = _first_occurrence_remap(
        triangles_old, len(world_flat)
    )
    u = u_grid.reshape(-1)[new_to_old]
    v = v_grid.reshape(-1)[new_to_old]
    qh = h_grid.reshape(-1)[new_to_old]
    world_reordered = world_flat[new_to_old]

    vertex_count = len(new_to_old)
    use_uint32 = vertex_count > 65536
    index_dtype = "<u4" if use_uint32 else "<u2"
    hwm = _high_water_encode(triangles, use_uint32=use_uint32)
    west_old, south_old, east_old, north_old = _edge_original_indices(n)
    edge_arrays = tuple(
        np.asarray(old_to_new[edge], dtype=index_dtype)
        for edge in (west_old, south_old, east_old, north_old)
    )

    sphere = _bounding_sphere(world_reordered)
    hop = compute_horizon_occlusion_point(world_reordered, radii)
    centre = np.mean(world_reordered, axis=0)
    header = struct.pack(
        "<3d2f4d3d",
        float(centre[0]), float(centre[1]), float(centre[2]),
        hmin, hmax,
        sphere[0], sphere[1], sphere[2], sphere[3],
        hop[0], hop[1], hop[2],
    )
    parts = [header, struct.pack("<I", vertex_count)]
    for values in (u, v, qh):
        parts.append(_zigzag_delta_encode(values).tobytes(order="C"))
    raw = b"".join(parts)
    alignment = 4 if use_uint32 else 2
    raw += b"\x00" * ((-len(raw)) % alignment)
    raw += struct.pack("<I", int(np.asarray(triangles).shape[0]))
    raw += hwm.tobytes(order="C")
    for edge in edge_arrays:
        raw += struct.pack("<I", int(len(edge)))
        raw += edge.tobytes(order="C")

    path = quantized_mesh_path(geodetic, key)
    metadata = QuantizedMeshMetadata(
        path=path,
        vertex_count=vertex_count,
        triangle_count=int(np.asarray(triangles).shape[0]),
        minimum_height_m=hmin,
        maximum_height_m=hmax,
        bounding_sphere=sphere,
        horizon_occlusion_point_scaled=hop,
    )
    return raw, metadata


def quantized_mesh_path(geodetic: GeodeticTilePyramid, key: GeodeticTileKey) -> Path:
    key.validate()
    return (
        geodetic.root / "quantized_mesh" / str(key.level) / str(key.x)
        / f"{key.y}.terrain"
    )


def write_quantized_mesh_tile(
    geodetic: GeodeticTilePyramid,
    key: GeodeticTileKey,
    *,
    ellipsoid_radii_m: Sequence[float] | None = None,
    overwrite: bool = False,
) -> QuantizedMeshMetadata:
    key.validate()
    radius = float(geodetic.pyramid.planet_radius_m)
    radii = tuple(ellipsoid_radii_m or (radius, radius, radius))
    path = quantized_mesh_path(geodetic, key)
    raw, metadata = _quantized_mesh_payload(
        geodetic, key, ellipsoid_radii_m=radii
    )
    if overwrite or not path.exists():
        _atomic_bytes(path, raw)
    return metadata


def _parent(key: GeodeticTileKey) -> GeodeticTileKey | None:
    if key.level == 0:
        return None
    return GeodeticTileKey(key.level - 1, key.x // 2, key.y // 2)


def _ancestor_closure(keys: Iterable[GeodeticTileKey]) -> tuple[GeodeticTileKey, ...]:
    values: set[GeodeticTileKey] = set()
    for original in keys:
        key = original.validate()
        while True:
            values.add(key)
            parent = _parent(key)
            if parent is None:
                break
            key = parent
    return tuple(sorted(values))


def write_quantized_mesh_tileset(
    geodetic: GeodeticTilePyramid,
    keys: Iterable[GeodeticTileKey],
    *,
    ellipsoid_radii_m: Sequence[float] | None = None,
    name: str = "Worldgen Terrain",
) -> Path:
    """Materialize requested tiles plus ancestors and write a valid static layer.json."""
    requested = tuple(keys)
    if not requested:
        raise ValueError("at least one geodetic tile key is required")
    closure = _ancestor_closure(requested)
    radius = float(geodetic.pyramid.planet_radius_m)
    radii = tuple(ellipsoid_radii_m or (radius, radius, radius))
    for key in closure:
        write_quantized_mesh_tile(geodetic, key, ellipsoid_radii_m=radii)

    max_level = max(key.level for key in closure)
    availability: list[list[dict[str, int]]] = [[] for _ in range(max_level + 1)]
    for key in closure:
        availability[key.level].append(
            {"startX": key.x, "startY": key.y, "endX": key.x, "endY": key.y}
        )
    for level in availability:
        level.sort(key=lambda item: (item["startY"], item["startX"]))
    version = geodetic.pyramid._source_hash()[:16]
    payload = {
        "name": str(name),
        "description": "Sparse terrain exported by artifexian-auto-worldgen",
        "version": version,
        "format": "quantized-mesh-1.0",
        "scheme": "tms",
        "projection": "EPSG:4326",
        "minzoom": 0,
        "maxzoom": max_level,
        "bounds": [-180.0, -90.0, 180.0, 90.0],
        "tiles": ["{z}/{x}/{y}.terrain?v={version}"],
        "available": availability,
    }
    root = geodetic.root / "quantized_mesh"
    path = root / "layer.json"
    _atomic_json(path, payload)
    return path


__all__ = [
    "QuantizedMeshMetadata",
    "compute_horizon_occlusion_point",
    "quantized_mesh_path",
    "write_quantized_mesh_tile",
    "write_quantized_mesh_tileset",
]
