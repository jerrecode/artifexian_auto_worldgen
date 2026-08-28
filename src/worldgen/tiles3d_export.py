from __future__ import annotations

"""Standards-oriented GLB and OGC 3D Tiles 1.1 terrain export.

The exporter consumes the lazy EPSG:4326/TMS interoperability pyramid but writes
planet-centred Cartesian GLB geometry, so the terrain can represent arbitrary
spherical world radii without pretending the generated planet is WGS84 Earth.

The initial 3D Tiles exporter is an explicit hierarchy for a supplied/materialized
set of geodetic tiles. Implicit-tiling subtree output is intentionally deferred until
availability is backed by either static materialization or the viewer service; a
static tileset must not claim non-existent content is available.
"""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from typing import Iterable, Mapping

import numpy as np

from .geodetic_tiles import (
    GeodeticTileKey,
    GeodeticTilePyramid,
    geodetic_meters_per_sample,
    geodetic_tile_geometry,
)
from .terrain_mesh import _boundary_vertex_indices, _local_basis, _surface_triangles


@dataclass(slots=True, frozen=True)
class GLBTerrainMetadata:
    path: Path
    vertex_count: int
    triangle_count: int
    skirt_vertex_count: int
    origin_ecef_m: np.ndarray
    basis_east_north_up: np.ndarray
    minimum_height_m: float
    maximum_height_m: float


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
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    _atomic_bytes(path, raw)


def _pad4(data: bytes, fill: bytes) -> bytes:
    padding = (-len(data)) % 4
    return data + fill * padding


def _glb_bytes(
    positions: np.ndarray,
    triangles: np.ndarray,
    *,
    transform_matrix: np.ndarray,
) -> bytes:
    pos = np.ascontiguousarray(positions, dtype="<f4")
    tri = np.ascontiguousarray(triangles)
    index_component = 5123 if tri.dtype.itemsize <= 2 else 5125
    index_dtype = "<u2" if index_component == 5123 else "<u4"
    idx = np.ascontiguousarray(tri.reshape(-1), dtype=index_dtype)
    pos_bytes = pos.tobytes(order="C")
    idx_offset = (len(pos_bytes) + 3) & ~3
    binary = pos_bytes + b"\x00" * (idx_offset - len(pos_bytes)) + idx.tobytes(order="C")
    binary = _pad4(binary, b"\x00")

    pmin = np.min(pos, axis=0).astype(float).tolist()
    pmax = np.max(pos, axis=0).astype(float).tolist()
    matrix = np.asarray(transform_matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("transform_matrix must be 4x4")
    gltf = {
        "asset": {"version": "2.0", "generator": "artifexian-auto-worldgen"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "matrix": matrix.T.reshape(-1).tolist()}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_bytes), "target": 34962},
            {
                "buffer": 0,
                "byteOffset": idx_offset,
                "byteLength": int(idx.nbytes),
                "target": 34963,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(len(pos)),
                "type": "VEC3",
                "min": pmin,
                "max": pmax,
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": index_component,
                "count": int(idx.size),
                "type": "SCALAR",
                "min": [int(idx.min()) if idx.size else 0],
                "max": [int(idx.max()) if idx.size else 0],
            },
        ],
    }
    json_chunk = _pad4(
        json.dumps(gltf, separators=(",", ":"), ensure_ascii=True).encode("utf-8"), b" "
    )
    total = 12 + 8 + len(json_chunk) + 8 + len(binary)
    return b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, total),
            struct.pack("<I4s", len(json_chunk), b"JSON"),
            json_chunk,
            struct.pack("<I4s", len(binary), b"BIN\x00"),
            binary,
        )
    )


def _geodetic_mesh(
    geodetic: GeodeticTilePyramid,
    key: GeodeticTileKey,
    *,
    skirt_depth_m: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, float]:
    n = int(geodetic.spec.tile_size)
    geom = geodetic_tile_geometry(key, n)
    elevation = np.asarray(geodetic.load_field(key, "elevation_m"), dtype=np.float64)
    radius = float(geodetic.pyramid.planet_radius_m)
    world = geom.xyz * (radius + elevation)[..., None]
    world_flat = world.reshape(-1, 3)
    centre_index = (n // 2) * (n + 1) + (n // 2)
    centre_direction = geom.xyz.reshape(-1, 3)[centre_index]
    origin = world_flat[centre_index].copy()
    basis = _local_basis(centre_direction)
    local_top = (world_flat - origin) @ basis.T
    surface = _surface_triangles(world_flat, n)
    mps = geodetic_meters_per_sample(radius, key.level, n)
    depth = max(5.0, min(5000.0, 0.08 * mps)) if skirt_depth_m is None else float(skirt_depth_m)
    if not math.isfinite(depth) or depth < 0:
        raise ValueError("skirt_depth_m must be finite and non-negative")
    boundary = _boundary_vertex_indices(n)
    if depth > 0:
        directions = geom.xyz.reshape(-1, 3)[boundary]
        skirt_world = world_flat[boundary] - depth * directions
        skirt_local = (skirt_world - origin) @ basis.T
        positions = np.concatenate((local_top, skirt_local), axis=0)
        base = len(local_top)
        sides = np.empty((2 * len(boundary), 3), dtype=np.int64)
        for i in range(len(boundary)):
            j = (i + 1) % len(boundary)
            ti, tj = int(boundary[i]), int(boundary[j])
            li, lj = base + i, base + j
            sides[2 * i] = (ti, li, tj)
            sides[2 * i + 1] = (tj, li, lj)
        triangles = np.concatenate((surface, sides), axis=0)
        skirt_count = len(boundary)
    else:
        positions = local_top
        triangles = surface
        skirt_count = 0
    index_dtype = np.uint16 if len(positions) <= np.iinfo(np.uint16).max else np.uint32
    return (
        positions.astype(np.float32),
        triangles.astype(index_dtype),
        origin,
        basis,
        int(skirt_count),
        float(np.min(elevation)),
        float(np.max(elevation)),
    )


def geodetic_glb_path(geodetic: GeodeticTilePyramid, key: GeodeticTileKey) -> Path:
    key.validate()
    return (
        geodetic.root / "3dtiles" / "content" / f"z{key.level:02d}"
        / f"x{key.x:08d}" / f"y{key.y:08d}.glb"
    )


def write_geodetic_glb(
    geodetic: GeodeticTilePyramid,
    key: GeodeticTileKey,
    *,
    skirt_depth_m: float | None = None,
    overwrite: bool = False,
) -> GLBTerrainMetadata:
    path = geodetic_glb_path(geodetic, key)
    if path.exists() and not overwrite and skirt_depth_m is None:
        # Rebuild lightweight metadata from the source height tile; the content file
        # is immutable for the same geodetic source/specification.
        elevation = np.asarray(geodetic.load_field(key, "elevation_m"), dtype=np.float64)
        positions, triangles, origin, basis, skirts, hmin, hmax = _geodetic_mesh(
            geodetic, key, skirt_depth_m=None
        )
        return GLBTerrainMetadata(path, len(positions), len(triangles), skirts, origin, basis, hmin, hmax)
    positions, triangles, origin, basis, skirts, hmin, hmax = _geodetic_mesh(
        geodetic, key, skirt_depth_m=skirt_depth_m
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 0] = basis[0]
    transform[:3, 1] = basis[1]
    transform[:3, 2] = basis[2]
    transform[:3, 3] = origin
    _atomic_bytes(path, _glb_bytes(positions, triangles, transform_matrix=transform))
    return GLBTerrainMetadata(
        path=path,
        vertex_count=len(positions),
        triangle_count=len(triangles),
        skirt_vertex_count=skirts,
        origin_ecef_m=origin,
        basis_east_north_up=basis,
        minimum_height_m=hmin,
        maximum_height_m=hmax,
    )


def _parent(key: GeodeticTileKey) -> GeodeticTileKey | None:
    if key.level == 0:
        return None
    return GeodeticTileKey(key.level - 1, key.x // 2, key.y // 2)


def _ancestor_closure(keys: Iterable[GeodeticTileKey]) -> set[GeodeticTileKey]:
    closure: set[GeodeticTileKey] = set()
    for value in keys:
        key = value.validate()
        while True:
            closure.add(key)
            parent = _parent(key)
            if parent is None:
                break
            key = parent
    return closure


def _tile_bounding_sphere(
    geodetic: GeodeticTilePyramid,
    key: GeodeticTileKey,
    *,
    minimum_height_m: float,
    maximum_height_m: float,
) -> list[float]:
    geom = geodetic_tile_geometry(key, 4)
    centre_dir = np.asarray(geom.xyz[2, 2], dtype=np.float64)
    radius_planet = float(geodetic.pyramid.planet_radius_m)
    mid_h = 0.5 * (minimum_height_m + maximum_height_m)
    centre = centre_dir * (radius_planet + mid_h)
    points = []
    flat = geom.xyz.reshape(-1, 3)
    for height in (minimum_height_m, maximum_height_m):
        points.append(flat * (radius_planet + height))
    samples = np.concatenate(points, axis=0)
    bound = float(np.max(np.linalg.norm(samples - centre[None, :], axis=1)))
    # Small numerical/conservatism margin for curvature extrema between the 5x5
    # samples and for float representation in clients.
    bound += max(1.0, 1e-6 * radius_planet)
    return [float(centre[0]), float(centre[1]), float(centre[2]), bound]


def write_explicit_3d_tileset(
    geodetic: GeodeticTilePyramid,
    keys: Iterable[GeodeticTileKey],
    *,
    output_path: str | Path | None = None,
) -> Path:
    """Write an OGC 3D Tiles 1.1 explicit hierarchy for selected terrain content.

    Ancestor nodes are synthesized as bounding/fallback nodes even when only deep
    leaves were requested. Content is attached only to keys supplied by the caller;
    therefore the static tileset never claims unavailable files are present.
    """
    requested = tuple(sorted(set(key.validate() for key in keys)))
    if not requested:
        raise ValueError("at least one geodetic tile key is required")
    closure = _ancestor_closure(requested)
    requested_set = set(requested)
    global_elevation = np.asarray(
        np.load(geodetic.pyramid.source_path, allow_pickle=False)["elevation_km"],
        dtype=np.float64,
    ) * 1000.0
    global_min = float(np.nanmin(global_elevation))
    global_max = float(np.nanmax(global_elevation))
    metadata: dict[GeodeticTileKey, GLBTerrainMetadata] = {}
    for key in requested:
        metadata[key] = write_geodetic_glb(geodetic, key)

    children_by_parent: dict[GeodeticTileKey | None, list[GeodeticTileKey]] = {}
    for key in closure:
        children_by_parent.setdefault(_parent(key), []).append(key)
    for children in children_by_parent.values():
        children.sort()

    content_root = geodetic.root / "3dtiles"

    def node(key: GeodeticTileKey) -> dict[str, object]:
        if key in metadata:
            hmin = metadata[key].minimum_height_m
            hmax = metadata[key].maximum_height_m
        else:
            hmin, hmax = global_min, global_max
        value: dict[str, object] = {
            "boundingVolume": {
                "sphere": _tile_bounding_sphere(
                    geodetic, key, minimum_height_m=hmin, maximum_height_m=hmax
                )
            },
            "geometricError": geodetic_meters_per_sample(
                geodetic.pyramid.planet_radius_m,
                key.level,
                geodetic.spec.tile_size,
            ),
            "refine": "REPLACE",
        }
        if key in requested_set:
            rel = metadata[key].path.relative_to(content_root).as_posix()
            value["content"] = {"uri": rel}
        children = children_by_parent.get(key, [])
        if children:
            value["children"] = [node(child) for child in children]
        return value

    roots = children_by_parent.get(None, [])
    if not roots:
        raise RuntimeError("3D Tiles hierarchy has no root geodetic tiles")
    radius = float(geodetic.pyramid.planet_radius_m)
    root_node: dict[str, object] = {
        "boundingVolume": {"sphere": [0.0, 0.0, 0.0, radius + max(abs(global_min), abs(global_max)) + 10.0]},
        "geometricError": math.pi * radius,
        "refine": "REPLACE",
        "children": [node(root) for root in roots],
    }
    payload = {
        "asset": {"version": "1.1", "generator": "artifexian-auto-worldgen"},
        "geometricError": math.pi * radius,
        "root": root_node,
        "properties": {
            "worldgen": {
                "source_sha256": geodetic.pyramid._source_hash(),
                "projection_source": "EPSG:4326/TMS lazy retiling",
                "static_availability": "content entries exist only for explicitly supplied keys",
            }
        },
    }
    path = Path(output_path) if output_path is not None else content_root / "tileset.json"
    if not path.is_absolute():
        path = geodetic.pyramid.world_root / path
    _atomic_json(path, payload)
    return path


__all__ = [
    "GLBTerrainMetadata",
    "geodetic_glb_path",
    "write_explicit_3d_tileset",
    "write_geodetic_glb",
]
