from __future__ import annotations

"""Render-ready local meshes derived from sparse cube-sphere terrain tiles."""

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile

import numpy as np

from .planet_tiles import (
    PlanetTilePyramid,
    TileKey,
    approximate_meters_per_sample,
    tile_geometry,
)


@dataclass(slots=True, frozen=True)
class TerrainMesh:
    """A tile-local ENU-like mesh with optional downward boundary skirts.

    Positions are relative to ``origin_ecef_m`` so float32 vertices retain metre and
    sub-metre precision even for Earth-sized planets.  ``basis_east_north_up`` maps
    local columns back to planet-centred Cartesian coordinates.
    """

    positions_local_m: np.ndarray
    triangle_indices: np.ndarray
    origin_ecef_m: np.ndarray
    basis_east_north_up: np.ndarray
    grid_vertex_count: int
    skirt_vertex_count: int
    skirt_depth_m: float
    meters_per_sample_approx: float

    def reconstruct_ecef_m(self) -> np.ndarray:
        return (
            self.origin_ecef_m[None, :]
            + np.asarray(self.positions_local_m, dtype=np.float64)
            @ np.asarray(self.basis_east_north_up, dtype=np.float64)
        )


def _local_basis(up: np.ndarray) -> np.ndarray:
    up = np.asarray(up, dtype=np.float64)
    up /= max(float(np.linalg.norm(up)), 1e-300)
    east = np.cross(np.asarray((0.0, 0.0, 1.0)), up)
    if float(np.linalg.norm(east)) < 1e-10:
        east = np.cross(np.asarray((0.0, 1.0, 0.0)), up)
    east /= max(float(np.linalg.norm(east)), 1e-300)
    north = np.cross(up, east)
    north /= max(float(np.linalg.norm(north)), 1e-300)
    # Rows are basis vectors. local @ basis => ECEF delta.
    return np.stack((east, north, up), axis=0)


def _boundary_vertex_indices(n: int) -> np.ndarray:
    """Clockwise perimeter of an (n+1)x(n+1) vertex grid, corners once."""
    stride = n + 1
    top = [i for i in range(0, n + 1)]
    right = [j * stride + n for j in range(1, n + 1)]
    bottom = [n * stride + i for i in range(n - 1, -1, -1)]
    left = [j * stride for j in range(n - 1, 0, -1)]
    return np.asarray(top + right + bottom + left, dtype=np.int64)


def _surface_triangles(world_vertices: np.ndarray, n: int) -> np.ndarray:
    stride = n + 1
    tris = np.empty((2 * n * n, 3), dtype=np.int64)
    k = 0
    for y in range(n):
        row = y * stride
        next_row = (y + 1) * stride
        for x in range(n):
            a = row + x
            b = a + 1
            c = next_row + x
            d = c + 1
            tris[k] = (a, b, c)
            tris[k + 1] = (b, d, c)
            k += 2
    if tris.size:
        tri = tris[0]
        pa, pb, pc = world_vertices[tri]
        outward = float(np.dot(np.cross(pb - pa, pc - pa), pa))
        if outward < 0.0:
            tris[:, [1, 2]] = tris[:, [2, 1]]
    return tris


def build_terrain_mesh(
    pyramid: PlanetTilePyramid,
    key: TileKey,
    *,
    skirt_depth_m: float | None = None,
) -> TerrainMesh:
    """Build one render mesh without generating any sibling tile.

    Same-LOD neighbours share top-surface edge vertices.  Skirts extend the perimeter
    inward along the local radial direction and conceal transient T-junction gaps when
    a renderer displays neighbouring tiles at different LODs.
    """
    key.validate()
    n = int(pyramid.spec.tile_size)
    geom = tile_geometry(key, n)
    elevation = np.asarray(pyramid.load_field(key, "elevation_m"), dtype=np.float64)
    if elevation.shape != (n + 1, n + 1):
        raise RuntimeError(
            f"elevation tile has shape {elevation.shape}, expected {(n + 1, n + 1)}"
        )
    radius = float(pyramid.planet_radius_m)
    world = geom.xyz * (radius + elevation)[..., None]
    world_flat = world.reshape(-1, 3)

    centre_index = (n // 2) * (n + 1) + (n // 2)
    centre_direction = geom.xyz.reshape(-1, 3)[centre_index]
    origin = world_flat[centre_index].copy()
    basis = _local_basis(centre_direction)
    local_top = (world_flat - origin) @ basis.T

    mps = approximate_meters_per_sample(radius, key.level, n)
    if skirt_depth_m is None:
        depth = max(5.0, min(5000.0, 0.08 * mps))
    else:
        depth = float(skirt_depth_m)
    if not np.isfinite(depth) or depth < 0.0:
        raise ValueError("skirt_depth_m must be finite and non-negative")

    surface_tris = _surface_triangles(world_flat, n)
    boundary = _boundary_vertex_indices(n)
    if depth > 0.0:
        directions = geom.xyz.reshape(-1, 3)[boundary]
        skirt_world = world_flat[boundary] - depth * directions
        skirt_local = (skirt_world - origin) @ basis.T
        positions = np.concatenate((local_top, skirt_local), axis=0)
        skirt_base = local_top.shape[0]
        side_tris = np.empty((2 * len(boundary), 3), dtype=np.int64)
        for i in range(len(boundary)):
            j = (i + 1) % len(boundary)
            top_i = int(boundary[i])
            top_j = int(boundary[j])
            low_i = skirt_base + i
            low_j = skirt_base + j
            side_tris[2 * i] = (top_i, low_i, top_j)
            side_tris[2 * i + 1] = (top_j, low_i, low_j)
        triangles = np.concatenate((surface_tris, side_tris), axis=0)
        skirt_count = len(boundary)
    else:
        positions = local_top
        triangles = surface_tris
        skirt_count = 0

    index_dtype = np.uint16 if len(positions) <= np.iinfo(np.uint16).max else np.uint32
    return TerrainMesh(
        positions_local_m=np.asarray(positions, dtype=np.float32),
        triangle_indices=np.asarray(triangles, dtype=index_dtype),
        origin_ecef_m=np.asarray(origin, dtype=np.float64),
        basis_east_north_up=np.asarray(basis, dtype=np.float64),
        grid_vertex_count=(n + 1) * (n + 1),
        skirt_vertex_count=int(skirt_count),
        skirt_depth_m=float(depth),
        meters_per_sample_approx=float(mps),
    )


def mesh_cache_path(pyramid: PlanetTilePyramid, key: TileKey) -> Path:
    key.validate()
    return (
        pyramid.root
        / "meshes"
        / f"z{key.level:02d}"
        / key.face
        / f"x{key.x:08d}"
        / f"y{key.y:08d}.npz"
    )


def write_terrain_mesh(
    pyramid: PlanetTilePyramid,
    key: TileKey,
    *,
    skirt_depth_m: float | None = None,
    overwrite: bool = False,
) -> Path:
    """Atomically cache a render mesh for one addressed terrain tile."""
    path = mesh_cache_path(pyramid, key)
    if path.exists() and not overwrite and skirt_depth_m is None:
        return path
    mesh = build_terrain_mesh(pyramid, key, skirt_depth_m=skirt_depth_m)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(
                f,
                positions_local_m=mesh.positions_local_m,
                triangle_indices=mesh.triangle_indices,
                origin_ecef_m=mesh.origin_ecef_m,
                basis_east_north_up=mesh.basis_east_north_up,
                grid_vertex_count=np.asarray(mesh.grid_vertex_count, dtype=np.int64),
                skirt_vertex_count=np.asarray(mesh.skirt_vertex_count, dtype=np.int64),
                skirt_depth_m=np.asarray(mesh.skirt_depth_m, dtype=np.float64),
                meters_per_sample_approx=np.asarray(
                    mesh.meters_per_sample_approx, dtype=np.float64
                ),
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return path


__all__ = [
    "TerrainMesh",
    "build_terrain_mesh",
    "mesh_cache_path",
    "write_terrain_mesh",
]
