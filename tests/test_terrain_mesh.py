from __future__ import annotations

import json

import numpy as np

from worldgen.planet_tiles import PlanetTilePyramid, TileKey, TilePyramidSpec
from worldgen.terrain_mesh import build_terrain_mesh, write_terrain_mesh


def _world(root):
    h, w = 16, 32
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elevation = (
        0.8 * np.sin(2 * np.pi * xx / w)
        + 0.35 * np.cos(np.pi * (yy + 0.5) / h)
    ).astype(np.float32)
    np.savez(root / "world_arrays.npz", lat=lat, lon=lon, elevation_km=elevation)
    (root / "world.json").write_text(
        json.dumps({"seed": 1122, "astronomy": {"planet": {"radius_earth": 1.0}}}),
        encoding="utf-8",
    )


def test_mesh_contains_grid_surface_and_perimeter_skirt(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, elevation_detail_strength=0.2),
    )
    mesh = build_terrain_mesh(pyramid, TileKey("px", 4, 7, 6), skirt_depth_m=25.0)
    grid_count = 17 * 17
    perimeter_count = 4 * 16
    assert mesh.grid_vertex_count == grid_count
    assert mesh.skirt_vertex_count == perimeter_count
    assert mesh.positions_local_m.shape == (grid_count + perimeter_count, 3)
    assert mesh.triangle_indices.shape == (2 * 16 * 16 + 2 * perimeter_count, 3)
    assert int(np.max(mesh.triangle_indices)) < len(mesh.positions_local_m)
    assert mesh.skirt_depth_m == 25.0


def test_local_coordinates_reconstruct_planet_scale_ecef(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=20, elevation_detail_strength=0.0),
    )
    mesh = build_terrain_mesh(pyramid, TileKey("pz", 3, 2, 5), skirt_depth_m=0.0)
    ecef = mesh.reconstruct_ecef_m()
    radius = np.linalg.norm(ecef, axis=1)
    assert np.all(np.isfinite(ecef))
    assert abs(float(np.median(radius)) - pyramid.planet_radius_m) < 3000.0
    assert mesh.positions_local_m.dtype == np.float32
    assert mesh.origin_ecef_m.dtype == np.float64


def test_adjacent_same_lod_mesh_top_edges_match_in_ecef(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=18, elevation_detail_strength=0.3),
    )
    a = build_terrain_mesh(pyramid, TileKey("py", 3, 3, 2), skirt_depth_m=0.0)
    b = build_terrain_mesh(pyramid, TileKey("py", 3, 4, 2), skirt_depth_m=0.0)
    ea = a.reconstruct_ecef_m()[: a.grid_vertex_count].reshape(19, 19, 3)
    eb = b.reconstruct_ecef_m()[: b.grid_vertex_count].reshape(19, 19, 3)
    np.testing.assert_allclose(ea[:, -1], eb[:, 0], rtol=0.0, atol=0.75)


def test_mesh_cache_is_sparse_and_atomic_result_is_reusable(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, elevation_detail_strength=0.1),
    )
    key = TileKey("nx", 6, 22, 31)
    path = write_terrain_mesh(pyramid, key)
    assert path.exists()
    sibling = path.with_name("y00000032.npz")
    assert not sibling.exists()
    with np.load(path, allow_pickle=False) as z:
        assert z["positions_local_m"].shape[1] == 3
        assert z["triangle_indices"].shape[1] == 3
    assert write_terrain_mesh(pyramid, key) == path
