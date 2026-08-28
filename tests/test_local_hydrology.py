from __future__ import annotations

import json

import numpy as np

from worldgen.local_hydrology import (
    LocalHydrologySolver,
    LocalHydrologySpec,
    _patch_geometry,
    _priority_flood_open,
    _resolved_elevation_patch,
)
from worldgen.planet_tiles import PlanetTilePyramid, TileKey, TilePyramidSpec


def _world(root):
    h, w = 18, 36
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elevation = (
        1.3
        + 0.9 * np.cos(2.0 * np.pi * xx / w)
        + 0.35 * np.cos(np.pi * (yy + 0.5) / h)
    ).astype(np.float32)
    runoff = (
        250.0
        + 800.0 * np.clip(np.cos(np.deg2rad(lat))[:, None], 0.0, 1.0)
        + 40.0 * np.sin(2 * np.pi * xx / w)
    ).astype(np.float32)
    rivers = np.zeros((h, w), dtype=bool)
    rivers[:, w // 2] = True
    precipitation = (runoff * 1.7).astype(np.float32)
    annual_temperature = (22.0 - 0.32 * np.abs(lat)[:, None]).astype(np.float32)
    np.savez(
        root / "world_arrays.npz",
        lat=lat,
        lon=lon,
        elevation_km=elevation,
        runoff_mm_year=runoff,
        annual_precipitation_mm=precipitation,
        annual_temperature_c=annual_temperature,
        rivers=rivers,
    )
    (root / "world.json").write_text(
        json.dumps({"seed": 1234, "astronomy": {"planet": {"radius_earth": 1.0}}}),
        encoding="utf-8",
    )


def test_open_priority_flood_never_wraps_or_changes_perimeter_seed_heights():
    z = np.array(
        [
            [9.0, 8.0, 7.0, 6.0, 5.0],
            [8.0, 4.0, 4.0, 4.0, 6.0],
            [7.0, 4.0, 1.0, 4.0, 7.0],
            [6.0, 4.0, 4.0, 4.0, 8.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
        ]
    )
    filled = _priority_flood_open(z, np.zeros_like(z, dtype=bool), epsilon_m=0.01)
    np.testing.assert_array_equal(filled[0], z[0])
    np.testing.assert_array_equal(filled[-1], z[-1])
    np.testing.assert_array_equal(filled[:, 0], z[:, 0])
    np.testing.assert_array_equal(filled[:, -1], z[:, -1])
    assert filled[2, 2] > z[2, 2]


def test_halo_resolved_elevation_core_matches_authoritative_tile(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=20, elevation_detail_strength=0.35),
    )
    key = TileKey("px", 4, 7, 6)
    halo = 5
    geom = _patch_geometry(key, pyramid.spec.tile_size, halo)
    patch = _resolved_elevation_patch(pyramid, key, geom)
    core = patch[halo : halo + 21, halo : halo + 21]
    tile = np.asarray(pyramid.load_field(key, "elevation_m"), dtype=float)
    np.testing.assert_allclose(core, tile, rtol=0.0, atol=2e-4)


def test_local_hydrology_returns_bounded_d8_and_inherited_runoff(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=24, elevation_detail_strength=0.4),
    )
    solver = LocalHydrologySolver(
        pyramid, spec=LocalHydrologySpec(halo_cells=8, stream_quantile=0.96)
    )
    result = solver.solve(TileKey("px", 5, 15, 14))
    expected = (25, 25)
    assert result.filled_elevation_m.shape == expected
    assert result.flow_direction_d8.shape == expected
    assert result.runoff_mm_year.shape == expected
    assert result.drainage_area_km2.shape == expected
    assert result.discharge_index.shape == expected
    assert result.streams.shape == expected
    assert np.all((result.flow_direction_d8 >= -1) & (result.flow_direction_d8 <= 7))
    assert np.all(result.runoff_mm_year >= 0.0)
    assert np.all(result.drainage_area_km2 >= 0.0)
    assert np.all((result.discharge_index >= 0.0) & (result.discharge_index <= 1.0 + 1e-6))
    assert result.metadata["runoff_semantics"] == "inherited global runoff_mm_year"
    assert result.metadata["flow_direction_semantics"]["not_global_flow_to"] is True


def test_inherited_major_river_cells_are_always_preserved_as_streams(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(tmp_path, spec=TilePyramidSpec(tile_size=32))
    solver = LocalHydrologySolver(pyramid, spec=LocalHydrologySpec(halo_cells=6))
    # Root +X includes longitude around 0 degrees where the synthetic major river lies.
    result = solver.solve(TileKey("px", 0, 0, 0))
    inherited = np.asarray(result.inherited_major_river, dtype=bool)
    assert np.any(inherited)
    assert np.all(np.asarray(result.streams, dtype=bool)[inherited])


def test_local_hydrology_cache_is_sparse_and_reusable(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, elevation_detail_strength=0.2),
    )
    solver = LocalHydrologySolver(
        pyramid, spec=LocalHydrologySpec(halo_cells=4, stream_quantile=0.95)
    )
    key = TileKey("pz", 7, 61, 70)
    first = solver.solve(key)
    assert solver._metadata_path(key).exists()
    sibling = TileKey("pz", 7, 62, 70)
    assert not solver._metadata_path(sibling).exists()
    second = solver.solve(key)
    np.testing.assert_array_equal(first.discharge_index, second.discharge_index)
    np.testing.assert_array_equal(first.flow_direction_d8, second.flow_direction_d8)
