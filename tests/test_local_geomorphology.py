from __future__ import annotations

import json

import numpy as np

from worldgen.local_geomorphology import LocalGeomorphologySolver
from worldgen.planet_tiles import PlanetTilePyramid, TileKey, TilePyramidSpec, tile_geometry
from worldgen.river_constraints import RiverConstraintGenerator


def _write_world(root) -> None:
    h, w = 8, 16
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elevation = (
        1.4 + 0.45 * np.sin(2.0 * np.pi * xx / w) + 0.15 * yy / h
    ).astype(np.float32)
    rivers = np.ones((h, w), dtype=np.float32)
    stream_order = np.full((h, w), 4, dtype=np.int16)
    discharge = np.full((h, w), 0.75, dtype=np.float32)
    width = np.full((h, w), 0.65, dtype=np.float32)
    runoff = np.full((h, w), 700.0, dtype=np.float32)
    annual_precip = np.full((h, w), 1200.0, dtype=np.float32)
    annual_temp = np.full((h, w), 14.0, dtype=np.float32)
    np.savez(
        root / "world_arrays.npz",
        lat=lat,
        lon=lon,
        elevation_km=elevation,
        rivers=rivers,
        stream_order=stream_order,
        discharge_index=discharge,
        river_width_proxy=width,
        runoff_mm_year=runoff,
        annual_precipitation_mm=annual_precip,
        annual_temperature_c=annual_temp,
    )
    (root / "world.json").write_text(
        json.dumps({"seed": 8080, "astronomy": {"planet": {"radius_earth": 1.0}}}),
        encoding="utf-8",
    )


def test_river_constraints_preserve_global_major_channel_authority(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, elevation_detail_strength=0.25, maximum_level=8),
    )
    key = TileKey("px", 4, 7, 6)
    result = RiverConstraintGenerator(pyramid).generate(key)
    geom = tile_geometry(key, 16)
    parent_elevation = np.asarray(pyramid._sample_source_field("elevation_m", geom), dtype=float)
    assert np.all(result.major_river_mask)
    assert np.all(result.parent_stream_order >= 3)
    assert np.all(result.constraint_strength > 0)
    assert np.all(np.asarray(result.channel_floor_m) <= parent_elevation + 1e-5)
    ancestry = result.metadata["quadtree_ancestry_root_to_parent"]
    assert len(ancestry) == key.level
    assert ancestry[0]["level"] == 0
    assert ancestry[-1]["level"] == key.level - 1


def test_local_geomorphology_is_watertight_mass_accounted_and_river_constrained(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, elevation_detail_strength=0.35, maximum_level=8),
    )
    key = TileKey("px", 4, 7, 6)
    base = np.asarray(pyramid.load_field(key, "elevation_m"))
    result = LocalGeomorphologySolver(pyramid).solve(key)
    evolved = np.asarray(result.elevation_m)

    assert evolved.shape == base.shape == (17, 17)
    np.testing.assert_array_equal(evolved[0, :], base[0, :])
    np.testing.assert_array_equal(evolved[-1, :], base[-1, :])
    np.testing.assert_array_equal(evolved[:, 0], base[:, 0])
    np.testing.assert_array_equal(evolved[:, -1], base[:, -1])
    assert np.all(np.asarray(result.erosion_m) >= 0.0)
    assert np.all(np.asarray(result.deposition_m) >= 0.0)
    assert np.isfinite(np.asarray(result.procedural_detail_m)).all()
    assert np.isfinite(np.asarray(result.procedural_coherence)).all()
    assert np.all((np.asarray(result.procedural_coherence) >= 0.0) & (np.asarray(result.procedural_coherence) <= 1.0))
    assert np.any(np.abs(np.asarray(result.procedural_detail_m)[1:-1, 1:-1]) > 0.0)
    assert np.allclose(np.asarray(result.procedural_detail_m)[0, :], 0.0)
    assert np.allclose(np.asarray(result.procedural_detail_m)[-1, :], 0.0)
    assert np.allclose(np.asarray(result.procedural_detail_m)[:, 0], 0.0)
    assert np.allclose(np.asarray(result.procedural_detail_m)[:, -1], 0.0)
    assert np.any(np.asarray(result.major_river_constraint)[1:-1, 1:-1])
    assert np.any(evolved[1:-1, 1:-1] < base[1:-1, 1:-1])
    assert float(result.metadata["sediment_closure_relative"]) < 1e-12
    eroded = float(result.metadata["eroded_sediment_volume_m3"])
    deposited = float(result.metadata["deposited_sediment_volume_m3"])
    exported = float(result.metadata["exported_sediment_volume_m3"])
    assert eroded >= 0 and deposited >= 0 and exported >= 0
    assert abs(eroded - deposited - exported) <= max(eroded, 1.0) * 1e-12
    assert result.metadata["procedural_semantics"].startswith("zero-area-mean")
    assert result.metadata["procedural_octaves_executed"] >= 1


def test_local_geomorphology_cache_is_deterministic(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, elevation_detail_strength=0.20, maximum_level=8),
    )
    key = TileKey("py", 3, 2, 4)
    solver = LocalGeomorphologySolver(pyramid)
    first = solver.solve(key)
    first_elevation = np.asarray(first.elevation_m).copy()
    second = solver.solve(key)
    np.testing.assert_array_equal(np.asarray(second.elevation_m), first_elevation)
    assert second.metadata == first.metadata
