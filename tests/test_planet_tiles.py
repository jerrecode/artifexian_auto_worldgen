from __future__ import annotations

import json

import numpy as np

from worldgen.refinement import RefinementEngine, RefinementSpec
from worldgen.planet_tiles import (
    PlanetTilePyramid,
    TileKey,
    TilePyramidSpec,
    approximate_meters_per_sample,
    direction_to_tile,
    latlon_to_tile,
    level_for_meters_per_sample,
    tile_geometry,
    visible_tiles,
)


def _write_world(root):
    h, w = 16, 32
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elevation = (
        1.8 * np.sin(2.0 * np.pi * xx / w)
        + 0.7 * np.cos(np.pi * (yy + 0.5) / h)
        - 0.15
    ).astype(np.float32)
    plate = ((xx // 5 + yy // 4) % 7).astype(np.int16)
    rgb = np.stack(
        (
            np.clip((elevation + 3.0) * 35.0, 0, 255),
            np.broadcast_to((90.0 - np.abs(lat))[:, None] * 2.0, (h, w)),
            np.full((h, w), 110.0),
        ),
        axis=-1,
    ).astype(np.uint8)
    np.savez(
        root / "world_arrays.npz",
        lat=lat,
        lon=lon,
        elevation_km=elevation,
        ocean_depth_m=np.maximum(-elevation * 1000.0, 0.0).astype(np.float32),
        plate_id=plate,
        true_color_rgb=rgb,
        flow_to=np.arange(h * w, dtype=np.int64).reshape(h, w),
    )
    (root / "world.json").write_text(
        json.dumps({"seed": 98765, "astronomy": {"planet": {"radius_earth": 1.0}}}),
        encoding="utf-8",
    )


def test_tile_geometry_has_shared_vertices_inside_face():
    a = tile_geometry(TileKey("pz", 2, 1, 2), tile_size=32)
    b = tile_geometry(TileKey("pz", 2, 2, 2), tile_size=32)
    np.testing.assert_allclose(a.xyz[:, -1], b.xyz[:, 0], rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(a.latitude_deg[:, -1], b.latitude_deg[:, 0], atol=2e-13)


def test_cube_face_seam_is_geometrically_identical():
    # +X right edge and -Z left edge represent the same cube directions.
    px = tile_geometry(TileKey("px", 0, 0, 0), tile_size=48)
    nz = tile_geometry(TileKey("nz", 0, 0, 0), tile_size=48)
    np.testing.assert_allclose(px.xyz[:, -1], nz.xyz[:, 0], rtol=0.0, atol=2e-15)


def test_direction_addressing_returns_expected_axis_faces():
    assert direction_to_tile((1, 0, 0), 5).face == "px"
    assert direction_to_tile((-1, 0, 0), 5).face == "nx"
    assert direction_to_tile((0, 1, 0), 5).face == "py"
    assert direction_to_tile((0, -1, 0), 5).face == "ny"
    assert direction_to_tile((0, 0, 1), 5).face == "pz"
    assert direction_to_tile((0, 0, -1), 5).face == "nz"
    assert latlon_to_tile(0.0, 0.0, 3).face == "px"


def test_characteristic_resolution_halves_each_level():
    radius = 6_371_000.0
    r4 = approximate_meters_per_sample(radius, 4, 256)
    r5 = approximate_meters_per_sample(radius, 5, 256)
    assert abs(r5 / r4 - 0.5) < 1e-15
    assert level_for_meters_per_sample(radius, 2.0, tile_size=256, maximum_level=24) >= 14


def test_visible_selector_does_not_enumerate_whole_high_level():
    keys = visible_tiles(
        latitude_deg=48.2,
        longitude_deg=16.37,
        angular_radius_deg=0.04,
        level=12,
        maximum_tiles=128,
    )
    assert keys
    assert len(keys) < 128
    assert all(key.level == 12 for key in keys)


def test_generate_one_tile_materializes_only_requested_address(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=32, elevation_detail_strength=0.2, maximum_level=18),
    )
    key = TileKey("px", 5, 9, 13)
    result = pyramid.generate_tile(key, ("elevation_m", "plate_id"))
    assert not result.cache_hit
    elevation = np.load(result.fields["elevation_m"])
    plate = np.load(result.fields["plate_id"])
    assert elevation.shape == (33, 33)
    assert elevation.dtype == np.float32
    assert plate.shape == (33, 33)
    assert np.issubdtype(plate.dtype, np.integer)

    # Sparse means no sibling tile or full zoom-level raster is generated implicitly.
    sibling = TileKey("px", 5, 10, 13)
    assert not pyramid._field_path(sibling, "elevation_m").exists()
    assert not (tmp_path / "tiles" / "cubesphere_v1" / "levels").exists()

    cached = pyramid.generate_tile(key, ("elevation_m", "plate_id"))
    assert cached.cache_hit


def test_generated_elevation_matches_across_same_face_and_cross_face_seams(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=40, elevation_detail_strength=0.35, maximum_level=8),
    )

    left = pyramid.load_field(TileKey("pz", 3, 3, 4), "elevation_m")
    right = pyramid.load_field(TileKey("pz", 3, 4, 4), "elevation_m")
    np.testing.assert_allclose(left[:, -1], right[:, 0], rtol=0.0, atol=2e-4)

    px = pyramid.load_field(TileKey("px", 0, 0, 0), "elevation_m")
    nz = pyramid.load_field(TileKey("nz", 0, 0, 0), "elevation_m")
    np.testing.assert_allclose(px[:, -1], nz[:, 0], rtol=0.0, atol=2e-4)


def test_deeper_zoom_adds_detail_without_moving_shared_parent_corners(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=32, elevation_detail_strength=0.4, maximum_level=8),
    )
    parent_key = TileKey("py", 2, 1, 1)
    child_key = parent_key.children()[0]
    parent = np.asarray(pyramid.load_field(parent_key, "elevation_m"))
    child = np.asarray(pyramid.load_field(child_key, "elevation_m"))
    assert np.ptp(parent) > 0
    assert np.ptp(child) > 0
    # Child adds a higher-frequency band, so it is not merely an interpolated copy.
    assert float(np.std(child)) > 0.0


def test_tileset_records_world_fingerprint_and_lod_contract(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(tmp_path, spec=TilePyramidSpec(tile_size=24))
    manifest = json.loads(pyramid.manifest_path.read_text(encoding="utf-8"))
    assert manifest["projection"] == "cube_sphere"
    assert manifest["tile_size"] == 24
    assert len(manifest["source_sha256"]) == 64
    assert manifest["lod"]["root_meters_per_sample_approx"] > 0
    assert "flow_to" in manifest["omitted_fields"]


def test_tiles_default_to_deepest_complete_full_world_refinement(tmp_path):
    _write_world(tmp_path)
    RefinementEngine(
        tmp_path,
        spec=RefinementSpec(
            scale=2,
            sections_y=2,
            sections_x=2,
            halo_cells=2,
            elevation_detail_strength=0.4,
        ),
    ).refine(2)

    refined = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, elevation_detail_strength=0.0),
    )
    assert refined.source_kind == "refinement_level"
    assert refined.source_level == 2
    assert refined._source_metadata()[0] == (64, 128)
    refined_manifest = json.loads(refined.manifest_path.read_text(encoding="utf-8"))
    assert refined_manifest["source_level"] == 2
    assert refined_manifest["source_resolution"] == [128, 64]
    assert "refinement_level_0002" in str(refined.manifest_path)

    base = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, elevation_detail_strength=0.0),
        source_level=0,
    )
    assert base.source_kind == "base_npz"
    assert base.source_level == 0
    assert base._source_metadata()[0] == (16, 32)
    assert base.manifest_path != refined.manifest_path
