from __future__ import annotations

import json

import numpy as np

from worldgen.local_downscaling import LocalClimateSpec, LocalTileDownscaler
from worldgen.planet_tiles import PlanetTilePyramid, TileKey, TilePyramidSpec, tile_geometry


def _world(root):
    h, w = 16, 32
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elevation = (
        1.1 * np.sin(2.0 * np.pi * xx / w)
        + 0.45 * np.cos(np.pi * (yy + 0.5) / h)
    ).astype(np.float32)
    annual = (18.0 - 0.33 * np.abs(lat)[:, None] + 0.15 * np.cos(2 * np.pi * xx / w)).astype(np.float32)
    monthly = np.stack(
        [annual + 7.0 * np.cos(2.0 * np.pi * month / 12.0) for month in range(12)],
        axis=0,
    ).astype(np.float32)
    np.savez(
        root / "world_arrays.npz",
        lat=lat,
        lon=lon,
        elevation_km=elevation,
        annual_temperature_c=annual,
        temperature_c_monthly=monthly,
    )
    (root / "world.json").write_text(
        json.dumps({"seed": 2468, "astronomy": {"planet": {"radius_earth": 1.0}}}),
        encoding="utf-8",
    )


def test_downscaled_temperature_matches_lapse_rate_relation(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=24, elevation_detail_strength=0.5),
    )
    key = TileKey("px", 5, 14, 12)
    spec = LocalClimateSpec(lapse_rate_k_per_km=6.5)
    down = LocalTileDownscaler(pyramid, climate_spec=spec)
    actual = np.asarray(down.annual_temperature_c(key))

    geom = tile_geometry(key, pyramid.spec.tile_size)
    base_temp = np.asarray(pyramid._sample_source_field("annual_temperature_c", geom), dtype=float)
    base_elev = np.asarray(pyramid._sample_source_field("elevation_m", geom), dtype=float)
    resolved_elev = np.asarray(pyramid.load_field(key, "elevation_m"), dtype=float)
    expected = base_temp - 6.5 * (resolved_elev - base_elev) / 1000.0
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-5)


def test_monthly_downscaling_preserves_inherited_seasonal_delta(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=18, elevation_detail_strength=0.35),
    )
    key = TileKey("pz", 4, 6, 5)
    down = LocalTileDownscaler(pyramid)
    monthly = np.asarray(down.monthly_temperature_c(key))
    annual = np.asarray(down.annual_temperature_c(key))
    assert monthly.shape == (12, 19, 19)
    assert annual.shape == (19, 19)
    # Terrain correction is identical in each month, so the local Jan-Jul
    # difference is exactly the inherited Jan-Jul difference.
    geom = tile_geometry(key, pyramid.spec.tile_size)
    inherited = np.asarray(
        pyramid._sample_source_field("temperature_c_monthly", geom), dtype=float
    )
    np.testing.assert_allclose(monthly[0] - monthly[6], inherited[0] - inherited[6], atol=3e-5)


def test_downscaled_temperature_is_continuous_across_same_face_seam(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=20, elevation_detail_strength=0.4),
    )
    down = LocalTileDownscaler(pyramid)
    a = np.asarray(down.annual_temperature_c(TileKey("py", 3, 3, 2)))
    b = np.asarray(down.annual_temperature_c(TileKey("py", 3, 4, 2)))
    np.testing.assert_allclose(a[:, -1], b[:, 0], rtol=0.0, atol=3e-5)


def test_downscaled_temperature_is_continuous_across_cube_face_seam(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=20, elevation_detail_strength=0.4),
    )
    down = LocalTileDownscaler(pyramid)
    px = np.asarray(down.annual_temperature_c(TileKey("px", 0, 0, 0)))
    nz = np.asarray(down.annual_temperature_c(TileKey("nz", 0, 0, 0)))
    np.testing.assert_allclose(px[:, -1], nz[:, 0], rtol=0.0, atol=3e-5)


def test_downscaling_cache_is_sparse(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, elevation_detail_strength=0.2),
    )
    down = LocalTileDownscaler(pyramid)
    key = TileKey("nx", 7, 45, 50)
    first = down.annual_temperature_c(key)
    assert first.shape == (17, 17)
    path = down._path(key, "annual_temperature_c")
    assert path.exists()
    sibling = TileKey("nx", 7, 46, 50)
    assert not down._path(sibling, "annual_temperature_c").exists()
    second = down.annual_temperature_c(key)
    np.testing.assert_array_equal(first, second)
