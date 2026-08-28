from __future__ import annotations

import json

import numpy as np

from worldgen.local_surface import LocalSurfaceGenerator
from worldgen.planet_tiles import PlanetTilePyramid, TileKey, TilePyramidSpec, tile_geometry


def _write_world(root) -> None:
    h, w = 8, 16
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elevation = (1.2 + 0.3 * np.sin(2.0 * np.pi * xx / w)).astype(np.float32)
    annual_t = (16.0 - 0.12 * np.abs(lat)[:, None] + np.zeros((h, w))).astype(np.float32)
    monthly_t = np.empty((12, h, w), dtype=np.float32)
    precipitation = np.empty((12, h, w), dtype=np.float32)
    for month in range(12):
        monthly_t[month] = annual_t + 9.0 * np.cos(2.0 * np.pi * month / 12.0)
        precipitation[month] = 80.0 + 15.0 * np.sin(2.0 * np.pi * month / 12.0)
    wind_u = np.full((12, h, w), 6.0, dtype=np.float32)
    wind_v = np.zeros((12, h, w), dtype=np.float32)
    soil = np.full((h, w), 0.55, dtype=np.float32)
    snow = np.full((h, w), 0.08, dtype=np.float32)
    vegetation = np.full((h, w), 0.62, dtype=np.float32)
    albedo = np.full((h, w), 0.19, dtype=np.float32)
    np.savez(
        root / "world_arrays.npz",
        lat=lat,
        lon=lon,
        elevation_km=elevation,
        annual_temperature_c=annual_t,
        temperature_c_monthly=monthly_t,
        wind_u_monthly=wind_u,
        wind_v_monthly=wind_v,
        precipitation_mm_monthly=precipitation,
        soil_moisture_index=soil,
        snow_persistence=snow,
        vegetation_fraction=vegetation,
        surface_albedo=albedo,
    )
    (root / "world.json").write_text(
        json.dumps({"seed": 5150, "astronomy": {"planet": {"radius_earth": 1.0}}}),
        encoding="utf-8",
    )


def test_local_surface_generates_bounded_fields_and_inherited_boundaries(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, elevation_detail_strength=0.3, maximum_level=8),
    )
    key = TileKey("px", 4, 7, 7)
    generator = LocalSurfaceGenerator(pyramid)
    paths = generator.generate(key)
    assert set(paths) == {
        "soil_moisture_index",
        "snow_persistence",
        "vegetation_fraction",
        "surface_albedo",
        "biome_code",
    }

    geom = tile_geometry(key, 16)
    continuous = {
        "soil_moisture_index": (0.0, 1.0),
        "snow_persistence": (0.0, 1.0),
        "vegetation_fraction": (0.0, 1.0),
        "surface_albedo": (0.02, 0.95),
    }
    for field, (lo, hi) in continuous.items():
        values = np.load(paths[field])
        assert values.shape == (17, 17)
        assert np.all(np.isfinite(values))
        assert float(values.min()) >= lo - 1e-7
        assert float(values.max()) <= hi + 1e-7
        parent = np.asarray(pyramid._sample_source_field(field, geom), dtype=np.float32)
        np.testing.assert_array_equal(values[0, :], parent[0, :])
        np.testing.assert_array_equal(values[-1, :], parent[-1, :])
        np.testing.assert_array_equal(values[:, 0], parent[:, 0])
        np.testing.assert_array_equal(values[:, -1], parent[:, -1])

    biome = np.load(paths["biome_code"])
    assert biome.dtype == np.uint8
    assert set(np.unique(biome)).issubset({0, 1, 2, 3, 4, 5})


def test_local_surface_is_deterministic_and_cache_stable(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, elevation_detail_strength=0.25, maximum_level=8),
    )
    key = TileKey("pz", 3, 3, 2)
    generator = LocalSurfaceGenerator(pyramid)
    first = generator.generate(key)
    bytes_before = {name: path.read_bytes() for name, path in first.items()}
    second = generator.generate(key)
    assert first == second
    for name, path in second.items():
        assert path.read_bytes() == bytes_before[name]
