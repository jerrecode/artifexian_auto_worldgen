from __future__ import annotations

import json

import numpy as np

from worldgen.geodetic_tiles import (
    GeodeticTileKey,
    GeodeticTilePyramid,
    GeodeticTileSpec,
    geodetic_meters_per_sample,
    geodetic_tile_bounds_deg,
    geodetic_tile_geometry,
    internal_detail_level_for_geodetic,
)
from worldgen.planet_tiles import PlanetTilePyramid, TilePyramidSpec


def _write_world(root) -> None:
    h, w = 8, 16
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elevation = (
        1.2 * np.sin(2.0 * np.pi * xx / w)
        + 0.4 * np.cos(np.pi * (yy + 0.5) / h)
    ).astype(np.float32)
    temperature = (20.0 - 0.25 * np.abs(lat)[:, None] + np.zeros((h, w))).astype(np.float32)
    np.savez(
        root / "world_arrays.npz",
        lat=lat,
        lon=lon,
        elevation_km=elevation,
        annual_temperature_c=temperature,
    )
    (root / "world.json").write_text(
        json.dumps({"seed": 1234, "astronomy": {"planet": {"radius_earth": 1.0}}}),
        encoding="utf-8",
    )


def test_geodetic_root_has_exactly_two_global_hemisphere_tiles():
    west = GeodeticTileKey(0, 0, 0)
    east = GeodeticTileKey(0, 1, 0)
    assert geodetic_tile_bounds_deg(west) == (-180.0, -90.0, 0.0, 90.0)
    assert geodetic_tile_bounds_deg(east) == (0.0, -90.0, 180.0, 90.0)
    west.validate(); east.validate()


def test_geodetic_children_exactly_partition_parent():
    parent = GeodeticTileKey(3, 5, 2)
    children = parent.children()
    pw, ps, pe, pn = geodetic_tile_bounds_deg(parent)
    bounds = [geodetic_tile_bounds_deg(child) for child in children]
    assert min(b[0] for b in bounds) == pw
    assert min(b[1] for b in bounds) == ps
    assert max(b[2] for b in bounds) == pe
    assert max(b[3] for b in bounds) == pn


def test_adjacent_geodetic_tiles_share_exact_vertex_edge_geometry():
    left = GeodeticTileKey(4, 12, 5)
    right = GeodeticTileKey(4, 13, 5)
    gl = geodetic_tile_geometry(left, 32)
    gr = geodetic_tile_geometry(right, 32)
    np.testing.assert_array_equal(gl.latitude_deg[:, -1], gr.latitude_deg[:, 0])
    np.testing.assert_array_equal(gl.longitude_deg[:, -1], gr.longitude_deg[:, 0])
    np.testing.assert_allclose(gl.xyz[:, -1], gr.xyz[:, 0], rtol=0.0, atol=2e-15)


def test_north_south_geodetic_tiles_share_exact_vertex_edge_geometry():
    south = GeodeticTileKey(4, 10, 5)
    north = GeodeticTileKey(4, 10, 6)
    gs = geodetic_tile_geometry(south, 32)
    gn = geodetic_tile_geometry(north, 32)
    np.testing.assert_array_equal(gs.latitude_deg[-1, :], gn.latitude_deg[0, :])
    np.testing.assert_array_equal(gs.longitude_deg[-1, :], gn.longitude_deg[0, :])
    np.testing.assert_allclose(gs.xyz[-1, :], gn.xyz[0, :], rtol=0.0, atol=2e-15)


def test_geodetic_resolution_halves_each_zoom_and_detail_band_tracks_scale():
    radius = 6_371_000.0
    a = geodetic_meters_per_sample(radius, 8, 256)
    b = geodetic_meters_per_sample(radius, 9, 256)
    assert abs(a / b - 2.0) < 1e-14
    assert internal_detail_level_for_geodetic(0) == 0
    assert internal_detail_level_for_geodetic(1) == 0
    assert internal_detail_level_for_geodetic(9) == 8


def test_geodetic_pyramid_is_sparse_cached_and_inherits_arbitrary_fields(tmp_path):
    _write_world(tmp_path)
    cube = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, elevation_detail_strength=0.25, maximum_level=12),
    )
    geo = GeodeticTilePyramid(cube, spec=GeodeticTileSpec(tile_size=16, maximum_level=12))
    key = GeodeticTileKey(7, 87, 41)
    first = geo.generate_tile(key, ("elevation_m", "annual_temperature_c"))
    assert not first.cache_hit
    assert first.fields["elevation_m"].exists()
    assert first.fields["annual_temperature_c"].exists()
    elevation = np.load(first.fields["elevation_m"])
    temperature = np.load(first.fields["annual_temperature_c"])
    assert elevation.shape == temperature.shape == (17, 17)
    second = geo.generate_tile(key, ("elevation_m", "annual_temperature_c"))
    assert second.cache_hit
    field_files = list((geo.root / "fields" / "elevation_m").rglob("*.npy"))
    assert field_files == [first.fields["elevation_m"]]
    manifest = json.loads(geo.manifest_path.read_text(encoding="utf-8"))
    assert manifest["projection"] == "EPSG:4326"
    assert manifest["scheme"] == "tms"
    assert len(manifest["root_tiles"]) == 2
