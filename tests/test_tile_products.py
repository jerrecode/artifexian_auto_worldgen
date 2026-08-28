from __future__ import annotations

import json

import numpy as np

from worldgen.planet_tiles import PlanetTilePyramid, TileKey, TilePyramidSpec
from worldgen.tile_products import TileProductExporter


def _world(root):
    h, w = 12, 24
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elevation = (
        1.25 * np.sin(2.0 * np.pi * xx / w)
        + 0.5 * np.cos(np.pi * (yy + 0.5) / h)
        - 0.1
    ).astype(np.float32)
    temp = (15.0 - 0.28 * np.abs(lat)[:, None] + 0.1 * np.cos(2 * np.pi * xx / w)).astype(np.float32)
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip((elevation + 2.0) * 55.0, 0, 255)
    rgb[..., 1] = np.clip((temp + 30.0) * 4.0, 0, 255)
    rgb[..., 2] = np.clip(100.0 + 40.0 * np.sin(2 * np.pi * xx / w), 0, 255)
    monthly = np.stack([temp + 4.0 * np.cos(2 * np.pi * i / 12.0) for i in range(12)]).astype(np.float32)
    np.savez(
        root / "world_arrays.npz",
        lat=lat,
        lon=lon,
        elevation_km=elevation,
        annual_temperature_c=temp,
        temperature_c_monthly=monthly,
        true_color_rgb=rgb,
    )
    (root / "world.json").write_text(
        json.dumps({"seed": 999, "astronomy": {"planet": {"radius_earth": 1.0}}}),
        encoding="utf-8",
    )


def test_uint16_height_encoding_is_global_and_reversible_within_one_code(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=24, elevation_detail_strength=0.4, maximum_level=14),
    )
    products = TileProductExporter(pyramid)
    key = TileKey("px", 6, 25, 31)
    elevation = np.asarray(pyramid.load_field(key, "elevation_m"), dtype=float)
    encoded = np.asarray(products.height_u16(key))
    decoded = products.height_encoding.decode(encoded)
    assert encoded.dtype == np.uint16
    assert np.max(np.abs(decoded - elevation)) <= products.height_encoding.meters_per_code * 0.51
    assert products.encoding_path.exists()


def test_height_codes_share_exact_same_lod_edge(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=20, elevation_detail_strength=0.35),
    )
    products = TileProductExporter(pyramid)
    a = np.asarray(products.height_u16(TileKey("pz", 4, 7, 8)))
    b = np.asarray(products.height_u16(TileKey("pz", 4, 8, 8)))
    np.testing.assert_array_equal(a[:, -1], b[:, 0])


def test_cross_face_height_encoding_has_no_quantization_seam(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=18, elevation_detail_strength=0.25),
    )
    products = TileProductExporter(pyramid)
    px = np.asarray(products.height_u16(TileKey("px", 0, 0, 0)))
    nz = np.asarray(products.height_u16(TileKey("nz", 0, 0, 0)))
    np.testing.assert_array_equal(px[:, -1], nz[:, 0])


def test_true_color_product_is_bilinear_and_png_exportable(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(tmp_path, spec=TilePyramidSpec(tile_size=16))
    products = TileProductExporter(pyramid)
    key = TileKey("py", 3, 3, 4)
    rgb = np.asarray(products.inherited_true_color_rgb(key))
    assert rgb.shape == (17, 17, 3)
    assert rgb.dtype == np.uint8
    path = products.true_color_png(key)
    assert path.exists()
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_height_png_uses_global_u16_codes(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(tmp_path, spec=TilePyramidSpec(tile_size=16))
    products = TileProductExporter(pyramid)
    key = TileKey("ny", 2, 1, 2)
    path = products.height_png(key)
    assert path.exists()
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert products.height_encoding.meters_per_code > 0


def test_diagnostic_terrain_temperature_image_uses_local_downscaling(tmp_path):
    _world(tmp_path)
    pyramid = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, elevation_detail_strength=0.5),
    )
    products = TileProductExporter(pyramid)
    key = TileKey("px", 5, 17, 12)
    rgb = np.asarray(products.terrain_temperature_rgb(key))
    assert rgb.shape == (17, 17, 3)
    assert rgb.dtype == np.uint8
    assert np.ptp(rgb.astype(np.int16)) > 0
    assert products.terrain_temperature_png(key).exists()
