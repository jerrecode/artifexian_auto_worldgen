from __future__ import annotations

import json

import numpy as np

from worldgen.planet_tiles import PlanetTilePyramid, TilePyramidSpec, latlon_to_tile
from worldgen.vector_tiles import VectorTilePyramid


def _write_world(root) -> None:
    h, w = 8, 16
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    elevation = np.where(lon[None, :] >= 0.0, 1.0, -1.0)
    elevation = np.broadcast_to(elevation, (h, w)).astype(np.float32)
    np.savez(root / "world_arrays.npz", lat=lat, lon=lon, elevation_km=elevation)
    (root / "world.json").write_text(
        json.dumps({"seed": 9001, "astronomy": {"planet": {"radius_earth": 1.0}}}),
        encoding="utf-8",
    )
    features = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "settlement:center",
                "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                "properties": {"feature_class": "settlement", "name": "Centre"},
            },
            {
                "type": "Feature",
                "id": "river:test",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-20.0, 0.0], [20.0, 0.0]],
                },
                "properties": {"feature_class": "river_centerline"},
            },
            {
                "type": "Feature",
                "id": "resource:east",
                "geometry": {"type": "Point", "coordinates": [70.0, 20.0]},
                "properties": {"feature_class": "resource_deposit"},
            },
        ],
    }
    (root / "features.geojson").write_text(json.dumps(features), encoding="utf-8")


def test_vector_tile_contains_only_intersecting_points_and_line_pieces(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(tmp_path, spec=TilePyramidSpec(tile_size=16, maximum_level=8))
    vectors = VectorTilePyramid(pyramid)
    key = latlon_to_tile(0.0, 0.0, 3)
    path = vectors.generate_tile(key)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["type"] == "FeatureCollection"
    assert payload["worldgen_vector_tile"]["key"]["level"] == 3
    classes = [f.get("properties", {}).get("feature_class") for f in payload["features"]]
    assert "settlement" in classes
    assert "river_centerline" in classes
    assert "resource_deposit" not in classes


def test_vector_tile_derives_shoreline_from_global_zero_crossing(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(tmp_path, spec=TilePyramidSpec(tile_size=16, maximum_level=8))
    vectors = VectorTilePyramid(pyramid)
    key = latlon_to_tile(0.0, 0.0, 2)
    payload = json.loads(vectors.generate_tile(key).read_text(encoding="utf-8"))
    shoreline = [
        f for f in payload["features"]
        if f.get("properties", {}).get("feature_class") == "shoreline"
    ]
    assert shoreline
    assert all(f["geometry"]["type"] == "LineString" for f in shoreline)


def test_vector_tile_is_deterministic_and_cached(tmp_path):
    _write_world(tmp_path)
    pyramid = PlanetTilePyramid(tmp_path, spec=TilePyramidSpec(tile_size=16, maximum_level=8))
    vectors = VectorTilePyramid(pyramid)
    key = latlon_to_tile(0.0, 0.0, 4)
    path = vectors.generate_tile(key)
    before = path.read_bytes()
    again = vectors.generate_tile(key)
    assert again == path
    assert again.read_bytes() == before
