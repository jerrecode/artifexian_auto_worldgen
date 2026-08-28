from __future__ import annotations

import json
import struct

import numpy as np

from worldgen.geodetic_tiles import (
    GeodeticTileKey,
    GeodeticTilePyramid,
    GeodeticTileSpec,
)
from worldgen.planet_tiles import PlanetTilePyramid, TilePyramidSpec
from worldgen.tiles3d_export import write_explicit_3d_tileset, write_geodetic_glb


def _write_world(root) -> None:
    h, w = 8, 16
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elevation = (
        0.8 * np.sin(2.0 * np.pi * xx / w) + 0.3 * np.cos(np.pi * (yy + 0.5) / h)
    ).astype(np.float32)
    np.savez(root / "world_arrays.npz", lat=lat, lon=lon, elevation_km=elevation)
    (root / "world.json").write_text(
        json.dumps({"seed": 71, "astronomy": {"planet": {"radius_earth": 1.0}}}),
        encoding="utf-8",
    )


def _pyramids(tmp_path):
    cube = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, elevation_detail_strength=0.2, maximum_level=8),
    )
    geo = GeodeticTilePyramid(cube, spec=GeodeticTileSpec(tile_size=16, maximum_level=8))
    return cube, geo


def test_glb_export_has_valid_container_chunks_and_local_transform(tmp_path):
    _write_world(tmp_path)
    _cube, geo = _pyramids(tmp_path)
    key = GeodeticTileKey(3, 7, 3)
    result = write_geodetic_glb(geo, key)
    raw = result.path.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", raw, 0)
    assert magic == b"glTF"
    assert version == 2
    assert total == len(raw)
    json_len, json_type = struct.unpack_from("<I4s", raw, 12)
    assert json_type == b"JSON"
    payload = json.loads(raw[20 : 20 + json_len].decode("utf-8"))
    assert payload["asset"]["version"] == "2.0"
    assert payload["meshes"][0]["primitives"][0]["mode"] == 4
    matrix = np.asarray(payload["nodes"][0]["matrix"], dtype=float).reshape(4, 4).T
    np.testing.assert_allclose(matrix[:3, 3], result.origin_ecef_m, rtol=0, atol=1e-6)
    assert result.vertex_count > 17 * 17
    assert result.skirt_vertex_count > 0
    assert result.triangle_count > 2 * 16 * 16


def test_explicit_3d_tileset_synthesizes_ancestors_but_only_advertises_requested_content(tmp_path):
    _write_world(tmp_path)
    _cube, geo = _pyramids(tmp_path)
    deep = GeodeticTileKey(3, 7, 3)
    sibling = GeodeticTileKey(3, 6, 3)
    path = write_explicit_3d_tileset(geo, (deep, sibling))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["asset"]["version"] == "1.1"
    assert payload["root"]["refine"] == "REPLACE"
    assert payload["root"]["boundingVolume"]["sphere"][3] > geo.pyramid.planet_radius_m

    content_uris: list[str] = []
    node_count = 0

    def walk(node):
        nonlocal node_count
        node_count += 1
        if "content" in node:
            content_uris.append(node["content"]["uri"])
        for child in node.get("children", []):
            walk(child)

    walk(payload["root"])
    assert node_count > 3  # synthetic root + ancestry + requested leaves
    assert len(content_uris) == 2
    assert all((path.parent / uri).exists() for uri in content_uris)
    assert all(uri.endswith(".glb") for uri in content_uris)


def test_3d_tiles_export_is_deterministic_for_same_key_set(tmp_path):
    _write_world(tmp_path)
    _cube, geo = _pyramids(tmp_path)
    keys = (GeodeticTileKey(2, 2, 1), GeodeticTileKey(2, 3, 1))
    path = write_explicit_3d_tileset(geo, keys)
    first = path.read_bytes()
    again = write_explicit_3d_tileset(geo, reversed(keys))
    assert again == path
    assert again.read_bytes() == first
