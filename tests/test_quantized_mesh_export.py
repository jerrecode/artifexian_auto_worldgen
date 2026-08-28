from __future__ import annotations

import json
import math
import struct

import numpy as np

from worldgen.geodetic_tiles import GeodeticTileKey, GeodeticTilePyramid, GeodeticTileSpec
from worldgen.planet_tiles import PlanetTilePyramid, TilePyramidSpec
from worldgen.quantized_mesh_export import (
    compute_horizon_occlusion_point,
    write_quantized_mesh_tile,
    write_quantized_mesh_tileset,
)


def _write_world(root) -> None:
    h, w = 8, 16
    lat = 90.0 - (np.arange(h) + 0.5) * 180.0 / h
    lon = -180.0 + (np.arange(w) + 0.5) * 360.0 / w
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    elevation = (
        0.7 * np.sin(2.0 * np.pi * xx / w) + 0.2 * np.cos(np.pi * (yy + 0.5) / h)
    ).astype(np.float32)
    np.savez(root / "world_arrays.npz", lat=lat, lon=lon, elevation_km=elevation)
    (root / "world.json").write_text(
        json.dumps({"seed": 31415, "astronomy": {"planet": {"radius_earth": 1.0}}}),
        encoding="utf-8",
    )


def _geo(tmp_path):
    cube = PlanetTilePyramid(
        tmp_path,
        spec=TilePyramidSpec(tile_size=16, elevation_detail_strength=0.2, maximum_level=8),
    )
    return GeodeticTilePyramid(cube, spec=GeodeticTileSpec(tile_size=16, maximum_level=8))


def _zigzag_decode(encoded: np.ndarray) -> np.ndarray:
    e = np.asarray(encoded, dtype=np.int64)
    delta = (e >> 1) ^ (-(e & 1))
    return np.cumsum(delta, dtype=np.int64)


def _decode_tile(raw: bytes):
    header = struct.unpack_from("<3d2f4d3d", raw, 0)
    offset = struct.calcsize("<3d2f4d3d")
    vertex_count = struct.unpack_from("<I", raw, offset)[0]
    offset += 4
    arrays = []
    for _ in range(3):
        encoded = np.frombuffer(raw, dtype="<u2", count=vertex_count, offset=offset)
        arrays.append(_zigzag_decode(encoded))
        offset += 2 * vertex_count
    use32 = vertex_count > 65536
    align = 4 if use32 else 2
    offset += (-offset) % align
    triangle_count = struct.unpack_from("<I", raw, offset)[0]
    offset += 4
    dtype = "<u4" if use32 else "<u2"
    codes = np.frombuffer(raw, dtype=dtype, count=triangle_count * 3, offset=offset)
    offset += np.dtype(dtype).itemsize * triangle_count * 3
    highest = 0
    decoded = np.empty(len(codes), dtype=np.int64)
    for i, code_value in enumerate(codes.tolist()):
        code = int(code_value)
        decoded[i] = highest - code
        if code == 0:
            highest += 1
    edges = []
    for _ in range(4):
        count = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        values = np.frombuffer(raw, dtype=dtype, count=count, offset=offset).copy()
        offset += np.dtype(dtype).itemsize * count
        edges.append(values)
    return header, vertex_count, arrays, triangle_count, decoded, edges, offset


def test_horizon_occlusion_point_is_finite_and_outside_unit_scaled_sphere():
    radius = 6_371_000.0
    directions = np.asarray(
        [[1.0, 0.0, 0.0], [0.99, 0.1, 0.0], [0.99, 0.0, 0.1]], dtype=float
    )
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    vertices = directions * (radius + 1200.0)
    hop = np.asarray(
        compute_horizon_occlusion_point(vertices, (radius, radius, radius)), dtype=float
    )
    assert np.all(np.isfinite(hop))
    assert np.linalg.norm(hop) >= 1.0
    assert hop[0] > 0.0


def test_quantized_mesh_binary_roundtrips_core_structure(tmp_path):
    _write_world(tmp_path)
    geo = _geo(tmp_path)
    key = GeodeticTileKey(3, 7, 3)
    metadata = write_quantized_mesh_tile(geo, key)
    raw = metadata.path.read_bytes()
    header, vertex_count, arrays, triangle_count, indices, edges, consumed = _decode_tile(raw)
    u, v, h = arrays
    assert consumed == len(raw)
    assert vertex_count == 17 * 17
    assert triangle_count == 2 * 16 * 16
    assert len(indices) == triangle_count * 3
    assert int(indices.min()) == 0
    assert int(indices.max()) == vertex_count - 1
    assert set((int(u.min()), int(u.max()))) == {0, 32767}
    assert set((int(v.min()), int(v.max()))) == {0, 32767}
    assert int(h.min()) >= 0 and int(h.max()) <= 32767
    assert all(len(edge) == 17 for edge in edges)
    assert math.isfinite(header[3]) and math.isfinite(header[4])
    assert header[8] > 0.0  # bounding sphere radius
    assert all(math.isfinite(value) for value in header[9:12])


def test_quantized_mesh_edges_contain_exact_quantized_tile_boundaries(tmp_path):
    _write_world(tmp_path)
    geo = _geo(tmp_path)
    key = GeodeticTileKey(4, 12, 5)
    metadata = write_quantized_mesh_tile(geo, key)
    _header, _vc, arrays, _tc, _indices, edges, _consumed = _decode_tile(metadata.path.read_bytes())
    u, v, _h = arrays
    west, south, east, north = edges
    assert np.all(u[west] == 0)
    assert np.all(u[east] == 32767)
    assert np.all(v[south] == 0)
    assert np.all(v[north] == 32767)


def test_quantized_mesh_layer_materializes_ancestor_closure_and_exact_availability(tmp_path):
    _write_world(tmp_path)
    geo = _geo(tmp_path)
    leaf = GeodeticTileKey(3, 7, 3)
    layer_path = write_quantized_mesh_tileset(geo, (leaf,))
    layer = json.loads(layer_path.read_text(encoding="utf-8"))
    assert layer["format"] == "quantized-mesh-1.0"
    assert layer["projection"] == "EPSG:4326"
    assert layer["scheme"] == "tms"
    assert layer["maxzoom"] == 3
    assert layer["tiles"] == ["{z}/{x}/{y}.terrain?v={version}"]
    expected = [
        GeodeticTileKey(0, 1, 0),
        GeodeticTileKey(1, 3, 1),
        GeodeticTileKey(2, 3, 1),
        leaf,
    ]
    # Parent x/y are integer halves of the child at each successive level.
    expected = []
    current = leaf
    while True:
        expected.append(current)
        if current.level == 0:
            break
        current = GeodeticTileKey(current.level - 1, current.x // 2, current.y // 2)
    expected.reverse()
    for key in expected:
        path = layer_path.parent / str(key.level) / str(key.x) / f"{key.y}.terrain"
        assert path.exists()
        rects = layer["available"][key.level]
        assert {
            "startX": key.x,
            "startY": key.y,
            "endX": key.x,
            "endY": key.y,
        } in rects
    assert sum(len(level) for level in layer["available"]) == len(expected)
