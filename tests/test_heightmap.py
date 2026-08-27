from __future__ import annotations

import json
import struct

import numpy as np

from worldgen.heightmap import height_to_uint16, write_heightmap_png16


def test_heightmap_uses_deepest_and_highest_points_as_full_range(tmp_path):
    elevation = np.array([[-8.0, -2.0, 0.0], [1.0, 4.0, 7.0]], dtype=np.float32)
    encoded, meta = height_to_uint16(elevation)
    assert encoded.dtype == np.uint16
    assert int(encoded.min()) == 0
    assert int(encoded.max()) == 65535
    assert meta["minimum_elevation_km"] == -8.0
    assert meta["maximum_elevation_km"] == 7.0
    # Sea level is an interior value, not a clipping boundary.
    assert 0 < int(encoded[0, 2]) < 65535

    path = tmp_path / "height.png"
    metadata = tmp_path / "height.json"
    write_heightmap_png16(path, elevation, metadata_path=metadata)
    payload = path.read_bytes()
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    ihdr_len = struct.unpack(">I", payload[8:12])[0]
    assert ihdr_len == 13
    assert payload[12:16] == b"IHDR"
    width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", payload[16:29])
    assert (width, height) == (3, 2)
    assert bit_depth == 16
    assert color_type == 0
    sidecar = json.loads(metadata.read_text(encoding="utf-8"))
    assert sidecar["normalization"].startswith("global minimum")
