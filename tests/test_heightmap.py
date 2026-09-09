from __future__ import annotations

import json
import struct

import numpy as np

from worldgen.heightmap import (
    height_to_uint16,
    height_to_uint32,
    write_heightmap_png16,
    write_heightmap_tiff32,
)


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


def test_uint32_heightmap_uses_one_full_range_integer_channel():
    elevation = np.array(
        [[-11.25, -1.0, 0.0], [0.001, 3.5, 9.48]],
        dtype=np.float64,
    )
    encoded, meta = height_to_uint32(elevation)
    assert encoded.dtype == np.uint32
    assert encoded.ndim == 2
    assert int(encoded.min()) == 0
    assert int(encoded.max()) == 2**32 - 1
    assert 0 < int(encoded[0, 2]) < 2**32 - 1
    assert meta["encoding_max"] == float(2**32 - 1)
    assert meta["quantization_step_m"] > 0.0
    assert meta["quantization_step_m"] < 1.0e-3


def test_uint32_tiff_is_single_channel_min_is_black_when_render_extra_available(tmp_path):
    tifffile = __import__("pytest").importorskip("tifffile")
    elevation = np.array(
        [[-8.0, -2.0, 0.0], [1.0, 4.0, 7.0]],
        dtype=np.float64,
    )
    path = tmp_path / "height32.tif"
    metadata = tmp_path / "height32.json"
    write_heightmap_tiff32(path, elevation, metadata_path=metadata, chunk_rows=1)
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        assert tuple(page.shape) == (2, 3)
        assert page.dtype == np.dtype(np.uint32)
        assert int(page.samplesperpixel) == 1
        assert str(page.photometric.name).upper() == "MINISBLACK"
        data = page.asarray()
    assert int(data.min()) == 0
    assert int(data.max()) == 2**32 - 1
    sidecar = json.loads(metadata.read_text(encoding="utf-8"))
    assert sidecar["channels"] == 1
    assert sidecar["tiff_bits_per_sample"] == 32
    assert sidecar["tiff_photometric"] == "min-is-black"
