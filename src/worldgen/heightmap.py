from __future__ import annotations

"""Lossless full-relief grayscale height-map utilities.

The encoded range is the complete modeled elevation/bathymetry range: the global
minimum elevation maps to 0 and the global maximum maps to 65535. Sea level is not
used as a clipping or normalization boundary, so bathymetry remains part of the
height field rather than becoming a flat water mask.
"""

from pathlib import Path
import binascii
import json
import os
import struct
import tempfile
import zlib

import numpy as np


def height_to_uint16(elevation_km: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Map finite elevation/bathymetry to the complete unsigned-16-bit range."""
    a = np.asarray(elevation_km, dtype=np.float64)
    if a.ndim != 2 or a.size == 0:
        raise ValueError("elevation_km must be a non-empty 2-D field")
    finite = np.isfinite(a)
    if not np.any(finite):
        raise ValueError("elevation_km contains no finite values")
    lo = float(np.min(a[finite]))
    hi = float(np.max(a[finite]))
    out = np.zeros(a.shape, dtype=np.uint16)
    if hi > lo:
        scaled = (np.nan_to_num(a, nan=lo, posinf=hi, neginf=lo) - lo) / (hi - lo)
        out = np.rint(np.clip(scaled, 0.0, 1.0) * 65535.0).astype(np.uint16)
    metadata = {
        "minimum_elevation_km": lo,
        "maximum_elevation_km": hi,
        "sea_level_code": float(np.clip((0.0 - lo) / max(hi - lo, 1e-30), 0.0, 1.0) * 65535.0),
        "encoding_min": 0.0,
        "encoding_max": 65535.0,
    }
    return out, metadata


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)


def _encode_gray16_png(array: np.ndarray) -> bytes:
    a = np.asarray(array, dtype=np.uint16)
    if a.ndim != 2:
        raise ValueError("16-bit PNG encoder requires a 2-D array")
    h, w = map(int, a.shape)
    # PNG stores 16-bit samples in network (big-endian) byte order. Filter type 0
    # keeps the encoder dependency-free and deterministic; zlib still compresses
    # smooth terrain effectively.
    be = a.astype(">u2", copy=False)
    raw = b"".join(b"\x00" + be[row].tobytes(order="C") for row in range(h))
    header = struct.pack(">IIBBBBB", w, h, 16, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw, level=6))
        + _png_chunk(b"IEND", b"")
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        try:
            dfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def write_heightmap_png16(
    path: str | Path,
    elevation_km: np.ndarray,
    *,
    metadata_path: str | Path | None = None,
) -> dict[str, float]:
    """Write a crash-safe 16-bit grayscale PNG spanning deepest ocean to highest peak."""
    p = Path(path)
    encoded, metadata = height_to_uint16(elevation_km)
    _atomic_write_bytes(p, _encode_gray16_png(encoded))
    if metadata_path is not None:
        mp = Path(metadata_path)
        text = json.dumps(
            {
                **metadata,
                "units": "km relative to modeled sea level",
                "normalization": "global minimum -> 0; global maximum -> 65535; sea level is not clipped",
                "png_bit_depth": 16,
                "png_color_type": "grayscale",
            },
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        _atomic_write_bytes(mp, text)
    return metadata


__all__ = ["height_to_uint16", "write_heightmap_png16"]
