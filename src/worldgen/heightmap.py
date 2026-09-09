from __future__ import annotations

"""Lossless single-channel full-relief height-map utilities.

The encoded range always spans the complete modeled elevation/bathymetry range.
Sea level is never used as a clipping boundary. PNG output is true grayscale
(color type 0) at 16 bits/sample. The high-precision TIFF output is one
min-is-black unsigned 32-bit sample per pixel; it does not pack height into RGB.
"""

from pathlib import Path
import binascii
import json
import math
import os
import struct
import tempfile
import zlib

import numpy as np


def _validate_height_field(elevation_km: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    a = np.asarray(elevation_km, dtype=np.float64)
    if a.ndim != 2 or a.size == 0:
        raise ValueError("elevation_km must be a non-empty 2-D field")
    finite = np.isfinite(a)
    if not np.any(finite):
        raise ValueError("elevation_km contains no finite values")
    return a, finite, float(np.min(a[finite])), float(np.max(a[finite]))


def height_to_uint16(elevation_km: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Map finite elevation/bathymetry to the complete unsigned-16-bit range."""
    a, _finite, lo, hi = _validate_height_field(elevation_km)
    out = np.zeros(a.shape, dtype=np.uint16)
    if hi > lo:
        scaled = (np.nan_to_num(a, nan=lo, posinf=hi, neginf=lo) - lo) / (hi - lo)
        out = np.rint(np.clip(scaled, 0.0, 1.0) * 65535.0).astype(np.uint16)
    metadata = {
        "minimum_elevation_km": lo,
        "maximum_elevation_km": hi,
        "sea_level_code": float(
            np.clip((0.0 - lo) / max(hi - lo, 1e-30), 0.0, 1.0) * 65535.0
        ),
        "encoding_min": 0.0,
        "encoding_max": 65535.0,
        "quantization_step_m": float((hi - lo) * 1000.0 / 65535.0) if hi > lo else 0.0,
    }
    return out, metadata


def height_to_uint32(elevation_km: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Map finite elevation/bathymetry to the complete unsigned-32-bit range."""
    a, _finite, lo, hi = _validate_height_field(elevation_km)
    maximum_code = float(2**32 - 1)
    out = np.zeros(a.shape, dtype=np.uint32)
    if hi > lo:
        scaled = (np.nan_to_num(a, nan=lo, posinf=hi, neginf=lo) - lo) / (hi - lo)
        out = np.rint(np.clip(scaled, 0.0, 1.0) * maximum_code).astype(np.uint32)
    metadata = {
        "minimum_elevation_km": lo,
        "maximum_elevation_km": hi,
        "sea_level_code": float(
            np.clip((0.0 - lo) / max(hi - lo, 1e-30), 0.0, 1.0) * maximum_code
        ),
        "encoding_min": 0.0,
        "encoding_max": maximum_code,
        "quantization_step_m": float((hi - lo) * 1000.0 / maximum_code) if hi > lo else 0.0,
    }
    return out, metadata


def _finite_min_max_chunked(
    values: np.ndarray,
    *,
    chunk_rows: int = 256,
) -> tuple[float, float]:
    a = np.asarray(values)
    if a.ndim != 2 or a.size == 0:
        raise ValueError("height field must be a non-empty 2-D array")
    rows = max(1, int(chunk_rows))
    lo = math.inf
    hi = -math.inf
    found = False
    for y0 in range(0, a.shape[0], rows):
        y1 = min(a.shape[0], y0 + rows)
        part = np.asarray(a[y0:y1], dtype=np.float64)
        finite = part[np.isfinite(part)]
        if finite.size:
            found = True
            lo = min(lo, float(np.min(finite)))
            hi = max(hi, float(np.max(finite)))
    if not found:
        raise ValueError("height field contains no finite values")
    return lo, hi


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return (
        struct.pack(">I", len(payload))
        + body
        + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)
    )


def _encode_gray16_png(array: np.ndarray) -> bytes:
    a = np.asarray(array, dtype=np.uint16)
    if a.ndim != 2:
        raise ValueError("16-bit PNG encoder requires a 2-D array")
    h, w = map(int, a.shape)
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
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
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
    """Write a crash-safe one-channel 16-bit grayscale PNG."""
    p = Path(path)
    encoded, metadata = height_to_uint16(elevation_km)
    _atomic_write_bytes(p, _encode_gray16_png(encoded))
    if metadata_path is not None:
        _atomic_write_bytes(
            Path(metadata_path),
            json.dumps(
                {
                    **metadata,
                    "units": "km relative to modeled sea level",
                    "normalization": (
                        "global minimum -> 0; global maximum -> 65535; "
                        "sea level is not clipped"
                    ),
                    "png_bit_depth": 16,
                    "png_color_type": "grayscale",
                    "png_color_type_code": 0,
                    "channels": 1,
                },
                indent=2,
                sort_keys=True,
            ).encode("utf-8"),
        )
    return metadata


def write_heightmap_tiff32(
    path: str | Path,
    elevation_km: np.ndarray,
    *,
    metadata_path: str | Path | None = None,
    chunk_rows: int = 128,
) -> dict[str, float]:
    """Write a one-channel uint32 BigTIFF using bounded working memory."""
    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover - render-extra CI exercises this
        raise RuntimeError(
            "32-bit TIFF output requires the 'render' extra (tifffile)"
        ) from exc

    p = Path(path)
    a = np.asarray(elevation_km)
    lo, hi = _finite_min_max_chunked(a, chunk_rows=chunk_rows)
    maximum_code = float(2**32 - 1)
    metadata = {
        "minimum_elevation_km": lo,
        "maximum_elevation_km": hi,
        "sea_level_code": float(
            np.clip((0.0 - lo) / max(hi - lo, 1e-30), 0.0, 1.0) * maximum_code
        ),
        "encoding_min": 0.0,
        "encoding_max": maximum_code,
        "quantization_step_m": float((hi - lo) * 1000.0 / maximum_code) if hi > lo else 0.0,
    }

    p.parent.mkdir(parents=True, exist_ok=True)
    fd, encoded_name = tempfile.mkstemp(
        prefix=f".{p.name}.encoded.", suffix=".u32", dir=p.parent
    )
    os.close(fd)
    encoded_path = Path(encoded_name)
    encoded = np.memmap(encoded_path, mode="w+", dtype=np.uint32, shape=a.shape)
    rows = max(1, int(chunk_rows))
    try:
        if hi > lo:
            for y0 in range(0, a.shape[0], rows):
                y1 = min(a.shape[0], y0 + rows)
                part = np.asarray(a[y0:y1], dtype=np.float64)
                scaled = (
                    np.nan_to_num(part, nan=lo, posinf=hi, neginf=lo) - lo
                ) / (hi - lo)
                encoded[y0:y1] = np.rint(
                    np.clip(scaled, 0.0, 1.0) * maximum_code
                ).astype(np.uint32)
        else:
            encoded[:] = 0
        encoded.flush()

        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{p.name}.", suffix=".tif", dir=p.parent
        )
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            tifffile.imwrite(
                tmp,
                encoded,
                photometric="minisblack",
                compression="deflate",
                predictor=True,
                metadata=None,
                bigtiff=True,
            )
            os.replace(tmp, p)
        finally:
            tmp.unlink(missing_ok=True)
    finally:
        del encoded
        encoded_path.unlink(missing_ok=True)

    if metadata_path is not None:
        _atomic_write_bytes(
            Path(metadata_path),
            json.dumps(
                {
                    **metadata,
                    "units": "km relative to modeled sea level",
                    "normalization": (
                        "global minimum -> 0; global maximum -> 4294967295; "
                        "sea level is not clipped"
                    ),
                    "tiff_bits_per_sample": 32,
                    "tiff_sample_format": "unsigned integer",
                    "tiff_photometric": "min-is-black",
                    "channels": 1,
                    "bigtiff": True,
                    "note": (
                        "Integer code precision can exceed physical/model accuracy; "
                        "the TIFF preserves the normalized numerical field without RGB packing."
                    ),
                },
                indent=2,
                sort_keys=True,
            ).encode("utf-8"),
        )
    return metadata


__all__ = [
    "height_to_uint16",
    "height_to_uint32",
    "write_heightmap_png16",
    "write_heightmap_tiff32",
]
