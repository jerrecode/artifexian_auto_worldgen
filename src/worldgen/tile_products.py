from __future__ import annotations

"""Viewer-oriented raster products derived from sparse scientific terrain tiles.

Scientific arrays remain the authority.  This module provides globally consistent
encodings suitable for texture/height streaming.  In particular, each height tile
uses the *same* uint16 decode range, avoiding the discontinuities caused by
normalizing every tile independently.
"""

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np

from .local_downscaling import LocalTileDownscaler
from .planet_tiles import PlanetTilePyramid, TileKey, tile_geometry


@dataclass(slots=True, frozen=True)
class HeightEncoding:
    minimum_m: float
    maximum_m: float
    levels: int = 65535

    def validate(self) -> "HeightEncoding":
        if not math.isfinite(self.minimum_m) or not math.isfinite(self.maximum_m):
            raise ValueError("height encoding bounds must be finite")
        if self.maximum_m <= self.minimum_m:
            raise ValueError("height encoding maximum must exceed minimum")
        if int(self.levels) < 2:
            raise ValueError("height encoding levels must be >= 2")
        return self

    @property
    def meters_per_code(self) -> float:
        return (self.maximum_m - self.minimum_m) / float(self.levels)

    def encode(self, elevation_m: np.ndarray) -> np.ndarray:
        z = np.asarray(elevation_m, dtype=np.float64)
        scaled = (z - self.minimum_m) / (self.maximum_m - self.minimum_m)
        return np.rint(np.clip(scaled, 0.0, 1.0) * self.levels).astype(np.uint16)

    def decode(self, encoded: np.ndarray) -> np.ndarray:
        q = np.asarray(encoded, dtype=np.float64)
        return self.minimum_m + q / float(self.levels) * (
            self.maximum_m - self.minimum_m
        )


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, np.asarray(values), allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))


def _png_bytes(values: np.ndarray) -> bytes:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - matplotlib currently brings Pillow
        raise RuntimeError(
            "PNG tile export requires Pillow; install the project render dependencies"
        ) from exc
    import io

    image = Image.fromarray(np.asarray(values))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()


class TileProductExporter:
    """Cache deterministic height and imagery products for addressed globe tiles."""

    def __init__(
        self,
        pyramid: PlanetTilePyramid,
        *,
        downscaler: LocalTileDownscaler | None = None,
    ) -> None:
        self.pyramid = pyramid
        self.downscaler = downscaler or LocalTileDownscaler(pyramid)
        self.root = pyramid.root / "products" / "viewer_v1"
        self.encoding_path = self.root / "height_encoding.json"
        self._height_encoding: HeightEncoding | None = None

    def _detail_envelope_m(self) -> float:
        amplitude = float(self.pyramid._elevation_detail_amplitude_m())
        if amplitude <= 0.0 or self.pyramid.spec.maximum_level <= 0:
            return 0.0
        ratio = 2.0 ** (-float(self.pyramid.spec.detail_hurst_exponent))
        bands = int(self.pyramid.spec.maximum_level)
        geometric = (1.0 - ratio**bands) / max(1.0 - ratio, 1e-12)
        # Each band is a sum of H sine waves divided by sqrt(H/2).  This is a
        # conservative deterministic bound, not an RMS estimate.
        harmonic_bound = math.sqrt(2.0 * float(self.pyramid.spec.detail_harmonics))
        return amplitude * geometric * harmonic_bound

    @property
    def height_encoding(self) -> HeightEncoding:
        if self._height_encoding is not None:
            return self._height_encoding
        if self.encoding_path.exists():
            payload = json.loads(self.encoding_path.read_text(encoding="utf-8"))
            self._height_encoding = HeightEncoding(
                float(payload["minimum_m"]),
                float(payload["maximum_m"]),
                int(payload.get("levels", 65535)),
            ).validate()
            return self._height_encoding
        with np.load(self.pyramid.source_path, allow_pickle=False) as z:
            if "elevation_km" not in z:
                raise KeyError("world_arrays.npz does not contain elevation_km")
            source = np.asarray(z["elevation_km"], dtype=np.float64) * 1000.0
        finite = source[np.isfinite(source)]
        if finite.size == 0:
            raise RuntimeError("source elevation contains no finite samples")
        envelope = self._detail_envelope_m()
        lo = float(np.min(finite)) - envelope
        hi = float(np.max(finite)) + envelope
        margin = max(1.0, 0.005 * (hi - lo))
        encoding = HeightEncoding(lo - margin, hi + margin).validate()
        _atomic_json(
            self.encoding_path,
            {
                **asdict(encoding),
                "meters_per_code": encoding.meters_per_code,
                "source_sha256": self.pyramid._source_hash(),
                "semantics": (
                    "global uint16 decode range shared by every cube-sphere tile; "
                    "bounds include a conservative deterministic detail envelope"
                ),
            },
        )
        self._height_encoding = encoding
        return encoding

    def _product_path(self, product: str, key: TileKey, suffix: str) -> Path:
        key.validate()
        return (
            self.root
            / product
            / f"z{key.level:02d}"
            / key.face
            / f"x{key.x:08d}"
            / f"y{key.y:08d}{suffix}"
        )

    def height_u16(self, key: TileKey) -> np.ndarray:
        path = self._product_path("height_u16", key, ".npy")
        if path.exists():
            return np.load(path, mmap_mode="r", allow_pickle=False)
        elevation = np.asarray(self.pyramid.load_field(key, "elevation_m"), dtype=np.float64)
        encoded = self.height_encoding.encode(elevation)
        _atomic_save_npy(path, encoded)
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def height_png(self, key: TileKey) -> Path:
        path = self._product_path("height_u16", key, ".png")
        if path.exists():
            return path
        encoded = np.asarray(self.height_u16(key), dtype=np.uint16)
        _atomic_write(path, _png_bytes(encoded))
        return path

    def inherited_true_color_rgb(self, key: TileKey) -> np.ndarray:
        """Bilinearly sample the global true-colour product into this tile.

        This is a display resampling of the global appearance solution. It does not
        invent higher-resolution vegetation/soil/cloud physics.
        """
        path = self._product_path("true_color_rgb", key, ".npy")
        if path.exists():
            return np.load(path, mmap_mode="r", allow_pickle=False)
        geom = tile_geometry(key, self.pyramid.spec.tile_size)
        (h, w), fields = self.pyramid._source_metadata()
        if "true_color_rgb" not in fields:
            raise KeyError("world_arrays.npz does not contain true_color_rgb")
        sy, sx = self.pyramid._source_coordinates(geom)
        with np.load(self.pyramid.source_path, allow_pickle=False) as z:
            src = np.asarray(z["true_color_rgb"], dtype=np.float64)
        sampled = self.pyramid._sample_array(
            src,
            sy=sy,
            sx=sx,
            source_shape=(h, w),
            mode="linear",
        )
        rgb = np.clip(np.rint(sampled), 0, 255).astype(np.uint8)
        _atomic_save_npy(path, rgb)
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def true_color_png(self, key: TileKey) -> Path:
        path = self._product_path("true_color_rgb", key, ".png")
        if path.exists():
            return path
        rgb = np.asarray(self.inherited_true_color_rgb(key), dtype=np.uint8)
        _atomic_write(path, _png_bytes(rgb))
        return path

    def terrain_temperature_rgb(self, key: TileKey) -> np.ndarray:
        """Simple diagnostic RGB combining resolved relief and local temperature.

        This is intentionally a diagnostic visualization rather than a replacement
        for the physically derived global true-colour appearance model.
        """
        path = self._product_path("terrain_temperature_rgb", key, ".npy")
        if path.exists():
            return np.load(path, mmap_mode="r", allow_pickle=False)
        elevation = np.asarray(self.pyramid.load_field(key, "elevation_m"), dtype=np.float64)
        temperature = np.asarray(self.downscaler.annual_temperature_c(key), dtype=np.float64)
        ocean = elevation < 0.0
        height_norm = np.clip((elevation + 500.0) / 4500.0, 0.0, 1.0)
        warm = np.clip((temperature + 25.0) / 60.0, 0.0, 1.0)
        cold = 1.0 - warm
        rgb = np.empty((*elevation.shape, 3), dtype=np.float64)
        rgb[..., 0] = 55.0 + 145.0 * warm + 30.0 * height_norm
        rgb[..., 1] = 75.0 + 120.0 * (1.0 - np.abs(warm - 0.55) * 1.6)
        rgb[..., 2] = 65.0 + 150.0 * cold + 25.0 * height_norm
        rgb[ocean, 0] = 12.0 + 22.0 * warm[ocean]
        rgb[ocean, 1] = 60.0 + 65.0 * warm[ocean]
        rgb[ocean, 2] = 120.0 + 95.0 * cold[ocean]
        snow = (~ocean) & (temperature < -1.0)
        snow_mix = np.clip((-temperature - 1.0) / 14.0, 0.0, 0.92)
        rgb[snow] = (
            rgb[snow] * (1.0 - snow_mix[snow, None])
            + 242.0 * snow_mix[snow, None]
        )
        result = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        _atomic_save_npy(path, result)
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def terrain_temperature_png(self, key: TileKey) -> Path:
        path = self._product_path("terrain_temperature_rgb", key, ".png")
        if path.exists():
            return path
        rgb = np.asarray(self.terrain_temperature_rgb(key), dtype=np.uint8)
        _atomic_write(path, _png_bytes(rgb))
        return path


__all__ = ["HeightEncoding", "TileProductExporter"]
