from __future__ import annotations

"""Lazy EPSG:4326/TMS retiling above the authoritative cube-sphere world.

Cesium quantized-mesh and several geospatial clients require a geodetic TMS pyramid
rather than worldgen's internal six-face cube-sphere addresses.  This module creates
that interoperability address space without ever materializing a complete zoom
level.  Each requested geodetic tile is evaluated from the same absolute planetary
function used by cube tiles, so this is a projection/tiling adapter rather than a
second terrain generator.
"""

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np

from .planet_tiles import PlanetTilePyramid, TileGeometry


@dataclass(slots=True, frozen=True, order=True)
class GeodeticTileKey:
    """Global-geodetic TMS address: z0 contains two 180x180 degree root tiles."""

    level: int
    x: int
    y: int

    def validate(self) -> "GeodeticTileKey":
        z = int(self.level)
        if not 0 <= z <= 30:
            raise ValueError("geodetic tile level must be in [0,30]")
        width = 1 << (z + 1)
        height = 1 << z
        if not 0 <= int(self.x) < width:
            raise ValueError(f"x must be in [0,{width}) at level {z}")
        if not 0 <= int(self.y) < height:
            raise ValueError(f"y must be in [0,{height}) at level {z}")
        return self

    @property
    def width(self) -> int:
        return 1 << (int(self.level) + 1)

    @property
    def height(self) -> int:
        return 1 << int(self.level)

    def children(self) -> tuple["GeodeticTileKey", ...]:
        z = int(self.level) + 1
        return (
            GeodeticTileKey(z, 2 * self.x, 2 * self.y),
            GeodeticTileKey(z, 2 * self.x + 1, 2 * self.y),
            GeodeticTileKey(z, 2 * self.x, 2 * self.y + 1),
            GeodeticTileKey(z, 2 * self.x + 1, 2 * self.y + 1),
        )


@dataclass(slots=True, frozen=True)
class GeodeticTileSpec:
    tile_size: int = 256
    maximum_level: int = 24

    def validate(self) -> "GeodeticTileSpec":
        if not 16 <= int(self.tile_size) <= 2048:
            raise ValueError("tile_size must be in [16,2048]")
        if not 0 <= int(self.maximum_level) <= 30:
            raise ValueError("maximum_level must be in [0,30]")
        return self


@dataclass(slots=True, frozen=True)
class GeodeticTileResult:
    key: GeodeticTileKey
    fields: Mapping[str, Path]
    metadata_path: Path
    cache_hit: bool


def geodetic_tile_bounds_deg(key: GeodeticTileKey) -> tuple[float, float, float, float]:
    """Return `(west, south, east, north)` degrees for TMS south-origin y."""
    key.validate()
    span = 180.0 / float(1 << int(key.level))
    west = -180.0 + int(key.x) * span
    east = west + span
    south = -90.0 + int(key.y) * span
    north = south + span
    return west, south, east, north


def geodetic_tile_geometry(key: GeodeticTileKey, tile_size: int = 256) -> TileGeometry:
    """Return `(N+1)^2` shared-edge vertices ordered south-to-north, west-to-east."""
    key.validate()
    n = int(tile_size)
    if n < 1:
        raise ValueError("tile_size must be positive")
    west, south, east, north = geodetic_tile_bounds_deg(key)
    lon_1d = np.linspace(west, east, n + 1, dtype=np.float64)
    lat_1d = np.linspace(south, north, n + 1, dtype=np.float64)
    lon, lat = np.meshgrid(lon_1d, lat_1d)
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    c = np.cos(lat_r)
    xyz = np.stack(
        (c * np.cos(lon_r), c * np.sin(lon_r), np.sin(lat_r)), axis=-1
    )
    return TileGeometry(xyz=xyz, latitude_deg=lat, longitude_deg=lon)


def geodetic_meters_per_sample(
    planet_radius_m: float, level: int, tile_size: int = 256
) -> float:
    radius = float(planet_radius_m)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("planet_radius_m must be finite and positive")
    if int(level) < 0 or int(tile_size) <= 0:
        raise ValueError("level must be >=0 and tile_size positive")
    return math.pi * radius / (int(tile_size) * (1 << int(level)))


def internal_detail_level_for_geodetic(level: int) -> int:
    """Map geodetic angular scale to the nearest internal cube detail band."""
    return max(0, int(level) - 1)


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


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
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


class GeodeticTilePyramid:
    """Sparse geodetic/TMS view of one `PlanetTilePyramid`."""

    schema_version = 1

    def __init__(
        self,
        pyramid: PlanetTilePyramid,
        *,
        spec: GeodeticTileSpec | None = None,
    ) -> None:
        self.pyramid = pyramid
        self.spec = (spec or GeodeticTileSpec(tile_size=pyramid.spec.tile_size)).validate()
        self.root = pyramid.root / "exports" / "geodetic_tms_v1"
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "geodetic_tileset.json"
        self._ensure_manifest()

    def _ensure_manifest(self) -> None:
        payload = {
            "schema_version": self.schema_version,
            "projection": "EPSG:4326",
            "scheme": "tms",
            "root_tiles": [asdict(GeodeticTileKey(0, 0, 0)), asdict(GeodeticTileKey(0, 1, 0))],
            "tile_size": int(self.spec.tile_size),
            "maximum_level": int(self.spec.maximum_level),
            "planet_radius_m": float(self.pyramid.planet_radius_m),
            "source_sha256": self.pyramid._source_hash(),
            "source_cube_tile_schema_version": self.pyramid.schema_version,
            "semantics": "lazy global-geodetic TMS export address space evaluated from the authoritative deterministic world function",
        }
        if self.manifest_path.exists():
            current = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if current != payload:
                raise RuntimeError("existing geodetic export cache has incompatible source/specification")
        else:
            _atomic_json(self.manifest_path, payload)

    def _field_path(self, key: GeodeticTileKey, field: str) -> Path:
        key.validate()
        return (
            self.root / "fields" / str(field) / f"z{key.level:02d}"
            / f"x{key.x:08d}" / f"y{key.y:08d}.npy"
        )

    def _metadata_path(self, key: GeodeticTileKey) -> Path:
        return (
            self.root / "metadata" / f"z{key.level:02d}"
            / f"x{key.x:08d}" / f"y{key.y:08d}.json"
        )

    def _generate_field(self, key: GeodeticTileKey, field: str, geom: TileGeometry) -> np.ndarray:
        if field == "ocean_depth_m":
            elevation = self._generate_field(key, "elevation_m", geom)
            return np.maximum(-np.asarray(elevation, dtype=np.float64), 0.0).astype(np.float32)
        inherited = self.pyramid._sample_source_field(field, geom)
        if field == "elevation_m":
            base = np.asarray(inherited, dtype=np.float64)
            detail_level = internal_detail_level_for_geodetic(key.level)
            detail = self.pyramid._spectral_detail(geom.xyz, detail_level)
            modulation = 0.65 + 0.35 * np.tanh(np.abs(base) / 1500.0)
            return (base + modulation * detail).astype(np.float32)
        return inherited

    def generate_tile(
        self,
        key: GeodeticTileKey,
        fields: Sequence[str] = ("elevation_m",),
    ) -> GeodeticTileResult:
        key.validate()
        if key.level > self.spec.maximum_level:
            raise ValueError(
                f"tile level {key.level} exceeds configured maximum {self.spec.maximum_level}"
            )
        requested = tuple(dict.fromkeys(str(value) for value in fields))
        if not requested:
            raise ValueError("at least one field must be requested")
        paths = {field: self._field_path(key, field) for field in requested}
        meta = self._metadata_path(key)
        if all(path.exists() for path in paths.values()) and meta.exists():
            return GeodeticTileResult(key, paths, meta, True)
        geom = geodetic_tile_geometry(key, self.spec.tile_size)
        for field, path in paths.items():
            if not path.exists():
                _atomic_save_npy(path, self._generate_field(key, field, geom))
        previous: set[str] = set()
        if meta.exists():
            try:
                previous.update(json.loads(meta.read_text(encoding="utf-8")).get("generated_fields", []))
            except (json.JSONDecodeError, TypeError):
                pass
        west, south, east, north = geodetic_tile_bounds_deg(key)
        payload = {
            "schema_version": self.schema_version,
            "key": asdict(key),
            "bounds_degrees": {"west": west, "south": south, "east": east, "north": north},
            "vertex_shape": [self.spec.tile_size + 1, self.spec.tile_size + 1],
            "meters_per_sample_meridional": geodetic_meters_per_sample(
                self.pyramid.planet_radius_m, key.level, self.spec.tile_size
            ),
            "internal_detail_level": internal_detail_level_for_geodetic(key.level),
            "generated_fields": sorted(previous | set(requested)),
            "source_sha256": self.pyramid._source_hash(),
        }
        _atomic_json(meta, payload)
        return GeodeticTileResult(key, paths, meta, False)

    def load_field(self, key: GeodeticTileKey, field: str, *, generate: bool = True) -> np.ndarray:
        path = self._field_path(key, field)
        if not path.exists():
            if not generate:
                raise FileNotFoundError(path)
            self.generate_tile(key, (field,))
        return np.load(path, mmap_mode="r", allow_pickle=False)


__all__ = [
    "GeodeticTileKey",
    "GeodeticTilePyramid",
    "GeodeticTileResult",
    "GeodeticTileSpec",
    "geodetic_meters_per_sample",
    "geodetic_tile_bounds_deg",
    "geodetic_tile_geometry",
    "internal_detail_level_for_geodetic",
]
