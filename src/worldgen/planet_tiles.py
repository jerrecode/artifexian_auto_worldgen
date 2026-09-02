from __future__ import annotations

"""Sparse cube-sphere level-of-detail tiles for generated planets.

The global world simulation remains the low-frequency physical authority.  This
module samples that state onto fixed-size cube-sphere terrain tiles and adds only
absolute-coordinate deterministic sub-grid relief.  Tiles are generated and cached
independently, so increasing zoom does not require composing an exponentially large
global raster.

This is an LOD/storage/rendering backend, not a claim that interpolated fields plus
spectral sub-grid relief constitute a metre-scale physical erosion simulation.
Future local geomorphology kernels can use the same tile address space and halos
without changing the viewer/cache contract.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np

from .topology import (
    _map_spherical_lattice_indices,
    apply_bilinear_sampler,
    prepare_spherical_bilinear_sampler,
)


CUBE_FACES = ("px", "nx", "py", "ny", "pz", "nz")
_TILE_SCHEMA_VERSION = 2
_OMITTED_FIELDS = {"flow_to"}


@dataclass(slots=True, frozen=True, order=True)
class TileKey:
    """One cube-sphere quadtree address."""

    face: str
    level: int
    x: int
    y: int

    def validate(self) -> "TileKey":
        if self.face not in CUBE_FACES:
            raise ValueError(f"face must be one of {CUBE_FACES}, got {self.face!r}")
        if int(self.level) < 0:
            raise ValueError("tile level must be >= 0")
        side = 1 << int(self.level)
        if not 0 <= int(self.x) < side or not 0 <= int(self.y) < side:
            raise ValueError(
                f"tile x/y must be in [0, {side}) at level {self.level}, got {(self.x, self.y)}"
            )
        return self

    @property
    def side(self) -> int:
        return 1 << int(self.level)

    def children(self) -> tuple["TileKey", ...]:
        z = int(self.level) + 1
        x = int(self.x) * 2
        y = int(self.y) * 2
        return (
            TileKey(self.face, z, x, y),
            TileKey(self.face, z, x + 1, y),
            TileKey(self.face, z, x, y + 1),
            TileKey(self.face, z, x + 1, y + 1),
        )


@dataclass(slots=True, frozen=True)
class TilePyramidSpec:
    """Stable storage/generation contract for one planetary tile pyramid."""

    tile_size: int = 256
    elevation_detail_strength: float = 0.20
    detail_hurst_exponent: float = 0.65
    detail_harmonics: int = 6
    maximum_level: int = 24

    def validate(self) -> "TilePyramidSpec":
        if not 16 <= int(self.tile_size) <= 2048:
            raise ValueError("tile_size must be in [16, 2048]")
        if not math.isfinite(float(self.elevation_detail_strength)) or not (
            0.0 <= float(self.elevation_detail_strength) <= 2.0
        ):
            raise ValueError("elevation_detail_strength must be finite and in [0, 2]")
        if not math.isfinite(float(self.detail_hurst_exponent)) or not (
            0.1 <= float(self.detail_hurst_exponent) <= 1.5
        ):
            raise ValueError("detail_hurst_exponent must be finite and in [0.1, 1.5]")
        if not 1 <= int(self.detail_harmonics) <= 32:
            raise ValueError("detail_harmonics must be in [1, 32]")
        if not 0 <= int(self.maximum_level) <= 30:
            raise ValueError("maximum_level must be in [0, 30]")
        return self


@dataclass(slots=True, frozen=True)
class TileGeometry:
    xyz: np.ndarray
    latitude_deg: np.ndarray
    longitude_deg: np.ndarray


@dataclass(slots=True, frozen=True)
class TileResult:
    key: TileKey
    fields: Mapping[str, Path]
    metadata_path: Path
    cache_hit: bool


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


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
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


def _file_sha256(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _cube_direction(face: str, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Map cube-face coordinates to normalized planet-centred directions.

    ``s`` grows left-to-right and ``t`` top-to-bottom.  The orientation is the
    conventional cubemap layout and is chosen so shared face edges evaluate to the
    same 3-D direction (possibly in reversed traversal order).
    """
    if face == "px":
        xyz = np.stack((np.ones_like(s), -t, -s), axis=-1)
    elif face == "nx":
        xyz = np.stack((-np.ones_like(s), -t, s), axis=-1)
    elif face == "py":
        xyz = np.stack((s, np.ones_like(s), t), axis=-1)
    elif face == "ny":
        xyz = np.stack((s, -np.ones_like(s), -t), axis=-1)
    elif face == "pz":
        xyz = np.stack((s, -t, np.ones_like(s)), axis=-1)
    elif face == "nz":
        xyz = np.stack((-s, -t, -np.ones_like(s)), axis=-1)
    else:
        raise ValueError(f"unknown cube face: {face!r}")
    return xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)


def tile_geometry(key: TileKey, tile_size: int = 256) -> TileGeometry:
    """Return shared-edge vertex geometry for one tile.

    Arrays have shape ``(tile_size + 1, tile_size + 1)`` so adjacent terrain tiles
    share an identical vertex row/column rather than relying on skirts to conceal
    cracks.
    """
    key.validate()
    n = int(tile_size)
    if n < 1:
        raise ValueError("tile_size must be positive")
    side = key.side
    qx = int(key.x) + np.arange(n + 1, dtype=np.float64) / float(n)
    qy = int(key.y) + np.arange(n + 1, dtype=np.float64) / float(n)
    s = -1.0 + 2.0 * qx[None, :] / float(side)
    t = -1.0 + 2.0 * qy[:, None] / float(side)
    s, t = np.broadcast_arrays(s, t)
    xyz = _cube_direction(key.face, s, t)
    lat = np.rad2deg(np.arcsin(np.clip(xyz[..., 2], -1.0, 1.0)))
    lon = np.rad2deg(np.arctan2(xyz[..., 1], xyz[..., 0]))
    return TileGeometry(xyz=xyz, latitude_deg=lat, longitude_deg=lon)


def latlon_to_unit(latitude_deg: float, longitude_deg: float) -> np.ndarray:
    lat = math.radians(float(latitude_deg))
    lon = math.radians(float(longitude_deg))
    c = math.cos(lat)
    return np.asarray((c * math.cos(lon), c * math.sin(lon), math.sin(lat)), dtype=np.float64)


def direction_to_tile(direction: Sequence[float], level: int) -> TileKey:
    """Return the cube tile containing a unit (or normalizable) direction."""
    p = np.asarray(direction, dtype=np.float64)
    if p.shape != (3,) or np.any(~np.isfinite(p)):
        raise ValueError("direction must be a finite length-3 vector")
    norm = float(np.linalg.norm(p))
    if norm <= 0:
        raise ValueError("direction cannot be zero")
    x, y, z = p / norm
    ax, ay, az = abs(x), abs(y), abs(z)
    if ax >= ay and ax >= az:
        if x >= 0:
            face, s, t = "px", -z / ax, -y / ax
        else:
            face, s, t = "nx", z / ax, -y / ax
    elif ay >= ax and ay >= az:
        if y >= 0:
            face, s, t = "py", x / ay, z / ay
        else:
            face, s, t = "ny", x / ay, -z / ay
    else:
        if z >= 0:
            face, s, t = "pz", x / az, -y / az
        else:
            face, s, t = "nz", -x / az, -y / az
    side = 1 << int(level)
    fx = np.clip((s + 1.0) * 0.5 * side, 0.0, np.nextafter(float(side), 0.0))
    fy = np.clip((t + 1.0) * 0.5 * side, 0.0, np.nextafter(float(side), 0.0))
    return TileKey(face, int(level), int(math.floor(float(fx))), int(math.floor(float(fy)))).validate()


def latlon_to_tile(latitude_deg: float, longitude_deg: float, level: int) -> TileKey:
    return direction_to_tile(latlon_to_unit(latitude_deg, longitude_deg), level)


def _angular_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def _tile_bounding_cap(key: TileKey) -> tuple[np.ndarray, float]:
    geom = tile_geometry(key, 1)
    corners = np.asarray(
        (geom.xyz[0, 0], geom.xyz[0, -1], geom.xyz[-1, 0], geom.xyz[-1, -1]),
        dtype=np.float64,
    )
    center = np.sum(corners, axis=0)
    center /= max(float(np.linalg.norm(center)), 1e-300)
    radius = max(_angular_distance(center, corner) for corner in corners)
    return center, radius


def visible_tiles(
    *,
    latitude_deg: float,
    longitude_deg: float,
    angular_radius_deg: float,
    level: int,
    maximum_tiles: int = 4096,
) -> tuple[TileKey, ...]:
    """Hierarchically select tiles intersecting a spherical viewing cap.

    This traverses only intersecting quadtree branches; it never enumerates all
    ``6 * 4**level`` tiles.  A renderer can call it with the current view centre,
    angular field of view and selected LOD, then load/generate only returned keys.
    """
    if int(level) < 0:
        raise ValueError("level must be >= 0")
    view = latlon_to_unit(latitude_deg, longitude_deg)
    radius = math.radians(max(0.0, min(180.0, float(angular_radius_deg))))
    pending = [TileKey(face, 0, 0, 0) for face in CUBE_FACES]
    selected: list[TileKey] = []
    while pending:
        key = pending.pop()
        center, tile_radius = _tile_bounding_cap(key)
        if _angular_distance(view, center) > radius + tile_radius:
            continue
        if key.level == int(level):
            selected.append(key)
            if len(selected) > int(maximum_tiles):
                raise RuntimeError(
                    "visible tile selection exceeded maximum_tiles; choose a coarser LOD or smaller view cap"
                )
        else:
            pending.extend(reversed(key.children()))
    return tuple(sorted(selected))


def approximate_meters_per_sample(
    planet_radius_m: float, level: int, tile_size: int = 256
) -> float:
    """Characteristic cube-face centre resolution for LOD planning."""
    radius = float(planet_radius_m)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("planet_radius_m must be finite and positive")
    if int(level) < 0 or int(tile_size) <= 0:
        raise ValueError("level must be >= 0 and tile_size positive")
    return math.pi * radius / (2.0 * int(tile_size) * (1 << int(level)))


def level_for_meters_per_sample(
    planet_radius_m: float,
    target_m_per_sample: float,
    *,
    tile_size: int = 256,
    maximum_level: int = 24,
) -> int:
    target = float(target_m_per_sample)
    if not math.isfinite(target) or target <= 0:
        raise ValueError("target_m_per_sample must be finite and positive")
    root = approximate_meters_per_sample(planet_radius_m, 0, tile_size)
    return min(int(maximum_level), max(0, int(math.ceil(math.log2(root / target)))))


class PlanetTilePyramid:
    """On-demand persistent cube-sphere tile generator for one completed world."""

    schema_version = _TILE_SCHEMA_VERSION

    def __init__(
        self,
        world_root: str | Path,
        *,
        spec: TilePyramidSpec | None = None,
        planet_radius_m: float | None = None,
        source_level: int | None = None,
    ) -> None:
        self.world_root = Path(world_root).expanduser().resolve()
        self.spec = (spec or TilePyramidSpec()).validate()
        self.base_source_path = self.world_root / "world_arrays.npz"
        if not self.base_source_path.exists():
            raise FileNotFoundError(
                f"{self.base_source_path} does not exist; generate the base world with NPZ output first"
            )
        self.source_level = 0
        self.source_kind = "base_npz"
        self.source_path = self.base_source_path
        self._source_arrays_dir: Path | None = None
        self._source_index: dict[str, object] | None = None
        self._select_source(source_level)
        cache_root = self.world_root / "tiles" / "cubesphere_v1"
        self.root = (
            cache_root
            if self.source_level == 0
            else cache_root / f"refinement_level_{self.source_level:04d}"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "tileset.json"
        self._source_sha256: str | None = None
        self._source_shape: tuple[int, int] | None = None
        self._root_seed: int | None = None
        self._planet_radius_m = (
            float(planet_radius_m) if planet_radius_m is not None else self._discover_planet_radius_m()
        )
        if not math.isfinite(self._planet_radius_m) or self._planet_radius_m <= 0:
            raise ValueError("planet radius must be finite and positive")
        self._ensure_manifest()

    def _select_source(self, requested_level: int | None) -> None:
        """Select the complete deepest refinement by default.

        Refinement levels are already globally composed static ``.npy`` maps.  The
        older tile path nevertheless always sampled ``world_arrays.npz``, making
        successful refinement invisible to every later LOD/precompute operation.
        """
        manifest_path = self.world_root / "refinement" / "manifest.json"
        manifest: dict[str, object] | None = None
        if manifest_path.exists():
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"unreadable refinement manifest: {manifest_path}") from exc
            if not isinstance(loaded, dict):
                raise RuntimeError(f"invalid refinement manifest: {manifest_path}")
            manifest = loaded

        if requested_level is None:
            level = int(manifest.get("deepest_complete_level", 0)) if manifest else 0
        else:
            level = int(requested_level)
        if level < 0:
            raise ValueError("source_level must be >= 0")
        if level == 0:
            return
        if manifest is None:
            raise ValueError(
                f"refinement source level {level} was requested but no refinement manifest exists"
            )
        record = manifest.get("levels", {}).get(str(level)) if isinstance(manifest.get("levels"), dict) else None
        if not isinstance(record, dict) or record.get("complete") is not True:
            raise ValueError(f"refinement source level {level} is not complete")
        index_path = self.world_root / str(
            record.get("index", f"refinement/levels/level_{level:04d}/index.json")
        )
        if not index_path.exists():
            raise FileNotFoundError(f"refinement source index does not exist: {index_path}")
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"unreadable refinement source index: {index_path}") from exc
        if index.get("complete") is not True or int(index.get("level", -1)) != level:
            raise RuntimeError(f"refinement source index is not a complete level {level}: {index_path}")
        arrays_dir = index_path.parent / "arrays"
        for required in ("lat", "lon", "elevation_km"):
            if not (arrays_dir / f"{required}.npy").exists():
                raise FileNotFoundError(
                    f"refinement source level {level} lacks required array {required!r}"
                )
        self.source_level = level
        self.source_kind = "refinement_level"
        self.source_path = index_path
        self._source_arrays_dir = arrays_dir
        self._source_index = index

    @property
    def planet_radius_m(self) -> float:
        return self._planet_radius_m

    def _world_json(self) -> Mapping[str, object]:
        path = self.world_root / "world.json"
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def _discover_planet_radius_m(self) -> float:
        payload = self._world_json()

        def walk(value):
            if isinstance(value, dict):
                if "radius_m" in value:
                    try:
                        radius = float(value["radius_m"])
                        if math.isfinite(radius) and radius > 0:
                            return radius
                    except (TypeError, ValueError):
                        pass
                if "radius_earth" in value:
                    try:
                        radius = float(value["radius_earth"]) * 6.371e6
                        if math.isfinite(radius) and radius > 0:
                            return radius
                    except (TypeError, ValueError):
                        pass
                for child in value.values():
                    found = walk(child)
                    if found is not None:
                        return found
            elif isinstance(value, list):
                for child in value:
                    found = walk(child)
                    if found is not None:
                        return found
            return None

        found = walk(payload)
        if found is None:
            raise ValueError(
                "planet radius was not found in world.json; pass planet_radius_m explicitly"
            )
        return float(found)

    def _read_seed(self) -> int:
        if self._root_seed is None:
            value = self._world_json().get("seed", 0)
            try:
                self._root_seed = int(value)
            except (TypeError, ValueError):
                self._root_seed = 0
        return self._root_seed

    def _source_metadata(self) -> tuple[tuple[int, int], tuple[str, ...]]:
        if self.source_kind == "refinement_level":
            assert self._source_index is not None and self._source_arrays_dir is not None
            resolution = self._source_index.get("resolution")
            entries = self._source_index.get("entries")
            if not (
                isinstance(resolution, list)
                and len(resolution) == 2
                and isinstance(entries, dict)
            ):
                raise RuntimeError(f"invalid refinement source index: {self.source_path}")
            fields = tuple(
                sorted(
                    name
                    for name, entry in entries.items()
                    if isinstance(name, str)
                    and isinstance(entry, dict)
                    and not entry.get("omitted_at_refined_levels")
                    and (self._source_arrays_dir / f"{name}.npy").exists()
                )
            )
            self._source_shape = (int(resolution[1]), int(resolution[0]))
            return self._source_shape, fields
        if self._source_shape is not None:
            with np.load(self.base_source_path, allow_pickle=False) as z:
                return self._source_shape, tuple(sorted(z.files))
        with np.load(self.base_source_path, allow_pickle=False) as z:
            if "lat" not in z or "lon" not in z:
                raise ValueError("world_arrays.npz must contain lat and lon")
            self._source_shape = (int(len(z["lat"])), int(len(z["lon"])))
            return self._source_shape, tuple(sorted(z.files))

    def _load_source_array(self, name: str) -> np.ndarray:
        if self.source_kind == "refinement_level":
            assert self._source_arrays_dir is not None
            path = self._source_arrays_dir / f"{name}.npy"
            if not path.exists():
                raise KeyError(f"source field {name!r} is not present in refinement level {self.source_level}")
            return np.load(path, mmap_mode="r", allow_pickle=False)
        with np.load(self.base_source_path, allow_pickle=False) as z:
            if name not in z:
                raise KeyError(f"source field {name!r} is not present in world_arrays.npz")
            return np.asarray(z[name])

    def _source_hash(self) -> str:
        if self._source_sha256 is None:
            if self.source_kind == "base_npz":
                self._source_sha256 = _file_sha256(self.base_source_path)
            else:
                assert self._source_arrays_dir is not None
                _shape, fields = self._source_metadata()
                digest = hashlib.sha256()
                digest.update(b"worldgen-refinement-source-v1\0")
                digest.update(self.source_path.read_bytes())
                for name in fields:
                    path = self._source_arrays_dir / f"{name}.npy"
                    stat = path.stat()
                    digest.update(name.encode("utf-8"))
                    digest.update(b"\0")
                    digest.update(str(int(stat.st_size)).encode("ascii"))
                    digest.update(b"\0")
                    digest.update(str(int(stat.st_mtime_ns)).encode("ascii"))
                    digest.update(b"\0")
                self._source_sha256 = digest.hexdigest()
        return self._source_sha256

    def _ensure_manifest(self) -> None:
        shape, fields = self._source_metadata()
        expected = {
            "schema_version": self.schema_version,
            "projection": "cube_sphere",
            "faces": list(CUBE_FACES),
            "addressing": "face/level/x/y",
            "tile_size": int(self.spec.tile_size),
            "terrain_vertex_shape": [int(self.spec.tile_size) + 1, int(self.spec.tile_size) + 1],
            "planet_radius_m": self.planet_radius_m,
            "source_kind": self.source_kind,
            "source_level": int(self.source_level),
            "source_path": str(self.source_path.relative_to(self.world_root)),
            "source_resolution": [shape[1], shape[0]],
            "source_sha256": self._source_hash(),
            "source_fields": list(fields),
            "omitted_fields": sorted(_OMITTED_FIELDS),
            "spec": asdict(self.spec),
            "lod": {
                "root_meters_per_sample_approx": approximate_meters_per_sample(
                    self.planet_radius_m, 0, self.spec.tile_size
                ),
                "maximum_level": int(self.spec.maximum_level),
                "semantics": "each level halves characteristic metres per sample; high levels are generated lazily",
            },
            "limitations": [
                "global coupled physics is inherited from the selected complete global source rather than rerun independently per tile",
                "high-frequency elevation is deterministic spectral sub-grid relief, not yet a local fluvial/landslide simulation",
                "flow_to is omitted because global raster receiver indices are invalid in a tile address space",
            ],
        }
        if self.manifest_path.exists():
            current = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if current != expected:
                raise RuntimeError(
                    "existing tile pyramid was created from a different world or TilePyramidSpec; remove it or use a separate world output directory"
                )
        else:
            _atomic_write_json(self.manifest_path, expected)

    def _field_path(self, key: TileKey, field: str) -> Path:
        key.validate()
        return (
            self.root
            / "fields"
            / str(field)
            / f"z{key.level:02d}"
            / key.face
            / f"x{key.x:08d}"
            / f"y{key.y:08d}.npy"
        )

    def _metadata_path(self, key: TileKey) -> Path:
        return (
            self.root
            / "metadata"
            / f"z{key.level:02d}"
            / key.face
            / f"x{key.x:08d}"
            / f"y{key.y:08d}.json"
        )

    def _source_coordinates(self, geom: TileGeometry) -> tuple[np.ndarray, np.ndarray]:
        (h, w), _ = self._source_metadata()
        sy = (90.0 - geom.latitude_deg) * (h / 180.0) - 0.5
        sx = ((geom.longitude_deg + 180.0) % 360.0) * (w / 360.0) - 0.5
        return sy, sx

    def _sample_array(
        self,
        values: np.ndarray,
        *,
        sy: np.ndarray,
        sx: np.ndarray,
        source_shape: tuple[int, int],
        mode: str,
    ) -> np.ndarray:
        a = np.asarray(values)
        h, w = source_shape
        if a.ndim >= 2 and tuple(a.shape[-2:]) == (h, w):
            axes = (a.ndim - 2, a.ndim - 1)
        elif a.ndim >= 2 and tuple(a.shape[:2]) == (h, w):
            axes = (0, 1)
        else:
            raise ValueError(f"array shape {a.shape} has no source spatial axes {(h, w)}")
        moved = np.moveaxis(a, axes, (-2, -1))
        leading = moved.shape[:-2]
        flat = moved.reshape((-1, h, w)) if leading else moved.reshape((1, h, w))
        if mode == "nearest":
            iy = np.floor(sy + 0.5).astype(np.int64)
            ix = np.floor(sx + 0.5).astype(np.int64)
            iy, ix = _map_spherical_lattice_indices(iy, ix, (h, w))
            sampled = np.stack([part[iy, ix] for part in flat], axis=0)
        else:
            sampler = prepare_spherical_bilinear_sampler(sy, sx, (h, w))
            sampled = np.stack([apply_bilinear_sampler(part, sampler) for part in flat], axis=0)
        if leading:
            sampled = sampled.reshape((*leading, *sy.shape))
        else:
            sampled = sampled[0]
        return np.moveaxis(sampled, (-2, -1), axes) if leading else sampled

    def _sample_source_field(self, field: str, geom: TileGeometry) -> np.ndarray:
        if field == "latitude_deg":
            return geom.latitude_deg.astype(np.float64, copy=True)
        if field == "longitude_deg":
            return geom.longitude_deg.astype(np.float64, copy=True)
        if field in _OMITTED_FIELDS:
            raise ValueError(f"field {field!r} cannot be inherited into sparse tiles")
        source_name = "elevation_km" if field == "elevation_m" else field
        (h, w), source_fields = self._source_metadata()
        if source_name not in source_fields:
            raise KeyError(f"source field {source_name!r} is not present in selected source level {self.source_level}")
        sy, sx = self._source_coordinates(geom)
        values = self._load_source_array(source_name)
        mode = "nearest" if values.dtype.kind in "biuUSO" else "linear"
        sampled = self._sample_array(values, sy=sy, sx=sx, source_shape=(h, w), mode=mode)
        if field == "elevation_m":
            return np.asarray(sampled, dtype=np.float64) * 1000.0
        if source_name.startswith("true_color"):
            return np.clip(np.rint(sampled), 0, 255).astype(np.uint8)
        return sampled

    def _elevation_detail_amplitude_m(self) -> float:
        if self.spec.elevation_detail_strength <= 0:
            return 0.0
        try:
            elevation = np.asarray(self._load_source_array("elevation_km"), dtype=np.float64)
        except KeyError:
            return 0.0
        finite = elevation[np.isfinite(elevation)]
        if finite.size == 0:
            return 0.0
        lo, hi = np.percentile(finite, (2.0, 98.0))
        relief_m = max(float(hi - lo) * 1000.0, 100.0)
        return float(self.spec.elevation_detail_strength) * min(1200.0, 0.10 * relief_m)

    def _spectral_detail(self, xyz: np.ndarray, level: int) -> np.ndarray:
        if int(level) <= 0 or self.spec.elevation_detail_strength <= 0:
            return np.zeros(xyz.shape[:-1], dtype=np.float64)
        (source_h, _), _fields = self._source_metadata()
        base_frequency = max(8.0, source_h / 5.0)
        amplitude0 = self._elevation_detail_amplitude_m()
        result = np.zeros(xyz.shape[:-1], dtype=np.float64)
        norm = math.sqrt(max(1.0, self.spec.detail_harmonics / 2.0))
        for band in range(1, int(level) + 1):
            h = hashlib.blake2b(digest_size=8)
            h.update(str(self._read_seed()).encode("ascii"))
            h.update(b"\0planet_tile_elevation\0")
            h.update(str(band).encode("ascii"))
            rng = np.random.default_rng(int.from_bytes(h.digest(), "little"))
            frequency = base_frequency * (2.0 ** (band - 1))
            band_field = np.zeros(xyz.shape[:-1], dtype=np.float64)
            for _ in range(int(self.spec.detail_harmonics)):
                axis = rng.normal(size=3)
                axis /= max(float(np.linalg.norm(axis)), 1e-300)
                phase = float(rng.uniform(0.0, 2.0 * math.pi))
                band_field += np.sin(
                    frequency * np.tensordot(xyz, axis, axes=([-1], [0])) + phase
                )
            amplitude = amplitude0 * (2.0 ** (-self.spec.detail_hurst_exponent * (band - 1)))
            result += amplitude * band_field / norm
        return result

    def _generate_field(self, key: TileKey, field: str, geom: TileGeometry) -> np.ndarray:
        if field == "ocean_depth_m":
            elevation = self._generate_field(key, "elevation_m", geom)
            return np.maximum(-np.asarray(elevation, dtype=np.float64), 0.0).astype(np.float32)
        inherited = self._sample_source_field(field, geom)
        if field == "elevation_m":
            base = np.asarray(inherited, dtype=np.float64)
            detail = self._spectral_detail(geom.xyz, key.level)
            modulation = 0.65 + 0.35 * np.tanh(np.abs(base) / 1500.0)
            return (base + modulation * detail).astype(np.float32)
        return inherited

    def _tile_metadata(self, key: TileKey, geom: TileGeometry, fields: Iterable[str]) -> dict[str, object]:
        n = self.spec.tile_size
        mid = n // 2
        centre = geom.xyz[mid, mid]
        dx = _angular_distance(centre, geom.xyz[mid, min(mid + 1, n)]) * self.planet_radius_m
        dy = _angular_distance(centre, geom.xyz[min(mid + 1, n), mid]) * self.planet_radius_m
        corners = [
            [float(geom.latitude_deg[0, 0]), float(geom.longitude_deg[0, 0])],
            [float(geom.latitude_deg[0, -1]), float(geom.longitude_deg[0, -1])],
            [float(geom.latitude_deg[-1, -1]), float(geom.longitude_deg[-1, -1])],
            [float(geom.latitude_deg[-1, 0]), float(geom.longitude_deg[-1, 0])],
        ]
        return {
            "schema_version": self.schema_version,
            "key": asdict(key),
            "vertex_shape": [n + 1, n + 1],
            "centre_latitude_deg": float(geom.latitude_deg[mid, mid]),
            "centre_longitude_deg": float(geom.longitude_deg[mid, mid]),
            "corner_latlon_deg": corners,
            "meters_per_sample_centre": float(0.5 * (dx + dy)),
            "generated_fields": sorted(set(map(str, fields))),
            "source_sha256": self._source_hash(),
            "terrain_semantics": "global simulation inherited + deterministic absolute-coordinate spectral sub-grid relief",
        }

    def generate_tile(self, key: TileKey, fields: Sequence[str] = ("elevation_m",)) -> TileResult:
        key.validate()
        if key.level > self.spec.maximum_level:
            raise ValueError(
                f"tile level {key.level} exceeds configured maximum {self.spec.maximum_level}"
            )
        requested = tuple(dict.fromkeys(str(field) for field in fields))
        if not requested:
            raise ValueError("at least one field must be requested")
        paths = {field: self._field_path(key, field) for field in requested}
        meta_path = self._metadata_path(key)
        if all(path.exists() for path in paths.values()) and meta_path.exists():
            return TileResult(key, paths, meta_path, True)

        geom = tile_geometry(key, self.spec.tile_size)
        for field, path in paths.items():
            if path.exists():
                continue
            values = self._generate_field(key, field, geom)
            _atomic_save_npy(path, values)
        previous_fields: set[str] = set()
        if meta_path.exists():
            try:
                previous_fields.update(json.loads(meta_path.read_text(encoding="utf-8")).get("generated_fields", []))
            except (json.JSONDecodeError, TypeError):
                pass
        metadata = self._tile_metadata(key, geom, previous_fields | set(requested))
        _atomic_write_json(meta_path, metadata)
        return TileResult(key, paths, meta_path, False)

    def load_field(self, key: TileKey, field: str, *, generate: bool = True) -> np.ndarray:
        path = self._field_path(key, field)
        if not path.exists():
            if not generate:
                raise FileNotFoundError(path)
            self.generate_tile(key, (field,))
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def select_visible(
        self,
        *,
        latitude_deg: float,
        longitude_deg: float,
        angular_radius_deg: float,
        level: int,
        maximum_tiles: int = 4096,
    ) -> tuple[TileKey, ...]:
        return visible_tiles(
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            angular_radius_deg=angular_radius_deg,
            level=level,
            maximum_tiles=maximum_tiles,
        )

    def level_for_resolution(self, target_m_per_sample: float) -> int:
        return level_for_meters_per_sample(
            self.planet_radius_m,
            target_m_per_sample,
            tile_size=self.spec.tile_size,
            maximum_level=self.spec.maximum_level,
        )


__all__ = [
    "CUBE_FACES",
    "PlanetTilePyramid",
    "TileGeometry",
    "TileKey",
    "TilePyramidSpec",
    "TileResult",
    "approximate_meters_per_sample",
    "direction_to_tile",
    "latlon_to_tile",
    "latlon_to_unit",
    "level_for_meters_per_sample",
    "tile_geometry",
    "visible_tiles",
]
