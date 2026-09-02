from __future__ import annotations

"""Recursive, resumable, spherical refinement of generated world datasets.

This module deliberately separates *global* simulation from *local* refinement.
Globally coupled physics is not rerun independently inside arbitrary tiles. Instead,
each refinement level inherits the complete composed parent state, samples it with
spherical seam/pole semantics, applies deterministic sub-grid kernels, and only
then crops halo regions. All siblings therefore see the same parent/global forcing
at their boundaries.

Every completed level is composed into a full random-access ``.npy`` dataset. A
later invocation refines that composed level, which permits sections of sections to
arbitrary depth without accumulating independent sibling boundary conditions.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from typing import Any, Callable, Iterable

import numpy as np

from .heightmap import write_heightmap_png16
from .topology import (
    _map_spherical_lattice_indices,
    apply_bilinear_sampler,
    prepare_spherical_bilinear_sampler,
)


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(slots=True, frozen=True)
class RefinementSpec:
    scale: int = 2
    sections_y: int = 2
    sections_x: int = 2
    halo_cells: int = 8
    elevation_detail_strength: float = 0.20
    keep_sections: bool = False

    def validate(self) -> "RefinementSpec":
        if int(self.scale) < 2:
            raise ValueError("refinement scale must be >= 2")
        if int(self.sections_y) < 1 or int(self.sections_x) < 1:
            raise ValueError("refinement section counts must be >= 1")
        if int(self.halo_cells) < 0:
            raise ValueError("refinement halo must be >= 0")
        if not math.isfinite(float(self.elevation_detail_strength)) or not (
            0.0 <= float(self.elevation_detail_strength) <= 2.0
        ):
            raise ValueError("elevation detail strength must be finite and in [0, 2]")
        return self


@dataclass(slots=True, frozen=True)
class RefinementNode:
    node_id: str
    parent_id: str
    level: int
    row: int
    col: int
    source_bounds: tuple[int, int, int, int]
    core_bounds: tuple[int, int, int, int]

    @property
    def shape(self) -> tuple[int, int]:
        y0, y1, x0, x1 = self.core_bounds
        return y1 - y0, x1 - x0


_SKIP_REFINEMENT = {
    # Receiver indices refer to the source raster's flattened address space. They
    # cannot be interpolated. A later hydrology-refinement kernel must rebuild them.
    "flow_to",
}

_VECTOR_COMPONENT_PREFIXES = (
    "ocean_current_u",
    "ocean_current_v",
    "wind_u",
    "wind_v",
    "global_circulation_u",
    "global_circulation_v",
    "humidity_transport_u",
    "humidity_transport_v",
)


def _emit(callback: ProgressCallback | None, **payload: Any) -> None:
    if callback is not None:
        callback(payload)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, np.asarray(array), allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _atomic_copy_file(source: Path, target: Path) -> None:
    """Publish a stable alias without exposing a partially copied product."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    tmp = Path(tmp_name)
    try:
        with source.open("rb") as src, os.fdopen(fd, "wb") as dst:
            shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp, target)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _spatial_axes(shape: tuple[int, ...], height: int, width: int) -> tuple[int, int] | None:
    if len(shape) >= 2 and tuple(shape[-2:]) == (height, width):
        return len(shape) - 2, len(shape) - 1
    if len(shape) >= 3 and tuple(shape[:2]) == (height, width):
        return 0, 1
    return None


def _field_mode(name: str, dtype: np.dtype) -> str:
    if name in _SKIP_REFINEMENT:
        return "skip"
    if name.startswith("true_color"):
        return "linear"
    if dtype.kind in "buiUSO":
        return "nearest"
    return "linear"


def _partition(start: int, stop: int, count: int) -> list[tuple[int, int]]:
    length = stop - start
    if length <= 0:
        raise ValueError("cannot partition an empty interval")
    if count > length:
        raise ValueError(
            f"section count {count} exceeds available parent cells {length}; reduce sections or refine first"
        )
    edges = [start + (length * i) // count for i in range(count + 1)]
    return [(edges[i], edges[i + 1]) for i in range(count)]


def _mapped_indices_with_parity(
    y: np.ndarray, x: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spherical integer mapping plus odd/even pole-crossing parity for vectors."""
    h, w = map(int, shape)
    yy = np.asarray(y, dtype=np.int64).copy()
    xx = np.asarray(x, dtype=np.int64).copy()
    parity = np.zeros(np.broadcast_shapes(yy.shape, xx.shape), dtype=np.int8)
    yy, xx = np.broadcast_arrays(yy, xx)
    yy = yy.copy(); xx = xx.copy()
    while np.any((yy < 0) | (yy >= h)):
        north = yy < 0
        south = yy >= h
        crossed = north | south
        parity ^= crossed.astype(np.int8)
        if np.any(north):
            yy = np.where(north, -yy - 1, yy)
            xx = np.where(north, xx + w // 2, xx)
        if np.any(south):
            yy = np.where(south, 2 * h - yy - 1, yy)
            xx = np.where(south, xx + w // 2, xx)
    return yy.astype(np.int32), (xx % w).astype(np.int32), parity


def _signed_bilinear_sample(values: np.ndarray, sy: np.ndarray, sx: np.ndarray) -> np.ndarray:
    """Bilinear sampling for tangent-vector components across reflected poles."""
    a = np.asarray(values)
    h, w = a.shape
    y0 = np.floor(sy).astype(np.int64); y1 = y0 + 1
    x0 = np.floor(sx).astype(np.int64); x1 = x0 + 1
    fy = sy - y0; fx = sx - x0
    y00, x00, p00 = _mapped_indices_with_parity(y0, x0, (h, w))
    y01, x01, p01 = _mapped_indices_with_parity(y0, x1, (h, w))
    y10, x10, p10 = _mapped_indices_with_parity(y1, x0, (h, w))
    y11, x11, p11 = _mapped_indices_with_parity(y1, x1, (h, w))
    s00 = 1.0 - 2.0 * p00; s01 = 1.0 - 2.0 * p01
    s10 = 1.0 - 2.0 * p10; s11 = 1.0 - 2.0 * p11
    return (
        a[y00, x00] * s00 * (1.0 - fy) * (1.0 - fx)
        + a[y01, x01] * s01 * (1.0 - fy) * fx
        + a[y10, x10] * s10 * fy * (1.0 - fx)
        + a[y11, x11] * s11 * fy * fx
    )


def _sample_2d(
    values: np.ndarray,
    sy: np.ndarray,
    sx: np.ndarray,
    *,
    mode: str,
    signed_vector: bool,
) -> np.ndarray:
    a = np.asarray(values)
    if mode == "nearest":
        yy = np.floor(sy + 0.5).astype(np.int64)
        xx = np.floor(sx + 0.5).astype(np.int64)
        yy, xx = _map_spherical_lattice_indices(yy, xx, a.shape)
        return a[yy, xx]
    if signed_vector:
        return _signed_bilinear_sample(a, sy, sx)
    sampler = prepare_spherical_bilinear_sampler(sy, sx, a.shape)
    return apply_bilinear_sampler(a, sampler)


def _sample_spatial(
    values: np.ndarray,
    axes: tuple[int, int],
    sy: np.ndarray,
    sx: np.ndarray,
    *,
    mode: str,
    signed_vector: bool,
) -> np.ndarray:
    a = np.asarray(values)
    moved = np.moveaxis(a, axes, (-2, -1))
    leading = moved.shape[:-2]
    if not leading:
        sampled = _sample_2d(moved, sy, sx, mode=mode, signed_vector=signed_vector)
    else:
        flat = moved.reshape((-1, *moved.shape[-2:]))
        out = [
            _sample_2d(part, sy, sx, mode=mode, signed_vector=signed_vector)
            for part in flat
        ]
        sampled = np.stack(out, axis=0).reshape((*leading, *sy.shape))
    return np.moveaxis(sampled, (-2, -1), axes)


def _global_fine_coordinates(
    core_bounds: tuple[int, int, int, int],
    target_shape: tuple[int, int],
    scale: int,
    halo: int,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice]]:
    y0, y1, x0, x1 = core_bounds
    h, w = target_shape
    gy = np.arange(y0 - halo, y1 + halo, dtype=np.float64)
    gx = np.arange(x0 - halo, x1 + halo, dtype=np.float64)
    yy, xx = np.meshgrid(gy, gx, indexing="ij")
    sy = (yy + 0.5) / float(scale) - 0.5
    sx = (xx + 0.5) / float(scale) - 0.5
    crop = (slice(halo, halo + y1 - y0), slice(halo, halo + x1 - x0))
    # ``target_shape`` is intentionally accepted here because these global fine
    # indices are also the coordinate system used by deterministic detail kernels.
    if h <= 0 or w <= 0:
        raise ValueError("target shape must be positive")
    return sy, sx, crop


def _target_lat_lon(
    core_bounds: tuple[int, int, int, int], target_shape: tuple[int, int], halo: int
) -> tuple[np.ndarray, np.ndarray]:
    y0, y1, x0, x1 = core_bounds
    h, w = target_shape
    gy = np.arange(y0 - halo, y1 + halo, dtype=np.float64)
    gx = np.arange(x0 - halo, x1 + halo, dtype=np.float64)
    lat = 90.0 - (gy[:, None] + 0.5) * (180.0 / h)
    lon = -180.0 + (gx[None, :] + 0.5) * (360.0 / w)
    # Geographic coordinates outside a polar edge are mapped by reflecting
    # latitude and advancing longitude 180 degrees, matching raster topology.
    lat = np.broadcast_to(lat, (len(gy), len(gx))).copy()
    lon = np.broadcast_to(lon, lat.shape).copy()
    while np.any((lat > 90.0) | (lat < -90.0)):
        north = lat > 90.0
        south = lat < -90.0
        lat = np.where(north, 180.0 - lat, lat)
        lon = np.where(north, lon + 180.0, lon)
        lat = np.where(south, -180.0 - lat, lat)
        lon = np.where(south, lon + 180.0, lon)
    return lat, ((lon + 180.0) % 360.0) - 180.0


def _seed_for(root_seed: int, level: int, field: str) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(int(root_seed)).encode("ascii"))
    h.update(b"\0refinement\0")
    h.update(str(int(level)).encode("ascii"))
    h.update(b"\0")
    h.update(field.encode("utf-8"))
    return int.from_bytes(h.digest(), "little", signed=False)


def _spherical_detail(
    lat_deg: np.ndarray, lon_deg: np.ndarray, *, seed: int, frequency: float
) -> np.ndarray:
    """Evaluate globally continuous deterministic sub-grid structure on the sphere."""
    lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
    c = np.cos(lat)
    xyz = np.stack((c * np.cos(lon), c * np.sin(lon), np.sin(lat)), axis=-1)
    rng = np.random.default_rng(seed)
    detail = np.zeros(lat.shape, dtype=np.float64)
    total_weight = 0.0
    for octave, weight in enumerate((1.0, 0.55, 0.30)):
        freq = float(frequency) * (2.0**octave)
        for _ in range(4):
            axis = rng.normal(size=3)
            axis /= max(float(np.linalg.norm(axis)), 1e-12)
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            detail += weight * np.sin(freq * np.tensordot(xyz, axis, axes=([-1], [0])) + phase)
            total_weight += weight
    detail /= max(total_weight, 1e-12)
    std = float(np.std(detail))
    if std > 1e-12:
        detail /= std
    return detail


def _elevation_detail_amplitude(parent_elevation: np.ndarray, strength: float) -> float:
    a = np.asarray(parent_elevation, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if finite.size == 0 or strength <= 0:
        return 0.0
    lo, hi = np.percentile(finite, [2.0, 98.0])
    robust_relief = max(float(hi - lo), 0.01)
    return float(strength) * min(0.20, max(0.002, 0.03 * robust_relief))


def _apply_refinement_kernel(
    name: str,
    expanded: np.ndarray,
    axes: tuple[int, int],
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    level: int,
    root_seed: int,
    parent_height: int,
    elevation_amplitude_km: float,
) -> np.ndarray:
    if name != "elevation_km" or elevation_amplitude_km <= 0.0:
        return expanded
    moved = np.moveaxis(np.asarray(expanded, dtype=np.float64), axes, (-2, -1))
    if moved.ndim != 2:
        return expanded
    detail = _spherical_detail(
        lat,
        lon,
        seed=_seed_for(root_seed, level, name),
        frequency=max(8.0, parent_height / 5.0),
    )
    # Absolute elevation is a smooth global modulator, not a tile-local statistic;
    # therefore siblings cannot acquire different amplitudes at their shared edge.
    modulation = 0.70 + 0.30 * np.tanh(np.abs(moved) / 1.5)
    return moved + elevation_amplitude_km * modulation * detail


def _bounds_nodes(
    parent_nodes: Iterable[dict[str, Any]], level: int, spec: RefinementSpec
) -> list[RefinementNode]:
    nodes: list[RefinementNode] = []
    for parent in parent_nodes:
        py0, py1, px0, px1 = map(int, parent["core_bounds"])
        ys = _partition(py0, py1, spec.sections_y)
        xs = _partition(px0, px1, spec.sections_x)
        parent_id = str(parent["node_id"])
        for row, (y0, y1) in enumerate(ys):
            for col, (x0, x1) in enumerate(xs):
                node_id = f"{parent_id}/r{row}c{col}"
                nodes.append(
                    RefinementNode(
                        node_id=node_id,
                        parent_id=parent_id,
                        level=level,
                        row=row,
                        col=col,
                        source_bounds=(y0, y1, x0, x1),
                        core_bounds=(
                            y0 * spec.scale,
                            y1 * spec.scale,
                            x0 * spec.scale,
                            x1 * spec.scale,
                        ),
                    )
                )
    return nodes


class RefinementEngine:
    """Persisted recursive refinement graph for one generated world directory."""

    schema_version = 1

    def __init__(
        self,
        world_root: str | Path,
        *,
        spec: RefinementSpec | None = None,
        resume: bool = True,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.world_root = Path(world_root).expanduser().resolve()
        self.refine_root = self.world_root / "refinement"
        self.levels_root = self.refine_root / "levels"
        self.manifest_path = self.refine_root / "manifest.json"
        self.spec = (spec or RefinementSpec()).validate()
        self.resume = bool(resume)
        self.progress = progress
        self.refine_root.mkdir(parents=True, exist_ok=True)
        self.levels_root.mkdir(parents=True, exist_ok=True)

    def _level_dir(self, level: int) -> Path:
        return self.levels_root / f"level_{int(level):04d}"

    def _level_index_path(self, level: int) -> Path:
        return self._level_dir(level) / "index.json"

    def _load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {
            "schema_version": self.schema_version,
            "world_root": str(self.world_root),
            "deepest_complete_level": -1,
            "levels": {},
            "limitations": [
                "flow_to is not interpolated because raster receiver indices must be rebuilt by a future hydrology refinement kernel",
                "global atmosphere/ocean/tectonic solves are inherited from the parent level rather than incorrectly solved independently per tile",
            ],
        }

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        _atomic_write_json(self.manifest_path, manifest)

    def _load_index(self, level: int) -> dict[str, Any]:
        return json.loads(self._level_index_path(level).read_text(encoding="utf-8"))

    def _array_path(self, level: int, name: str) -> Path:
        return self._level_dir(level) / "arrays" / f"{name}.npy"

    def _node_dir(self, level: int, node: RefinementNode) -> Path:
        safe = node.node_id.replace("/", "__")
        return self._level_dir(level) / "nodes" / safe

    def _prepare_base_level(self, manifest: dict[str, Any]) -> None:
        if self._level_index_path(0).exists():
            if manifest.get("deepest_complete_level", -1) < 0:
                manifest["deepest_complete_level"] = 0
                self._save_manifest(manifest)
            return
        source = self.world_root / "world_arrays.npz"
        if not source.exists():
            raise FileNotFoundError(
                f"{source} does not exist; generate the base world with NPZ output before --refine"
            )
        t0 = time.perf_counter()
        _emit(self.progress, event="base_start", path="level[0]", message="materializing base NPZ for random access")
        with np.load(source, allow_pickle=False) as z:
            if "lat" not in z or "lon" not in z:
                raise ValueError("world_arrays.npz must contain lat and lon coordinates")
            height = int(len(z["lat"])); width = int(len(z["lon"]))
            entries: dict[str, Any] = {}
            arrays_dir = self._level_dir(0) / "arrays"
            arrays_dir.mkdir(parents=True, exist_ok=True)
            names = sorted(z.files)
            for i, name in enumerate(names, 1):
                arr = np.asarray(z[name])
                path = arrays_dir / f"{name}.npy"
                _atomic_save_npy(path, arr)
                axes = _spatial_axes(arr.shape, height, width)
                mode = "coordinate" if name in {"lat", "lon"} else _field_mode(name, arr.dtype)
                entries[name] = {
                    "shape": list(arr.shape),
                    "dtype": arr.dtype.str,
                    "spatial_axes": None if axes is None else list(axes),
                    "mode": mode,
                }
                _emit(
                    self.progress,
                    event="base_field",
                    path="level[0]",
                    current=i,
                    total=len(names),
                    field=name,
                )
        index = {
            "schema_version": self.schema_version,
            "level": 0,
            "resolution": [width, height],
            "source": str(source),
            "entries": entries,
            "nodes": [{"node_id": "root", "parent_id": "", "core_bounds": [0, height, 0, width]}],
            "complete": True,
        }
        _atomic_write_json(self._level_index_path(0), index)
        manifest["root_seed"] = self._read_root_seed()
        manifest["base_resolution"] = [width, height]
        manifest["deepest_complete_level"] = max(0, int(manifest.get("deepest_complete_level", -1)))
        manifest["levels"]["0"] = {
            "resolution": [width, height],
            "node_count": 1,
            "complete": True,
        }
        self._save_manifest(manifest)
        _emit(self.progress, event="base_done", path="level[0]", seconds=time.perf_counter() - t0)

    def _read_root_seed(self) -> int:
        path = self.world_root / "world.json"
        if path.exists():
            try:
                return int(json.loads(path.read_text(encoding="utf-8")).get("seed", 0))
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
        return 0

    def _parent_nodes(self, level: int, index: dict[str, Any]) -> list[dict[str, Any]]:
        nodes = index.get("nodes") or []
        if nodes:
            return list(nodes)
        width, height = map(int, index["resolution"])
        return [{"node_id": "root", "parent_id": "", "core_bounds": [0, height, 0, width]}]

    def _process_node_field(
        self,
        parent_level: int,
        target_level: int,
        node: RefinementNode,
        name: str,
        entry: dict[str, Any],
        target_shape: tuple[int, int],
        root_seed: int,
        elevation_amplitude_km: float,
    ) -> Path | None:
        mode = str(entry.get("mode", "linear"))
        if mode == "skip" or name in {"lat", "lon"}:
            return None
        axes_raw = entry.get("spatial_axes")
        if axes_raw is None:
            return None
        axes = (int(axes_raw[0]), int(axes_raw[1]))
        node_dir = self._node_dir(target_level, node)
        out_path = node_dir / "arrays" / f"{name}.npy"
        if self.resume and out_path.exists():
            return out_path

        source = np.load(self._array_path(parent_level, name), mmap_mode="r", allow_pickle=False)
        sy, sx, crop2d = _global_fine_coordinates(
            node.core_bounds,
            target_shape,
            self.spec.scale,
            self.spec.halo_cells,
        )
        signed = name.startswith(_VECTOR_COMPONENT_PREFIXES)
        expanded = _sample_spatial(source, axes, sy, sx, mode=mode, signed_vector=signed)
        lat, lon = _target_lat_lon(node.core_bounds, target_shape, self.spec.halo_cells)
        expanded = _apply_refinement_kernel(
            name,
            expanded,
            axes,
            lat=lat,
            lon=lon,
            level=target_level,
            root_seed=root_seed,
            parent_height=int(source.shape[axes[0]]),
            elevation_amplitude_km=elevation_amplitude_km,
        )
        crop = [slice(None)] * np.ndim(expanded)
        crop[axes[0]] = crop2d[0]
        crop[axes[1]] = crop2d[1]
        core = np.asarray(expanded[tuple(crop)])
        dtype = np.dtype(entry["dtype"])
        if name.startswith("true_color"):
            core = np.clip(np.rint(core), 0, 255)
        if mode == "nearest" or dtype.kind in "iu":
            core = np.rint(core) if dtype.kind in "iu" else core
        core = core.astype(dtype, copy=False)
        _atomic_save_npy(out_path, core)
        return out_path

    def _process_level(
        self,
        manifest: dict[str, Any],
        parent_level: int,
        target_level: int,
    ) -> None:
        parent_index = self._load_index(parent_level)
        parent_width, parent_height = map(int, parent_index["resolution"])
        target_height = parent_height * self.spec.scale
        target_width = parent_width * self.spec.scale
        target_shape = (target_height, target_width)
        parent_nodes = self._parent_nodes(parent_level, parent_index)
        nodes = _bounds_nodes(parent_nodes, target_level, self.spec)
        root_seed = int(manifest.get("root_seed", 0))

        elevation_amplitude = 0.0
        if "elevation_km" in parent_index["entries"]:
            parent_elevation = np.load(
                self._array_path(parent_level, "elevation_km"), mmap_mode="r", allow_pickle=False
            )
            elevation_amplitude = _elevation_detail_amplitude(
                parent_elevation, self.spec.elevation_detail_strength
            )

        target_dir = self._level_dir(target_level)
        target_dir.mkdir(parents=True, exist_ok=True)
        level_state = {
            "schema_version": self.schema_version,
            "level": target_level,
            "parent_level": parent_level,
            "resolution": [target_width, target_height],
            "spec": asdict(self.spec),
            "nodes": [asdict(node) for node in nodes],
            "complete": False,
        }
        _atomic_write_json(target_dir / "level_state.json", level_state)

        refinable = [
            name
            for name, entry in parent_index["entries"].items()
            if entry.get("spatial_axes") is not None
            and name not in {"lat", "lon"}
            and entry.get("mode") != "skip"
        ]
        total = len(nodes) * len(refinable)
        done = 0
        level_t0 = time.perf_counter()
        _emit(
            self.progress,
            event="level_start",
            path=f"level[{target_level}]",
            level=target_level,
            nodes=len(nodes),
            fields=len(refinable),
            total=total,
            resolution=[target_width, target_height],
        )

        # Elevation is processed first because ocean depth can then be derived from
        # the refined full-relief field instead of independently interpolated.
        ordered = sorted(refinable, key=lambda n: (n != "elevation_km", n == "ocean_depth_m", n))
        for node_index, node in enumerate(nodes, 1):
            node_t0 = time.perf_counter()
            for name in ordered:
                t0 = time.perf_counter()
                entry = parent_index["entries"][name]
                if name == "ocean_depth_m":
                    elevation_path = self._node_dir(target_level, node) / "arrays" / "elevation_km.npy"
                    out_path = self._node_dir(target_level, node) / "arrays" / "ocean_depth_m.npy"
                    if not (self.resume and out_path.exists()) and elevation_path.exists():
                        elev = np.load(elevation_path, mmap_mode="r", allow_pickle=False)
                        _atomic_save_npy(out_path, np.maximum(-np.asarray(elev, float) * 1000.0, 0.0).astype(np.float32))
                    else:
                        self._process_node_field(
                            parent_level,
                            target_level,
                            node,
                            name,
                            entry,
                            target_shape,
                            root_seed,
                            elevation_amplitude,
                        )
                else:
                    self._process_node_field(
                        parent_level,
                        target_level,
                        node,
                        name,
                        entry,
                        target_shape,
                        root_seed,
                        elevation_amplitude,
                    )
                done += 1
                _emit(
                    self.progress,
                    event="field_done",
                    path=f"level[{target_level}]/{node.node_id}",
                    level=target_level,
                    node=node.node_id,
                    node_index=node_index,
                    node_total=len(nodes),
                    field=name,
                    current=done,
                    total=total,
                    seconds=time.perf_counter() - t0,
                )
            _atomic_write_json(
                self._node_dir(target_level, node) / "status.json",
                {"complete": True, "node": asdict(node), "seconds": time.perf_counter() - node_t0},
            )
            _emit(
                self.progress,
                event="node_done",
                path=f"level[{target_level}]/{node.node_id}",
                node=node.node_id,
                current=node_index,
                total=len(nodes),
                seconds=time.perf_counter() - node_t0,
            )

        index = self._compose_level(parent_index, target_level, nodes, target_shape)
        level_state["complete"] = True
        level_state["seconds"] = time.perf_counter() - level_t0
        _atomic_write_json(target_dir / "level_state.json", level_state)
        level_record = {
            "resolution": [target_width, target_height],
            "node_count": len(nodes),
            "complete": True,
            "spec": asdict(self.spec),
            "index": str(self._level_index_path(target_level).relative_to(self.world_root)),
            "heightmap": str((target_dir / "maps" / "height_grayscale_16bit.png").relative_to(self.world_root)),
        }
        manifest["levels"][str(target_level)] = level_record
        manifest["latest"] = self._publish_latest_level(target_level, index)
        manifest["deepest_complete_level"] = target_level
        self._save_manifest(manifest)
        if not self.spec.keep_sections:
            shutil.rmtree(target_dir / "nodes", ignore_errors=True)
        _emit(
            self.progress,
            event="level_done",
            path=f"level[{target_level}]",
            level=target_level,
            seconds=time.perf_counter() - level_t0,
            resolution=[target_width, target_height],
            fields=len(index["entries"]),
        )

    def _compose_level(
        self,
        parent_index: dict[str, Any],
        target_level: int,
        nodes: list[RefinementNode],
        target_shape: tuple[int, int],
    ) -> dict[str, Any]:
        target_h, target_w = target_shape
        target_dir = self._level_dir(target_level)
        arrays_dir = target_dir / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)
        entries: dict[str, Any] = {}
        t0 = time.perf_counter()
        _emit(self.progress, event="compose_start", path=f"level[{target_level}]/compose")

        for name, entry in parent_index["entries"].items():
            axes_raw = entry.get("spatial_axes")
            if name == "lat":
                arr = 90.0 - (np.arange(target_h, dtype=np.float64) + 0.5) * (180.0 / target_h)
                _atomic_save_npy(arrays_dir / "lat.npy", arr)
                entries[name] = {**entry, "shape": [target_h], "dtype": arr.dtype.str, "spatial_axes": None}
                continue
            if name == "lon":
                arr = -180.0 + (np.arange(target_w, dtype=np.float64) + 0.5) * (360.0 / target_w)
                _atomic_save_npy(arrays_dir / "lon.npy", arr)
                entries[name] = {**entry, "shape": [target_w], "dtype": arr.dtype.str, "spatial_axes": None}
                continue
            if axes_raw is None:
                src = self._array_path(target_level - 1, name)
                dst = arrays_dir / f"{name}.npy"
                if not dst.exists():
                    shutil.copy2(src, dst)
                entries[name] = dict(entry)
                continue
            if entry.get("mode") == "skip":
                entries[name] = {**entry, "omitted_at_refined_levels": True}
                continue

            axes = (int(axes_raw[0]), int(axes_raw[1]))
            parent_shape = list(entry["shape"])
            out_shape = list(parent_shape)
            out_shape[axes[0]] = target_h
            out_shape[axes[1]] = target_w
            final_path = arrays_dir / f"{name}.npy"
            tmp_path = arrays_dir / f".{name}.compose.npy"
            if final_path.exists() and self.resume:
                entries[name] = {**entry, "shape": out_shape}
                continue
            mm = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.dtype(entry["dtype"]), shape=tuple(out_shape))
            for node in nodes:
                node_path = self._node_dir(target_level, node) / "arrays" / f"{name}.npy"
                if not node_path.exists():
                    raise FileNotFoundError(f"missing completed node field: {node_path}")
                part = np.load(node_path, mmap_mode="r", allow_pickle=False)
                y0, y1, x0, x1 = node.core_bounds
                sl = [slice(None)] * len(out_shape)
                sl[axes[0]] = slice(y0, y1)
                sl[axes[1]] = slice(x0, x1)
                mm[tuple(sl)] = part
            mm.flush()
            del mm
            os.replace(tmp_path, final_path)
            entries[name] = {**entry, "shape": out_shape}
            _emit(
                self.progress,
                event="compose_field",
                path=f"level[{target_level}]/compose",
                field=name,
            )

        index = {
            "schema_version": self.schema_version,
            "level": target_level,
            "resolution": [target_w, target_h],
            "entries": entries,
            "nodes": [asdict(node) for node in nodes],
            "complete": True,
            "omitted_fields": sorted(
                name for name, entry in parent_index["entries"].items() if entry.get("mode") == "skip"
            ),
        }
        _atomic_write_json(self._level_index_path(target_level), index)
        elevation_path = arrays_dir / "elevation_km.npy"
        if elevation_path.exists():
            elevation = np.load(elevation_path, mmap_mode="r", allow_pickle=False)
            maps = target_dir / "maps"
            write_heightmap_png16(
                maps / "height_grayscale_16bit.png",
                elevation,
                metadata_path=maps / "height_grayscale_16bit.json",
            )
            index["heightmap_full_relief"] = "maps/height_grayscale_16bit.png"
            _atomic_write_json(self._level_index_path(target_level), index)
        _emit(
            self.progress,
            event="compose_done",
            path=f"level[{target_level}]/compose",
            seconds=time.perf_counter() - t0,
        )
        return index

    def _publish_latest_level(
        self, target_level: int, index: dict[str, Any]
    ) -> dict[str, Any]:
        """Expose the deepest complete full-world arrays and map at stable paths.

        The authoritative arrays remain the composed per-level ``.npy`` files; the
        manifest provides stable random-access pointers without duplicating what may
        be hundreds of gigabytes.  The comparatively small PNG height product is
        atomically copied to ``maps/`` so ordinary map browsing shows the refined
        world instead of the unchanged base render.
        """
        level_dir = self._level_dir(target_level)
        arrays = {
            name: str((level_dir / "arrays" / f"{name}.npy").relative_to(self.world_root))
            for name, entry in sorted(index["entries"].items())
            if not entry.get("omitted_at_refined_levels")
            and (level_dir / "arrays" / f"{name}.npy").exists()
        }
        source_height = level_dir / "maps" / "height_grayscale_16bit.png"
        source_height_meta = level_dir / "maps" / "height_grayscale_16bit.json"
        latest_height = self.world_root / "maps" / "02c_height_refined_latest_16bit.png"
        latest_height_meta = self.world_root / "maps" / "02c_height_refined_latest_16bit.json"
        if source_height.exists():
            _atomic_copy_file(source_height, latest_height)
        if source_height_meta.exists():
            _atomic_copy_file(source_height_meta, latest_height_meta)

        payload: dict[str, Any] = {
            "schema_version": 1,
            "level": int(target_level),
            "resolution": list(map(int, index["resolution"])),
            "index": str(self._level_index_path(target_level).relative_to(self.world_root)),
            "arrays": arrays,
            "heightmap_full_relief": (
                str(latest_height.relative_to(self.world_root)) if source_height.exists() else None
            ),
            "storage_semantics": (
                "arrays are complete composed full-world static files; section payloads are not required"
            ),
        }
        _atomic_write_json(self.refine_root / "latest.json", payload)
        _atomic_write_json(self.world_root / "maps" / "refinement_latest.json", payload)
        return {
            "level": int(target_level),
            "resolution": list(map(int, index["resolution"])),
            "manifest": str((self.refine_root / "latest.json").relative_to(self.world_root)),
            "heightmap": payload["heightmap_full_relief"],
        }

    def refine(self, levels: int = 1) -> dict[str, Any]:
        if int(levels) < 1:
            raise ValueError("levels must be >= 1")
        manifest = self._load_manifest()
        self._prepare_base_level(manifest)
        completed = 0
        while completed < int(levels):
            parent_level = int(manifest.get("deepest_complete_level", 0))
            target_level = parent_level + 1
            # If a crash left a target directory without a completed manifest level,
            # processing the same target with ``resume=True`` reuses every atomic node field.
            self._process_level(manifest, parent_level, target_level)
            completed += 1
        return manifest


__all__ = ["RefinementEngine", "RefinementNode", "RefinementSpec"]
