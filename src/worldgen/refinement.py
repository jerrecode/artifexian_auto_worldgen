from __future__ import annotations

"""Public recursive-refinement API and hot-path optimizations.

The durable implementation lives in :mod:`worldgen.refinement_core`. This module
adds invariants that are important enough to keep explicit at the public API:

* sub-grid detail is normalized globally-by-construction and cannot depend on how
  the sphere happened to be partitioned into child sections;
* spherical interpolation geometry is prepared once per child and reused by every
  field in that child instead of rebuilding the same large index/weight arrays;
* resumability is provenance-safe: incomplete child payloads are never reused under
  a different refinement specification, and a changed base NPZ cannot silently
  inherit an older refinement tree.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from . import refinement_core as _core
from .refinement_core import *  # noqa: F401,F403


def _partition_invariant_spherical_detail(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    *,
    seed: int,
    frequency: float,
) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
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
            detail += weight * np.sin(
                freq * np.tensordot(xyz, axis, axes=([-1], [0])) + phase
            )
            total_weight += weight
    return detail / max(total_weight, 1e-12)


_core._spherical_detail = _partition_invariant_spherical_detail


def _file_sha256(path: Path, *, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _source_fingerprint(path: Path, *, include_hash: bool = True) -> dict[str, Any]:
    st = path.stat()
    out: dict[str, Any] = {
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }
    if include_hash:
        out["sha256"] = _file_sha256(path)
    return out


def _sample_with_context(
    values: np.ndarray,
    axes: tuple[int, int],
    context: dict[str, Any],
    *,
    mode: str,
    signed_vector: bool,
) -> np.ndarray:
    """Sample a spatial field using one child node's reusable interpolation plan."""
    a = np.asarray(values)
    moved = np.moveaxis(a, axes, (-2, -1))
    sy = context["sy"]
    sx = context["sx"]

    def sample_one(part: np.ndarray) -> np.ndarray:
        if mode == "nearest":
            return part[context["nearest_y"], context["nearest_x"]]
        if signed_vector:
            return _core._signed_bilinear_sample(part, sy, sx)
        return _core.apply_bilinear_sampler(part, context["linear_sampler"])

    leading = moved.shape[:-2]
    if not leading:
        sampled = sample_one(moved)
    else:
        flat = moved.reshape((-1, *moved.shape[-2:]))
        sampled = np.stack([sample_one(part) for part in flat], axis=0).reshape(
            (*leading, *sy.shape)
        )
    return np.moveaxis(sampled, (-2, -1), axes)


class RefinementEngine(_core.RefinementEngine):
    """Refinement engine with bounded sampler reuse and provenance-safe resume."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sampling_context_key: tuple[Any, ...] | None = None
        self._sampling_context: dict[str, Any] | None = None

    def _reset_sampling_context(self) -> None:
        self._sampling_context_key = None
        self._sampling_context = None

    def _reset_refinement_levels(self, manifest: dict[str, Any]) -> None:
        shutil.rmtree(self.levels_root, ignore_errors=True)
        self.levels_root.mkdir(parents=True, exist_ok=True)
        manifest["deepest_complete_level"] = -1
        manifest["levels"] = {}
        self._reset_sampling_context()
        self._save_manifest(manifest)

    def _prepare_base_level(self, manifest: dict[str, Any]) -> None:
        source = self.world_root / "world_arrays.npz"
        index_path = self._level_index_path(0)
        if index_path.exists() and source.exists():
            index = self._load_index(0)
            stored = index.get("source_fingerprint")
            if isinstance(stored, dict) and stored.get("sha256"):
                st = source.stat()
                stat_same = (
                    int(stored.get("size_bytes", -1)) == int(st.st_size)
                    and int(stored.get("mtime_ns", -1)) == int(st.st_mtime_ns)
                )
                if not stat_same:
                    current_hash = _file_sha256(source)
                    if current_hash != str(stored["sha256"]):
                        if self.resume:
                            raise RuntimeError(
                                "world_arrays.npz changed after refinement level 0 was materialized; "
                                "rerun with --no-resume to rebuild the refinement hierarchy from the new base world"
                            )
                        self._reset_refinement_levels(manifest)
                    else:
                        stored["size_bytes"] = int(st.st_size)
                        stored["mtime_ns"] = int(st.st_mtime_ns)
                        index["source_fingerprint"] = stored
                        _core._atomic_write_json(index_path, index)

        super()._prepare_base_level(manifest)

        # Record a strong fingerprint after first materialization. The fast stat
        # fields avoid rehashing a multi-GB NPZ on ordinary resume; SHA-256 is
        # recomputed only when size/mtime indicates that the source may have changed.
        source = self.world_root / "world_arrays.npz"
        if source.exists():
            index = self._load_index(0)
            if not isinstance(index.get("source_fingerprint"), dict) or not index["source_fingerprint"].get("sha256"):
                index["source_fingerprint"] = _source_fingerprint(source, include_hash=True)
                _core._atomic_write_json(index_path, index)

    def _node_sampling_context(
        self,
        target_level: int,
        node: _core.RefinementNode,
        target_shape: tuple[int, int],
    ) -> dict[str, Any]:
        key = (
            int(target_level),
            node.node_id,
            tuple(map(int, target_shape)),
            int(self.spec.scale),
            int(self.spec.halo_cells),
        )
        if key == self._sampling_context_key and self._sampling_context is not None:
            return self._sampling_context

        sy, sx, crop2d = _core._global_fine_coordinates(
            node.core_bounds,
            target_shape,
            self.spec.scale,
            self.spec.halo_cells,
        )
        parent_shape = (
            int(target_shape[0]) // int(self.spec.scale),
            int(target_shape[1]) // int(self.spec.scale),
        )
        nearest_y = np.floor(sy + 0.5).astype(np.int64)
        nearest_x = np.floor(sx + 0.5).astype(np.int64)
        nearest_y, nearest_x = _core._map_spherical_lattice_indices(
            nearest_y, nearest_x, parent_shape
        )
        context = {
            "sy": sy,
            "sx": sx,
            "crop2d": crop2d,
            "lat_lon": _core._target_lat_lon(
                node.core_bounds, target_shape, self.spec.halo_cells
            ),
            "nearest_y": nearest_y,
            "nearest_x": nearest_x,
            "linear_sampler": _core.prepare_spherical_bilinear_sampler(
                sy, sx, parent_shape
            ),
        }
        self._sampling_context_key = key
        self._sampling_context = context
        return context

    def _process_level(
        self,
        manifest: dict[str, Any],
        parent_level: int,
        target_level: int,
    ) -> None:
        parent_index = self._load_index(parent_level)
        parent_width, parent_height = map(int, parent_index["resolution"])
        expected_resolution = [
            parent_width * int(self.spec.scale),
            parent_height * int(self.spec.scale),
        ]
        target_dir = self._level_dir(target_level)
        state_path = target_dir / "level_state.json"

        if target_dir.exists():
            if not self.resume:
                shutil.rmtree(target_dir, ignore_errors=True)
                self._reset_sampling_context()
            elif state_path.exists():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"incomplete refinement state is unreadable: {state_path}; use --no-resume to rebuild it"
                    ) from exc
                compatible = (
                    int(state.get("parent_level", -1)) == int(parent_level)
                    and list(state.get("resolution", [])) == expected_resolution
                    and state.get("spec") == asdict(self.spec)
                )
                if not compatible:
                    raise RuntimeError(
                        "incomplete refinement payloads were created with a different parent/specification; "
                        "use --no-resume to discard that incomplete depth before changing --refine-scale, "
                        "--refine-sections, --refine-halo-cells, or detail settings"
                    )
            elif any(target_dir.iterdir()):
                raise RuntimeError(
                    f"refinement target {target_dir} contains untracked partial data; use --no-resume to rebuild it"
                )

        super()._process_level(manifest, parent_level, target_level)

    def _process_node_field(
        self,
        parent_level: int,
        target_level: int,
        node: _core.RefinementNode,
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

        source = np.load(
            self._array_path(parent_level, name), mmap_mode="r", allow_pickle=False
        )
        context = self._node_sampling_context(target_level, node, target_shape)
        signed = name.startswith(_core._VECTOR_COMPONENT_PREFIXES)
        expanded = _sample_with_context(
            source,
            axes,
            context,
            mode=mode,
            signed_vector=signed,
        )
        lat, lon = context["lat_lon"]
        expanded = _core._apply_refinement_kernel(
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
        crop2d = context["crop2d"]
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
        _core._atomic_save_npy(out_path, core)
        return out_path


__all__ = _core.__all__
