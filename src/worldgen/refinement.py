from __future__ import annotations

"""Public recursive-refinement API and hot-path optimizations.

The durable implementation lives in :mod:`worldgen.refinement_core`. This module
adds two invariants that are important enough to keep explicit at the public API:

* sub-grid detail is normalized globally-by-construction and cannot depend on how
  the sphere happened to be partitioned into child sections;
* spherical interpolation geometry is prepared once per child and reused by every
  field in that child instead of rebuilding the same large index/weight arrays for
  dozens of climate/ocean/geology/resource layers.
"""

from pathlib import Path
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
            # Pole-reflected tangent vectors need corner-specific sign parity, so
            # they retain the specialized sampler rather than the scalar plan.
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
    """Refinement engine with bounded one-node spherical sampling-plan reuse."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sampling_context_key: tuple[Any, ...] | None = None
        self._sampling_context: dict[str, Any] | None = None

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
        # Keep exactly one child plan resident. This captures the dominant reuse
        # pattern (many fields per node) without letting a deep refinement tree turn
        # interpolation caches into an unbounded RAM consumer.
        self._sampling_context_key = key
        self._sampling_context = context
        return context

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


# Preserve the public names of the core module while replacing its engine with the
# optimized subclass above.
__all__ = _core.__all__
