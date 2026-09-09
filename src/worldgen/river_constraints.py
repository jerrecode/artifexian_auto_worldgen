from __future__ import annotations

"""Hierarchical major-river constraints for sparse local refinement.

Continental drainage is solved by the global world. Local high-resolution terrain is
therefore not allowed to rediscover or reroute major rivers from scratch. This module
projects the global river mask, stream order, discharge and width proxies into one
cube-sphere tile and turns them into an explicit channel-floor constraint with stable
quadtree ancestry metadata.
"""

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np
from scipy import ndimage

from .lod import parent_chain
from .planet_tiles import PlanetTilePyramid, TileKey, tile_geometry


@dataclass(slots=True, frozen=True)
class RiverConstraintSpec:
    minimum_major_stream_order: int = 3
    minimum_channel_depth_m: float = 0.5
    maximum_channel_depth_m: float = 8.0
    corridor_half_width_cells: int = 7
    corridor_sigma_cells: float = 3.0
    corridor_mask_threshold: float = 0.12

    def validate(self) -> "RiverConstraintSpec":
        if int(self.minimum_major_stream_order) < 1:
            raise ValueError("minimum_major_stream_order must be >= 1")
        if not math.isfinite(float(self.minimum_channel_depth_m)) or self.minimum_channel_depth_m < 0:
            raise ValueError("minimum_channel_depth_m must be finite and non-negative")
        if not math.isfinite(float(self.maximum_channel_depth_m)) or self.maximum_channel_depth_m < self.minimum_channel_depth_m:
            raise ValueError("maximum_channel_depth_m must be finite and >= minimum_channel_depth_m")
        if not 1 <= int(self.corridor_half_width_cells) <= 64:
            raise ValueError("corridor_half_width_cells must be in [1,64]")
        if not math.isfinite(float(self.corridor_sigma_cells)) or self.corridor_sigma_cells <= 0.0:
            raise ValueError("corridor_sigma_cells must be positive")
        if not 0.0 < float(self.corridor_mask_threshold) < 1.0:
            raise ValueError("corridor_mask_threshold must be in (0,1)")
        return self


@dataclass(slots=True, frozen=True)
class RiverConstraintResult:
    major_river_mask: np.ndarray
    parent_stream_order: np.ndarray
    parent_discharge_index: np.ndarray
    parent_width_proxy: np.ndarray
    constraint_strength: np.ndarray
    channel_floor_m: np.ndarray
    metadata: Mapping[str, object]


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


class RiverConstraintGenerator:
    def __init__(
        self,
        pyramid: PlanetTilePyramid,
        *,
        spec: RiverConstraintSpec | None = None,
    ) -> None:
        self.pyramid = pyramid
        self.spec = (spec or RiverConstraintSpec()).validate()
        self.root = pyramid.root / "derived" / "river_constraints_v1"

    def _path(self, key: TileKey, field: str) -> Path:
        return (
            self.root / field / f"z{key.level:02d}" / key.face
            / f"x{key.x:08d}" / f"y{key.y:08d}.npy"
        )

    def _metadata_path(self, key: TileKey) -> Path:
        return (
            self.root / "metadata" / f"z{key.level:02d}" / key.face
            / f"x{key.x:08d}" / f"y{key.y:08d}.json"
        )

    def _sample_optional(self, name: str, geom, default: float = 0.0) -> np.ndarray:
        _shape, fields = self.pyramid._source_metadata()
        if name not in fields:
            return np.full(geom.latitude_deg.shape, default, dtype=np.float64)
        return np.asarray(self.pyramid._sample_source_field(name, geom), dtype=np.float64)

    def _load_cached(self, key: TileKey) -> RiverConstraintResult | None:
        names = (
            "major_river_mask",
            "parent_stream_order",
            "parent_discharge_index",
            "parent_width_proxy",
            "constraint_strength",
            "channel_floor_m",
        )
        paths = {name: self._path(key, name) for name in names}
        meta = self._metadata_path(key)
        if not meta.exists() or not all(path.exists() for path in paths.values()):
            return None
        return RiverConstraintResult(
            major_river_mask=np.load(paths["major_river_mask"], mmap_mode="r", allow_pickle=False),
            parent_stream_order=np.load(paths["parent_stream_order"], mmap_mode="r", allow_pickle=False),
            parent_discharge_index=np.load(paths["parent_discharge_index"], mmap_mode="r", allow_pickle=False),
            parent_width_proxy=np.load(paths["parent_width_proxy"], mmap_mode="r", allow_pickle=False),
            constraint_strength=np.load(paths["constraint_strength"], mmap_mode="r", allow_pickle=False),
            channel_floor_m=np.load(paths["channel_floor_m"], mmap_mode="r", allow_pickle=False),
            metadata=json.loads(meta.read_text(encoding="utf-8")),
        )

    def generate(self, key: TileKey) -> RiverConstraintResult:
        key.validate()
        cached = self._load_cached(key)
        if cached is not None:
            return cached
        geom = tile_geometry(key, self.pyramid.spec.tile_size)
        elevation = np.asarray(
            self.pyramid._sample_source_field("elevation_m", geom), dtype=np.float64
        )
        rivers = self._sample_optional("rivers", geom)
        order = np.maximum(self._sample_optional("stream_order", geom), 0.0)
        discharge = np.clip(self._sample_optional("discharge_index", geom), 0.0, None)
        width = np.clip(self._sample_optional("river_width_proxy", geom), 0.0, None)
        major_raw = (
            (rivers >= 0.5)
            | (order >= int(self.spec.minimum_major_stream_order))
        )
        major_raw &= elevation >= 0.0

        max_order = max(float(np.max(order)), 1.0)
        order_norm = np.clip(order / max_order, 0.0, 1.0)
        discharge_norm = np.clip(discharge, 0.0, 1.0)
        width_norm = (
            width / max(float(np.max(width)), 1.0e-12)
            if np.any(width > 0)
            else np.zeros_like(width)
        )
        raw_strength = np.maximum.reduce(
            (order_norm, discharge_norm, np.clip(width_norm, 0.0, 1.0))
        )
        raw_strength *= major_raw

        if np.any(major_raw):
            distance = ndimage.distance_transform_edt(~major_raw)
            sigma = max(float(self.spec.corridor_sigma_cells), 0.5)
            support = np.exp(-0.5 * np.square(distance / sigma))
            support[
                distance > float(self.spec.corridor_half_width_cells)
            ] = 0.0
            # Carry the parent river importance laterally into the corridor while
            # allowing the local D16 hydrology to choose its actual centreline.
            strength_seed = ndimage.maximum_filter(
                raw_strength,
                size=2 * int(self.spec.corridor_half_width_cells) + 1,
                mode="nearest",
            )
            strength = (
                np.clip(strength_seed, 0.0, 1.0)
                * support
                * (elevation >= 0.0)
            )
        else:
            support = np.zeros_like(elevation, dtype=np.float64)
            strength = np.zeros_like(elevation, dtype=np.float64)
        major = (
            support >= float(self.spec.corridor_mask_threshold)
        ) & (elevation >= 0.0)

        depth = (
            float(self.spec.minimum_channel_depth_m)
            + (
                float(self.spec.maximum_channel_depth_m)
                - float(self.spec.minimum_channel_depth_m)
            )
            * np.clip(strength, 0.0, 1.0)
        )
        # This is a shallow topology guide, not the final channel. The actual
        # centreline and final incision are recomputed on evolved high-resolution
        # terrain after geomorphology.
        floor = elevation - depth * np.clip(strength, 0.0, 1.0)

        arrays = {
            "major_river_mask": major.astype(np.bool_),
            "parent_stream_order": np.rint(order).astype(np.int16),
            "parent_discharge_index": discharge.astype(np.float32),
            "parent_width_proxy": width.astype(np.float32),
            "constraint_strength": strength.astype(np.float32),
            "channel_floor_m": floor.astype(np.float32),
        }
        for name, values in arrays.items():
            _atomic_save_npy(self._path(key, name), values)
        ancestry = [asdict(parent) for parent in parent_chain(key)]
        metadata = {
            "schema_version": 2,
            "key": asdict(key),
            "quadtree_ancestry_root_to_parent": ancestry,
            "source_sha256": self.pyramid._source_hash(),
            "spec": asdict(self.spec),
            "raw_parent_river_cells": int(np.count_nonzero(major_raw)),
            "major_river_corridor_cells": int(np.count_nonzero(major)),
            "semantics": (
                "continental river topology is projected as a shallow broad corridor; "
                "the high-resolution D16 solver selects a terrain-conforming and "
                "meandering centreline inside that corridor rather than copying the "
                "coarse parent raster line"
            ),
        }
        _atomic_json(self._metadata_path(key), metadata)
        return RiverConstraintResult(metadata=metadata, **arrays)


__all__ = [
    "RiverConstraintGenerator",
    "RiverConstraintResult",
    "RiverConstraintSpec",
]
