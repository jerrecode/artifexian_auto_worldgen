from __future__ import annotations

"""Bounded tile-local geomorphology constrained by global river topology.

This is a finite-domain refinement operator, not a replacement for the global
landscape evolution model. It consumes the open-boundary local hydrology and
hierarchical river constraints, applies bounded stream-power incision, mass-accounted
sediment retention/deposition and conservative hillslope smoothing, then anchors the
entire perturbation to zero at the tile perimeter so independently generated tiles
remain watertight.
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

from .local_hydrology import LocalHydrologySolver, _sample_area_km2
from .local_orography import edge_anchor_taper, terrain_frame
from .planet_tiles import PlanetTilePyramid, TileKey, tile_geometry
from .river_constraints import RiverConstraintGenerator


@dataclass(slots=True, frozen=True)
class LocalGeomorphologySpec:
    stream_power_area_exponent: float = 0.5
    stream_power_slope_exponent: float = 1.0
    drainage_area_scale_km2: float = 40.0
    max_fluvial_erosion_m: float = 4.0
    sediment_retention_fraction: float = 0.45
    max_deposition_m: float = 3.0
    hillslope_diffusion_fraction: float = 0.12
    edge_anchor_cells: int = 4

    def validate(self) -> "LocalGeomorphologySpec":
        for name in (
            "stream_power_area_exponent",
            "stream_power_slope_exponent",
            "drainage_area_scale_km2",
            "max_fluvial_erosion_m",
            "max_deposition_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.drainage_area_scale_km2 <= 0:
            raise ValueError("drainage_area_scale_km2 must be positive")
        if not 0.0 <= float(self.sediment_retention_fraction) <= 1.0:
            raise ValueError("sediment_retention_fraction must be in [0,1]")
        if not 0.0 <= float(self.hillslope_diffusion_fraction) <= 0.5:
            raise ValueError("hillslope_diffusion_fraction must be in [0,0.5]")
        if int(self.edge_anchor_cells) < 1:
            raise ValueError("edge_anchor_cells must be >= 1")
        return self


@dataclass(slots=True, frozen=True)
class LocalGeomorphologyResult:
    elevation_m: np.ndarray
    erosion_m: np.ndarray
    deposition_m: np.ndarray
    hillslope_adjustment_m: np.ndarray
    major_river_constraint: np.ndarray
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


class LocalGeomorphologySolver:
    def __init__(
        self,
        pyramid: PlanetTilePyramid,
        *,
        spec: LocalGeomorphologySpec | None = None,
    ) -> None:
        self.pyramid = pyramid
        self.spec = (spec or LocalGeomorphologySpec()).validate()
        self.hydrology = LocalHydrologySolver(pyramid)
        self.rivers = RiverConstraintGenerator(pyramid)
        self.root = pyramid.root / "derived" / "local_geomorphology_v1"

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

    def _load_cached(self, key: TileKey) -> LocalGeomorphologyResult | None:
        names = (
            "elevation_m",
            "erosion_m",
            "deposition_m",
            "hillslope_adjustment_m",
            "major_river_constraint",
        )
        paths = {name: self._path(key, name) for name in names}
        meta = self._metadata_path(key)
        if not meta.exists() or not all(path.exists() for path in paths.values()):
            return None
        return LocalGeomorphologyResult(
            elevation_m=np.load(paths["elevation_m"], mmap_mode="r", allow_pickle=False),
            erosion_m=np.load(paths["erosion_m"], mmap_mode="r", allow_pickle=False),
            deposition_m=np.load(paths["deposition_m"], mmap_mode="r", allow_pickle=False),
            hillslope_adjustment_m=np.load(paths["hillslope_adjustment_m"], mmap_mode="r", allow_pickle=False),
            major_river_constraint=np.load(paths["major_river_constraint"], mmap_mode="r", allow_pickle=False),
            metadata=json.loads(meta.read_text(encoding="utf-8")),
        )

    def solve(self, key: TileKey) -> LocalGeomorphologyResult:
        key.validate()
        cached = self._load_cached(key)
        if cached is not None:
            return cached
        cfg = self.spec
        geom = tile_geometry(key, self.pyramid.spec.tile_size)
        base = np.asarray(self.pyramid.load_field(key, "elevation_m"), dtype=np.float64)
        land = base >= 0.0
        taper = edge_anchor_taper(base.shape, cfg.edge_anchor_cells)
        area_km2 = _sample_area_km2(geom.xyz, self.pyramid.planet_radius_m)
        area_m2 = area_km2 * 1.0e6
        hydro = self.hydrology.solve(key)
        river = self.rivers.generate(key)
        _normal, slope_deg, _ge, _gs = terrain_frame(
            geom.xyz, base, self.pyramid.planet_radius_m
        )
        slope = np.tan(np.deg2rad(np.clip(slope_deg, 0.0, 80.0)))
        drainage = np.maximum(np.asarray(hydro.drainage_area_km2, dtype=np.float64), 0.0)
        discharge = np.clip(np.asarray(hydro.discharge_index, dtype=np.float64), 0.0, 1.0)
        area_term = (drainage / (drainage + cfg.drainage_area_scale_km2)) ** cfg.stream_power_area_exponent
        slope_term = np.maximum(slope, 0.0) ** cfg.stream_power_slope_exponent
        stream_power = area_term * slope_term * (0.25 + 0.75 * discharge)
        stream_power = np.clip(stream_power, 0.0, 1.0)
        fluvial = cfg.max_fluvial_erosion_m * stream_power * land * taper

        major = np.asarray(river.major_river_mask, dtype=bool) & land
        channel_floor = np.asarray(river.channel_floor_m, dtype=np.float64)
        # The global river constraint may require a local high-frequency ridge to be
        # lowered, but never raises an already-lower resolved channel.
        channel_lowering = np.maximum(base - channel_floor, 0.0) * major * taper
        erosion = np.maximum(fluvial, channel_lowering)
        eroded_volume_m3 = float(np.sum(erosion * area_m2))

        slope_norm = np.clip(slope / 0.20, 0.0, 1.0)
        strength = np.clip(np.asarray(river.constraint_strength, dtype=np.float64), 0.0, 1.0)
        deposit_weights = (
            land
            * taper
            * (0.15 + 0.85 * discharge)
            * (1.0 - slope_norm)
            * (1.0 - 0.85 * strength)
        )
        target_retained = eroded_volume_m3 * cfg.sediment_retention_fraction
        denom = float(np.sum(deposit_weights * area_m2))
        if target_retained > 0.0 and denom > 0.0:
            deposition = target_retained * deposit_weights / denom
            deposition = np.minimum(deposition, cfg.max_deposition_m)
        else:
            deposition = np.zeros_like(base)
        deposited_volume_m3 = float(np.sum(deposition * area_m2))
        exported_volume_m3 = max(eroded_volume_m3 - deposited_volume_m3, 0.0)

        evolved = base - erosion + deposition
        smoothed = ndimage.gaussian_filter(evolved, sigma=1.0, mode="nearest")
        hillslope = cfg.hillslope_diffusion_fraction * (smoothed - evolved) * land * taper
        # Remove any area-weighted net terrain volume introduced by the finite-domain
        # smoothing step. The correction also vanishes at the perimeter.
        net = float(np.sum(hillslope * area_m2))
        correction_weight = land * taper
        correction_denom = float(np.sum(correction_weight * area_m2))
        if correction_denom > 0.0:
            hillslope -= correction_weight * (net / correction_denom)
        evolved = evolved + hillslope
        # Reapply exact parent/base perimeter anchoring after all floating operations.
        anchored = base + taper * (evolved - base)
        anchored[0, :] = base[0, :]
        anchored[-1, :] = base[-1, :]
        anchored[:, 0] = base[:, 0]
        anchored[:, -1] = base[:, -1]

        closure = (
            abs(eroded_volume_m3 - deposited_volume_m3 - exported_volume_m3)
            / max(eroded_volume_m3, 1.0)
        )
        arrays = {
            "elevation_m": anchored.astype(np.float32),
            "erosion_m": erosion.astype(np.float32),
            "deposition_m": deposition.astype(np.float32),
            "hillslope_adjustment_m": hillslope.astype(np.float32),
            "major_river_constraint": major.astype(np.bool_),
        }
        for name, values in arrays.items():
            _atomic_save_npy(self._path(key, name), values)
        metadata = {
            "schema_version": 1,
            "key": asdict(key),
            "source_sha256": self.pyramid._source_hash(),
            "spec": asdict(cfg),
            "upstream": {
                "hydrology": "local_hydrology_v1",
                "river_constraints": "river_constraints_v1",
            },
            "eroded_sediment_volume_m3": eroded_volume_m3,
            "deposited_sediment_volume_m3": deposited_volume_m3,
            "exported_sediment_volume_m3": exported_volume_m3,
            "sediment_closure_relative": closure,
            "major_river_constraint_cells": int(np.count_nonzero(major)),
            "boundary_semantics": "all geomorphic terrain perturbations vanish on the tile perimeter; output core remains watertight with independently generated neighbours",
            "limitations": [
                "single bounded local evolution pass uses the current local drainage graph rather than iterating hydrology to full landscape equilibrium",
                "continental drainage/discharge remains inherited from the global hierarchy",
            ],
        }
        _atomic_json(self._metadata_path(key), metadata)
        return LocalGeomorphologyResult(metadata=metadata, **arrays)


__all__ = [
    "LocalGeomorphologyResult",
    "LocalGeomorphologySolver",
    "LocalGeomorphologySpec",
]
