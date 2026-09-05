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
from .planet_tiles import (
    PlanetTilePyramid,
    TileKey,
    approximate_meters_per_sample,
    tile_geometry,
)
from .procedural_erosion import phase_cell_octave_xyz
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
    procedural_detail_enabled: bool = True
    procedural_octaves: int = 3
    procedural_base_wavelength_samples: float = 36.0
    procedural_min_samples_per_wavelength: float = 5.0
    procedural_amplitude_m: float = 1.25
    procedural_gain: float = 0.52
    procedural_lacunarity: float = 2.0
    procedural_cell_scale: float = 0.72
    procedural_steering_strength: float = 0.28

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
        if not isinstance(self.procedural_detail_enabled, bool):
            raise TypeError("procedural_detail_enabled must be bool")
        if not 1 <= int(self.procedural_octaves) <= 8:
            raise ValueError("procedural_octaves must be in [1,8]")
        for name in (
            "procedural_base_wavelength_samples",
            "procedural_min_samples_per_wavelength",
            "procedural_amplitude_m",
            "procedural_cell_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(float(self.procedural_gain)) or not 0.0 < float(self.procedural_gain) <= 1.0:
            raise ValueError("procedural_gain must be in (0,1]")
        if not math.isfinite(float(self.procedural_lacunarity)) or float(self.procedural_lacunarity) <= 1.0:
            raise ValueError("procedural_lacunarity must be > 1")
        if not math.isfinite(float(self.procedural_steering_strength)) or float(self.procedural_steering_strength) < 0.0:
            raise ValueError("procedural_steering_strength must be finite and non-negative")
        return self


@dataclass(slots=True, frozen=True)
class LocalGeomorphologyResult:
    elevation_m: np.ndarray
    erosion_m: np.ndarray
    deposition_m: np.ndarray
    hillslope_adjustment_m: np.ndarray
    procedural_detail_m: np.ndarray
    procedural_coherence: np.ndarray
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
            "procedural_detail_m",
            "procedural_coherence",
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
            procedural_detail_m=np.load(paths["procedural_detail_m"], mmap_mode="r", allow_pickle=False),
            procedural_coherence=np.load(paths["procedural_coherence"], mmap_mode="r", allow_pickle=False),
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
        _normal, slope_deg, grad_east, grad_south = terrain_frame(
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

        # Add unresolved deterministic phase-cell morphology without changing the
        # physical sediment ledger. The field is zero-area-mean on active land and
        # tapers exactly to the inherited parent terrain at the tile boundary.
        procedural = np.zeros_like(base)
        procedural_coherence = np.zeros_like(base)
        executed_octaves = 0
        if cfg.procedural_detail_enabled and np.any(land):
            sample_m = approximate_meters_per_sample(
                self.pyramid.planet_radius_m,
                key.level,
                self.pyramid.spec.tile_size,
            )
            base_wavelength_m = sample_m * float(cfg.procedural_base_wavelength_samples)
            runoff_n = 1.0 - np.exp(
                -np.maximum(np.asarray(hydro.runoff_mm_year, dtype=np.float64), 0.0) / 650.0
            )
            discharge_n = np.clip(
                np.asarray(hydro.discharge_index, dtype=np.float64), 0.0, 1.0
            )
            local_strength = (
                np.sqrt(runoff_n * (0.20 + 0.80 * discharge_n))
                * (0.30 + 0.70 * np.clip(slope / 0.15, 0.0, 1.0))
                * land
            )

            direction_s = -np.asarray(grad_south, dtype=np.float64)
            direction_e = -np.asarray(grad_east, dtype=np.float64)
            direction_norm = np.hypot(direction_s, direction_e)
            direction_s = np.divide(
                direction_s, np.maximum(direction_norm, 1.0e-12)
            )
            direction_e = np.divide(
                direction_e,
                np.maximum(direction_norm, 1.0e-12),
                out=np.ones_like(direction_e),
                where=direction_norm > 1.0e-12,
            )

            unit = np.asarray(geom.xyz, dtype=np.float64)
            zaxis = np.zeros_like(unit)
            zaxis[..., 2] = 1.0
            east_basis = np.cross(zaxis, unit)
            east_norm = np.linalg.norm(east_basis, axis=-1, keepdims=True)
            polar = east_norm[..., 0] < 1.0e-10
            if np.any(polar):
                fallback = np.zeros_like(unit)
                fallback[..., 1] = 1.0
                east_basis[polar] = np.cross(fallback[polar], unit[polar])
                east_norm = np.linalg.norm(east_basis, axis=-1, keepdims=True)
            east_basis /= np.maximum(east_norm, 1.0e-15)
            south_basis = -np.cross(unit, east_basis)

            phase_mask = np.zeros_like(base)
            raw_detail = np.zeros_like(base)
            for octave in range(int(cfg.procedural_octaves)):
                wavelength_m = base_wavelength_m / (
                    float(cfg.procedural_lacunarity) ** octave
                )
                if wavelength_m < float(cfg.procedural_min_samples_per_wavelength) * sample_m:
                    break
                direction_xyz = (
                    direction_s[..., None] * south_basis
                    + direction_e[..., None] * east_basis
                )
                perpendicular = np.cross(unit, direction_xyz)
                cosine, sine, coherence = phase_cell_octave_xyz(
                    unit,
                    self.pyramid.planet_radius_m / 1000.0,
                    np.full(base.shape, wavelength_m / 1000.0, dtype=np.float64),
                    perpendicular,
                    cell_scale=float(cfg.procedural_cell_scale),
                    seed=int(self.pyramid._read_seed()) ^ 0x4C4F43414C,
                    octave=octave + 8,
                )
                phase_mask = 1.0 - (1.0 - phase_mask) * (
                    1.0 - np.clip(coherence * (0.55 + 0.45 * local_strength), 0.0, 1.0)
                )
                raw_detail += (
                    float(cfg.procedural_amplitude_m)
                    * (float(cfg.procedural_gain) ** octave)
                    * local_strength
                    * phase_mask
                    * cosine
                )
                procedural_coherence = np.maximum(procedural_coherence, coherence)
                turn = (
                    float(cfg.procedural_steering_strength)
                    * np.sign(sine)
                    * coherence
                    * local_strength
                    * (float(cfg.procedural_gain) ** octave)
                )
                ct, st = np.cos(turn), np.sin(turn)
                new_s = direction_s * ct - direction_e * st
                new_e = direction_s * st + direction_e * ct
                new_norm = np.hypot(new_s, new_e)
                direction_s = np.divide(new_s, np.maximum(new_norm, 1.0e-12))
                direction_e = np.divide(new_e, np.maximum(new_norm, 1.0e-12))
                executed_octaves += 1

            active_weight = area_m2 * land * taper
            denom_detail = float(np.sum(active_weight))
            if denom_detail > 0.0:
                mean_detail = float(np.sum(raw_detail * active_weight) / denom_detail)
                procedural = (raw_detail - mean_detail) * land * taper
        # Reapply the general finite-domain anchoring once, then add the
        # procedural perturbation exactly once. The procedural field already
        # contains its edge taper; applying the outer taper to it again would
        # square the taper and change its area-weighted zero-mean semantics.
        anchored = base + taper * (evolved - base) + procedural
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
            "procedural_detail_m": procedural.astype(np.float32),
            "procedural_coherence": procedural_coherence.astype(np.float32),
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
            "procedural_detail_enabled": bool(cfg.procedural_detail_enabled),
            "procedural_octaves_executed": int(executed_octaves),
            "procedural_max_absolute_detail_m": float(np.max(np.abs(procedural))),
            "procedural_area_weighted_mean_m": (
                float(np.sum(procedural * area_m2) / np.sum(area_m2))
                if float(np.sum(area_m2)) > 0.0
                else 0.0
            ),
            "procedural_semantics": (
                "zero-area-mean unresolved morphology applied once after "
                "finite-domain anchoring; excluded from the physical sediment mass ledger"
            ),
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
