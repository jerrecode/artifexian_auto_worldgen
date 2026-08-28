from __future__ import annotations

"""Tile-local physical downscaling derived from the global world state.

The first downscaling operator deliberately uses a pointwise invariant: resolved
sub-grid relief modifies near-surface temperature by an environmental lapse rate
relative to the inherited global topography.  Because every term is evaluated at an
absolute planet position, independently requested tiles agree exactly at same-LOD
shared vertices and across cube-face seams.

Non-local processes such as drainage, orographic rain shadows and sediment routing
are intentionally not approximated here; they require halo/supertile boundary
conditions and are implemented separately.
"""

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np

from .planet_tiles import PlanetTilePyramid, TileKey, tile_geometry


@dataclass(slots=True, frozen=True)
class LocalClimateSpec:
    lapse_rate_k_per_km: float = 6.5
    temperature_floor_c: float = -160.0
    temperature_ceiling_c: float = 80.0

    def validate(self) -> "LocalClimateSpec":
        if not math.isfinite(float(self.lapse_rate_k_per_km)) or not 0.0 <= float(
            self.lapse_rate_k_per_km
        ) <= 20.0:
            raise ValueError("lapse_rate_k_per_km must be finite and in [0, 20]")
        if not math.isfinite(float(self.temperature_floor_c)) or not math.isfinite(
            float(self.temperature_ceiling_c)
        ):
            raise ValueError("temperature bounds must be finite")
        if self.temperature_ceiling_c <= self.temperature_floor_c:
            raise ValueError("temperature ceiling must exceed floor")
        return self


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


class LocalTileDownscaler:
    """Derive zoom-dependent physical fields without materializing a global LOD."""

    def __init__(
        self,
        pyramid: PlanetTilePyramid,
        *,
        climate_spec: LocalClimateSpec | None = None,
    ) -> None:
        self.pyramid = pyramid
        self.climate_spec = (climate_spec or LocalClimateSpec()).validate()
        self.root = pyramid.root / "derived" / "local_downscaling_v1"

    def _path(self, key: TileKey, field: str) -> Path:
        key.validate()
        return (
            self.root
            / field
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

    def _terrain_delta_m(self, key: TileKey) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        geom = tile_geometry(key, self.pyramid.spec.tile_size)
        inherited = np.asarray(
            self.pyramid._sample_source_field("elevation_m", geom), dtype=np.float64
        )
        resolved = np.asarray(
            self.pyramid.load_field(key, "elevation_m"), dtype=np.float64
        )
        if inherited.shape != resolved.shape:
            raise RuntimeError("inherited and resolved tile elevation shapes disagree")
        return geom, inherited, resolved

    def annual_temperature_c(self, key: TileKey) -> np.ndarray:
        """Terrain-resolved annual temperature using parent climate as boundary state."""
        path = self._path(key, "annual_temperature_c")
        if path.exists():
            return np.load(path, mmap_mode="r", allow_pickle=False)
        geom, inherited_elevation_m, resolved_elevation_m = self._terrain_delta_m(key)
        base = np.asarray(
            self.pyramid._sample_source_field("annual_temperature_c", geom),
            dtype=np.float64,
        )
        delta_k = -float(self.climate_spec.lapse_rate_k_per_km) * (
            resolved_elevation_m - inherited_elevation_m
        ) / 1000.0
        result = np.clip(
            base + delta_k,
            self.climate_spec.temperature_floor_c,
            self.climate_spec.temperature_ceiling_c,
        ).astype(np.float32)
        _atomic_save_npy(path, result)
        self._update_metadata(key)
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def monthly_temperature_c(self, key: TileKey) -> np.ndarray:
        """Apply the same terrain correction to each inherited monthly temperature."""
        path = self._path(key, "temperature_c_monthly")
        if path.exists():
            return np.load(path, mmap_mode="r", allow_pickle=False)
        geom, inherited_elevation_m, resolved_elevation_m = self._terrain_delta_m(key)
        base = np.asarray(
            self.pyramid._sample_source_field("temperature_c_monthly", geom),
            dtype=np.float64,
        )
        if base.ndim != 3 or base.shape[-2:] != resolved_elevation_m.shape:
            raise RuntimeError(
                "temperature_c_monthly must sample to (month, y, x) for local downscaling"
            )
        delta_k = -float(self.climate_spec.lapse_rate_k_per_km) * (
            resolved_elevation_m - inherited_elevation_m
        ) / 1000.0
        result = np.clip(
            base + delta_k[None, :, :],
            self.climate_spec.temperature_floor_c,
            self.climate_spec.temperature_ceiling_c,
        ).astype(np.float32)
        _atomic_save_npy(path, result)
        self._update_metadata(key)
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def _update_metadata(self, key: TileKey) -> None:
        path = self._metadata_path(key)
        fields = []
        for field in ("annual_temperature_c", "temperature_c_monthly"):
            if self._path(key, field).exists():
                fields.append(field)
        payload = {
            "schema_version": 1,
            "key": asdict(key),
            "source_tile_schema_version": self.pyramid.schema_version,
            "source_sha256": self.pyramid._source_hash(),
            "climate_spec": asdict(self.climate_spec),
            "generated_fields": fields,
            "semantics": {
                "temperature": (
                    "global inherited temperature plus environmental lapse-rate correction "
                    "for resolved minus inherited elevation"
                ),
                "nonlocal_processes": (
                    "orographic precipitation, circulation feedback and hydrology are not "
                    "re-solved by this pointwise operator"
                ),
            },
        }
        _atomic_json(path, payload)


__all__ = ["LocalClimateSpec", "LocalTileDownscaler"]
