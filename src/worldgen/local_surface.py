from __future__ import annotations

"""High-resolution surface-state products derived from local terrain and climate.

These products refine already-simulated global ecology/appearance fields rather than
replacing them. Continuous fields blend to inherited values at tile boundaries so
independently generated neighbours remain compatible. The categorical biome code is
a documented local ecoregime proxy, not a claim of a newly solved ecological model.
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
from .local_orography import LocalOrographicDownscaler, edge_anchor_taper
from .planet_tiles import PlanetTilePyramid, TileKey, tile_geometry


@dataclass(slots=True, frozen=True)
class LocalSurfaceSpec:
    edge_anchor_cells: int = 3
    parent_blend_fraction: float = 0.45
    annual_aridity_scale_mm: float = 650.0
    vegetation_temperature_optimum_c: float = 18.0
    vegetation_temperature_width_c: float = 25.0

    def validate(self) -> "LocalSurfaceSpec":
        if int(self.edge_anchor_cells) < 1:
            raise ValueError("edge_anchor_cells must be >= 1")
        if not 0.0 <= float(self.parent_blend_fraction) <= 1.0:
            raise ValueError("parent_blend_fraction must be in [0, 1]")
        if not math.isfinite(float(self.annual_aridity_scale_mm)) or self.annual_aridity_scale_mm <= 0:
            raise ValueError("annual_aridity_scale_mm must be finite and positive")
        if not math.isfinite(float(self.vegetation_temperature_width_c)) or self.vegetation_temperature_width_c <= 0:
            raise ValueError("vegetation_temperature_width_c must be finite and positive")
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


def _blend_parent(parent: np.ndarray, local: np.ndarray, taper: np.ndarray, parent_fraction: float) -> np.ndarray:
    parent_a = np.asarray(parent, dtype=np.float64)
    local_a = np.asarray(local, dtype=np.float64)
    if parent_a.shape != local_a.shape or parent_a.shape != taper.shape:
        raise ValueError("parent/local/taper shapes must agree")
    interior = float(parent_fraction) * parent_a + (1.0 - float(parent_fraction)) * local_a
    return parent_a + np.asarray(taper, dtype=np.float64) * (interior - parent_a)


class LocalSurfaceGenerator:
    def __init__(
        self,
        pyramid: PlanetTilePyramid,
        *,
        spec: LocalSurfaceSpec | None = None,
    ) -> None:
        self.pyramid = pyramid
        self.spec = (spec or LocalSurfaceSpec()).validate()
        self.temperature = LocalTileDownscaler(pyramid)
        self.orography = LocalOrographicDownscaler(pyramid)
        self.root = pyramid.root / "derived" / "local_surface_v1"

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

    def _optional_parent(self, field: str, geom, fallback: np.ndarray) -> np.ndarray:
        try:
            value = self.pyramid._sample_source_field(field, geom)
        except KeyError:
            return np.asarray(fallback, dtype=np.float64)
        return np.asarray(value, dtype=np.float64)

    def generate(self, key: TileKey) -> dict[str, Path]:
        key.validate()
        names = (
            "soil_moisture_index",
            "snow_persistence",
            "vegetation_fraction",
            "surface_albedo",
            "biome_code",
        )
        paths = {name: self._path(key, name) for name in names}
        if all(path.exists() for path in paths.values()) and self._metadata_path(key).exists():
            return paths

        geom = tile_geometry(key, self.pyramid.spec.tile_size)
        elevation = np.asarray(self.pyramid.load_field(key, "elevation_m"), dtype=np.float64)
        annual_temp = np.asarray(self.temperature.annual_temperature_c(key), dtype=np.float64)
        monthly_temp = np.asarray(self.temperature.monthly_temperature_c(key), dtype=np.float64)
        oro_paths = self.orography.generate(key)
        slope = np.asarray(np.load(oro_paths["slope_deg"], mmap_mode="r"), dtype=np.float64)
        monthly_precip = np.asarray(
            np.load(oro_paths["precipitation_mm_monthly"], mmap_mode="r"), dtype=np.float64
        )
        annual_precip = np.sum(monthly_precip, axis=0, dtype=np.float64)
        taper = edge_anchor_taper(elevation.shape, self.spec.edge_anchor_cells)
        land = elevation >= 0.0

        # A bounded aridity/PET proxy: hotter terrain needs more annual water to
        # achieve the same moisture state; steep terrain retains less soil water.
        thermal_demand = self.spec.annual_aridity_scale_mm * np.exp(
            0.025 * np.clip(annual_temp - 10.0, -30.0, 45.0)
        )
        climatic_moisture = annual_precip / np.maximum(annual_precip + thermal_demand, 1e-12)
        slope_retention = np.exp(-np.clip(slope, 0.0, 75.0) / 55.0)
        moisture_raw = np.clip(climatic_moisture * slope_retention * land, 0.0, 1.0)
        parent_moisture = self._optional_parent("soil_moisture_index", geom, moisture_raw)
        moisture = np.clip(
            _blend_parent(parent_moisture, moisture_raw, taper, self.spec.parent_blend_fraction),
            0.0,
            1.0,
        )
        moisture[~land] = 1.0

        snowfall = np.sum(monthly_precip * (monthly_temp <= 0.0), axis=0, dtype=np.float64)
        snow_supply = snowfall / np.maximum(annual_precip, 1e-9)
        cold_fraction = np.mean(monthly_temp <= 0.0, axis=0)
        cold_intensity = np.clip((-annual_temp + 3.0) / 25.0, 0.0, 1.0)
        snow_raw = np.clip(snow_supply * (0.35 + 0.65 * cold_fraction) * (0.4 + 0.6 * cold_intensity), 0.0, 1.0)
        parent_snow = self._optional_parent("snow_persistence", geom, snow_raw)
        snow = np.clip(
            _blend_parent(parent_snow, snow_raw, taper, self.spec.parent_blend_fraction),
            0.0,
            1.0,
        )
        snow[~land] = 0.0

        temp_suitability = np.exp(
            -((annual_temp - self.spec.vegetation_temperature_optimum_c) / self.spec.vegetation_temperature_width_c) ** 2
        )
        terrain_penalty = np.exp(-np.clip(slope, 0.0, 80.0) / 65.0)
        vegetation_raw = np.clip(1.35 * moisture * temp_suitability * terrain_penalty * (1.0 - 0.75 * snow), 0.0, 1.0)
        parent_vegetation = self._optional_parent("vegetation_fraction", geom, vegetation_raw)
        vegetation = np.clip(
            _blend_parent(parent_vegetation, vegetation_raw, taper, self.spec.parent_blend_fraction),
            0.0,
            1.0,
        )
        vegetation[~land] = 0.0

        fallback_albedo = np.where(land, 0.22, 0.07)
        parent_albedo = self._optional_parent("surface_albedo", geom, fallback_albedo)
        local_albedo = np.clip(
            parent_albedo
            + snow * (0.78 - parent_albedo)
            - 0.08 * vegetation
            + 0.025 * (1.0 - moisture) * land,
            0.02,
            0.95,
        )
        albedo = np.clip(
            _blend_parent(parent_albedo, local_albedo, taper, self.spec.parent_blend_fraction),
            0.02,
            0.95,
        )

        # Compact local ecoregime proxy. Codes are intentionally simple and stable:
        # 0 ocean, 1 ice/tundra, 2 desert, 3 grass/shrub, 4 temperate/boreal forest,
        # 5 warm wet forest. This is not a replacement for the global Köppen field.
        biome = np.full(elevation.shape, 3, dtype=np.uint8)
        biome[~land] = 0
        biome[land & ((annual_temp < -5.0) | (snow > 0.62))] = 1
        biome[land & (annual_precip < 300.0) & (annual_temp >= -5.0)] = 2
        biome[land & (vegetation >= 0.48) & (annual_temp <= 22.0)] = 4
        biome[land & (vegetation >= 0.58) & (annual_temp > 22.0) & (annual_precip >= 1200.0)] = 5

        values = {
            "soil_moisture_index": moisture.astype(np.float32),
            "snow_persistence": snow.astype(np.float32),
            "vegetation_fraction": vegetation.astype(np.float32),
            "surface_albedo": albedo.astype(np.float32),
            "biome_code": biome,
        }
        for name, value in values.items():
            _atomic_save_npy(paths[name], value)
        metadata = {
            "schema_version": 1,
            "key": asdict(key),
            "source_sha256": self.pyramid._source_hash(),
            "spec": asdict(self.spec),
            "generated_fields": list(names),
            "upstream": {
                "temperature": "local_downscaling_v1",
                "wind_precipitation_slope_normals": "local_orography_v1",
            },
            "biome_codes": {
                "0": "ocean",
                "1": "ice_or_tundra_proxy",
                "2": "desert_proxy",
                "3": "grass_or_shrub_proxy",
                "4": "temperate_or_boreal_forest_proxy",
                "5": "warm_wet_forest_proxy",
            },
            "semantics": "local climate/terrain refinement blended to inherited global surface state; biome_code is an ecoregime proxy, not a new global biome authority",
        }
        _atomic_json(self._metadata_path(key), metadata)
        return paths


__all__ = ["LocalSurfaceGenerator", "LocalSurfaceSpec"]
