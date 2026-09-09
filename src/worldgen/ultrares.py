from __future__ import annotations

"""Scale-aware ultra-resolution terrain refinement and bottom-up reconstruction.

The globally coupled world remains the low-frequency physical authority.  This module
refines terrain in a sparse cube-sphere hierarchy without pretending interpolation is
new information:

* the base tile sampler is run with spectral detail disabled;
* local open-boundary hydrology resolves tributaries at the finer sampling scale;
* the legacy stream-power/sediment operator and phase-cell procedural erosion are
  evaluated in the same local geomorphology solve;
* the procedural wavelength band is derived from the source and target metres/sample
  so it occupies only newly resolvable spatial frequencies;
* the deepest tiles are the terrain authority and coarser tiles/full-view products are
  reconstructed from them, never independently regenerated.

This is intentionally a terrain/near-surface refinement layer.  Atmosphere, ocean and
tectonic state are inherited from the globally coupled source rather than falsely
claiming that an arbitrary local tile has independent global boundary conditions.
"""

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from threading import local
from typing import Any, Iterable, Mapping

import numpy as np
from scipy import ndimage

from .heightmap import write_heightmap_png16
from .local_geomorphology import LocalGeomorphologySolver, LocalGeomorphologySpec
from .procedural_erosion import phase_cell_octave_xyz
from .planet_tiles import (
    CUBE_FACES,
    PlanetTilePyramid,
    TileGeometry,
    TileKey,
    TilePyramidSpec,
    approximate_meters_per_sample,
)


class UltraResolutionTilePyramid(PlanetTilePyramid):
    """PlanetTilePyramid with a bounded in-process cache for compact authority arrays.

    Ultra-resolution terrain repeatedly samples a deliberately compact source NPZ.
    Keeping only those selected arrays resident avoids decompressing/re-reading the
    same global field for every one of the 96 deepest tiles.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._ultra_source_cache: dict[str, np.ndarray] = {}
        super().__init__(*args, **kwargs)

    def _load_source_array(self, name: str) -> np.ndarray:
        if self.source_kind != "base_npz":
            return super()._load_source_array(name)
        cached = self._ultra_source_cache.get(name)
        if cached is not None:
            return cached
        with np.load(self.base_source_path, allow_pickle=False) as z:
            if name not in z:
                raise KeyError(f"source field {name!r} is not present in world_arrays.npz")
            values = np.asarray(z[name])
        self._ultra_source_cache[name] = values
        return values


    def _xyz_geometry(self, xyz: np.ndarray) -> TileGeometry:
        unit = np.asarray(xyz, dtype=np.float64)
        lat = np.rad2deg(np.arcsin(np.clip(unit[..., 2], -1.0, 1.0)))
        lon = np.rad2deg(np.arctan2(unit[..., 1], unit[..., 0]))
        return TileGeometry(xyz=unit, latitude_deg=lat, longitude_deg=lon)

    def _context_field(
        self,
        name: str,
        geom: TileGeometry,
        *,
        default: float = 0.0,
    ) -> np.ndarray:
        _shape, fields = self._source_metadata()
        if name not in fields:
            return np.full(geom.latitude_deg.shape, float(default), dtype=np.float64)
        return np.asarray(self._sample_source_field(name, geom), dtype=np.float64)

    def _spectral_detail(self, xyz: np.ndarray, level: int) -> np.ndarray:
        """Globally continuous, tectonically conditioned sub-grid terrain.

        Unlike the old tile-local sinusoidal filler this field is evaluated from
        absolute spherical coordinates and global tectonic authority. Shared tile
        vertices therefore evaluate identically, so no edge taper/grid imprint is
        needed. Wavelengths fill the band below the global source resolution down
        to four deepest-tile samples at the requested LOD.  Because the native
        16384-wide height product spans about two deepest samples per output pixel,
        that makes the smallest half-wave approximately one output pixel without
        aliasing below the raster Nyquist limit.
        """
        if int(level) <= 0 or float(self.spec.elevation_detail_strength) <= 0.0:
            return np.zeros(np.asarray(xyz).shape[:-1], dtype=np.float64)
        unit = np.asarray(xyz, dtype=np.float64)
        geom = self._xyz_geometry(unit)
        (_source_h, source_w), _fields = self._source_metadata()
        source_m = 2.0 * math.pi * float(self.planet_radius_m) / float(source_w)
        sample_m = approximate_meters_per_sample(
            self.planet_radius_m, int(level), int(self.spec.tile_size)
        )
        # Four source/tile samples per wavelength is a conservative resolved
        # bandwidth. With z3 tiles sampled twice as finely as the 16384-wide
        # deliverable, the shortest ridge/valley half-wave is about one output
        # pixel: the map is saturated without violating Nyquist.
        coarsest_m = max(3.95 * source_m, 4.0 * sample_m)
        finest_m = max(4.0 * sample_m, 800.0)
        if coarsest_m <= finest_m:
            wavelengths = [finest_m]
        else:
            wavelengths = []
            w = coarsest_m
            while w >= finest_m * (1.0 - 1.0e-12) and len(wavelengths) < 10:
                wavelengths.append(float(w))
                w /= 1.82
            if wavelengths[-1] > finest_m * 1.20 and len(wavelengths) < 10:
                wavelengths.append(float(finest_m))

        elevation = self._context_field("elevation_m", geom)
        mountain = np.clip(self._context_field("mountain_strength", geom), 0.0, 1.0)
        convergence = np.clip(
            self._context_field("convergence_strength", geom), 0.0, 1.0
        )
        strain = np.clip(self._context_field("strain_field", geom), 0.0, 1.0)
        paleo = np.clip(self._context_field("paleo_convergence", geom), 0.0, 1.0)
        age = np.maximum(self._context_field("orogen_age_myr", geom, default=900.0), 0.0)
        inherited_orogen = paleo * np.exp(-age / 480.0)
        orogen = np.clip(
            np.maximum(mountain, 0.90 * convergence + 0.55 * strain + 0.55 * inherited_orogen),
            0.0,
            1.0,
        )
        land = elevation >= 0.0
        relief_context = 0.32 + 0.68 * np.tanh(np.abs(elevation) / 1700.0)
        coast_guard = 0.30 + 0.70 * np.tanh(np.abs(elevation) / 260.0)

        seed = int(self._read_seed()) ^ 0x5445525241494E32
        result = np.zeros(unit.shape[:-1], dtype=np.float64)
        reference_m = max(float(wavelengths[0]), 1.0)
        ridge_mean = 1.0 - 2.0 / math.pi
        for octave, wavelength_m in enumerate(wavelengths):
            rng = np.random.default_rng(seed + 0x9E3779B1 * (octave + 1))
            octave_field = np.zeros(result.shape, dtype=np.float64)
            octave_ridge = np.zeros(result.shape, dtype=np.float64)
            for orientation in range(2):
                axis = rng.normal(size=3)
                axis /= max(float(np.linalg.norm(axis)), 1.0e-15)
                tangent = axis - np.sum(unit * axis, axis=-1, keepdims=True) * unit
                norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
                fallback_axis = np.array([0.0, 0.0, 1.0])
                fallback = fallback_axis - (
                    np.sum(unit * fallback_axis, axis=-1, keepdims=True) * unit
                )
                tangent = np.where(norm > 1.0e-10, tangent, fallback)
                tangent /= np.maximum(
                    np.linalg.norm(tangent, axis=-1, keepdims=True), 1.0e-12
                )
                cosine, _sine, coherence = phase_cell_octave_xyz(
                    unit,
                    self.planet_radius_m / 1000.0,
                    np.full(result.shape, wavelength_m / 1000.0, dtype=np.float64),
                    tangent,
                    cell_scale=0.68,
                    seed=seed ^ (0xA511E9B3 * (orientation + 1)),
                    octave=octave + 17,
                )
                octave_field += coherence * cosine
                octave_ridge += coherence * (
                    (1.0 - np.abs(cosine)) - ridge_mean
                )
            octave_field *= 0.5
            octave_ridge *= 0.5
            scale = (wavelength_m / reference_m) ** float(
                self.spec.detail_hurst_exponent
            )
            background_amp_m = 48.0 * scale
            orogenic_amp_m = 390.0 * scale
            ocean_amp_m = 22.0 * scale
            amplitude = np.where(
                land,
                background_amp_m * relief_context + orogenic_amp_m * orogen,
                ocean_amp_m * (0.55 + 0.45 * relief_context),
            )
            shape_field = (
                octave_field * (0.72 - 0.30 * orogen)
                + octave_ridge * (0.55 + 1.10 * orogen)
            )
            result += amplitude * shape_field * coast_guard

        return (
            float(self.spec.elevation_detail_strength)
            * np.asarray(result, dtype=np.float64)
        )


@dataclass(slots=True, frozen=True)
class UltraResolutionSpec:
    """Numerical contract for one ultra-resolution terrain build."""

    base_linear_multiplier: float = 4.0
    subsection_linear_multiplier: float = 3.0
    tile_size: int = 1024
    min_samples_per_wavelength: float = 4.0
    procedural_lacunarity: float = 2.0
    procedural_hurst_exponent: float = 0.65
    procedural_amplitude_fraction: float = 0.65
    terrain_detail_strength: float = 1.15
    workers: int = 2
    reconstruction_chunk_rows: int = 64

    def validate(self) -> "UltraResolutionSpec":
        for name in ("base_linear_multiplier", "subsection_linear_multiplier"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 1.0:
                raise ValueError(f"{name} must be finite and >= 1")
        if not 64 <= int(self.tile_size) <= 2048:
            raise ValueError("tile_size must be in [64, 2048]")
        if not math.isfinite(float(self.min_samples_per_wavelength)) or float(
            self.min_samples_per_wavelength
        ) < 2.0:
            raise ValueError("min_samples_per_wavelength must be finite and >= 2")
        if not math.isfinite(float(self.procedural_lacunarity)) or float(
            self.procedural_lacunarity
        ) <= 1.0:
            raise ValueError("procedural_lacunarity must be > 1")
        if not 0.1 <= float(self.procedural_hurst_exponent) <= 1.5:
            raise ValueError("procedural_hurst_exponent must be in [0.1, 1.5]")
        if not 0.0 < float(self.procedural_amplitude_fraction) <= 1.0:
            raise ValueError("procedural_amplitude_fraction must be in (0, 1]")
        if not math.isfinite(float(self.terrain_detail_strength)) or not (
            0.0 < float(self.terrain_detail_strength) <= 2.5
        ):
            raise ValueError("terrain_detail_strength must be in (0, 2.5]")
        if not 1 <= int(self.workers) <= 32:
            raise ValueError("workers must be in [1, 32]")
        if int(self.reconstruction_chunk_rows) < 1:
            raise ValueError("reconstruction_chunk_rows must be positive")
        return self


@dataclass(slots=True, frozen=True)
class UltraResolutionPlan:
    source_width: int
    source_height: int
    source_equatorial_m_per_sample: float
    fullview_width: int
    fullview_height: int
    base_level: int
    finest_level: int
    base_m_per_sample: float
    finest_m_per_sample: float
    actual_base_multiplier: float
    actual_subsection_multiplier: float
    finest_tile_count: int
    tile_size: int


@dataclass(slots=True, frozen=True)
class UltraResolutionReport:
    plan: Mapping[str, Any]
    geomorphology_spec: Mapping[str, Any]
    finest_tiles_completed: int
    audit_path: str
    fullview_heightmap: str
    reconstruction_root: str


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, np.asarray(values), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


ULTRARES_AUTHORITY_FIELDS = (
    "lat",
    "lon",
    "elevation_km",
    "runoff_mm_year",
    "annual_precipitation_mm",
    "annual_temperature_c",
    "rivers",
    "stream_order",
    "discharge_index",
    "river_width_proxy",
)

ULTRARES_OPTIONAL_AUTHORITY_FIELDS = (
    "temperature_c_monthly",
    "precipitation_mm_monthly",
    "wind_u_monthly",
    "wind_v_monthly",
    "humidity_proxy_monthly",
    "koppen",
    "continentality_index_c",
    "soil_moisture_index",
    "snow_persistence",
    "vegetation_fraction",
    "surface_albedo",
    "cloud_fraction_annual",
    "true_color_rgb",
    "fog",
    "thunderstorm_level",
    "lightning_flashes_km2_year",
    "tornado_potential",
    "blizzard",
    "sandstorm",
    "duststorm",
    "hurricane_genesis",
    "aurora",
    "sea_ice_max",
    "sea_ice_min",
    "coral_reef",
    "rock_code",
    "bedrock_code",
    "mountain_strength",
    "ruggedness",
    "convergence_strength",
    "strain_field",
    "paleo_convergence",
    "orogen_age_myr",
    "stress_field",
)


def compact_world_authority(
    world_root: str | Path,
    *,
    keep_fields: tuple[str, ...] = ULTRARES_AUTHORITY_FIELDS,
    prune_rendered_maps: bool = True,
) -> dict[str, Any]:
    """Keep only exact global fields required by terrain LOD refinement."""
    root = Path(world_root).expanduser().resolve()
    source = root / "world_arrays.npz"
    if not source.exists():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as z:
        missing = [name for name in keep_fields if name not in z.files]
        if missing:
            raise KeyError(
                "world_arrays.npz is missing ultra-resolution authority fields: "
                + ", ".join(missing)
            )
        selected = list(keep_fields)
        selected.extend(
            name for name in ULTRARES_OPTIONAL_AUTHORITY_FIELDS
            if name in z.files and name not in selected
        )
        selected.extend(
            name for name in z.files
            if name.startswith("resource_") and name not in selected
        )
        arrays = {name: np.asarray(z[name]) for name in selected}

    tmp = root / ".world_arrays.ultrares.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, source)

    source_maps = root / "ultrares" / "source_maps"
    source_maps.mkdir(parents=True, exist_ok=True)
    maps_root = root / "maps"
    for name in (
        "02_elevation.png",
        "02b_height_grayscale_16bit.png",
        "02b_height_grayscale_16bit.json",
        "05_precipitation_annual.png",
        "08c_erosion.png",
        "15_true_color.png",
    ):
        path = maps_root / name
        if path.exists():
            shutil.copy2(path, source_maps / name)

    shutil.rmtree(root / "checkpoints", ignore_errors=True)
    if prune_rendered_maps and maps_root.exists():
        shutil.rmtree(maps_root, ignore_errors=True)

    report = {
        "fields": list(arrays),
        "source_resolution": [int(len(arrays["lon"])), int(len(arrays["lat"]))],
        "npz_bytes": int(source.stat().st_size),
        "rendered_source_maps": sorted(
            path.name for path in source_maps.iterdir() if path.is_file()
        ),
        "semantics": (
            "values copied exactly from the completed full-detail global solve; "
            "only redundant arrays/checkpoints/renders were pruned"
        ),
    }
    _atomic_json(root / "ultrares" / "authority_compaction.json", report)
    return report


def _world_payload(world_root: Path) -> dict[str, Any]:
    path = world_root / "world.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _nested_number(
    payload: Mapping[str, Any], path: tuple[str, ...], default: float
) -> float:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return float(default)
        value = value[key]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _source_resolution(pyramid: PlanetTilePyramid) -> tuple[int, int]:
    (height, width), _fields = pyramid._source_metadata()
    return int(width), int(height)


def source_equatorial_meters_per_sample(pyramid: PlanetTilePyramid) -> float:
    width, _height = _source_resolution(pyramid)
    return 2.0 * math.pi * float(pyramid.planet_radius_m) / float(width)


def _level_for_target(
    pyramid: PlanetTilePyramid, target_m_per_sample: float
) -> int:
    target = float(target_m_per_sample)
    for level in range(int(pyramid.spec.maximum_level) + 1):
        if approximate_meters_per_sample(
            pyramid.planet_radius_m, level, pyramid.spec.tile_size
        ) <= target:
            return level
    return int(pyramid.spec.maximum_level)


def make_ultra_resolution_plan(
    pyramid: PlanetTilePyramid,
    spec: UltraResolutionSpec | None = None,
) -> UltraResolutionPlan:
    cfg = (spec or UltraResolutionSpec()).validate()
    source_width, source_height = _source_resolution(pyramid)
    source_mps = source_equatorial_meters_per_sample(pyramid)
    desired_base = source_mps / float(cfg.base_linear_multiplier)
    base_level = _level_for_target(pyramid, desired_base)
    base_mps = approximate_meters_per_sample(
        pyramid.planet_radius_m, base_level, pyramid.spec.tile_size
    )
    desired_fine = base_mps / float(cfg.subsection_linear_multiplier)
    finest_level = max(base_level, _level_for_target(pyramid, desired_fine))
    fine_mps = approximate_meters_per_sample(
        pyramid.planet_radius_m, finest_level, pyramid.spec.tile_size
    )
    fullview_width = int(round(source_width * float(cfg.base_linear_multiplier)))
    fullview_height = int(round(source_height * float(cfg.base_linear_multiplier)))
    if fullview_width != 2 * fullview_height:
        raise ValueError("ultra-resolution fullview must remain canonical 2:1")
    return UltraResolutionPlan(
        source_width=source_width,
        source_height=source_height,
        source_equatorial_m_per_sample=float(source_mps),
        fullview_width=fullview_width,
        fullview_height=fullview_height,
        base_level=int(base_level),
        finest_level=int(finest_level),
        base_m_per_sample=float(base_mps),
        finest_m_per_sample=float(fine_mps),
        actual_base_multiplier=float(source_mps / base_mps),
        actual_subsection_multiplier=float(base_mps / fine_mps),
        finest_tile_count=int(6 * (4 ** finest_level)),
        tile_size=int(pyramid.spec.tile_size),
    )


def derive_scale_aware_geomorphology_spec(
    pyramid: PlanetTilePyramid,
    plan: UltraResolutionPlan,
    spec: UltraResolutionSpec | None = None,
) -> LocalGeomorphologySpec:
    """Derive local erosion magnitudes and wavelengths from physical scale.

    The coarsest procedural wavelength is the source raster's conservative
    four-sample resolution limit.  The finest allowed wavelength is the target
    tile's four-sample limit.  Thus the local shader cannot double-count spatial
    frequencies that the global source already resolves and cannot emit aliased
    detail below the target resolution.
    """
    cfg = (spec or UltraResolutionSpec()).validate()
    payload = _world_payload(pyramid.world_root)
    source_mps = float(plan.source_equatorial_m_per_sample)
    fine_mps = float(plan.finest_m_per_sample)
    ratio = source_mps / fine_mps

    global_proc_wave_km = _nested_number(
        payload, ("config", "procedural_erosion", "base_wavelength_km"), 420.0
    )
    global_proc_amp_m = _nested_number(
        payload, ("config", "procedural_erosion", "base_amplitude_m"), 24.0
    )
    global_proc_gain = _nested_number(
        payload, ("config", "procedural_erosion", "gain"), 0.52
    )
    global_fluvial_cap = _nested_number(
        payload, ("config", "hydrology", "max_fluvial_erosion_m_per_iteration"), 15.0
    )
    deposition_strength = _nested_number(
        payload, ("config", "hydrology", "deposition_strength"), 0.54
    )
    hillslope_strength = _nested_number(
        payload, ("config", "hydrology", "hillslope_diffusion_strength"), 0.028
    )
    stream_m = _nested_number(payload, ("config", "hydrology", "stream_power_m"), 0.50)
    stream_n = _nested_number(payload, ("config", "hydrology", "stream_power_n"), 1.00)

    min_samples = float(cfg.min_samples_per_wavelength)
    # Keep the upper endpoint infinitesimally below the source-resolvable band.
    coarsest_wavelength_m = min_samples * source_mps * (1.0 - 1.0e-9)
    base_wavelength_samples = coarsest_wavelength_m / fine_mps
    finest_wavelength_m = min_samples * fine_mps

    requested_lac = float(cfg.procedural_lacunarity)
    span = max(coarsest_wavelength_m / finest_wavelength_m, 1.0)
    if span <= 1.0 + 1.0e-12:
        octaves = 1
        lac = requested_lac
    else:
        # Use enough octaves to reach the finest safe wavelength, then make the
        # lacunarity infinitesimally smaller when necessary so the last octave
        # lands exactly on that four-sample boundary instead of stopping one
        # octave early because the coarsest endpoint is kept below the source
        # authority's own four-sample resolution limit.
        octaves = 1 + int(
            math.ceil(
                math.log(span) / math.log(requested_lac) - 1.0e-12
            )
        )
        octaves = max(2, min(octaves, 8))
        lac = span ** (1.0 / float(octaves - 1))
        if lac <= 1.0:
            lac = requested_lac

    scale_ratio = max(coarsest_wavelength_m / 1000.0 / max(global_proc_wave_km, 1.0e-9), 1.0e-9)
    procedural_amplitude = (
        global_proc_amp_m
        * min(scale_ratio, 1.0) ** float(cfg.procedural_hurst_exponent)
        * float(cfg.procedural_amplitude_fraction)
    )

    # The local stream-power pass is a sub-grid correction, not another complete
    # geologic epoch.  Scale its displacement cap sublinearly with cell size.
    physical_cap = global_fluvial_cap * (fine_mps / source_mps) ** 0.35
    physical_cap = float(np.clip(physical_cap, 1.0, global_fluvial_cap))
    deposition_cap = max(0.5, physical_cap * float(np.clip(deposition_strength, 0.0, 1.0)))
    hillslope_fraction = float(
        np.clip(hillslope_strength * math.sqrt(max(ratio, 1.0)), 0.02, 0.15)
    )
    fine_km = fine_mps / 1000.0
    drainage_scale = max(8.0, 12.0 * fine_km * fine_km)

    return LocalGeomorphologySpec(
        stream_power_area_exponent=float(stream_m),
        stream_power_slope_exponent=float(stream_n),
        drainage_area_scale_km2=float(drainage_scale),
        max_fluvial_erosion_m=float(physical_cap),
        sediment_retention_fraction=float(np.clip(deposition_strength, 0.05, 0.85)),
        max_deposition_m=float(deposition_cap),
        hillslope_diffusion_fraction=hillslope_fraction,
        # Absolute-XYZ microrelief itself is seamless; only finite-domain
        # geomorphic displacement needs a narrow edge anchor.
        edge_anchor_cells=3,
        procedural_detail_enabled=True,
        procedural_octaves=int(octaves),
        procedural_base_wavelength_samples=float(base_wavelength_samples),
        procedural_min_samples_per_wavelength=min_samples,
        procedural_amplitude_m=float(max(procedural_amplitude, 0.05)),
        procedural_gain=float(
            np.clip(
                global_proc_gain
                ** (math.log(lac) / math.log(max(requested_lac, 1.000001))),
                0.20,
                0.90,
            )
        ),
        procedural_lacunarity=float(lac),
        procedural_cell_scale=0.72,
        procedural_steering_strength=0.34,
        final_channel_incision_m=float(
            np.clip(6.0 + 0.30 * global_fluvial_cap, 6.0, 14.0)
        ),
        final_channel_discharge_exponent=1.08,
    ).validate()


def _level_keys(level: int) -> Iterable[TileKey]:
    side = 1 << int(level)
    for face in CUBE_FACES:
        for y in range(side):
            for x in range(side):
                yield TileKey(face, int(level), x, y)


def _progress_path(world_root: Path) -> Path:
    return world_root / "ultrares" / "progress.json"


def generate_finest_geomorphology(
    pyramid: PlanetTilePyramid,
    plan: UltraResolutionPlan,
    geomorphology_spec: LocalGeomorphologySpec,
    *,
    workers: int = 2,
) -> int:
    """Generate every deepest terrain subsection with bounded parallelism."""
    keys = tuple(_level_keys(plan.finest_level))
    worker_count = max(1, int(workers))
    thread_state = local()
    completed = 0
    progress_path = _progress_path(pyramid.world_root)
    started = time.monotonic()

    def solver() -> LocalGeomorphologySolver:
        value = getattr(thread_state, "solver", None)
        if value is None:
            value = LocalGeomorphologySolver(pyramid, spec=geomorphology_spec)
            thread_state.solver = value
        return value

    def run_one(key: TileKey) -> TileKey:
        # The base tile already contains seamless absolute-XYZ tectonic microrelief.
        # Hydrology/geomorphology then consume that field at native LOD.
        pyramid.generate_tile(key, ("elevation_m",))
        local_solver = solver()
        local_solver.solve(key)

        # z3 contains 384 x 1025² tiles. Keep only fields required by the final
        # scientific products/audit and delete reproducible scratch state per tile,
        # otherwise temporary D16 hydrology would exceed hosted-runner disk.
        for field in (
            "deposition_m",
            "hillslope_adjustment_m",
            "procedural_coherence",
            "tectonic_microdetail_m",
            "channel_incision_m",
            "final_flow_direction_d16",
            "final_meander_potential",
            "major_river_constraint",
        ):
            local_solver._path(key, field).unlink(missing_ok=True)

        for field in (
            "filled_elevation_m",
            "flow_direction_d8",
            "flow_direction_d16",
            "flow_angle_rad",
            "meander_potential",
            "runoff_mm_year",
            "drainage_area_km2",
            "discharge_index",
            "streams",
            "inherited_major_river",
        ):
            local_solver.hydrology._path(key, field).unlink(missing_ok=True)
        local_solver.hydrology._metadata_path(key).unlink(missing_ok=True)

        for field in (
            "major_river_mask",
            "parent_stream_order",
            "parent_discharge_index",
            "parent_width_proxy",
            "constraint_strength",
            "channel_floor_m",
        ):
            local_solver.rivers._path(key, field).unlink(missing_ok=True)
        local_solver.rivers._metadata_path(key).unlink(missing_ok=True)

        # Base elevation can always be re-evaluated exactly from global authority +
        # absolute-coordinate microrelief; do not retain a second 384-tile copy.
        pyramid._field_path(key, "elevation_m").unlink(missing_ok=True)
        pyramid._metadata_path(key).unlink(missing_ok=True)
        return key

    def publish(last: TileKey | None, state: str) -> None:
        elapsed = max(time.monotonic() - started, 0.0)
        rate = completed / elapsed if completed > 0 and elapsed > 0.0 else 0.0
        remaining = max(len(keys) - completed, 0)
        eta_seconds = remaining / rate if rate > 0.0 else None
        payload = {
            "state": state,
            "completed": completed,
            "total": len(keys),
            "last_key": asdict(last) if last is not None else None,
            "finest_level": int(plan.finest_level),
            "tile_size": int(plan.tile_size),
            "elapsed_seconds": elapsed,
            "tiles_per_second": rate,
            "eta_seconds": eta_seconds,
        }
        _atomic_json(progress_path, payload)
        if completed > 0 or state != "running":
            eta_text = (
                "unknown"
                if eta_seconds is None
                else f"{eta_seconds / 60.0:.1f} min"
            )
            print(
                "[ultrares] "
                f"{state}: {completed}/{len(keys)} tiles "
                f"({100.0 * completed / max(len(keys), 1):.1f}%), "
                f"{elapsed / 60.0:.1f} min elapsed, ETA {eta_text}",
                flush=True,
            )

    last_key: TileKey | None = None
    publish(None, "running")
    if worker_count == 1:
        for key in keys:
            last_key = run_one(key)
            completed += 1
            if completed % 8 == 0 or completed == len(keys):
                publish(last_key, "running")
    else:
        iterator = iter(keys)
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="worldgen-ultrares") as executor:
            pending: dict[Future[TileKey], TileKey] = {}

            def fill() -> None:
                while len(pending) < 2 * worker_count:
                    try:
                        key = next(iterator)
                    except StopIteration:
                        return
                    pending[executor.submit(run_one, key)] = key

            fill()
            while pending:
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    last_key = future.result()
                    completed += 1
                    if completed % 8 == 0 or completed == len(keys):
                        publish(last_key, "running")
                fill()
    publish(last_key, "complete")
    return completed


def _geomorph_path(
    pyramid: PlanetTilePyramid, key: TileKey, field: str
) -> Path:
    return (
        pyramid.root
        / "derived"
        / "local_geomorphology_v1"
        / field
        / f"z{key.level:02d}"
        / key.face
        / f"x{key.x:08d}"
        / f"y{key.y:08d}.npy"
    )


def _reconstructed_path(
    world_root: Path, key: TileKey, field: str
) -> Path:
    return (
        world_root
        / "ultrares"
        / "reconstructed"
        / field
        / f"z{key.level:02d}"
        / key.face
        / f"x{key.x:08d}"
        / f"y{key.y:08d}.npy"
    )


def _merge_children_downsample(
    children: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    """Compose four equal-LOD children and decimate the nested vertex lattice by 2."""
    a00, a10, a01, a11 = (np.asarray(a) for a in children)
    shape = a00.shape
    if any(np.asarray(a).shape != shape for a in children):
        raise ValueError("all child arrays must have identical shapes")
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] < 3:
        raise ValueError("reconstruction currently requires square 2-D vertex fields")
    n = shape[0] - 1
    fine = np.empty((2 * n + 1, 2 * n + 1), dtype=np.result_type(*children))
    fine[: n + 1, : n + 1] = a00
    fine[: n + 1, n:] = a10
    fine[n:, : n + 1] = a01
    fine[n:, n:] = a11
    return np.asarray(fine[::2, ::2])


def reconstruct_parent_levels(
    pyramid: PlanetTilePyramid,
    plan: UltraResolutionPlan,
    *,
    fields: tuple[str, ...] = (
        "elevation_m",
        "erosion_m",
        "procedural_detail_m",
    ),
) -> Path:
    """Propagate the deepest terrain solution upward through every parent LOD."""
    root = pyramid.world_root / "ultrares" / "reconstructed"
    finest = int(plan.finest_level)
    for level in range(finest - 1, -1, -1):
        side = 1 << level
        for field in fields:
            for face in CUBE_FACES:
                for y in range(side):
                    for x in range(side):
                        parent = TileKey(face, level, x, y)
                        child_keys = (
                            TileKey(face, level + 1, 2 * x, 2 * y),
                            TileKey(face, level + 1, 2 * x + 1, 2 * y),
                            TileKey(face, level + 1, 2 * x, 2 * y + 1),
                            TileKey(face, level + 1, 2 * x + 1, 2 * y + 1),
                        )
                        arrays = []
                        for child in child_keys:
                            if child.level == finest:
                                path = _geomorph_path(pyramid, child, field)
                            else:
                                path = _reconstructed_path(pyramid.world_root, child, field)
                            if not path.exists():
                                raise FileNotFoundError(path)
                            arrays.append(np.load(path, mmap_mode="r", allow_pickle=False))
                        parent_values = _merge_children_downsample(tuple(arrays))
                        _atomic_save_npy(
                            _reconstructed_path(pyramid.world_root, parent, field),
                            parent_values,
                        )
    _atomic_json(
        pyramid.world_root / "ultrares" / "reconstruction.json",
        {
            "source_level": finest,
            "deepest_is_authority": True,
            "fields": list(fields),
            "parent_rule": (
                "four child vertex grids are composed exactly and decimated by two; "
                "coarser terrain is therefore derived from the finest solved terrain"
            ),
        },
    )
    return root


def _inverse_cube_coordinates(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ax, ay, az = np.abs(x), np.abs(y), np.abs(z)
    major_x = (ax >= ay) & (ax >= az)
    major_y = (~major_x) & (ay >= az)
    major_z = ~(major_x | major_y)
    face = np.empty(x.shape, dtype=np.int8)
    s = np.empty(x.shape, dtype=np.float64)
    t = np.empty(x.shape, dtype=np.float64)

    m = major_x & (x >= 0)
    face[m] = 0; s[m] = -z[m] / ax[m]; t[m] = -y[m] / ax[m]
    m = major_x & (x < 0)
    face[m] = 1; s[m] = z[m] / ax[m]; t[m] = -y[m] / ax[m]
    m = major_y & (y >= 0)
    face[m] = 2; s[m] = x[m] / ay[m]; t[m] = z[m] / ay[m]
    m = major_y & (y < 0)
    face[m] = 3; s[m] = x[m] / ay[m]; t[m] = -z[m] / ay[m]
    m = major_z & (z >= 0)
    face[m] = 4; s[m] = x[m] / az[m]; t[m] = -y[m] / az[m]
    m = major_z & (z < 0)
    face[m] = 5; s[m] = -x[m] / az[m]; t[m] = -y[m] / az[m]
    return face, s, t


def _sample_reconstructed_level(
    world_root: Path,
    *,
    field: str,
    level: int,
    tile_size: int,
    width: int,
    height: int,
    chunk_rows: int,
    output_path: Path,
) -> Path:
    """Reproject one reconstructed cube-sphere LOD to an equirectangular fullview."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=(height, width)
    )
    faces = CUBE_FACES
    side = 1 << int(level)
    n = int(tile_size)
    lon = -math.pi + (np.arange(width, dtype=np.float64) + 0.5) * (
        2.0 * math.pi / width
    )
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    cache: dict[tuple[int, int, int], np.ndarray] = {}
    for y0 in range(0, height, int(chunk_rows)):
        y1 = min(height, y0 + int(chunk_rows))
        lat = math.pi / 2.0 - (np.arange(y0, y1, dtype=np.float64) + 0.5) * (
            math.pi / height
        )
        cos_lat = np.cos(lat)[:, None]
        sin_lat = np.sin(lat)[:, None]
        xx = cos_lat * cos_lon[None, :]
        yy = cos_lat * sin_lon[None, :]
        zz = np.broadcast_to(sin_lat, xx.shape)
        face, s, t = _inverse_cube_coordinates(xx, yy, zz)
        qx = np.clip(
            (s + 1.0) * 0.5 * side,
            0.0,
            np.nextafter(float(side), 0.0),
        )
        qy = np.clip(
            (t + 1.0) * 0.5 * side,
            0.0,
            np.nextafter(float(side), 0.0),
        )
        tx = np.floor(qx).astype(np.int16)
        ty = np.floor(qy).astype(np.int16)
        u = (qx - tx) * n
        v = (qy - ty) * n
        tile_code = face.astype(np.int32) * (side * side) + ty.astype(np.int32) * side + tx.astype(np.int32)
        chunk = np.empty(xx.shape, dtype=np.float32)

        for code in np.unique(tile_code):
            mask = tile_code == code
            fi = int(code // (side * side))
            rem = int(code % (side * side))
            cy = rem // side
            cx = rem % side
            cache_key = (fi, cx, cy)
            values = cache.get(cache_key)
            if values is None:
                key = TileKey(faces[fi], int(level), cx, cy)
                path = _reconstructed_path(world_root, key, field)
                values = np.load(path, mmap_mode="r", allow_pickle=False)
                cache[cache_key] = values
            uu = np.clip(u[mask], 0.0, n)
            vv = np.clip(v[mask], 0.0, n)
            x0i = np.minimum(np.floor(uu).astype(np.int64), n - 1)
            y0i = np.minimum(np.floor(vv).astype(np.int64), n - 1)
            fx = uu - x0i
            fy = vv - y0i
            x1i = x0i + 1
            y1i = y0i + 1
            a = np.asarray(values)
            sampled = (
                a[y0i, x0i] * (1.0 - fx) * (1.0 - fy)
                + a[y0i, x1i] * fx * (1.0 - fy)
                + a[y1i, x0i] * (1.0 - fx) * fy
                + a[y1i, x1i] * fx * fy
            )
            chunk[mask] = sampled.astype(np.float32)
        out[y0:y1] = chunk
    out.flush()
    return output_path


def reconstruct_fullview(
    pyramid: PlanetTilePyramid,
    plan: UltraResolutionPlan,
    spec: UltraResolutionSpec | None = None,
) -> Path:
    """Build the 4x-linear full-view map from the finest terrain via its parent chain."""
    cfg = (spec or UltraResolutionSpec()).validate()
    root = pyramid.world_root / "ultrares" / "fullview"
    root.mkdir(parents=True, exist_ok=True)
    elevation_npy = _sample_reconstructed_level(
        pyramid.world_root,
        field="elevation_m",
        level=plan.base_level,
        tile_size=plan.tile_size,
        width=plan.fullview_width,
        height=plan.fullview_height,
        chunk_rows=cfg.reconstruction_chunk_rows,
        output_path=root / "elevation_m.npy",
    )
    elevation_m = np.load(elevation_npy, mmap_mode="r", allow_pickle=False)
    png = root / "height_grayscale_16bit.png"
    write_heightmap_png16(
        png,
        np.asarray(elevation_m, dtype=np.float64) / 1000.0,
        metadata_path=root / "height_grayscale_16bit.json",
    )

    for field in ("erosion_m", "procedural_detail_m"):
        _sample_reconstructed_level(
            pyramid.world_root,
            field=field,
            level=plan.base_level,
            tile_size=plan.tile_size,
            width=plan.fullview_width,
            height=plan.fullview_height,
            chunk_rows=cfg.reconstruction_chunk_rows,
            output_path=root / f"{field}.npy",
        )
    _atomic_json(
        root / "provenance.json",
        {
            "resolution": [plan.fullview_width, plan.fullview_height],
            "reconstructed_from_finest_level": plan.finest_level,
            "sampled_from_reconstructed_parent_level": plan.base_level,
            "terrain_authority": "deepest local geomorphology tiles",
            "not_a_pixel_upscale": True,
        },
    )
    return png


def _same_face_seam_max(
    pyramid: PlanetTilePyramid, level: int
) -> float:
    side = 1 << int(level)
    maximum = 0.0
    for face in CUBE_FACES:
        for y in range(side):
            for x in range(side):
                key = TileKey(face, level, x, y)
                a = np.load(
                    _geomorph_path(pyramid, key, "elevation_m"),
                    mmap_mode="r",
                    allow_pickle=False,
                )
                if x + 1 < side:
                    b = np.load(
                        _geomorph_path(
                            pyramid, TileKey(face, level, x + 1, y), "elevation_m"
                        ),
                        mmap_mode="r",
                        allow_pickle=False,
                    )
                    maximum = max(
                        maximum, float(np.max(np.abs(np.asarray(a[:, -1]) - np.asarray(b[:, 0]))))
                    )
                if y + 1 < side:
                    b = np.load(
                        _geomorph_path(
                            pyramid, TileKey(face, level, x, y + 1), "elevation_m"
                        ),
                        mmap_mode="r",
                        allow_pickle=False,
                    )
                    maximum = max(
                        maximum, float(np.max(np.abs(np.asarray(a[-1, :]) - np.asarray(b[0, :]))))
                    )
    return maximum


def _same_face_seam_gradient_rms(
    pyramid: PlanetTilePyramid, level: int
) -> float:
    """RMS mismatch of one-sided terrain derivatives across same-face tile seams."""
    side = 1 << int(level)
    sumsq = 0.0
    count = 0
    for face in CUBE_FACES:
        for y in range(side):
            for x in range(side):
                key = TileKey(face, level, x, y)
                a = np.asarray(
                    np.load(
                        _geomorph_path(pyramid, key, "elevation_m"),
                        mmap_mode="r",
                        allow_pickle=False,
                    ),
                    dtype=np.float64,
                )
                if x + 1 < side:
                    b = np.asarray(
                        np.load(
                            _geomorph_path(
                                pyramid,
                                TileKey(face, level, x + 1, y),
                                "elevation_m",
                            ),
                            mmap_mode="r",
                            allow_pickle=False,
                        ),
                        dtype=np.float64,
                    )
                    da = a[:, -1] - a[:, -2]
                    db = b[:, 1] - b[:, 0]
                    diff = da - db
                    sumsq += float(np.sum(np.square(diff)))
                    count += int(diff.size)
                if y + 1 < side:
                    b = np.asarray(
                        np.load(
                            _geomorph_path(
                                pyramid,
                                TileKey(face, level, x, y + 1),
                                "elevation_m",
                            ),
                            mmap_mode="r",
                            allow_pickle=False,
                        ),
                        dtype=np.float64,
                    )
                    da = a[-1, :] - a[-2, :]
                    db = b[1, :] - b[0, :]
                    diff = da - db
                    sumsq += float(np.sum(np.square(diff)))
                    count += int(diff.size)
    return math.sqrt(sumsq / max(count, 1))


def audit_ultra_resolution(
    pyramid: PlanetTilePyramid,
    plan: UltraResolutionPlan,
    geomorphology_spec: LocalGeomorphologySpec,
    spec: UltraResolutionSpec | None = None,
) -> dict[str, Any]:
    """Audit spatial bandwidth, grid neutrality, rivers, relief and mass closure."""
    cfg = (spec or UltraResolutionSpec()).validate()
    source_limit_m = (
        cfg.min_samples_per_wavelength * plan.source_equatorial_m_per_sample
    )
    fine_limit_m = cfg.min_samples_per_wavelength * plan.finest_m_per_sample

    # The selected high-resolution deliverable is 2x the ordinary fullview.
    # z3 terrain is another factor of two finer than that output, so a four-native-
    # sample wavelength has a one-output-pixel half-wave (Nyquist saturation).
    native_output_width = int(plan.fullview_width * 2)
    native_output_height = int(plan.fullview_height * 2)
    native_output_m_per_pixel = (
        2.0 * math.pi * float(pyramid.planet_radius_m)
        / float(native_output_width)
    )
    terrain_finest_wavelength_m = max(
        4.0 * float(plan.finest_m_per_sample),
        800.0,
    )
    terrain_min_feature_width_m = 0.5 * terrain_finest_wavelength_m
    feature_pixels = (
        terrain_min_feature_width_m / native_output_m_per_pixel
    )

    rows: list[dict[str, Any]] = []
    closure_max = 0.0
    simultaneous = 0
    active_tiles = 0
    residual_band_energy: list[float] = []
    physical_rms: list[float] = []
    procedural_rms: list[float] = []
    microdetail_rms: list[float] = []
    finest_samples: list[float] = []
    coarsest_wavelengths: list[float] = []

    terrain_grid_real = 0.0
    terrain_grid_imag = 0.0
    terrain_grid_eighth_real = 0.0
    terrain_grid_eighth_imag = 0.0
    terrain_grid_weight = 0.0
    river_grid_real = 0.0
    river_grid_imag = 0.0
    river_grid_eighth_real = 0.0
    river_grid_eighth_imag = 0.0
    river_direction_count = 0
    turn_weighted = 0.0
    turn_transition_count = 0
    max_straight_run = 0
    sinuosity_values: list[tuple[float, int]] = []

    land_area = 0.0
    mountain_1500_area = 0.0
    mountain_2500_area = 0.0
    rugged_area = 0.0

    for key in _level_keys(plan.finest_level):
        meta_path = (
            pyramid.root
            / "derived"
            / "local_geomorphology_v1"
            / "metadata"
            / f"z{key.level:02d}"
            / key.face
            / f"x{key.x:08d}"
            / f"y{key.y:08d}.json"
        )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        evolved = np.asarray(
            np.load(
                _geomorph_path(pyramid, key, "elevation_m"),
                mmap_mode="r",
                allow_pickle=False,
            ),
            dtype=np.float64,
        )
        interior = (
            evolved[8:-8, 8:-8]
            if min(evolved.shape) > 20
            else evolved
        )
        band = interior - ndimage.gaussian_filter(
            interior, sigma=2.0, mode="nearest"
        )
        band_rms = (
            float(np.sqrt(np.mean(np.square(band))))
            if band.size
            else 0.0
        )
        residual_band_energy.append(band_rms)

        p_rms = float(meta.get("physical_erosion_rms_m", 0.0))
        q_rms = float(meta.get("procedural_detail_rms_m", 0.0))
        m_rms = float(meta.get("tectonic_microdetail_rms_m", 0.0))
        physical_rms.append(p_rms)
        procedural_rms.append(q_rms)
        microdetail_rms.append(m_rms)
        closure = abs(float(meta.get("sediment_closure_relative", 0.0)))
        closure_max = max(closure_max, closure)
        if p_rms > 1.0e-8 or q_rms > 1.0e-8 or m_rms > 1.0e-8:
            active_tiles += 1
        if bool(meta.get("legacy_and_procedural_simultaneous", False)):
            simultaneous += 1

        grid_angular = meta.get("terrain_grid_angular_moments")
        if isinstance(grid_angular, dict):
            terrain_grid_real += float(grid_angular.get("fourth_real", 0.0))
            terrain_grid_imag += float(grid_angular.get("fourth_imag", 0.0))
            terrain_grid_eighth_real += float(
                grid_angular.get("eighth_real", 0.0)
            )
            terrain_grid_eighth_imag += float(
                grid_angular.get("eighth_imag", 0.0)
            )
            terrain_grid_weight += float(grid_angular.get("weight", 0.0))
        else:
            grid = meta.get("terrain_grid_fourfold_moment", {})
            terrain_grid_real += float(grid.get("real", 0.0))
            terrain_grid_imag += float(grid.get("imag", 0.0))
            terrain_grid_weight += float(grid.get("weight", 0.0))

        routing = meta.get("final_routing_metrics", {})
        direction_count = int(routing.get("stream_direction_count", 0) or 0)
        river_direction_count += direction_count
        river_grid_real += float(
            routing.get("directional_fourfold_moment_real", 0.0) or 0.0
        )
        river_grid_imag += float(
            routing.get("directional_fourfold_moment_imag", 0.0) or 0.0
        )
        river_grid_eighth_real += float(
            routing.get("directional_eighth_moment_real", 0.0) or 0.0
        )
        river_grid_eighth_imag += float(
            routing.get("directional_eighth_moment_imag", 0.0) or 0.0
        )
        transition_count = int(
            routing.get("stream_transition_count", 0) or 0
        )
        turn_transition_count += transition_count
        turn_weighted += (
            float(routing.get("stream_turn_fraction_gt10deg", 0.0) or 0.0)
            * transition_count
        )
        max_straight_run = max(
            max_straight_run,
            int(routing.get("max_straight_run_cells", 0) or 0),
        )
        sinuosity = routing.get("median_sampled_sinuosity")
        if sinuosity is not None and direction_count > 0:
            sinuosity_values.append((float(sinuosity), direction_count))

        land_area += float(meta.get("land_area_m2", 0.0))
        mountain_1500_area += float(
            meta.get("mountain_area_above_1500_m_m2", 0.0)
        )
        mountain_2500_area += float(
            meta.get("mountain_area_above_2500_m_m2", 0.0)
        )
        rugged_area += float(
            meta.get("rugged_land_area_slope_ge_10deg_m2", 0.0)
        )

        fs = meta.get("procedural_finest_samples_per_wavelength")
        cw = meta.get("procedural_coarsest_wavelength_m")
        if fs is not None:
            finest_samples.append(float(fs))
        if cw is not None:
            coarsest_wavelengths.append(float(cw))

        rows.append(
            {
                "key": asdict(key),
                "elevation_high_frequency_rms_m": band_rms,
                "physical_erosion_rms_m": p_rms,
                "procedural_detail_rms_m": q_rms,
                "tectonic_microdetail_rms_m": m_rms,
                "sediment_closure_relative": closure,
                "procedural_finest_samples_per_wavelength": fs,
                "procedural_coarsest_wavelength_m": cw,
                "procedural_finest_wavelength_m": meta.get(
                    "procedural_finest_wavelength_m"
                ),
                "final_routing_metrics": routing,
                "simultaneous": bool(
                    meta.get("legacy_and_procedural_simultaneous", False)
                ),
            }
        )

    seam_max = _same_face_seam_max(pyramid, plan.finest_level)
    seam_gradient_rms = _same_face_seam_gradient_rms(
        pyramid, plan.finest_level
    )
    aggregate_physical = (
        float(np.sqrt(np.mean(np.square(physical_rms))))
        if physical_rms else 0.0
    )
    aggregate_procedural = (
        float(np.sqrt(np.mean(np.square(procedural_rms))))
        if procedural_rms else 0.0
    )
    aggregate_microdetail = (
        float(np.sqrt(np.mean(np.square(microdetail_rms))))
        if microdetail_rms else 0.0
    )
    aggregate_band = (
        float(np.sqrt(np.mean(np.square(residual_band_energy))))
        if residual_band_energy else 0.0
    )
    min_finest_samples = (
        min(finest_samples) if finest_samples else float("inf")
    )
    max_finest_samples = (
        max(finest_samples) if finest_samples else 0.0
    )
    max_coarsest = (
        max(coarsest_wavelengths) if coarsest_wavelengths else 0.0
    )

    terrain_grid_anisotropy = (
        abs(complex(terrain_grid_real, terrain_grid_imag))
        / max(terrain_grid_weight, 1.0e-30)
    )
    terrain_grid_eighth_anisotropy = (
        abs(complex(terrain_grid_eighth_real, terrain_grid_eighth_imag))
        / max(terrain_grid_weight, 1.0e-30)
    )
    terrain_grid_worst_anisotropy = max(
        terrain_grid_anisotropy,
        terrain_grid_eighth_anisotropy,
    )
    river_grid_anisotropy = (
        abs(complex(river_grid_real, river_grid_imag))
        / max(river_direction_count, 1)
    )
    river_grid_eighth_anisotropy = (
        abs(complex(river_grid_eighth_real, river_grid_eighth_imag))
        / max(river_direction_count, 1)
    )
    river_grid_worst_anisotropy = max(
        river_grid_anisotropy,
        river_grid_eighth_anisotropy,
    )
    turn_fraction = (
        turn_weighted / max(turn_transition_count, 1)
    )
    if sinuosity_values:
        sinuosity_num = sum(value * weight for value, weight in sinuosity_values)
        sinuosity_den = sum(weight for _value, weight in sinuosity_values)
        mean_tile_median_sinuosity = sinuosity_num / max(sinuosity_den, 1)
    else:
        mean_tile_median_sinuosity = None

    mountain_1500_fraction = mountain_1500_area / max(land_area, 1.0)
    mountain_2500_fraction = mountain_2500_area / max(land_area, 1.0)
    rugged_fraction = rugged_area / max(land_area, 1.0)

    checks = {
        "base_linear_resolution_at_least_requested": (
            plan.actual_base_multiplier + 1.0e-9 >= cfg.base_linear_multiplier
        ),
        "subsection_resolution_at_least_requested": (
            plan.actual_subsection_multiplier + 1.0e-9
            >= cfg.subsection_linear_multiplier
        ),
        "native_heightmap_bandwidth_saturated": (
            0.90 <= feature_pixels <= 1.15
        ),
        "new_high_frequency_terrain_exists": aggregate_band > 0.25,
        "tectonic_microdetail_active": aggregate_microdetail > 1.0,
        "terrain_square_grid_imprint_below_limit": (
            terrain_grid_anisotropy < 0.28
            and terrain_grid_eighth_anisotropy < 0.36
        ),
        "legacy_erosion_active": aggregate_physical > 1.0e-5,
        "procedural_erosion_active": aggregate_procedural > 1.0e-5,
        "legacy_and_procedural_overlap_on_active_tiles": (
            simultaneous >= max(1, int(0.20 * max(active_tiles, 1)))
        ),
        "procedural_not_below_sampling_limit": (
            min_finest_samples >= cfg.min_samples_per_wavelength - 1.0e-5
        ),
        "procedural_reaches_finest_safe_band": (
            max_finest_samples
            <= cfg.min_samples_per_wavelength
            * cfg.procedural_lacunarity
            + 1.0e-3
        ),
        "procedural_does_not_overlap_source_resolved_band": (
            max_coarsest <= source_limit_m * (1.0 + 1.0e-6)
        ),
        "procedural_band_reaches_target_resolution": (
            min(
                (
                    float(row["procedural_finest_wavelength_m"])
                    for row in rows
                    if row["procedural_finest_wavelength_m"] is not None
                ),
                default=float("inf"),
            )
            <= fine_limit_m
            * cfg.procedural_lacunarity
            * (1.0 + 1.0e-6)
        ),
        "procedural_magnitude_not_overwhelming_legacy": (
            aggregate_procedural <= max(aggregate_physical * 3.0, 0.25)
        ),
        "final_rivers_exist": river_direction_count > 100,
        "river_square_grid_imprint_below_limit": (
            river_grid_anisotropy < 0.58
            and river_grid_eighth_anisotropy < 0.72
        ),
        "rivers_turn_at_resolved_scale": turn_fraction > 0.025,
        "river_straight_runs_bounded": max_straight_run <= 220,
        "rivers_have_resolved_sinuosity": (
            mean_tile_median_sinuosity is not None
            and mean_tile_median_sinuosity >= 1.015
        ),
        "mountain_area_present": mountain_1500_fraction >= 0.08,
        "high_mountain_area_present": mountain_2500_fraction >= 0.015,
        "rugged_mountain_relief_present": rugged_fraction >= 0.002,
        "sediment_mass_closure": closure_max <= 1.0e-8,
        "same_face_tile_seams_watertight": seam_max <= 1.0e-5,
        "same_face_derivative_seams_bounded": seam_gradient_rms <= 80.0,
    }
    report = {
        "schema_version": 2,
        "plan": asdict(plan),
        "derived_geomorphology_spec": asdict(geomorphology_spec),
        "sampling": {
            "source_safe_min_wavelength_m": float(source_limit_m),
            "finest_safe_min_wavelength_m": float(fine_limit_m),
            "minimum_samples_per_wavelength": float(
                cfg.min_samples_per_wavelength
            ),
            "native_output_resolution": [
                native_output_width, native_output_height
            ],
            "native_output_equatorial_m_per_pixel": float(
                native_output_m_per_pixel
            ),
            "terrain_finest_wavelength_m": float(
                terrain_finest_wavelength_m
            ),
            "terrain_minimum_feature_half_wave_m": float(
                terrain_min_feature_width_m
            ),
            "minimum_feature_width_in_native_output_pixels": float(
                feature_pixels
            ),
        },
        "aggregate": {
            "active_tiles": active_tiles,
            "simultaneous_legacy_and_procedural_tiles": simultaneous,
            "elevation_high_frequency_rms_m": aggregate_band,
            "tectonic_microdetail_rms_m": aggregate_microdetail,
            "terrain_square_grid_fourfold_anisotropy": float(
                terrain_grid_anisotropy
            ),
            "terrain_square_grid_eighth_anisotropy": float(
                terrain_grid_eighth_anisotropy
            ),
            "terrain_square_grid_worst_anisotropy": float(
                terrain_grid_worst_anisotropy
            ),
            "legacy_physical_erosion_rms_m": aggregate_physical,
            "procedural_detail_rms_m": aggregate_procedural,
            "river_square_grid_fourfold_anisotropy": float(
                river_grid_anisotropy
            ),
            "river_square_grid_eighth_anisotropy": float(
                river_grid_eighth_anisotropy
            ),
            "river_square_grid_worst_anisotropy": float(
                river_grid_worst_anisotropy
            ),
            "river_stream_direction_count": int(river_direction_count),
            "river_turn_fraction_gt10deg": float(turn_fraction),
            "river_max_straight_run_cells": int(max_straight_run),
            "river_mean_tile_median_sinuosity": (
                None
                if mean_tile_median_sinuosity is None
                else float(mean_tile_median_sinuosity)
            ),
            "land_fraction_above_1500_m": float(mountain_1500_fraction),
            "land_fraction_above_2500_m": float(mountain_2500_fraction),
            "rugged_land_fraction_slope_ge_10deg": float(rugged_fraction),
            "max_sediment_closure_relative": closure_max,
            "same_face_seam_max_abs_m": seam_max,
            "same_face_seam_gradient_rms_m": seam_gradient_rms,
            "min_finest_samples_per_wavelength": (
                None if not finest_samples else min_finest_samples
            ),
            "max_coarsest_procedural_wavelength_m": max_coarsest,
        },
        "checks": checks,
        "all_checks_passed": bool(all(checks.values())),
        "tiles": rows,
    }
    out = pyramid.world_root / "ultrares" / "terrain_detail_audit.json"
    _atomic_json(out, report)
    if not report["all_checks_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            "ultra-resolution terrain audit failed: " + ", ".join(failed)
        )
    return report


def run_ultra_resolution(
    world_root: str | Path,
    *,
    spec: UltraResolutionSpec | None = None,
) -> UltraResolutionReport:
    """Execute the complete deepest-first terrain refinement/reconstruction chain."""
    cfg = (spec or UltraResolutionSpec()).validate()
    root = Path(world_root).expanduser().resolve()
    pyramid = UltraResolutionTilePyramid(
        root,
        spec=TilePyramidSpec(
            tile_size=int(cfg.tile_size),
            # This is not interpolation/noise-for-pixels: UltraResolutionTilePyramid
            # overrides _spectral_detail with a globally continuous, tectonically
            # conditioned physical microrelief field evaluated in absolute XYZ.
            elevation_detail_strength=float(cfg.terrain_detail_strength),
            detail_hurst_exponent=float(cfg.procedural_hurst_exponent),
            detail_harmonics=1,
            maximum_level=12,
        ),
    )
    plan = make_ultra_resolution_plan(pyramid, cfg)
    geomorph = derive_scale_aware_geomorphology_spec(pyramid, plan, cfg)
    plan_path = root / "ultrares" / "plan.json"
    _atomic_json(
        plan_path,
        {
            "plan": asdict(plan),
            "spec": asdict(cfg),
            "geomorphology_spec": asdict(geomorph),
            "semantics": {
                "base_fullview": (
                    "4x-linear terrain base reconstructed from deeper solved tiles"
                ),
                "subsections": (
                    "requested >=3x subsection refinement selects the next power-of-two LOD; "
                    "for the Earth production profile this is z3, 4x finer than the 8192 base"
                ),
                "detail": (
                    "globally continuous tectonic microrelief fills newly resolvable terrain "
                    "frequencies before hydrology; legacy stream-power and phase-cell erosion "
                    "then operate on that refined terrain"
                ),
            },
        },
    )
    completed = generate_finest_geomorphology(
        pyramid, plan, geomorph, workers=cfg.workers
    )
    audit = audit_ultra_resolution(pyramid, plan, geomorph, cfg)
    reconstruction_root = reconstruct_parent_levels(pyramid, plan)
    fullview = reconstruct_fullview(pyramid, plan, cfg)
    return UltraResolutionReport(
        plan=asdict(plan),
        geomorphology_spec=asdict(geomorph),
        finest_tiles_completed=int(completed),
        audit_path=str(root / "ultrares" / "terrain_detail_audit.json"),
        fullview_heightmap=str(fullview),
        reconstruction_root=str(reconstruction_root),
    )


__all__ = [
    "ULTRARES_AUTHORITY_FIELDS",
    "ULTRARES_OPTIONAL_AUTHORITY_FIELDS",
    "UltraResolutionPlan",
    "UltraResolutionTilePyramid",
    "UltraResolutionReport",
    "UltraResolutionSpec",
    "audit_ultra_resolution",
    "compact_world_authority",
    "derive_scale_aware_geomorphology_spec",
    "generate_finest_geomorphology",
    "make_ultra_resolution_plan",
    "reconstruct_fullview",
    "reconstruct_parent_levels",
    "run_ultra_resolution",
    "source_equatorial_meters_per_sample",
]
