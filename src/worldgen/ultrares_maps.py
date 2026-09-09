from __future__ import annotations

"""Full-view map reconstruction from the deepest ultra-resolution cube-sphere tiles.

No generative-image model is involved.  Every pixel is sampled from scientific tile
arrays produced by the world generator.  Terrain-dependent climate/surface products
are recomputed on the final eroded terrain.  Fields without a defensible local solver
(resource geology and some weather hazards) retain the global simulation as their
physical authority and are sampled through the deepest tile geometry without
inventing sub-grid deposits/events.
"""

from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .heightmap import write_heightmap_png16, write_heightmap_tiff32
from .local_downscaling import LocalClimateSpec
from .local_orography import (
    OrographicDownscalingSpec,
    cube_vertex_area_weights,
    downscale_wind,
    edge_anchor_taper,
    redistribute_precipitation,
    terrain_frame,
)
from .local_surface import LocalSurfaceSpec, _blend_parent
from .planet_tiles import (
    CUBE_FACES,
    TileKey,
    TilePyramidSpec,
    tile_geometry,
)
from .ultrares import (
    UltraResolutionPlan,
    UltraResolutionTilePyramid,
    _inverse_cube_coordinates,
)


KOPPEN_CLASSES = (
    "Af", "Am", "As", "Aw",
    "BWh", "BWk", "BSh", "BSk",
    "Csa", "Csb", "Csc", "Cwa", "Cwb", "Cwc", "Cfa", "Cfb", "Cfc",
    "Dsa", "Dsb", "Dsc", "Dsd", "Dwa", "Dwb", "Dwc", "Dwd",
    "Dfa", "Dfb", "Dfc", "Dfd",
    "ET", "EF", "EXO", "UNK",
)
KOPPEN_TO_CODE = {name: i for i, name in enumerate(KOPPEN_CLASSES)}

BIOME_LABELS = {
    0: "ocean",
    1: "ice / tundra",
    2: "desert",
    3: "grass / shrub",
    4: "temperate / boreal forest",
    5: "warm wet forest",
}

RESOURCE_GROUPS: Mapping[str, tuple[str, ...]] = {
    "fuel": ("resource_peat", "resource_coal"),
    "copper": (
        "resource_copper_rich", "resource_native_copper_mvt",
        "resource_secondary_vms_copper", "resource_arsenical_copper",
        "resource_bronze_vms", "resource_iocg", "resource_tin_copper",
    ),
    "gold": (
        "resource_gold_placer", "resource_gold_laterite", "resource_gold_gossan",
        "resource_porphyry_gold",
    ),
    "silver": (
        "resource_silver_epithermal", "resource_silver_vms",
        "resource_silver_placer", "resource_lead_silver",
    ),
    "iron": (
        "resource_bog_iron", "resource_skarn_iron", "resource_iron_laterite",
        "resource_oolitic_iron", "resource_hydrothermal_iron",
    ),
    "tin": ("resource_tin_belt", "resource_tin_placer", "resource_tin_copper"),
    "lead_zinc": ("resource_lead", "resource_lead_silver", "resource_zinc", "resource_sedex"),
    "gemstones": (
        "resource_kimberlite_diamond", "resource_pegmatite_gems", "resource_jadeite",
        "resource_turquoise", "resource_agate", "resource_malachite_azurite",
        "resource_opal_smithsonite", "resource_gem_placer",
    ),
    "salt": ("resource_salt_flat", "resource_halite"),
}

WEATHER_GROUPS = (
    "fog",
    "severe_convection",
    "blizzard",
    "dust_sandstorm",
    "hurricane_genesis",
    "aurora",
)

# Palettes are explicit renderer metadata, not learned/generated imagery.
PALETTES: Mapping[str, tuple[tuple[float, tuple[int, int, int]], ...]] = {
    "elevation": (
        (0.00, (7, 22, 54)), (0.24, (17, 67, 117)), (0.43, (66, 139, 176)),
        (0.50, (210, 196, 140)), (0.55, (79, 133, 65)), (0.70, (146, 126, 76)),
        (0.86, (128, 102, 85)), (1.00, (245, 245, 245)),
    ),
    "temperature": (
        (0.00, (42, 30, 120)), (0.20, (46, 103, 190)), (0.40, (68, 188, 214)),
        (0.56, (235, 232, 94)), (0.73, (241, 142, 45)), (1.00, (150, 25, 25)),
    ),
    "water": (
        (0.00, (247, 251, 255)), (0.20, (198, 219, 239)), (0.45, (107, 174, 214)),
        (0.70, (33, 113, 181)), (1.00, (8, 48, 107)),
    ),
    "humidity": (
        (0.00, (161, 112, 65)), (0.35, (220, 190, 116)), (0.58, (87, 160, 95)),
        (0.78, (52, 137, 163)), (1.00, (34, 70, 150)),
    ),
    "vegetation": (
        (0.00, (94, 70, 44)), (0.25, (170, 155, 82)), (0.50, (107, 155, 70)),
        (0.75, (45, 119, 52)), (1.00, (15, 70, 30)),
    ),
    "snow": ((0.00, (18, 28, 55)), (0.35, (93, 139, 181)), (0.75, (205, 225, 236)), (1.00, (255, 255, 255))),
    "gray": ((0.00, (0, 0, 0)), (1.00, (255, 255, 255))),
    "wind": (
        (0.00, (30, 40, 90)), (0.30, (45, 150, 180)), (0.55, (105, 190, 105)),
        (0.78, (240, 205, 70)), (1.00, (180, 45, 35)),
    ),
    "erosion": (
        (0.00, (255, 252, 235)), (0.30, (240, 190, 110)), (0.60, (205, 105, 55)),
        (0.82, (135, 55, 40)), (1.00, (60, 20, 25)),
    ),
    "signed": (
        (0.00, (35, 80, 170)), (0.35, (120, 180, 220)), (0.50, (245, 245, 245)),
        (0.65, (235, 155, 120)), (1.00, (165, 40, 40)),
    ),
    "resource": (
        (0.00, (15, 12, 20)), (0.22, (60, 40, 90)), (0.45, (120, 70, 135)),
        (0.70, (205, 125, 70)), (1.00, (255, 230, 105)),
    ),
    "hazard": (
        (0.00, (16, 20, 28)), (0.25, (65, 55, 90)), (0.50, (160, 75, 70)),
        (0.75, (225, 130, 45)), (1.00, (255, 235, 120)),
    ),
}

BIOME_PALETTE = {
    0: (35, 90, 155),
    1: (210, 225, 232),
    2: (215, 185, 100),
    3: (145, 165, 75),
    4: (54, 120, 55),
    5: (25, 85, 42),
}

RESOURCE_PALETTE = {
    0: (90, 65, 40),    # fuel
    1: (196, 103, 54),  # copper
    2: (226, 190, 57),  # gold
    3: (185, 190, 200), # silver
    4: (155, 62, 50),   # iron
    5: (170, 140, 105), # tin
    6: (93, 105, 120),  # lead/zinc
    7: (105, 70, 165),  # gems
    8: (225, 220, 195), # salt
}
WEATHER_PALETTE = {
    0: (165, 178, 185),
    1: (220, 110, 45),
    2: (205, 225, 245),
    3: (175, 125, 70),
    4: (175, 45, 80),
    5: (80, 205, 145),
}


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


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


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, sort_keys=True).encode("utf-8")
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


def _field_path(root: Path, category: str, field: str, key: TileKey) -> Path:
    return (
        root / "ultrares" / "deepest_products" / category / field
        / f"z{key.level:02d}" / key.face
        / f"x{key.x:08d}" / f"y{key.y:08d}.npy"
    )


def _geomorph_path(pyramid: UltraResolutionTilePyramid, field: str, key: TileKey) -> Path:
    return (
        pyramid.root / "derived" / "local_geomorphology_v1" / field
        / f"z{key.level:02d}" / key.face
        / f"x{key.x:08d}" / f"y{key.y:08d}.npy"
    )


def _hydrology_path(pyramid: UltraResolutionTilePyramid, field: str, key: TileKey) -> Path:
    return (
        pyramid.root / "derived" / "local_hydrology_v1" / field
        / f"z{key.level:02d}" / key.face
        / f"x{key.x:08d}" / f"y{key.y:08d}.npy"
    )


def _base_tile_path(pyramid: UltraResolutionTilePyramid, field: str, key: TileKey) -> Path:
    return (
        pyramid.root / "fields" / field
        / f"z{key.level:02d}" / key.face
        / f"x{key.x:08d}" / f"y{key.y:08d}.npy"
    )


def _level_keys(level: int) -> Iterable[TileKey]:
    side = 1 << int(level)
    for face in CUBE_FACES:
        for y in range(side):
            for x in range(side):
                yield TileKey(face, int(level), x, y)


def _sample_optional(
    pyramid: UltraResolutionTilePyramid,
    field: str,
    geom,
    fallback: np.ndarray | float,
) -> np.ndarray:
    try:
        return np.asarray(pyramid._sample_source_field(field, geom))
    except KeyError:
        return np.asarray(fallback)


def _save_product(root: Path, category: str, field: str, key: TileKey, values: np.ndarray) -> None:
    _atomic_save_npy(_field_path(root, category, field, key), values)


def _classify_koppen_geographic(
    temp: np.ndarray,
    precip: np.ndarray,
    latitude_deg: np.ndarray,
) -> np.ndarray:
    """Köppen classifier using true geographic hemisphere on arbitrary tile geometry."""
    t = np.asarray(temp, dtype=np.float64)
    p = np.asarray(precip, dtype=np.float64)
    lat = np.asarray(latitude_deg, dtype=np.float64)
    if t.shape[0] != 12 or p.shape != t.shape or lat.shape != t.shape[1:]:
        raise ValueError("Köppen tile classification requires (12,H,W) temp/precip and (H,W) latitude")
    h, w = lat.shape
    out = np.full((h, w), "", dtype="<U3")
    t_ann = t.mean(0)
    p_ann = p.sum(0)
    t_min = t.min(0)
    t_max = t.max(0)
    months_gt10 = (t > 10.0).sum(0)
    north = lat >= 0.0
    summer_n = np.array([3, 4, 5, 6, 7, 8])
    winter_n = np.array([9, 10, 11, 0, 1, 2])
    p_sum_n = p[summer_n].sum(0)
    p_win_n = p[winter_n].sum(0)
    p_summer = np.where(north, p_sum_n, p_win_n)
    p_winter = np.where(north, p_win_n, p_sum_n)
    frac_s = p_summer / np.maximum(p_ann, 1.0e-6)
    arid_threshold = np.maximum(
        20.0 * t_ann
        + np.where(frac_s >= 0.70, 280.0, np.where(frac_s >= 0.30, 140.0, 0.0)),
        0.0,
    )
    desert = p_ann < 0.5 * arid_threshold
    steppe = (~desert) & (p_ann < arid_threshold)
    hot = t_ann >= 18.0
    out[desert & hot] = "BWh"
    out[desert & ~hot] = "BWk"
    out[steppe & hot] = "BSh"
    out[steppe & ~hot] = "BSk"
    bmask = desert | steppe

    ef = (~bmask) & (t_max < 0.0)
    et = (~bmask) & (t_max >= 0.0) & (t_max < 10.0)
    out[ef] = "EF"
    out[et] = "ET"

    tropical = (~bmask) & (t_min >= 18.0)
    pmin = p.min(0)
    af = tropical & (pmin >= 60.0)
    am = tropical & ~af & (pmin >= 100.0 - p_ann / 25.0)
    out[af] = "Af"
    out[am] = "Am"

    pmin_s_n = p[summer_n].min(0)
    pmin_w_n = p[winter_n].min(0)
    pmin_s = np.where(north, pmin_s_n, pmin_w_n)
    pmin_w = np.where(north, pmin_w_n, pmin_s_n)
    other_tropical = tropical & ~(af | am)
    out[other_tropical & (pmin_s < pmin_w)] = "As"
    out[other_tropical & ~(pmin_s < pmin_w)] = "Aw"

    base = (~bmask) & ~tropical & ~(ef | et)
    c_mask = base & (t_min > 0.0) & (t_min < 18.0) & (t_max > 10.0)
    d_mask = base & (t_min <= 0.0) & (t_max > 10.0)
    pmax_s_n = p[summer_n].max(0)
    pmax_w_n = p[winter_n].max(0)
    pmax_s = np.where(north, pmax_s_n, pmax_w_n)
    pmax_w = np.where(north, pmax_w_n, pmax_s_n)
    dry_s = (pmin_s < 40.0) & (pmin_s < pmax_w / 3.0)
    dry_w = pmin_w < pmax_s / 10.0
    second = np.where(dry_s, "s", np.where(dry_w, "w", "f"))
    third_c = np.where(
        (t_max >= 22.0) & (months_gt10 >= 4),
        "a",
        np.where(months_gt10 >= 4, "b", "c"),
    )
    third_d = np.where(
        (t_max >= 22.0) & (months_gt10 >= 4),
        "a",
        np.where(months_gt10 >= 4, "b", np.where(t_min <= -38.0, "d", "c")),
    )
    for sec in ("s", "w", "f"):
        for third in ("a", "b", "c"):
            out[c_mask & (second == sec) & (third_c == third)] = "C" + sec + third
        for third in ("a", "b", "c", "d"):
            out[d_mask & (second == sec) & (third_d == third)] = "D" + sec + third
    out[out == ""] = "UNK"
    return out


def _koppen_codes(classes: np.ndarray) -> np.ndarray:
    a = np.asarray(classes).astype("<U3", copy=False)
    out = np.full(a.shape, KOPPEN_TO_CODE["UNK"], dtype=np.uint8)
    for name, code in KOPPEN_TO_CODE.items():
        out[a == name] = code
    return out


def _refined_true_color(
    parent_rgb: np.ndarray,
    parent_vegetation: np.ndarray,
    parent_snow: np.ndarray,
    parent_albedo: np.ndarray,
    vegetation: np.ndarray,
    snow: np.ndarray,
    albedo: np.ndarray,
    normal_xyz: np.ndarray,
    land: np.ndarray,
    taper: np.ndarray,
) -> np.ndarray:
    rgb = np.clip(np.asarray(parent_rgb, dtype=np.float64) / 255.0, 0.0, 1.0)
    pveg = np.clip(np.asarray(parent_vegetation, dtype=np.float64), 0.0, 1.0)
    psnow = np.clip(np.asarray(parent_snow, dtype=np.float64), 0.0, 1.0)
    palb = np.clip(np.asarray(parent_albedo, dtype=np.float64), 0.02, 0.95)

    veg_delta = np.asarray(vegetation, dtype=np.float64) - pveg
    snow_delta = np.asarray(snow, dtype=np.float64) - psnow
    bare = np.array([0.40, 0.32, 0.22], dtype=np.float64)
    green = np.array([0.08, 0.30, 0.07], dtype=np.float64)
    white = np.array([0.96, 0.97, 0.98], dtype=np.float64)

    positive_veg = np.clip(veg_delta, 0.0, 1.0)[..., None] * 0.55
    negative_veg = np.clip(-veg_delta, 0.0, 1.0)[..., None] * 0.40
    local = rgb * (1.0 - positive_veg) + green * positive_veg
    local = local * (1.0 - negative_veg) + bare * negative_veg

    positive_snow = np.clip(snow_delta, 0.0, 1.0)[..., None] * 0.85
    local = local * (1.0 - positive_snow) + white * positive_snow

    brightness = np.sqrt(
        np.clip(np.asarray(albedo, dtype=np.float64) / palb, 0.55, 1.65)
    )[..., None]
    local *= brightness

    sun = np.array([0.62, -0.43, 0.65], dtype=np.float64)
    sun /= np.linalg.norm(sun)
    shade = np.clip(np.sum(np.asarray(normal_xyz, dtype=np.float64) * sun, axis=-1), -0.2, 1.0)
    shade = 0.82 + 0.24 * ((shade + 0.2) / 1.2)
    shade = 1.0 + np.asarray(taper) * (shade - 1.0)
    local *= shade[..., None]

    blend = (np.asarray(taper, dtype=np.float64) * np.asarray(land, dtype=np.float64))[..., None]
    rgb = rgb + blend * (local - rgb)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def generate_climate_surface_products(
    pyramid: UltraResolutionTilePyramid,
    plan: UltraResolutionPlan,
) -> dict[str, object]:
    """Recompute terrain-sensitive local climate/surface fields on final deepest terrain."""
    climate_cfg = LocalClimateSpec().validate()
    oro_cfg = OrographicDownscalingSpec().validate()
    surface_cfg = LocalSurfaceSpec().validate()
    root = pyramid.world_root
    fields = (
        "annual_temperature_c",
        "annual_precipitation_mm",
        "humidity_proxy_annual",
        "wind_speed_annual_m_s",
        "storminess_index",
        "cloud_fraction",
        "soil_moisture_index",
        "snow_persistence",
        "vegetation_fraction",
        "surface_albedo",
        "biome_code",
        "koppen_code",
        "slope_deg",
        "true_color_rgb",
    )

    for key in _level_keys(plan.finest_level):
        geom = tile_geometry(key, pyramid.spec.tile_size)
        final_elevation = np.asarray(
            np.load(_geomorph_path(pyramid, "elevation_m", key), mmap_mode="r", allow_pickle=False),
            dtype=np.float64,
        )
        inherited_elevation = np.asarray(
            pyramid._sample_source_field("elevation_m", geom), dtype=np.float64
        )
        relief_delta_km = (final_elevation - inherited_elevation) / 1000.0

        try:
            base_temp_monthly = np.asarray(
                pyramid._sample_source_field("temperature_c_monthly", geom), dtype=np.float64
            )
        except KeyError:
            annual = np.asarray(
                pyramid._sample_source_field("annual_temperature_c", geom), dtype=np.float64
            )
            base_temp_monthly = np.broadcast_to(annual[None, ...], (12, *annual.shape)).copy()
        local_temp_monthly = np.clip(
            base_temp_monthly - climate_cfg.lapse_rate_k_per_km * relief_delta_km[None, ...],
            climate_cfg.temperature_floor_c,
            climate_cfg.temperature_ceiling_c,
        )
        annual_temp = np.mean(local_temp_monthly, axis=0)

        normal, slope, grad_e, grad_s = terrain_frame(
            geom.xyz, final_elevation, pyramid.planet_radius_m
        )
        taper = edge_anchor_taper(final_elevation.shape, surface_cfg.edge_anchor_cells)
        area_weights = cube_vertex_area_weights(geom.xyz)

        base_u = np.asarray(pyramid._sample_source_field("wind_u_monthly", geom), dtype=np.float64)
        base_v = np.asarray(pyramid._sample_source_field("wind_v_monthly", geom), dtype=np.float64)
        base_p = np.asarray(
            pyramid._sample_source_field("precipitation_mm_monthly", geom), dtype=np.float64
        )
        local_u, local_v = downscale_wind(
            base_u, base_v, grad_e, grad_s, taper, spec=oro_cfg
        )
        local_p = redistribute_precipitation(
            base_p, local_u, local_v, grad_e, grad_s, taper, area_weights, spec=oro_cfg
        )
        local_p64 = np.asarray(local_p, dtype=np.float64)
        annual_p = np.sum(local_p64, axis=0)
        wind_speed = np.mean(
            np.hypot(np.asarray(local_u, dtype=np.float64), np.asarray(local_v, dtype=np.float64)),
            axis=0,
        )
        storminess = np.std(local_p64, axis=0) / np.maximum(np.mean(local_p64, axis=0) + 20.0, 20.0)
        storminess = np.clip(storminess / 2.5, 0.0, 1.0)

        base_h = _sample_optional(
            pyramid,
            "humidity_proxy_monthly",
            geom,
            np.zeros_like(local_p64),
        )
        base_h = np.asarray(base_h, dtype=np.float64)
        if base_h.shape == local_p64.shape and np.any(base_h > 0.0):
            precip_ratio = np.clip((local_p64 + 5.0) / (base_p + 5.0), 0.4, 2.5)
            temp_delta = local_temp_monthly - base_temp_monthly
            local_h = np.maximum(base_h, 0.0) * np.power(precip_ratio, 0.28) * np.exp(-0.012 * temp_delta)
            humidity = np.mean(local_h, axis=0)
        else:
            demand = surface_cfg.annual_aridity_scale_mm * np.exp(
                0.025 * np.clip(annual_temp - 10.0, -30.0, 45.0)
            )
            humidity = annual_p / np.maximum(annual_p + demand, 1.0)

        land = final_elevation >= 0.0
        thermal_demand = surface_cfg.annual_aridity_scale_mm * np.exp(
            0.025 * np.clip(annual_temp - 10.0, -30.0, 45.0)
        )
        climatic_moisture = annual_p / np.maximum(annual_p + thermal_demand, 1.0e-12)
        slope_retention = np.exp(-np.clip(slope, 0.0, 75.0) / 55.0)
        moisture_raw = np.clip(climatic_moisture * slope_retention * land, 0.0, 1.0)
        parent_moisture = np.asarray(
            _sample_optional(pyramid, "soil_moisture_index", geom, moisture_raw),
            dtype=np.float64,
        )
        moisture = np.clip(
            _blend_parent(
                parent_moisture, moisture_raw, taper, surface_cfg.parent_blend_fraction
            ),
            0.0,
            1.0,
        )
        moisture[~land] = 1.0

        snowfall = np.sum(local_p64 * (local_temp_monthly <= 0.0), axis=0)
        snow_supply = snowfall / np.maximum(annual_p, 1.0e-9)
        cold_fraction = np.mean(local_temp_monthly <= 0.0, axis=0)
        cold_intensity = np.clip((-annual_temp + 3.0) / 25.0, 0.0, 1.0)
        snow_raw = np.clip(
            snow_supply
            * (0.35 + 0.65 * cold_fraction)
            * (0.4 + 0.6 * cold_intensity),
            0.0,
            1.0,
        )
        parent_snow = np.asarray(
            _sample_optional(pyramid, "snow_persistence", geom, snow_raw), dtype=np.float64
        )
        snow = np.clip(
            _blend_parent(parent_snow, snow_raw, taper, surface_cfg.parent_blend_fraction),
            0.0,
            1.0,
        )
        snow[~land] = 0.0

        temp_suitability = np.exp(
            -(
                (annual_temp - surface_cfg.vegetation_temperature_optimum_c)
                / surface_cfg.vegetation_temperature_width_c
            )
            ** 2
        )
        terrain_penalty = np.exp(-np.clip(slope, 0.0, 80.0) / 65.0)
        vegetation_raw = np.clip(
            1.35
            * moisture
            * temp_suitability
            * terrain_penalty
            * (1.0 - 0.75 * snow),
            0.0,
            1.0,
        )
        parent_vegetation = np.asarray(
            _sample_optional(pyramid, "vegetation_fraction", geom, vegetation_raw),
            dtype=np.float64,
        )
        vegetation = np.clip(
            _blend_parent(
                parent_vegetation,
                vegetation_raw,
                taper,
                surface_cfg.parent_blend_fraction,
            ),
            0.0,
            1.0,
        )
        vegetation[~land] = 0.0

        fallback_albedo = np.where(land, 0.22, 0.07)
        parent_albedo = np.asarray(
            _sample_optional(pyramid, "surface_albedo", geom, fallback_albedo),
            dtype=np.float64,
        )
        local_albedo = np.clip(
            parent_albedo
            + snow * (0.78 - parent_albedo)
            - 0.08 * vegetation
            + 0.025 * (1.0 - moisture) * land,
            0.02,
            0.95,
        )
        albedo = np.clip(
            _blend_parent(
                parent_albedo, local_albedo, taper, surface_cfg.parent_blend_fraction
            ),
            0.02,
            0.95,
        )

        biome = np.full(final_elevation.shape, 3, dtype=np.uint8)
        biome[~land] = 0
        biome[land & ((annual_temp < -5.0) | (snow > 0.62))] = 1
        biome[land & (annual_p < 300.0) & (annual_temp >= -5.0)] = 2
        biome[land & (vegetation >= 0.48) & (annual_temp <= 22.0)] = 4
        biome[
            land
            & (vegetation >= 0.58)
            & (annual_temp > 22.0)
            & (annual_p >= 1200.0)
        ] = 5
        koppen = _koppen_codes(
            _classify_koppen_geographic(
                local_temp_monthly,
                local_p64,
                np.asarray(geom.latitude_deg, dtype=np.float64),
            )
        )

        parent_cloud = np.asarray(
            _sample_optional(
                pyramid,
                "cloud_fraction_annual",
                geom,
                np.zeros(final_elevation.shape, dtype=np.float64),
            ),
            dtype=np.float64,
        )
        wet_month_fraction = np.mean(local_p64 > 10.0, axis=0)
        cloud_raw = np.clip(0.55 * parent_cloud + 0.45 * wet_month_fraction, 0.0, 1.0)
        cloud = parent_cloud + taper * (cloud_raw - parent_cloud)
        cloud = np.clip(cloud, 0.0, 1.0)

        parent_rgb = np.asarray(
            pyramid._sample_source_field("true_color_rgb", geom), dtype=np.uint8
        )
        rgb = _refined_true_color(
            parent_rgb,
            parent_vegetation,
            parent_snow,
            parent_albedo,
            vegetation,
            snow,
            albedo,
            normal,
            land,
            taper,
        )

        values = {
            "annual_temperature_c": annual_temp.astype(np.float32),
            "annual_precipitation_mm": annual_p.astype(np.float32),
            "humidity_proxy_annual": humidity.astype(np.float32),
            "wind_speed_annual_m_s": wind_speed.astype(np.float32),
            "storminess_index": storminess.astype(np.float32),
            "cloud_fraction": cloud.astype(np.float32),
            "soil_moisture_index": moisture.astype(np.float32),
            "snow_persistence": snow.astype(np.float32),
            "vegetation_fraction": vegetation.astype(np.float32),
            "surface_albedo": albedo.astype(np.float32),
            "biome_code": biome,
            "koppen_code": koppen,
            "slope_deg": slope.astype(np.float32),
            "true_color_rgb": rgb,
        }
        for field, values_a in values.items():
            _save_product(root, "climate_surface", field, key, values_a)

    metadata = {
        "category": "climate_surface",
        "fields": list(fields),
        "source": "deepest final geomorphology elevation plus inherited global climate boundary state",
        "terrain_sensitive_recomputed": True,
        "true_color_semantics": (
            "global physical appearance recolored only by locally recomputed vegetation, "
            "snow, albedo and final-terrain normal; perturbations taper to parent at tile edges"
        ),
        "koppen_semantics": "Köppen class recomputed from locally downscaled monthly temperature and precipitation",
    }
    _atomic_json(root / "ultrares" / "deepest_products" / "climate_surface.json", metadata)
    return metadata


def generate_weather_products(
    pyramid: UltraResolutionTilePyramid,
    plan: UltraResolutionPlan,
) -> dict[str, object]:
    root = pyramid.world_root
    source_fields = (
        "fog",
        "thunderstorm_level",
        "lightning_flashes_km2_year",
        "tornado_potential",
        "blizzard",
        "sandstorm",
        "duststorm",
        "hurricane_genesis",
        "aurora",
    )
    available = set(pyramid._source_metadata()[1])
    scales: dict[str, float] = {}
    for field in source_fields:
        if field not in available:
            scales[field] = 1.0
            continue
        a = np.asarray(pyramid._load_source_array(field), dtype=np.float64)
        finite = a[np.isfinite(a) & (a > 0.0)]
        scales[field] = max(
            float(np.percentile(finite, 99.0)) if finite.size else 1.0,
            1.0e-12,
        )

    for key in _level_keys(plan.finest_level):
        geom = tile_geometry(key, pyramid.spec.tile_size)
        surface_root = root / "ultrares" / "deepest_products" / "climate_surface"
        moisture = np.asarray(
            np.load(
                surface_root / "soil_moisture_index" / f"z{key.level:02d}" / key.face
                / f"x{key.x:08d}" / f"y{key.y:08d}.npy",
                mmap_mode="r",
                allow_pickle=False,
            ),
            dtype=np.float64,
        )
        snow = np.asarray(
            np.load(
                surface_root / "snow_persistence" / f"z{key.level:02d}" / key.face
                / f"x{key.x:08d}" / f"y{key.y:08d}.npy",
                mmap_mode="r",
                allow_pickle=False,
            ),
            dtype=np.float64,
        )
        storm = np.asarray(
            np.load(
                surface_root / "storminess_index" / f"z{key.level:02d}" / key.face
                / f"x{key.x:08d}" / f"y{key.y:08d}.npy",
                mmap_mode="r",
                allow_pickle=False,
            ),
            dtype=np.float64,
        )

        norm: dict[str, np.ndarray] = {}
        for field in source_fields:
            if field in available:
                raw = np.asarray(
                    pyramid._sample_source_field(field, geom), dtype=np.float64
                )
                norm[field] = np.clip(np.maximum(raw, 0.0) / scales[field], 0.0, 1.0)
            else:
                norm[field] = np.zeros(moisture.shape, dtype=np.float64)

        fog = np.clip(norm["fog"] * (0.65 + 0.35 * moisture), 0.0, 1.0)
        severe = np.maximum.reduce(
            (
                norm["thunderstorm_level"],
                norm["tornado_potential"],
                norm["lightning_flashes_km2_year"],
            )
        )
        severe = np.clip(severe * (0.60 + 0.40 * storm), 0.0, 1.0)
        blizzard = np.clip(norm["blizzard"] * (0.55 + 0.45 * snow), 0.0, 1.0)
        dust = np.maximum(norm["sandstorm"], norm["duststorm"])
        dust = np.clip(dust * (0.62 + 0.38 * (1.0 - moisture)), 0.0, 1.0)
        hurricane = norm["hurricane_genesis"]
        aurora = norm["aurora"]
        stack = np.stack((fog, severe, blizzard, dust, hurricane, aurora), axis=0)
        code = np.argmax(stack, axis=0).astype(np.uint8)
        strength = np.max(stack, axis=0).astype(np.float32)

        values = {
            "fog": fog.astype(np.float32),
            "severe_convection": severe.astype(np.float32),
            "blizzard": blizzard.astype(np.float32),
            "dust_sandstorm": dust.astype(np.float32),
            "hurricane_genesis": hurricane.astype(np.float32),
            "aurora": aurora.astype(np.float32),
            "dominant_weather_code": code,
            "dominant_weather_strength": strength,
        }
        for field, a in values.items():
            _save_product(root, "weather", field, key, a)

    metadata = {
        "category": "weather",
        "fields": list(WEATHER_GROUPS) + ["dominant_weather_code", "dominant_weather_strength"],
        "source_percentile_scales": scales,
        "semantics": (
            "global simulated weather-hazard authority sampled onto deepest tiles; fog, "
            "convective, blizzard and dust intensities are locally modulated by final-terrain "
            "surface moisture/snow/storminess without inventing new storm tracks"
        ),
    }
    _atomic_json(root / "ultrares" / "deepest_products" / "weather.json", metadata)
    return metadata


def generate_resource_products(
    pyramid: UltraResolutionTilePyramid,
    plan: UltraResolutionPlan,
) -> dict[str, object]:
    root = pyramid.world_root
    available = set(pyramid._source_metadata()[1])
    used: dict[str, list[str]] = {}
    for key in _level_keys(plan.finest_level):
        geom = tile_geometry(key, pyramid.spec.tile_size)
        groups: list[np.ndarray] = []
        for group, candidates in RESOURCE_GROUPS.items():
            present = [name for name in candidates if name in available]
            used[group] = present
            values = [
                np.clip(
                    np.asarray(pyramid._sample_source_field(name, geom), dtype=np.float64),
                    0.0,
                    1.0,
                )
                for name in present
            ]
            if values:
                group_field = np.maximum.reduce(values)
            else:
                group_field = np.zeros(geom.latitude_deg.shape, dtype=np.float64)
            groups.append(group_field)
            _save_product(root, "resources", group, key, group_field.astype(np.float32))
        stack = np.stack(groups, axis=0)
        _save_product(
            root,
            "resources",
            "dominant_resource_code",
            key,
            np.argmax(stack, axis=0).astype(np.uint8),
        )
        _save_product(
            root,
            "resources",
            "dominant_resource_strength",
            key,
            np.max(stack, axis=0).astype(np.float32),
        )

    metadata = {
        "category": "resources",
        "groups": {key: list(value) for key, value in RESOURCE_GROUPS.items()},
        "available_group_inputs": used,
        "semantics": (
            "resource suitability remains the full global geological/hydrological simulation "
            "authority; it is sampled through the deepest tile geometry but no fictitious "
            "sub-grid ore bodies are synthesized"
        ),
    }
    _atomic_json(root / "ultrares" / "deepest_products" / "resources.json", metadata)
    return metadata


def _sample_deepest_to_npy(
    plan: UltraResolutionPlan,
    resolver: Callable[[TileKey], Path],
    output_path: Path,
    *,
    mode: str = "linear",
    output_dtype: np.dtype | str | None = None,
    chunk_rows: int = 64,
) -> Path:
    level = int(plan.finest_level)
    side = 1 << level
    n = int(plan.tile_size)
    first = np.asarray(
        np.load(resolver(TileKey(CUBE_FACES[0], level, 0, 0)), mmap_mode="r", allow_pickle=False)
    )
    if first.shape[0] != n + 1 or first.shape[1] != n + 1:
        raise ValueError(f"deepest tile shape must begin {(n + 1, n + 1)}, got {first.shape}")
    trailing = first.shape[2:]
    dtype = np.dtype(output_dtype or first.dtype)
    shape = (int(plan.fullview_height), int(plan.fullview_width), *trailing)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(output_path, mode="w+", dtype=dtype, shape=shape)

    width = int(plan.fullview_width)
    height = int(plan.fullview_height)
    lon = -math.pi + (np.arange(width, dtype=np.float64) + 0.5) * (2.0 * math.pi / width)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)
    cache: dict[tuple[int, int, int], np.ndarray] = {}

    for y_start in range(0, height, int(chunk_rows)):
        y_stop = min(height, y_start + int(chunk_rows))
        lat = math.pi / 2.0 - (np.arange(y_start, y_stop, dtype=np.float64) + 0.5) * (
            math.pi / height
        )
        cos_lat = np.cos(lat)[:, None]
        sin_lat = np.sin(lat)[:, None]
        xx = cos_lat * cos_lon[None, :]
        yy = cos_lat * sin_lon[None, :]
        zz = np.broadcast_to(sin_lat, xx.shape)
        face, s, t = _inverse_cube_coordinates(xx, yy, zz)
        qx = np.clip((s + 1.0) * 0.5 * side, 0.0, np.nextafter(float(side), 0.0))
        qy = np.clip((t + 1.0) * 0.5 * side, 0.0, np.nextafter(float(side), 0.0))
        tx = np.floor(qx).astype(np.int16)
        ty = np.floor(qy).astype(np.int16)
        u = (qx - tx) * n
        v = (qy - ty) * n
        tile_code = (
            face.astype(np.int32) * (side * side)
            + ty.astype(np.int32) * side
            + tx.astype(np.int32)
        )
        chunk = np.empty((y_stop - y_start, width, *trailing), dtype=dtype)

        for code in np.unique(tile_code):
            mask = tile_code == code
            fi = int(code // (side * side))
            rem = int(code % (side * side))
            cy = rem // side
            cx = rem % side
            cache_key = (fi, cx, cy)
            tile = cache.get(cache_key)
            if tile is None:
                tile = np.load(
                    resolver(TileKey(CUBE_FACES[fi], level, cx, cy)),
                    mmap_mode="r",
                    allow_pickle=False,
                )
                cache[cache_key] = tile
            a = np.asarray(tile)
            uu = np.clip(u[mask], 0.0, n)
            vv = np.clip(v[mask], 0.0, n)
            if mode == "nearest":
                xi = np.clip(np.rint(uu).astype(np.int64), 0, n)
                yi = np.clip(np.rint(vv).astype(np.int64), 0, n)
                sampled = a[yi, xi]
            else:
                x0 = np.minimum(np.floor(uu).astype(np.int64), n - 1)
                y0 = np.minimum(np.floor(vv).astype(np.int64), n - 1)
                fx = uu - x0
                fy = vv - y0
                x1 = x0 + 1
                y1 = y0 + 1
                if trailing:
                    fx = fx.reshape((-1,) + (1,) * len(trailing))
                    fy = fy.reshape((-1,) + (1,) * len(trailing))
                sampled = (
                    a[y0, x0] * (1.0 - fx) * (1.0 - fy)
                    + a[y0, x1] * fx * (1.0 - fy)
                    + a[y1, x0] * (1.0 - fx) * fy
                    + a[y1, x1] * fx * fy
                )
            chunk[mask] = np.asarray(sampled, dtype=dtype)
        out[y_start:y_stop] = chunk
    out.flush()
    return output_path


def _ramp(values: np.ndarray, anchors: Sequence[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    positions = np.array([p for p, _ in anchors], dtype=np.float64)
    colors = np.array([c for _, c in anchors], dtype=np.float64)
    idx = np.searchsorted(positions, x, side="right") - 1
    idx = np.clip(idx, 0, len(positions) - 2)
    p0 = positions[idx]
    p1 = positions[idx + 1]
    frac = np.divide(x - p0, np.maximum(p1 - p0, 1.0e-12))
    rgb = colors[idx] * (1.0 - frac[..., None]) + colors[idx + 1] * frac[..., None]
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _sample_percentiles(a: np.ndarray, low: float, high: float) -> tuple[float, float]:
    sample = np.asarray(a[::8, ::8], dtype=np.float64).reshape(-1)
    sample = sample[np.isfinite(sample)]
    if not sample.size:
        return 0.0, 1.0
    lo = float(np.percentile(sample, low))
    hi = float(np.percentile(sample, high))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _draw_scalar_legend(
    image: Image.Image,
    *,
    title: str,
    units: str,
    lo: float,
    hi: float,
    anchors: Sequence[tuple[float, tuple[int, int, int]]],
) -> None:
    draw = ImageDraw.Draw(image)
    font = _font(30)
    small = _font(24)
    x0, y0 = 70, image.height - 260
    x1, y1 = min(image.width - 70, x0 + 1900), image.height - 55
    draw.rectangle((x0, y0, x1, y1), fill=(12, 14, 18))
    draw.text((x0 + 28, y0 + 20), title, fill=(245, 245, 245), font=font)
    bar_x0, bar_y0 = x0 + 28, y0 + 86
    bar_w, bar_h = 1450, 42
    for i in range(bar_w):
        t = i / max(bar_w - 1, 1)
        color = tuple(int(v) for v in _ramp(np.array([[t]]), anchors)[0, 0])
        draw.line((bar_x0 + i, bar_y0, bar_x0 + i, bar_y0 + bar_h), fill=color)
    draw.rectangle((bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h), outline=(235, 235, 235))
    draw.text((bar_x0, bar_y0 + 52), f"{lo:.4g} {units}".strip(), fill=(235, 235, 235), font=small)
    high_text = f"{hi:.4g} {units}".strip()
    draw.text((bar_x0 + bar_w - 330, bar_y0 + 52), high_text, fill=(235, 235, 235), font=small)


def _render_scalar(
    data_path: Path,
    png_path: Path,
    *,
    title: str,
    units: str = "",
    palette: str = "water",
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    fixed_range: tuple[float, float] | None = None,
    power: float = 1.0,
    signed: bool = False,
    chunk_rows: int = 128,
) -> dict[str, object]:
    a = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if a.ndim != 2:
        raise ValueError(f"scalar renderer requires 2-D data, got {a.shape}")
    if fixed_range is None:
        lo, hi = _sample_percentiles(a, low_percentile, high_percentile)
    else:
        lo, hi = map(float, fixed_range)
    if signed:
        sample = np.asarray(a[::8, ::8], dtype=np.float64)
        finite = sample[np.isfinite(sample)]
        scale = float(np.percentile(np.abs(finite), high_percentile)) if finite.size else 1.0
        scale = max(scale, 1.0e-12)
        lo, hi = -scale, scale
    anchors = PALETTES[palette]
    tmp = png_path.with_suffix(".rgb.tmp")
    rgb = np.memmap(tmp, mode="w+", dtype=np.uint8, shape=(a.shape[0], a.shape[1], 3))
    for start in range(0, a.shape[0], chunk_rows):
        stop = min(a.shape[0], start + chunk_rows)
        part = np.asarray(a[start:stop], dtype=np.float64)
        norm = (part - lo) / max(hi - lo, 1.0e-12)
        norm = np.clip(norm, 0.0, 1.0)
        if power != 1.0:
            norm = np.power(norm, float(power))
        rgb[start:stop] = _ramp(norm, anchors)
    rgb.flush()
    image = Image.fromarray(np.asarray(rgb), mode="RGB")
    _draw_scalar_legend(image, title=title, units=units, lo=lo, hi=hi, anchors=anchors)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path, format="PNG", compress_level=6)
    del image
    del rgb
    tmp.unlink(missing_ok=True)
    return {
        "file": png_path.name,
        "title": title,
        "units": units,
        "range": [lo, hi],
        "palette": palette,
        "resolution": [int(a.shape[1]), int(a.shape[0])],
    }


def _draw_categorical_legend(
    image: Image.Image,
    *,
    title: str,
    palette: Mapping[int, tuple[int, int, int]],
    labels: Mapping[int, str],
    present: Sequence[int],
    columns: int = 3,
) -> None:
    draw = ImageDraw.Draw(image)
    title_font = _font(30)
    font = _font(22)
    items = [(code, labels.get(code, str(code))) for code in present]
    rows = max(1, math.ceil(len(items) / max(columns, 1)))
    box_w = min(image.width - 100, 2500)
    box_h = 85 + rows * 38 + 25
    x0, y0 = 60, image.height - box_h - 55
    draw.rectangle((x0, y0, x0 + box_w, y0 + box_h), fill=(12, 14, 18))
    draw.text((x0 + 24, y0 + 18), title, fill=(245, 245, 245), font=title_font)
    col_w = (box_w - 48) // max(columns, 1)
    for idx, (code, label) in enumerate(items):
        col = idx // rows
        row = idx % rows
        px = x0 + 24 + col * col_w
        py = y0 + 70 + row * 38
        color = palette.get(code, (180, 180, 180))
        draw.rectangle((px, py, px + 26, py + 26), fill=color, outline=(230, 230, 230))
        draw.text((px + 36, py - 2), label, fill=(240, 240, 240), font=font)


def _render_categorical(
    data_path: Path,
    png_path: Path,
    *,
    title: str,
    palette: Mapping[int, tuple[int, int, int]],
    labels: Mapping[int, str],
    columns: int = 3,
) -> dict[str, object]:
    a = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if a.ndim != 2:
        raise ValueError("categorical renderer requires 2-D data")
    out = np.empty((a.shape[0], a.shape[1], 3), dtype=np.uint8)
    present = sorted(int(v) for v in np.unique(np.asarray(a)))
    for code in present:
        out[np.asarray(a) == code] = palette.get(code, (180, 180, 180))
    image = Image.fromarray(out, mode="RGB")
    _draw_categorical_legend(
        image, title=title, palette=palette, labels=labels, present=present, columns=columns
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path, format="PNG", compress_level=6)
    return {
        "file": png_path.name,
        "title": title,
        "categories": {str(code): labels.get(code, str(code)) for code in present},
        "resolution": [int(a.shape[1]), int(a.shape[0])],
    }


def _render_true_color(data_path: Path, png_path: Path, title: str) -> dict[str, object]:
    a = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("true-color renderer requires HxWx3")
    image = Image.fromarray(np.asarray(a, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    font = _font(30)
    draw.rectangle((60, 60, 900, 115), fill=(12, 14, 18))
    draw.text((78, 70), title, fill=(245, 245, 245), font=font)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path, format="PNG", compress_level=6)
    return {
        "file": png_path.name,
        "title": title,
        "resolution": [int(a.shape[1]), int(a.shape[0])],
        "encoding": "8-bit RGB",
    }


def _terrain_relief(
    elevation_path: Path,
    slope_path: Path,
    png_path: Path,
) -> dict[str, object]:
    elev = np.load(elevation_path, mmap_mode="r", allow_pickle=False)
    slope = np.load(slope_path, mmap_mode="r", allow_pickle=False)
    lo, hi = _sample_percentiles(elev, 0.3, 99.7)
    # Put sea level near the centre of the palette when possible.
    ocean_span = max(0.0 - lo, 1.0)
    land_span = max(hi, 1.0)
    tmp = png_path.with_suffix(".rgb.tmp")
    rgb = np.memmap(tmp, mode="w+", dtype=np.uint8, shape=(elev.shape[0], elev.shape[1], 3))
    anchors = PALETTES["elevation"]
    for start in range(0, elev.shape[0], 128):
        stop = min(elev.shape[0], start + 128)
        z = np.asarray(elev[start:stop], dtype=np.float64)
        norm = np.where(
            z < 0.0,
            0.49 * np.clip((z - lo) / ocean_span, 0.0, 1.0),
            0.50 + 0.50 * np.clip(z / land_span, 0.0, 1.0),
        )
        part = _ramp(norm, anchors).astype(np.float64)
        relief = 1.0 - 0.20 * np.clip(np.asarray(slope[start:stop], dtype=np.float64) / 60.0, 0.0, 1.0)
        part *= relief[..., None]
        rgb[start:stop] = np.clip(np.rint(part), 0, 255).astype(np.uint8)
    rgb.flush()
    image = Image.fromarray(np.asarray(rgb), mode="RGB")
    _draw_scalar_legend(
        image,
        title="Final terrain relief — reconstructed from deepest eroded tiles",
        units="m elevation",
        lo=lo,
        hi=hi,
        anchors=anchors,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path, format="PNG", compress_level=6)
    del image
    del rgb
    tmp.unlink(missing_ok=True)
    return {
        "file": png_path.name,
        "title": "Final terrain relief",
        "resolution": [int(elev.shape[1]), int(elev.shape[0])],
        "range_m": [lo, hi],
    }


def _render_river_composite(
    drainage_path: Path,
    discharge_path: Path,
    streams_path: Path,
    png_path: Path,
) -> dict[str, object]:
    drainage = np.load(drainage_path, mmap_mode="r", allow_pickle=False)
    discharge = np.load(discharge_path, mmap_mode="r", allow_pickle=False)
    streams = np.load(streams_path, mmap_mode="r", allow_pickle=False)
    d = np.log1p(np.maximum(np.asarray(drainage, dtype=np.float64), 0.0))
    _, d_hi = _sample_percentiles(d, 1.0, 99.8)
    dn = np.clip(d / max(d_hi, 1.0e-12), 0.0, 1.0)
    qn = np.clip(np.asarray(discharge, dtype=np.float64), 0.0, 1.0)
    intensity = np.clip((0.65 * dn + 0.35 * qn) * (0.15 + 0.85 * (np.asarray(streams) > 0)), 0.0, 1.0)
    rgb = _ramp(intensity, PALETTES["water"])
    image = Image.fromarray(rgb, mode="RGB")
    _draw_scalar_legend(
        image,
        title="Deepest-tile river / drainage hierarchy",
        units="relative",
        lo=0.0,
        hi=1.0,
        anchors=PALETTES["water"],
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path, format="PNG", compress_level=6)
    return {
        "file": png_path.name,
        "title": "Deepest-tile river / drainage hierarchy",
        "resolution": [int(drainage.shape[1]), int(drainage.shape[0])],
    }


def _render_dominant_composite(
    code_path: Path,
    strength_path: Path,
    png_path: Path,
    *,
    title: str,
    palette: Mapping[int, tuple[int, int, int]],
    labels: Mapping[int, str],
) -> dict[str, object]:
    code = np.load(code_path, mmap_mode="r", allow_pickle=False)
    strength = np.clip(
        np.asarray(np.load(strength_path, mmap_mode="r", allow_pickle=False), dtype=np.float64),
        0.0,
        1.0,
    )
    rgb = np.zeros((code.shape[0], code.shape[1], 3), dtype=np.uint8)
    present = sorted(int(v) for v in np.unique(np.asarray(code)))
    for category in present:
        base = np.array(palette.get(category, (180, 180, 180)), dtype=np.float64)
        mask = np.asarray(code) == category
        s = strength[mask][:, None]
        color = 18.0 + s * (base[None, :] - 18.0)
        rgb[mask] = np.clip(np.rint(color), 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    _draw_categorical_legend(
        image,
        title=title,
        palette=palette,
        labels=labels,
        present=present,
        columns=3,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(png_path, format="PNG", compress_level=6)
    return {
        "file": png_path.name,
        "title": title,
        "categories": {str(v): labels.get(v, str(v)) for v in present},
        "resolution": [int(code.shape[1]), int(code.shape[0])],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _remove_fullview_temp(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def reconstruct_fullview_maps(
    world_root: str | Path,
    *,
    cleanup_deepest_products: bool = True,
    cleanup_heavy_solver_caches: bool = True,
) -> Path:
    root = Path(world_root).expanduser().resolve()
    plan_payload = json.loads((root / "ultrares" / "plan.json").read_text(encoding="utf-8"))
    plan = UltraResolutionPlan(**plan_payload["plan"])
    pyramid = UltraResolutionTilePyramid(
        root,
        spec=TilePyramidSpec(
            tile_size=int(plan.tile_size),
            elevation_detail_strength=0.0,
            detail_hurst_exponent=0.65,
            detail_harmonics=1,
            maximum_level=12,
        ),
    )

    output = root / "ultrares" / "fullview_maps"
    temp = root / "ultrares" / "fullview_map_arrays"
    output.mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    def sample(
        name: str,
        resolver: Callable[[TileKey], Path],
        *,
        mode: str = "linear",
        dtype: str | np.dtype | None = None,
    ) -> Path:
        return _sample_deepest_to_npy(
            plan,
            resolver,
            temp / f"{name}.npy",
            mode=mode,
            output_dtype=dtype,
        )

    # Terrain and erosion use the actual final deepest geomorphology tiles.
    elev = sample("elevation_m", lambda key: _geomorph_path(pyramid, "elevation_m", key), dtype="float32")
    erosion = sample("erosion_m", lambda key: _geomorph_path(pyramid, "erosion_m", key), dtype="float32")
    proc = sample("procedural_detail_m", lambda key: _geomorph_path(pyramid, "procedural_detail_m", key), dtype="float32")
    base = sample("base_elevation_m", lambda key: _base_tile_path(pyramid, "elevation_m", key), dtype="float32")
    combined_path = temp / "combined_geomorphic_delta_m.npy"
    combined = np.lib.format.open_memmap(
        combined_path,
        mode="w+",
        dtype=np.float32,
        shape=(plan.fullview_height, plan.fullview_width),
    )
    e_arr = np.load(elev, mmap_mode="r", allow_pickle=False)
    b_arr = np.load(base, mmap_mode="r", allow_pickle=False)
    for start in range(0, plan.fullview_height, 128):
        stop = min(plan.fullview_height, start + 128)
        combined[start:stop] = np.asarray(e_arr[start:stop], dtype=np.float32) - np.asarray(
            b_arr[start:stop], dtype=np.float32
        )
    combined.flush()
    del combined

    # Rivers: materialize fullview while the local hydrology cache still exists.
    drainage = sample(
        "drainage_area_km2",
        lambda key: _hydrology_path(pyramid, "drainage_area_km2", key),
        dtype="float32",
    )
    discharge = sample(
        "discharge_index",
        lambda key: _hydrology_path(pyramid, "discharge_index", key),
        dtype="float32",
    )
    streams = sample(
        "streams",
        lambda key: _hydrology_path(pyramid, "streams", key),
        mode="nearest",
        dtype="uint8",
    )
    river_png = output / "04_rivers_deepest_tiles.png"
    manifest.append(_render_river_composite(drainage, discharge, streams, river_png))
    manifest.append(
        _render_scalar(
            drainage,
            output / "05_drainage_area.png",
            title="Local drainage area",
            units="km²",
            palette="water",
            low_percentile=0.0,
            high_percentile=99.8,
            power=0.42,
        )
    )
    manifest.append(
        _render_scalar(
            discharge,
            output / "06_discharge_index.png",
            title="Local discharge index",
            units="relative",
            palette="water",
            fixed_range=(0.0, 1.0),
        )
    )
    _remove_fullview_temp(drainage, discharge, streams)
    if cleanup_heavy_solver_caches:
        shutil.rmtree(pyramid.root / "derived" / "local_hydrology_v1", ignore_errors=True)
        shutil.rmtree(pyramid.root / "derived" / "river_constraints_v1", ignore_errors=True)

    # Recompute terrain-sensitive climate/surface products using final elevation.
    generate_climate_surface_products(pyramid, plan)
    surface_fields: dict[str, Path] = {}
    surface_modes = {
        "biome_code": ("nearest", "uint8"),
        "koppen_code": ("nearest", "uint8"),
        "true_color_rgb": ("linear", "uint8"),
    }
    for field in (
        "annual_temperature_c", "annual_precipitation_mm", "humidity_proxy_annual",
        "wind_speed_annual_m_s", "storminess_index", "cloud_fraction",
        "soil_moisture_index", "snow_persistence", "vegetation_fraction",
        "surface_albedo", "biome_code", "koppen_code", "slope_deg", "true_color_rgb",
    ):
        mode, dtype = surface_modes.get(field, ("linear", "float32"))
        surface_fields[field] = sample(
            field,
            lambda key, field=field: _field_path(root, "climate_surface", field, key),
            mode=mode,
            dtype=dtype,
        )

    # Terrain render now uses slope recomputed against final eroded terrain.
    manifest.insert(0, _terrain_relief(elev, surface_fields["slope_deg"], output / "00_terrain_relief.png"))
    height_png = output / "01_terrain_heightmap_grayscale16.png"
    height_meta = output / "01_terrain_heightmap_grayscale16.json"
    write_heightmap_png16(
        height_png,
        np.asarray(np.load(elev, mmap_mode="r", allow_pickle=False), dtype=np.float64) / 1000.0,
        metadata_path=height_meta,
    )
    manifest.insert(1, {
        "file": height_png.name,
        "metadata_file": height_meta.name,
        "title": "Final terrain heightmap",
        "encoding": "lossless 16-bit grayscale; global min->0 global max->65535",
        "resolution": [plan.fullview_width, plan.fullview_height],
    })
    manifest.insert(2, _render_scalar(
        elev,
        output / "02_elevation_heatmap.png",
        title="Final elevation / bathymetry",
        units="m",
        palette="elevation",
        low_percentile=0.2,
        high_percentile=99.8,
    ))
    manifest.insert(3, _render_scalar(
        surface_fields["slope_deg"],
        output / "03_slope.png",
        title="Final terrain slope",
        units="degrees",
        palette="erosion",
        fixed_range=(0.0, 65.0),
    ))

    manifest.extend(
        (
            _render_scalar(
                surface_fields["annual_temperature_c"],
                output / "07_temperature_annual_heatmap.png",
                title="Annual mean surface temperature — final terrain downscaled",
                units="°C",
                palette="temperature",
                low_percentile=0.5,
                high_percentile=99.5,
            ),
            _render_scalar(
                surface_fields["annual_precipitation_mm"],
                output / "08_precipitation_annual.png",
                title="Annual precipitation — final-terrain orographic redistribution",
                units="mm/year",
                palette="water",
                fixed_range=(0.0, max(1.0, _sample_percentiles(np.load(surface_fields["annual_precipitation_mm"], mmap_mode="r"), 0.0, 99.7)[1])),
                power=0.52,
            ),
            _render_scalar(
                surface_fields["humidity_proxy_annual"],
                output / "09_humidity_proxy_annual.png",
                title="Annual atmospheric humidity proxy — locally downscaled",
                units="proxy",
                palette="humidity",
                low_percentile=0.5,
                high_percentile=99.5,
            ),
            _render_scalar(
                surface_fields["wind_speed_annual_m_s"],
                output / "10_wind_speed_annual.png",
                title="Annual mean near-surface wind speed — terrain downscaled",
                units="m/s",
                palette="wind",
                low_percentile=0.0,
                high_percentile=99.5,
            ),
            _render_scalar(
                surface_fields["storminess_index"],
                output / "11_storminess_index.png",
                title="Local precipitation-seasonality storminess index",
                units="0–1",
                palette="hazard",
                fixed_range=(0.0, 1.0),
            ),
            _render_scalar(
                surface_fields["cloud_fraction"],
                output / "12_cloud_fraction.png",
                title="Annual cloud fraction proxy",
                units="0–1",
                palette="gray",
                fixed_range=(0.0, 1.0),
            ),
            _render_scalar(
                surface_fields["soil_moisture_index"],
                output / "13_soil_moisture.png",
                title="Final-terrain soil moisture index",
                units="0–1",
                palette="humidity",
                fixed_range=(0.0, 1.0),
            ),
            _render_scalar(
                surface_fields["vegetation_fraction"],
                output / "14_vegetation_fraction.png",
                title="Final-terrain vegetation fraction",
                units="0–1",
                palette="vegetation",
                fixed_range=(0.0, 1.0),
            ),
            _render_scalar(
                surface_fields["snow_persistence"],
                output / "15_snow_persistence.png",
                title="Snow persistence",
                units="0–1",
                palette="snow",
                fixed_range=(0.0, 1.0),
            ),
            _render_scalar(
                surface_fields["surface_albedo"],
                output / "16_surface_albedo.png",
                title="Surface albedo",
                units="fraction",
                palette="gray",
                fixed_range=(0.02, 0.95),
            ),
        )
    )
    manifest.append(
        _render_categorical(
            surface_fields["biome_code"],
            output / "17_biome_ecoregime_with_legend.png",
            title="Local ecoregime / biome proxy — final terrain",
            palette=BIOME_PALETTE,
            labels=BIOME_LABELS,
            columns=2,
        )
    )
    koppen_palette = {
        i: tuple(int(v) for v in _ramp(np.array([[i / max(len(KOPPEN_CLASSES) - 1, 1)]]), PALETTES["temperature"])[0, 0])
        for i in range(len(KOPPEN_CLASSES))
    }
    koppen_labels = {i: name for i, name in enumerate(KOPPEN_CLASSES)}
    manifest.append(
        _render_categorical(
            surface_fields["koppen_code"],
            output / "18_koppen_climate_with_legend.png",
            title="Köppen-Geiger classes recomputed from deepest-tile climate",
            palette=koppen_palette,
            labels=koppen_labels,
            columns=4,
        )
    )
    manifest.append(
        _render_true_color(
            surface_fields["true_color_rgb"],
            output / "19_true_color_refined.png",
            "Refined true color — actual final terrain/surface fields",
        )
    )
    manifest.extend(
        (
            _render_scalar(
                erosion,
                output / "20_erosion_legacy_stream_power.png",
                title="Legacy physical stream-power erosion",
                units="m",
                palette="erosion",
                fixed_range=(0.0, max(0.01, _sample_percentiles(np.load(erosion, mmap_mode="r"), 0.0, 99.7)[1])),
                power=0.65,
            ),
            _render_scalar(
                proc,
                output / "21_erosion_procedural_phase_cell.png",
                title="Procedural phase-cell erosion morphology",
                units="m displacement",
                palette="signed",
                signed=True,
                high_percentile=99.7,
            ),
            _render_scalar(
                combined_path,
                output / "22_erosion_combined_geomorphic_delta.png",
                title="Combined local geomorphic terrain delta",
                units="m",
                palette="signed",
                signed=True,
                high_percentile=99.7,
            ),
        )
    )

    # Fullview climate/surface arrays have all been rendered. Weather generation still
    # needs the deepest climate/surface tiles, but not these ~gigabyte fullview arrays.
    _remove_fullview_temp(*surface_fields.values())

    # Weather hazards use the climate/surface tile products for physically bounded local modulation.
    generate_weather_products(pyramid, plan)
    if cleanup_deepest_products:
        shutil.rmtree(
            root / "ultrares" / "deepest_products" / "climate_surface",
            ignore_errors=True,
        )
    weather_paths: dict[str, Path] = {}
    for field in (*WEATHER_GROUPS, "dominant_weather_code", "dominant_weather_strength"):
        weather_paths[field] = sample(
            f"weather_{field}",
            lambda key, field=field: _field_path(root, "weather", field, key),
            mode="nearest" if field == "dominant_weather_code" else "linear",
            dtype="uint8" if field == "dominant_weather_code" else "float32",
        )
    weather_labels = {i: name.replace("_", " ") for i, name in enumerate(WEATHER_GROUPS)}
    manifest.append(
        _render_dominant_composite(
            weather_paths["dominant_weather_code"],
            weather_paths["dominant_weather_strength"],
            output / "23_weather_hazards_with_legend.png",
            title="Dominant weather-hazard regime",
            palette=WEATHER_PALETTE,
            labels=weather_labels,
        )
    )
    weather_number = 24
    for field in WEATHER_GROUPS:
        manifest.append(
            _render_scalar(
                weather_paths[field],
                output / f"{weather_number:02d}_weather_{field}.png",
                title=f"Weather hazard: {field.replace('_', ' ')}",
                units="normalized simulated intensity",
                palette="hazard",
                fixed_range=(0.0, 1.0),
            )
        )
        weather_number += 1
    _remove_fullview_temp(*weather_paths.values())
    if cleanup_deepest_products:
        shutil.rmtree(
            root / "ultrares" / "deepest_products" / "weather",
            ignore_errors=True,
        )

    # Resource suitability remains global-geology-authoritative but is rebuilt through deepest tiles.
    generate_resource_products(pyramid, plan)
    resource_paths: dict[str, Path] = {}
    for field in (*RESOURCE_GROUPS.keys(), "dominant_resource_code", "dominant_resource_strength"):
        resource_paths[field] = sample(
            f"resource_{field}",
            lambda key, field=field: _field_path(root, "resources", field, key),
            mode="nearest" if field == "dominant_resource_code" else "linear",
            dtype="uint8" if field == "dominant_resource_code" else "float32",
        )
    resource_labels = {i: name.replace("_", " / ") for i, name in enumerate(RESOURCE_GROUPS)}
    manifest.append(
        _render_dominant_composite(
            resource_paths["dominant_resource_code"],
            resource_paths["dominant_resource_strength"],
            output / "30_resources_dominant_with_legend.png",
            title="Dominant resource-suitability group",
            palette=RESOURCE_PALETTE,
            labels=resource_labels,
        )
    )
    number = 31
    for field in RESOURCE_GROUPS:
        manifest.append(
            _render_scalar(
                resource_paths[field],
                output / f"{number:02d}_resource_{field}.png",
                title=f"Resource suitability: {field.replace('_', ' / ')}",
                units="0–1 suitability",
                palette="resource",
                fixed_range=(0.0, 1.0),
            )
        )
        number += 1
    _remove_fullview_temp(*resource_paths.values())
    if cleanup_deepest_products:
        shutil.rmtree(
            root / "ultrares" / "deepest_products" / "resources",
            ignore_errors=True,
        )

    # Scientific temporary fullview arrays used only to render images can now go.
    _remove_fullview_temp(
        base,
        erosion,
        proc,
        combined_path,
        *surface_fields.values(),
    )
    # Keep the final fullview elevation NPY as a numeric companion to the image suite.
    final_elevation = output / "elevation_m.npy"
    os.replace(elev, final_elevation)

    # Remove terrain intermediates no longer needed for the requested image suite.
    if cleanup_heavy_solver_caches:
        for field in ("deposition_m", "hillslope_adjustment_m", "procedural_coherence", "major_river_constraint"):
            shutil.rmtree(
                pyramid.root / "derived" / "local_geomorphology_v1" / field,
                ignore_errors=True,
            )

    for row in manifest:
        path = output / str(row["file"])
        row["bytes"] = int(path.stat().st_size)
        row["sha256"] = _sha256(path)

    manifest_payload = {
        "schema_version": 1,
        "resolution": [plan.fullview_width, plan.fullview_height],
        "deepest_level": plan.finest_level,
        "tile_size": plan.tile_size,
        "deepest_tile_count": plan.finest_tile_count,
        "source_resolution": [plan.source_width, plan.source_height],
        "provenance": (
            "all map pixels reconstructed from deepest cube-sphere tile arrays; terrain-sensitive "
            "climate/surface maps recomputed on final eroded terrain; global-only resource/weather "
            "authority sampled through deepest geometry without invented sub-grid geology/events"
        ),
        "files": manifest,
    }
    _atomic_json(output / "map_manifest.json", manifest_payload)
    shutil.rmtree(temp, ignore_errors=True)
    return output


__all__ = [
    "BIOME_LABELS",
    "KOPPEN_CLASSES",
    "RESOURCE_GROUPS",
    "WEATHER_GROUPS",
    "generate_climate_surface_products",
    "generate_resource_products",
    "generate_weather_products",
    "reconstruct_fullview_maps",
]
