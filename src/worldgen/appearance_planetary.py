from __future__ import annotations

"""Final composition-aware surface/atmosphere appearance pass.

The legacy appearance model is intentionally retained as the Earth-calibrated surface
renderer.  This module replaces only assumptions that are chemically wrong for
composition-driven worlds: blue-water oceans, a 0 C definition of every solid
condensate, water-only soil humidity semantics, and atmosphere-free cloud composites.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .appearance import _ROCK_RGB, _render_rgb
from .aerosol_cloud_decks import apply_composition_cloud_decks
from .planetary_optics import (
    atmosphere_visible_optics,
    composite_top_of_atmosphere,
    ground_liquid_humidity_index,
    liquid_mixture_optics,
    render_surface_liquid_rgb,
)


@dataclass(slots=True)
class PlanetaryAppearanceDiagnostics:
    surface_liquid_rgb: np.ndarray
    atmospheric_haze_optical_depth: np.ndarray
    ground_liquid_humidity_index: np.ndarray
    solid_condensate_persistence: np.ndarray
    metadata: dict

    def to_dict(self) -> dict:
        return dict(self.metadata)


def _solid_condensate_persistence(world: dict[str, Any]) -> np.ndarray:
    terrain = world["terrain"]
    hydro = world["hydrology"]
    forcing = world.get("condensate_hydrology")
    land = np.asarray(terrain.land, dtype=bool)
    stored = np.maximum(
        np.asarray(getattr(hydro, "snowpack_mm", np.zeros(land.shape)), dtype=np.float64), 0.0
    )
    stored_index = stored / (stored + 35.0)
    if forcing is None:
        return np.asarray(world["appearance"].snow_persistence, dtype=np.float32)
    solid = np.maximum(np.asarray(forcing.annual_solid_input_mm, dtype=np.float64), 0.0)
    total = np.maximum(np.asarray(forcing.annual_total_precipitation_depth_mm, dtype=np.float64), 0.0)
    solid_fraction = np.divide(
        solid,
        np.maximum(total, 1.0e-12),
        out=np.zeros_like(solid),
        where=total > 1.0e-12,
    )
    supply = solid / (solid + 90.0)
    persistence = np.clip(0.62 * stored_index + 0.38 * solid_fraction * supply, 0.0, 1.0)
    return (persistence * land).astype(np.float32)


def _organic_surface_deposition(world: dict[str, Any]) -> np.ndarray:
    shape = world["grid"].shape
    volatile = world.get("volatile_cycle")
    if volatile is None:
        return np.zeros(shape, dtype=np.float32)
    result = np.zeros(shape, dtype=np.float64)
    weights = {
        "THOLIN": 1.0,
        "HCN": 0.55,
        "C2H2": 0.45,
        "C2H4": 0.35,
        "S8": 0.20,
    }
    for key, weight in weights.items():
        field = getattr(volatile, "photochemical_deposition_by_species", {}).get(key)
        if field is not None:
            result += float(weight) * np.asarray(field, dtype=np.float64)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _biological_surface_plausible(world: dict[str, Any]) -> bool:
    """Conservatively suppress Earth vegetation on clearly non-Earth surface chemistry.

    This is not an astrobiology model; it merely prevents the legacy renderer from
    painting green forests on a 700 K sulfur/CO2 world or a methane-lake cryogenic moon.
    """
    liquids = world.get("surface_liquids")
    optics = liquid_mixture_optics(liquids)
    water_fraction = float(optics["volume_fractions"].get("H2O", 0.0))
    mean_t = float(np.sum(
        np.asarray(world["climate"].annual_temperature_c, dtype=np.float64)
        * np.asarray(world["grid"].cell_area_weights, dtype=np.float64)
    ))
    atmosphere = getattr(world["astronomy"], "atmosphere", {}) or {}
    fractions = atmosphere.get("fractions", {}) or {}
    oxygen = float(fractions.get("O2", 0.0))
    return water_fraction >= 0.45 and -25.0 <= mean_t <= 60.0 and oxygen >= 0.01


def _rerender_surface_base(world: dict[str, Any], cfg: Any, solid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    appearance = world["appearance"]
    terrain = world["terrain"]
    land = np.asarray(terrain.land, dtype=bool)
    plausible_bio = _biological_surface_plausible(world)
    if plausible_bio:
        vegetation = np.asarray(appearance.vegetation_fraction, dtype=np.float32)
        forest = np.asarray(appearance.forest_fraction, dtype=np.float32)
        grass = np.asarray(appearance.grass_fraction, dtype=np.float32)
        bare = np.asarray(appearance.bare_ground_fraction, dtype=np.float32)
    else:
        vegetation = np.zeros(land.shape, dtype=np.float32)
        forest = np.zeros(land.shape, dtype=np.float32)
        grass = np.zeros(land.shape, dtype=np.float32)
        bare = land.astype(np.float32)
        appearance.vegetation_fraction = vegetation
        appearance.forest_fraction = forest
        appearance.grass_fraction = grass
        appearance.bare_ground_fraction = bare

    kwargs = (
        world["grid"], terrain, world["ocean"], world["geology"], world["weather"]
    )
    annual = _render_rgb(
        *kwargs, vegetation, forest, grass, bare, solid,
        np.asarray(appearance.water_turbidity, dtype=np.float32), cfg.appearance,
    )
    # Seasonal biological change is deliberately not synthesized for exotic worlds.
    # For plausible H2O/O2 worlds retain the already calculated January/July surface
    # variability; otherwise use the chemically corrected annual substrate as baseline.
    if plausible_bio:
        jan = np.asarray(appearance.true_color_january_rgb, dtype=np.float32) / 255.0
        jul = np.asarray(appearance.true_color_july_rgb, dtype=np.float32) / 255.0
    else:
        jan = annual.copy()
        jul = annual.copy()
    return annual.astype(np.float32), jan.astype(np.float32), jul.astype(np.float32)


def apply_composition_aware_appearance(world: dict[str, Any], cfg: Any) -> dict[str, Any]:
    if str(getattr(cfg.astronomy, "greenhouse_model", "legacy")) != "composition":
        return world

    appearance = world["appearance"]
    terrain = world["terrain"]
    land = np.asarray(terrain.land, dtype=bool)
    liquids = world.get("surface_liquids")
    volatile = world.get("volatile_cycle")
    forcing = world.get("condensate_hydrology")

    ground_humidity = ground_liquid_humidity_index(world["hydrology"], land, forcing)
    solid = _solid_condensate_persistence(world)
    appearance.soil_moisture_index = ground_humidity.astype(np.float32)
    appearance.snow_persistence = solid.astype(np.float32)

    annual, jan, jul = _rerender_surface_base(world, cfg, solid)
    organic = _organic_surface_deposition(world)
    optics = atmosphere_visible_optics(world["astronomy"], volatile, shape=world["grid"].shape)
    optics = apply_composition_cloud_decks(optics, volatile)
    sky_reflection = np.clip(
        0.72 * np.asarray(optics.rayleigh_scatter_rgb, dtype=np.float64)
        + 0.28 * np.asarray(optics.haze_rgb, dtype=np.float64),
        0.0,
        1.0,
    )

    liquid_rgb = np.zeros((*land.shape, 3), dtype=np.float32)
    liquid_meta: dict[str, Any] = {"composition_volume_fraction": {}}
    if liquids is not None and np.any(np.asarray(liquids.liquid_mask, dtype=bool)):
        wet = np.asarray(liquids.liquid_mask, dtype=bool)
        bed_rgb = _ROCK_RGB[
            np.clip(np.asarray(world["geology"].rock_code, dtype=int), 0, len(_ROCK_RGB) - 1)
        ]
        liquid_rgb, liquid_meta = render_surface_liquid_rgb(
            np.asarray(liquids.liquid_depth_m, dtype=np.float64),
            bed_rgb,
            liquids,
            atmosphere_reflection_rgb=sky_reflection,
            turbidity=np.asarray(appearance.water_turbidity, dtype=np.float64),
            organic_deposition=organic,
        )
        annual[wet] = liquid_rgb[wet]
        jan[wet] = liquid_rgb[wet]
        jul[wet] = liquid_rgb[wet]

        # The exotic-ocean freezing model replaces the water-specific 0 C sea-ice
        # assumption in the base renderer.
        exotic_ocean = world.get("exotic_ocean")
        if exotic_ocean is not None:
            ice_fraction = np.clip(
                np.asarray(exotic_ocean.sea_ice_fraction, dtype=np.float64), 0.0, 1.0
            )[..., None]
            ice_rgb = np.asarray([0.89, 0.93, 0.94], dtype=np.float64)
            annual[wet] = annual[wet] * (1.0 - 0.78 * ice_fraction[wet]) + ice_rgb * (0.78 * ice_fraction[wet])
            jan[wet] = jan[wet] * (1.0 - 0.78 * ice_fraction[wet]) + ice_rgb * (0.78 * ice_fraction[wet])
            jul[wet] = jul[wet] * (1.0 - 0.78 * ice_fraction[wet]) + ice_rgb * (0.78 * ice_fraction[wet])

    # Wet pore space darkens the substrate and picks up a very small tint from the
    # active surface liquid. Because ground_humidity is driven by final bucket storage
    # plus annual liquid precipitation, rainfall/methane-rain/etc. visibly wets ground.
    gh = np.asarray(ground_humidity, dtype=np.float64)[..., None]
    darken = 1.0 - 0.20 * gh
    annual[land] *= darken[land]
    jan[land] *= darken[land]
    jul[land] *= darken[land]
    if liquid_meta.get("composition_volume_fraction"):
        wet_tint = np.asarray(sky_reflection, dtype=np.float64)
        tint = 0.035 * gh
        annual[land] = annual[land] * (1.0 - tint[land]) + wet_tint * tint[land]
        jan[land] = jan[land] * (1.0 - tint[land]) + wet_tint * tint[land]
        jul[land] = jul[land] * (1.0 - tint[land]) + wet_tint * tint[land]

    # Deposited refractory photochemistry affects dry ground as well as liquids.
    if np.any(organic > 0):
        oo = np.clip(np.asarray(organic, dtype=np.float64), 0.0, 1.0)[..., None]
        deposit_rgb = np.asarray([0.30, 0.17, 0.075], dtype=np.float64)
        annual[land] = annual[land] * (1.0 - 0.42 * oo[land]) + deposit_rgb * (0.42 * oo[land])
        jan[land] = jan[land] * (1.0 - 0.42 * oo[land]) + deposit_rgb * (0.42 * oo[land])
        jul[land] = jul[land] * (1.0 - 0.42 * oo[land]) + deposit_rgb * (0.42 * oo[land])

    annual = np.clip(annual, 0.0, 1.0)
    jan = np.clip(jan, 0.0, 1.0)
    jul = np.clip(jul, 0.0, 1.0)
    appearance.true_color_rgb = np.rint(annual * 255.0).astype(np.uint8)
    appearance.true_color_january_rgb = np.rint(jan * 255.0).astype(np.uint8)
    appearance.true_color_july_rgb = np.rint(jul * 255.0).astype(np.uint8)

    appearance.true_color_with_clouds_rgb = np.rint(
        composite_top_of_atmosphere(
            annual, appearance.cloud_fraction_annual, optics,
            cloud_max_optical_opacity=float(cfg.appearance.cloud_max_optical_opacity),
        ) * 255.0
    ).astype(np.uint8)
    appearance.true_color_january_with_clouds_rgb = np.rint(
        composite_top_of_atmosphere(
            jan, appearance.cloud_fraction_monthly[0], optics,
            cloud_max_optical_opacity=float(cfg.appearance.cloud_max_optical_opacity),
        ) * 255.0
    ).astype(np.uint8)
    appearance.true_color_july_with_clouds_rgb = np.rint(
        composite_top_of_atmosphere(
            jul, appearance.cloud_fraction_monthly[6], optics,
            cloud_max_optical_opacity=float(cfg.appearance.cloud_max_optical_opacity),
        ) * 255.0
    ).astype(np.uint8)

    haze = np.asarray(optics.haze_optical_depth, dtype=np.float32)
    if haze.ndim == 0:
        haze = np.full(land.shape, float(haze), dtype=np.float32)
    diagnostics = PlanetaryAppearanceDiagnostics(
        surface_liquid_rgb=np.asarray(liquid_rgb, dtype=np.float32),
        atmospheric_haze_optical_depth=haze,
        ground_liquid_humidity_index=np.asarray(ground_humidity, dtype=np.float32),
        solid_condensate_persistence=np.asarray(solid, dtype=np.float32),
        metadata={
            "model": "composition-aware liquid optics + generic condensate ground storage + broadband atmospheric visible transfer",
            "surface_liquid_optics": liquid_meta,
            "atmosphere_visible_optics": optics.metadata,
            "ground_humidity_semantics": "fractional pore-volume wetness of the active liquid condensate mixture; not intrinsically H2O",
            "ground_humidity_precipitation_coupling": "final conservative soil storage plus annual thermodynamically liquid condensate input",
            "solid_condensate_semantics": "generic stored/falling solid condensate; replaces hard-coded 0 C water snow on exotic worlds",
            "earth_vegetation_suppressed": not _biological_surface_plausible(world),
        },
    )
    world["planetary_appearance"] = diagnostics
    appearance.metadata = {
        **appearance.metadata,
        **diagnostics.metadata,
        "mean_land_ground_liquid_humidity_index": float(
            np.average(ground_humidity[land], weights=world["grid"].cell_area_weights[land])
        ) if np.any(land) else 0.0,
    }
    return world


__all__ = ["PlanetaryAppearanceDiagnostics", "apply_composition_aware_appearance"]
