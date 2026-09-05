from __future__ import annotations

"""Build physically conditioned parameter fields for procedural erosion."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from atmogen import fluid_transport_properties, liquid_mixture_transport_properties

from .grid import normalize01
from .lithology_properties import properties_for_codes


@dataclass(slots=True)
class ErosionForcing:
    strength: np.ndarray
    preferred_scale_km: np.ndarray
    detail: np.ndarray
    ridge_valley_target: np.ndarray
    orientation_south: np.ndarray
    orientation_east: np.ndarray
    ridge_rounding: np.ndarray
    crease_rounding: np.ndarray
    fluvial_activity: np.ndarray
    pluvial_activity: np.ndarray
    glacial_activity: np.ndarray
    marine_activity: np.ndarray
    chemical_weathering: np.ndarray
    freeze_thaw_activity: np.ndarray
    soil_saturation: np.ndarray
    fluid_mechanical_factor: np.ndarray
    metadata: dict


def _sat(values: np.ndarray, scale: float, power: float = 1.0) -> np.ndarray:
    x = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    return 1.0 - np.exp(-np.power(x / max(float(scale), 1.0e-12), float(power)))


def _fluid_factor(astronomy: Any, condensate_hydrology: Any | None, shape: tuple[int, int]) -> tuple[np.ndarray, dict]:
    gravity = float(getattr(astronomy, "planet", {}).get("surface_gravity_m_s2", 9.80665))
    factor = 1.0
    source = "water_reference"
    species = "H2O"
    mixture = None

    if condensate_hydrology is not None:
        species_mass: dict[str, float] = {}
        for key, values in getattr(condensate_hydrology, "species_monthly_mass_kg_m2", {}).items():
            mass = float(np.sum(np.maximum(np.asarray(values, dtype=np.float64), 0.0)))
            if mass > 0:
                species_mass[str(key)] = mass
        if species_mass:
            mixture = liquid_mixture_transport_properties(species_mass_kg=species_mass)
            species = max(species_mass, key=species_mass.get)
            source = "condensate_mass_weighted_mixture"

    if mixture is not None:
        rho = mixture.density_kg_m3
        mu = mixture.dynamic_viscosity_pa_s
        sigma = mixture.surface_tension_n_m
    else:
        props = fluid_transport_properties(species)
        if props is None:
            props = fluid_transport_properties("H2O")
            source += "+unsupported_species_water_fallback"
        assert props is not None
        rho = props.density_kg_m3
        mu = props.dynamic_viscosity_pa_s
        sigma = props.surface_tension_n_m

    factor = (
        (rho / 997.0) ** 0.55
        * (gravity / 9.80665) ** 0.45
        * (1.0e-3 / max(mu, 1.0e-12)) ** 0.16
        * (0.072 / max(sigma, 1.0e-12)) ** 0.10
    )
    factor = float(np.clip(factor, 0.08, 3.0))
    return np.full(shape, factor, dtype=np.float32), {
        "dominant_condensate": species,
        "fluid_property_source": source,
        "fluid_mechanical_factor": factor,
        "gravity_m_s2": gravity,
    }


def _freeze_thaw(climate: Any, condensate_hydrology: Any | None, moisture: np.ndarray, frost_susceptibility: np.ndarray) -> np.ndarray:
    temp_k = np.asarray(climate.temperature_c, dtype=np.float64) + 273.15
    species = "H2O"
    if condensate_hydrology is not None:
        species = str(getattr(condensate_hydrology, "reference_species", "H2O"))
    props = fluid_transport_properties(species)
    freezing = 273.15 if props is None or props.freezing_temperature_k is None else float(props.freezing_temperature_k)
    shifted = temp_k - freezing
    crossings = np.mean(shifted * np.roll(shifted, -1, axis=0) < 0.0, axis=0)
    continentality = np.asarray(getattr(climate, "continentality_index_c", np.ptp(temp_k, axis=0)), dtype=np.float64)
    continentality_factor = _sat(continentality, 18.0, 0.85)
    return np.clip(crossings * continentality_factor * (0.25 + 0.75 * moisture) * frost_susceptibility, 0.0, 2.0)


def build_erosion_forcing(
    grid,
    terrain,
    ocean,
    climate,
    hydrology,
    geology,
    astronomy,
    cfg,
    *,
    condensate_hydrology: Any | None = None,
    cryogeology: Any | None = None,
) -> ErosionForcing:
    shape = terrain.elevation_km.shape
    land = np.asarray(terrain.land, dtype=bool)
    ocean_mask = np.asarray(terrain.ocean, dtype=bool)
    rock_codes = getattr(geology, "bedrock_code", None)
    if rock_codes is None:
        rock_codes = getattr(geology, "rock_code", None)
    if rock_codes is None:
        raise ValueError("geology must provide bedrock_code or rock_code for erosion forcing")
    rock = np.asarray(rock_codes, dtype=np.int64)
    if rock.shape != shape:
        raise ValueError(
            f"geology rock-code field shape {rock.shape} does not match terrain shape {shape}"
        )
    lith = properties_for_codes(rock)

    soil = np.asarray(getattr(hydrology, "soil_water_storage_mm", np.zeros(shape)), dtype=np.float64)
    soil_capacity = np.maximum(lith["soil_capacity_mm"], 1.0)
    soil_saturation = np.clip(soil / soil_capacity, 0.0, 1.25) * land

    surface_runoff = np.asarray(
        getattr(hydrology, "surface_runoff_mm_year", getattr(hydrology, "runoff", np.zeros(shape))),
        dtype=np.float64,
    )
    discharge = np.asarray(getattr(hydrology, "discharge_index", np.zeros(shape)), dtype=np.float64)
    storm = np.clip(np.asarray(getattr(hydrology, "storminess_index", np.zeros(shape)), dtype=np.float64), 0.0, 1.0)

    if condensate_hydrology is not None:
        liquid_precip = np.asarray(condensate_hydrology.annual_liquid_input_mm, dtype=np.float64)
        solid_precip = np.asarray(condensate_hydrology.annual_solid_input_mm, dtype=np.float64)
    else:
        annual_p = np.maximum(np.asarray(climate.annual_precipitation_mm, dtype=np.float64), 0.0)
        snow_fraction = np.clip(np.asarray(getattr(climate, "snow_fraction", np.zeros(shape)), dtype=np.float64), 0.0, 1.0)
        liquid_precip = annual_p * (1.0 - snow_fraction)
        solid_precip = annual_p * snow_fraction

    fluid_factor, fluid_meta = _fluid_factor(astronomy, condensate_hydrology, shape)
    k_mech = np.clip(lith["mechanical_erodibility"] / 0.82, 0.25, 2.5)
    runoff_term = _sat(surface_runoff, 650.0, 0.85)
    discharge_term = _sat(discharge, 0.22, 0.80)
    precip_term = _sat(liquid_precip, 850.0, 0.82)

    fluvial = np.sqrt(runoff_term * discharge_term) * (0.45 + 0.55 * np.clip(soil_saturation, 0.0, 1.0))
    fluvial *= k_mech * fluid_factor * land
    pluvial = precip_term * (0.35 + 0.65 * storm) * (0.55 + 0.45 * np.clip(soil_saturation, 0.0, 1.0))
    pluvial *= k_mech * fluid_factor * land

    temp = np.asarray(climate.annual_temperature_c, dtype=np.float64)
    cold = np.clip((4.0 - temp) / 22.0, 0.0, 1.0)
    solid_supply = _sat(solid_precip, 350.0, 0.8)
    glacial = cold * solid_supply * lith["glacial_abrasion_susceptibility"] * land
    if cryogeology is not None:
        basal = np.asarray(getattr(cryogeology, "basal_melt_fraction", np.zeros(shape)), dtype=np.float64)
        fracture = np.asarray(getattr(cryogeology, "brittle_fracture_index", np.zeros(shape)), dtype=np.float64)
        glacial *= 0.75 + 0.25 * np.clip(basal + fracture, 0.0, 1.0)

    gy, gx = grid.ops.metric_gradient(np.asarray(terrain.elevation_km, dtype=np.float64))
    slope = np.hypot(gx, gy)
    slope_n = normalize01(slope, robust=True)
    current = normalize01(np.asarray(getattr(ocean, "current_speed", np.zeros(shape)), dtype=np.float64), robust=True)
    shelf = np.asarray(getattr(terrain, "shelf", np.zeros(shape, dtype=bool)), dtype=bool)
    marine = ocean_mask * (0.10 + 0.90 * shelf) * slope_n * (0.35 + 0.65 * current)

    thermal_weather = np.exp(-((temp - 18.0) / 28.0) ** 2)
    chemical = (
        lith["chemical_weatherability"]
        * thermal_weather
        * precip_term
        * (0.35 + 0.65 * np.clip(soil_saturation, 0.0, 1.0))
        * land
    )
    freeze_thaw = _freeze_thaw(climate, condensate_hydrology, np.clip(soil_saturation, 0.0, 1.0), lith["frost_susceptibility"]) * land

    strength = (
        float(cfg.fluvial_weight) * fluvial
        + float(cfg.pluvial_weight) * pluvial
        + float(cfg.glacial_weight) * glacial
        + float(cfg.marine_weight) * marine
        + float(cfg.chemical_weight) * chemical
        + float(cfg.freeze_thaw_weight) * freeze_thaw
    )
    strength = np.clip(strength, 0.0, float(cfg.max_local_strength))

    drainage_density = np.asarray(getattr(hydrology, "subgrid_drainage_density_km_per_km2", np.zeros(shape)), dtype=np.float64)
    density_n = normalize01(drainage_density, robust=True)
    regime_scale = 1.0 / (0.55 + 1.10 * density_n)
    regime_scale *= 1.0 + 0.65 * np.clip(glacial, 0.0, 1.0) + 0.35 * ocean_mask
    preferred_scale = np.clip(
        float(cfg.base_wavelength_km) * regime_scale,
        float(cfg.min_wavelength_km),
        float(cfg.max_wavelength_km),
    )

    # Ridge/valley classification uses curvature plus real drainage topology rather
    # than absolute altitude, so high valleys and submarine ridges classify correctly.
    lap = grid.ops.divergence(gx, gy)
    abs_lap = np.abs(lap[np.isfinite(lap)])
    curv_scale = float(np.quantile(abs_lap, 0.80)) if abs_lap.size else 1.0
    curv = np.tanh(-lap / max(curv_scale, 1.0e-12))
    twi = normalize01(np.asarray(getattr(hydrology, "topographic_wetness_index", np.zeros(shape)), dtype=np.float64), robust=True)
    hand = normalize01(np.asarray(getattr(hydrology, "height_above_nearest_drainage_m", np.zeros(shape)), dtype=np.float64), robust=True)
    channels = np.asarray(getattr(hydrology, "channel_class", np.zeros(shape)), dtype=np.float64)
    valley = np.clip(0.45 * twi + 0.35 * (channels > 0) + 0.20 * (1.0 - hand), 0.0, 1.0)
    ridge = np.clip(0.65 * hand + 0.35 * (1.0 - twi), 0.0, 1.0)
    target = np.tanh(1.10 * curv + 0.85 * ridge - 1.05 * valley)

    norm = np.hypot(gy, gx)
    orientation_south = np.divide(-gy, norm, out=np.zeros_like(gy), where=norm > 1.0e-12)
    orientation_east = np.divide(-gx, norm, out=np.ones_like(gx), where=norm > 1.0e-12)

    cohesion = np.clip(lith["cohesion"], 0.0, 1.0)
    sediment_softness = np.clip(k_mech / 2.5, 0.0, 1.0)
    ridge_rounding = np.clip(0.22 + 0.42 * glacial + 0.22 * sediment_softness, 0.0, 1.0)
    crease_rounding = np.clip(0.10 + 0.38 * glacial + 0.30 * sediment_softness + 0.12 * (1.0 - cohesion), 0.0, 1.0)
    detail = np.clip(0.35 + 0.35 * storm + 0.30 * density_n - 0.38 * glacial, 0.10, 1.0)

    meta = {
        "model": "environment-conditioned procedural erosion forcing",
        "strength_semantics": "dimensionless process activity; not an erosion mass rate",
        "scale_semantics": "preferred procedural wavelength in kilometres",
        "freeze_thaw_semantics": "actual monthly phase-threshold crossings modulated by moisture and continentality",
        "chemical_weathering_semantics": "substrate/moisture/temperature screening only; no general solvent-reaction kinetics",
        **fluid_meta,
    }
    return ErosionForcing(
        strength=np.asarray(strength, np.float32),
        preferred_scale_km=np.asarray(preferred_scale, np.float32),
        detail=np.asarray(detail, np.float32),
        ridge_valley_target=np.asarray(target, np.float32),
        orientation_south=np.asarray(orientation_south, np.float32),
        orientation_east=np.asarray(orientation_east, np.float32),
        ridge_rounding=np.asarray(ridge_rounding, np.float32),
        crease_rounding=np.asarray(crease_rounding, np.float32),
        fluvial_activity=np.asarray(fluvial, np.float32),
        pluvial_activity=np.asarray(pluvial, np.float32),
        glacial_activity=np.asarray(glacial, np.float32),
        marine_activity=np.asarray(marine, np.float32),
        chemical_weathering=np.asarray(chemical, np.float32),
        freeze_thaw_activity=np.asarray(freeze_thaw, np.float32),
        soil_saturation=np.asarray(soil_saturation, np.float32),
        fluid_mechanical_factor=np.asarray(fluid_factor, np.float32),
        metadata=meta,
    )


__all__ = ["ErosionForcing", "build_erosion_forcing"]
