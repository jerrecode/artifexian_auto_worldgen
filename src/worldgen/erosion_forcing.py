from __future__ import annotations

"""Build physically conditioned parameter fields for procedural erosion."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from atmogen import (
    fluid_transport_properties,
    liquid_mixture_transport_fields,
    liquid_mixture_transport_properties,
)

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


def _mechanical_factor_from_properties(
    density_kg_m3: np.ndarray | float,
    dynamic_viscosity_pa_s: np.ndarray | float,
    surface_tension_n_m: np.ndarray | float,
    gravity_m_s2: float,
) -> np.ndarray:
    rho = np.maximum(np.asarray(density_kg_m3, dtype=np.float64), 1.0e-12)
    mu = np.maximum(np.asarray(dynamic_viscosity_pa_s, dtype=np.float64), 1.0e-12)
    sigma = np.maximum(np.asarray(surface_tension_n_m, dtype=np.float64), 1.0e-12)
    factor = (
        (rho / 997.0) ** 0.55
        * (float(gravity_m_s2) / 9.80665) ** 0.45
        * (1.0e-3 / mu) ** 0.16
        * (0.072 / sigma) ** 0.10
    )
    return np.clip(factor, 0.08, 3.0)


def _liquid_species_mass_fields(
    grid,
    condensate_hydrology: Any | None,
    shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Return annual precipitating liquid mass per cell [kg] by condensate.

    The condensate bridge stores total species precipitation mass plus separate
    liquid/solid volume depths. Their depth ratio therefore supplies the phase
    fraction without assuming another density. Multiplying the resulting kg/m² by
    spherical cell area gives true per-cell mass for atmogen's mixture API.
    """
    if condensate_hydrology is None:
        return {}
    monthly_mass = getattr(
        condensate_hydrology, "species_monthly_mass_kg_m2", {}
    ) or {}
    monthly_liquid = getattr(
        condensate_hydrology, "species_monthly_liquid_depth_mm", {}
    ) or {}
    monthly_solid = getattr(
        condensate_hydrology, "species_monthly_solid_depth_mm", {}
    ) or {}
    if not monthly_mass:
        return {}

    area_m2 = (
        np.asarray(grid.cell_area_weights, dtype=np.float64)
        * 4.0
        * np.pi
        * (float(grid.radius_km) * 1000.0) ** 2
    )
    if area_m2.shape != shape or not np.isfinite(area_m2).all() or np.any(area_m2 <= 0.0):
        raise ValueError("grid cell areas must be finite, positive, and match erosion shape")

    out: dict[str, np.ndarray] = {}
    for raw_key, raw_mass in monthly_mass.items():
        key = str(raw_key)
        mass_source = np.asarray(raw_mass)
        liquid_source = np.asarray(
            monthly_liquid.get(raw_key, np.zeros_like(mass_source))
        )
        solid_source = np.asarray(
            monthly_solid.get(raw_key, np.zeros_like(mass_source))
        )
        expected = (12, *shape)
        if (
            mass_source.shape != expected
            or liquid_source.shape != expected
            or solid_source.shape != expected
        ):
            raise ValueError(
                f"condensate phase fields for {key!r} must have shape {expected}"
            )

        # Accumulate month by month instead of promoting several complete
        # (12,H,W) tensors to float64. Peak new working memory therefore scales
        # with one raster slice rather than the full seasonal cube.
        annual_liquid_mass_kg_m2 = np.zeros(shape, dtype=np.float64)
        for month in range(12):
            mass = np.asarray(mass_source[month], dtype=np.float64)
            liquid = np.asarray(liquid_source[month], dtype=np.float64)
            solid = np.asarray(solid_source[month], dtype=np.float64)
            if (
                not np.isfinite(mass).all()
                or not np.isfinite(liquid).all()
                or not np.isfinite(solid).all()
                or np.any(mass < 0.0)
                or np.any(liquid < 0.0)
                or np.any(solid < 0.0)
            ):
                raise ValueError(
                    f"condensate phase fields for {key!r} must be finite and non-negative"
                )
            phase_depth = liquid + solid
            condensed = phase_depth > 1.0e-12
            np.divide(
                liquid,
                phase_depth,
                out=phase_depth,
                where=condensed,
            )
            phase_depth[~condensed] = 0.0
            annual_liquid_mass_kg_m2 += mass * phase_depth

        cell_mass = annual_liquid_mass_kg_m2 * area_m2
        if np.any(cell_mass > 0.0):
            out[key] = cell_mass
    return out


def _fluid_factor(
    grid,
    astronomy: Any,
    condensate_hydrology: Any | None,
    shape: tuple[int, int],
) -> tuple[np.ndarray, dict]:
    gravity = float(
        getattr(astronomy, "planet", {}).get("surface_gravity_m_s2", 9.80665)
    )
    if not np.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("surface gravity must be finite and positive")

    water = fluid_transport_properties("H2O")
    assert water is not None
    water_factor = float(
        _mechanical_factor_from_properties(
            water.density_kg_m3,
            water.dynamic_viscosity_pa_s,
            water.surface_tension_n_m,
            gravity,
        )
    )
    factor = np.full(shape, water_factor, dtype=np.float64)
    source = "water_reference"
    species = "H2O"
    active = np.zeros(shape, dtype=bool)
    liquid_species = _liquid_species_mass_fields(
        grid, condensate_hydrology, shape
    )

    unsupported = sorted(
        key
        for key, values in liquid_species.items()
        if np.any(values > 0.0) and fluid_transport_properties(key) is None
    )
    if liquid_species:
        totals = {
            key: float(np.sum(values, dtype=np.float64))
            for key, values in liquid_species.items()
        }
        species = max(totals, key=totals.get)

    if liquid_species and not unsupported:
        total_species_mass = {
            key: float(np.sum(values, dtype=np.float64))
            for key, values in liquid_species.items()
        }
        global_mixture = liquid_mixture_transport_properties(
            species_mass_kg=total_species_mass
        )
        spatial = liquid_mixture_transport_fields(
            species_mass_kg=liquid_species
        )
        if global_mixture is not None and spatial is not None:
            fallback = float(
                _mechanical_factor_from_properties(
                    global_mixture.density_kg_m3,
                    global_mixture.dynamic_viscosity_pa_s,
                    global_mixture.surface_tension_n_m,
                    gravity,
                )
            )
            factor.fill(fallback)
            active = np.asarray(spatial.active_mask, dtype=bool)
            local = _mechanical_factor_from_properties(
                spatial.density_kg_m3,
                spatial.dynamic_viscosity_pa_s,
                spatial.surface_tension_n_m,
                gravity,
            )
            factor[active] = local[active]
            source = "spatial_condensate_liquid_mixture+global_liquid_mixture_fallback"
    elif unsupported:
        source = "unsupported_liquid_species_water_reference_fallback"
    elif condensate_hydrology is not None:
        source = "water_reference+no_liquid_condensate"

    if not np.isfinite(factor).all():
        raise RuntimeError("fluid mechanical factor became non-finite")
    factor = np.clip(factor, 0.08, 3.0)
    weights = np.asarray(grid.cell_area_weights, dtype=np.float64)
    mean_factor = float(np.sum(factor * weights))
    active_fraction = float(np.sum(weights[active])) if np.any(active) else 0.0
    return np.asarray(factor, dtype=np.float32), {
        "dominant_condensate": species,
        "liquid_transport_species": sorted(liquid_species),
        "unsupported_liquid_transport_species": unsupported,
        "fluid_property_source": source,
        "fluid_mechanical_factor": mean_factor,
        "fluid_mechanical_factor_min": float(np.min(factor)),
        "fluid_mechanical_factor_max": float(np.max(factor)),
        "fluid_mechanical_spatial_active_fraction": active_fraction,
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
        raise ValueError(
            "geology must provide bedrock_code or rock_code for erosion forcing"
        )
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

    fluid_factor, fluid_meta = _fluid_factor(
        grid, astronomy, condensate_hydrology, shape
    )
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
