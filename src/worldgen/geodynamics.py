from __future__ import annotations

"""Automatic reduced-order silicate and cryogenic geodynamic regime selection."""

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


@dataclass(slots=True)
class GeodynamicRegimeResult:
    regime: str
    silicate_regime: str
    cryogenic_regime: str
    internal_heat_flux_w_m2: float
    tidal_fraction: float
    convective_vigor_index: float
    lid_mobility_index: float
    resurfacing_index: float
    cryotectonic_activity_index: float
    estimated_elastic_lid_km: float
    estimated_ice_shell_mobility: float
    drivers: dict[str, float]
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(-x))


def build_geodynamic_regime(
    astronomy: Any,
    tectonics_config: Any,
    climate: Any | None = None,
    exotic_ocean: Any | None = None,
) -> GeodynamicRegimeResult:
    planet = getattr(astronomy, "planet", {}) or {}
    interior = getattr(astronomy, "interior", {}) or {}
    total_heat = max(float(interior.get("total_internal_heat_flux_w_m2_approx", 0.0)), 0.0)
    tidal = max(float(interior.get("tidal_heating_flux_w_m2", 0.0)), 0.0)
    radiogenic = max(float(interior.get("radiogenic_heat_flux_w_m2", 0.0)), 0.0)
    tidal_fraction = tidal / max(total_heat, 1e-12)
    mass = max(float(planet.get("mass_earth", 1.0)), 0.01)
    radius = max(float(planet.get("radius_earth", 1.0)), 0.05)
    gravity_g = max(float(planet.get("surface_gravity_g", mass / radius**2)), 0.02)
    mean_t_k = 273.15
    if climate is not None:
        mean_t_k = float(np.mean(np.asarray(climate.annual_temperature_c, dtype=float))) + 273.15

    requested = str(getattr(tectonics_config, "geological_activity_mode", "auto"))
    configured_strength = max(float(getattr(tectonics_config, "activity_strength", 1.0)), 0.0)
    ice_shell_km = max(float(getattr(tectonics_config, "ice_shell_thickness_km", 0.0)), 0.0)
    ice_mode = str(getattr(tectonics_config, "ice_geology_mode", "auto"))

    # Heat-flux thresholds are deliberately broad because mantle rheology/composition
    # is not solved. Mass/gravity weakly raise the pressure/retention contribution,
    # while very cold surfaces strengthen the stagnant lid.
    heat_term = math.log10(max(total_heat, 1e-6) / 0.025)
    mass_term = 0.18 * math.log(max(mass, 0.05))
    cold_lid = float(np.clip((273.0 - mean_t_k) / 180.0, -0.5, 1.5))
    convective = _sigmoid(1.35 * heat_term + mass_term - 0.30 * cold_lid) * min(configured_strength, 2.5) / max(1.0, configured_strength)
    tidal_drive = _sigmoid((tidal - 0.02) / 0.015) if tidal > 0 else 0.0
    mobile = np.clip(0.68 * convective + 0.18 * tidal_drive + 0.10 * math.log1p(mass) - 0.20 * max(cold_lid, 0.0), 0.0, 1.0)

    # Explicit user regimes remain authoritative, but automatic mode chooses among
    # physically distinct end-members rather than a binary active/inactive switch.
    if requested != "auto":
        silicate = {
            "active": "mobile_lid",
            "stagnant_lid": "stagnant_lid",
            "inactive": "inactive",
            "tidal": "tidally_forced",
        }.get(requested, requested)
    elif total_heat < 0.008 and tidal < 0.003:
        silicate = "inactive"
    elif tidal_fraction > 0.65 and tidal > 0.025:
        silicate = "tidally_forced"
    elif total_heat > 0.35:
        silicate = "heat_pipe_or_magma_dominated"
    elif mobile > 0.58 and total_heat > 0.045:
        silicate = "mobile_lid"
    elif total_heat > 0.015:
        silicate = "stagnant_lid"
    else:
        silicate = "weakly_active_lid"

    if silicate == "mobile_lid":
        resurfacing = np.clip(0.35 + 0.65 * convective, 0, 1)
    elif silicate == "heat_pipe_or_magma_dominated":
        resurfacing = np.clip(0.72 + 0.28 * convective, 0, 1)
    elif silicate == "tidally_forced":
        resurfacing = np.clip(0.40 + 0.55 * tidal_drive, 0, 1)
    elif silicate == "stagnant_lid":
        resurfacing = np.clip(0.08 + 0.35 * convective, 0, 0.55)
    else:
        resurfacing = np.clip(0.04 + 0.18 * convective, 0, 0.3)

    # Elastic-lid thickness is a diagnostic inverse of heat and mobility, scaled by
    # gravity. It is not a flexural inversion.
    elastic_lid_km = 12.0 + 105.0 * (1.0 - convective) * math.sqrt(gravity_g)
    elastic_lid_km /= 1.0 + 1.8 * tidal_drive
    elastic_lid_km = float(np.clip(elastic_lid_km, 2.0, 220.0))

    # Cryogenic tectonics depends on there being an ice-rich surface/ocean system.
    ocean_class = "dry" if exotic_ocean is None else str(getattr(exotic_ocean, "ocean_class", "dry"))
    water_or_cryo = any(token in ocean_class for token in ("aqueous", "ammonia", "hydrocarbon", "exotic")) or ice_shell_km > 0
    antifreeze = 0.0
    if exotic_ocean is not None:
        comp = getattr(exotic_ocean, "composition_mass_fraction", {}) or {}
        antifreeze = float(comp.get("NH3", 0.0) + 0.7 * comp.get("CH3OH", 0.0))
    shell_scale = max(ice_shell_km, 3.0) if water_or_cryo else 1000.0
    basal_heat = total_heat * (1.0 + 0.8 * tidal_fraction)
    shell_mobility = np.clip(
        (basal_heat / 0.05) ** 0.55 * (25.0 / shell_scale) ** 0.40 * (1.0 + 2.0 * antifreeze),
        0.0, 2.0,
    ) if water_or_cryo else 0.0
    cryo_index = float(np.clip(0.55 * shell_mobility + 0.45 * tidal_drive, 0.0, 1.0))
    if ice_mode == "inactive" or not water_or_cryo:
        cryo = "inactive"
        cryo_index = 0.0
    elif ice_mode == "active":
        cryo = "active_cryotectonics"
        cryo_index = max(cryo_index, 0.55)
    elif cryo_index > 0.72:
        cryo = "active_cryotectonics"
    elif cryo_index > 0.35:
        cryo = "episodic_cryotectonics"
    else:
        cryo = "conductive_ice_shell"

    if cryo in {"active_cryotectonics", "episodic_cryotectonics"} and silicate in {"inactive", "weakly_active_lid"}:
        regime = "cryogeologically_active"
    elif silicate == "tidally_forced" and cryo == "active_cryotectonics":
        regime = "tidal_silicate_and_cryo_active"
    else:
        regime = silicate

    drivers = {
        "radiogenic_heat_flux_w_m2": radiogenic,
        "tidal_heating_flux_w_m2": tidal,
        "surface_temperature_k": mean_t_k,
        "mass_earth": mass,
        "radius_earth": radius,
        "gravity_g": gravity_g,
        "configured_activity_strength": configured_strength,
        "ice_shell_thickness_km": ice_shell_km,
        "antifreeze_mass_fraction_proxy": antifreeze,
    }
    metadata = {
        "model": "automatic reduced-order geodynamic regime classifier",
        "requested_silicate_mode": requested,
        "requested_ice_mode": ice_mode,
        "selection_basis": "internal heat, tidal fraction, mass/gravity, surface thermal lid, ice-shell thickness and antifreeze fraction",
        "limitations": "No mantle mineralogy/rheology, Rayleigh-number convection solve, plate-yield criterion, orbital resonance evolution, or viscoelastic shell FEM.",
    }
    return GeodynamicRegimeResult(
        regime=regime,
        silicate_regime=silicate,
        cryogenic_regime=cryo,
        internal_heat_flux_w_m2=float(total_heat),
        tidal_fraction=float(tidal_fraction),
        convective_vigor_index=float(convective),
        lid_mobility_index=float(mobile),
        resurfacing_index=float(resurfacing),
        cryotectonic_activity_index=float(cryo_index),
        estimated_elastic_lid_km=elastic_lid_km,
        estimated_ice_shell_mobility=float(shell_mobility),
        drivers=drivers,
        metadata=metadata,
    )


__all__ = ["GeodynamicRegimeResult", "build_geodynamic_regime"]
