from __future__ import annotations

"""Explicit worldgen-to-atmogen SI-unit and result boundary."""

from dataclasses import asdict
import importlib.metadata
import os
from typing import Any, Mapping

import numpy as np

import atmogen


ATMOGEN_COMPATIBLE_REVISION = "ea48caee2efb01f6e1a63d0abf1f2f139dddcd7b"


def atmogen_runtime_metadata() -> dict[str, Any]:
    try:
        distribution_version = importlib.metadata.version("atmogen")
    except importlib.metadata.PackageNotFoundError:
        distribution_version = atmogen.__version__
    return {
        "package_version": distribution_version,
        "api_schema_version": atmogen.API_SCHEMA_VERSION,
        "data_schema_version": atmogen.DATA_SCHEMA_VERSION,
        "database_sha256": atmogen.BUILTIN_DATABASE.revision_hash,
        "compatible_git_revision": ATMOGEN_COMPATIBLE_REVISION,
        "runtime_git_revision": os.environ.get("ATMOGEN_GIT_COMMIT"),
    }


def _normalize(values: Mapping[str, float]) -> dict[str, float]:
    positive = {str(key): float(value) for key, value in values.items() if float(value) > 0}
    total = sum(positive.values())
    if total <= 0:
        raise ValueError("atmosphere composition must contain a positive amount")
    return {key: value / total for key, value in positive.items()}


class AtmogenAdapter:
    """Maps one representative world column without importing worldgen from atmogen."""

    def __init__(self, world_config) -> None:
        self.world_config = world_config

    def _settings(self) -> atmogen.SolverSettings:
        cfg = self.world_config.atmogen
        return atmogen.SolverSettings(
            fidelity=atmogen.Fidelity(cfg.fidelity), vertical_layers=int(cfg.vertical_layers),
            top_pressure_pa=float(self.world_config.astronomy.atmosphere_top_pressure_bar) * 1e5,
            chemistry_mode=str(cfg.chemistry_mode), radiation_mode=str(cfg.radiation_mode),
            cloud_mode=str(cfg.cloud_mode), max_iterations=int(cfg.max_iterations),
            relative_temperature_tolerance=float(cfg.relative_temperature_tolerance),
            composition_tolerance=float(cfg.composition_tolerance),
            energy_tolerance_w_m2=float(cfg.energy_tolerance_w_m2), relaxation=float(cfg.relaxation),
            allow_fidelity_fallback=bool(cfg.allow_fidelity_fallback),
        )

    def solve(self, astronomy_result):
        cfg = self.world_config
        acfg = cfg.astronomy
        species = _normalize(acfg.atmosphere)
        # Compatibility conversion deliberately retains molecular input as an
        # initial-state hint; metadata never calls it an equilibrium specification.
        elements = atmogen.species_moles_to_elements(species)
        inventory = atmogen.ElementInventory(elements, species, "legacy_species_composition_initial_state")
        surface_mass = {str(key): float(value) * float(cfg.atmogen.surface_inventory_reference_mass_kg)
                        for key, value in acfg.surface_volatiles.items() if float(value) > 0}
        planet = atmogen.PlanetPhysicalState(
            radius_m=float(astronomy_result.planet["radius_earth"]) * 6.371e6,
            gravity_m_s2=float(astronomy_result.planet["surface_gravity_m_s2"]),
            surface_pressure_pa=float(acfg.atmosphere_pressure_bar) * 1e5,
            # The old composition-greenhouse value is not used as the authoritative
            # initial state: start from the orbit/albedo equilibrium temperature.
            initial_surface_temperature_k=float(astronomy_result.planet["equilibrium_temperature_k"]),
            surface_albedo_initial=float(acfg.albedo),
            internal_heat_flux_w_m2=float(astronomy_result.interior.get("total_internal_heat_flux_w_m2_approx", 0.0)),
        )
        flux = 1361.0 * float(astronomy_result.star["luminosity_solar"]) / float(astronomy_result.planet["semimajor_axis_au"])**2
        star = atmogen.blackbody_stellar_spectrum(float(astronomy_result.star["effective_temperature_k"]), flux)
        result = atmogen.solve_planet(planet=planet, star=star, inventory=inventory,
                                      surface=atmogen.SurfaceReservoirs(surface_mass), settings=self._settings())
        self.apply_to_astronomy(astronomy_result, result)
        return result

    @staticmethod
    def apply_to_astronomy(astronomy_result, result) -> None:
        temperature_k = float(result.atmosphere.temperature_k[0])
        fractions = {key: float(value) for key, value in result.atmosphere.mole_fractions.items()}
        pressure_bar = float(astronomy_result.atmosphere["surface_pressure_bar"])
        astronomy_result.atmosphere.update({
            "fractions": fractions,
            "partial_pressures_bar": {key: pressure_bar * value for key, value in fractions.items()},
            "mean_molar_mass_g_mol": result.atmosphere.mean_molar_mass_kg_mol * 1000.0,
            "hydrostatic_thickness_km_approx": float(result.atmosphere.altitude_m[-1]) / 1000.0,
            "effective_thickness_km_approx": float(result.atmosphere.altitude_m[-1]) / 1000.0,
            "greenhouse_model": "atmogen",
            "greenhouse_temperature_increment_k_approx": temperature_k - float(astronomy_result.planet["equilibrium_temperature_k"]),
            "greenhouse_optical_depth_terms": {"model": "atmogen_semi_gray", "total": result.energy_budget.longwave_optical_depth},
            "atmogen": result_summary(result),
        })
        astronomy_result.planet.update({
            "mean_surface_temperature_c_approx": temperature_k - 273.15,
            "greenhouse_increment_k_approx": temperature_k - float(astronomy_result.planet["equilibrium_temperature_k"]),
            "bond_albedo": float(result.spectra.bond_albedo),
            "geometric_albedo_approx": float(result.spectra.geometric_albedo_approx),
        })
        astronomy_result.volatile_chemistry["authority"] = "atmogen"
        astronomy_result.volatile_chemistry["atmogen"] = result_summary(result)


def result_summary(result) -> dict[str, Any]:
    return {
        "provenance": dict(result.provenance),
        "convergence": asdict(result.convergence),
        "diagnostics": dict(result.diagnostics),
        "surface_temperature_k": float(result.atmosphere.temperature_k[0]),
        "surface_pressure_pa": float(result.atmosphere.pressure_interface_pa[0]),
        "mean_molar_mass_kg_mol": float(result.atmosphere.mean_molar_mass_kg_mol),
        "mole_fractions": dict(result.atmosphere.mole_fractions),
        "phase_reservoirs": {
            "atmospheric_mass_kg": dict(result.surface.atmospheric_mass_kg),
            "liquid_mass_kg": dict(result.surface.liquid_mass_kg),
            "solid_mass_kg": dict(result.surface.solid_mass_kg),
            "liquid_volume_m3": dict(result.surface.liquid_volume_m3),
            "surface_vapor_mole_fractions": dict(result.surface.surface_vapor_mole_fractions),
            "liquid_phases": [asdict(phase) for phase in result.surface.liquid_phases],
            "activity_model": str(result.surface.activity_model),
            "fallbacks": list(result.surface.fallbacks),
            "mass_closure_relative": float(result.surface.mass_closure_relative),
        },
        "clouds": asdict(result.clouds),
        "energy_budget": asdict(result.energy_budget),
        "spectra": {
            "bond_albedo": float(result.spectra.bond_albedo),
            "geometric_albedo_approx": float(result.spectra.geometric_albedo_approx),
            "visible_srgb": list(result.spectra.visible_srgb),
            "shortwave_wavelength_m": result.spectra.shortwave_wavelength_m.tolist(),
            "spectral_albedo": result.spectra.spectral_albedo.tolist(),
            "outgoing_thermal_wavelength_m": result.spectra.thermal_wavelength_m.tolist(),
            "outgoing_thermal_w_m2_m": result.spectra.outgoing_thermal_w_m2_m.tolist(),
        },
    }
