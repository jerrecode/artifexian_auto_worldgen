from __future__ import annotations

"""Explicit worldgen-to-atmogen SI-unit and result boundary."""

from dataclasses import asdict
import importlib.metadata
import os
from typing import Any, Mapping

import numpy as np

import atmogen


ATMOGEN_COMPATIBLE_REVISION = "32f688391c6c24a97d06e406258b320c241ae648"


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
    positive = {
        str(key): float(value) for key, value in values.items() if float(value) > 0
    }
    total = sum(positive.values())
    if total <= 0:
        raise ValueError("atmosphere composition must contain a positive amount")
    return {key: value / total for key, value in positive.items()}


def _optional_float(cfg, name: str) -> float | None:
    value = getattr(cfg, name, None)
    return None if value is None else float(value)


class AtmogenAdapter:
    """Maps worldgen state into the standalone atmogen SI API."""

    def __init__(self, world_config) -> None:
        self.world_config = world_config

    def _settings(self) -> atmogen.SolverSettings:
        cfg = self.world_config.atmogen
        return atmogen.SolverSettings(
            fidelity=atmogen.Fidelity(cfg.fidelity),
            vertical_layers=int(cfg.vertical_layers),
            top_pressure_pa=float(
                self.world_config.astronomy.atmosphere_top_pressure_bar
            )
            * 1e5,
            chemistry_mode=str(cfg.chemistry_mode),
            radiation_mode=str(cfg.radiation_mode),
            temperature_profile_mode=str(cfg.temperature_profile_mode),
            gray_optical_depth_pressure_exponent=float(
                cfg.gray_optical_depth_pressure_exponent
            ),
            moist_condensible=str(cfg.moist_condensible),
            moist_saturation_threshold=float(cfg.moist_saturation_threshold),
            moist_max_saturation_mixing_ratio=float(
                cfg.moist_max_saturation_mixing_ratio
            ),
            moist_allow_estimated_saturation=bool(
                cfg.moist_allow_estimated_saturation
            ),
            cloud_mode=str(cfg.cloud_mode),
            activity_model=str(getattr(cfg, "activity_model", "auto")),
            liquid_phase_split=bool(getattr(cfg, "liquid_phase_split", True)),
            vertical_transport_mode=str(
                getattr(cfg, "vertical_transport_mode", "none")
            ),
            eddy_diffusivity_m2_s=float(
                getattr(cfg, "eddy_diffusivity_m2_s", 50.0)
            ),
            cloud_suspended_fraction=float(
                getattr(cfg, "cloud_suspended_fraction", 0.01)
            ),
            cloud_condensate_column_cap_kg_m2=float(
                getattr(cfg, "cloud_condensate_column_cap_kg_m2", 0.2)
            ),
            cloud_particle_median_radius_m=float(
                getattr(cfg, "cloud_particle_median_radius_m", 10e-6)
            ),
            cloud_particle_geometric_std=float(
                getattr(cfg, "cloud_particle_geometric_std", 1.4)
            ),
            cloud_particle_density_kg_m3=_optional_float(
                cfg, "cloud_particle_density_kg_m3"
            ),
            cloud_refractive_index_real=_optional_float(
                cfg, "cloud_refractive_index_real"
            ),
            cloud_refractive_index_imag=float(
                getattr(cfg, "cloud_refractive_index_imag", 0.0)
            ),
            gas_dynamic_viscosity_pa_s=float(
                getattr(cfg, "gas_dynamic_viscosity_pa_s", 1.8e-5)
            ),
            cloud_microphysics_timestep_s=float(
                getattr(cfg, "cloud_microphysics_timestep_s", 3600.0)
            ),
            cloud_reevaporation_timescale_s=_optional_float(
                cfg, "cloud_reevaporation_timescale_s"
            ),
            cloud_quadrature_order=int(getattr(cfg, "cloud_quadrature_order", 12)),
            max_iterations=int(cfg.max_iterations),
            relative_temperature_tolerance=float(
                cfg.relative_temperature_tolerance
            ),
            composition_tolerance=float(cfg.composition_tolerance),
            energy_tolerance_w_m2=float(cfg.energy_tolerance_w_m2),
            relaxation=float(cfg.relaxation),
            allow_fidelity_fallback=bool(cfg.allow_fidelity_fallback),
        )

    def _inventory(
        self,
    ) -> tuple[
        dict[str, float], atmogen.ElementInventory, atmogen.SurfaceReservoirs
    ]:
        cfg = self.world_config
        species = _normalize(cfg.astronomy.atmosphere)
        elements = atmogen.species_moles_to_elements(species)
        inventory = atmogen.ElementInventory(
            elements, species, "legacy_species_composition_initial_state"
        )
        surface_mass = {
            str(key): float(value)
            * float(cfg.atmogen.surface_inventory_reference_mass_kg)
            for key, value in cfg.astronomy.surface_volatiles.items()
            if float(value) > 0
        }
        return species, inventory, atmogen.SurfaceReservoirs(surface_mass)

    def stellar_spectrum(self, astronomy_result) -> atmogen.StellarSpectrum:
        flux = (
            1361.0
            * float(astronomy_result.star["luminosity_solar"])
            / float(astronomy_result.planet["semimajor_axis_au"]) ** 2
        )
        return atmogen.blackbody_stellar_spectrum(
            float(astronomy_result.star["effective_temperature_k"]), flux
        )

    def planet_state(
        self,
        astronomy_result,
        *,
        initial_surface_temperature_k: float | None = None,
    ) -> atmogen.PlanetPhysicalState:
        acfg = self.world_config.astronomy
        return atmogen.PlanetPhysicalState(
            radius_m=float(astronomy_result.planet["radius_earth"]) * 6.371e6,
            gravity_m_s2=float(
                astronomy_result.planet["surface_gravity_m_s2"]
            ),
            surface_pressure_pa=float(acfg.atmosphere_pressure_bar) * 1e5,
            initial_surface_temperature_k=(
                float(astronomy_result.planet["equilibrium_temperature_k"])
                if initial_surface_temperature_k is None
                else float(initial_surface_temperature_k)
            ),
            surface_albedo_initial=float(acfg.albedo),
            internal_heat_flux_w_m2=float(
                astronomy_result.interior.get(
                    "total_internal_heat_flux_w_m2_approx", 0.0
                )
            ),
        )

    def solve(self, astronomy_result):
        _species, inventory, surface = self._inventory()
        result = atmogen.solve_planet(
            planet=self.planet_state(astronomy_result),
            star=self.stellar_spectrum(astronomy_result),
            inventory=inventory,
            surface=surface,
            settings=self._settings(),
        )
        self.apply_to_astronomy(astronomy_result, result)
        return result

    def solve_columns(
        self,
        astronomy_result,
        *,
        initial_surface_temperature_k: np.ndarray,
        stellar_flux_scale: np.ndarray,
    ) -> tuple[atmogen.PlanetChemistryResult, ...]:
        """Solve representative geographic columns without duplicating atmogen physics."""
        return self.solve_columns_with_diagnostics(
            astronomy_result,
            initial_surface_temperature_k=initial_surface_temperature_k,
            stellar_flux_scale=stellar_flux_scale,
        ).results

    def solve_columns_with_diagnostics(
        self,
        astronomy_result,
        *,
        initial_surface_temperature_k: np.ndarray,
        stellar_flux_scale: np.ndarray,
    ) -> atmogen.ColumnBatchResult:
        """Solve representative columns with atmogen-owned cache identities."""
        temperatures = np.asarray(
            initial_surface_temperature_k, dtype=float
        ).reshape(-1)
        scales = np.asarray(stellar_flux_scale, dtype=float).reshape(-1)
        if temperatures.shape != scales.shape or temperatures.size == 0:
            raise ValueError(
                "column temperatures and stellar forcing scales must have equal "
                "non-zero length"
            )
        if np.any(~np.isfinite(temperatures)) or np.any(temperatures <= 0):
            raise ValueError("column temperatures must be finite and positive")
        if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("column stellar forcing scales must be finite and positive")
        _species, inventory, surface = self._inventory()
        columns = tuple(
            atmogen.ColumnInput(
                planet=self.planet_state(
                    astronomy_result,
                    initial_surface_temperature_k=float(temp),
                ),
                inventory=inventory,
                surface=surface,
                stellar_flux_scale=float(scale),
            )
            for temp, scale in zip(temperatures, scales, strict=True)
        )
        return atmogen.solve_columns_with_diagnostics(
            atmogen.ColumnBatchInput(
                columns=columns, star=self.stellar_spectrum(astronomy_result)
            ),
            settings=self._settings(),
        )

    @staticmethod
    def apply_to_astronomy(astronomy_result, result) -> None:
        temperature_k = float(result.atmosphere.temperature_k[0])
        fractions = {
            key: float(value)
            for key, value in result.atmosphere.mole_fractions.items()
        }
        pressure_bar = float(astronomy_result.atmosphere["surface_pressure_bar"])
        astronomy_result.atmosphere.update(
            {
                "fractions": fractions,
                "partial_pressures_bar": {
                    key: pressure_bar * value for key, value in fractions.items()
                },
                "mean_molar_mass_g_mol": (
                    result.atmosphere.mean_molar_mass_kg_mol * 1000.0
                ),
                "hydrostatic_thickness_km_approx": (
                    float(result.atmosphere.altitude_m[-1]) / 1000.0
                ),
                "effective_thickness_km_approx": (
                    float(result.atmosphere.altitude_m[-1]) / 1000.0
                ),
                "greenhouse_model": "atmogen",
                "greenhouse_temperature_increment_k_approx": (
                    temperature_k
                    - float(astronomy_result.planet["equilibrium_temperature_k"])
                ),
                "greenhouse_optical_depth_terms": {
                    "model": "atmogen_semi_gray",
                    "total": result.energy_budget.longwave_optical_depth,
                },
                "atmogen": result_summary(result),
            }
        )
        astronomy_result.planet.update(
            {
                "mean_surface_temperature_c_approx": temperature_k - 273.15,
                "greenhouse_increment_k_approx": (
                    temperature_k
                    - float(astronomy_result.planet["equilibrium_temperature_k"])
                ),
                "bond_albedo": float(result.spectra.bond_albedo),
                "geometric_albedo_approx": float(
                    result.spectra.geometric_albedo_approx
                ),
            }
        )
        astronomy_result.volatile_chemistry["authority"] = "atmogen"
        astronomy_result.volatile_chemistry["atmogen"] = result_summary(result)


def result_summary(result) -> dict[str, Any]:
    vertical = result.vertical
    atmosphere = result.atmosphere
    return {
        "provenance": dict(result.provenance),
        "convergence": asdict(result.convergence),
        "diagnostics": dict(result.diagnostics),
        "surface_temperature_k": float(atmosphere.temperature_k[0]),
        "surface_pressure_pa": float(atmosphere.pressure_interface_pa[0]),
        "mean_molar_mass_kg_mol": float(atmosphere.mean_molar_mass_kg_mol),
        "mole_fractions": dict(atmosphere.mole_fractions),
        "vertical_profile": {
            "pressure_pa": atmosphere.pressure_pa.tolist(),
            "pressure_interface_pa": atmosphere.pressure_interface_pa.tolist(),
            "altitude_m": atmosphere.altitude_m.tolist(),
            "temperature_k": atmosphere.temperature_k.tolist(),
            "density_kg_m3": atmosphere.density_kg_m3.tolist(),
            "hydrostatic_relative_residual": float(
                atmosphere.hydrostatic_relative_residual
            ),
            "temperature_profile_model": result.diagnostics.get(
                "temperature_profile_model"
            ),
        },
        "phase_reservoirs": {
            "atmospheric_mass_kg": dict(result.surface.atmospheric_mass_kg),
            "liquid_mass_kg": dict(result.surface.liquid_mass_kg),
            "solid_mass_kg": dict(result.surface.solid_mass_kg),
            "liquid_volume_m3": dict(result.surface.liquid_volume_m3),
            "surface_vapor_mole_fractions": dict(
                result.surface.surface_vapor_mole_fractions
            ),
            "liquid_phases": [
                asdict(phase) for phase in result.surface.liquid_phases
            ],
            "activity_model": str(result.surface.activity_model),
            "fallbacks": list(result.surface.fallbacks),
            "mass_closure_relative": float(result.surface.mass_closure_relative),
        },
        "clouds": asdict(result.clouds),
        "vertical_processes": {
            "model": str(vertical.model),
            "fallbacks": list(vertical.fallbacks),
            "layer_count": int(vertical.layer_thickness_m.size),
            "eddy_diffusivity_m2_s": vertical.eddy_diffusivity_m2_s.tolist(),
            "mixing_timescale_s": vertical.mixing_timescale_s.tolist(),
            "cloud_condensate_kg_m2": vertical.cloud_condensate_kg_m2.tolist(),
            "cloud_settling_velocity_m_s": (
                vertical.cloud_settling_velocity_m_s.tolist()
            ),
            "cloud_sedimentation_flux_kg_m2_s": (
                vertical.cloud_sedimentation_flux_kg_m2_s.tolist()
            ),
            "surface_precipitation_kg_m2_per_step": float(
                vertical.surface_precipitation_kg_m2
            ),
            "precipitation_reevaporated_kg_m2": (
                vertical.precipitation_reevaporated_kg_m2.tolist()
            ),
            "reevaporation_latent_cooling_w_m2_diagnostic": float(
                vertical.reevaporation_latent_cooling_w_m2
            ),
            "mass_closure_relative": float(vertical.mass_closure_relative),
        },
        "energy_budget": asdict(result.energy_budget),
        "spectra": {
            "bond_albedo": float(result.spectra.bond_albedo),
            "geometric_albedo_approx": float(
                result.spectra.geometric_albedo_approx
            ),
            "visible_srgb": list(result.spectra.visible_srgb),
            "shortwave_wavelength_m": (
                result.spectra.shortwave_wavelength_m.tolist()
            ),
            "spectral_albedo": result.spectra.spectral_albedo.tolist(),
            "outgoing_thermal_wavelength_m": (
                result.spectra.thermal_wavelength_m.tolist()
            ),
            "outgoing_thermal_w_m2_m": (
                result.spectra.outgoing_thermal_w_m2_m.tolist()
            ),
        },
    }
