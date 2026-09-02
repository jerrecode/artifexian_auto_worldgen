from dataclasses import asdict

import numpy as np

from worldgen.astronomy import build_astronomy
from worldgen.atmogen_adapter import (
    ATMOGEN_COMPATIBLE_REVISION,
    AtmogenAdapter,
    atmogen_runtime_metadata,
    result_summary,
)
from worldgen.checkpoint import stage_cache_key
from worldgen.config import WorldConfig
from worldgen.fingerprints import stage_source_fingerprint
from worldgen.grid import SphereGrid
from worldgen.rng import RngPool
from worldgen.surface_liquids import (
    integrate_liquid_volume_m3,
    place_partitioned_liquids,
)


def configured_world() -> WorldConfig:
    cfg = WorldConfig()
    cfg.astronomy.semimajor_axis_au = 1.0
    cfg.astronomy.greenhouse_model = "composition"
    cfg.atmogen.enabled = True
    cfg.atmogen.chemistry_mode = "fixed_species"
    cfg.atmogen.vertical_layers = 12
    return cfg.validate()


def test_adapter_replaces_old_greenhouse_authority_and_records_revision():
    cfg = configured_world()
    astro = build_astronomy(cfg.astronomy, RngPool(cfg.seed)("astronomy"))
    old_model = astro.atmosphere["greenhouse_model"]
    result = AtmogenAdapter(cfg).solve(astro)
    assert old_model == "composition"
    assert astro.atmosphere["greenhouse_model"] == "atmogen"
    assert astro.volatile_chemistry["authority"] == "atmogen"
    assert result.convergence.converged
    assert abs(result.energy_budget.imbalance_w_m2) < 1e-9
    assert result.surface.mass_closure_relative < 1e-12
    metadata = atmogen_runtime_metadata()
    assert metadata["compatible_git_revision"] == ATMOGEN_COMPATIBLE_REVISION
    assert metadata["package_version"] == "0.12.0"
    assert metadata["api_schema_version"] == 11
    assert metadata["data_schema_version"] == 4


def test_standard_fidelity_exposes_nonisothermal_vertical_profile():
    cfg = configured_world()
    cfg.atmogen.fidelity = "STANDARD"
    cfg.atmogen.vertical_layers = 16
    cfg.validate()
    astro = build_astronomy(cfg.astronomy, RngPool(cfg.seed)("astronomy"))
    result = AtmogenAdapter(cfg).solve(astro)
    summary = result_summary(result)
    profile = summary["vertical_profile"]
    temperature = np.asarray(profile["temperature_k"], dtype=float)
    pressure = np.asarray(profile["pressure_pa"], dtype=float)

    assert result.convergence.converged
    assert result.diagnostics["temperature_profile_model"] == (
        "dry_gray_radiative_convective"
    )
    assert profile["temperature_profile_model"] == "dry_gray_radiative_convective"
    assert temperature.size == 16
    assert pressure.size == 16
    assert np.ptp(temperature) > 1.0
    assert temperature[0] > temperature[-1]
    assert np.all(np.diff(pressure) < 0.0)
    assert profile["hydrostatic_relative_residual"] < 2e-12
    assert atmogen_runtime_metadata()["package_version"] == "0.12.0"


def test_local_column_elevation_changes_pressure_with_explicit_provenance():
    cfg = configured_world()
    astro = build_astronomy(cfg.astronomy, RngPool(cfg.seed)("astronomy"))
    batch = AtmogenAdapter(cfg).solve_columns_with_diagnostics(
        astro,
        initial_surface_temperature_k=np.asarray([280.0, 280.0]),
        stellar_flux_scale=np.asarray([1.0, 1.0]),
        surface_elevation_m=np.asarray([0.0, 2500.0]),
    )
    sea_level, mountain = batch.results
    assert mountain.atmosphere.pressure_interface_pa[0] < (
        sea_level.atmosphere.pressure_interface_pa[0]
    )
    assert batch.diagnostics.surface_boundary_modes == (
        "hydrostatic_adjusted",
        "hydrostatic_adjusted",
    )
    assert batch.diagnostics.surface_boundaries[1]["elevation_delta_m"] == 2500.0


def test_high_fidelity_worldgen_adapter_uses_water_saturated_profile():
    cfg = configured_world()
    cfg.astronomy.star_mass_solar = 1.0
    cfg.astronomy.planet_mass_earth = 1.0
    cfg.astronomy.planet_density_g_cm3 = 5.51
    cfg.astronomy.atmosphere_pressure_bar = 1.01325
    cfg.astronomy.atmosphere = {
        "N2": 0.7808,
        "O2": 0.2095,
        "Ar": 0.0093,
        "CO2": 0.0004,
    }
    cfg.astronomy.surface_volatiles = {"H2O": 1.05}
    cfg.atmogen.fidelity = "HIGH"
    cfg.atmogen.vertical_layers = 20
    cfg.atmogen.temperature_profile_mode = "auto"
    cfg.atmogen.moist_condensible = "H2O"
    cfg.validate()

    astro = build_astronomy(cfg.astronomy, RngPool(cfg.seed)("astronomy"))
    result = AtmogenAdapter(cfg).solve(astro)

    assert result.convergence.converged
    assert result.diagnostics["temperature_profile_model"] == (
        "dilute_saturated_gray_radiative_convective"
    )
    assert result.diagnostics["moist_condensible"] == "H2O"
    assert result.diagnostics["saturated_convective_constraint_layers"] > 0
    assert result.atmosphere.temperature_k[0] > result.atmosphere.temperature_k[-1]
    assert result.atmosphere.hydrostatic_relative_residual < 2e-12


def test_atmogen_phase_volume_uses_worldgen_spherical_geometry():
    cfg = configured_world()
    astro = build_astronomy(cfg.astronomy, RngPool(cfg.seed)("astronomy"))
    result = AtmogenAdapter(cfg).solve(astro)
    grid = SphereGrid(64, 32, 6371.0)
    bed = np.linspace(-5, 5, grid.width * grid.height).reshape(grid.shape)
    inventories = {"H2O": cfg.atmogen.surface_inventory_reference_mass_kg}
    placed = place_partitioned_liquids(
        grid,
        bed,
        inventories,
        result.surface,
        temperature_k=float(result.atmosphere.temperature_k[0]),
    )
    direct = integrate_liquid_volume_m3(grid, bed, placed.liquid_level_km)
    assert (
        abs(direct - placed.total_liquid_volume_m3)
        / max(placed.total_liquid_volume_m3, 1)
        < 1e-9
    )
    assert placed.metadata["phase_authority"] == "atmogen"


def test_atmogen_settings_change_only_dependent_stage_key_inputs():
    a = configured_world()
    b = configured_world()
    b.atmogen.vertical_layers = 36
    # Tectonics has no atmogen dependency; its stage-scoped input stays identical.
    tect_a = {
        key: a.to_dict()[key]
        for key in ("seed", "resolution", "tectonics", "noise")
    }
    tect_b = {
        key: b.to_dict()[key]
        for key in ("seed", "resolution", "tectonics", "noise")
    }
    assert tect_a == tect_b
    fp = stage_source_fingerprint("atmogen_column")
    atm_a = {"astronomy": asdict(a.astronomy), "atmogen": asdict(a.atmogen)}
    atm_b = {"astronomy": asdict(b.astronomy), "atmogen": asdict(b.atmogen)}
    assert stage_cache_key("atmogen_column", atm_a, fp) != stage_cache_key(
        "atmogen_column", atm_b, fp
    )
