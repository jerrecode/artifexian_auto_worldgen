from __future__ import annotations

from types import SimpleNamespace
import numpy as np

from worldgen.config import WorldConfig
from worldgen.grid import SphereGrid
from worldgen.planetary_chemistry import (
    detect_condensates,
    evaluate_photochemistry,
)
from worldgen.exotic_ocean import build_exotic_ocean
from worldgen.geodynamics import build_geodynamic_regime
from worldgen.geomorphic_fluids import build_geomorphic_fluid_parameters
from worldgen.pipeline import WorldPipeline


def _astronomy_stub(*, role="planet", heat=0.08, tidal=0.0, mass=1.0, radius=1.0, gravity=9.81):
    return SimpleNamespace(
        star={"luminosity_solar": 1.0, "effective_temperature_k": 5772.0},
        planet={
            "semimajor_axis_au": 1.0,
            "body_role": role,
            "mass_earth": mass,
            "radius_earth": radius,
            "surface_gravity_g": gravity / 9.80665,
            "surface_gravity_m_s2": gravity,
        },
        atmosphere={"surface_pressure_bar": 1.0, "fractions": {"N2": 0.78, "O2": 0.21, "CO2": 0.01}},
        interior={
            "total_internal_heat_flux_w_m2_approx": heat,
            "tidal_heating_flux_w_m2": tidal,
            "radiogenic_heat_flux_w_m2": max(heat - tidal, 0.0),
        },
    )


def test_earthlike_oxygen_photochemistry_generates_trace_ozone():
    astro = _astronomy_stub()
    products = evaluate_photochemistry(astro, {"N2": 0.78, "O2": 0.21, "CO2": 0.01})
    assert "O3" in products
    assert 0.0 < products["O3"].abundance_proxy < 0.01
    assert products["O3"].production_index > 0.0


def test_titanlike_nitrogen_methane_photochemistry_generates_tholin_and_nitriles():
    astro = _astronomy_stub(role="moon", heat=0.02, tidal=0.015, gravity=1.35)
    products = evaluate_photochemistry(astro, {"N2": 0.95, "CH4": 0.05})
    for key in ("THOLIN", "C2H6", "C2H2", "HCN"):
        assert key in products
        assert products[key].production_index > 0.0
    assert products["THOLIN"].aerosol
    assert products["THOLIN"].deposited


def test_venuslike_trace_so2_and_water_generate_sulfuric_acid_aerosol():
    astro = _astronomy_stub(gravity=8.87)
    astro.planet["semimajor_axis_au"] = 0.723
    products = evaluate_photochemistry(
        astro,
        {"CO2": 0.964775, "N2": 0.035, "SO2": 1.5e-4, "H2O": 2.0e-5, "Ar": 5.5e-5},
    )
    assert "H2SO4" in products
    assert products["H2SO4"].aerosol
    assert products["H2SO4"].production_index > 0.0


def test_multiple_condensates_can_be_active_without_single_species_exclusivity():
    temp = np.full((16, 32), 94.0 - 273.15)
    condensates = detect_condensates(
        temp,
        {"N2": 0.90, "CH4": 0.06, "C2H6": 0.04},
        1.5,
    )
    assert "CH4" in condensates
    assert "C2H6" in condensates
    assert condensates["CH4"].near_saturation_area_fraction > 0.0
    assert condensates["C2H6"].near_saturation_area_fraction > 0.0


def test_ammonia_water_mixture_depresses_effective_freezing_temperature():
    grid = SphereGrid(32, 16, 1560.0)
    parts = {
        "H2O": SimpleNamespace(liquid_mass_kg=7e18, liquid_density_kg_m3=1000.0),
        "NH3": SimpleNamespace(liquid_mass_kg=3e18, liquid_density_kg_m3=680.0),
    }
    liquids = SimpleNamespace(
        partitions=parts,
        liquid_mask=np.ones(grid.shape, bool),
        liquid_depth_m=np.full(grid.shape, 1000.0, np.float32),
    )
    climate = SimpleNamespace(
        annual_temperature_c=np.full(grid.shape, -70.0),
        temperature_c=np.stack([np.full(grid.shape, -70.0 + 2 * np.sin(m / 12 * 2*np.pi)) for m in range(12)]),
        wind_u=np.zeros((12, *grid.shape)),
        wind_v=np.zeros((12, *grid.shape)),
    )
    ocean = SimpleNamespace(current_speed=np.full(grid.shape, 0.2))
    exo = build_exotic_ocean(grid, _astronomy_stub(role="moon", gravity=1.3), climate, ocean, liquids)
    assert exo.ocean_class == "ammonia-water cryo-ocean"
    assert exo.effective_freezing_temperature_k < 230.0
    assert exo.bulk_density_kg_m3 > 700.0


def test_automatic_geodynamics_detects_tidally_forced_cryo_activity():
    astro = _astronomy_stub(role="moon", heat=0.18, tidal=0.15, gravity=1.3)
    cfg = SimpleNamespace(
        geological_activity_mode="auto",
        activity_strength=1.0,
        ice_shell_thickness_km=20.0,
        ice_geology_mode="auto",
    )
    climate = SimpleNamespace(annual_temperature_c=np.full((8, 16), -120.0))
    exo = SimpleNamespace(ocean_class="ammonia-water cryo-ocean", composition_mass_fraction={"H2O": 0.8, "NH3": 0.2})
    result = build_geodynamic_regime(astro, cfg, climate=climate, exotic_ocean=exo)
    assert result.silicate_regime == "tidally_forced"
    assert result.cryogenic_regime in {"active_cryotectonics", "episodic_cryotectonics"}
    assert result.tidal_fraction > 0.7


def test_methane_geomorphic_scaling_uses_actual_fluid_and_low_gravity():
    astro = _astronomy_stub(role="moon", gravity=1.35)
    exo = SimpleNamespace(
        composition_mass_fraction={"CH4": 0.7, "C2H6": 0.3},
        bulk_density_kg_m3=460.0,
        dynamic_viscosity_mpa_s=0.20,
        surface_tension_mn_m=18.0,
    )
    params = build_geomorphic_fluid_parameters(astro, exo)
    assert params.active_fluid == "CH4"
    assert 0.08 <= params.stream_power_multiplier < 1.0
    assert params.evaporation_loss_multiplier > 1.0


def _tiny_advanced_config() -> WorldConfig:
    c = WorldConfig(seed=61231)
    c.resolution.width = 64
    c.resolution.height = 32
    c.resolution.history_myr = 100
    c.resolution.history_step_myr = 50
    c.tectonics.plate_count = 5
    c.tectonics.min_subplates_per_plate = 1
    c.tectonics.max_subplates_per_plate = 2
    c.tectonics.mean_subplates_per_plate = 1.4
    c.tectonics.shape_control_points_per_subplate = 1
    c.tectonics.history_grid_height = 24
    c.tectonics.hotspot_count = 3
    c.tectonics.boundary_deformation_iterations = 1
    c.climate.moisture_iterations = 5
    c.climate.thermal_memory_spinup_years = 1
    c.ocean.current_iterations = 10
    c.ocean.heat_transport_iterations = 3
    c.hydrology.surface_evolution_iterations = 1
    c.hydrology.sediment_routing_passes = 3
    c.weather.hurricane_seed_count = 0
    c.society.enabled = False
    c.output.save_json = False
    c.output.save_npz = False
    c.output.save_png = False
    c.output.save_report = False
    c.simulation.earth_system_passes = 1
    c.simulation.final_climate_ocean_passes = 1
    c.astronomy.greenhouse_model = "composition"
    c.astronomy.atmosphere = {"N2": 0.95, "CH4": 0.05}
    c.astronomy.atmosphere_pressure_bar = 1.5
    c.astronomy.surface_volatiles = {"CH4": 8e-6, "C2H6": 4e-6}
    c.astronomy.surface_condensible = "CH4"
    c.climate.condensible_species = "CH4"
    c.ocean.fluid_species = "CH4"
    return c.validate()


def test_canonical_pipeline_exports_advanced_planetary_layers_and_final_optics():
    world = WorldPipeline(_tiny_advanced_config(), progress=None).generate()
    climate_temperature_k = np.asarray(world["climate"].temperature_c, dtype=float) + 273.15
    assert np.isfinite(climate_temperature_k).all()
    assert float(np.min(climate_temperature_k)) >= 1.0 - 1.0e-9
    assert world["climate"].metadata["minimum_model_temperature_k"] == 1.0
    for key in (
        "surface_liquids", "volatile_cycle", "exotic_ocean", "geodynamics",
        "cryogeology", "geomorphic_fluid_parameters", "exotic_geomorphology",
        "condensate_hydrology", "planetary_appearance",
    ):
        assert key in world
    assert world["geomorphic_fluid_parameters"].active_fluid in {"CH4", "C2H6"}
    assert world["geodynamics"].regime
    assert world["volatile_cycle"].metadata["precipitation_partition_conservative"]

    pa = world["planetary_appearance"]
    hydro = world["hydrology"]
    np.testing.assert_allclose(
        world["appearance"].soil_moisture_index,
        pa.ground_liquid_humidity_index,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(hydro.soil_liquid_storage_mm, hydro.soil_water_storage_mm)
    np.testing.assert_array_equal(hydro.subsurface_liquid_storage_mm, hydro.groundwater_storage_mm)
    np.testing.assert_array_equal(hydro.solid_condensate_storage_mm, hydro.snowpack_mm)
    assert pa.metadata["ground_humidity_semantics"].startswith("fractional pore-volume wetness")
    wet = np.asarray(world["surface_liquids"].liquid_mask, dtype=bool)
    if np.any(wet):
        assert "CH4" in pa.metadata["surface_liquid_optics"]["composition_volume_fraction"]
        liquid_rgb = np.asarray(pa.surface_liquid_rgb, dtype=float)[wet]
        assert np.all(np.isfinite(liquid_rgb))
        assert np.any(liquid_rgb > 0.0)
    else:
        # This compact integration fixture retains Earth's stellar forcing.  At its
        # resulting warm surface temperature the configured trace hydrocarbons remain
        # vapor, so claiming a liquid optical mixture would be physically incorrect.
        assert world["surface_liquids"].total_liquid_mass_kg == 0.0
        assert pa.metadata["surface_liquid_optics"]["composition_volume_fraction"] == {}
    clear = np.asarray(world["appearance"].true_color_rgb, dtype=np.int16)
    toa = np.asarray(world["appearance"].true_color_with_clouds_rgb, dtype=np.int16)
    assert clear.shape == toa.shape == (*world["grid"].shape, 3)
    assert np.mean(np.abs(clear.astype(float) - toa.astype(float))) > 1.0
