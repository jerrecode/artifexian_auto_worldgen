from __future__ import annotations

import numpy as np

from worldgen.config import WorldConfig
from worldgen.pipeline import WorldPipeline
from worldgen.surface_liquids import integrate_liquid_volume_m3


def _tiny_composition_world() -> WorldConfig:
    c = WorldConfig(seed=24681357)
    c.resolution.width = 64
    c.resolution.height = 32
    c.resolution.history_myr = 50
    c.resolution.history_step_myr = 50
    c.tectonics.plate_count = 6
    c.tectonics.mean_subplates_per_plate = 2.0
    c.tectonics.min_subplates_per_plate = 2
    c.tectonics.max_subplates_per_plate = 3
    c.tectonics.shape_control_points_per_subplate = 1
    c.tectonics.hotspot_count = 2
    c.climate.moisture_iterations = 4
    c.climate.thermal_memory_spinup_years = 2
    c.hydrology.surface_evolution_iterations = 0
    c.weather.hurricane_seed_count = 0
    c.weather.hurricane_max_steps = 4
    c.society.enabled = False
    c.simulation.earth_system_passes = 1
    c.simulation.final_climate_ocean_passes = 1
    c.astronomy.star_mass_solar = 1.0
    c.astronomy.semimajor_axis_au = 1.0
    c.astronomy.greenhouse_model = "composition"
    c.astronomy.thermodynamics_backend = "builtin"
    c.astronomy.atmosphere_pressure_bar = 1.0
    c.astronomy.atmosphere = {"N2": 0.78, "O2": 0.21, "Ar": 0.0096, "CO2": 0.0004}
    c.astronomy.surface_volatiles = {"H2O": 0.05}
    c.astronomy.surface_condensible = "H2O"
    c.climate.condensible_species = "H2O"
    c.output.save_json = False
    c.output.save_npz = False
    c.output.save_png = False
    c.output.save_report = False
    return c.validate()


def test_canonical_pipeline_applies_inventory_controlled_surface_liquid_geometry():
    world = WorldPipeline(_tiny_composition_world(), progress=None).generate()
    assert "surface_liquids" in world
    liquids = world["surface_liquids"]
    assert liquids.total_liquid_mass_kg > 0.0
    assert liquids.total_liquid_volume_m3 > 0.0
    assert np.any(liquids.liquid_mask)
    np.testing.assert_array_equal(world["terrain"].ocean, liquids.liquid_mask)
    np.testing.assert_allclose(
        world["ocean"].depth_m,
        liquids.liquid_depth_m,
        rtol=0.0,
        atol=0.0,
    )
    integrated = integrate_liquid_volume_m3(
        world["grid"],
        liquids.relative_surface_elevation_km + liquids.liquid_level_km,
        liquids.liquid_level_km,
    )
    rel_error = abs(integrated - liquids.total_liquid_volume_m3) / liquids.total_liquid_volume_m3
    assert rel_error < 1e-8
    assert world["coupling_summary"]["surface_liquid_equilibrium"]["enabled"] is True


def test_legacy_world_does_not_enable_inventory_sea_level_solver():
    c = _tiny_composition_world()
    c.astronomy.greenhouse_model = "legacy"
    world = WorldPipeline(c, progress=None).generate()
    assert "surface_liquids" not in world
