from worldgen.config import WorldConfig
from worldgen.pipeline import WorldPipeline


def _tiny_adaptive_config() -> WorldConfig:
    cfg = WorldConfig(seed=424242)
    cfg.resolution.width = 64
    cfg.resolution.height = 32
    cfg.resolution.history_step_myr = 100
    cfg.tectonics.plate_count = 4
    cfg.tectonics.hotspot_count = 2
    cfg.tectonics.mean_subplates_per_plate = 3.0
    cfg.tectonics.min_subplates_per_plate = 2
    cfg.tectonics.max_subplates_per_plate = 4
    cfg.tectonics.history_grid_height = 24
    cfg.tectonics.boundary_detail_octaves = 2
    cfg.tectonics.boundary_deformation_iterations = 0
    cfg.noise.octaves = 2
    cfg.noise.wave_count = 2
    cfg.noise.domain_warp_strength = 0.0
    cfg.ocean.current_iterations = 4
    cfg.ocean.heat_transport_iterations = 2
    cfg.climate.moisture_iterations = 3
    cfg.climate.thermal_memory_spinup_years = 2
    cfg.hydrology.surface_evolution_iterations = 0
    cfg.hydrology.sediment_routing_passes = 2
    cfg.hydrology.max_river_centerlines = 8
    cfg.weather.hurricane_seed_count = 0
    cfg.society.enabled = False

    cfg.simulation.adaptive_convergence = True
    cfg.simulation.min_earth_system_passes = 2
    cfg.simulation.max_earth_system_passes = 4
    cfg.simulation.required_consecutive_converged_passes = 1
    cfg.simulation.convergence_temperature_c = 1.0e9
    cfg.simulation.convergence_precip_mm_year = 1.0e9
    cfg.simulation.convergence_elevation_m = 1.0e9

    cfg.simulation.adaptive_final_coupling = True
    cfg.simulation.min_final_climate_ocean_passes = 1
    cfg.simulation.max_final_climate_ocean_passes = 3
    cfg.simulation.required_consecutive_final_converged_passes = 1
    cfg.simulation.final_convergence_temperature_c = 1.0e9
    cfg.simulation.final_convergence_precip_mm_year = 1.0e9

    cfg.output.save_png = False
    cfg.output.save_npz = False
    cfg.output.save_json = False
    cfg.output.save_report = False
    return cfg.validate()


def test_adaptive_pipeline_stops_at_minimum_when_residuals_are_inside_tolerance():
    cfg = _tiny_adaptive_config()
    world = WorldPipeline(cfg, progress=None).generate(None)
    summary = world["coupling_summary"]
    assert summary["earth_system_mode"] == "adaptive"
    assert summary["earth_system_passes_executed"] == 2
    assert summary["earth_system_stop_reason"] == "converged"
    assert summary["final_coupling_passes_executed"] == 1
    assert summary["final_coupling_stop_reason"] == "converged"

    macro = [x for x in world["coupling_history"] if x["stage"].startswith("earth_system_pass_")]
    assert len(macro) == 2
    assert macro[0]["full_fidelity"] is False
    assert macro[1]["full_fidelity"] is True
    assert macro[1]["convergence"]["stop"] is True
