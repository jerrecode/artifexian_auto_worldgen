from __future__ import annotations

import numpy as np

from worldgen import WorldConfig, WorldPipeline


def test_public_pipeline_includes_procedural_erosion_layer_but_defaults_off():
    cfg = WorldConfig().validate()
    assert cfg.procedural_erosion.enabled is False
    assert WorldPipeline.__module__ == "worldgen.pipeline_procedural_erosion"


def test_procedural_erosion_config_validation_rejects_invalid_scale_order():
    cfg = WorldConfig()
    cfg.procedural_erosion.min_wavelength_km = 100.0
    cfg.procedural_erosion.max_wavelength_km = 50.0
    try:
        cfg.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("invalid procedural wavelength range was accepted")


def test_enabled_procedural_erosion_runs_through_canonical_recoupling():
    cfg = WorldConfig(seed=606060)
    cfg.resolution.width = 64
    cfg.resolution.height = 32
    cfg.resolution.history_myr = 50
    cfg.resolution.history_step_myr = 25
    cfg.tectonics.plate_count = 6
    cfg.tectonics.history_grid_height = 32
    cfg.tectonics.shape_control_points_per_subplate = 1
    cfg.climate.moisture_iterations = 6
    cfg.hydrology.surface_evolution_iterations = 1
    cfg.simulation.earth_system_passes = 1
    cfg.simulation.final_climate_ocean_passes = 1
    cfg.weather.hurricane_seed_count = 2
    cfg.weather.hurricane_max_steps = 8
    cfg.society.enabled = False
    cfg.output.save_json = False
    cfg.output.save_npz = False
    cfg.output.save_png = False
    cfg.output.save_report = False
    cfg.procedural_erosion.enabled = True
    cfg.procedural_erosion.octaves = 2
    cfg.procedural_erosion.base_wavelength_km = 5000.0
    cfg.procedural_erosion.min_wavelength_km = 2500.0
    cfg.procedural_erosion.max_wavelength_km = 7000.0
    cfg.procedural_erosion.base_amplitude_m = 8.0
    cfg.procedural_erosion.max_displacement_m = 20.0
    cfg.validate()

    world = WorldPipeline(cfg, progress=None).generate()
    result = world["procedural_erosion"]
    forcing = world["procedural_erosion_forcing"]

    assert np.isfinite(result.delta_height_m).all()
    assert np.any(np.abs(result.delta_height_m) > 0.0)
    assert result.metadata["octaves_executed"] >= 1
    assert forcing.strength.shape == world["terrain"].elevation_km.shape
    assert world["coupling_summary"]["procedural_erosion"]["recoupled"] is True
    assert world["terrain"].metadata["procedural_erosion_active"] is True
