import math

from worldgen.benchmarks import (
    PROFILES,
    benchmark_drainage_graph,
    benchmark_priority_flood,
    config_for_profile,
)


def test_benchmark_profiles_are_valid_2_to_1_worlds():
    for name, profile in PROFILES.items():
        assert profile.width == 2 * profile.height, name
        assert profile.width >= 128
        assert profile.history_step_myr > 0
        assert profile.climate_iterations > 0


def test_micro_benchmark_config_disables_expensive_output_only():
    cfg = config_for_profile("micro", seed=12345)
    assert cfg.seed == 12345
    assert (cfg.resolution.width, cfg.resolution.height) == (128, 64)
    assert not cfg.output.save_png
    assert not cfg.output.save_npz
    assert not cfg.output.save_json
    assert not cfg.output.save_report
    # The benchmark is still a physical pipeline, not a fake/no-op workload.
    assert cfg.simulation.earth_system_passes >= 1
    assert cfg.hydrology.surface_evolution_iterations >= 1


def test_priority_flood_benchmark_preserves_reference_semantics():
    result = benchmark_priority_flood(width=64, height=32, seed=7)
    assert result["shape"] == [32, 64]
    assert result["reference"]["wall_seconds"] >= 0.0
    assert result["selected_backend"]["wall_seconds"] >= 0.0
    assert math.isfinite(result["speedup"])
    assert result["speedup"] > 0.0
    assert result["max_abs_difference_km"] <= 1.0e-12


def test_drainage_benchmark_routes_every_source_to_outlet():
    width, height = 64, 32
    result = benchmark_drainage_graph(width=width, height=height)
    assert result["nodes"] == width * height
    assert result["outlet_accumulation"] == float(width * height)
    assert result["build"]["wall_seconds"] >= 0.0
    assert result["accumulate"]["wall_seconds"] >= 0.0
