from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_terrain_fidelity_runs.py"
SPEC = importlib.util.spec_from_file_location("compare_terrain_fidelity_runs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_compare_reports_native_gain_and_expected_directional_improvements():
    old = {
        "all_checks_passed": True,
        "plan": {"finest_level": 2, "finest_m_per_sample": 2450.0},
        "sampling": {
            "native_output_resolution": [8192, 4096],
            "native_output_equatorial_m_per_pixel": 4900.0,
            "terrain_minimum_feature_half_wave_m": 4900.0,
        },
        "aggregate": {
            "terrain_square_grid_fourfold_anisotropy": 0.30,
            "river_square_grid_fourfold_anisotropy": 0.50,
            "river_max_straight_run_cells": 240,
            "river_turn_fraction_gt10deg": 0.02,
            "river_mean_tile_median_sinuosity": 1.01,
            "land_fraction_above_1500_m": 0.06,
            "land_fraction_above_2500_m": 0.008,
            "rugged_land_fraction_slope_ge_10deg": 0.001,
            "elevation_high_frequency_rms_m": 0.4,
            "tectonic_microdetail_rms_m": 0.0,
            "max_sediment_closure_relative": 1e-9,
            "same_face_seam_max_abs_m": 0.0,
            "same_face_seam_gradient_rms_m": 50.0,
        },
        "checks": {},
    }
    new = {
        "all_checks_passed": True,
        "plan": {"finest_level": 3, "finest_m_per_sample": 1225.0},
        "sampling": {
            "native_output_resolution": [16384, 8192],
            "native_output_equatorial_m_per_pixel": 2450.0,
            "terrain_minimum_feature_half_wave_m": 2450.0,
        },
        "aggregate": {
            "terrain_square_grid_fourfold_anisotropy": 0.15,
            "river_square_grid_fourfold_anisotropy": 0.30,
            "river_max_straight_run_cells": 80,
            "river_turn_fraction_gt10deg": 0.12,
            "river_mean_tile_median_sinuosity": 1.08,
            "land_fraction_above_1500_m": 0.12,
            "land_fraction_above_2500_m": 0.03,
            "rugged_land_fraction_slope_ge_10deg": 0.01,
            "elevation_high_frequency_rms_m": 18.0,
            "tectonic_microdetail_rms_m": 24.0,
            "max_sediment_closure_relative": 1e-10,
            "same_face_seam_max_abs_m": 0.0,
            "same_face_seam_gradient_rms_m": 30.0,
        },
        "checks": {
            "native_heightmap_bandwidth_saturated": True,
            "terrain_square_grid_imprint_below_limit": True,
            "river_square_grid_imprint_below_limit": True,
            "rivers_turn_at_resolved_scale": True,
            "river_straight_runs_bounded": True,
            "rivers_have_resolved_sinuosity": True,
            "mountain_area_present": True,
            "high_mountain_area_present": True,
            "rugged_mountain_relief_present": True,
            "same_face_tile_seams_watertight": True,
            "same_face_derivative_seams_bounded": True,
        },
    }

    result = MODULE.compare(old, new)
    assert result["native_linear_resolution_gain"] == 2.0
    assert result["sampling"]["native_output_equatorial_m_per_pixel"]["improved"] is True
    assert result["metrics"]["river_max_straight_run_cells"]["improved"] is True
    assert result["metrics"]["river_mean_tile_median_sinuosity"]["improved"] is True
    assert result["metrics"]["land_fraction_above_2500_m"]["improved"] is True
    assert result["regressed_metric_count"] == 0
