from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_native_terrain_rivers.py"
SPEC = importlib.util.spec_from_file_location("audit_native_terrain_rivers", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_authoritative_native_audit_requires_every_new_terrain_and_river_gate():
    checks = {name: True for name in MODULE.REQUIRED_TERRAIN_CHECKS}
    report = {
        "all_checks_passed": True,
        "checks": checks,
        "aggregate": {
            "river_mean_tile_median_sinuosity": 1.08,
            "terrain_square_grid_fourfold_anisotropy": 0.12,
        },
        "sampling": {
            "native_output_resolution": [16384, 8192],
            "minimum_feature_width_in_native_output_pixels": 1.0,
        },
    }
    result = MODULE.audit_authoritative_report(report)
    assert result["all_checks_passed"] is True
    assert result["missing_required_checks"] == []
    assert result["failed_required_checks"] == []
    assert result["sampling"]["native_output_resolution"] == [16384, 8192]

    checks["rivers_have_resolved_sinuosity"] = False
    result = MODULE.audit_authoritative_report(report)
    assert result["failed_required_checks"] == ["rivers_have_resolved_sinuosity"]


def test_fourfold_gradient_diagnostic_distinguishes_axis_grid_from_mixed_relief():
    y, x = np.mgrid[:128, :128]
    axis_grid = np.sin(2.0 * np.pi * x / 8.0)
    grid_metric = MODULE._fourfold_gradient_anisotropy(axis_grid)
    assert grid_metric["fourfold_anisotropy"] > 0.85

    mixed = (
        np.sin(2.0 * np.pi * (0.77 * x + 0.31 * y) / 13.0)
        + 0.8 * np.sin(2.0 * np.pi * (-0.28 * x + 0.91 * y) / 17.0)
        + 0.6 * np.sin(2.0 * np.pi * (0.53 * x - 0.64 * y) / 23.0)
    )
    mixed_metric = MODULE._fourfold_gradient_anisotropy(mixed)
    assert mixed_metric["fourfold_anisotropy"] < grid_metric["fourfold_anisotropy"]


def test_projected_internal_tile_seam_diagnostic_is_zero_on_uniform_render():
    values = np.full((256, 512), 37.0, dtype=np.float64)
    result = MODULE._river_tile_seam_diagnostic(values, level=3, samples_per_curve=96)
    assert result["level"] == 3
    assert result["sample_count"] > 0
    assert abs(float(result["median_abs_jump"])) <= 1.0e-12
    assert abs(float(result["p95_abs_jump"])) <= 1.0e-12
    assert abs(float(result["median_jump_ratio_to_baseline"])) <= 1.0e-3
    assert abs(float(result["p95_jump_ratio_to_baseline"])) <= 1.0e-3


def test_eighth_moment_detects_axis_plus_diagonal_d8_symmetry():
    y, x = np.mgrid[:256, :256]
    field = np.zeros((256, 256), dtype=np.float64)
    field[:128] = np.sin(2.0 * np.pi * x[:128] / 12.0)
    field[128:] = np.sin(
        2.0 * np.pi * (x[128:] + y[128:]) / (12.0 * np.sqrt(2.0))
    )
    metrics = MODULE._gradient_angular_anisotropy(field)
    assert metrics["angular_moment_4"] < 0.25
    assert metrics["angular_moment_8"] > 0.80
