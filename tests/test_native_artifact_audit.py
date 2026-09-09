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
