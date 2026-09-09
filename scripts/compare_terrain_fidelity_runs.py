#!/usr/bin/env python3
from __future__ import annotations

"""Compare two ultra-resolution terrain-detail audits.

The report is deliberately metric-based: a new run is not considered an
improvement merely because its raster is larger. It compares physical/native
sampling, terrain bandwidth, grid anisotropy, river routing geometry, mountain
coverage, ruggedness, erosion magnitudes and seam quality.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


LOWER_IS_BETTER = (
    "terrain_square_grid_fourfold_anisotropy",
    "river_square_grid_fourfold_anisotropy",
    "river_max_straight_run_cells",
    "max_sediment_closure_relative",
    "same_face_seam_max_abs_m",
    "same_face_seam_gradient_rms_m",
)

HIGHER_IS_BETTER = (
    "river_turn_fraction_gt10deg",
    "river_mean_tile_median_sinuosity",
    "land_fraction_above_1500_m",
    "land_fraction_above_2500_m",
    "rugged_land_fraction_slope_ge_10deg",
    "elevation_high_frequency_rms_m",
    "tectonic_microdetail_rms_m",
)

SAMPLING_LOWER_IS_BETTER = (
    "native_output_equatorial_m_per_pixel",
    "terrain_minimum_feature_half_wave_m",
)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _delta(old: float | None, new: float | None, *, lower_is_better: bool) -> dict[str, Any]:
    if old is None or new is None:
        return {
            "old": old,
            "new": new,
            "absolute_change": None,
            "relative_change": None,
            "improved": None,
        }
    change = new - old
    relative = change / abs(old) if abs(old) > 1.0e-30 else None
    improved = new < old if lower_is_better else new > old
    return {
        "old": old,
        "new": new,
        "absolute_change": change,
        "relative_change": relative,
        "improved": bool(improved),
    }


def compare(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    old_agg = old.get("aggregate", {})
    new_agg = new.get("aggregate", {})
    old_sampling = old.get("sampling", {})
    new_sampling = new.get("sampling", {})

    metrics: dict[str, Any] = {}
    for name in LOWER_IS_BETTER:
        metrics[name] = _delta(
            _number(old_agg.get(name)),
            _number(new_agg.get(name)),
            lower_is_better=True,
        )
    for name in HIGHER_IS_BETTER:
        metrics[name] = _delta(
            _number(old_agg.get(name)),
            _number(new_agg.get(name)),
            lower_is_better=False,
        )

    sampling: dict[str, Any] = {}
    for name in SAMPLING_LOWER_IS_BETTER:
        sampling[name] = _delta(
            _number(old_sampling.get(name)),
            _number(new_sampling.get(name)),
            lower_is_better=True,
        )

    old_resolution = old_sampling.get("native_output_resolution")
    new_resolution = new_sampling.get("native_output_resolution")
    old_plan = old.get("plan", {})
    new_plan = new.get("plan", {})
    old_checks = old.get("checks", {})
    new_checks = new.get("checks", {})

    old_width = (
        float(old_resolution[0])
        if isinstance(old_resolution, list) and len(old_resolution) == 2
        else None
    )
    new_width = (
        float(new_resolution[0])
        if isinstance(new_resolution, list) and len(new_resolution) == 2
        else None
    )
    linear_gain = (
        new_width / old_width
        if old_width is not None and new_width is not None and old_width > 0
        else None
    )

    critical_new_checks = {
        name: new_checks.get(name)
        for name in (
            "native_heightmap_bandwidth_saturated",
            "terrain_square_grid_imprint_below_limit",
            "river_square_grid_imprint_below_limit",
            "rivers_turn_at_resolved_scale",
            "river_straight_runs_bounded",
            "rivers_have_resolved_sinuosity",
            "mountain_area_present",
            "high_mountain_area_present",
            "rugged_mountain_relief_present",
            "same_face_tile_seams_watertight",
            "same_face_derivative_seams_bounded",
        )
    }

    comparable = [
        item["improved"]
        for item in [*metrics.values(), *sampling.values()]
        if item["improved"] is not None
    ]
    return {
        "schema_version": 1,
        "old": {
            "all_checks_passed": bool(old.get("all_checks_passed", False)),
            "native_output_resolution": old_resolution,
            "finest_level": old_plan.get("finest_level"),
            "finest_m_per_sample": old_plan.get("finest_m_per_sample"),
        },
        "new": {
            "all_checks_passed": bool(new.get("all_checks_passed", False)),
            "native_output_resolution": new_resolution,
            "finest_level": new_plan.get("finest_level"),
            "finest_m_per_sample": new_plan.get("finest_m_per_sample"),
        },
        "native_linear_resolution_gain": linear_gain,
        "sampling": sampling,
        "metrics": metrics,
        "critical_new_checks": critical_new_checks,
        "comparable_metric_count": len(comparable),
        "improved_metric_count": int(sum(bool(value) for value in comparable)),
        "regressed_metric_count": int(sum(not bool(value) for value in comparable)),
        "note": (
            "Individual metrics have different physical trade-offs. This comparison "
            "does not replace the new run's absolute acceptance gates or visual review."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_audit", type=Path)
    parser.add_argument("new_audit", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    old = json.loads(args.old_audit.read_text(encoding="utf-8"))
    new = json.loads(args.new_audit.read_text(encoding="utf-8"))
    result = compare(old, new)
    raw = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(raw + "\n", encoding="utf-8")
    print(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
