#!/usr/bin/env python3
from __future__ import annotations

"""Independent QA for native terrain/river artifacts.

This utility analyzes exported artifacts rather than generator-internal arrays. It
complements terrain_detail_audit.json by checking the actual downloadable files:
TIFF sample/channel metadata, raster dimensions, encoded height range, and coarse
render-level directional diagnostics.

Directional image diagnostics are reported, not treated as physical truth. The
authoritative routing/terrain acceptance criteria remain the generator's tile-space
audit because map projection itself can introduce orientation bias.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REQUIRED_TERRAIN_CHECKS = (
    "native_heightmap_bandwidth_saturated",
    "new_high_frequency_terrain_exists",
    "tectonic_microdetail_active",
    "terrain_square_grid_imprint_below_limit",
    "legacy_erosion_active",
    "procedural_erosion_active",
    "legacy_and_procedural_overlap_on_active_tiles",
    "procedural_not_below_sampling_limit",
    "procedural_reaches_finest_safe_band",
    "procedural_does_not_overlap_source_resolved_band",
    "procedural_band_reaches_target_resolution",
    "procedural_magnitude_not_overwhelming_legacy",
    "final_rivers_exist",
    "river_square_grid_imprint_below_limit",
    "rivers_turn_at_resolved_scale",
    "river_straight_runs_bounded",
    "rivers_have_resolved_sinuosity",
    "mountain_area_present",
    "high_mountain_area_present",
    "rugged_mountain_relief_present",
    "sediment_mass_closure",
    "same_face_tile_seams_watertight",
    "same_face_derivative_seams_bounded",
)


def _fourfold_gradient_anisotropy(values: np.ndarray) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 3:
        raise ValueError("directional diagnostic requires a 2-D raster of at least 3x3")
    gy, gx = np.gradient(a)
    weight = np.hypot(gx, gy)
    finite = np.isfinite(weight) & np.isfinite(gx) & np.isfinite(gy)
    if not np.any(finite):
        return {"fourfold_anisotropy": 0.0, "gradient_weight": 0.0}
    threshold = float(np.percentile(weight[finite], 55.0))
    active = finite & (weight > max(threshold, 1.0e-12))
    if not np.any(active):
        return {"fourfold_anisotropy": 0.0, "gradient_weight": 0.0}
    angle = np.arctan2(gy[active], gx[active])
    w = weight[active]
    moment = np.sum(w * np.exp(4j * angle))
    total = float(np.sum(w))
    return {
        "fourfold_anisotropy": float(abs(moment) / max(total, 1.0e-30)),
        "gradient_weight": total,
    }


def audit_authoritative_report(report: Mapping[str, Any]) -> dict[str, Any]:
    checks = report.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("terrain audit is missing a checks mapping")
    missing = [name for name in REQUIRED_TERRAIN_CHECKS if name not in checks]
    failed = [name for name in REQUIRED_TERRAIN_CHECKS if checks.get(name) is not True]
    aggregate = report.get("aggregate", {})
    sampling = report.get("sampling", {})
    return {
        "all_checks_passed": bool(report.get("all_checks_passed", False)),
        "required_check_count": len(REQUIRED_TERRAIN_CHECKS),
        "missing_required_checks": missing,
        "failed_required_checks": failed,
        "aggregate": {
            key: aggregate.get(key)
            for key in (
                "terrain_square_grid_fourfold_anisotropy",
                "river_square_grid_fourfold_anisotropy",
                "river_stream_direction_count",
                "river_turn_fraction_gt10deg",
                "river_max_straight_run_cells",
                "river_mean_tile_median_sinuosity",
                "land_fraction_above_1500_m",
                "land_fraction_above_2500_m",
                "rugged_land_fraction_slope_ge_10deg",
                "elevation_high_frequency_rms_m",
                "tectonic_microdetail_rms_m",
                "legacy_physical_erosion_rms_m",
                "procedural_detail_rms_m",
            )
        },
        "sampling": {
            key: sampling.get(key)
            for key in (
                "native_output_resolution",
                "native_output_equatorial_m_per_pixel",
                "terrain_finest_wavelength_m",
                "terrain_minimum_feature_half_wave_m",
                "minimum_feature_width_in_native_output_pixels",
            )
        },
    }


def inspect_height_tiff(
    path: Path,
    *,
    expected_width: int,
    expected_height: int,
    sample_stride: int,
    scan_range: bool,
) -> dict[str, Any]:
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError("height TIFF inspection requires tifffile") from exc

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        shape = tuple(int(v) for v in page.shape)
        dtype = np.dtype(page.dtype)
        samples = int(page.samplesperpixel)
        photometric = str(page.photometric.name).upper()
        result: dict[str, Any] = {
            "path": str(path),
            "shape": list(shape),
            "dtype": str(dtype),
            "samples_per_pixel": samples,
            "photometric": photometric,
            "is_bigtiff": bool(tif.is_bigtiff),
            "bytes": int(path.stat().st_size),
            "dimension_contract_ok": shape == (expected_height, expected_width),
            "single_channel_uint32_ok": (
                dtype == np.dtype(np.uint32)
                and samples == 1
                and photometric == "MINISBLACK"
            ),
        }
        if scan_range:
            data = page.asarray()
            result["encoded_min"] = int(np.min(data))
            result["encoded_max"] = int(np.max(data))
            result["uses_full_uint32_range"] = (
                result["encoded_min"] == 0
                and result["encoded_max"] == 2**32 - 1
            )
            stride = max(1, int(sample_stride))
            sampled = np.asarray(data[::stride, ::stride], dtype=np.float64)
            result["sampled_directional"] = _fourfold_gradient_anisotropy(sampled)
        return result


def inspect_river_png(
    path: Path,
    *,
    expected_width: int,
    expected_height: int,
    diagnostic_width: int,
) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("river PNG inspection requires Pillow") from exc

    with Image.open(path) as image:
        original_size = tuple(int(v) for v in image.size)
        mode = image.mode
        target_w = max(256, min(int(diagnostic_width), original_size[0]))
        target_h = max(128, int(round(target_w * original_size[1] / original_size[0])))
        gray = image.convert("L").resize((target_w, target_h), resample=Image.Resampling.BOX)
        values = np.asarray(gray, dtype=np.float64)

    try:
        from scipy import ndimage
        residual = values - ndimage.gaussian_filter(values, sigma=3.0, mode="nearest")
    except ImportError:
        residual = values - float(np.mean(values))

    return {
        "path": str(path),
        "size": list(original_size),
        "mode": mode,
        "bytes": int(path.stat().st_size),
        "dimension_contract_ok": original_size == (expected_width, expected_height),
        "diagnostic_resolution": [int(target_w), int(target_h)],
        "render_directional": _fourfold_gradient_anisotropy(residual),
        "grayscale_std": float(np.std(values)),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("terrain_audit", type=Path)
    p.add_argument("height_tiff", type=Path)
    p.add_argument("river_png", type=Path)
    p.add_argument("--width", type=int, default=16384)
    p.add_argument("--height", type=int, default=8192)
    p.add_argument("--sample-stride", type=int, default=16)
    p.add_argument("--river-diagnostic-width", type=int, default=2048)
    p.add_argument(
        "--no-height-range-scan",
        action="store_true",
        help="skip decoding TIFF pixels; metadata/channel checks still run",
    )
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--strict",
        action="store_true",
        help="return nonzero unless authoritative gates and exported-file contracts pass",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    audit = json.loads(args.terrain_audit.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "authoritative": audit_authoritative_report(audit),
        "height_tiff": inspect_height_tiff(
            args.height_tiff,
            expected_width=args.width,
            expected_height=args.height,
            sample_stride=args.sample_stride,
            scan_range=not args.no_height_range_scan,
        ),
        "river_png": inspect_river_png(
            args.river_png,
            expected_width=args.width,
            expected_height=args.height,
            diagnostic_width=args.river_diagnostic_width,
        ),
        "interpretation": {
            "authoritative_grid_and_river_metrics": (
                "tile-space generator audit; use these for pass/fail"
            ),
            "render_directional_metrics": (
                "projection/render diagnostics only; report rather than equate with physical anisotropy"
            ),
        },
    }
    raw = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(raw + "\n", encoding="utf-8")
    print(raw)

    if not args.strict:
        return 0
    authoritative = report["authoritative"]
    height = report["height_tiff"]
    river = report["river_png"]
    ok = (
        authoritative["all_checks_passed"]
        and not authoritative["missing_required_checks"]
        and not authoritative["failed_required_checks"]
        and height["dimension_contract_ok"]
        and height["single_channel_uint32_ok"]
        and height.get("uses_full_uint32_range", True)
        and river["dimension_contract_ok"]
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
