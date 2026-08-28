#!/usr/bin/env python3
from __future__ import annotations

"""Audit composition-aware liquid/atmosphere rendering from saved public artifacts."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
    x = np.asarray(a, dtype=np.float64)[mask]
    y = np.asarray(b, dtype=np.float64)[mask]
    good = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(good) < 30 or np.std(x[good]) < 1e-12 or np.std(y[good]) < 1e-12:
        return None
    return float(np.corrcoef(x[good], y[good])[0, 1])


def _rgb_mean(rgb: np.ndarray, mask: np.ndarray) -> list[float] | None:
    if not np.any(mask):
        return None
    a = np.asarray(rgb, dtype=np.float64)
    if a.dtype == np.uint8 or float(np.nanmax(a)) > 1.5:
        a = a / 255.0
    return [float(x) for x in np.mean(a[mask], axis=0)]


def audit(root: Path) -> dict[str, Any]:
    world_arrays = root / "world_arrays.npz"
    if not world_arrays.exists():
        raise FileNotFoundError(world_arrays)
    with np.load(world_arrays, allow_pickle=False) as z:
        required = (
            "true_color_rgb",
            "true_color_with_clouds_rgb",
            "surface_liquid_true_color_rgb",
            "atmospheric_haze_optical_depth",
            "ground_liquid_humidity_index",
            "liquid_condensate_input_mm_year",
            "soil_liquid_storage_mm",
        )
        missing = [key for key in required if key not in z.files]
        if missing:
            raise ValueError(f"missing composition-aware saved optical fields: {missing}")
        clear = np.asarray(z["true_color_rgb"])
        toa = np.asarray(z["true_color_with_clouds_rgb"])
        liquid_rgb = np.asarray(z["surface_liquid_true_color_rgb"])
        haze = np.asarray(z["atmospheric_haze_optical_depth"], dtype=np.float64)
        humidity = np.asarray(z["ground_liquid_humidity_index"], dtype=np.float64)
        liquid_precip = np.asarray(z["liquid_condensate_input_mm_year"], dtype=np.float64)
        soil_storage = np.asarray(z["soil_liquid_storage_mm"], dtype=np.float64)

    liquid_path = root / "surface_liquids.npz"
    if not liquid_path.exists():
        raise FileNotFoundError(liquid_path)
    with np.load(liquid_path, allow_pickle=False) as z:
        wet = np.asarray(z["liquid_mask"], dtype=bool)
        depth = np.asarray(z["liquid_depth_m"], dtype=np.float64)

    if humidity.shape != wet.shape or clear.shape != (*wet.shape, 3):
        raise ValueError("optical/hydrology fields do not share the canonical spatial shape")
    land = ~wet
    world = _load_json(root / "world.json")
    pa = world.get("planetary_appearance", {}) if isinstance(world, dict) else {}
    liquid_meta = pa.get("surface_liquid_optics", {}) if isinstance(pa, dict) else {}
    atmosphere_meta = pa.get("atmosphere_visible_optics", {}) if isinstance(pa, dict) else {}
    fractions = liquid_meta.get("composition_volume_fraction", {}) if isinstance(liquid_meta, dict) else {}
    water_fraction = float(fractions.get("H2O", 0.0)) if isinstance(fractions, dict) else 0.0

    finite = {
        "haze": bool(np.all(np.isfinite(haze))),
        "ground_humidity": bool(np.all(np.isfinite(humidity))),
        "liquid_condensate_input": bool(np.all(np.isfinite(liquid_precip))),
        "soil_liquid_storage": bool(np.all(np.isfinite(soil_storage))),
    }
    humidity_bounds_ok = bool(float(np.min(humidity)) >= -1e-7 and float(np.max(humidity)) <= 1.0 + 1e-7)
    wet_rgb = _rgb_mean(liquid_rgb, wet)
    clear_wet_rgb = _rgb_mean(clear, wet)
    toa_mean = _rgb_mean(toa, np.ones(wet.shape, dtype=bool))
    clear_mean = _rgb_mean(clear, np.ones(wet.shape, dtype=bool))
    atmospheric_change = float(np.mean(np.abs(toa.astype(np.float64) - clear.astype(np.float64))))
    humidity_precip_corr = _corr(humidity, liquid_precip, land)
    humidity_storage_corr = _corr(humidity, soil_storage, land)

    high_low_precip_humidity_delta = None
    lp = liquid_precip[land]
    gh = humidity[land]
    good = np.isfinite(lp) & np.isfinite(gh)
    if np.count_nonzero(good) >= 100 and np.ptp(lp[good]) > 1e-6:
        q25, q75 = np.percentile(lp[good], [25, 75])
        low = good & (lp <= q25)
        high = good & (lp >= q75)
        if np.any(low) and np.any(high):
            high_low_precip_humidity_delta = float(np.mean(gh[high]) - np.mean(gh[low]))

    hydrocarbon_or_other_nonwater = bool(fractions) and water_fraction < 0.2
    nonwater_not_terrestrial_blue = True
    if hydrocarbon_or_other_nonwater and wet_rgb is not None:
        # Non-water liquids can reflect a blue sky, but the bulk column itself should
        # not have the extreme blue-minus-red contrast of the legacy H2O ocean paint.
        nonwater_not_terrestrial_blue = bool((wet_rgb[2] - wet_rgb[0]) < 0.32)

    invariants = {
        "all_optical_scalar_fields_finite": bool(all(finite.values())),
        "ground_liquid_humidity_bounded_0_1": humidity_bounds_ok,
        "surface_liquid_rgb_present_when_liquid_exists": bool((not np.any(wet)) or (wet_rgb is not None and max(wet_rgb) > 0.0)),
        "top_of_atmosphere_differs_from_clear_surface": atmospheric_change > 0.5,
        "nonwater_liquid_not_legacy_blue_water": nonwater_not_terrestrial_blue,
        "liquid_depth_nonnegative": bool(float(np.min(depth)) >= -1e-7),
    }

    result = {
        "schema_version": 1,
        "all_invariants_passed": bool(all(invariants.values())),
        "invariants": invariants,
        "surface_liquid": {
            "composition_volume_fraction": fractions,
            "wet_cell_count": int(np.count_nonzero(wet)),
            "wet_cell_fraction": float(np.mean(wet)),
            "max_depth_m": float(np.max(depth)) if depth.size else 0.0,
            "mean_rendered_rgb": wet_rgb,
            "mean_clear_map_rgb_on_liquid": clear_wet_rgb,
            "water_volume_fraction": water_fraction,
        },
        "atmosphere": {
            "mean_clear_surface_rgb": clear_mean,
            "mean_top_of_atmosphere_rgb": toa_mean,
            "mean_absolute_rgb_change_uint8": atmospheric_change,
            "mean_haze_optical_depth": float(np.mean(haze)),
            "max_haze_optical_depth": float(np.max(haze)),
            "gas_transmittance_rgb": atmosphere_meta.get("gas_transmittance_rgb") if isinstance(atmosphere_meta, dict) else None,
            "rayleigh_tau_rgb": atmosphere_meta.get("rayleigh_tau_rgb") if isinstance(atmosphere_meta, dict) else None,
            "molecular_absorption_tau_rgb": atmosphere_meta.get("molecular_absorption_tau_rgb") if isinstance(atmosphere_meta, dict) else None,
            "haze_rgb": atmosphere_meta.get("haze_rgb") if isinstance(atmosphere_meta, dict) else None,
            "cloud_rgb": atmosphere_meta.get("cloud_rgb") if isinstance(atmosphere_meta, dict) else None,
        },
        "ground_liquid": {
            "mean_land_humidity_index": float(np.mean(humidity[land])) if np.any(land) else 0.0,
            "max_humidity_index": float(np.max(humidity)) if humidity.size else 0.0,
            "mean_land_liquid_condensate_input_mm_year": float(np.mean(liquid_precip[land])) if np.any(land) else 0.0,
            "mean_land_soil_liquid_storage_mm": float(np.mean(soil_storage[land])) if np.any(land) else 0.0,
            "humidity_vs_liquid_precipitation_correlation": humidity_precip_corr,
            "humidity_vs_soil_liquid_storage_correlation": humidity_storage_corr,
            "upper_minus_lower_precip_quartile_humidity": high_low_precip_humidity_delta,
        },
        "finite_fields": finite,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("world_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    result = audit(args.world_dir)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 1 if args.enforce and not result["all_invariants_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
