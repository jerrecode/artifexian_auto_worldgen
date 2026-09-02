#!/usr/bin/env python3
from __future__ import annotations

"""Build reproducible maximum-complexity validation configs for reference worlds."""

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "config" / "maximal_realism_safe.yaml"
PROFILE_FILES = {
    "titan": ROOT / "config" / "titan_like.yaml",
    "super_earth": ROOT / "config" / "super_earth.yaml",
}

EARTH_OVERLAY: dict[str, Any] = {
    "astronomy": {
        "star_mass_solar": 1.0,
        "body_role": "planet",
        "planet_mass_earth": 1.0,
        "planet_density_g_cm3": 5.514,
        "semimajor_axis_au": 1.0,
        "eccentricity": 0.0167,
        "axial_tilt_deg": 23.44,
        "rotation_hours": 23.934,
        "albedo": 0.30,
        "greenhouse_model": "composition",
        "atmosphere_pressure_bar": 1.0,
        "atmosphere_top_pressure_bar": 1.0e-7,
        "atmosphere": {"N2": 0.78084, "O2": 0.20946, "Ar": 0.00934, "CO2": 0.00036},
        "surface_volatiles": {"H2O": 1.0},
        "surface_condensible": "H2O",
        "radiogenic_heat_flux_w_m2": 0.087,
    },
    "tectonics": {"geological_activity_mode": "active", "activity_strength": 1.0, "ice_geology_mode": "inactive"},
    "climate": {"condensible_species": "H2O", "precip_scale_mm_year": 1100.0, "phase_coupled_evaporation": True},
    "ocean": {"fluid_species": "H2O"},
}

ATMOGEN_REFERENCE = {
    "enabled": True,
    "fidelity": "REFERENCE",
    "chemistry_mode": "equilibrium",
    "vertical_layers": 64,
    "radiation_mode": "semi_gray_spectral_shortwave",
    "temperature_profile_mode": "auto",
    "representative_columns_enabled": True,
    "representative_column_count": 16,
    "representative_feedback_relaxation": 0.25,
    # High-pressure phase-coupled equilibrium columns are stiff and SLSQP introduces
    # ~1e-7 absolute mole-fraction iterate noise even when element closure, thermal
    # residual and energy closure are substantially tighter.  2e-7 is therefore a
    # numerical-noise-aware outer fixed-point tolerance, not a chemistry mass-balance
    # relaxation; the inner Gibbs solver retains its own strict closure checks.
    "composition_tolerance": 2.0e-7,
    "relaxation": 0.35,
    "max_iterations": 100,
    "allow_fidelity_fallback": True,
}


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _apply_profile(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge structural tuning but replace physical composition inventories exactly."""
    out = _merge(base, overlay)
    astronomy = overlay.get("astronomy", {})
    for key in ("atmosphere", "surface_volatiles"):
        if key in astronomy:
            out.setdefault("astronomy", {})[key] = copy.deepcopy(astronomy[key])
    return out


def build(profile: str, seed: int) -> dict[str, Any]:
    cfg = _load(BASE)
    overlay = EARTH_OVERLAY if profile == "earth" else _load(PROFILE_FILES[profile])
    cfg = _apply_profile(cfg, overlay)
    cfg["seed"] = int(seed)
    cfg["atmogen"] = _merge(cfg.get("atmogen", {}), ATMOGEN_REFERENCE)
    cfg.setdefault("output", {})["save_npz"] = True
    cfg["output"]["save_png"] = True
    cfg["output"]["save_json"] = True
    cfg["output"]["save_report"] = True
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=("earth", "titan", "super_earth"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    cfg = build(args.profile, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(yaml.safe_dump({"profile": args.profile, "seed": args.seed, "resolution": cfg["resolution"], "astronomy": cfg["astronomy"], "atmogen": cfg["atmogen"]}, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
