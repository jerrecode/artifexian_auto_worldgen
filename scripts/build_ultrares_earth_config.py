from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


def merge(base, overlay):
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=Path("config/ultra_detail.yaml"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2026090707)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise TypeError("base config must contain a mapping")

    cfg["seed"] = int(args.seed)
    cfg.setdefault("resolution", {}).update({"width": 2048, "height": 1024})

    astronomy = cfg.setdefault("astronomy", {})
    astronomy["greenhouse_model"] = "composition"
    astronomy["thermodynamics_backend"] = "auto"
    astronomy["surface_volatiles"] = {"H2O": 1.0}
    astronomy["surface_condensible"] = "H2O"
    # Preserve the Earth-like bulk atmosphere from ultra_detail while making the
    # inventory explicit for atmogen's composition-aware path.
    astronomy["atmosphere"] = {
        "N2": 0.7800,
        "O2": 0.2090,
        "Ar": 0.0093,
        "CO2": 0.0006,
        "H2O": 0.0011,
    }

    climate = cfg.setdefault("climate", {})
    climate["condensible_species"] = "H2O"
    climate["phase_coupled_evaporation"] = True
    cfg.setdefault("ocean", {})["fluid_species"] = "H2O"

    cfg["tectonics"] = merge(
        cfg.get("tectonics", {}),
        {
            # Active but still Earth-like: more resolved convergent/subplate
            # structure and stronger relief, without changing planet gravity.
            "plate_count": 20,
            "mountain_uplift_km": 6.2,
            "terrain_noise_km": 0.72,
            "mean_subplates_per_plate": 7.0,
            "min_subplates_per_plate": 4,
            "max_subplates_per_plate": 12,
            "boundary_warp_deg": 4.4,
            "boundary_detail_octaves": 9,
            "boundary_deformation_iterations": 4,
            "strain_boundary_warp_deg": 3.8,
            "geological_activity_mode": "active",
            "activity_strength": 1.12,
        },
    )
    cfg["terrain"] = merge(
        cfg.get("terrain", {}),
        {
            # Preserve broad geologic elevations while retaining sharper active
            # orogens for the subsequent kilometre-scale refinement.
            "erosion_m_per_myr": 4.0,
            "fractal_octaves": 9,
            "relief_detail_strength": 0.68,
            "fault_block_relief_km": 1.05,
            "rift_shoulder_uplift_km": 1.0,
        },
    )
    cfg["hydrology"] = merge(
        cfg.get("hydrology", {}),
        {
            "surface_evolution_iterations": 6,
            "sediment_routing_passes": 30,
            "river_meander_strength": 1.0,
            "lateral_erosion_fraction": 0.38,
            "meander_microrelief_m": 14.0,
        },
    )

    cfg["atmogen"] = merge(
        cfg.get("atmogen", {}),
        {
            "enabled": True,
            "fidelity": "REFERENCE",
            "chemistry_mode": "equilibrium",
            "vertical_layers": 64,
            "radiation_mode": "semi_gray_spectral_shortwave",
            "temperature_profile_mode": "auto",
            "representative_columns_enabled": True,
            "representative_column_count": 16,
            "representative_feedback_relaxation": 0.25,
            "composition_tolerance": 2.0e-7,
            "relaxation": 0.35,
            "max_iterations": 100,
            "allow_fidelity_fallback": True,
        },
    )
    cfg["procedural_erosion"] = merge(
        cfg.get("procedural_erosion", {}),
        {
            "enabled": True,
            "octaves": 6,
            "min_samples_per_wavelength": 4.0,
            "phase_chunk_rows": 64,
            "recouple_after_canonical_pass": True,
            "zero_mean_displacement": True,
        },
    )
    cfg.setdefault("output", {}).update(
        {
            "save_npz": True,
            "save_png": True,
            "save_json": True,
            "save_report": True,
            "compress_npz": True,
            "map_dpi": 180,
            "rgb_dpi": 200,
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(
        yaml.safe_dump(
            {
                "seed": cfg["seed"],
                "source_resolution": cfg["resolution"],
                "global_detail_profile": "ultra_detail",
                "atmogen": cfg["atmogen"],
                "procedural_erosion": cfg["procedural_erosion"],
                "terrain_output_contract": {
                    "reconstructed_base_resolution": [8192, 4096],
                    "native_terrain_map_resolution": [16384, 8192],
                    "requested_subsection_linear_multiplier": 3,
                    "expected_discrete_lod_multiplier": 4,
                    "minimum_resolved_wave_samples_native_tile": 4,
                    "minimum_feature_half_wave_pixels_native_map": 1,
                    "height_products": [
                        "single-channel grayscale uint16 PNG",
                        "single-channel min-is-black uint32 TIFF",
                    ],
                },
            },
            sort_keys=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
