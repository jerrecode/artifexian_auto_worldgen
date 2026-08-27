from __future__ import annotations

"""Cryogeology and volatile-ice resurfacing for cold planetary bodies."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .grid import SphereGrid, normalize01, smooth_periodic


@dataclass(slots=True)
class CryogeologyResult:
    ice_shell_thickness_km: np.ndarray
    basal_melt_fraction: np.ndarray
    brittle_fracture_index: np.ndarray
    diapirism_index: np.ndarray
    cryovolcanism_index: np.ndarray
    chaos_terrain_index: np.ndarray
    plume_venting_index: np.ndarray
    sublimation_erosion_index: np.ndarray
    volatile_frost_deposition_index: np.ndarray
    clathrate_destabilization_index: np.ndarray
    cryomagma_composition: dict[str, float]
    metadata: dict

    def to_dict(self) -> dict:
        return {
            **self.metadata,
            "cryomagma_composition": dict(self.cryomagma_composition),
            "mean_ice_shell_thickness_km": float(np.mean(self.ice_shell_thickness_km)),
            "max_cryovolcanism_index": float(np.max(self.cryovolcanism_index)),
            "max_plume_venting_index": float(np.max(self.plume_venting_index)),
        }


def _cryomagma_composition(exotic_ocean: Any | None) -> dict[str, float]:
    if exotic_ocean is None:
        return {}
    comp = dict(getattr(exotic_ocean, "composition_mass_fraction", {}) or {})
    candidates = {k: v for k, v in comp.items() if k in {"H2O", "NH3", "CH3OH", "CH4", "C2H6", "CO2"} and v > 0}
    total = sum(candidates.values())
    return {} if total <= 0 else {k: float(v / total) for k, v in candidates.items()}


def build_cryogeology(
    grid: SphereGrid,
    astronomy: Any,
    terrain: Any,
    climate: Any,
    tectonics: Any,
    tectonics_config: Any,
    geodynamics: Any,
    exotic_ocean: Any | None = None,
    volatile_cycle: Any | None = None,
) -> CryogeologyResult:
    shape = grid.shape
    temp_k = np.asarray(climate.annual_temperature_c, dtype=np.float64) + 273.15
    mean_t = float(np.sum(temp_k * grid.cell_area_weights))
    interior = getattr(astronomy, "interior", {}) or {}
    heat = max(float(interior.get("total_internal_heat_flux_w_m2_approx", 0.0)), 0.0)
    tidal = max(float(interior.get("tidal_heating_flux_w_m2", 0.0)), 0.0)
    configured_shell = max(float(getattr(tectonics_config, "ice_shell_thickness_km", 0.0)), 0.0)
    cryo_activity = float(getattr(geodynamics, "cryotectonic_activity_index", 0.0))
    shell_mobility = float(getattr(geodynamics, "estimated_ice_shell_mobility", 0.0))

    comp = _cryomagma_composition(exotic_ocean)
    water = comp.get("H2O", 0.0)
    antifreeze = comp.get("NH3", 0.0) + 0.7 * comp.get("CH3OH", 0.0)
    hydrocarbon = comp.get("CH4", 0.0) + comp.get("C2H6", 0.0)
    ocean_present = bool(comp)

    # If an ice shell was not explicitly configured but a cold water-rich ocean is
    # present, infer a broad conductive shell scale from heat flux.  The relation is
    # intentionally bounded and diagnostic rather than a Stefan solution.
    if configured_shell > 0:
        base_shell = configured_shell
        shell_source = "configured"
    elif ocean_present and water > 0.25 and mean_t < 260.0:
        base_shell = float(np.clip(18.0 * (0.06 / max(heat, 0.004)) ** 0.55, 3.0, 140.0))
        shell_source = "inferred_from_heat_flux_and_cold_water_rich_surface"
    else:
        base_shell = 0.0
        shell_source = "none"

    # Tectonic stress and internal heating thin the shell; cold poles/high terrain
    # thicken it.  Existing tectonic fields are used as spatial priors even on worlds
    # whose silicate lithosphere is mostly inactive because tidal/ice stresses can
    # inherit long-wavelength structural heterogeneity.
    stress = normalize01(
        0.40 * np.asarray(getattr(tectonics, "stress_field", np.zeros(shape)), float)
        + 0.30 * np.asarray(getattr(tectonics, "strain_field", np.zeros(shape)), float)
        + 0.20 * np.asarray(getattr(tectonics, "hotspot_strength", np.zeros(shape)), float)
        + 0.10 * np.asarray(getattr(tectonics, "lip_strength", np.zeros(shape)), float),
        robust=True,
    )
    cold = normalize01(np.maximum(0.0, 260.0 - temp_k), robust=True)
    topographic = normalize01(np.maximum(np.asarray(terrain.elevation_km, float), 0.0), robust=True)
    if base_shell > 0:
        shell = base_shell * (1.0 + 0.28 * cold + 0.10 * topographic - 0.42 * cryo_activity * stress)
        shell *= 1.0 / (1.0 + 0.9 * antifreeze)
        shell = np.clip(shell, max(0.5, 0.12 * base_shell), 3.0 * base_shell)
    else:
        shell = np.zeros(shape, dtype=np.float64)

    # Basal melting rises with heat, shell mobility and antifreeze abundance and is
    # focused where the shell is anomalously thin.
    if base_shell > 0:
        thin = np.clip(base_shell / np.maximum(shell, 0.25) - 0.55, 0.0, 2.5)
        heat_drive = np.clip((heat / 0.06) ** 0.55, 0.0, 2.0)
        basal = np.clip((0.20 + 0.80 * thin) * heat_drive * (0.55 + 0.45 * shell_mobility) * (1.0 + 1.8 * antifreeze), 0.0, 1.0)
    else:
        basal = np.zeros(shape, dtype=np.float64)

    # Brittle fractures require a shell, stress and enough thermal contrast for a
    # brittle lid. Tidal forcing adds a planet-wide oscillatory stress contribution.
    tidal_drive = np.clip(tidal / 0.08, 0.0, 2.0)
    fracture = normalize01(stress * (0.45 + 0.55 * cold) + 0.30 * tidal_drive, robust=True)
    fracture *= (shell > 0)
    fracture = smooth_periodic(fracture, (0.8, 1.15))

    # Warm buoyant ice/cryomagma diapirs are favored by basal melt and a shell thick
    # enough to convect, whereas direct venting is favored by thin highly fractured
    # shells. This gives Europa-style chaos and Enceladus-style vents different priors.
    thick_enough = np.clip(shell / max(base_shell, 1.0), 0.0, 2.0) if base_shell > 0 else np.zeros(shape)
    diapir = normalize01(basal * (0.35 + 0.65 * thick_enough) * (0.5 + 0.5 * cryo_activity), robust=True)
    vent = normalize01(basal * fracture / np.maximum(shell / max(base_shell, 1.0), 0.12), robust=True) if base_shell > 0 else np.zeros(shape)
    vent *= np.clip(0.45 + 0.55 * tidal_drive + 0.25 * antifreeze, 0.0, 1.5)
    vent = np.clip(vent, 0.0, 1.0)
    cryovolcanism = normalize01(0.45 * vent + 0.35 * diapir + 0.20 * basal, robust=True) * (0.25 + 0.75 * cryo_activity)
    chaos = normalize01(diapir * fracture * (0.45 + 0.55 * basal), robust=True)

    # Surface volatile loss/deposition couples directly to the multicomponent cycle.
    sublimation = normalize01(np.maximum(temp_k - 120.0, 0.0) * (0.35 + 0.65 * (1.0 - cold)), robust=True)
    frost = np.zeros(shape, dtype=np.float64)
    if volatile_cycle is not None:
        for cyc in getattr(volatile_cycle, "species", {}).values():
            sublimation = np.maximum(sublimation, np.asarray(cyc.sublimation_potential, float))
            frost = np.maximum(frost, np.asarray(cyc.frost_deposition_index, float))
    sublimation *= (shell > 0) | (water + hydrocarbon > 0)

    clath = np.zeros(shape, dtype=np.float64)
    if exotic_ocean is not None:
        clath_stable = np.asarray(getattr(exotic_ocean, "clathrate_stability_index", np.zeros(shape)), float)
        # Destabilization is strongest where a stable reservoir is heated/fractured.
        clath = normalize01(clath_stable * (0.35 * basal + 0.65 * fracture), robust=True)

    metadata = {
        "model": "reduced-order conductive/convective ice shell + fracture/diapir/vent cryogeology",
        "shell_source": shell_source,
        "configured_or_inferred_base_shell_km": float(base_shell),
        "mean_surface_temperature_k": mean_t,
        "internal_heat_flux_w_m2": float(heat),
        "tidal_heating_flux_w_m2": float(tidal),
        "antifreeze_mass_fraction_proxy": float(antifreeze),
        "hydrocarbon_cryofluid_fraction_proxy": float(hydrocarbon),
        "cryogenic_regime": str(getattr(geodynamics, "cryogenic_regime", "unknown")),
        "limitations": "No viscoelastic tidal FEM, grain-size-dependent ice rheology, two-phase porous flow, explicit ocean pressure, or fracture propagation mechanics.",
    }
    return CryogeologyResult(
        ice_shell_thickness_km=np.asarray(shell, dtype=np.float32),
        basal_melt_fraction=np.asarray(basal, dtype=np.float32),
        brittle_fracture_index=np.asarray(fracture, dtype=np.float32),
        diapirism_index=np.asarray(diapir, dtype=np.float32),
        cryovolcanism_index=np.asarray(cryovolcanism, dtype=np.float32),
        chaos_terrain_index=np.asarray(chaos, dtype=np.float32),
        plume_venting_index=np.asarray(vent, dtype=np.float32),
        sublimation_erosion_index=np.asarray(sublimation, dtype=np.float32),
        volatile_frost_deposition_index=np.asarray(frost, dtype=np.float32),
        clathrate_destabilization_index=np.asarray(clath, dtype=np.float32),
        cryomagma_composition=comp,
        metadata=metadata,
    )


__all__ = ["CryogeologyResult", "build_cryogeology"]
