from __future__ import annotations

"""Reduced-order multicomponent ocean/sea state for exotic planetary liquids."""

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from .grid import SphereGrid, normalize01, smooth_periodic
from .planetary_chemistry import CHEMICALS


@dataclass(slots=True)
class ExoticOceanResult:
    composition_mass_fraction: dict[str, float]
    bulk_density_kg_m3: float
    dynamic_viscosity_mpa_s: float
    surface_tension_mn_m: float
    effective_freezing_temperature_k: float
    ocean_class: str
    mixed_layer_depth_m: np.ndarray
    density_anomaly_kg_m3: np.ndarray
    stratification_index: np.ndarray
    sea_ice_fraction: np.ndarray
    brine_or_solute_concentration_index: np.ndarray
    clathrate_stability_index: np.ndarray
    hydrothermal_exchange_index: np.ndarray
    metadata: dict

    def to_dict(self) -> dict:
        return {
            **self.metadata,
            "composition_mass_fraction": dict(self.composition_mass_fraction),
            "bulk_density_kg_m3": float(self.bulk_density_kg_m3),
            "dynamic_viscosity_mpa_s": float(self.dynamic_viscosity_mpa_s),
            "surface_tension_mn_m": float(self.surface_tension_mn_m),
            "effective_freezing_temperature_k": float(self.effective_freezing_temperature_k),
            "ocean_class": self.ocean_class,
        }


def _mixture_composition(surface_liquids: Any) -> dict[str, float]:
    masses: dict[str, float] = {}
    for key, part in getattr(surface_liquids, "partitions", {}).items():
        mass = max(float(getattr(part, "liquid_mass_kg", 0.0)), 0.0)
        if mass > 0:
            masses[str(key)] = mass
    total = sum(masses.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in masses.items()}


def _property(key: str, attr: str, fallback: float) -> float:
    sp = CHEMICALS.get(key)
    if sp is None:
        return fallback
    value = getattr(sp, attr)
    return fallback if value is None or not math.isfinite(float(value)) else float(value)


def _mixture_density(comp: dict[str, float], surface_liquids: Any) -> float:
    # Ideal volume-additive mixture density; surface-liquid partition density wins
    # because it may come from CoolProp at the actual modeled T/P.
    denom = 0.0
    for key, w in comp.items():
        part = getattr(surface_liquids, "partitions", {}).get(key)
        rho = float(getattr(part, "liquid_density_kg_m3", 0.0)) if part is not None else 0.0
        if rho <= 0:
            rho = _property(key, "liquid_density_kg_m3", 1000.0)
        denom += w / max(rho, 1e-12)
    return 1000.0 if denom <= 0 else 1.0 / denom


def _mixture_log_property(comp: dict[str, float], attr: str, fallback: float) -> float:
    if not comp:
        return fallback
    return float(math.exp(sum(w * math.log(max(_property(k, attr, fallback), 1e-9)) for k, w in comp.items())))


def _ocean_class(comp: dict[str, float]) -> str:
    if not comp:
        return "dry"
    h2o = comp.get("H2O", 0.0)
    hydrocarbons = comp.get("CH4", 0.0) + comp.get("C2H6", 0.0) + comp.get("C3H8", 0.0)
    nh3 = comp.get("NH3", 0.0)
    if h2o >= 0.55 and nh3 >= 0.03:
        return "ammonia-water cryo-ocean"
    if h2o >= 0.55:
        return "aqueous ocean"
    if hydrocarbons >= 0.55:
        return "hydrocarbon sea"
    if comp.get("SO2", 0.0) >= 0.45:
        return "sulfur-dioxide sea"
    if comp.get("CO2", 0.0) >= 0.45:
        return "carbon-dioxide fluid reservoir"
    return "mixed exotic ocean"


def _effective_freezing_k(comp: dict[str, float]) -> float:
    if not comp:
        return 0.0
    # Dominant pure-component melting/triple point, then bounded eutectic-like
    # depressions for common cryomagma antifreezes. These are screening relations.
    base = sum(
        w * (_property(k, "triple_or_melting_k", 273.15) if CHEMICALS.get(k) else 273.15)
        for k, w in comp.items()
    )
    if comp.get("H2O", 0.0) > 0.2:
        nh3 = comp.get("NH3", 0.0)
        methanol = comp.get("CH3OH", 0.0)
        # NH3-H2O can remain liquid close to ~176 K near its eutectic composition;
        # use a smooth bounded approximation rather than claiming a full phase diagram.
        nh3_dep = 97.0 * min(1.0, nh3 / 0.32) * min(1.0, comp.get("H2O", 0.0) / 0.68)
        meth_dep = 70.0 * min(1.0, methanol / 0.45) * min(1.0, comp.get("H2O", 0.0) / 0.55)
        base -= max(nh3_dep, meth_dep)
    return float(max(base, 20.0))


def build_exotic_ocean(
    grid: SphereGrid,
    astronomy: Any,
    climate: Any,
    ocean: Any,
    surface_liquids: Any,
    volatile_cycle: Any | None = None,
) -> ExoticOceanResult:
    comp = _mixture_composition(surface_liquids)
    rho = _mixture_density(comp, surface_liquids) if comp else 0.0
    viscosity = _mixture_log_property(comp, "viscosity_mpa_s", 1.0) if comp else 0.0
    tension = sum(w * _property(k, "surface_tension_mn_m", 35.0) for k, w in comp.items()) if comp else 0.0
    freezing_k = _effective_freezing_k(comp)
    ocean_class = _ocean_class(comp)

    wet = np.asarray(getattr(surface_liquids, "liquid_mask", np.zeros(grid.shape, bool)), dtype=bool)
    depth = np.asarray(getattr(surface_liquids, "liquid_depth_m", np.zeros(grid.shape)), dtype=np.float64)
    temp_k = np.asarray(climate.annual_temperature_c, dtype=np.float64) + 273.15
    seasonal_amp = np.ptp(np.asarray(climate.temperature_c, dtype=np.float64), axis=0)
    currents = np.asarray(getattr(ocean, "current_speed", np.zeros(grid.shape)), dtype=np.float64)
    winds = np.hypot(
        np.asarray(climate.wind_u, dtype=np.float64).mean(axis=0),
        np.asarray(climate.wind_v, dtype=np.float64).mean(axis=0),
    )

    if comp:
        thermal_expansion = 2.0e-4
        if ocean_class == "hydrocarbon sea":
            thermal_expansion = 1.2e-3
        elif "ammonia" in ocean_class:
            thermal_expansion = 4.0e-4
        tref = float(np.average(temp_k[wet], weights=grid.cell_area_weights[wet])) if np.any(wet) else float(np.average(temp_k, weights=grid.cell_area_weights))
        density_anom = -rho * thermal_expansion * (temp_k - tref) * wet
    else:
        density_anom = np.zeros(grid.shape, dtype=np.float64)

    # Mixed-layer proxy: stronger winds/currents and lower viscosity deepen mixing;
    # shallow seas are capped by actual water-column depth.
    forcing = normalize01(0.62 * winds + 0.38 * currents, robust=True)
    viscosity_penalty = math.sqrt(max(viscosity, 0.02)) if viscosity > 0 else 1.0
    mixed = (8.0 + 95.0 * forcing / viscosity_penalty) * wet
    mixed = np.minimum(mixed, np.maximum(depth, 0.0))

    # Seasonal thermal contrast and weak mechanical forcing increase stratification;
    # current/wind mixing erodes it.
    strat = normalize01(seasonal_amp * wet, robust=True) * (1.0 - 0.55 * forcing) * wet
    if freezing_k > 0:
        freeze_margin = freezing_k - temp_k
        sea_ice = np.clip(0.5 + freeze_margin / 8.0, 0.0, 1.0) * wet
    else:
        sea_ice = np.zeros(grid.shape, dtype=np.float64)

    # Freezing excludes many dissolved/low-melting components from the solid phase,
    # concentrating the residual liquid.  This is a brine/solute concentration index,
    # not a salinity in PSU.
    solute_component = 1.0 - comp.get("H2O", 0.0) if comp else 0.0
    solute = normalize01(sea_ice * (0.15 + solute_component) + 0.25 * strat, robust=True) * wet

    # CH4-water clathrate is favored at cold temperatures and pressure, represented by
    # liquid depth as a hydrostatic-pressure proxy. Other guest species can contribute
    # weakly but are not assigned detailed hydrate phase boundaries here.
    guest = comp.get("CH4", 0.0) + 0.35 * comp.get("C2H6", 0.0) + 0.15 * comp.get("CO2", 0.0)
    water = comp.get("H2O", 0.0)
    clath = normalize01(
        guest * water * np.clip((285.0 - temp_k) / 90.0, 0.0, 1.0) * np.log1p(np.maximum(depth, 0.0)),
        robust=True,
    ) * wet

    interior = getattr(astronomy, "interior", {}) or {}
    heat = max(float(interior.get("total_internal_heat_flux_w_m2_approx", 0.0)), 0.0)
    hydrothermal = normalize01((heat + 1e-5) * np.sqrt(np.maximum(depth, 0.0)) * (0.4 + 0.6 * strat), robust=True) * wet
    hydrothermal = smooth_periodic(hydrothermal, (1.0, 1.4)) * wet

    metadata = {
        "model": "volume-additive liquid mixture + reduced-order density/mixing/freezing/stratification state",
        "ocean_class": ocean_class,
        "mean_mixed_layer_depth_m": float(np.average(mixed[wet], weights=grid.cell_area_weights[wet])) if np.any(wet) else 0.0,
        "mean_sea_ice_fraction": float(np.average(sea_ice[wet], weights=grid.cell_area_weights[wet])) if np.any(wet) else 0.0,
        "mixture_model": "ideal volume density; logarithmic viscosity mixing; linear surface-tension mixing",
        "freezing_model": "pure-component weighted screening temperature plus bounded NH3/CH3OH-H2O eutectic depression",
        "limitations": "No EOS mixture fugacity, salinity chemistry, vertical primitive-equation ocean, double diffusion, or explicit sea-ice dynamics.",
    }
    return ExoticOceanResult(
        composition_mass_fraction=comp,
        bulk_density_kg_m3=float(rho),
        dynamic_viscosity_mpa_s=float(viscosity),
        surface_tension_mn_m=float(tension),
        effective_freezing_temperature_k=float(freezing_k),
        ocean_class=ocean_class,
        mixed_layer_depth_m=mixed.astype(np.float32),
        density_anomaly_kg_m3=np.asarray(density_anom, dtype=np.float32),
        stratification_index=np.asarray(strat, dtype=np.float32),
        sea_ice_fraction=np.asarray(sea_ice, dtype=np.float32),
        brine_or_solute_concentration_index=np.asarray(solute, dtype=np.float32),
        clathrate_stability_index=np.asarray(clath, dtype=np.float32),
        hydrothermal_exchange_index=np.asarray(hydrothermal, dtype=np.float32),
        metadata=metadata,
    )


__all__ = ["ExoticOceanResult", "build_exotic_ocean"]
