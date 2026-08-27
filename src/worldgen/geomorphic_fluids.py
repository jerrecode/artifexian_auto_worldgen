from __future__ import annotations

"""Fluid-property-dependent geomorphic scaling for non-water rivers and seas."""

from dataclasses import asdict, dataclass
import copy
import math
from typing import Any

import numpy as np

from .grid import SphereGrid, normalize01, smooth_periodic
from .planetary_chemistry import CHEMICALS


@dataclass(slots=True)
class GeomorphicFluidParameters:
    active_fluid: str
    density_kg_m3: float
    viscosity_mpa_s: float
    surface_tension_mn_m: float
    gravity_m_s2: float
    stream_power_multiplier: float
    runoff_multiplier: float
    sediment_capacity_multiplier: float
    deposition_multiplier: float
    lateral_erosion_multiplier: float
    delta_retention_multiplier: float
    hillslope_multiplier: float
    evaporation_loss_multiplier: float
    substrate_erodibility_multiplier: float
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class ExoticGeomorphologyResult:
    fluid_erosion_potential: np.ndarray
    fluid_deposition_potential: np.ndarray
    evaporite_deposition_index: np.ndarray
    organic_sediment_deposition_index: np.ndarray
    sublimation_landform_index: np.ndarray
    cryogenic_mass_wasting_index: np.ndarray
    metadata: dict

    def to_dict(self) -> dict:
        return dict(self.metadata)


def _screening_fluid_properties(species: str) -> tuple[float, float, float]:
    """Return screening-grade liquid density, viscosity and surface tension.

    This fallback is used when a volatile is hydrologically active but the current
    global surface-liquid equilibrium contains no mobile sea.  That is physically
    distinct from saying the world is a water world: episodic rain/runoff can still
    be methane-, ethane-, CO2-, etc. dominated even when standing liquid is absent.
    """
    sp = CHEMICALS.get(species)
    if sp is None:
        return 997.0, 1.0, 72.0
    rho = 997.0 if sp.liquid_density_kg_m3 is None else float(sp.liquid_density_kg_m3)
    mu = 1.0 if sp.viscosity_mpa_s is None else float(sp.viscosity_mpa_s)
    sigma = 72.0 if sp.surface_tension_mn_m is None else float(sp.surface_tension_mn_m)
    return max(rho, 30.0), max(mu, 0.01), max(sigma, 1.0)


def _volatile_cycle_reference_fluid(volatile_cycle: Any | None) -> str | None:
    if volatile_cycle is None:
        return None
    cycles = getattr(volatile_cycle, "species", {}) or {}
    if cycles:
        def cycle_score(item):
            key, cyc = item
            precip = np.asarray(getattr(cyc, "annual_precipitation_mm_equivalent", 0.0), dtype=float)
            mean_precip = float(np.mean(precip)) if precip.size else 0.0
            frac = max(float(getattr(cyc, "atmospheric_fraction", 0.0)), 0.0)
            cond = max(float(getattr(cyc, "condensation_index", 0.0)), 0.0)
            return mean_precip + 100.0 * frac * cond
        return max(cycles.items(), key=cycle_score)[0]

    condensates = getattr(volatile_cycle, "condensates", {}) or {}
    candidates = [
        (key, cand)
        for key, cand in condensates.items()
        if bool(getattr(cand, "precipitation_capable", False))
        and not bool(getattr(cand, "aerosol_only", False))
    ]
    if candidates:
        return max(
            candidates,
            key=lambda kv: max(float(getattr(kv[1], "atmospheric_fraction", 0.0)), 0.0)
            * max(float(getattr(kv[1], "condensation_index", 0.0)), 0.0),
        )[0]
    return None


def build_geomorphic_fluid_parameters(
    astronomy: Any,
    exotic_ocean: Any | None,
    volatile_cycle: Any | None = None,
    cryogeology: Any | None = None,
) -> GeomorphicFluidParameters:
    planet = getattr(astronomy, "planet", {}) or {}
    gravity = max(float(planet.get("surface_gravity_m_s2", 9.80665)), 0.05)
    comp = {} if exotic_ocean is None else dict(getattr(exotic_ocean, "composition_mass_fraction", {}) or {})
    if comp:
        active = max(comp, key=comp.get)
        rho = max(float(exotic_ocean.bulk_density_kg_m3), 30.0)
        mu = max(float(exotic_ocean.dynamic_viscosity_mpa_s), 0.01)
        sigma = max(float(exotic_ocean.surface_tension_mn_m), 1.0)
        selection_source = "mobile_surface_liquid_mixture"
    else:
        active = _volatile_cycle_reference_fluid(volatile_cycle) or "H2O"
        rho, mu, sigma = _screening_fluid_properties(active)
        selection_source = "active_precipitating_volatile" if active != "H2O" else "water_reference_fallback"

    # Bed shear scales with rho*g and turbulent competence rises as viscosity falls.
    # Exponents are intentionally sub-linear to avoid suppressing Titan-like fluvial
    # geomorphology merely because a simulation iteration has no explicit duration.
    density_term = (rho / 997.0) ** 0.55
    gravity_term = (gravity / 9.80665) ** 0.45
    viscosity_term = (1.0 / mu) ** 0.16
    tension_term = (72.0 / sigma) ** 0.10
    stream = float(np.clip(density_term * gravity_term * viscosity_term * tension_term, 0.08, 3.0))

    # Low-viscosity hydrocarbons infiltrate/evaporate readily but can still generate
    # intense episodic runoff. Ammonia-water behaves closer to water while very
    # viscous fluids reduce effective channel transport.
    runoff = float(np.clip((1.0 / mu) ** 0.10 * (rho / 997.0) ** 0.15, 0.45, 1.7))
    evap = float(np.clip((997.0 / rho) ** 0.25 * (1.0 / mu) ** 0.08, 0.55, 2.4))
    capacity = float(np.clip(stream * (1.0 / mu) ** 0.10, 0.08, 3.5))
    deposition = float(np.clip(1.0 / math.sqrt(max(capacity, 0.08)), 0.45, 2.3))
    lateral = float(np.clip(stream ** 0.65 * (72.0 / sigma) ** 0.12, 0.15, 2.2))
    delta_retention = float(np.clip(deposition * (rho / 997.0) ** 0.10, 0.45, 1.8))

    substrate = 1.0
    if cryogeology is not None:
        cryo = float(np.mean(np.asarray(cryogeology.cryovolcanism_index, dtype=float)))
        frost = float(np.mean(np.asarray(cryogeology.volatile_frost_deposition_index, dtype=float)))
        # Fresh fractured ice and volatile-rich regolith are more readily reworked
        # than competent silicate bedrock in this reduced-order parameterization.
        substrate *= 1.0 + 0.55 * cryo + 0.35 * frost
    if volatile_cycle is not None:
        tholin = getattr(volatile_cycle, "photochemical_deposition_by_species", {}).get("THOLIN")
        if tholin is not None:
            substrate *= 1.0 + 0.35 * float(np.mean(np.asarray(tholin, dtype=float)))
    substrate = float(np.clip(substrate, 0.6, 2.2))

    hillslope = float(np.clip((gravity / 9.80665) ** 0.30 / max(substrate, 0.4) ** 0.15, 0.35, 1.8))
    return GeomorphicFluidParameters(
        active_fluid=active,
        density_kg_m3=rho,
        viscosity_mpa_s=mu,
        surface_tension_mn_m=sigma,
        gravity_m_s2=gravity,
        stream_power_multiplier=stream,
        runoff_multiplier=runoff,
        sediment_capacity_multiplier=capacity,
        deposition_multiplier=deposition,
        lateral_erosion_multiplier=lateral,
        delta_retention_multiplier=delta_retention,
        hillslope_multiplier=hillslope,
        evaporation_loss_multiplier=evap,
        substrate_erodibility_multiplier=substrate,
        metadata={
            "model": "dimensionless Shields/stream-power-inspired fluid property scaling",
            "reference_fluid": "liquid water near 288 K and Earth gravity",
            "active_fluid_selection_source": selection_source,
            "limitations": "No grain-resolved Shields curve, sediment-size distribution, cohesive bank mechanics, infiltration PDE, or explicit storm duration.",
        },
    )


def scaled_hydrology_config(base_cfg: Any, params: GeomorphicFluidParameters, *, iterations: int | None = None):
    """Return a copied HydrologyConfig adapted to the active planetary fluid."""
    cfg = copy.deepcopy(base_cfg)
    cfg.runoff_base_fraction = float(np.clip(cfg.runoff_base_fraction * params.runoff_multiplier, 0.03, 0.92))
    cfg.max_fluvial_erosion_m_per_iteration = max(
        0.05,
        float(cfg.max_fluvial_erosion_m_per_iteration)
        * params.stream_power_multiplier
        * params.substrate_erodibility_multiplier,
    )
    cfg.deposition_strength = float(np.clip(cfg.deposition_strength * params.deposition_multiplier, 0.05, 1.0))
    cfg.lateral_erosion_fraction = float(np.clip(cfg.lateral_erosion_fraction * params.lateral_erosion_multiplier, 0.02, 1.0))
    cfg.delta_retention_fraction = float(np.clip(cfg.delta_retention_fraction * params.delta_retention_multiplier, 0.05, 1.0))
    cfg.hillslope_diffusion_strength = max(0.001, float(cfg.hillslope_diffusion_strength) * params.hillslope_multiplier)
    if iterations is not None:
        cfg.surface_evolution_iterations = max(0, int(iterations))
    return cfg


def build_exotic_geomorphology(
    grid: SphereGrid,
    terrain: Any,
    climate: Any,
    hydrology: Any,
    params: GeomorphicFluidParameters,
    volatile_cycle: Any | None = None,
    cryogeology: Any | None = None,
) -> ExoticGeomorphologyResult:
    land = np.asarray(terrain.land, dtype=bool)
    discharge = normalize01(np.asarray(hydrology.discharge_index, dtype=float), robust=True)
    slope_y, slope_x = grid.ops.metric_gradient(np.asarray(terrain.elevation_km, dtype=float))
    slope = normalize01(np.hypot(slope_x, slope_y), robust=True)
    erosion = normalize01(
        discharge ** 0.55 * slope ** 0.85 * params.stream_power_multiplier * params.substrate_erodibility_multiplier,
        robust=True,
    ) * land
    deposition = normalize01(
        discharge ** 0.45 * (1.0 - slope) * params.deposition_multiplier,
        robust=True,
    ) * land

    # Evaporites are favored where condensable precipitation/runoff reaches closed or
    # low-gradient basins but evaporation potential is high.  They are compositional
    # deposits, not necessarily terrestrial salts.
    evap = np.zeros(grid.shape, dtype=float)
    organic = np.zeros(grid.shape, dtype=float)
    if volatile_cycle is not None:
        for cyc in getattr(volatile_cycle, "species", {}).values():
            evap = np.maximum(evap, np.asarray(cyc.evaporation_potential, float))
        for key in ("THOLIN", "HCN", "C2H2", "S8", "H2SO4"):
            arr = getattr(volatile_cycle, "photochemical_deposition_by_species", {}).get(key)
            if arr is not None:
                organic += np.asarray(arr, float)
    evaporite = normalize01(evap * (1.0 - slope) * (0.35 + 0.65 * discharge), robust=True) * land
    organic = normalize01(organic * (0.45 + 0.55 * (1.0 - discharge)), robust=True) * land

    if cryogeology is None:
        sublimation = np.zeros(grid.shape, dtype=float)
        wasting = np.zeros(grid.shape, dtype=float)
    else:
        sublimation = normalize01(
            np.asarray(cryogeology.sublimation_erosion_index, float) * (0.4 + 0.6 * slope), robust=True
        ) * land
        wasting = normalize01(
            np.asarray(cryogeology.brittle_fracture_index, float)
            * np.asarray(cryogeology.basal_melt_fraction, float)
            * (0.35 + 0.65 * slope),
            robust=True,
        ) * land
        wasting = smooth_periodic(wasting, (0.7, 1.0)) * land

    metadata = {
        "active_fluid": params.active_fluid,
        "stream_power_multiplier": params.stream_power_multiplier,
        "sediment_capacity_multiplier": params.sediment_capacity_multiplier,
        "deposition_multiplier": params.deposition_multiplier,
        "model": "fluid-aware fluvial/evaporitic/organic/sublimation geomorphic diagnostics",
    }
    return ExoticGeomorphologyResult(
        fluid_erosion_potential=erosion.astype(np.float32),
        fluid_deposition_potential=deposition.astype(np.float32),
        evaporite_deposition_index=evaporite.astype(np.float32),
        organic_sediment_deposition_index=organic.astype(np.float32),
        sublimation_landform_index=np.asarray(sublimation, dtype=np.float32),
        cryogenic_mass_wasting_index=np.asarray(wasting, dtype=np.float32),
        metadata=metadata,
    )


__all__ = [
    "GeomorphicFluidParameters", "ExoticGeomorphologyResult",
    "build_geomorphic_fluid_parameters", "scaled_hydrology_config", "build_exotic_geomorphology",
]
