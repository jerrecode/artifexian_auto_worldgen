from __future__ import annotations

"""Mass-conservative bridge from multicomponent climate condensation to hydrology.

The global climate transport remains a reduced-order single reference moisture tracer,
but composition-aware worlds can contain several simultaneously condensable species.
This module interprets the reference precipitation raster as a *mass flux* of the
climate's active condensable, partitions that mass among every thermodynamically
eligible condensate, and converts each species back to its own liquid-volume depth.

That distinction matters for methane/ethane/water/ammonia mixtures: one millimetre of
water and one millimetre of methane do not carry the same mass.  The forcing therefore
tracks kg m-2 first and only then derives hydrologic depths.  The partition is exactly
mass conservative to floating-point precision.

This is still a reduced-order bridge, not a cloud microphysics solver.  The expensive
atmospheric circulation is not duplicated per species; species share the transported
reference condensate flux according to abundance, saturation state and configured
surface reservoirs.
"""

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .planetary_chemistry import CHEMICALS, detect_condensates, normalize_loose_composition
from .volatile_cycle import _seasonal_species_propensity


@dataclass(slots=True)
class CondensateHydrologyForcing:
    reference_species: str
    monthly_reference_mass_kg_m2: np.ndarray
    monthly_total_precipitation_depth_mm: np.ndarray
    monthly_liquid_input_mm: np.ndarray
    monthly_solid_input_mm: np.ndarray
    monthly_thaw_fraction: np.ndarray
    species_monthly_mass_kg_m2: dict[str, np.ndarray]
    species_monthly_liquid_depth_mm: dict[str, np.ndarray]
    species_monthly_solid_depth_mm: dict[str, np.ndarray]
    metadata: dict

    @property
    def annual_total_precipitation_depth_mm(self) -> np.ndarray:
        return np.sum(self.monthly_total_precipitation_depth_mm, axis=0)

    @property
    def annual_liquid_input_mm(self) -> np.ndarray:
        return np.sum(self.monthly_liquid_input_mm, axis=0)

    @property
    def annual_solid_input_mm(self) -> np.ndarray:
        return np.sum(self.monthly_solid_input_mm, axis=0)

    def to_dict(self) -> dict:
        return {
            **self.metadata,
            "reference_species": self.reference_species,
            "species": sorted(self.species_monthly_mass_kg_m2),
        }


class HydrologyClimateView:
    """Read-only climate facade exposing multicomponent precipitation to hydrology."""

    __slots__ = ("base", "hydrologic_forcing")

    def __init__(self, base: Any, forcing: CondensateHydrologyForcing):
        self.base = base
        self.hydrologic_forcing = forcing

    @property
    def precipitation_mm(self) -> np.ndarray:
        return self.hydrologic_forcing.monthly_total_precipitation_depth_mm

    @property
    def annual_precipitation_mm(self) -> np.ndarray:
        return self.hydrologic_forcing.annual_total_precipitation_depth_mm

    @property
    def snow_fraction(self) -> np.ndarray:
        total = self.hydrologic_forcing.annual_total_precipitation_depth_mm
        solid = self.hydrologic_forcing.annual_solid_input_mm
        return np.divide(
            solid,
            np.maximum(total, 1.0e-12),
            out=np.zeros_like(total, dtype=np.float32),
            where=total > 1.0e-12,
        ).astype(np.float32)

    def __getattr__(self, name: str):
        return getattr(self.base, name)


def _density_kg_m3(species: str) -> float:
    record = CHEMICALS.get(species)
    if record is None or record.liquid_density_kg_m3 is None:
        return 1000.0
    return max(float(record.liquid_density_kg_m3), 1.0)


def _surface_inventory_weights(surface_volatiles: Mapping[str, float] | None) -> dict[str, float]:
    if not surface_volatiles:
        return {}
    out: dict[str, float] = {}
    for raw, amount in surface_volatiles.items():
        key = str(raw).strip()
        if key not in CHEMICALS:
            match = next((candidate for candidate in CHEMICALS if candidate.lower() == key.lower()), None)
            if match is None:
                continue
            key = match
        value = float(amount)
        if math.isfinite(value) and value > 0.0:
            out[key] = out.get(key, 0.0) + value
    return out


def _normalized_surface_phase(
    species: str,
    monthly_temperature_c: np.ndarray,
    partial_pressure_bar: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return liquid/solid fractions whose sum is exactly one where precipitating.

    ``_seasonal_species_propensity`` already contains the same saturation/phase logic
    as the volatile-cycle layer.  Its raw liquid+solid fractions can fall below one in
    a supercritical corner.  A precipitation flux cannot disappear, so the hydrology
    bridge normalizes the condensed phases and uses the melting/triple temperature as
    a deterministic fallback when the screening phase model is indeterminate.
    """
    _cond, liquid, solid = _seasonal_species_propensity(
        species, monthly_temperature_c, partial_pressure_bar
    )
    phase_sum = liquid + solid
    record = CHEMICALS[species]
    t_k = np.asarray(monthly_temperature_c, dtype=np.float64) + 273.15
    if record.triple_or_melting_k is None:
        fallback_liquid = np.ones_like(t_k)
    else:
        fallback_liquid = (t_k >= float(record.triple_or_melting_k)).astype(np.float64)
    liquid_n = np.divide(
        liquid,
        np.maximum(phase_sum, 1.0e-30),
        out=fallback_liquid.copy(),
        where=phase_sum > 1.0e-12,
    )
    liquid_n = np.clip(liquid_n, 0.0, 1.0)
    return liquid_n, 1.0 - liquid_n


def build_condensate_hydrology_forcing(
    astronomy: Any,
    climate: Any,
    *,
    surface_volatiles: Mapping[str, float] | None = None,
) -> CondensateHydrologyForcing:
    """Partition reference precipitation among all hydrologically active condensates.

    The reference climate precipitation depth is converted to kg/m² using the active
    condensable's liquid density.  Species shares are then applied to that mass, so
    the sum of all species mass fluxes equals the reference mass flux exactly.  Each
    species is subsequently converted to its own liquid-equivalent depth.
    """
    base_precip = np.maximum(np.asarray(climate.precipitation_mm, dtype=np.float64), 0.0)
    monthly_temp = np.asarray(climate.temperature_c, dtype=np.float64)
    if base_precip.ndim != 3 or monthly_temp.shape != base_precip.shape:
        raise ValueError("climate precipitation and temperature must have shape (month,y,x)")

    atmosphere = getattr(astronomy, "atmosphere", {}) or {}
    composition = normalize_loose_composition(atmosphere.get("fractions", {}))
    pressure_bar = max(float(atmosphere.get("surface_pressure_bar", 1.0)), 1.0e-12)
    reference_species = str(
        getattr(climate, "metadata", {}).get("active_condensible_species", "H2O")
    )
    if reference_species not in CHEMICALS:
        reference_species = "H2O"
    reference_density = _density_kg_m3(reference_species)
    reference_mass = base_precip * reference_density / 1000.0

    condensates = detect_condensates(
        np.asarray(climate.annual_temperature_c, dtype=np.float64),
        composition,
        pressure_bar,
    )
    eligible = [
        key for key, candidate in condensates.items()
        if candidate.precipitation_capable
        and not candidate.aerosol_only
        and key in CHEMICALS
        and CHEMICALS[key].liquid_density_kg_m3 is not None
    ]
    if reference_species not in eligible:
        eligible.append(reference_species)

    inventory = _surface_inventory_weights(surface_volatiles)
    reservoir_bonus = {
        key: min(2.0, 0.35 * math.log1p(value * 1.0e5))
        for key, value in inventory.items()
    }

    weights: list[np.ndarray] = []
    liquid_phase: dict[str, np.ndarray] = {}
    solid_phase: dict[str, np.ndarray] = {}
    for key in eligible:
        candidate = condensates.get(key)
        fraction = float(composition.get(key, 0.0))
        partial = pressure_bar * fraction if candidate is None else float(candidate.partial_pressure_bar)
        cond, _raw_liquid, _raw_solid = _seasonal_species_propensity(key, monthly_temp, partial)
        abundance_weight = math.sqrt(max(fraction, 1.0e-18))
        # The active reference condensable must remain available even when its
        # atmospheric screening abundance is tiny but a configured surface reservoir
        # continuously supplies it (e.g. an H2O ocean under a dry atmosphere).
        if key == reference_species:
            abundance_weight = max(abundance_weight, 1.0e-5)
        bonus = 1.0 + reservoir_bonus.get(key, 0.0)
        weights.append(cond * abundance_weight * bonus)
        liquid_phase[key], solid_phase[key] = _normalized_surface_phase(key, monthly_temp, partial)

    stack = np.stack(weights, axis=0)
    denom = np.sum(stack, axis=0)
    shares = np.divide(
        stack,
        np.maximum(denom[None, ...], 1.0e-30),
        out=np.zeros_like(stack),
        where=denom[None, ...] > 1.0e-20,
    )
    # Where every screening weight vanishes but the circulation produced reference
    # precipitation, assign that flux to the reference condensable.  This maintains
    # exact closure without inventing an untracked sink.
    ref_index = eligible.index(reference_species)
    no_weight = denom <= 1.0e-20
    if np.any(no_weight):
        shares[:, no_weight] = 0.0
        shares[ref_index, no_weight] = 1.0

    species_mass: dict[str, np.ndarray] = {}
    species_liquid_depth: dict[str, np.ndarray] = {}
    species_solid_depth: dict[str, np.ndarray] = {}
    liquid_total = np.zeros_like(base_precip)
    solid_total = np.zeros_like(base_precip)
    thaw_numerator = np.zeros_like(base_precip)
    thaw_denominator = np.zeros_like(base_precip)

    running_mass = np.zeros_like(reference_mass)
    for i, key in enumerate(eligible):
        if i == len(eligible) - 1:
            mass = reference_mass - running_mass
        else:
            mass = reference_mass * shares[i]
            running_mass += mass
        mass = np.maximum(mass, 0.0)
        rho = _density_kg_m3(key)
        liquid_fraction = liquid_phase[key]
        solid_fraction = solid_phase[key]
        liquid_depth = mass * liquid_fraction / rho * 1000.0
        solid_depth = mass * solid_fraction / rho * 1000.0
        species_mass[key] = mass.astype(np.float32)
        species_liquid_depth[key] = liquid_depth.astype(np.float32)
        species_solid_depth[key] = solid_depth.astype(np.float32)
        liquid_total += liquid_depth
        solid_total += solid_depth
        thaw_numerator += solid_depth * liquid_fraction
        thaw_denominator += solid_depth

    thaw_fraction = np.divide(
        thaw_numerator,
        np.maximum(thaw_denominator, 1.0e-30),
        out=np.zeros_like(thaw_numerator),
        where=thaw_denominator > 1.0e-12,
    )
    # If no solid precipitates in a cell/month, phase state still tells the bucket
    # whether previously stored condensate should thaw.  Use the species-share weighted
    # liquid phase as that memory-compatible fallback.
    phase_weighted_liquid = np.zeros_like(base_precip)
    for i, key in enumerate(eligible):
        phase_weighted_liquid += shares[i] * liquid_phase[key]
    thaw_fraction = np.where(thaw_denominator > 1.0e-12, thaw_fraction, phase_weighted_liquid)
    thaw_fraction = np.clip(thaw_fraction, 0.0, 1.0)

    summed_mass = np.zeros_like(reference_mass)
    for value in species_mass.values():
        summed_mass += np.asarray(value, dtype=np.float64)
    mass_residual = summed_mass - reference_mass
    absolute_mass = float(np.sum(np.abs(reference_mass)))
    relative_mass_residual = float(np.sum(np.abs(mass_residual)) / max(absolute_mass, 1.0e-30))

    total_depth = liquid_total + solid_total
    metadata = {
        "model": "single transported reference condensate partitioned into simultaneous species by saturation/abundance/reservoir state; exact mass closure before density-to-volume conversion",
        "reference_species": reference_species,
        "reference_liquid_density_kg_m3": reference_density,
        "active_hydrologic_species": list(eligible),
        "mass_conservation_relative_l1_residual": relative_mass_residual,
        "reference_condensate_mass_kg_m2_global_sum": float(np.sum(reference_mass)),
        "partitioned_condensate_mass_kg_m2_global_sum": float(np.sum(summed_mass)),
        "mean_volume_depth_ratio_to_reference": float(
            np.mean(total_depth / np.maximum(base_precip, 1.0e-12))
        ),
        "volume_additivity": "species condensate volumes are additive after exact mass partition; non-ideal liquid excess volume is not modeled",
        "limitations": "shared atmospheric transport trajectory; no per-species cloud microphysics, latent-energy feedback or precipitation fall-speed solver",
    }
    return CondensateHydrologyForcing(
        reference_species=reference_species,
        monthly_reference_mass_kg_m2=reference_mass.astype(np.float32),
        monthly_total_precipitation_depth_mm=total_depth.astype(np.float32),
        monthly_liquid_input_mm=liquid_total.astype(np.float32),
        monthly_solid_input_mm=solid_total.astype(np.float32),
        monthly_thaw_fraction=thaw_fraction.astype(np.float32),
        species_monthly_mass_kg_m2=species_mass,
        species_monthly_liquid_depth_mm=species_liquid_depth,
        species_monthly_solid_depth_mm=species_solid_depth,
        metadata=metadata,
    )


def climate_for_hydrology(
    astronomy: Any,
    climate: Any,
    *,
    surface_volatiles: Mapping[str, float] | None = None,
) -> tuple[HydrologyClimateView, CondensateHydrologyForcing]:
    forcing = build_condensate_hydrology_forcing(
        astronomy, climate, surface_volatiles=surface_volatiles
    )
    return HydrologyClimateView(climate, forcing), forcing


__all__ = [
    "CondensateHydrologyForcing",
    "HydrologyClimateView",
    "build_condensate_hydrology_forcing",
    "climate_for_hydrology",
]
