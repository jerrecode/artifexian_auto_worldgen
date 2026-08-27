from __future__ import annotations

"""Multicomponent atmospheric volatile-cycle diagnostics and precipitation fields.

The legacy climate solver still transports one reference moisture tracer for speed.
This layer conservatively decomposes that transported condensate flux among every
chemically/thermodynamically plausible condensate, while also tracking frost,
evaporation/sublimation potential, and photochemical aerosol deposition. It is
therefore immediately useful to methane/ethane, CO2, ammonia and mixed-condensable
worlds without duplicating the expensive circulation solve.
"""

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .grid import SphereGrid, normalize01, smooth_periodic
from .planetary_chemistry import (
    CHEMICALS,
    CondensateCandidate,
    PhotochemicalProduct,
    _approx_saturation_bar,
    detect_condensates,
    evaluate_photochemistry,
    normalize_loose_composition,
    stellar_radiation_indices,
)


@dataclass(slots=True)
class SpeciesCycle:
    species: str
    atmospheric_fraction: float
    partial_pressure_bar: float
    condensation_index: float
    annual_precipitation_mm_equivalent: np.ndarray
    liquid_precipitation_fraction: np.ndarray
    solid_precipitation_fraction: np.ndarray
    frost_deposition_index: np.ndarray
    evaporation_potential: np.ndarray
    sublimation_potential: np.ndarray
    reservoir_exchange_index: np.ndarray
    metadata: dict

    def summary(self) -> dict:
        return {
            "species": self.species,
            "atmospheric_fraction": float(self.atmospheric_fraction),
            "partial_pressure_bar": float(self.partial_pressure_bar),
            "condensation_index": float(self.condensation_index),
            **self.metadata,
        }


@dataclass(slots=True)
class VolatileCycleResult:
    species: dict[str, SpeciesCycle]
    photochemical_products: dict[str, PhotochemicalProduct]
    condensates: dict[str, CondensateCandidate]
    total_condensate_precipitation_mm: np.ndarray
    aerosol_optical_depth_proxy: np.ndarray
    aerosol_deposition_index: np.ndarray
    photochemical_deposition_by_species: dict[str, np.ndarray]
    radiation: dict[str, float]
    metadata: dict

    def to_dict(self) -> dict:
        return {
            **self.metadata,
            "radiation": dict(self.radiation),
            "condensates": {k: v.to_dict() for k, v in self.condensates.items()},
            "photochemical_products": {k: v.to_dict() for k, v in self.photochemical_products.items()},
            "species": {k: v.summary() for k, v in self.species.items()},
        }


def _surface_inventory_weights(surface_volatiles: Mapping[str, float] | None) -> dict[str, float]:
    if not surface_volatiles:
        return {}
    out: dict[str, float] = {}
    for raw, value in surface_volatiles.items():
        key = str(raw).strip()
        if key not in CHEMICALS:
            match = next((k for k in CHEMICALS if k.lower() == key.lower()), None)
            if match is None:
                continue
            key = match
        x = float(value)
        if math.isfinite(x) and x > 0:
            out[key] = out.get(key, 0.0) + x
    return out


def _seasonal_species_propensity(
    species: str,
    monthly_temperature_c: np.ndarray,
    partial_pressure_bar: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp = CHEMICALS[species]
    t_k = np.asarray(monthly_temperature_c, dtype=np.float64) + 273.15
    psat = _approx_saturation_bar(species, t_k)
    ratio = np.zeros_like(t_k)
    finite = np.isfinite(psat) & (psat > 0)
    ratio[finite] = max(float(partial_pressure_bar), 0.0) / np.maximum(psat[finite], 1e-30)
    cond = np.clip((np.log10(np.maximum(ratio, 1e-12)) + 0.70) / 1.10, 0.0, 1.0)
    if sp.triple_or_melting_k is None:
        solid = np.zeros_like(cond)
    else:
        solid = 1.0 / (1.0 + np.exp((t_k - float(sp.triple_or_melting_k)) / 2.5))
    if sp.critical_k is None:
        supercritical = np.zeros_like(cond)
    else:
        supercritical = (t_k >= float(sp.critical_k)).astype(float)
    liquid = (1.0 - solid) * (1.0 - supercritical)
    return cond, np.clip(liquid, 0.0, 1.0), np.clip(solid, 0.0, 1.0)


def build_volatile_cycle(
    grid: SphereGrid,
    astronomy: Any,
    climate: Any,
    *,
    surface_volatiles: Mapping[str, float] | None = None,
    surface_liquids: Any | None = None,
) -> VolatileCycleResult:
    atmosphere = getattr(astronomy, "atmosphere", {}) or {}
    composition = normalize_loose_composition(atmosphere.get("fractions", {}))
    pressure_bar = max(float(atmosphere.get("surface_pressure_bar", 1.0)), 1e-12)
    photochem = evaluate_photochemistry(astronomy, composition)
    condensates = detect_condensates(
        climate.annual_temperature_c,
        composition,
        pressure_bar,
        photochemical_products=photochem,
    )
    radiation = stellar_radiation_indices(astronomy)
    inventory = _surface_inventory_weights(surface_volatiles)

    reservoir_bonus: dict[str, float] = {}
    for key, amount in inventory.items():
        reservoir_bonus[key] = min(2.0, 0.35 * math.log1p(amount * 1e5))
    if surface_liquids is not None:
        for key, part in getattr(surface_liquids, "partitions", {}).items():
            if getattr(part, "liquid_mass_kg", 0.0) > 0:
                reservoir_bonus[key] = max(reservoir_bonus.get(key, 0.0), 0.8)

    monthly_temp = np.asarray(climate.temperature_c, dtype=np.float64)
    monthly_base_precip = np.maximum(np.asarray(climate.precipitation_mm, dtype=np.float64), 0.0)
    annual_base = np.maximum(np.asarray(climate.annual_precipitation_mm, dtype=np.float64), 0.0)
    humidity_monthly = np.clip(np.asarray(climate.humidity_proxy, dtype=np.float64), 0.0, 4.0)
    if monthly_temp.ndim != 3 or monthly_temp.shape[1:] != grid.shape:
        raise ValueError("climate.temperature_c must have shape [month,height,width]")
    if humidity_monthly.shape != monthly_temp.shape:
        raise ValueError("climate.humidity_proxy must match monthly temperature shape")
    humidity_annual = np.mean(humidity_monthly, axis=0)

    eligible = [k for k, v in condensates.items() if v.precipitation_capable and not v.aerosol_only]
    monthly_weights: dict[str, np.ndarray] = {}
    phase_liquid: dict[str, np.ndarray] = {}
    phase_solid: dict[str, np.ndarray] = {}
    for key in eligible:
        candidate = condensates[key]
        cond, liquid, solid = _seasonal_species_propensity(key, monthly_temp, candidate.partial_pressure_bar)
        abundance_weight = math.sqrt(max(candidate.atmospheric_fraction, 1e-18))
        bonus = 1.0 + reservoir_bonus.get(key, 0.0)
        monthly_weights[key] = cond * abundance_weight * bonus
        phase_liquid[key] = liquid
        phase_solid[key] = solid

    if monthly_weights:
        stack = np.stack([monthly_weights[k] for k in eligible], axis=0)
        denom = np.sum(stack, axis=0)
        shares = np.divide(
            stack,
            denom[None, ...],
            out=np.zeros_like(stack),
            where=denom[None, ...] > 1e-12,
        )
    else:
        shares = np.zeros((0, *monthly_temp.shape), dtype=np.float64)

    cycles: dict[str, SpeciesCycle] = {}
    total_precip = np.zeros(grid.shape, dtype=np.float64)
    for i, key in enumerate(eligible):
        monthly_species = monthly_base_precip * shares[i]
        annual_species = monthly_species.sum(axis=0)
        total_precip += annual_species
        liq_w = phase_liquid[key]
        sol_w = phase_solid[key]
        precip_sum = np.sum(monthly_species, axis=0)
        liquid_frac = np.divide(
            np.sum(monthly_species * liq_w, axis=0),
            precip_sum,
            out=np.zeros(grid.shape, dtype=np.float64),
            where=precip_sum > 1e-12,
        )
        solid_frac = np.divide(
            np.sum(monthly_species * sol_w, axis=0),
            precip_sum,
            out=np.zeros(grid.shape, dtype=np.float64),
            where=precip_sum > 1e-12,
        )
        sp = CHEMICALS[key]
        mean_t_k = np.asarray(climate.annual_temperature_c, dtype=np.float64) + 273.15
        psat = _approx_saturation_bar(key, mean_t_k)
        partial = condensates[key].partial_pressure_bar
        undersat = np.clip(1.0 - partial / np.maximum(psat, 1e-30), 0.0, 1.0)
        if sp.critical_k is not None:
            undersat *= mean_t_k < sp.critical_k
        reservoir = float(np.clip(reservoir_bonus.get(key, 0.0), 0.0, 2.0))
        warm = (
            1.0
            if sp.triple_or_melting_k is None
            else 1.0 / (1.0 + np.exp(-(mean_t_k - sp.triple_or_melting_k) / 3.0))
        )
        evap = normalize01(undersat * humidity_annual * warm, robust=True) * min(1.0, 0.3 + reservoir)
        sub = normalize01(undersat * humidity_annual * (1.0 - warm), robust=True) * min(1.0, 0.2 + reservoir)
        frost = normalize01((1.0 - warm) * np.clip(1.0 - undersat, 0.0, 1.0), robust=True)
        exchange = normalize01(
            annual_species / np.maximum(annual_base + 1.0, 1.0) + evap + sub,
            robust=True,
        )
        cycles[key] = SpeciesCycle(
            species=key,
            atmospheric_fraction=float(condensates[key].atmospheric_fraction),
            partial_pressure_bar=float(partial),
            condensation_index=float(condensates[key].condensation_index),
            annual_precipitation_mm_equivalent=annual_species.astype(np.float32),
            liquid_precipitation_fraction=liquid_frac.astype(np.float32),
            solid_precipitation_fraction=solid_frac.astype(np.float32),
            frost_deposition_index=frost.astype(np.float32),
            evaporation_potential=np.asarray(evap, dtype=np.float32),
            sublimation_potential=np.asarray(sub, dtype=np.float32),
            reservoir_exchange_index=exchange.astype(np.float32),
            metadata={
                "mean_annual_precipitation_mm_equivalent": float(
                    np.sum(annual_species * grid.cell_area_weights)
                ),
                "global_liquid_precipitation_fraction": float(
                    np.sum(liquid_frac * grid.cell_area_weights)
                ),
                "global_solid_precipitation_fraction": float(
                    np.sum(solid_frac * grid.cell_area_weights)
                ),
                "surface_reservoir_weight_bonus": float(reservoir_bonus.get(key, 0.0)),
            },
        )

    aerosol_products = {k: p for k, p in photochem.items() if p.aerosol or p.deposited}
    aerosol_optical = np.zeros(grid.shape, dtype=np.float64)
    aerosol_dep = np.zeros(grid.shape, dtype=np.float64)
    by_species: dict[str, np.ndarray] = {}
    insolation_shape = 0.55 + 0.45 * np.cos(np.deg2rad(grid.lat)) ** 2
    scavenging = normalize01(total_precip + 0.15 * annual_base, robust=True)
    for key, product in aerosol_products.items():
        production = float(product.production_index)
        source = production * insolation_shape
        if product.aerosol:
            haze = smooth_periodic(
                source * (0.55 + 0.45 * humidity_annual),
                (2.0, 3.5),
            )
            aerosol_optical += haze
        dry = 0.25 + (0.35 if key == "THOLIN" else 0.10)
        dep = source * (dry + (1.0 - dry) * scavenging)
        dep = normalize01(dep, robust=True) * production
        by_species[key] = dep.astype(np.float32)
        aerosol_dep += dep
    aerosol_optical = normalize01(aerosol_optical, robust=True)
    aerosol_dep = normalize01(aerosol_dep, robust=True)

    metadata = {
        "model": "multicomponent diagnostic volatile cycle decomposed from transported reference-moisture flux",
        "active_precipitating_species": eligible,
        "active_aerosol_or_deposit_species": sorted(aerosol_products),
        "base_transport_precipitation_global_mean_mm": float(
            np.sum(annual_base * grid.cell_area_weights)
        ),
        "resolved_condensate_precipitation_global_mean_mm": float(
            np.sum(total_precip * grid.cell_area_weights)
        ),
        "precipitation_partition_conservative": True,
        "limitations": (
            "Species share one precomputed circulation/moisture transport field; latent heats do not yet feed back "
            "independently into atmospheric dynamics and cloud microphysics is diagnostic."
        ),
    }
    return VolatileCycleResult(
        species=cycles,
        photochemical_products=photochem,
        condensates=condensates,
        total_condensate_precipitation_mm=total_precip.astype(np.float32),
        aerosol_optical_depth_proxy=aerosol_optical.astype(np.float32),
        aerosol_deposition_index=aerosol_dep.astype(np.float32),
        photochemical_deposition_by_species=by_species,
        radiation=radiation,
        metadata=metadata,
    )


__all__ = ["SpeciesCycle", "VolatileCycleResult", "build_volatile_cycle"]
