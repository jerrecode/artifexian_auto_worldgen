from __future__ import annotations

"""Reduced-order planetary thermophysics shared by astronomy and climate.

The built-in phase model is intentionally lightweight and deterministic.  It uses
triple/critical points plus integrated Clausius-Clapeyron saturation curves and is
suitable for procedural screening and raster-scale climate work.  When CoolProp is
installed, scalar phase queries can use its real-fluid equations of state instead.

This module does not claim chemical-equilibrium, photochemical, cloud-microphysics,
or full radiative-transfer accuracy.  It provides a physically interpretable bridge
between composition/pressure, phase stability, greenhouse forcing and tidal heat.
"""

from dataclasses import asdict, dataclass
import importlib.util
import math
from typing import Mapping, Sequence

import numpy as np

R_GAS = 8.31446261815324
G = 6.67430e-11
M_EARTH = 5.9722e24
R_EARTH = 6.371e6


@dataclass(slots=True, frozen=True)
class SpeciesThermo:
    formula: str
    coolprop_name: str | None
    molar_mass_g_mol: float
    triple_temperature_k: float
    triple_pressure_bar: float
    critical_temperature_k: float
    critical_pressure_bar: float
    normal_boiling_temperature_k: float | None
    latent_vaporization_kj_mol: float
    liquid_density_kg_m3: float


# Values are compact engineering references for the deterministic fallback.  The
# optional CoolProp backend should be preferred when high-accuracy real-fluid
# properties are required near saturation/critical regions.
SPECIES: dict[str, SpeciesThermo] = {
    "H2O": SpeciesThermo("H2O", "Water", 18.01528, 273.16, 0.00611657, 647.096, 220.64, 373.124, 40.65, 997.0),
    "CO2": SpeciesThermo("CO2", "CarbonDioxide", 44.0095, 216.592, 5.185, 304.1282, 73.773, None, 15.3, 1100.0),
    "CH4": SpeciesThermo("CH4", "Methane", 16.0425, 90.694, 0.11696, 190.564, 45.992, 111.667, 8.19, 422.0),
    "C2H6": SpeciesThermo("C2H6", "Ethane", 30.069, 90.368, 1.14e-5, 305.322, 48.722, 184.569, 14.72, 544.0),
    "NH3": SpeciesThermo("NH3", "Ammonia", 17.03052, 195.40, 0.0606, 405.40, 113.33, 239.82, 23.35, 682.0),
    "N2": SpeciesThermo("N2", "Nitrogen", 28.0134, 63.151, 0.1252, 126.192, 33.958, 77.355, 5.56, 808.0),
    "O2": SpeciesThermo("O2", "Oxygen", 31.9988, 54.361, 0.00146, 154.581, 50.43, 90.188, 6.82, 1141.0),
    "SO2": SpeciesThermo("SO2", "SulfurDioxide", 64.066, 197.67, 0.0167, 430.64, 78.84, 263.05, 24.9, 1430.0),
    "Ar": SpeciesThermo("Ar", "Argon", 39.948, 83.806, 0.6889, 150.687, 48.63, 87.302, 6.43, 1395.0),
    "H2": SpeciesThermo("H2", "Hydrogen", 2.01588, 13.957, 0.0720, 33.145, 12.964, 20.369, 0.90, 71.0),
    "He": SpeciesThermo("He", "Helium", 4.002602, "", 0.0, 5.1953, 2.2746, 4.222, 0.083, 125.0),
}

# Repair helium's absence of an ordinary triple point without burdening the public
# table with a nullable type.  Helium is never selected as a condensable by default.
SPECIES["He"] = SpeciesThermo("He", "Helium", 4.002602, 2.1768, 0.0504, 5.1953, 2.2746, 4.222, 0.083, 125.0)

ALIASES = {
    "water": "H2O", "h2o": "H2O",
    "carbon dioxide": "CO2", "carbondioxide": "CO2", "co2": "CO2",
    "methane": "CH4", "ch4": "CH4",
    "ethane": "C2H6", "c2h6": "C2H6",
    "ammonia": "NH3", "nh3": "NH3",
    "nitrogen": "N2", "n2": "N2",
    "oxygen": "O2", "o2": "O2",
    "sulfur dioxide": "SO2", "sulphur dioxide": "SO2", "so2": "SO2",
    "argon": "Ar", "ar": "Ar",
    "hydrogen": "H2", "h2": "H2",
    "helium": "He", "he": "He",
}


def canonical_species(name: str) -> str:
    text = str(name).strip()
    if text in SPECIES:
        return text
    key = ALIASES.get(text.lower())
    if key is None:
        raise KeyError(f"unsupported thermodynamic species: {name!r}")
    return key


def coolprop_available() -> bool:
    return importlib.util.find_spec("CoolProp") is not None


def _builtin_saturation_pressure_bar(species: str, temperature_k: np.ndarray | float) -> np.ndarray:
    sp = SPECIES[canonical_species(species)]
    t = np.asarray(temperature_k, dtype=np.float64)
    safe_t = np.maximum(t, 1.0)
    dh = sp.latent_vaporization_kj_mol * 1000.0

    # Above the triple point use a vaporization curve anchored either at the normal
    # boiling point or (for CO2) at the triple point.  Below it, use an approximate
    # sublimation enthalpy so frost/snow stability remains pressure-sensitive.
    if sp.normal_boiling_temperature_k is not None:
        anchor_t = sp.normal_boiling_temperature_k
        anchor_p = 1.01325
    else:
        anchor_t = sp.triple_temperature_k
        anchor_p = sp.triple_pressure_bar
    ln_p_liq = math.log(max(anchor_p, 1e-30)) - dh / R_GAS * (1.0 / safe_t - 1.0 / anchor_t)
    dh_sub = dh * 1.24
    ln_p_sol = math.log(max(sp.triple_pressure_bar, 1e-30)) - dh_sub / R_GAS * (
        1.0 / safe_t - 1.0 / sp.triple_temperature_k
    )
    p = np.exp(np.where(t < sp.triple_temperature_k, ln_p_sol, ln_p_liq))
    p = np.where(t >= sp.critical_temperature_k, sp.critical_pressure_bar, p)
    return np.clip(p, 1e-12, sp.critical_pressure_bar)


def saturation_pressure_bar(
    species: str,
    temperature_k: np.ndarray | float,
    *,
    backend: str = "auto",
) -> np.ndarray | float:
    """Pure-species saturation pressure in bar.

    Array calls always use the deterministic vectorized fallback.  Scalar calls use
    CoolProp when requested/available and the state lies in its saturation domain.
    """
    key = canonical_species(species)
    arr = np.asarray(temperature_k)
    use_cp = backend in {"auto", "coolprop"} and arr.ndim == 0 and coolprop_available()
    if use_cp:
        sp = SPECIES[key]
        if sp.coolprop_name is not None:
            try:
                from CoolProp.CoolProp import PropsSI
                tk = float(arr)
                if sp.triple_temperature_k < tk < sp.critical_temperature_k:
                    value = float(PropsSI("P", "T", tk, "Q", 0, sp.coolprop_name)) / 1e5
                    if math.isfinite(value) and value > 0:
                        return value
            except Exception:
                if backend == "coolprop":
                    raise
    result = _builtin_saturation_pressure_bar(key, arr)
    return float(result) if result.ndim == 0 else result


def _map_coolprop_phase(text: str) -> str:
    phase = text.lower()
    if "supercritical" in phase:
        return "supercritical"
    if "liquid" in phase or "twophase" in phase:
        return "liquid"
    if "gas" in phase:
        return "gas"
    return phase


def phase_at(species: str, temperature_k: float, pressure_bar: float, *, backend: str = "auto") -> str:
    """Return solid/liquid/gas/supercritical for a pure substance at T and P."""
    key = canonical_species(species)
    sp = SPECIES[key]
    t = float(temperature_k)
    p = max(float(pressure_bar), 0.0)
    if backend in {"auto", "coolprop"} and coolprop_available() and sp.coolprop_name is not None:
        try:
            # CoolProp generally does not model the solid region; use its EOS where
            # valid and retain the triple-point fallback below that domain.
            if t >= sp.triple_temperature_k:
                from CoolProp.CoolProp import PhaseSI
                return _map_coolprop_phase(str(PhaseSI("T", t, "P", max(p, 1e-12) * 1e5, sp.coolprop_name)))
        except Exception:
            if backend == "coolprop":
                raise
    if t >= sp.critical_temperature_k:
        return "supercritical" if p >= sp.critical_pressure_bar else "gas"
    if t < sp.triple_temperature_k:
        return "solid" if p >= float(saturation_pressure_bar(key, t, backend="builtin")) else "gas"
    psat = float(saturation_pressure_bar(key, t, backend="builtin"))
    return "liquid" if p >= psat else "gas"


def phase_code_grid(species: str, temperature_k: np.ndarray, pressure_bar: float) -> np.ndarray:
    """Vectorized pure-phase code: 0 gas, 1 liquid, 2 solid, 3 supercritical."""
    key = canonical_species(species)
    sp = SPECIES[key]
    t = np.asarray(temperature_k, dtype=np.float64)
    p = float(pressure_bar)
    psat = np.asarray(saturation_pressure_bar(key, t, backend="builtin"), dtype=np.float64)
    out = np.zeros(t.shape, dtype=np.uint8)
    out[(t < sp.triple_temperature_k) & (p >= psat)] = 2
    out[(t >= sp.triple_temperature_k) & (t < sp.critical_temperature_k) & (p >= psat)] = 1
    out[(t >= sp.critical_temperature_k) & (p >= sp.critical_pressure_bar)] = 3
    return out


def normalize_composition(composition: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, value in composition.items():
        key = canonical_species(name)
        x = float(value)
        if not math.isfinite(x) or x < 0:
            raise ValueError(f"invalid atmospheric fraction for {name}: {value!r}")
        out[key] = out.get(key, 0.0) + x
    total = sum(out.values())
    if total <= 0:
        raise ValueError("atmospheric fractions must sum to a positive value")
    return {k: v / total for k, v in out.items()}


def mean_molar_mass_g_mol(composition: Mapping[str, float]) -> float:
    comp = normalize_composition(composition)
    return sum(v * SPECIES[k].molar_mass_g_mol for k, v in comp.items())


def greenhouse_optical_depth(composition: Mapping[str, float], pressure_bar: float) -> dict[str, float]:
    """Composition/pressure-sensitive grey infrared optical-depth proxy.

    This is a reduced-order radiative model, not line-by-line spectroscopy.  The
    pressure gate prevents a thin Mars-like CO2 atmosphere from receiving the same
    opacity scaling as dense Venus-like CO2, while high-pressure H2 gets a separate
    collision-induced-absorption term.
    """
    comp = normalize_composition(composition)
    p = max(float(pressure_bar), 1e-9)
    partial = {k: p * v for k, v in comp.items()}
    pressure_gate = min(1.0, (p / 0.5) ** 0.35)
    broadening = 1.0 + 0.15 * math.log1p(p)
    terms = {
        "background_collision": 0.04 * p ** 1.25,
        "CO2": 0.22 * math.sqrt(partial.get("CO2", 0.0) / 4.2e-4) * pressure_gate,
        "H2O": 0.42 * math.sqrt(partial.get("H2O", 0.0) / 1.2e-2) * pressure_gate,
        "CH4": 0.0050 * math.sqrt(partial.get("CH4", 0.0) / 1.8e-6) * pressure_gate,
        "NH3": 0.012 * math.sqrt(partial.get("NH3", 0.0) / 1.0e-6) * pressure_gate,
        "SO2": 0.006 * math.sqrt(partial.get("SO2", 0.0) / 1.0e-7) * pressure_gate,
        "H2_CIA": 0.12 * partial.get("H2", 0.0) ** 2,
    }
    for key in ("CO2", "H2O", "CH4", "NH3", "SO2"):
        terms[key] *= broadening
    terms["total"] = max(0.0, sum(terms.values()))
    return terms


def composition_greenhouse_temperature_k(
    equilibrium_temperature_k: float,
    composition: Mapping[str, float],
    pressure_bar: float,
) -> tuple[float, dict[str, float]]:
    terms = greenhouse_optical_depth(composition, pressure_bar)
    tau = terms["total"]
    surface = float(equilibrium_temperature_k) * (1.0 + 0.75 * tau) ** 0.25
    return surface, terms


def select_active_condensible(
    composition: Mapping[str, float],
    surface_volatiles: Mapping[str, float],
    temperature_k: float,
    pressure_bar: float,
    *,
    requested: str = "auto",
) -> str | None:
    if requested != "auto":
        return canonical_species(requested)
    comp = normalize_composition(composition)
    candidates: dict[str, float] = {}
    for name, inventory in surface_volatiles.items():
        key = canonical_species(name)
        if float(inventory) > 0:
            candidates[key] = candidates.get(key, 0.0) + float(inventory)
    for key, frac in comp.items():
        if key in {"H2O", "CO2", "CH4", "C2H6", "NH3", "N2", "SO2"} and frac > 1e-6:
            candidates[key] = candidates.get(key, 0.0) + frac
    if not candidates:
        return None
    best: tuple[float, str] | None = None
    for key, abundance in candidates.items():
        sp = SPECIES[key]
        psat = float(saturation_pressure_bar(key, temperature_k, backend="builtin"))
        partial = max(comp.get(key, 0.0) * pressure_bar, 1e-12)
        proximity = abs(math.log10(max(psat, 1e-12) / partial))
        # Stable surface liquid/solid inventory is a strong preference, followed by
        # atmospheric species near condensation conditions.
        state = phase_at(key, temperature_k, max(pressure_bar, partial), backend="builtin")
        phase_bonus = -2.0 if state in {"liquid", "solid"} else 0.0
        score = proximity + phase_bonus - 0.15 * math.log1p(max(abundance, 0.0))
        if best is None or score < best[0]:
            best = (score, key)
    return None if best is None else best[1]


def relative_vapor_capacity(species: str, temperature_k: np.ndarray, reference_temperature_k: float) -> np.ndarray:
    p = np.asarray(saturation_pressure_bar(species, temperature_k, backend="builtin"), dtype=np.float64)
    pref = float(saturation_pressure_bar(species, reference_temperature_k, backend="builtin"))
    return np.clip(p / max(pref, 1e-12), 0.02, 50.0)


def tidal_heating_power_w(
    *,
    satellite_radius_earth: float,
    primary_mass_earth: float,
    orbit_km: float,
    eccentricity: float,
    love_number_k2: float,
    quality_factor_q: float,
) -> float:
    """Synchronous, small-eccentricity equilibrium-tide heating rate.

    E_dot = 21/2 * (k2/Q) * n^5 R^5/G * e^2.
    Obliquity tides, forced libration and resonance evolution are excluded.
    """
    if quality_factor_q <= 0 or orbit_km <= 0 or primary_mass_earth <= 0:
        return 0.0
    a = float(orbit_km) * 1000.0
    n = math.sqrt(G * float(primary_mass_earth) * M_EARTH / a**3)
    r = float(satellite_radius_earth) * R_EARTH
    e = max(float(eccentricity), 0.0)
    return 10.5 * (float(love_number_k2) / float(quality_factor_q)) * n**5 * r**5 / G * e**2


def tidal_heating_flux_w_m2(**kwargs: float) -> float:
    power = tidal_heating_power_w(**kwargs)
    area = 4.0 * math.pi * (float(kwargs["satellite_radius_earth"]) * R_EARTH) ** 2
    return power / max(area, 1.0)


def geological_activity_regime(total_internal_heat_flux_w_m2: float) -> str:
    q = max(float(total_internal_heat_flux_w_m2), 0.0)
    if q < 0.015:
        return "geologically_inactive"
    if q < 0.05:
        return "weak_or_stagnant_lid"
    if q < 0.25:
        return "active"
    if q < 1.0:
        return "strongly_active"
    return "extreme_tidally_active"


def atmosphere_diagnostics(
    *,
    composition: Mapping[str, float],
    pressure_bar: float,
    temperature_k: float,
    gravity_m_s2: float,
) -> dict[str, object]:
    comp = normalize_composition(composition)
    mw = mean_molar_mass_g_mol(comp)
    partial = {k: v * pressure_bar for k, v in comp.items()}
    scale_height_km = R_GAS * max(float(temperature_k), 1.0) / ((mw / 1000.0) * max(float(gravity_m_s2), 1e-9)) / 1000.0
    density = pressure_bar * 1e5 * (mw / 1000.0) / (R_GAS * max(float(temperature_k), 1.0))
    column_mass = pressure_bar * 1e5 / max(float(gravity_m_s2), 1e-9)
    return {
        "surface_pressure_bar": float(pressure_bar),
        "fractions": comp,
        "partial_pressures_bar": partial,
        "mean_molar_mass_g_mol": float(mw),
        "scale_height_km_approx": float(scale_height_km),
        "surface_density_kg_m3_approx": float(density),
        "atmospheric_column_mass_kg_m2": float(column_mass),
    }


def species_metadata() -> dict[str, dict[str, object]]:
    return {name: asdict(sp) for name, sp in SPECIES.items()}


__all__ = [
    "SpeciesThermo", "SPECIES", "canonical_species", "coolprop_available",
    "saturation_pressure_bar", "phase_at", "phase_code_grid", "normalize_composition",
    "mean_molar_mass_g_mol", "greenhouse_optical_depth",
    "composition_greenhouse_temperature_k", "select_active_condensible",
    "relative_vapor_capacity", "tidal_heating_power_w", "tidal_heating_flux_w_m2",
    "geological_activity_regime", "atmosphere_diagnostics", "species_metadata",
]
