from __future__ import annotations

"""Reduced-order planetary thermophysics shared by astronomy and climate.

The vectorized built-in backend uses triple/critical data plus integrated
Clausius-Clapeyron vaporization and sublimation curves. Scalar fluid-region queries
can optionally delegate to CoolProp. These models are for procedural planetary
screening and climate coupling, not full chemical equilibrium, photochemistry,
cloud microphysics, or line-by-line radiative transfer.
"""

from dataclasses import asdict, dataclass
import importlib.util
import math
from typing import Mapping

import numpy as np

R_GAS = 8.31446261815324
G = 6.67430e-11
M_EARTH = 5.9722e24
R_EARTH = 6.371e6


def _finite_positive(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return out


def _finite_nonnegative(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return out


def _positive_temperature_array(temperature_k: np.ndarray | float) -> np.ndarray:
    t = np.asarray(temperature_k, dtype=np.float64)
    if np.any(~np.isfinite(t)) or np.any(t <= 0.0):
        raise ValueError("temperature_k must be finite and positive")
    return t


def _validate_backend(backend: str) -> str:
    if backend not in {"auto", "builtin", "coolprop"}:
        raise ValueError("backend must be auto, builtin, or coolprop")
    return backend


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
    latent_sublimation_kj_mol: float
    liquid_density_kg_m3: float


# Screening-grade pure-component constants. CoolProp, when installed and when a
# fluid is available there, is preferred for scalar liquid properties. The built-in
# curves deliberately trade precision for vectorizable phase screening.
SPECIES: dict[str, SpeciesThermo] = {
    "H2O": SpeciesThermo("H2O", "Water", 18.01528, 273.16, 0.00611657, 647.096, 220.64, 373.124, 40.65, 51.0, 997.0),
    "CO2": SpeciesThermo("CO2", "CarbonDioxide", 44.0095, 216.592, 5.185, 304.1282, 73.773, None, 15.3, 25.2, 1100.0),
    "CH4": SpeciesThermo("CH4", "Methane", 16.0425, 90.694, 0.11696, 190.564, 45.992, 111.667, 8.19, 9.7, 422.0),
    "C2H6": SpeciesThermo("C2H6", "Ethane", 30.069, 90.368, 1.14e-5, 305.322, 48.722, 184.569, 14.72, 17.0, 544.0),
    "NH3": SpeciesThermo("NH3", "Ammonia", 17.03052, 195.40, 0.0606, 405.40, 113.33, 239.82, 23.35, 29.0, 682.0),
    "N2": SpeciesThermo("N2", "Nitrogen", 28.0134, 63.151, 0.1252, 126.192, 33.958, 77.355, 5.56, 6.8, 808.0),
    "O2": SpeciesThermo("O2", "Oxygen", 31.9988, 54.361, 0.00146, 154.581, 50.43, 90.188, 6.82, 7.9, 1141.0),
    "SO2": SpeciesThermo("SO2", "SulfurDioxide", 64.066, 197.67, 0.0167, 430.64, 78.84, 263.05, 24.9, 31.0, 1430.0),
    "Ar": SpeciesThermo("Ar", "Argon", 39.948, 83.806, 0.6889, 150.687, 48.63, 87.302, 6.43, 7.8, 1395.0),
    "H2": SpeciesThermo("H2", "Hydrogen", 2.01588, 13.957, 0.0720, 33.145, 12.964, 20.369, 0.90, 1.0, 71.0),
    "He": SpeciesThermo("He", "Helium", 4.002602, 2.1768, 0.0504, 5.1953, 2.2746, 4.222, 0.083, 0.10, 125.0),
    "CO": SpeciesThermo("CO", "CarbonMonoxide", 28.0101, 68.16, 0.153, 132.86, 34.94, 81.64, 6.04, 7.6, 789.0),
    "H2S": SpeciesThermo("H2S", "HydrogenSulfide", 34.0809, 187.67, 0.23, 373.10, 89.63, 212.87, 18.7, 24.0, 900.0),
    "C3H8": SpeciesThermo("C3H8", "Propane", 44.0956, 85.53, 1.7e-8, 369.89, 42.51, 231.04, 19.0, 22.0, 493.0),
    "C2H4": SpeciesThermo("C2H4", "Ethylene", 28.0532, 103.99, 0.0012, 282.35, 50.42, 169.38, 13.5, 16.0, 570.0),
    # Acetylene's triple pressure lies above one atmosphere, so there is no ordinary
    # 1-atm liquid boiling point; use the triple point as the vapor-pressure anchor.
    "C2H2": SpeciesThermo("C2H2", None, 26.0373, 192.4, 1.28, 308.3, 61.4, None, 16.7, 20.0, 620.0),
    "CH3OH": SpeciesThermo("CH3OH", "Methanol", 32.0419, 175.61, 1.9e-6, 512.6, 80.9, 337.63, 35.3, 40.0, 792.0),
    "HCN": SpeciesThermo("HCN", None, 27.0253, 259.86, 0.19, 456.7, 53.9, 299.2, 25.0, 31.0, 690.0),
}

ALIASES = {
    "water": "H2O", "h2o": "H2O", "carbon dioxide": "CO2", "carbondioxide": "CO2", "co2": "CO2",
    "methane": "CH4", "ch4": "CH4", "ethane": "C2H6", "c2h6": "C2H6", "ammonia": "NH3", "nh3": "NH3",
    "nitrogen": "N2", "n2": "N2", "oxygen": "O2", "o2": "O2", "sulfur dioxide": "SO2", "sulphur dioxide": "SO2", "so2": "SO2",
    "argon": "Ar", "ar": "Ar", "hydrogen": "H2", "h2": "H2", "helium": "He", "he": "He",
    "carbon monoxide": "CO", "carbonmonoxide": "CO", "co": "CO",
    "hydrogen sulfide": "H2S", "hydrogen sulphide": "H2S", "h2s": "H2S",
    "propane": "C3H8", "c3h8": "C3H8", "ethylene": "C2H4", "ethene": "C2H4", "c2h4": "C2H4",
    "acetylene": "C2H2", "ethyne": "C2H2", "c2h2": "C2H2", "methanol": "CH3OH", "ch3oh": "CH3OH",
    "hydrogen cyanide": "HCN", "hcn": "HCN",
}


def canonical_species(name: str) -> str:
    text = str(name).strip()
    if text in SPECIES: return text
    key = ALIASES.get(text.lower())
    if key is None: raise KeyError(f"unsupported thermodynamic species: {name!r}")
    return key


def coolprop_available() -> bool:
    return importlib.util.find_spec("CoolProp") is not None


def _builtin_saturation_pressure_bar(species: str, temperature_k: np.ndarray | float) -> np.ndarray:
    sp = SPECIES[canonical_species(species)]
    t = _positive_temperature_array(temperature_k)
    safe_t = t
    if sp.normal_boiling_temperature_k is not None:
        anchor_t, anchor_p = sp.normal_boiling_temperature_k, 1.01325
    else:
        anchor_t, anchor_p = sp.triple_temperature_k, sp.triple_pressure_bar
    ln_liq = math.log(max(anchor_p, 1e-30)) - sp.latent_vaporization_kj_mol * 1000.0 / R_GAS * (1.0 / safe_t - 1.0 / anchor_t)
    ln_sol = math.log(max(sp.triple_pressure_bar, 1e-30)) - sp.latent_sublimation_kj_mol * 1000.0 / R_GAS * (1.0 / safe_t - 1.0 / sp.triple_temperature_k)
    p = np.exp(np.where(t < sp.triple_temperature_k, ln_sol, ln_liq))
    p = np.where(t >= sp.critical_temperature_k, sp.critical_pressure_bar, p)
    return np.clip(p, 1e-14, sp.critical_pressure_bar)


def saturation_pressure_bar(species: str, temperature_k: np.ndarray | float, *, backend: str = "auto") -> np.ndarray | float:
    key = canonical_species(species)
    backend = _validate_backend(backend)
    arr = _positive_temperature_array(temperature_k)
    if backend in {"auto", "coolprop"} and arr.ndim == 0 and coolprop_available():
        sp = SPECIES[key]
        if sp.coolprop_name is not None:
            try:
                from CoolProp.CoolProp import PropsSI
                tk = float(arr)
                if sp.triple_temperature_k < tk < sp.critical_temperature_k:
                    value = float(PropsSI("P", "T", tk, "Q", 0, sp.coolprop_name)) / 1e5
                    if math.isfinite(value) and value > 0: return value
            except Exception:
                if backend == "coolprop": raise
    result = _builtin_saturation_pressure_bar(key, arr)
    return float(result) if result.ndim == 0 else result


def _map_coolprop_phase(text: str) -> str:
    phase = text.lower()
    if "supercritical" in phase: return "supercritical"
    if "liquid" in phase or "twophase" in phase: return "liquid"
    if "gas" in phase: return "gas"
    return phase


def phase_at(species: str, temperature_k: float, pressure_bar: float, *, backend: str = "auto") -> str:
    key = canonical_species(species)
    backend = _validate_backend(backend)
    sp = SPECIES[key]
    t = _finite_positive("temperature_k", temperature_k)
    p = _finite_nonnegative("pressure_bar", pressure_bar)
    if backend in {"auto", "coolprop"} and coolprop_available() and sp.coolprop_name is not None:
        try:
            if t >= sp.triple_temperature_k:
                from CoolProp.CoolProp import PhaseSI
                return _map_coolprop_phase(str(PhaseSI("T", t, "P", max(p, 1e-12) * 1e5, sp.coolprop_name)))
        except Exception:
            if backend == "coolprop": raise
    if t >= sp.critical_temperature_k: return "supercritical" if p >= sp.critical_pressure_bar else "gas"
    psat = float(saturation_pressure_bar(key, t, backend="builtin"))
    if t < sp.triple_temperature_k: return "solid" if p >= psat else "gas"
    return "liquid" if p >= psat else "gas"


def phase_code_grid(species: str, temperature_k: np.ndarray, pressure_bar: float) -> np.ndarray:
    key = canonical_species(species)
    sp = SPECIES[key]
    t = _positive_temperature_array(temperature_k)
    p = _finite_nonnegative("pressure_bar", pressure_bar)
    psat = np.asarray(saturation_pressure_bar(key, t, backend="builtin"), dtype=np.float64); out = np.zeros(t.shape, dtype=np.uint8)
    out[(t < sp.triple_temperature_k) & (p >= psat)] = 2
    out[(t >= sp.triple_temperature_k) & (t < sp.critical_temperature_k) & (p >= psat)] = 1
    out[(t >= sp.critical_temperature_k) & (p >= sp.critical_pressure_bar)] = 3
    return out


def normalize_composition(composition: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, value in composition.items():
        key = canonical_species(name); x = float(value)
        if not math.isfinite(x) or x < 0: raise ValueError(f"invalid atmospheric fraction for {name}: {value!r}")
        out[key] = out.get(key, 0.0) + x
    total = sum(out.values())
    if total <= 0: raise ValueError("atmospheric fractions must sum to a positive value")
    return {k: v / total for k, v in out.items()}


def mean_molar_mass_g_mol(composition: Mapping[str, float]) -> float:
    comp = normalize_composition(composition)
    return sum(v * SPECIES[k].molar_mass_g_mol for k, v in comp.items())


def greenhouse_optical_depth(composition: Mapping[str, float], pressure_bar: float, *, path_length_factor: float = 1.0) -> dict[str, float]:
    comp = normalize_composition(composition)
    p = _finite_positive("pressure_bar", pressure_bar)
    path_factor = _finite_positive("path_length_factor", path_length_factor)
    partial = {k: p * v for k, v in comp.items()}
    pressure_gate = min(1.0, (p / 0.5) ** 0.35)
    broadening = 1.0 + 0.15 * math.log1p(p)
    path = float(np.clip(path_factor, 0.1, 10.0)) ** 0.45
    terms = {
        "background_collision": 0.04 * p**1.25,
        "CO2": 0.22 * math.sqrt(partial.get("CO2", 0.0) / 4.2e-4) * pressure_gate,
        "H2O": 0.42 * math.sqrt(partial.get("H2O", 0.0) / 1.2e-2) * pressure_gate,
        "CH4": 0.0050 * math.sqrt(partial.get("CH4", 0.0) / 1.8e-6) * pressure_gate,
        "NH3": 0.012 * math.sqrt(partial.get("NH3", 0.0) / 1.0e-6) * pressure_gate,
        "SO2": 0.006 * math.sqrt(partial.get("SO2", 0.0) / 1.0e-7) * pressure_gate,
        "H2S": 0.0025 * math.sqrt(partial.get("H2S", 0.0) / 1.0e-7) * pressure_gate,
        "H2_CIA": 0.12 * partial.get("H2", 0.0)**2,
    }
    for key in ("CO2", "H2O", "CH4", "NH3", "SO2", "H2S"): terms[key] *= broadening
    for key in tuple(terms): terms[key] *= path
    terms["total"] = max(0.0, sum(terms.values())); terms["path_length_factor"] = path_factor
    return terms


def composition_greenhouse_temperature_k(equilibrium_temperature_k: float, composition: Mapping[str, float], pressure_bar: float,
                                         *, path_length_factor: float = 1.0) -> tuple[float, dict[str, float]]:
    equilibrium = _finite_positive("equilibrium_temperature_k", equilibrium_temperature_k)
    terms = greenhouse_optical_depth(composition, pressure_bar, path_length_factor=path_length_factor)
    tau = terms["total"]
    return equilibrium * (1.0 + 0.75 * tau) ** 0.25, terms


def select_active_condensible(composition: Mapping[str, float], surface_volatiles: Mapping[str, float], temperature_k: float,
                              pressure_bar: float, *, requested: str = "auto") -> str | None:
    """Select the reference climate condensable, strongly preferring surface reservoirs.

    This function chooses one *transport reference* for the legacy climate moisture
    solver. The advanced volatile-cycle layer can subsequently activate multiple
    simultaneous condensates. Surface ocean/ice inventory outranks atmospheric
    background gases; supercritical species are excluded from automatic selection.
    """
    temperature = _finite_positive("temperature_k", temperature_k)
    pressure = _finite_positive("pressure_bar", pressure_bar)
    if requested != "auto":
        return canonical_species(requested)
    comp = normalize_composition(composition)
    surface: dict[str, float] = {}
    for name, inventory in surface_volatiles.items():
        key = canonical_species(name)
        amount = float(inventory)
        if not math.isfinite(amount) or amount < 0.0:
            raise ValueError("surface volatile inventories must be finite and non-negative")
        if amount > 0:
            surface[key] = surface.get(key, 0.0) + amount
    if surface:
        best: tuple[float, str] | None = None
        for key, abundance in surface.items():
            state = phase_at(key, temperature, pressure, backend="builtin")
            psat = float(saturation_pressure_bar(key, temperature, backend="builtin"))
            proximity = abs(math.log10(max(psat, 1e-12) / pressure))
            phase_bonus = -3.0 if state == "liquid" else (-1.5 if state == "solid" else 0.0)
            score = proximity + phase_bonus - 0.25 * math.log1p(abundance)
            if best is None or score < best[0]: best = (score, key)
        return None if best is None else best[1]

    condensable_candidates = {
        "H2O", "CO2", "CH4", "C2H6", "NH3", "N2", "SO2", "CO", "H2S",
        "C3H8", "C2H4", "C2H2", "CH3OH", "HCN",
    }
    best = None
    for key, frac in comp.items():
        if key not in condensable_candidates or frac <= 1e-6: continue
        sp = SPECIES[key]
        if temperature >= sp.critical_temperature_k: continue
        psat = float(saturation_pressure_bar(key, temperature, backend="builtin")); partial = max(frac * pressure, 1e-12)
        score = abs(math.log10(max(psat, 1e-12) / partial)) - 0.15 * math.log1p(frac)
        if best is None or score < best[0]: best = (score, key)
    return None if best is None else best[1]


def relative_vapor_capacity(species: str, temperature_k: np.ndarray, reference_temperature_k: float) -> np.ndarray:
    p = np.asarray(saturation_pressure_bar(species, temperature_k, backend="builtin"), dtype=np.float64)
    pref = float(saturation_pressure_bar(species, reference_temperature_k, backend="builtin"))
    return np.clip(p / max(pref, 1e-12), 0.02, 50.0)


def tidal_heating_power_w(*, satellite_radius_earth: float, primary_mass_earth: float, orbit_km: float, eccentricity: float,
                          love_number_k2: float, quality_factor_q: float) -> float:
    radius = _finite_positive("tidal heating satellite_radius_earth", satellite_radius_earth)
    primary_mass = _finite_positive("tidal heating primary_mass_earth", primary_mass_earth)
    orbit = _finite_positive("tidal heating orbit_km", orbit_km)
    eccentricity_value = _finite_nonnegative("tidal heating eccentricity", eccentricity)
    love = _finite_nonnegative("tidal heating love_number_k2", love_number_k2)
    quality = _finite_positive("tidal heating quality_factor_q", quality_factor_q)
    if eccentricity_value >= 1.0:
        raise ValueError("tidal heating eccentricity must be less than 1")
    a = orbit * 1000.0
    n = math.sqrt(G * primary_mass * M_EARTH / a**3)
    r = radius * R_EARTH
    return 10.5 * (love / quality) * n**5 * r**5 / G * eccentricity_value**2


def tidal_heating_flux_w_m2(**kwargs: float) -> float:
    power = tidal_heating_power_w(**kwargs)
    radius = _finite_positive("tidal heating satellite_radius_earth", kwargs["satellite_radius_earth"])
    area = 4.0 * math.pi * (radius * R_EARTH) ** 2
    return power / area


def geological_activity_regime(total_internal_heat_flux_w_m2: float) -> str:
    q = max(float(total_internal_heat_flux_w_m2), 0.0)
    if q < 0.015: return "geologically_inactive"
    if q < 0.05: return "weak_or_stagnant_lid"
    if q < 0.25: return "active"
    if q < 1.0: return "strongly_active"
    return "extreme_tidally_active"


def atmosphere_diagnostics(*, composition: Mapping[str, float], pressure_bar: float, temperature_k: float,
                           gravity_m_s2: float) -> dict[str, object]:
    pressure = _finite_positive("pressure_bar", pressure_bar)
    temperature = _finite_positive("temperature_k", temperature_k)
    gravity = _finite_positive("gravity_m_s2", gravity_m_s2)
    comp = normalize_composition(composition)
    mw = mean_molar_mass_g_mol(comp)
    partial = {k: v * pressure for k, v in comp.items()}
    scale_height_km = R_GAS * temperature / ((mw / 1000.0) * gravity) / 1000.0
    density = pressure * 1e5 * (mw / 1000.0) / (R_GAS * temperature)
    column_mass = pressure * 1e5 / gravity
    return {"surface_pressure_bar": pressure, "fractions": comp, "partial_pressures_bar": partial,
            "mean_molar_mass_g_mol": float(mw), "scale_height_km_approx": float(scale_height_km),
            "surface_density_kg_m3_approx": float(density), "atmospheric_column_mass_kg_m2": float(column_mass)}


def species_metadata() -> dict[str, dict[str, object]]:
    return {name: asdict(sp) for name, sp in SPECIES.items()}


__all__ = ["SpeciesThermo", "SPECIES", "canonical_species", "coolprop_available", "saturation_pressure_bar", "phase_at",
           "phase_code_grid", "normalize_composition", "mean_molar_mass_g_mol", "greenhouse_optical_depth",
           "composition_greenhouse_temperature_k", "select_active_condensible", "relative_vapor_capacity", "tidal_heating_power_w",
           "tidal_heating_flux_w_m2", "geological_activity_regime", "atmosphere_diagnostics", "species_metadata"]
