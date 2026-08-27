from __future__ import annotations

"""Reduced-order planetary chemistry, cloud-condensate and photochemistry layer.

The procedural generator intentionally separates *screening chemistry* from detailed
kinetics.  This module answers three questions cheaply and deterministically:

* which atmospheric/surface species are present in physically significant amounts;
* which of them can plausibly condense somewhere in the supplied temperature field;
* which important irradiation-driven products are favored by the stellar spectrum
  proxy and precursor abundances.

It is not a Gibbs-energy minimizer, reaction-network integrator, cloud microphysics
solver, or line-by-line photochemical model.  All generated values are therefore
explicitly dimensionless production/condensation indices unless a real thermodynamic
quantity is named.
"""

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np

from .planetary_physics import R_GAS, SPECIES, canonical_species, saturation_pressure_bar


@dataclass(slots=True, frozen=True)
class ChemicalSpecies:
    formula: str
    name: str
    molar_mass_g_mol: float
    normal_boiling_k: float | None
    triple_or_melting_k: float | None
    critical_k: float | None
    latent_vaporization_kj_mol: float | None
    liquid_density_kg_m3: float | None
    viscosity_mpa_s: float | None
    surface_tension_mn_m: float | None
    min_significant_fraction: float
    roles: tuple[str, ...]
    notes: str = ""


@dataclass(slots=True)
class PhotochemicalProduct:
    product: str
    pathway: str
    production_index: float
    abundance_proxy: float
    aerosol: bool
    deposited: bool
    reactant_limiter: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class CondensateCandidate:
    species: str
    atmospheric_fraction: float
    partial_pressure_bar: float
    supersaturated_area_fraction: float
    near_saturation_area_fraction: float
    condensation_index: float
    dominant_phase: str
    precipitation_capable: bool
    aerosol_only: bool

    def to_dict(self) -> dict:
        return asdict(self)


# Values below are deliberately screening-grade.  Well-supported bulk fluids are
# still delegated to planetary_physics/CoolProp whenever possible.  Extended species
# exist so exotic atmospheres can produce plausible clouds/aerosols and trace cycles
# without pretending that every candidate has a precision equation of state.
CHEMICALS: dict[str, ChemicalSpecies] = {
    "H2O": ChemicalSpecies("H2O", "water", 18.0153, 373.124, 273.16, 647.096, 40.65, 997.0, 1.00, 72.0, 1e-8, ("condensable", "ocean", "ice", "greenhouse", "weathering")),
    "CO2": ChemicalSpecies("CO2", "carbon dioxide", 44.0095, None, 216.59, 304.13, 15.3, 1100.0, 0.10, 17.0, 1e-7, ("condensable", "ice", "greenhouse", "photochemical")),
    "CH4": ChemicalSpecies("CH4", "methane", 16.0425, 111.67, 90.69, 190.56, 8.19, 422.0, 0.18, 17.0, 1e-8, ("condensable", "ocean", "greenhouse", "photochemical", "hydrology")),
    "C2H6": ChemicalSpecies("C2H6", "ethane", 30.069, 184.57, 90.37, 305.32, 14.72, 544.0, 0.24, 16.0, 1e-9, ("condensable", "ocean", "photochemical", "hydrology")),
    "NH3": ChemicalSpecies("NH3", "ammonia", 17.0305, 239.82, 195.4, 405.4, 23.35, 682.0, 0.35, 23.0, 1e-8, ("condensable", "cryomagma", "antifreeze", "greenhouse")),
    "N2": ChemicalSpecies("N2", "nitrogen", 28.0134, 77.36, 63.15, 126.19, 5.56, 808.0, 0.16, 8.9, 1e-7, ("condensable", "background_gas", "photochemical")),
    "O2": ChemicalSpecies("O2", "oxygen", 31.9988, 90.19, 54.36, 154.58, 6.82, 1141.0, 0.20, 13.2, 1e-7, ("condensable", "oxidant", "photochemical")),
    "SO2": ChemicalSpecies("SO2", "sulfur dioxide", 64.066, 263.05, 197.67, 430.64, 24.9, 1430.0, 0.40, 32.0, 1e-9, ("condensable", "greenhouse", "volcanic", "photochemical")),
    "H2": ChemicalSpecies("H2", "hydrogen", 2.0159, 20.37, 13.96, 33.15, 0.90, 71.0, 0.013, 2.4, 1e-6, ("background_gas", "reducing", "escape")),
    "He": ChemicalSpecies("He", "helium", 4.0026, 4.22, 2.18, 5.20, 0.083, 125.0, 0.004, 0.35, 1e-6, ("background_gas",)),
    "Ar": ChemicalSpecies("Ar", "argon", 39.948, 87.30, 83.81, 150.69, 6.43, 1395.0, 0.24, 13.0, 1e-7, ("condensable", "background_gas")),
    "CO": ChemicalSpecies("CO", "carbon monoxide", 28.010, 81.64, 68.15, 132.86, 6.0, 790.0, 0.17, 9.0, 1e-8, ("condensable", "reducing", "photochemical")),
    "H2S": ChemicalSpecies("H2S", "hydrogen sulfide", 34.081, 212.9, 187.6, 373.2, 18.7, 900.0, 0.25, 26.0, 1e-9, ("condensable", "reducing", "volcanic", "photochemical")),
    "HCN": ChemicalSpecies("HCN", "hydrogen cyanide", 27.026, 299.2, 259.9, 456.7, 25.0, 690.0, 0.20, 19.0, 1e-10, ("condensable", "nitrile", "photochemical", "organic_deposit")),
    "C2H2": ChemicalSpecies("C2H2", "acetylene", 26.038, 189.3, 192.4, 308.3, 16.7, 620.0, 0.20, 20.0, 1e-10, ("condensable", "photochemical", "organic_deposit")),
    "C2H4": ChemicalSpecies("C2H4", "ethylene", 28.054, 169.4, 104.0, 282.4, 13.5, 570.0, 0.18, 17.0, 1e-10, ("condensable", "photochemical")),
    "C3H8": ChemicalSpecies("C3H8", "propane", 44.096, 231.0, 85.5, 369.8, 19.0, 493.0, 0.20, 16.0, 1e-10, ("condensable", "ocean_minor", "photochemical")),
    "CH3OH": ChemicalSpecies("CH3OH", "methanol", 32.042, 337.8, 175.6, 512.6, 35.2, 792.0, 0.55, 22.6, 1e-10, ("condensable", "antifreeze", "organic")),
    "O3": ChemicalSpecies("O3", "ozone", 47.998, 161.2, 80.7, 261.0, 15.0, 1350.0, 1.0, 38.0, 1e-12, ("oxidant", "photochemical", "uv_shield")),
    "H2O2": ChemicalSpecies("H2O2", "hydrogen peroxide", 34.015, 423.0, 272.7, 730.0, 45.0, 1450.0, 1.25, 80.0, 1e-12, ("oxidant", "radiolytic", "ice_chemistry")),
    "H2SO4": ChemicalSpecies("H2SO4", "sulfuric acid", 98.079, None, 283.5, None, None, 1840.0, 25.0, 55.0, 1e-12, ("aerosol", "acid_cloud", "photochemical"), "Bulk vapor pressure is not modeled; treated as aerosol/cloud product."),
    "NH4SH": ChemicalSpecies("NH4SH", "ammonium hydrosulfide", 51.11, None, None, None, None, None, None, None, 1e-12, ("aerosol", "salt_cloud"), "Reaction condensate; treated as solid cloud material."),
    "S8": ChemicalSpecies("S8", "elemental sulfur aerosol", 256.52, None, 388.4, None, None, None, None, None, 1e-12, ("aerosol", "sulfur_haze", "deposit")),
    "THOLIN": ChemicalSpecies("THOLIN", "complex nitrogen-rich organic haze", 100.0, None, None, None, None, None, None, None, 1e-12, ("aerosol", "organic_haze", "deposit"), "Pseudo-species representing a distribution of complex refractory organics."),
}

ALIASES = {
    "water": "H2O", "methane": "CH4", "ethane": "C2H6", "ammonia": "NH3",
    "nitrogen": "N2", "oxygen": "O2", "carbon dioxide": "CO2", "carbon monoxide": "CO",
    "sulfur dioxide": "SO2", "sulphur dioxide": "SO2", "hydrogen sulfide": "H2S",
    "hydrogen sulphide": "H2S", "hydrogen cyanide": "HCN", "acetylene": "C2H2",
    "ethylene": "C2H4", "propane": "C3H8", "methanol": "CH3OH", "ozone": "O3",
    "hydrogen peroxide": "H2O2", "sulfuric acid": "H2SO4", "sulphuric acid": "H2SO4",
    "ammonium hydrosulfide": "NH4SH", "tholin": "THOLIN", "tholins": "THOLIN",
}


def canonical_chemical(name: str) -> str:
    text = str(name).strip()
    if text in CHEMICALS:
        return text
    low = text.lower()
    if low in ALIASES:
        return ALIASES[low]
    for key in CHEMICALS:
        if key.lower() == low:
            return key
    raise KeyError(f"unsupported planetary chemical species: {name!r}")


def normalize_loose_composition(composition: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, value in composition.items():
        try:
            key = canonical_chemical(name)
        except KeyError:
            # Unknown trace species are deliberately ignored by the screening model
            # rather than breaking an otherwise usable speculative atmosphere.
            continue
        x = float(value)
        if math.isfinite(x) and x > 0:
            out[key] = out.get(key, 0.0) + x
    total = sum(out.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in out.items()}


def stellar_radiation_indices(astronomy) -> dict[str, float]:
    """Return bolometric, UV and energetic-particle proxies at the generated world."""
    star = getattr(astronomy, "star", {}) or {}
    planet = getattr(astronomy, "planet", {}) or {}
    lum = max(float(star.get("luminosity_solar", 1.0)), 1e-8)
    a = max(float(planet.get("semimajor_axis_au", 1.0)), 1e-4)
    teff = max(float(star.get("effective_temperature_k", 5772.0)), 1200.0)
    bolometric = lum / (a * a)
    # Hotter photospheres put a rapidly increasing fraction of their luminosity into
    # short wavelengths.  The exponent is intentionally conservative for a screening
    # proxy and is bounded so exotic stars cannot numerically dominate every reaction.
    uv_fraction_proxy = float(np.clip((teff / 5772.0) ** 5.0, 0.015, 25.0))
    uv = bolometric * uv_fraction_proxy
    # Magnetospheric/radiolytic forcing is enhanced for moons with non-zero tidal
    # forcing because the same giant-planet environments commonly provide energetic
    # particles.  It is a proxy, not a modeled magnetosphere.
    role = str(planet.get("body_role", "planet"))
    interior = getattr(astronomy, "interior", {}) or {}
    tidal = max(float(interior.get("tidal_heating_flux_w_m2", 0.0)), 0.0)
    particle = uv * (1.0 + (1.4 if role == "moon" else 0.0) + min(4.0, 3.0 * math.sqrt(tidal + 1e-12)))
    return {
        "bolometric_relative_earth": float(bolometric),
        "uv_relative_earth_proxy": float(uv),
        "energetic_particle_proxy": float(particle),
        "stellar_effective_temperature_k": float(teff),
    }


def _abundance(comp: Mapping[str, float], key: str) -> float:
    return max(float(comp.get(key, 0.0)), 0.0)


def evaluate_photochemistry(astronomy, composition: Mapping[str, float]) -> dict[str, PhotochemicalProduct]:
    """Evaluate major irradiation-driven planetary chemistry pathways.

    Products are abundance proxies suitable for haze/cloud/albedo/geology coupling.
    They must not be interpreted as kinetic steady-state mole fractions.
    """
    comp = normalize_loose_composition(composition)
    rad = stellar_radiation_indices(astronomy)
    uv = float(np.clip(rad["uv_relative_earth_proxy"], 0.0, 50.0))
    particle = float(np.clip(rad["energetic_particle_proxy"], 0.0, 100.0))
    products: dict[str, PhotochemicalProduct] = {}

    def add(product: str, pathway: str, production: float, abundance: float,
            *, aerosol: bool = False, deposited: bool = False, limiter: float = 0.0) -> None:
        p = float(np.clip(production, 0.0, 1.0))
        a = float(max(abundance, 0.0))
        old = products.get(product)
        if old is not None:
            p = 1.0 - (1.0 - old.production_index) * (1.0 - p)
            a += old.abundance_proxy
        products[product] = PhotochemicalProduct(product, pathway, p, a, aerosol, deposited, float(limiter))

    o2 = _abundance(comp, "O2")
    if o2 > 1e-5 and uv > 1e-4:
        limiter = min(1.0, o2 / 0.20)
        prod = (1.0 - math.exp(-0.8 * math.sqrt(uv))) * limiter
        # Ozone columns are tiny compared with O2; proxy intentionally remains trace.
        add("O3", "O2 photolysis + O + O2 three-body recombination", prod,
            min(2e-3, 8e-5 * prod * math.sqrt(max(o2 / 0.21, 1e-12))), limiter=limiter)

    n2, ch4 = _abundance(comp, "N2"), _abundance(comp, "CH4")
    if n2 > 1e-3 and ch4 > 1e-7 and (uv + particle) > 1e-4:
        limiter = min(1.0, n2 / 0.5, ch4 / 0.03)
        energy = 1.0 - math.exp(-0.30 * (uv + 0.35 * particle))
        base = limiter * energy
        add("C2H6", "CH4 photolysis/recombination", 0.70 * base, ch4 * 0.025 * base, limiter=limiter)
        add("C2H2", "hydrocarbon photochemical chain", 0.62 * base, ch4 * 0.008 * base, deposited=True, limiter=limiter)
        add("HCN", "N2/CH4 ion-neutral and radical chemistry", 0.52 * base, min(n2, ch4) * 0.004 * base, deposited=True, limiter=limiter)
        add("THOLIN", "N2/CH4 UV + energetic-particle organic polymerization", 0.78 * base,
            min(n2, ch4) * 0.015 * base, aerosol=True, deposited=True, limiter=limiter)

    co2 = _abundance(comp, "CO2")
    if co2 > 1e-6 and uv > 1e-4:
        limiter = min(1.0, co2 / 0.1)
        base = limiter * (1.0 - math.exp(-0.20 * uv))
        add("CO", "CO2 photolysis", 0.55 * base, co2 * 0.01 * base, limiter=limiter)
        if o2 < 0.05:
            add("O2", "CO2 photolysis oxygen recombination", 0.20 * base, co2 * 0.002 * base, limiter=limiter)

    so2, h2o = _abundance(comp, "SO2"), _abundance(comp, "H2O")
    oxidant = o2 + 4.0 * float(products.get("O3", PhotochemicalProduct("O3", "", 0, 0, False, False, 0)).abundance_proxy)
    if so2 > 1e-10 and h2o > 1e-10 and uv > 1e-4:
        limiter = min(1.0, so2 / 1e-4, h2o / 1e-4)
        base = limiter * (1.0 - math.exp(-0.25 * uv)) * (0.25 + 0.75 * min(1.0, oxidant / 0.02))
        add("H2SO4", "SO2 oxidation + hydration aerosol chemistry", 0.75 * base,
            min(so2, h2o) * 0.15 * base, aerosol=True, deposited=True, limiter=limiter)

    nh3, h2s = _abundance(comp, "NH3"), _abundance(comp, "H2S")
    if nh3 > 1e-9 and h2s > 1e-9:
        limiter = min(1.0, nh3 / 1e-4, h2s / 1e-4)
        base = limiter * min(1.0, 0.35 + 0.08 * uv)
        add("NH4SH", "NH3 + H2S cloud condensation/reaction", base,
            min(nh3, h2s) * 0.4 * base, aerosol=True, deposited=True, limiter=limiter)

    if h2s > 1e-9 and (uv + particle) > 1e-4:
        limiter = min(1.0, h2s / 1e-4)
        base = limiter * (1.0 - math.exp(-0.12 * (uv + particle)))
        add("S8", "H2S/SO2 photolysis and sulfur polymerization", 0.55 * base,
            h2s * 0.08 * base, aerosol=True, deposited=True, limiter=limiter)

    if h2o > 1e-8 and particle > 0.05:
        limiter = min(1.0, h2o / 1e-3)
        base = limiter * (1.0 - math.exp(-0.08 * particle))
        add("H2O2", "H2O ice/vapor radiolysis and oxidant recombination", 0.40 * base,
            h2o * 2e-4 * base, deposited=True, limiter=limiter)

    return products


def _approx_saturation_bar(key: str, temperature_k: np.ndarray) -> np.ndarray:
    """Screening saturation curve for extended species.

    Supported planetary_physics species use its better reduced-order implementation.
    Other bulk fluids use a one-point Clausius-Clapeyron approximation anchored at
    the normal boiling point.  Aerosol pseudo-species return zero and are handled as
    aerosol-only products by callers.
    """
    t = np.maximum(np.asarray(temperature_k, dtype=np.float64), 1.0)
    if key in SPECIES:
        return np.asarray(saturation_pressure_bar(key, t, backend="builtin"), dtype=np.float64)
    sp = CHEMICALS[key]
    if sp.normal_boiling_k is None or sp.latent_vaporization_kj_mol is None:
        return np.zeros_like(t)
    ln_p = math.log(1.01325) - sp.latent_vaporization_kj_mol * 1000.0 / R_GAS * (
        1.0 / t - 1.0 / float(sp.normal_boiling_k)
    )
    out = np.exp(np.clip(ln_p, -45.0, 12.0))
    if sp.critical_k is not None:
        out = np.where(t >= sp.critical_k, np.inf, out)
    return out


def detect_condensates(
    temperature_c: np.ndarray,
    composition: Mapping[str, float],
    surface_pressure_bar: float,
    *,
    photochemical_products: Mapping[str, PhotochemicalProduct] | None = None,
    minimum_area_fraction: float = 1e-4,
) -> dict[str, CondensateCandidate]:
    """Automatically identify all plausible atmospheric condensates.

    A species is retained when it is present above its significance threshold and at
    least a small fraction of the map is near saturation.  Multiple species may be
    active simultaneously; there is no single-condensable exclusivity.
    """
    t_k = np.asarray(temperature_c, dtype=np.float64) + 273.15
    comp = normalize_loose_composition(composition)
    if photochemical_products:
        for key, product in photochemical_products.items():
            if key in CHEMICALS and product.abundance_proxy > 0:
                comp[key] = comp.get(key, 0.0) + float(product.abundance_proxy)
        norm = sum(comp.values())
        if norm > 0:
            comp = {k: v / norm for k, v in comp.items()}

    pressure = max(float(surface_pressure_bar), 1e-12)
    result: dict[str, CondensateCandidate] = {}
    for key, frac in comp.items():
        if key not in CHEMICALS:
            continue
        sp = CHEMICALS[key]
        if frac < sp.min_significant_fraction:
            continue
        aerosol_only = "aerosol" in sp.roles and sp.normal_boiling_k is None
        partial = pressure * frac
        if aerosol_only:
            product = None if photochemical_products is None else photochemical_products.get(key)
            idx = 0.0 if product is None else float(product.production_index)
            if idx > 1e-5:
                result[key] = CondensateCandidate(
                    key, frac, partial, idx, idx, idx, "aerosol", False, True
                )
            continue
        psat = _approx_saturation_bar(key, t_k)
        finite = np.isfinite(psat) & (psat > 0)
        ratio = np.zeros_like(t_k)
        ratio[finite] = partial / np.maximum(psat[finite], 1e-30)
        supersat = float(np.mean(ratio >= 1.0))
        near = float(np.mean(ratio >= 0.25))
        if near < minimum_area_fraction:
            continue
        if sp.triple_or_melting_k is not None:
            cold = float(np.mean(t_k < sp.triple_or_melting_k))
        else:
            cold = 0.0
        if supersat <= 0 and near > 0:
            phase = "incipient_cloud"
        elif cold > 0.6:
            phase = "solid_precipitation"
        elif cold < 0.2:
            phase = "liquid_precipitation"
        else:
            phase = "mixed_phase_precipitation"
        index = float(np.clip(0.65 * supersat + 0.35 * near, 0.0, 1.0))
        result[key] = CondensateCandidate(
            key, frac, partial, supersat, near, index, phase,
            precipitation_capable=index > 1e-4, aerosol_only=False,
        )
    return result


def chemistry_metadata() -> dict:
    return {
        "species": {k: asdict(v) for k, v in CHEMICALS.items()},
        "model": "screening thermochemistry + abundance-limited photochemical pathway proxies",
        "limitations": (
            "No Gibbs minimization, reaction-network integration, vertical photolysis column, "
            "cloud microphysics, aqueous activity coefficients, or escape-coupled atmospheric evolution."
        ),
    }


__all__ = [
    "ChemicalSpecies", "PhotochemicalProduct", "CondensateCandidate", "CHEMICALS",
    "canonical_chemical", "normalize_loose_composition", "stellar_radiation_indices",
    "evaluate_photochemistry", "detect_condensates", "chemistry_metadata",
]
