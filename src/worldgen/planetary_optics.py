from __future__ import annotations

"""Composition-aware visible optics for planetary surfaces and atmospheres.

The world generator does not run a line-by-line visible radiative-transfer solver, but
it should still obey the most important optical constraints:

* pure hydrocarbon/ammonia/CO2 liquids are not painted terrestrial ocean-blue;
* liquid mixtures derive their visible appearance from the species actually present;
* transparent liquids reveal the bottom in shallow water and darken with optical path;
* atmospheric Rayleigh scattering scales with pressure, gravity, molecular mass and
  composition rather than using an Earth-only blue overlay;
* molecular visible absorption and photochemical/cloud aerosols tint the top-of-
  atmosphere image, including Titan-like organic haze and sulfur-rich cloud decks.

RGB coefficients are deliberately screening-grade broadband coefficients centered near
650/550/450 nm.  They are suitable for physically motivated procedural true-color,
not a substitute for laboratory refractive-index spectra or a multiple-scattering RT
model.
"""

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .planetary_chemistry import normalize_loose_composition


_RGB_WAVELENGTH_NM = np.asarray([650.0, 550.0, 450.0], dtype=np.float64)

# Approximate broadband liquid absorption coefficients [1/m].  H2O preferentially
# removes red light; the cryogenic hydrocarbons and several other pure molecular
# liquids are nearly colorless in the visible and therefore receive much smaller,
# flatter coefficients.  Deep transparent reservoirs still become dark because the
# optical path traverses the liquid twice before bottom light returns to the observer.
_LIQUID_ABSORPTION_RGB_M_INV: dict[str, np.ndarray] = {
    "H2O": np.asarray([0.34, 0.055, 0.016]),
    "CH4": np.asarray([0.0100, 0.0040, 0.0028]),
    "C2H6": np.asarray([0.0065, 0.0038, 0.0028]),
    "C3H8": np.asarray([0.0070, 0.0040, 0.0030]),
    "NH3": np.asarray([0.0070, 0.0055, 0.0045]),
    "CO2": np.asarray([0.0055, 0.0045, 0.0038]),
    "SO2": np.asarray([0.0045, 0.0060, 0.0130]),
    "CH3OH": np.asarray([0.0060, 0.0048, 0.0040]),
}

# Diffuse deep-column source colors.  For nearly colorless liquids these are neutral
# and dim; sky reflection and bottom transmission dominate until the column is deep.
_LIQUID_VOLUME_SCATTER_RGB: dict[str, np.ndarray] = {
    "H2O": np.asarray([0.020, 0.145, 0.340]),
    "CH4": np.asarray([0.045, 0.062, 0.070]),
    "C2H6": np.asarray([0.052, 0.058, 0.060]),
    "C3H8": np.asarray([0.052, 0.058, 0.060]),
    "NH3": np.asarray([0.050, 0.066, 0.070]),
    "CO2": np.asarray([0.048, 0.062, 0.068]),
    "SO2": np.asarray([0.080, 0.078, 0.060]),
    "CH3OH": np.asarray([0.050, 0.063, 0.067]),
}

_LIQUID_REFRACTIVE_INDEX = {
    "H2O": 1.333,
    "CH4": 1.286,
    "C2H6": 1.36,
    "C3H8": 1.34,
    "NH3": 1.33,
    "CO2": 1.20,
    "SO2": 1.34,
    "CH3OH": 1.329,
}

# Relative molecular Rayleigh efficiency at 550 nm.  This folds molecular
# polarizability/depolarization into a compact screening factor; column number density
# is handled separately from pressure, gravity and mean molar mass.
_RAYLEIGH_RELATIVE = {
    "H2": 0.40,
    "He": 0.07,
    "N2": 1.00,
    "O2": 0.92,
    "Ar": 0.64,
    "CO2": 1.45,
    "CH4": 1.18,
    "H2O": 0.86,
    "NH3": 1.30,
    "SO2": 2.10,
    "CO": 0.95,
    "H2S": 1.55,
}

# Broadband gas absorption optical depth per bar of partial pressure.  Visible gas
# absorption is generally weaker than Rayleigh/aerosol extinction; CH4 removes red
# preferentially, while SO2 is strongest toward the blue/near-UV edge.
_GAS_ABSORPTION_TAU_PER_BAR = {
    "CH4": np.asarray([2.8, 0.20, 0.035]),
    "H2O": np.asarray([0.080, 0.018, 0.005]),
    "CO2": np.asarray([0.010, 0.006, 0.004]),
    "SO2": np.asarray([0.025, 0.18, 0.85]),
    "NH3": np.asarray([0.055, 0.025, 0.012]),
    "H2S": np.asarray([0.080, 0.045, 0.025]),
    "O3": np.asarray([0.14, 0.32, 0.11]),
}

_AEROSOL_RGB = {
    "THOLIN": np.asarray([0.76, 0.36, 0.12]),
    "H2SO4": np.asarray([0.96, 0.93, 0.79]),
    "S8": np.asarray([0.94, 0.73, 0.20]),
    "NH4SH": np.asarray([0.90, 0.86, 0.70]),
    "HCN": np.asarray([0.72, 0.68, 0.60]),
    "C2H2": np.asarray([0.68, 0.61, 0.52]),
}

_CLOUD_RGB = {
    "H2O": np.asarray([0.95, 0.97, 0.99]),
    "CH4": np.asarray([0.94, 0.96, 0.97]),
    "C2H6": np.asarray([0.94, 0.95, 0.96]),
    "NH3": np.asarray([0.96, 0.95, 0.89]),
    "CO2": np.asarray([0.96, 0.97, 0.98]),
    "SO2": np.asarray([0.97, 0.93, 0.78]),
}


@dataclass(slots=True)
class AtmosphereVisibleOptics:
    gas_transmittance_rgb: np.ndarray
    rayleigh_scatter_rgb: np.ndarray
    haze_optical_depth: np.ndarray | float
    haze_rgb: np.ndarray
    cloud_rgb: np.ndarray
    metadata: dict


def liquid_volume_fractions(surface_liquids: Any | None) -> dict[str, float]:
    if surface_liquids is None:
        return {}
    volumes: dict[str, float] = {}
    for key, part in getattr(surface_liquids, "partitions", {}).items():
        volume = max(float(getattr(part, "liquid_volume_m3", 0.0)), 0.0)
        if volume > 0.0:
            volumes[str(key)] = volume
    total = sum(volumes.values())
    if total <= 0.0:
        return {}
    return {key: value / total for key, value in volumes.items()}


def _weighted_rgb_property(
    fractions: Mapping[str, float],
    table: Mapping[str, np.ndarray],
    fallback: np.ndarray,
) -> np.ndarray:
    if not fractions:
        return np.asarray(fallback, dtype=np.float64).copy()
    result = np.zeros(3, dtype=np.float64)
    weight = 0.0
    for key, fraction in fractions.items():
        f = max(float(fraction), 0.0)
        if f <= 0.0:
            continue
        result += f * np.asarray(table.get(key, fallback), dtype=np.float64)
        weight += f
    return result / max(weight, 1.0e-30)


def liquid_mixture_optics(surface_liquids: Any | None) -> dict[str, Any]:
    fractions = liquid_volume_fractions(surface_liquids)
    fallback_abs = np.asarray([0.008, 0.006, 0.005], dtype=np.float64)
    fallback_scatter = np.asarray([0.050, 0.060, 0.065], dtype=np.float64)
    absorption = _weighted_rgb_property(fractions, _LIQUID_ABSORPTION_RGB_M_INV, fallback_abs)
    scatter = _weighted_rgb_property(fractions, _LIQUID_VOLUME_SCATTER_RGB, fallback_scatter)
    refractive_index = 1.33
    if fractions:
        refractive_index = sum(
            max(float(frac), 0.0) * float(_LIQUID_REFRACTIVE_INDEX.get(key, 1.33))
            for key, frac in fractions.items()
        )
    r0 = ((refractive_index - 1.0) / max(refractive_index + 1.0, 1.0e-12)) ** 2
    return {
        "volume_fractions": fractions,
        "absorption_rgb_m_inv": absorption,
        "volume_scatter_rgb": scatter,
        "effective_refractive_index": float(refractive_index),
        "normal_incidence_fresnel_reflectance": float(np.clip(r0, 0.0, 0.25)),
    }


def render_surface_liquid_rgb(
    depth_m: np.ndarray,
    bed_rgb: np.ndarray,
    surface_liquids: Any,
    *,
    atmosphere_reflection_rgb: np.ndarray | None = None,
    turbidity: np.ndarray | None = None,
    organic_deposition: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render the actual modeled liquid mixture with Beer-Lambert bottom attenuation."""
    depth = np.maximum(np.asarray(depth_m, dtype=np.float64), 0.0)
    bed = np.clip(np.asarray(bed_rgb, dtype=np.float64), 0.0, 1.0)
    if bed.shape != (*depth.shape, 3):
        raise ValueError("bed_rgb must have shape (*depth_m.shape, 3)")
    optics = liquid_mixture_optics(surface_liquids)
    alpha = np.asarray(optics["absorption_rgb_m_inv"], dtype=np.float64)
    scatter = np.asarray(optics["volume_scatter_rgb"], dtype=np.float64)
    # Two-way path: illumination traverses the liquid to the bottom and reflected
    # bottom light traverses it again to the observer.
    transmission = np.exp(-2.0 * depth[..., None] * alpha[None, None, :])
    column = bed * transmission + scatter[None, None, :] * (1.0 - transmission)

    sky = np.asarray(
        [0.34, 0.48, 0.72] if atmosphere_reflection_rgb is None else atmosphere_reflection_rgb,
        dtype=np.float64,
    )
    fresnel = float(optics["normal_incidence_fresnel_reflectance"])
    # Wind/wave roughness raises the hemispheric reflected fraction above normal-
    # incidence Fresnel while retaining composition dependence through refractive index.
    reflected = float(np.clip(0.035 + 1.8 * fresnel, 0.035, 0.16))
    rgb = column * (1.0 - reflected) + sky[None, None, :] * reflected

    if turbidity is not None:
        tt = np.clip(np.asarray(turbidity, dtype=np.float64), 0.0, 1.0)[..., None]
        suspended = np.asarray([0.34, 0.38, 0.30], dtype=np.float64)
        rgb = rgb * (1.0 - 0.42 * tt) + suspended * (0.42 * tt)
    if organic_deposition is not None:
        oo = np.clip(np.asarray(organic_deposition, dtype=np.float64), 0.0, 1.0)[..., None]
        # Titan-like tholin/organic material dissolved or suspended near shores shifts
        # otherwise colorless hydrocarbon liquid toward dark amber-brown.
        organic = np.asarray([0.19, 0.105, 0.045], dtype=np.float64)
        rgb = rgb * (1.0 - 0.48 * oo) + organic * (0.48 * oo)

    meta = {
        "model": "composition-weighted broadband Beer-Lambert liquid column + bottom reflection + Fresnel/sky reflection",
        "composition_volume_fraction": dict(optics["volume_fractions"]),
        "absorption_rgb_m_inv": [float(x) for x in alpha],
        "effective_refractive_index": float(optics["effective_refractive_index"]),
        "normal_incidence_fresnel_reflectance": fresnel,
        "limitations": "screening broadband RGB coefficients; no wavelength-resolved multiple scattering, dissolved-solute spectrum or wave BRDF",
    }
    return np.clip(rgb, 0.0, 1.0).astype(np.float32), meta


def _composition(astronomy: Any) -> tuple[dict[str, float], float, float, float]:
    atmosphere = getattr(astronomy, "atmosphere", {}) or {}
    fractions = normalize_loose_composition(atmosphere.get("fractions", {}))
    pressure_bar = max(float(atmosphere.get("surface_pressure_bar", 0.0)), 0.0)
    molar_mass = max(float(atmosphere.get("mean_molar_mass_g_mol", 28.97)), 0.1)
    planet = getattr(astronomy, "planet", {}) or {}
    gravity_g = max(float(planet.get("surface_gravity_g", 1.0)), 0.01)
    return fractions, pressure_bar, molar_mass, gravity_g


def atmosphere_visible_optics(
    astronomy: Any,
    volatile_cycle: Any | None = None,
    *,
    shape: tuple[int, int] | None = None,
) -> AtmosphereVisibleOptics:
    fractions, pressure_bar, molar_mass, gravity_g = _composition(astronomy)
    rayleigh_mix = sum(
        float(frac) * float(_RAYLEIGH_RELATIVE.get(key, 1.0))
        for key, frac in fractions.items()
    ) if fractions else 1.0
    # Number column relative to Earth: P/g divided by mean molecular mass.
    number_column_rel = (
        pressure_bar / 1.01325 * (1.0 / gravity_g) * (28.97 / molar_mass)
    ) if pressure_bar > 0.0 else 0.0
    tau550 = 0.085 * number_column_rel * rayleigh_mix
    rayleigh_tau = tau550 * (550.0 / _RGB_WAVELENGTH_NM) ** 4

    absorption_tau = np.zeros(3, dtype=np.float64)
    for key, fraction in fractions.items():
        coeff = _GAS_ABSORPTION_TAU_PER_BAR.get(key)
        if coeff is not None:
            absorption_tau += pressure_bar * float(fraction) * coeff

    # A representative two-way slant factor for disk-averaged reflected light.  It is
    # intentionally bounded rather than pretending to solve viewing geometry per cell.
    gas_trans = np.exp(-1.55 * (rayleigh_tau + absorption_tau))
    scatter_raw = 1.0 - np.exp(-rayleigh_tau)
    if float(np.max(scatter_raw)) > 1.0e-12:
        scatter_rgb = scatter_raw / float(np.max(scatter_raw))
    else:
        scatter_rgb = np.asarray([0.55, 0.72, 1.0], dtype=np.float64)
    # Keep Rayleigh source blue-biased even for optically thick columns where all three
    # channels asymptotically scatter strongly.
    scatter_rgb *= np.asarray([0.72, 0.86, 1.0], dtype=np.float64)
    scatter_rgb = np.clip(scatter_rgb, 0.0, 1.0)

    haze_rgb = np.asarray([0.82, 0.84, 0.84], dtype=np.float64)
    haze_strength = 0.0
    haze_map: np.ndarray | float = 0.0
    product_colors: list[tuple[float, np.ndarray]] = []
    if volatile_cycle is not None:
        for key, product in getattr(volatile_cycle, "photochemical_products", {}).items():
            if not bool(getattr(product, "aerosol", False)):
                continue
            strength = max(float(getattr(product, "production_index", 0.0)), 0.0)
            if strength <= 0.0:
                continue
            product_colors.append((strength, _AEROSOL_RGB.get(str(key), np.asarray([0.78, 0.76, 0.70]))))
            haze_strength += strength
        if product_colors:
            total = sum(weight for weight, _ in product_colors)
            haze_rgb = sum(weight * color for weight, color in product_colors) / max(total, 1.0e-30)
        proxy = np.asarray(getattr(volatile_cycle, "aerosol_optical_depth_proxy", 0.0), dtype=np.float64)
        if proxy.ndim == 2:
            haze_map = np.maximum(0.0, proxy) * (0.45 + 3.8 * min(haze_strength, 2.0))
            # Titan-like tholin haze is globally persistent even where the normalized
            # production map is locally weak.
            tholin = getattr(volatile_cycle, "photochemical_products", {}).get("THOLIN")
            if tholin is not None:
                p = max(float(getattr(tholin, "production_index", 0.0)), 0.0)
                haze_map = haze_map + min(1.8, 0.75 * p)
        elif haze_strength > 0.0:
            haze_map = min(4.0, 1.2 * haze_strength)

    # Determine condensate/cloud color from actual precipitating species rather than
    # assuming every cloud is liquid/ice H2O.
    cloud_rgb = np.asarray([0.95, 0.97, 0.99], dtype=np.float64)
    cloud_weights: list[tuple[float, np.ndarray]] = []
    if volatile_cycle is not None:
        for key, candidate in getattr(volatile_cycle, "condensates", {}).items():
            if not bool(getattr(candidate, "precipitation_capable", False)):
                continue
            weight = max(float(getattr(candidate, "condensation_index", 0.0)), 0.0)
            weight *= max(float(getattr(candidate, "atmospheric_fraction", 0.0)), 1.0e-5) ** 0.5
            if weight > 0.0:
                cloud_weights.append((weight, _CLOUD_RGB.get(str(key), cloud_rgb)))
    if cloud_weights:
        total = sum(weight for weight, _ in cloud_weights)
        cloud_rgb = sum(weight * color for weight, color in cloud_weights) / max(total, 1.0e-30)
    if any(str(getattr(p, "product", key)) == "H2SO4" for key, p in getattr(volatile_cycle, "photochemical_products", {}).items()) if volatile_cycle is not None else False:
        cloud_rgb = 0.72 * cloud_rgb + 0.28 * _AEROSOL_RGB["H2SO4"]

    if shape is not None and np.isscalar(haze_map):
        haze_map = np.full(shape, float(haze_map), dtype=np.float32)

    meta = {
        "model": "pressure/gravity/molecular-mass scaled Rayleigh + broadband molecular absorption + photochemical aerosol extinction",
        "atmosphere_fractions": dict(fractions),
        "surface_pressure_bar": pressure_bar,
        "mean_molar_mass_g_mol": molar_mass,
        "surface_gravity_g": gravity_g,
        "rayleigh_tau_rgb": [float(x) for x in rayleigh_tau],
        "molecular_absorption_tau_rgb": [float(x) for x in absorption_tau],
        "gas_transmittance_rgb": [float(x) for x in gas_trans],
        "rayleigh_scatter_rgb": [float(x) for x in scatter_rgb],
        "haze_rgb": [float(x) for x in haze_rgb],
        "cloud_rgb": [float(x) for x in cloud_rgb],
        "aerosol_production_strength": float(haze_strength),
        "limitations": "three broadband visible channels; no wavelength-resolved multiple scattering, phase functions, polarization or limb geometry",
    }
    return AtmosphereVisibleOptics(
        gas_transmittance_rgb=np.asarray(gas_trans, dtype=np.float32),
        rayleigh_scatter_rgb=np.asarray(scatter_rgb, dtype=np.float32),
        haze_optical_depth=np.asarray(haze_map, dtype=np.float32) if not np.isscalar(haze_map) else float(haze_map),
        haze_rgb=np.asarray(haze_rgb, dtype=np.float32),
        cloud_rgb=np.asarray(cloud_rgb, dtype=np.float32),
        metadata=meta,
    )


def composite_top_of_atmosphere(
    surface_rgb: np.ndarray,
    cloud_fraction: np.ndarray,
    optics: AtmosphereVisibleOptics,
    *,
    cloud_max_optical_opacity: float = 0.88,
) -> np.ndarray:
    """Apply gas scattering/absorption, condensate clouds and upper aerosol haze."""
    rgb = np.clip(np.asarray(surface_rgb, dtype=np.float64), 0.0, 1.0)
    cloud = np.clip(np.asarray(cloud_fraction, dtype=np.float64), 0.0, 1.0)
    trans = np.asarray(optics.gas_transmittance_rgb, dtype=np.float64)
    scatter = np.asarray(optics.rayleigh_scatter_rgb, dtype=np.float64)
    # Rayleigh-scattered source brightness follows the amount removed from direct
    # transmission while retaining spectral color.
    gas_source = scatter * np.clip(1.0 - trans, 0.0, 1.0) * 0.72
    out = rgb * trans[None, None, :] + gas_source[None, None, :]

    c = np.clip(cloud * float(cloud_max_optical_opacity), 0.0, 0.97)[..., None]
    cloud_rgb = np.asarray(optics.cloud_rgb, dtype=np.float64)
    out = out * (1.0 - c) + cloud_rgb[None, None, :] * c

    haze_tau = np.asarray(optics.haze_optical_depth, dtype=np.float64)
    if haze_tau.ndim == 0:
        haze_tau = np.full(cloud.shape, float(haze_tau), dtype=np.float64)
    haze_trans = np.exp(-np.clip(haze_tau, 0.0, 8.0))[..., None]
    haze_rgb = np.asarray(optics.haze_rgb, dtype=np.float64)
    # Single-scattering source approximation. Strong haze can therefore obscure the
    # surface instead of merely applying a color filter.
    out = out * haze_trans + haze_rgb[None, None, :] * (1.0 - haze_trans) * 0.88
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def ground_liquid_humidity_index(
    hydrology: Any,
    land: np.ndarray,
    condensate_forcing: Any | None = None,
) -> np.ndarray:
    """Generic pore-liquid humidity: H2O, CH4, C2H6, NH3, or their active mixture.

    Final soil storage is the persistent component; current annual *liquid* condensate
    input provides a direct precipitation response.  Both are geometric liquid-depth
    quantities, so the index is independent of which molecular species occupies the
    pore volume.
    """
    lf = np.asarray(land, dtype=bool)
    storage = np.maximum(
        np.asarray(getattr(hydrology, "soil_water_storage_mm", np.zeros(lf.shape)), dtype=np.float64),
        0.0,
    )
    storage_sat = storage / (storage + 70.0)
    if condensate_forcing is not None:
        annual_liquid = np.maximum(
            np.asarray(condensate_forcing.annual_liquid_input_mm, dtype=np.float64), 0.0
        )
        precip_response = annual_liquid / (annual_liquid + 320.0)
    else:
        runoff = np.maximum(np.asarray(getattr(hydrology, "runoff", np.zeros(lf.shape)), dtype=np.float64), 0.0)
        precip_response = runoff / (runoff + 280.0)
    index = np.clip(0.76 * storage_sat + 0.24 * precip_response, 0.0, 1.0)
    return (index * lf).astype(np.float32)


__all__ = [
    "AtmosphereVisibleOptics",
    "atmosphere_visible_optics",
    "composite_top_of_atmosphere",
    "ground_liquid_humidity_index",
    "liquid_mixture_optics",
    "liquid_volume_fractions",
    "render_surface_liquid_rgb",
]
