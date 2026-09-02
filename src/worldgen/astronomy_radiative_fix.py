from __future__ import annotations

"""Compatibility install for physically correct Bond-albedo equilibrium temperatures.

Historically :mod:`worldgen.astronomy` normalized the zero-albedo 278.5 K reference
by ``0.7`` again when applying Bond albedo.  At Earth albedo that makes the
one-solar-flux equilibrium temperature 278.5 K instead of ~255 K and, when an orbit
is solved from a target temperature, places the planet too far from its star.

The installer keeps the established astronomy implementation and RNG behavior, but
feeds it an algebraically transformed temporary albedo so its legacy formula is
exactly equivalent to the physical ``(1-A)**0.25`` law.  For automatically selected
orbits we also solve the semimajor axis with the physical law before delegating.
This is intentionally a narrow compatibility shim and can be removed once the
literal formula in astronomy.py is migrated in a major-version cleanup.
"""

import copy
import math

import numpy as np

from . import astronomy as _astronomy
from .planetary_physics import composition_greenhouse_temperature_k

_ZERO_ALBEDO_EQUILIBRIUM_K = 278.5
_ORIGINAL_BUILD_ASTRONOMY = _astronomy.build_astronomy
_INSTALLED = False


def physical_equilibrium_temperature_k(
    *, luminosity_solar: float, semimajor_axis_au: float, bond_albedo: float
) -> float:
    """Blackbody equilibrium temperature using the configured Bond albedo."""
    lum = max(float(luminosity_solar), 0.0)
    a = float(semimajor_axis_au)
    albedo = float(bond_albedo)
    if not np.isfinite(a) or a <= 0:
        raise ValueError("semimajor_axis_au must be finite and positive")
    if not np.isfinite(albedo) or not 0.0 <= albedo < 1.0:
        raise ValueError("bond_albedo must be finite and in [0, 1)")
    return float(
        _ZERO_ALBEDO_EQUILIBRIUM_K
        * lum**0.25
        / math.sqrt(a)
        * (1.0 - albedo) ** 0.25
    )


def semimajor_axis_for_target_equilibrium_au(
    *, luminosity_solar: float, target_temperature_k: float, bond_albedo: float
) -> float:
    """Invert :func:`physical_equilibrium_temperature_k` for orbital distance."""
    lum = max(float(luminosity_solar), 0.0)
    target = float(target_temperature_k)
    albedo = float(bond_albedo)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("target_temperature_k must be finite and positive")
    if not np.isfinite(albedo) or not 0.0 <= albedo < 1.0:
        raise ValueError("bond_albedo must be finite and in [0, 1)")
    return float(
        math.sqrt(lum * (1.0 - albedo))
        * (_ZERO_ALBEDO_EQUILIBRIUM_K / target) ** 2
    )


def _target_equilibrium_temperature_k(cfg) -> float:
    target_surface = max(float(cfg.target_mean_surface_c) + 273.15, 100.0)
    if str(cfg.greenhouse_model) == "composition":
        surface_unit, _ = composition_greenhouse_temperature_k(
            1.0, cfg.atmosphere, cfg.atmosphere_pressure_bar
        )
        return target_surface / max(float(surface_unit), 1.0e-6)
    return max(target_surface - float(cfg.greenhouse_k), 100.0)


def _legacy_formula_equivalent_albedo(actual_albedo: float) -> float:
    """Transform A so legacy ((1-A)/0.7)^1/4 equals physical (1-A_actual)^1/4."""
    actual = float(actual_albedo)
    if not np.isfinite(actual) or not 0.0 <= actual < 1.0:
        raise ValueError("configured albedo must be finite and in [0, 1)")
    return float(1.0 - 0.7 * (1.0 - actual))


def build_astronomy_physical(cfg, rng):
    """Delegate to the established astronomy model with corrected radiative geometry."""
    work = copy.deepcopy(cfg)
    actual_albedo = float(cfg.albedo)
    lum = float(_astronomy._mass_luminosity(float(cfg.star_mass_solar)))
    hz_inner = math.sqrt(lum / 1.10)
    hz_outer = math.sqrt(lum / 0.53)

    if cfg.semimajor_axis_au is None:
        target_teq = _target_equilibrium_temperature_k(cfg)
        corrected_a = semimajor_axis_for_target_equilibrium_au(
            luminosity_solar=lum,
            target_temperature_k=target_teq,
            bond_albedo=actual_albedo,
        )
        work.semimajor_axis_au = float(
            np.clip(corrected_a, hz_inner * 1.01, hz_outer * 0.99)
        )

    # astronomy.py's existing formula divides by 0.7.  This transformed temporary
    # value cancels that historical normalization exactly without leaking into the
    # caller's WorldConfig; Atmogen still receives the real configured albedo.
    work.albedo = _legacy_formula_equivalent_albedo(actual_albedo)
    result = _ORIGINAL_BUILD_ASTRONOMY(work, rng)

    a = float(result.planet["semimajor_axis_au"])
    teq = physical_equilibrium_temperature_k(
        luminosity_solar=lum,
        semimajor_axis_au=a,
        bond_albedo=actual_albedo,
    )
    # The delegated result should already equal this to roundoff; write it
    # explicitly and publish provenance so future regressions are obvious.
    result.planet["equilibrium_temperature_k"] = teq
    result.planet["radiative_equilibrium_model"] = "zero_albedo_278.5K_times_(1-A)^0.25"
    result.planet["configured_surface_albedo"] = actual_albedo
    for item in result.planetary_system:
        flux = float(item.get("stellar_flux_earth", 0.0))
        item["equilibrium_temperature_k_approx"] = float(
            _ZERO_ALBEDO_EQUILIBRIUM_K
            * max(flux, 0.0) ** 0.25
            * (1.0 - actual_albedo) ** 0.25
        )
    return result


def install_astronomy_radiative_fix() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _astronomy.build_astronomy = build_astronomy_physical
    _INSTALLED = True


__all__ = [
    "physical_equilibrium_temperature_k",
    "semimajor_axis_for_target_equilibrium_au",
    "build_astronomy_physical",
    "install_astronomy_radiative_fix",
]
