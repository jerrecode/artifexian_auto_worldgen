from __future__ import annotations

"""Composition-triggered visible cloud-deck calibrations.

Some aerosol systems are not well represented by a local production-index tint alone.
Venus is the important current example: once SO2/H2O photochemistry sustains H2SO4,
the planet is covered by a many-optical-depth sulfuric-acid cloud deck.  This module
adds bounded global-deck behavior only when the relevant chemistry exists; it does not
infer a Venus-like deck from CO2 pressure alone.
"""

from typing import Any

import numpy as np

from .planetary_optics import AtmosphereVisibleOptics


_H2SO4_VISIBLE_RGB = np.asarray([0.96, 0.93, 0.79], dtype=np.float32)


def apply_composition_cloud_decks(
    optics: AtmosphereVisibleOptics,
    volatile_cycle: Any | None,
) -> AtmosphereVisibleOptics:
    if volatile_cycle is None:
        return optics
    products = getattr(volatile_cycle, "photochemical_products", {}) or {}
    acid = products.get("H2SO4")
    if acid is None or not bool(getattr(acid, "aerosol", False)):
        return optics
    production = max(float(getattr(acid, "production_index", 0.0)), 0.0)
    if production <= 1.0e-6:
        return optics

    # Observed Venus cloud columns are many tens of optical depths. The procedural
    # visible transfer clips at tau=8 because greater values are numerically
    # indistinguishable in the single-source approximation. A floor near 6 therefore
    # means "surface fully shrouded" without pretending the reduced-order renderer can
    # retrieve the actual vertical optical-depth profile.
    deck_tau = float(np.clip(6.0 + 8.0 * production, 6.0, 8.0))
    haze = np.asarray(optics.haze_optical_depth, dtype=np.float32)
    if haze.ndim == 0:
        haze = np.asarray(float(haze), dtype=np.float32)
        optics.haze_optical_depth = float(max(float(haze), deck_tau))
    else:
        optics.haze_optical_depth = np.maximum(haze, deck_tau).astype(np.float32)

    optics.haze_rgb = (0.20 * np.asarray(optics.haze_rgb, np.float32) + 0.80 * _H2SO4_VISIBLE_RGB).astype(np.float32)
    optics.cloud_rgb = (0.15 * np.asarray(optics.cloud_rgb, np.float32) + 0.85 * _H2SO4_VISIBLE_RGB).astype(np.float32)
    optics.metadata = {
        **optics.metadata,
        "optically_thick_cloud_deck": "H2SO4-H2O aerosol",
        "cloud_deck_effective_visible_tau_floor": deck_tau,
        "cloud_deck_interpretation": "bounded rendering opacity floor; observations indicate many tens of optical depths, while this three-band single-source renderer saturates at tau≈8",
    }
    return optics


__all__ = ["apply_composition_cloud_decks"]
