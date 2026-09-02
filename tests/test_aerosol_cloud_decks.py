from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from worldgen.aerosol_cloud_decks import apply_composition_cloud_decks
from worldgen.planetary_optics import atmosphere_visible_optics, composite_top_of_atmosphere


def test_h2so4_photochemistry_creates_optically_thick_cream_cloud_deck():
    astronomy = SimpleNamespace(
        atmosphere={
            "fractions": {"CO2": 0.964775, "N2": 0.035, "SO2": 1.5e-4, "H2O": 2.0e-5},
            "surface_pressure_bar": 92.0,
            "mean_molar_mass_g_mol": 43.4,
        },
        planet={"surface_gravity_g": 0.904},
    )
    acid = SimpleNamespace(product="H2SO4", production_index=0.72, aerosol=True)
    volatile = SimpleNamespace(
        photochemical_products={"H2SO4": acid},
        aerosol_optical_depth_proxy=np.full((4, 8), 0.02, dtype=np.float32),
        condensates={},
    )
    optics = atmosphere_visible_optics(astronomy, volatile, shape=(4, 8))
    optics = apply_composition_cloud_decks(optics, volatile)

    tau = np.asarray(optics.haze_optical_depth, dtype=float)
    assert float(np.min(tau)) >= 6.0
    assert optics.metadata["optically_thick_cloud_deck"] == "H2SO4-H2O aerosol"
    assert 6.0 <= float(optics.metadata["cloud_deck_effective_visible_tau_floor"]) <= 8.0
    assert float(optics.haze_rgb[0]) > float(optics.haze_rgb[2])

    # The deck should overwhelm large differences in the underlying visible surface.
    black = np.zeros((4, 8, 3), dtype=np.float32)
    white = np.ones((4, 8, 3), dtype=np.float32)
    no_cloud = np.zeros((4, 8), dtype=np.float32)
    black_toa = composite_top_of_atmosphere(black, no_cloud, optics)
    white_toa = composite_top_of_atmosphere(white, no_cloud, optics)
    assert float(np.mean(np.abs(black_toa - white_toa))) < 0.01
    mean = np.mean(black_toa, axis=(0, 1))
    assert mean[0] > mean[1] > mean[2]


def test_cloud_deck_calibration_is_inert_without_h2so4():
    astronomy = SimpleNamespace(
        atmosphere={
            "fractions": {"N2": 0.95, "CH4": 0.05},
            "surface_pressure_bar": 1.47,
            "mean_molar_mass_g_mol": 27.4,
        },
        planet={"surface_gravity_g": 0.14},
    )
    tholin = SimpleNamespace(product="THOLIN", production_index=0.7, aerosol=True)
    volatile = SimpleNamespace(
        photochemical_products={"THOLIN": tholin},
        aerosol_optical_depth_proxy=np.full((3, 6), 0.3, dtype=np.float32),
        condensates={},
    )
    optics = atmosphere_visible_optics(astronomy, volatile, shape=(3, 6))
    before_tau = np.asarray(optics.haze_optical_depth).copy()
    before_rgb = np.asarray(optics.haze_rgb).copy()
    returned = apply_composition_cloud_decks(optics, volatile)
    np.testing.assert_array_equal(returned.haze_optical_depth, before_tau)
    np.testing.assert_array_equal(returned.haze_rgb, before_rgb)
    assert "optically_thick_cloud_deck" not in returned.metadata
