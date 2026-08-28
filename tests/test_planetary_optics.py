from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from worldgen import tectonics
from worldgen.config import TectonicsConfig
from worldgen.grid import SphereGrid
from worldgen.planetary_optics import (
    atmosphere_visible_optics,
    composite_top_of_atmosphere,
    ground_liquid_humidity_index,
    liquid_mixture_optics,
    render_surface_liquid_rgb,
)


def _part(species: str, volume: float):
    return SimpleNamespace(species=species, liquid_volume_m3=volume)


def test_hydrocarbon_liquid_is_not_rendered_as_terrestrial_blue_water():
    methane = SimpleNamespace(partitions={"CH4": _part("CH4", 7.0), "C2H6": _part("C2H6", 3.0)})
    water = SimpleNamespace(partitions={"H2O": _part("H2O", 10.0)})
    mo = liquid_mixture_optics(methane)
    wo = liquid_mixture_optics(water)
    assert mo["volume_fractions"] == {"CH4": 0.7, "C2H6": 0.3}
    assert wo["absorption_rgb_m_inv"][0] > 20.0 * mo["absorption_rgb_m_inv"][0]

    depth = np.full((2, 3), 80.0)
    bed = np.full((2, 3, 3), [0.45, 0.38, 0.30], dtype=float)
    methane_rgb, _ = render_surface_liquid_rgb(
        depth, bed, methane, atmosphere_reflection_rgb=np.asarray([0.55, 0.66, 0.82])
    )
    water_rgb, _ = render_surface_liquid_rgb(
        depth, bed, water, atmosphere_reflection_rgb=np.asarray([0.55, 0.66, 0.82])
    )
    # Deep water is strongly blue; pure hydrocarbon liquid is comparatively neutral.
    water_blue_excess = float(np.mean(water_rgb[..., 2] - water_rgb[..., 0]))
    methane_blue_excess = float(np.mean(methane_rgb[..., 2] - methane_rgb[..., 0]))
    assert water_blue_excess > methane_blue_excess + 0.08


def test_titan_like_tholin_haze_tints_top_of_atmosphere_orange_and_absorbs_red_gas_band():
    astro = SimpleNamespace(
        atmosphere={
            "fractions": {"N2": 0.95, "CH4": 0.05},
            "surface_pressure_bar": 1.47,
            "mean_molar_mass_g_mol": 27.4,
        },
        planet={"surface_gravity_g": 0.14},
    )
    tholin = SimpleNamespace(product="THOLIN", production_index=1.0, aerosol=True)
    methane_cloud = SimpleNamespace(
        precipitation_capable=True, condensation_index=0.8, atmospheric_fraction=0.05
    )
    volatile = SimpleNamespace(
        photochemical_products={"THOLIN": tholin},
        aerosol_optical_depth_proxy=np.ones((3, 4), dtype=np.float32),
        condensates={"CH4": methane_cloud},
    )
    optics = atmosphere_visible_optics(astro, volatile, shape=(3, 4))
    assert optics.haze_rgb[0] > optics.haze_rgb[1] > optics.haze_rgb[2]
    assert optics.gas_transmittance_rgb[0] < optics.gas_transmittance_rgb[2]
    assert np.min(optics.haze_optical_depth) > 0.5

    surface = np.full((3, 4, 3), 0.25, dtype=np.float32)
    rendered = composite_top_of_atmosphere(surface, np.zeros((3, 4)), optics)
    mean = rendered.mean(axis=(0, 1))
    assert mean[0] > mean[1] > mean[2]


def test_ground_liquid_humidity_increases_with_liquid_precipitation_for_same_storage():
    land = np.ones((2, 2), dtype=bool)
    hydro = SimpleNamespace(
        soil_water_storage_mm=np.full((2, 2), 25.0),
        runoff=np.zeros((2, 2)),
    )
    dry = SimpleNamespace(annual_liquid_input_mm=np.full((2, 2), 5.0))
    wet = SimpleNamespace(annual_liquid_input_mm=np.full((2, 2), 900.0))
    a = ground_liquid_humidity_index(hydro, land, dry)
    b = ground_liquid_humidity_index(hydro, land, wet)
    assert np.all(b > a)
    assert np.all((b >= 0.0) & (b <= 1.0))


def test_canonical_tectonic_initializer_accepts_inactive_titan_speed_regime():
    cfg = TectonicsConfig(max_plate_speed_cm_yr=0.03)
    grid = SphereGrid(64, 32, radius_km=2574.7)
    rng = np.random.default_rng(1234)
    centers = tectonics.random_unit_vectors(rng, cfg.plate_count)
    _sub, _parent, desired, _continental = tectonics._initial_subplates(centers, cfg, grid, rng)
    speeds_cm_yr = np.linalg.norm(desired, axis=1) * grid.radius_km / 10.0
    assert np.all(np.isfinite(speeds_cm_yr))
    assert float(np.max(speeds_cm_yr)) < 0.05
    assert float(np.max(speeds_cm_yr)) > 0.0
