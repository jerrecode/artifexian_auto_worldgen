from __future__ import annotations

import math

import numpy as np
import pytest

from worldgen.planetary_physics import (
    atmosphere_diagnostics,
    composition_greenhouse_temperature_k,
    geological_activity_regime,
    phase_at,
    phase_code_grid,
    saturation_pressure_bar,
    select_active_condensible,
    tidal_heating_flux_w_m2,
    tidal_heating_power_w,
    greenhouse_optical_depth,
)


def test_water_phase_changes_with_temperature_at_one_bar():
    assert phase_at("H2O", 288.0, 1.0, backend="builtin") == "liquid"
    assert phase_at("H2O", 250.0, 1.0, backend="builtin") == "solid"
    assert phase_at("H2O", 400.0, 1.0, backend="builtin") == "gas"


def test_titan_temperature_pressure_supports_liquid_methane_and_ethane():
    assert phase_at("CH4", 94.0, 1.5, backend="builtin") == "liquid"
    assert phase_at("C2H6", 94.0, 1.5, backend="builtin") == "liquid"
    active = select_active_condensible(
        {"N2": 0.95, "CH4": 0.05},
        {"CH4": 0.7, "C2H6": 0.3},
        94.0,
        1.5,
    )
    assert active in {"CH4", "C2H6"}


def test_mars_like_co2_is_gaseous_at_mean_conditions_but_can_frost_when_cold():
    assert phase_at("CO2", 210.0, 0.006, backend="builtin") == "gas"
    assert phase_at("CO2", 145.0, 0.006, backend="builtin") == "solid"


def test_phase_grid_is_vectorized_and_uses_stable_codes():
    t = np.array([[250.0, 288.0, 400.0]], dtype=float)
    code = phase_code_grid("H2O", t, 1.0)
    assert code.dtype == np.uint8
    np.testing.assert_array_equal(code, np.array([[2, 1, 0]], dtype=np.uint8))


def test_saturation_pressure_increases_monotonically_below_critical_point():
    p = saturation_pressure_bar("CH4", np.array([95.0, 105.0, 115.0]), backend="builtin")
    assert np.all(np.diff(p) > 0)


def test_composition_greenhouse_responds_to_pressure_and_composition():
    earth_t, earth_terms = composition_greenhouse_temperature_k(
        255.0, {"N2": 0.78, "O2": 0.21, "CO2": 4.2e-4, "H2O": 0.00958}, 1.0
    )
    mars_t, _ = composition_greenhouse_temperature_k(210.0, {"CO2": 0.95, "N2": 0.027, "Ar": 0.023}, 0.006)
    venus_t, venus_terms = composition_greenhouse_temperature_k(232.0, {"CO2": 0.965, "N2": 0.035}, 92.0)
    assert 270.0 < earth_t < 310.0
    assert 210.0 <= mars_t < 240.0
    assert venus_t > 600.0
    assert venus_terms["CO2"] > earth_terms["CO2"]


def test_atmosphere_diagnostics_capture_scale_height_and_column_mass():
    d = atmosphere_diagnostics(
        composition={"N2": 0.78, "O2": 0.21, "CO2": 0.01},
        pressure_bar=1.0,
        temperature_k=288.0,
        gravity_m_s2=9.81,
    )
    assert 7.0 < d["scale_height_km_approx"] < 10.0
    assert 9_000.0 < d["atmospheric_column_mass_kg_m2"] < 12_000.0


def test_tidal_heating_scales_with_eccentricity_squared():
    base = dict(
        satellite_radius_earth=0.25,
        primary_mass_earth=317.8,
        orbit_km=421_700.0,
        love_number_k2=0.3,
        quality_factor_q=100.0,
    )
    a = tidal_heating_flux_w_m2(**base, eccentricity=0.002)
    b = tidal_heating_flux_w_m2(**base, eccentricity=0.004)
    assert a > 0
    assert math.isclose(b / a, 4.0, rel_tol=1e-12)
    assert geological_activity_regime(0.005) == "geologically_inactive"
    assert geological_activity_regime(0.09) == "active"
    assert geological_activity_regime(1.5) == "extreme_tidally_active"



@pytest.mark.parametrize("temperature", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_saturation_pressure_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature.*finite and positive"):
        saturation_pressure_bar("H2O", temperature, backend="builtin")


@pytest.mark.parametrize(
    "temperature,pressure",
    [
        (np.nan, 1.0),
        (np.inf, 1.0),
        (0.0, 1.0),
        (288.0, np.nan),
        (288.0, np.inf),
        (288.0, -1.0),
    ],
)
def test_phase_at_rejects_invalid_state(temperature, pressure):
    with pytest.raises(ValueError, match="temperature|pressure"):
        phase_at("H2O", temperature, pressure, backend="builtin")


def test_phase_grid_rejects_nonfinite_temperature():
    with pytest.raises(ValueError, match="temperature.*finite and positive"):
        phase_code_grid("H2O", np.asarray([[250.0, np.nan]]), 1.0)


@pytest.mark.parametrize("pressure", [np.nan, np.inf, -1.0, 0.0])
def test_greenhouse_rejects_invalid_pressure(pressure):
    with pytest.raises(ValueError, match="pressure_bar.*finite and positive"):
        greenhouse_optical_depth({"N2": 0.8, "CO2": 0.2}, pressure)


@pytest.mark.parametrize("temperature", [np.nan, np.inf, -1.0, 0.0])
def test_composition_greenhouse_rejects_invalid_equilibrium_temperature(temperature):
    with pytest.raises(ValueError, match="equilibrium_temperature_k.*finite and positive"):
        composition_greenhouse_temperature_k(
            temperature, {"N2": 0.8, "CO2": 0.2}, 1.0
        )


@pytest.mark.parametrize("inventory", [{"H2O": np.nan}, {"H2O": np.inf}, {"H2O": -1.0}])
def test_condensible_selection_rejects_invalid_surface_inventory(inventory):
    with pytest.raises(ValueError, match="surface volatile.*finite and non-negative"):
        select_active_condensible(
            {"N2": 1.0}, inventory, 280.0, 1.0
        )


@pytest.mark.parametrize(
    "override",
    [
        {"satellite_radius_earth": np.nan},
        {"satellite_radius_earth": -1.0},
        {"primary_mass_earth": np.inf},
        {"primary_mass_earth": 0.0},
        {"orbit_km": np.nan},
        {"orbit_km": 0.0},
        {"eccentricity": np.nan},
        {"eccentricity": -0.1},
        {"love_number_k2": np.nan},
        {"love_number_k2": -0.1},
        {"quality_factor_q": np.inf},
        {"quality_factor_q": 0.0},
    ],
)
def test_tidal_heating_rejects_invalid_physical_inputs(override):
    kwargs = dict(
        satellite_radius_earth=0.25,
        primary_mass_earth=317.8,
        orbit_km=421_700.0,
        eccentricity=0.004,
        love_number_k2=0.3,
        quality_factor_q=100.0,
    )
    kwargs.update(override)
    with pytest.raises(ValueError, match="tidal heating"):
        tidal_heating_power_w(**kwargs)


@pytest.mark.parametrize(
    "override",
    [
        {"pressure_bar": np.nan},
        {"pressure_bar": 0.0},
        {"temperature_k": np.inf},
        {"temperature_k": 0.0},
        {"gravity_m_s2": np.nan},
        {"gravity_m_s2": 0.0},
    ],
)
def test_atmosphere_diagnostics_rejects_invalid_state(override):
    kwargs = dict(
        composition={"N2": 0.8, "O2": 0.2},
        pressure_bar=1.0,
        temperature_k=288.0,
        gravity_m_s2=9.81,
    )
    kwargs.update(override)
    with pytest.raises(ValueError, match="pressure|temperature|gravity"):
        atmosphere_diagnostics(**kwargs)



@pytest.mark.parametrize("heat_flux", [np.nan, np.inf, -np.inf, -0.01])
def test_geological_activity_rejects_invalid_heat_flux(heat_flux):
    with pytest.raises(ValueError, match="internal heat flux.*finite and non-negative"):
        geological_activity_regime(heat_flux)
