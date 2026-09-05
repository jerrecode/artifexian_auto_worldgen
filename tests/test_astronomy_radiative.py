from __future__ import annotations

import pytest

from worldgen.astronomy import (
    ZERO_ALBEDO_EQUILIBRIUM_K,
    equilibrium_temperature_k,
    semimajor_axis_for_equilibrium_temperature_au,
)


def test_earth_bond_albedo_equilibrium_is_about_255_k():
    expected = ZERO_ALBEDO_EQUILIBRIUM_K * 0.70 ** 0.25
    got = equilibrium_temperature_k(
        luminosity_solar=1.0,
        semimajor_axis_au=1.0,
        bond_albedo=0.30,
    )
    assert got == pytest.approx(expected, rel=1e-12)
    assert 254.0 < got < 256.0


def test_zero_albedo_reference_is_preserved_at_one_au():
    assert equilibrium_temperature_k(
        luminosity_solar=1.0,
        semimajor_axis_au=1.0,
        bond_albedo=0.0,
    ) == pytest.approx(ZERO_ALBEDO_EQUILIBRIUM_K, rel=1e-12)


def test_orbit_inversion_round_trips_radiative_equilibrium():
    target = 255.15
    luminosity = 0.92 ** 4
    orbit = semimajor_axis_for_equilibrium_temperature_au(
        luminosity_solar=luminosity,
        target_temperature_k=target,
        bond_albedo=0.30,
    )
    recovered = equilibrium_temperature_k(
        luminosity_solar=luminosity,
        semimajor_axis_au=orbit,
        bond_albedo=0.30,
    )
    assert recovered == pytest.approx(target, rel=1e-12)
    assert orbit > 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"luminosity_solar": -1.0, "semimajor_axis_au": 1.0, "bond_albedo": 0.3},
        {"luminosity_solar": 1.0, "semimajor_axis_au": 0.0, "bond_albedo": 0.3},
        {"luminosity_solar": 1.0, "semimajor_axis_au": 1.0, "bond_albedo": 1.0},
    ],
)
def test_equilibrium_temperature_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        equilibrium_temperature_k(**kwargs)
