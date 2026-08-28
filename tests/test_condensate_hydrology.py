from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from worldgen.condensate_hydrology import (
    build_condensate_hydrology_forcing,
    climate_for_hydrology,
)
from worldgen.multicondensate_water_balance import build_multicondensate_water_balance


def _titan_like_climate(shape=(6, 12)):
    h, w = shape
    # Titan-like conditions deliberately place both methane and ethane in a mobile
    # condensed regime while retaining enough seasonality to exercise phase forcing.
    seasonal = np.asarray([-2.0, -1.2, -0.4, 0.4, 1.2, 2.0, 1.2, 0.4, -0.4, -1.2, -2.0, -2.4])
    base_c = 94.0 - 273.15
    temp = np.empty((12, h, w), dtype=np.float32)
    precip = np.empty_like(temp)
    for month in range(12):
        temp[month] = base_c + seasonal[month]
        precip[month] = 8.0 + 2.5 * np.cos(2.0 * np.pi * month / 12.0)
    humidity = np.full_like(temp, 0.72)
    return SimpleNamespace(
        temperature_c=temp,
        precipitation_mm=precip,
        annual_temperature_c=temp.mean(axis=0).astype(np.float32),
        annual_precipitation_mm=precip.sum(axis=0).astype(np.float32),
        humidity_proxy=humidity,
        snow_fraction=np.zeros((h, w), dtype=np.float32),
        metadata={"active_condensible_species": "CH4"},
    )


def _titan_like_astronomy():
    return SimpleNamespace(
        atmosphere={
            "surface_pressure_bar": 1.47,
            "fractions": {"N2": 0.94, "CH4": 0.045, "C2H6": 0.015},
        }
    )


def test_condensate_partition_conserves_reference_mass_and_uses_species_density():
    climate = _titan_like_climate()
    forcing = build_condensate_hydrology_forcing(
        _titan_like_astronomy(),
        climate,
        surface_volatiles={"CH4": 0.7, "C2H6": 0.3},
    )
    assert "CH4" in forcing.species_monthly_mass_kg_m2
    assert "C2H6" in forcing.species_monthly_mass_kg_m2
    reference = np.asarray(forcing.monthly_reference_mass_kg_m2, float)
    partitioned = sum(
        np.asarray(value, float)
        for value in forcing.species_monthly_mass_kg_m2.values()
    )
    np.testing.assert_allclose(partitioned, reference, rtol=2e-7, atol=2e-6)
    assert forcing.metadata["mass_conservation_relative_l1_residual"] < 2e-7

    # Some reference methane mass is assigned to denser ethane. The resulting total
    # volume-depth therefore must not be identical to the original methane depth.
    total_depth = np.asarray(forcing.monthly_total_precipitation_depth_mm, float)
    assert np.max(np.abs(total_depth - climate.precipitation_mm)) > 1e-4
    assert np.all(total_depth >= 0.0)
    np.testing.assert_allclose(
        forcing.monthly_liquid_input_mm + forcing.monthly_solid_input_mm,
        forcing.monthly_total_precipitation_depth_mm,
        rtol=2e-7,
        atol=2e-6,
    )


def test_multicondensate_bucket_closes_water_volume_and_exposes_climate_view():
    climate = _titan_like_climate()
    view, forcing = climate_for_hydrology(
        _titan_like_astronomy(),
        climate,
        surface_volatiles={"CH4": 0.7, "C2H6": 0.3},
    )
    np.testing.assert_array_equal(
        view.precipitation_mm,
        forcing.monthly_total_precipitation_depth_mm,
    )
    assert view.annual_precipitation_mm.shape == climate.annual_precipitation_mm.shape

    land = np.ones(climate.annual_temperature_c.shape, dtype=bool)
    cfg = SimpleNamespace(
        soil_storage_multiplier=1.0,
        groundwater_recession_fraction_month=0.065,
        storm_runoff_strength=1.0,
        water_balance_spinup_years=4,
    )
    result = build_multicondensate_water_balance(view, land, None, cfg)
    assert result.metadata["multicomponent_condensate_hydrology"] is True
    assert set(result.metadata["active_hydrologic_species"]) >= {"CH4", "C2H6"}
    assert result.metadata["condensate_mass_partition_relative_l1_residual"] < 2e-7
    assert result.metadata["max_absolute_water_balance_residual_mm_year"] < 0.02
    assert np.all(result.total_runoff_mm_year >= 0.0)
    assert np.all(result.groundwater_storage_mm >= 0.0)
    assert np.all(result.snowpack_mm >= 0.0)


def test_reference_mass_partition_degrades_safely_to_one_condensable():
    climate = _titan_like_climate()
    astronomy = SimpleNamespace(
        atmosphere={"surface_pressure_bar": 1.0, "fractions": {"N2": 0.99, "CH4": 0.01}}
    )
    forcing = build_condensate_hydrology_forcing(
        astronomy,
        climate,
        surface_volatiles={"CH4": 1.0},
    )
    assert "CH4" in forcing.species_monthly_mass_kg_m2
    reference = np.asarray(forcing.monthly_reference_mass_kg_m2, float)
    partitioned = sum(np.asarray(v, float) for v in forcing.species_monthly_mass_kg_m2.values())
    np.testing.assert_allclose(partitioned, reference, rtol=2e-7, atol=2e-6)
