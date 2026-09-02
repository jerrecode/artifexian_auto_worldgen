from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from worldgen.astronomy_radiative_fix import (
    physical_equilibrium_temperature_k,
    semimajor_axis_for_target_equilibrium_au,
)
from worldgen.drainage import DrainageGraph
from worldgen.grid import SphereGrid
from worldgen.hydrology_reliability import (
    channel_hierarchy_discharge_guarded,
    enforce_hydrology_guardrails,
    flow_directions_multidirection,
    lake_mask_volume_guarded,
    priority_flood_closed_aware,
)


def test_earth_equilibrium_temperature_uses_zero_albedo_reference_correctly() -> None:
    expected = 278.5 * 0.7 ** 0.25
    got = physical_equilibrium_temperature_k(
        luminosity_solar=1.0,
        semimajor_axis_au=1.0,
        bond_albedo=0.30,
    )
    assert got == pytest.approx(expected, rel=1e-12)
    assert 254.0 < got < 256.0


def test_target_equilibrium_orbit_inverts_physical_temperature_law() -> None:
    target = 255.15
    orbit = semimajor_axis_for_target_equilibrium_au(
        luminosity_solar=0.92**4,
        target_temperature_k=target,
        bond_albedo=0.30,
    )
    recovered = physical_equilibrium_temperature_k(
        luminosity_solar=0.92**4,
        semimajor_axis_au=orbit,
        bond_albedo=0.30,
    )
    assert recovered == pytest.approx(target, rel=1e-12)
    assert 0.80 < orbit < 0.90


def test_oceanless_priority_flood_preserves_closed_basins_and_poles() -> None:
    grid = SphereGrid(48, 24)
    yy, xx = np.indices(grid.shape)
    z = 1.0 + 0.002 * yy + 0.001 * np.cos(xx / 3.0)
    z[12, 17] -= 0.25
    ocean = np.zeros(grid.shape, dtype=bool)
    filled = priority_flood_closed_aware(z, ocean, grid)
    np.testing.assert_array_equal(filled, z)


def test_multidirection_flow_keeps_closed_minimum_as_sink() -> None:
    grid = SphereGrid(64, 32)
    lat = np.deg2rad(grid.lat)
    lon = np.deg2rad(grid.lon)
    z = 1.0 + 0.20 * (lat**2 + 0.35 * lon**2)
    center = np.unravel_index(np.argmin(z), z.shape)
    ocean = np.zeros(grid.shape, dtype=bool)
    flow = flow_directions_multidirection(z, ocean, grid).reshape(grid.shape)
    assert flow[center] == -1
    # Strictly downhill receivers guarantee an acyclic graph even without an ocean.
    DrainageGraph.from_receiver(flow, grid.shape)


def test_lake_soft_cap_is_enforced_for_first_giant_component() -> None:
    grid = SphereGrid(64, 32)
    z = np.zeros(grid.shape, dtype=float)
    filled = np.full(grid.shape, 0.100, dtype=float)
    land = np.ones(grid.shape, dtype=bool)
    drainage = np.full(grid.shape, 1.0e5)
    runoff_acc = np.full(grid.shape, 1000.0)
    climate = SimpleNamespace(
        annual_temperature_c=np.full(grid.shape, 10.0),
        annual_precipitation_mm=np.full(grid.shape, 900.0),
    )
    cfg = SimpleNamespace(
        lake_min_depth_m=5.0,
        lake_min_catchment_km2=180.0,
        lake_area_soft_cap_fraction_land=0.022,
    )
    lakes = lake_mask_volume_guarded(
        grid, z, filled, land, drainage, runoff_acc, climate, cfg
    )
    fraction = grid.weighted_fraction(lakes) / grid.weighted_fraction(land)
    # One raster cell of discretization slack is allowed around the requested cap.
    assert 0.0 < fraction < 0.03


def test_dry_planet_channel_hierarchy_cannot_classify_most_land_as_river() -> None:
    grid = SphereGrid(64, 32)
    n = grid.width * grid.height
    receiver = np.arange(n, dtype=np.int64) + 1
    receiver[-1] = -1
    graph = DrainageGraph.from_receiver(receiver, grid.shape)
    cell_area = grid.cell_area_weights * (4.0 * np.pi * grid.radius_km**2)
    drainage = graph.accumulate(cell_area)
    base = SimpleNamespace(
        runoff=np.full(grid.shape, 1.7, dtype=float),
        flow_to=receiver,
        drainage_area_km2=drainage,
        filled_elevation_km=np.linspace(2.0, 0.0, n).reshape(grid.shape),
    )
    water = SimpleNamespace(
        total_runoff_mm_year=np.full(grid.shape, 1.7),
        baseflow_mm_year=np.full(grid.shape, 0.25),
        storminess_index=np.full(grid.shape, 0.20),
    )
    cfg = SimpleNamespace(
        bankfull_storm_multiplier=3.0,
        channel_min_catchment_km2=0.0,
        max_subgrid_drainage_density_km_per_km2=3.2,
        max_resolved_river_cell_fraction_land=0.20,
        min_resolved_channel_discharge_m3_s=0.02,
        min_resolved_stream_discharge_m3_s=0.10,
        min_perennial_stream_discharge_m3_s=1.0,
        min_river_discharge_m3_s=10.0,
        min_major_river_discharge_m3_s=100.0,
    )
    _channel, _cls, rivers, *_rest = channel_hierarchy_discharge_guarded(
        grid, base, water, cfg
    )
    assert np.count_nonzero(rivers) / n <= 0.20 + 1.0 / n


def test_guardrail_rejects_previous_pathological_global_hydrology() -> None:
    result = SimpleNamespace(
        metadata={
            "river_area_fraction_of_land": 0.70,
            "watersheds": {"largest_basin_cells": 990},
        },
        base=SimpleNamespace(metadata={"lake_area_fraction_of_land": 0.63}),
    )
    terrain = SimpleNamespace(land=np.ones((20, 50), dtype=bool))
    cfg = SimpleNamespace()
    with pytest.raises(RuntimeError, match="hydrology reliability guardrail failure"):
        enforce_hydrology_guardrails(result, terrain, cfg)
