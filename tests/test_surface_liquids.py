from __future__ import annotations

import math
import numpy as np

from worldgen.grid import SphereGrid
from worldgen.surface_liquids import (
    integrate_liquid_volume_m3,
    partition_volatile_inventory,
    solve_global_liquid_level,
    solve_surface_liquids,
)


def _global_layer_volume(grid: SphereGrid, depth_m: float, bed_km: float = 0.0) -> float:
    r0 = grid.radius_km * 1000.0 + bed_km * 1000.0
    return 4.0 * math.pi * (r0 * r0 * depth_m + r0 * depth_m**2 + depth_m**3 / 3.0)


def test_flat_sphere_fill_recovers_requested_global_depth():
    grid = SphereGrid(32, 16, 6371.0)
    bed = np.zeros(grid.shape, dtype=np.float64)
    target = _global_layer_volume(grid, 125.0)
    level_km, depth_m, integrated = solve_global_liquid_level(grid, bed, target)
    assert abs(level_km - 0.125) < 1e-9
    np.testing.assert_allclose(depth_m, 125.0, rtol=0, atol=1e-5)
    assert abs(integrated - target) / target < 1e-11


def test_fill_starts_in_deepest_cells_before_spilling_upward():
    grid = SphereGrid(8, 4, 1000.0)
    bed = np.full(grid.shape, 1.0, dtype=np.float64)
    bed[1, 2] = -2.0
    bed[2, 5] = -1.0
    # Fill only partway from -2 km toward the next basin floor at -1 km.
    omega = grid.cell_area_weights[1, 2] * 4.0 * math.pi
    r0 = grid.radius_km * 1000.0 - 2000.0
    target = omega * (r0 * r0 * 400.0 + r0 * 400.0**2 + 400.0**3 / 3.0)
    level_km, depth_m, integrated = solve_global_liquid_level(grid, bed, target)
    assert -2.0 < level_km < -1.0
    assert depth_m[1, 2] > 0.0
    assert depth_m[2, 5] == 0.0
    assert np.count_nonzero(depth_m) == 1
    assert abs(integrated - target) / target < 1e-10


def test_integrated_volume_matches_solver_for_irregular_spherical_bed():
    grid = SphereGrid(40, 20, 3200.0)
    yy, xx = np.indices(grid.shape)
    bed = (1.2 * np.sin(xx / 5.0) - 2.4 * np.cos(yy / 4.0)).astype(float)
    target = 3.1e17
    level_km, _, integrated = solve_global_liquid_level(grid, bed, target)
    check = integrate_liquid_volume_m3(grid, bed, level_km)
    assert abs(integrated - target) / target < 1e-10
    assert abs(check - target) / target < 1e-10


def test_hot_water_inventory_moves_to_vapor_instead_of_liquid():
    grid = SphereGrid(24, 12)
    # The cool 288 K atmosphere can hold about 6.1e16 kg at 65% RH on this
    # Earth-radius grid, so the inventory must exceed that capacity for a liquid
    # reservoir to coexist with vapor. At 500 K a one-bar atmosphere can hold much
    # more vapor, exercising the intended temperature-dependent repartitioning.
    mass = 5e18
    cool = np.full(grid.shape, 288.0 - 273.15)
    hot = np.full(grid.shape, 500.0 - 273.15)
    p_cool = partition_volatile_inventory(
        grid, "H2O", mass, cool,
        surface_pressure_bar=1.0, gravity_m_s2=9.81,
        relative_humidity=0.65, thermodynamics_backend="builtin",
    )
    p_hot = partition_volatile_inventory(
        grid, "H2O", mass, hot,
        surface_pressure_bar=1.0, gravity_m_s2=9.81,
        relative_humidity=1.0, thermodynamics_backend="builtin",
    )
    assert p_cool.liquid_mass_kg > 0.0
    assert p_hot.vapor_mass_kg > p_cool.vapor_mass_kg
    assert p_hot.liquid_mass_kg < p_cool.liquid_mass_kg


def test_frozen_world_sequesters_condensed_inventory_as_solid():
    grid = SphereGrid(24, 12)
    temp = np.full(grid.shape, -80.0)
    part = partition_volatile_inventory(
        grid, "H2O", 1e20, temp,
        surface_pressure_bar=1.0, gravity_m_s2=9.81,
        relative_humidity=0.5, ice_fixation_efficiency=1.0,
        thermodynamics_backend="builtin",
    )
    assert part.solid_mass_kg > 0.99 * (part.total_mass_kg - part.vapor_mass_kg)
    assert part.liquid_mass_kg == 0.0


def test_lower_density_liquid_produces_larger_filled_volume_for_same_mass():
    grid = SphereGrid(32, 16, 2575.0)
    bed = np.zeros(grid.shape)
    temp_methane = np.full(grid.shape, 94.0 - 273.15)
    methane = solve_surface_liquids(
        grid, bed, temp_methane, {"CH4": 2e18},
        surface_pressure_bar=1.5, gravity_m_s2=1.35,
        relative_humidity=0.0, ice_fixation_efficiency=0.0,
        thermodynamics_backend="builtin",
    )
    temp_water = np.full(grid.shape, 290.0 - 273.15)
    water = solve_surface_liquids(
        grid, bed, temp_water, {"H2O": 2e18},
        surface_pressure_bar=1.5, gravity_m_s2=1.35,
        relative_humidity=0.0, ice_fixation_efficiency=0.0,
        thermodynamics_backend="builtin",
    )
    assert methane.total_liquid_mass_kg == water.total_liquid_mass_kg
    assert methane.total_liquid_volume_m3 > water.total_liquid_volume_m3
    assert methane.liquid_level_km > water.liquid_level_km


def test_multispecies_liquid_volumes_add_before_global_fill():
    grid = SphereGrid(24, 12, 3000.0)
    bed = np.linspace(-3.0, 2.0, grid.height)[:, None] + np.zeros(grid.shape)
    temp = np.full(grid.shape, 95.0 - 273.15)
    result = solve_surface_liquids(
        grid, bed, temp, {"CH4": 1e18, "C2H6": 1e18},
        surface_pressure_bar=1.5, gravity_m_s2=1.4,
        relative_humidity=0.0, ice_fixation_efficiency=0.0,
        thermodynamics_backend="builtin",
    )
    summed = sum(p.liquid_volume_m3 for p in result.partitions.values())
    assert result.total_liquid_volume_m3 == summed
    assert abs(result.integrated_volume_m3 - summed) / max(summed, 1.0) < 1e-10
    np.testing.assert_allclose(
        result.relative_surface_elevation_km,
        bed - result.liquid_level_km,
        rtol=0,
        atol=2e-7,
    )
