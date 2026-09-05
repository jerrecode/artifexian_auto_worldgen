from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from worldgen.config import WorldConfig
from worldgen.drainage import DrainageGraph
from worldgen.grid import SphereGrid
from worldgen.hydrology_natural_routing import flow_directions_continuous_local
from worldgen.hydrology_reliability import (
    channel_hierarchy_discharge_guarded,
    enforce_hydrology_guardrails,
    lake_mask_volume_guarded,
    priority_flood_closed_aware,
)
from worldgen.watershed_naturalism import build_watershed_hierarchy_natural


def test_oceanless_priority_flood_preserves_real_closed_basins():
    grid = SphereGrid(48, 24)
    yy, xx = np.indices(grid.shape)
    elevation = 1.0 + 0.002 * yy + 0.001 * np.cos(xx / 3.0)
    elevation[12, 17] -= 0.25
    ocean = np.zeros(grid.shape, dtype=bool)
    filled = priority_flood_closed_aware(elevation, ocean, grid)
    np.testing.assert_array_equal(filled, elevation)


def test_natural_routing_uses_only_adjacent_downhill_receivers():
    grid = SphereGrid(48, 24)
    y = np.linspace(2.0, 0.0, grid.height)[:, None]
    relief = 0.025 * np.sin(np.deg2rad(grid.lon * 5.0 + grid.lat * 2.0))
    elevation = y + relief
    ocean = np.zeros(grid.shape, dtype=bool)
    ocean[-2:] = True

    receiver = flow_directions_continuous_local(elevation, ocean, grid).reshape(grid.shape)
    valid_targets = []
    for dy, dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
        ny, nx = grid.ops.neighbor_indices(dy, dx)
        valid_targets.append(ny * grid.width + nx)
    valid_targets = np.stack(valid_targets, axis=0).reshape(8, -1)

    flat = receiver.ravel()
    active = flat >= 0
    assert np.all(np.any(valid_targets[:, active] == flat[active], axis=0))
    DrainageGraph.from_receiver(flat, grid.shape)


def test_lake_soft_cap_applies_even_to_first_giant_component():
    grid = SphereGrid(64, 32)
    elevation = np.zeros(grid.shape, dtype=float)
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
        grid, elevation, filled, land, drainage, runoff_acc, climate, cfg
    )
    fraction = grid.weighted_fraction(lakes) / grid.weighted_fraction(land)
    assert 0.0 < fraction < 0.03


def test_dry_world_discharge_guard_prevents_river_overclassification():
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
    _channel, _cls, rivers, *_ = channel_hierarchy_discharge_guarded(
        grid, base, water, cfg
    )
    assert np.count_nonzero(rivers) / n <= 0.20 + 1.0 / n


def test_reliability_guardrail_rejects_pathological_global_classification():
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


def test_major_watersheds_aggregate_comb_of_tiny_coastal_outlets():
    grid = SphereGrid(width=24, height=12, radius_km=100.0)
    h, w = grid.shape
    land = np.ones(grid.shape, dtype=bool)
    land[-1] = False
    receiver = np.full(grid.shape, -1, dtype=np.int64)
    for yy in range(h - 1):
        ny, nx = grid.ops.neighbor_indices(1, 0)
        target = ny * w + nx
        receiver[yy] = target[yy]
    receiver[~land] = -1
    graph = DrainageGraph.from_receiver(receiver.ravel(), grid.shape)

    cell_area = grid.cell_area_weights * (4.0 * np.pi * grid.radius_km**2)
    drainage = graph.accumulate(cell_area * land)
    elevation = np.linspace(1.0, -0.1, h)[:, None] * np.ones((1, w))
    slope = np.full(grid.shape, 0.01)
    channel = land & (drainage > np.percentile(drainage[land], 45.0))

    hierarchy = build_watershed_hierarchy_natural(
        grid,
        graph,
        land,
        elevation,
        drainage,
        slope,
        channel,
        subbasin_thresholds_km2=(1000.0, 300.0, 80.0),
    )
    meta = hierarchy.metadata
    assert meta["raw_terminal_outlet_basin_count"] == w
    assert 1 <= meta["outlet_basin_count"] < meta["raw_terminal_outlet_basin_count"]
    assert meta["median_basin_cells"] > meta["raw_terminal_median_basin_cells"]
    assert np.all(hierarchy.basin_id[land] > 0)


def test_reliability_configuration_requires_descending_subbasin_scales():
    cfg = WorldConfig()
    cfg.hydrology.subbasin_thresholds_km2 = (1000.0, 500.0, 100.0)
    cfg.validate()
    cfg.hydrology.subbasin_thresholds_km2 = (100.0, 500.0, 10.0)
    with pytest.raises(ValueError, match="strictly descending"):
        cfg.validate()


def test_public_hydrology_facade_installs_reliability_backends():
    import worldgen.hydrology as hydro
    import worldgen.hydrology_advanced as advanced
    import worldgen.hydrology_base as base

    assert base._priority_flood is priority_flood_closed_aware
    assert base._flow_directions is flow_directions_continuous_local
    assert base._lake_mask is lake_mask_volume_guarded
    assert advanced._channel_hierarchy is channel_hierarchy_discharge_guarded
    assert advanced.build_watershed_hierarchy is build_watershed_hierarchy_natural
    assert hydro._priority_flood is priority_flood_closed_aware


def test_endorheic_basins_are_not_merged_into_coastal_basin_groups():
    grid = SphereGrid(width=24, height=12, radius_km=100.0)
    h, w = grid.shape
    land = np.ones(grid.shape, dtype=bool)
    land[-1] = False
    receiver = np.full(grid.shape, -1, dtype=np.int64)

    # Most columns drain to the ocean. One interior column terminates in a genuine
    # internal sink and must remain a distinct major-basin label.
    ny, nx = grid.ops.neighbor_indices(1, 0)
    target = ny * w + nx
    for yy in range(h - 1):
        receiver[yy] = target[yy]
    sink_y, sink_x = h // 2, w // 2
    receiver[sink_y, sink_x] = -1
    for yy in range(sink_y):
        receiver[yy, sink_x] = (yy + 1) * w + sink_x
    receiver[~land] = -1

    graph = DrainageGraph.from_receiver(receiver.ravel(), grid.shape)
    cell_area = grid.cell_area_weights * (4.0 * np.pi * grid.radius_km**2)
    drainage = graph.accumulate(cell_area * land)
    elevation = np.linspace(1.0, -0.1, h)[:, None] * np.ones((1, w))
    slope = np.full(grid.shape, 0.01)
    channel = land & (drainage > np.percentile(drainage[land], 45.0))

    hierarchy = build_watershed_hierarchy_natural(
        grid,
        graph,
        land,
        elevation,
        drainage,
        slope,
        channel,
        subbasin_thresholds_km2=(1000.0, 300.0, 80.0),
    )
    internal_labels = set(np.unique(hierarchy.basin_id[land & ~hierarchy.exorheic]))
    exorheic_labels = set(np.unique(hierarchy.basin_id[land & hierarchy.exorheic]))
    internal_labels.discard(0)
    exorheic_labels.discard(0)
    assert internal_labels
    assert internal_labels.isdisjoint(exorheic_labels)
    assert hierarchy.metadata["preserved_endorheic_terminal_basins"] >= 1


def test_guardrail_prefers_spherical_area_fraction_over_cell_fraction():
    result = SimpleNamespace(
        metadata={
            "river_area_fraction_of_land": 0.05,
            "watersheds": {
                "largest_basin_cells": 99,
                "largest_basin_area_fraction_land": 0.50,
            },
        },
        base=SimpleNamespace(metadata={"lake_area_fraction_of_land": 0.02}),
    )
    terrain = SimpleNamespace(land=np.ones((10, 10), dtype=bool))
    cfg = SimpleNamespace(
        hydrology_fail_river_fraction_land=0.35,
        hydrology_fail_lake_fraction_land=0.10,
        hydrology_fail_largest_basin_fraction_land=0.95,
    )
    enforce_hydrology_guardrails(result, terrain, cfg)
    assert result.metadata["reliability_guardrails"]["largest_basin_fraction_land"] == pytest.approx(0.50)
