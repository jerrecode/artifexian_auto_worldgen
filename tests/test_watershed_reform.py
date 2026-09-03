from __future__ import annotations

import numpy as np

from worldgen.drainage import DrainageGraph
from worldgen.grid import SphereGrid
from worldgen.hydrology_natural_routing import flow_directions_continuous_local
from worldgen.watershed_naturalism import build_watershed_hierarchy_natural


def test_natural_routing_receiver_is_always_immediately_adjacent():
    grid = SphereGrid(width=48, height=24, radius_km=6371.0)
    # Broad slope plus smooth deterministic relief; keep a real ocean strip.
    y = np.linspace(2.0, 0.0, grid.height)[:, None]
    relief = 0.025 * np.sin(np.deg2rad(grid.lon * 5.0 + grid.lat * 2.0))
    z = y + relief
    ocean = np.zeros(grid.shape, dtype=bool)
    ocean[-2:] = True
    receiver = flow_directions_continuous_local(z, ocean, grid).reshape(grid.shape)

    valid_targets = []
    for dy, dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
        ny, nx = grid.ops.neighbor_indices(dy, dx)
        valid_targets.append(ny * grid.width + nx)
    valid_targets = np.stack(valid_targets, axis=0)
    rr = receiver.ravel()
    active = rr >= 0
    assert np.all(np.any(valid_targets.reshape(8, -1)[:, active] == rr[active], axis=0))
    # Strictly downhill receiver rule keeps the graph acyclic.
    DrainageGraph.from_receiver(rr, grid.shape)


def test_major_watersheds_merge_comb_of_tiny_terminal_outlets():
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

    ws = build_watershed_hierarchy_natural(
        grid, graph, land, elevation, drainage, slope, channel,
        subbasin_thresholds_km2=(1000.0, 300.0, 80.0),
    )
    meta = ws.metadata
    assert meta["raw_terminal_outlet_basin_count"] == w
    assert 1 <= meta["outlet_basin_count"] < meta["raw_terminal_outlet_basin_count"]
    assert meta["median_basin_cells"] > meta["raw_terminal_median_basin_cells"]
    assert meta["domain_decomposition"].startswith("none")
    assert np.all(ws.basin_id[land] > 0)
