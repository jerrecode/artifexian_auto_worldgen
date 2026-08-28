from __future__ import annotations

"""Hierarchical drainage-basin analysis on the canonical single-receiver graph.

The hydrology solver already routes every land cell through one acyclic receiver graph.
This module promotes that graph into explicit, queryable watershed structure instead
of conflating all rivers entering one connected ocean with one basin.  It also derives
terrain/hydrology metrics useful for groundwater, wetlands, ecology and recursive
hydrologic refinement.
"""

from dataclasses import dataclass
import numpy as np

from .drainage import DrainageGraph
from .grid import SphereGrid
from .mathops import optional_njit


@dataclass(slots=True)
class WatershedHierarchy:
    basin_id: np.ndarray
    subbasin_level_1: np.ndarray
    subbasin_level_2: np.ndarray
    subbasin_level_3: np.ndarray
    exorheic: np.ndarray
    distance_to_outlet_km: np.ndarray
    topographic_wetness_index: np.ndarray
    height_above_nearest_drainage_m: np.ndarray
    metadata: dict


@optional_njit(cache=True)
def _inherit_downstream_seed_kernel(
    receiver: np.ndarray,
    order: np.ndarray,
    seed_label: np.ndarray,
    land: np.ndarray,
) -> np.ndarray:
    out = seed_label.copy()
    for kk in range(order.size - 1, -1, -1):
        node = order[kk]
        if not land[node] or out[node] != 0:
            continue
        nxt = receiver[node]
        if nxt >= 0 and land[nxt]:
            out[node] = out[nxt]
    return out


@optional_njit(cache=True)
def _distance_to_outlet_kernel(
    receiver: np.ndarray,
    order: np.ndarray,
    edge_km: np.ndarray,
    land: np.ndarray,
) -> np.ndarray:
    out = np.zeros(receiver.size, dtype=np.float64)
    for kk in range(order.size - 1, -1, -1):
        node = order[kk]
        if not land[node]:
            continue
        nxt = receiver[node]
        if nxt >= 0 and land[nxt]:
            out[node] = edge_km[node] + out[nxt]
        elif nxt >= 0:
            out[node] = edge_km[node]
    return out


@optional_njit(cache=True)
def _hand_kernel(
    receiver: np.ndarray,
    order: np.ndarray,
    elevation_m: np.ndarray,
    land: np.ndarray,
    channel: np.ndarray,
) -> np.ndarray:
    reference = np.empty(receiver.size, dtype=np.float64)
    reference[:] = np.nan
    for node in range(receiver.size):
        if channel[node]:
            reference[node] = elevation_m[node]
    for kk in range(order.size - 1, -1, -1):
        node = order[kk]
        if not land[node] or channel[node]:
            continue
        nxt = receiver[node]
        if nxt >= 0 and land[nxt] and np.isfinite(reference[nxt]):
            reference[node] = reference[nxt]
        elif nxt >= 0 and not land[nxt]:
            reference[node] = 0.0
    out = np.zeros(receiver.size, dtype=np.float64)
    for node in range(receiver.size):
        if land[node] and np.isfinite(reference[node]):
            d = elevation_m[node] - reference[node]
            out[node] = d if d > 0.0 else 0.0
    return out


def _edge_length_km(grid: SphereGrid, receiver: np.ndarray) -> np.ndarray:
    recv = np.asarray(receiver, dtype=np.int64).ravel()
    n = recv.size
    src = np.arange(n, dtype=np.int64)
    good = recv >= 0
    out = np.zeros(n, dtype=np.float64)
    if np.any(good):
        xyz = np.asarray(grid.xyz, dtype=np.float64).reshape(-1, 3)
        dot = np.sum(xyz[src[good]] * xyz[recv[good]], axis=1)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        out[good] = float(grid.radius_km) * angle
    return out


def _outlet_labels(graph: DrainageGraph, land: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    recv = graph.receiver
    lf = np.asarray(land, dtype=bool).ravel()
    n = recv.size
    safe = np.where(recv >= 0, recv, 0)
    receiver_land = np.zeros(n, dtype=bool)
    valid = recv >= 0
    receiver_land[valid] = lf[safe[valid]]
    outlet = lf & ((recv < 0) | ~receiver_land)
    outlet_nodes = np.flatnonzero(outlet)
    seeds = np.zeros(n, dtype=np.int32)
    seeds[outlet_nodes] = np.arange(1, outlet_nodes.size + 1, dtype=np.int32)
    labels = _inherit_downstream_seed_kernel(recv, graph.order, seeds, lf)

    exo_seed = np.zeros(n, dtype=np.int32)
    if outlet_nodes.size:
        drains_to_ocean = recv[outlet_nodes] >= 0
        exo_seed[outlet_nodes] = drains_to_ocean.astype(np.int32) + 1
    exo_code = _inherit_downstream_seed_kernel(recv, graph.order, exo_seed, lf)
    exorheic = exo_code == 2
    return labels, exorheic, outlet


def _subbasin_partition(
    graph: DrainageGraph,
    land: np.ndarray,
    drainage_area_km2: np.ndarray,
    outlet: np.ndarray,
    threshold_km2: float,
) -> np.ndarray:
    """Partition by the nearest downstream scale-appropriate pour point.

    Pour points include outlet cells, threshold-crossing channel heads and major
    confluences.  Decreasing the threshold therefore gives a nested, progressively
    finer segmentation suitable for map display and local refinement scheduling.
    """
    recv = graph.receiver
    lf = np.asarray(land, dtype=bool).ravel()
    area = np.asarray(drainage_area_km2, dtype=np.float64).ravel()
    safe = np.where(recv >= 0, recv, 0)
    valid_edge = (recv >= 0) & lf & lf[safe]

    max_donor = np.zeros(recv.size, dtype=np.float64)
    if np.any(valid_edge):
        np.maximum.at(max_donor, safe[valid_edge], area[valid_edge])

    substantial = valid_edge & (area >= 0.28 * threshold_km2)
    donor_count = np.zeros(recv.size, dtype=np.int16)
    if np.any(substantial):
        np.add.at(donor_count, safe[substantial], 1)

    heads = lf & (area >= threshold_km2) & (max_donor < threshold_km2)
    confluence = lf & (area >= threshold_km2) & (donor_count >= 2)
    seeds_mask = np.asarray(outlet, bool).ravel() | heads | confluence
    seed_nodes = np.flatnonzero(seeds_mask)
    seed_label = np.zeros(recv.size, dtype=np.int32)
    seed_label[seed_nodes] = np.arange(1, seed_nodes.size + 1, dtype=np.int32)
    return _inherit_downstream_seed_kernel(recv, graph.order, seed_label, lf)


def build_watershed_hierarchy(
    grid: SphereGrid,
    graph: DrainageGraph,
    land: np.ndarray,
    elevation_km: np.ndarray,
    drainage_area_km2: np.ndarray,
    receiver_slope: np.ndarray,
    channel_mask: np.ndarray,
    *,
    subbasin_thresholds_km2: tuple[float, float, float] = (1.0e6, 1.0e5, 1.0e4),
) -> WatershedHierarchy:
    lf = np.asarray(land, dtype=bool).ravel()
    basin, exorheic, outlet = _outlet_labels(graph, land)

    t1, t2, t3 = (max(float(x), 1.0) for x in subbasin_thresholds_km2)
    level1 = _subbasin_partition(graph, land, drainage_area_km2, outlet, t1)
    level2 = _subbasin_partition(graph, land, drainage_area_km2, outlet, t2)
    level3 = _subbasin_partition(graph, land, drainage_area_km2, outlet, t3)

    edge = _edge_length_km(grid, graph.receiver)
    distance = _distance_to_outlet_kernel(graph.receiver, graph.order, edge, lf)

    area = np.asarray(drainage_area_km2, dtype=np.float64).ravel()
    slope = np.maximum(np.asarray(receiver_slope, dtype=np.float64).ravel(), 1.0e-7)
    cell_area = np.asarray(grid.cell_area_weights, dtype=np.float64).ravel() * (
        4.0 * np.pi * float(grid.radius_km) ** 2
    )
    width_m = np.sqrt(np.maximum(cell_area, 1.0e-9)) * 1000.0
    specific_area_m = np.maximum(area, 0.0) * 1.0e6 / np.maximum(width_m, 1.0)
    twi = np.zeros_like(area)
    twi[lf] = np.log((specific_area_m[lf] + 1.0) / slope[lf])

    hand = _hand_kernel(
        graph.receiver,
        graph.order,
        np.asarray(elevation_km, dtype=np.float64).ravel() * 1000.0,
        lf,
        np.asarray(channel_mask, dtype=bool).ravel(),
    )

    basin_land = basin[lf]
    unique, counts = np.unique(basin_land[basin_land > 0], return_counts=True)
    outlet_count = int(unique.size)
    internal_count = int(np.unique(basin[lf & ~exorheic]).size) if np.any(lf & ~exorheic) else 0
    metadata = {
        "basin_semantics": "unique terminal land outlet catchments; not connected-ocean labels",
        "outlet_basin_count": outlet_count,
        "endorheic_or_internal_basin_count": internal_count,
        "largest_basin_cells": int(np.max(counts)) if counts.size else 0,
        "median_basin_cells": float(np.median(counts)) if counts.size else 0.0,
        "subbasin_thresholds_km2": [t1, t2, t3],
        "all_land_assigned": bool(np.all(basin[lf] > 0)) if np.any(lf) else True,
        "distance_definition": "great-circle receiver-edge distance accumulated to terminal outlet",
        "hand_definition": "height above nearest downstream resolved channel/effective shoreline drainage",
        "twi_definition": "ln(specific catchment area / receiver slope), reduced-order raster proxy",
    }
    shape = graph.shape
    return WatershedHierarchy(
        basin.reshape(shape).astype(np.int32),
        level1.reshape(shape).astype(np.int32),
        level2.reshape(shape).astype(np.int32),
        level3.reshape(shape).astype(np.int32),
        exorheic.reshape(shape),
        distance.reshape(shape).astype(np.float32),
        twi.reshape(shape).astype(np.float32),
        hand.reshape(shape).astype(np.float32),
        metadata,
    )


__all__ = ["WatershedHierarchy", "build_watershed_hierarchy"]
