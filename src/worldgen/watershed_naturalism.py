from __future__ import annotations

"""Naturalized watershed hierarchy built on the global drainage DAG.

The raw terminal-outlet representation assigns a unique basin to every coastal outlet
cell.  At high resolution this creates tens of thousands of one-cell terminal basins
and conspicuous comb/grid patterns even when routing is numerically valid.  This
module keeps raw terminal catchments internally, but aggregates neighboring minor
coastal catchments into major drainage systems, derives nested subbasins from true
threshold crossings/confluences, and uses a thinned drainage reference for HAND.

All propagation is global on the canonical DrainageGraph; there are no independent
256x256 tile solves or stitched label fields.
"""

import math
import numpy as np

from .drainage import DrainageGraph
from .watersheds import WatershedHierarchy
from .mathops import optional_njit


@optional_njit(cache=True)
def _inherit_downstream_seed_kernel(receiver, order, seed_label, land):
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
def _distance_to_outlet_kernel(receiver, order, edge_km, land):
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
def _hand_kernel(receiver, order, elevation_m, land, channel):
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


def _edge_length_km(grid, receiver):
    recv = np.asarray(receiver, np.int64).ravel()
    src = np.arange(recv.size, dtype=np.int64)
    good = recv >= 0
    out = np.zeros(recv.size, dtype=np.float64)
    if np.any(good):
        xyz = np.asarray(grid.xyz, float).reshape(-1, 3)
        dot = np.sum(xyz[src[good]] * xyz[recv[good]], axis=1)
        out[good] = float(grid.radius_km) * np.arccos(np.clip(dot, -1.0, 1.0))
    return out


def _raw_outlet_labels(graph: DrainageGraph, land: np.ndarray):
    recv = graph.receiver
    lf = np.asarray(land, bool).ravel()
    safe = np.where(recv >= 0, recv, 0)
    valid = recv >= 0
    receiver_land = np.zeros(recv.size, dtype=bool)
    receiver_land[valid] = lf[safe[valid]]
    outlet = lf & ((recv < 0) | ~receiver_land)
    outlet_nodes = np.flatnonzero(outlet)
    seeds = np.zeros(recv.size, dtype=np.int32)
    seeds[outlet_nodes] = np.arange(1, outlet_nodes.size + 1, dtype=np.int32)
    labels = _inherit_downstream_seed_kernel(recv, graph.order, seeds, lf)

    exo_seed = np.zeros(recv.size, dtype=np.int32)
    if outlet_nodes.size:
        exo_seed[outlet_nodes] = (recv[outlet_nodes] >= 0).astype(np.int32) + 1
    exo_code = _inherit_downstream_seed_kernel(recv, graph.order, exo_seed, lf)
    return labels, exo_code == 2, outlet, outlet_nodes


def _major_basin_aggregation(
    grid,
    raw,
    outlet_nodes,
    land,
    drainage_area_km2,
    exorheic,
):
    """Merge minor coastal outlet catchments into nearby major drainage systems.

    Catchment boundaries themselves are never redrawn: whole raw terminal catchments
    are merged.  Therefore every displayed major-basin boundary remains an actual
    drainage divide from the global receiver graph rather than a Voronoi/pixel box.
    """
    lf = np.asarray(land, bool).ravel()
    raw = np.asarray(raw, np.int32).ravel()
    max_raw = int(raw.max(initial=0))
    if max_raw <= 0:
        return raw.copy(), {"raw_outlet_basin_count": 0, "major_basin_count": 0}

    cell_area = np.asarray(grid.cell_area_weights, float).ravel() * (
        4.0 * math.pi * float(grid.radius_km) ** 2
    )
    raw_area = np.bincount(raw, weights=cell_area, minlength=max_raw + 1)

    # Each raw label has exactly one terminal outlet; preserve that label->node map.
    outlet_for_label = np.full(max_raw + 1, -1, dtype=np.int64)
    if outlet_nodes.size:
        outlet_for_label[np.arange(1, outlet_nodes.size + 1)] = outlet_nodes

    comp_map, ncomp = grid.ops.connected_components(np.asarray(land, bool))
    comp_flat = np.asarray(comp_map, np.int32).ravel()
    xyz = np.asarray(grid.xyz, float).reshape(-1, 3)

    mapping = np.zeros(max_raw + 1, dtype=np.int32)
    next_major = 1
    anchor_counts = []
    target_anchor_area_km2 = max(1.0e5, 0.0025 * 4.0 * math.pi * float(grid.radius_km) ** 2)
    hard_major_min_km2 = max(1.5e4, 0.12 * target_anchor_area_km2)

    present_labels = np.flatnonzero(raw_area > 0.0)
    present_labels = present_labels[present_labels > 0]
    exo_flat = np.asarray(exorheic, bool).ravel()

    # Endorheic terminal basins are physically distinct drainage systems and must
    # never be merged into a nearby coastal/exorheic group merely because their
    # outlets are geographically close.
    internal_labels: list[int] = []
    exorheic_label = np.zeros(max_raw + 1, dtype=bool)
    for lab in present_labels:
        members = raw == int(lab)
        is_exo = bool(np.any(exo_flat[members]))
        exorheic_label[int(lab)] = is_exo
        if not is_exo:
            internal_labels.append(int(lab))

    for lab in internal_labels:
        mapping[lab] = next_major
        next_major += 1

    for comp in range(1, int(ncomp) + 1):
        labels = []
        for lab in present_labels:
            node = outlet_for_label[lab]
            if (
                node >= 0
                and comp_flat[node] == comp
                and exorheic_label[int(lab)]
            ):
                labels.append(int(lab))
        if not labels:
            continue
        labs = np.asarray(labels, dtype=np.int32)
        comp_area = float(np.sum(raw_area[labs]))
        target_n = max(1, int(math.ceil(comp_area / target_anchor_area_km2)))
        target_n = min(target_n, 256)

        major = labs[raw_area[labs] >= hard_major_min_km2]
        if major.size < target_n:
            order = labs[np.argsort(raw_area[labs])[::-1]]
            chosen = list(map(int, major.tolist()))
            chosen_set = set(chosen)
            for lab in order:
                li = int(lab)
                if li not in chosen_set:
                    chosen.append(li)
                    chosen_set.add(li)
                if len(chosen) >= target_n:
                    break
            major = np.asarray(chosen, dtype=np.int32)
        if major.size == 0:
            major = np.asarray([int(labs[np.argmax(raw_area[labs])])], dtype=np.int32)

        anchor_nodes = outlet_for_label[major]
        anchor_xyz = xyz[anchor_nodes]
        anchor_ids = np.arange(next_major, next_major + major.size, dtype=np.int32)
        next_major += int(major.size)
        anchor_counts.append(int(major.size))

        # Chunked spherical nearest-anchor assignment of whole terminal catchments.
        for start in range(0, labs.size, 2048):
            chunk = labs[start:start + 2048]
            nodes = outlet_for_label[chunk]
            dots = xyz[nodes] @ anchor_xyz.T
            nearest = np.argmax(dots, axis=1)
            mapping[chunk] = anchor_ids[nearest]

    major_labels = mapping[np.clip(raw, 0, max_raw)]
    major_labels[~lf] = 0
    vals, counts = np.unique(major_labels[lf & (major_labels > 0)], return_counts=True)
    return major_labels, {
        "raw_outlet_basin_count": int(present_labels.size),
        "major_basin_count": int(vals.size),
        "major_basin_median_cells": float(np.median(counts)) if counts.size else 0.0,
        "major_basin_largest_cells": int(np.max(counts)) if counts.size else 0,
        "major_basin_target_anchor_area_km2": float(target_anchor_area_km2),
        "major_basin_hard_anchor_min_km2": float(hard_major_min_km2),
        "major_basin_anchor_counts_by_landmass": anchor_counts,
        "preserved_endorheic_terminal_basins": int(len(internal_labels)),
        "aggregation_model": "whole exorheic terminal catchments merge to nearby major coastal outlets on the same connected landmass; endorheic basins remain independent and all boundaries remain true raw drainage divides",
    }


def _branch_partition(graph, land, area2d, macro_basin, threshold_km2):
    recv = graph.receiver
    lf = np.asarray(land, bool).ravel()
    area = np.asarray(area2d, float).ravel()
    macro = np.asarray(macro_basin, np.int32).ravel()
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
    confluences = lf & (area >= threshold_km2) & (donor_count >= 2)
    seed_nodes = np.flatnonzero(heads | confluences)
    seed = np.zeros(recv.size, dtype=np.int32)
    seed[seed_nodes] = np.arange(1, seed_nodes.size + 1, dtype=np.int32)
    branch = _inherit_downstream_seed_kernel(recv, graph.order, seed, lf)

    # Cells below the last scale-appropriate pour point remain in their major basin;
    # branch catchments receive unique labels.  This avoids turning every tiny coast
    # outlet into a displayed subbasin seed.
    macro_count = int(macro.max(initial=0))
    out = macro.copy()
    mask = branch > 0
    out[mask] = macro_count + branch[mask]
    out[~lf] = 0
    return out


def build_watershed_hierarchy_natural(
    grid,
    graph: DrainageGraph,
    land: np.ndarray,
    elevation_km: np.ndarray,
    drainage_area_km2: np.ndarray,
    receiver_slope: np.ndarray,
    channel_mask: np.ndarray,
    *,
    subbasin_thresholds_km2=(1.0e6, 1.0e5, 1.0e4),
) -> WatershedHierarchy:
    lf = np.asarray(land, bool).ravel()
    raw, exorheic, outlet, outlet_nodes = _raw_outlet_labels(graph, land)
    basin, aggregation_meta = _major_basin_aggregation(
        grid, raw, outlet_nodes, land, drainage_area_km2, exorheic
    )

    t1, t2, t3 = (max(float(v), 1.0) for v in subbasin_thresholds_km2)
    level1 = _branch_partition(graph, land, drainage_area_km2, basin, t1)
    level2 = _branch_partition(graph, land, drainage_area_km2, basin, t2)
    level3 = _branch_partition(graph, land, drainage_area_km2, basin, t3)

    edge = _edge_length_km(grid, graph.receiver)
    distance = _distance_to_outlet_kernel(graph.receiver, graph.order, edge, lf)

    area = np.asarray(drainage_area_km2, float).ravel()
    slope = np.maximum(np.asarray(receiver_slope, float).ravel(), 1.0e-7)
    cell_area = np.asarray(grid.cell_area_weights, float).ravel() * (
        4.0 * np.pi * float(grid.radius_km) ** 2
    )
    width_m = np.sqrt(np.maximum(cell_area, 1.0e-9)) * 1000.0
    specific_area_m = np.maximum(area, 0.0) * 1.0e6 / np.maximum(width_m, 1.0)
    twi = np.zeros_like(area)
    twi[lf] = np.log((specific_area_m[lf] + 1.0) / slope[lf])

    # HAND referenced every resolved rill previously; at high resolution that exposes
    # the receiver lattice.  Use only drainage paths with a meaningful upstream area.
    median_cell_area = float(np.median(cell_area[lf])) if np.any(lf) else float(np.median(cell_area))
    hand_min_area = max(250.0, 8.0 * median_cell_area)
    hand_channel = np.asarray(channel_mask, bool).ravel() & (area >= hand_min_area)
    hand = _hand_kernel(
        graph.receiver,
        graph.order,
        np.asarray(elevation_km, float).ravel() * 1000.0,
        lf,
        hand_channel,
    )

    raw_land = raw[lf]
    raw_unique, raw_counts = np.unique(raw_land[raw_land > 0], return_counts=True)
    major_land = basin[lf]
    major_unique, major_counts = np.unique(major_land[major_land > 0], return_counts=True)
    internal_count = int(np.unique(raw[lf & ~exorheic]).size) if np.any(lf & ~exorheic) else 0
    land_area = float(np.sum(cell_area[lf]))
    if major_unique.size and land_area > 0.0:
        major_area = np.bincount(
            basin[lf],
            weights=cell_area[lf],
            minlength=int(basin.max(initial=0)) + 1,
        )
        largest_basin_area_fraction_land = float(
            np.max(major_area[major_unique]) / land_area
        )
    else:
        largest_basin_area_fraction_land = 0.0

    metadata = {
        "basin_semantics": "major natural drainage systems formed by merging whole minor terminal catchments; raw outlet catchments retained diagnostically",
        "outlet_basin_count": int(major_unique.size),
        "raw_terminal_outlet_basin_count": int(raw_unique.size),
        "raw_terminal_median_basin_cells": float(np.median(raw_counts)) if raw_counts.size else 0.0,
        "endorheic_or_internal_basin_count": internal_count,
        "largest_basin_cells": int(np.max(major_counts)) if major_counts.size else 0,
        "largest_basin_area_fraction_land": largest_basin_area_fraction_land,
        "median_basin_cells": float(np.median(major_counts)) if major_counts.size else 0.0,
        "subbasin_thresholds_km2": [t1, t2, t3],
        "all_land_assigned": bool(np.all(basin[lf] > 0)) if np.any(lf) else True,
        "distance_definition": "great-circle receiver-edge distance accumulated globally to the true terminal outlet",
        "hand_definition": "height above nearest downstream scale-resolved drainage path/effective shoreline; tiny raster rills excluded",
        "hand_min_drainage_area_km2": float(hand_min_area),
        "twi_definition": "ln(specific catchment area / receiver slope), global DAG reduced-order raster proxy",
        "domain_decomposition": "none: watershed labels, subbasins, outlet distance and HAND are propagated on the full global drainage DAG",
        **aggregation_meta,
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


__all__ = ["build_watershed_hierarchy_natural"]
