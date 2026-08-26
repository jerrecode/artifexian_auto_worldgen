from __future__ import annotations

from dataclasses import dataclass
import weakref

import numpy as np

from .grid import connected_components_spherical
from .mathops import optional_njit


@optional_njit(cache=True)
def _accumulate_kernel(order: np.ndarray, receiver: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = values.copy()
    for k in range(order.size):
        i = int(order[k])
        j = int(receiver[i])
        if j >= 0:
            out[j] += out[i]
    return out


@optional_njit(cache=True)
def _propagate_roots_kernel(reverse_order: np.ndarray, receiver: np.ndarray, roots: np.ndarray) -> np.ndarray:
    out = roots.copy()
    for k in range(reverse_order.size):
        i = int(reverse_order[k])
        if out[i] != -2:
            continue
        j = int(receiver[i])
        out[i] = 0 if j < 0 else out[j]
    return out


@optional_njit(cache=True)
def _strahler_kernel(order: np.ndarray, receiver: np.ndarray, channel: np.ndarray, donor_count: np.ndarray) -> np.ndarray:
    n = receiver.size
    stream = np.zeros(n, np.uint8)
    max_in = np.zeros(n, np.uint8)
    count_max = np.zeros(n, np.uint8)
    remaining = donor_count.copy()
    for k in range(order.size):
        i = int(order[k])
        if not channel[i]:
            continue
        if stream[i] == 0:
            stream[i] = 1
        j = int(receiver[i])
        if j < 0 or not channel[j]:
            continue
        oi = int(stream[i])
        if oi > int(max_in[j]):
            max_in[j] = oi
            count_max[j] = 1
        elif oi == int(max_in[j]):
            count_max[j] = min(255, int(count_max[j]) + 1)
        remaining[j] -= 1
        if remaining[j] == 0:
            base = max(1, int(max_in[j]))
            stream[j] = min(255, base + (1 if int(count_max[j]) >= 2 else 0))
    for i in range(n):
        if channel[i] and stream[i] == 0:
            stream[i] = 1
    return stream


@dataclass(slots=True)
class DrainageGraph:
    """Reusable acyclic receiver graph for all downstream fluvial calculations."""

    receiver: np.ndarray
    order: np.ndarray
    donor_count: np.ndarray
    unresolved_count: int

    @classmethod
    def from_receiver(cls, flow: np.ndarray) -> "DrainageGraph":
        receiver = np.asarray(flow, dtype=np.int64).ravel()
        n = receiver.size
        donor_count = np.zeros(n, dtype=np.int32)
        valid = receiver >= 0
        if np.any(valid):
            np.add.at(donor_count, receiver[valid], 1)
        remaining = donor_count.copy()
        queue = np.empty(n, dtype=np.int64)
        head = 0
        tail = 0
        sources = np.flatnonzero(remaining == 0)
        queue[:len(sources)] = sources
        tail = len(sources)
        while head < tail:
            i = int(queue[head]); head += 1
            j = int(receiver[i])
            if j >= 0:
                remaining[j] -= 1
                if remaining[j] == 0:
                    queue[tail] = j
                    tail += 1
        unresolved = int(n - tail)
        if unresolved:
            # Filled terrain should be acyclic. Keep execution total and
            # deterministic if a pathological loop reaches this layer; callers
            # can expose unresolved_count as a scientific invariant failure.
            seen = np.zeros(n, dtype=bool)
            seen[queue[:tail]] = True
            rest = np.flatnonzero(~seen)
            queue[tail:tail + len(rest)] = rest
            tail += len(rest)
        return cls(receiver.astype(np.int64, copy=False), queue[:tail].copy(), donor_count, unresolved)

    def accumulate(self, source: np.ndarray) -> np.ndarray:
        values = np.asarray(source, dtype=np.float64).ravel()
        if values.size != self.receiver.size:
            raise ValueError("source size does not match drainage graph")
        return _accumulate_kernel(self.order, self.receiver, values)

    def basin_roots(self, ocean: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        ocean = np.asarray(ocean, dtype=bool)
        if ocean.shape != shape:
            raise ValueError("ocean mask shape mismatch")
        ocean_labels, _ = connected_components_spherical(ocean)
        roots = np.full(self.receiver.size, -2, dtype=np.int32)
        of = ocean.ravel()
        roots[of] = ocean_labels.ravel()[of]
        roots = _propagate_roots_kernel(self.order[::-1], self.receiver, roots)
        roots[roots < 0] = 0
        return roots.reshape(shape)

    def strahler(self, channel: np.ndarray) -> np.ndarray:
        ch = np.asarray(channel, dtype=bool).ravel()
        if ch.size != self.receiver.size:
            raise ValueError("channel size does not match drainage graph")
        # Donor count must count only channel donors for Strahler propagation.
        donors = np.zeros_like(self.donor_count)
        src = np.flatnonzero(ch & (self.receiver >= 0))
        if src.size:
            tgt = self.receiver[src]
            good = ch[tgt]
            np.add.at(donors, tgt[good], 1)
        return _strahler_kernel(self.order, self.receiver, ch, donors)


_GRAPH_CACHE: dict[int, tuple[weakref.ReferenceType[np.ndarray], DrainageGraph]] = {}


def graph_for(flow: np.ndarray) -> DrainageGraph:
    """Reuse graph topology while a receiver ndarray remains alive."""
    key = id(flow)
    cached = _GRAPH_CACHE.get(key)
    if cached is not None and cached[0]() is flow:
        return cached[1]
    graph = DrainageGraph.from_receiver(flow)

    def cleanup(_):
        _GRAPH_CACHE.pop(key, None)

    try:
        ref = weakref.ref(flow, cleanup)
        _GRAPH_CACHE[key] = (ref, graph)
    except TypeError:
        pass
    return graph


def accumulate(z: np.ndarray, flow: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Drop-in replacement for hydrology._accumulate without global elevation sorting."""
    graph = graph_for(flow)
    return graph.accumulate(source).reshape(np.asarray(z).shape)


def basins(flow: np.ndarray, ocean: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return graph_for(flow).basin_roots(ocean, shape)


def strahler_order(flow: np.ndarray, channel: np.ndarray) -> np.ndarray:
    return graph_for(flow).strahler(channel).reshape(np.asarray(channel).shape)


def install_into_hydrology() -> None:
    """Install optimized graph kernels into the existing hydrology module.

    Kept as an explicit compatibility bridge while the public hydrology API is
    stable. The scientific implementation remains in one reusable DrainageGraph
    object rather than three independent Python traversals.
    """
    from . import hydrology
    hydrology._accumulate = accumulate
    hydrology._basins = basins
    hydrology._strahler_order = strahler_order
