from __future__ import annotations

"""Deterministic Priority-Flood backends for spherical raster hydrology."""

import heapq
import importlib.util

import numpy as np

from .mathops import optional_njit

_FLOOD_NEIGHBORS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),             (0, 1),
    (1, -1),  (1, 0),   (1, 1),
)


def _seed_coastal(ocean: np.ndarray, grid) -> np.ndarray:
    land = ~np.asarray(ocean, dtype=bool)
    return land & grid.ops.binary_dilation(ocean, iterations=1)


def priority_flood_reference(
    elev: np.ndarray,
    ocean: np.ndarray,
    grid,
    *,
    epsilon_km: float = 1.0e-7,
) -> np.ndarray:
    """Reference Python/heapq Priority-Flood with spherical pole/seam topology."""
    h, w = elev.shape
    z = np.asarray(elev, dtype=np.float64).copy()
    oc = np.asarray(ocean, dtype=bool)
    if oc.shape != z.shape:
        raise ValueError("ocean shape must match elevation")
    visited = oc.copy()
    heap: list[tuple[float, int, int]] = []
    coastal = _seed_coastal(oc, grid)
    ys, xs = np.where(coastal)
    for y, x in zip(ys.tolist(), xs.tolist()):
        visited[y, x] = True
        heapq.heappush(heap, (float(z[y, x]), y, x))
    if not heap:
        land = ~oc
        for x in range(w):
            for y in (0, h - 1):
                if land[y, x] and not visited[y, x]:
                    visited[y, x] = True
                    heapq.heappush(heap, (float(z[y, x]), y, x))
    eps = max(float(epsilon_km), 0.0)
    while heap:
        cur, y, x = heapq.heappop(heap)
        for dy, dx in _FLOOD_NEIGHBORS:
            ny = y + dy
            nx = x + dx
            if ny < 0:
                ny = -ny - 1
                nx += w // 2
            elif ny >= h:
                ny = 2 * h - ny - 1
                nx += w // 2
            nx %= w
            if visited[ny, nx]:
                continue
            visited[ny, nx] = True
            nz = float(z[ny, nx])
            if nz <= cur:
                nz = cur + eps
                z[ny, nx] = nz
            heapq.heappush(heap, (nz, ny, nx))
    return z


@optional_njit(cache=True)
def _less(key_a: float, node_a: int, key_b: float, node_b: int) -> bool:
    return key_a < key_b or (key_a == key_b and node_a < node_b)


@optional_njit(cache=True)
def _heap_push(
    keys: np.ndarray,
    nodes: np.ndarray,
    size: int,
    key: float,
    node: int,
) -> int:
    i = size
    size += 1
    while i > 0:
        p = (i - 1) // 2
        if not _less(key, node, keys[p], nodes[p]):
            break
        keys[i] = keys[p]
        nodes[i] = nodes[p]
        i = p
    keys[i] = key
    nodes[i] = node
    return size


@optional_njit(cache=True)
def _heap_pop(keys: np.ndarray, nodes: np.ndarray, size: int):
    root_key = keys[0]
    root_node = nodes[0]
    size -= 1
    if size > 0:
        key = keys[size]
        node = nodes[size]
        i = 0
        while True:
            left = 2 * i + 1
            if left >= size:
                break
            right = left + 1
            child = left
            if right < size and _less(keys[right], nodes[right], keys[left], nodes[left]):
                child = right
            if not _less(keys[child], nodes[child], key, node):
                break
            keys[i] = keys[child]
            nodes[i] = nodes[child]
            i = child
        keys[i] = key
        nodes[i] = node
    return root_key, root_node, size


@optional_njit(cache=True)
def _priority_flood_heap_kernel(
    elev: np.ndarray,
    ocean: np.ndarray,
    coastal: np.ndarray,
    epsilon_km: float,
) -> np.ndarray:
    h, w = elev.shape
    n = h * w
    z = elev.copy()
    visited = ocean.copy()
    heap_keys = np.empty(n, dtype=np.float64)
    heap_nodes = np.empty(n, dtype=np.int64)
    size = 0

    # np.where in the reference implementation yields row-major ordering. Iterating
    # row-major here plus (key,node) heap ordering produces the same deterministic
    # tie resolution as Python tuples (key,y,x).
    for y in range(h):
        for x in range(w):
            if coastal[y, x]:
                visited[y, x] = True
                node = y * w + x
                size = _heap_push(heap_keys, heap_nodes, size, z[y, x], node)

    if size == 0:
        for x in range(w):
            for k in range(2):
                y = 0 if k == 0 else h - 1
                if not ocean[y, x] and not visited[y, x]:
                    visited[y, x] = True
                    node = y * w + x
                    size = _heap_push(heap_keys, heap_nodes, size, z[y, x], node)

    eps = epsilon_km if epsilon_km > 0.0 else 0.0
    dys = (-1, -1, -1, 0, 0, 1, 1, 1)
    dxs = (-1, 0, 1, -1, 1, -1, 0, 1)
    while size > 0:
        cur, node, size = _heap_pop(heap_keys, heap_nodes, size)
        y = node // w
        x = node - y * w
        for j in range(8):
            ny = y + dys[j]
            nx = x + dxs[j]
            if ny < 0:
                ny = -ny - 1
                nx += w // 2
            elif ny >= h:
                ny = 2 * h - ny - 1
                nx += w // 2
            nx %= w
            if visited[ny, nx]:
                continue
            visited[ny, nx] = True
            nz = z[ny, nx]
            if nz <= cur:
                nz = cur + eps
                z[ny, nx] = nz
            size = _heap_push(heap_keys, heap_nodes, size, nz, ny * w + nx)
    return z


def numba_priority_flood_available() -> bool:
    return importlib.util.find_spec("numba") is not None


def priority_flood(
    elev: np.ndarray,
    ocean: np.ndarray,
    grid,
    *,
    backend: str = "auto",
    epsilon_km: float = 1.0e-7,
) -> np.ndarray:
    """Fill depressions using the fastest available deterministic backend.

    ``auto`` uses the compiled custom binary heap when Numba is installed and the
    dependency-free heapq implementation otherwise. ``reference`` always selects
    the Python oracle, while ``numba`` requires Numba explicitly.
    """
    mode = str(backend).strip().lower()
    if mode not in {"auto", "reference", "numba"}:
        raise ValueError("priority-flood backend must be auto, reference, or numba")
    if mode == "reference" or (mode == "auto" and not numba_priority_flood_available()):
        return priority_flood_reference(elev, ocean, grid, epsilon_km=epsilon_km)
    if not numba_priority_flood_available():
        raise RuntimeError("priority-flood backend 'numba' requested but Numba is not installed")

    z = np.asarray(elev, dtype=np.float64)
    oc = np.asarray(ocean, dtype=np.bool_)
    if z.ndim != 2 or oc.shape != z.shape:
        raise ValueError("elevation and ocean must be equal-shaped 2-D arrays")
    coastal = np.asarray(_seed_coastal(oc, grid), dtype=np.bool_)
    return _priority_flood_heap_kernel(z, oc, coastal, max(float(epsilon_km), 0.0))


__all__ = [
    "priority_flood",
    "priority_flood_reference",
    "numba_priority_flood_available",
]
