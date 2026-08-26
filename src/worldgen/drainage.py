from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .mathops import optional_njit


@optional_njit(cache=True)
def _accumulate_kernel(receiver: np.ndarray, order: np.ndarray, source: np.ndarray) -> np.ndarray:
    out = source.copy()
    for k in range(order.size):
        node = order[k]
        nxt = receiver[node]
        if nxt >= 0:
            out[nxt] += out[node]
    return out


@optional_njit(cache=True)
def _propagate_roots_kernel(receiver: np.ndarray, order: np.ndarray, roots: np.ndarray) -> np.ndarray:
    out = roots.copy()
    for k in range(order.size - 1, -1, -1):
        node = order[k]
        if out[node] != 0:
            continue
        nxt = receiver[node]
        if nxt >= 0:
            out[node] = out[nxt]
    return out


@optional_njit(cache=True)
def _strahler_kernel(receiver: np.ndarray, channel: np.ndarray, order: np.ndarray) -> np.ndarray:
    n = receiver.size
    result = np.zeros(n, np.uint8)
    max_in = np.zeros(n, np.uint8)
    count_max = np.zeros(n, np.uint8)
    for k in range(order.size):
        node = order[k]
        if not channel[node]:
            continue
        base = max_in[node]
        if base == 0:
            result[node] = 1
        else:
            result[node] = base + (1 if count_max[node] >= 2 else 0)
        nxt = receiver[node]
        if nxt < 0 or not channel[nxt]:
            continue
        val = result[node]
        if val > max_in[nxt]:
            max_in[nxt] = val
            count_max[nxt] = 1
        elif val == max_in[nxt] and count_max[nxt] < 255:
            count_max[nxt] += 1
    return result


def topological_order(receiver: np.ndarray) -> np.ndarray:
    """Return upstream-to-downstream order for a single-receiver acyclic graph.

    Raises ``ValueError`` when receivers are out of bounds or a directed cycle is
    present. The order is reusable for every scalar flux routed over the same
    drainage network, eliminating repeated O(N log N) elevation sorts.
    """
    recv = np.asarray(receiver, dtype=np.int64).ravel()
    n = recv.size
    src = np.flatnonzero(recv >= 0)
    if src.size and (np.any(recv[src] >= n) or np.any(recv[src] < 0)):
        raise ValueError("receiver contains out-of-range index")
    indeg = np.zeros(n, dtype=np.int32)
    if src.size:
        np.add.at(indeg, recv[src], 1)
    queue = np.empty(n, dtype=np.int64)
    roots = np.flatnonzero(indeg == 0)
    qn = roots.size
    queue[:qn] = roots
    head = 0
    while head < qn:
        node = int(queue[head])
        head += 1
        nxt = int(recv[node])
        if nxt >= 0:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue[qn] = nxt
                qn += 1
    if qn != n:
        raise ValueError("receiver graph contains a directed cycle")
    return queue


@dataclass(slots=True)
class DrainageGraph:
    """Reusable topology for hydrological routing and material transport."""

    receiver: np.ndarray
    order: np.ndarray
    shape: tuple[int, int]

    @classmethod
    def from_receiver(cls, receiver: np.ndarray, shape: tuple[int, int]) -> "DrainageGraph":
        recv = np.asarray(receiver, dtype=np.int64).ravel()
        if recv.size != int(shape[0]) * int(shape[1]):
            raise ValueError("receiver size does not match shape")
        return cls(recv, topological_order(recv), tuple(map(int, shape)))

    def accumulate(self, source: np.ndarray, *, dtype=np.float64) -> np.ndarray:
        src = np.asarray(source, dtype=dtype).ravel()
        if src.size != self.receiver.size:
            raise ValueError("source size does not match drainage graph")
        return _accumulate_kernel(self.receiver, self.order, src).reshape(self.shape)

    def basin_roots(self, terminal_labels: np.ndarray) -> np.ndarray:
        roots = np.asarray(terminal_labels, dtype=np.int32).ravel()
        if roots.size != self.receiver.size:
            raise ValueError("terminal_labels size does not match drainage graph")
        return _propagate_roots_kernel(self.receiver, self.order, roots).reshape(self.shape)

    def strahler_order(self, channel: np.ndarray) -> np.ndarray:
        mask = np.asarray(channel, dtype=np.bool_).ravel()
        if mask.size != self.receiver.size:
            raise ValueError("channel size does not match drainage graph")
        return _strahler_kernel(self.receiver, mask, self.order).reshape(self.shape)

    def donor_count(self, subset: np.ndarray | None = None) -> np.ndarray:
        recv = self.receiver
        out = np.zeros(recv.size, dtype=np.int32)
        src = np.flatnonzero(recv >= 0)
        if subset is not None:
            keep = np.asarray(subset, dtype=bool).ravel()
            src = src[keep[src] & keep[recv[src]]]
        if src.size:
            np.add.at(out, recv[src], 1)
        return out.reshape(self.shape)

    def donor_csr(self) -> tuple[np.ndarray, np.ndarray]:
        """Return CSR offsets and donor indices for reverse graph traversal."""
        recv = self.receiver
        n = recv.size
        src = np.flatnonzero(recv >= 0)
        counts = np.zeros(n, dtype=np.int64)
        if src.size:
            np.add.at(counts, recv[src], 1)
        offsets = np.empty(n + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        donors = np.empty(src.size, dtype=np.int64)
        cursor = offsets[:-1].copy()
        for node in src:
            target = recv[node]
            pos = cursor[target]
            donors[pos] = node
            cursor[target] += 1
        return offsets, donors
