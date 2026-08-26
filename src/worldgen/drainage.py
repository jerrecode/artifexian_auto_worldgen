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


@optional_njit(cache=True)
def _topological_order_kernel(receiver: np.ndarray):
    """Numba-friendly Kahn traversal returning (queue, count, error_code)."""
    n = receiver.size
    indeg = np.zeros(n, dtype=np.int32)
    for node in range(n):
        nxt = receiver[node]
        if nxt < -1 or nxt >= n:
            return np.empty(0, dtype=np.int64), 0, 1
        if nxt >= 0:
            indeg[nxt] += 1

    queue = np.empty(n, dtype=np.int64)
    qn = 0
    for node in range(n):
        if indeg[node] == 0:
            queue[qn] = node
            qn += 1
    head = 0
    while head < qn:
        node = queue[head]
        head += 1
        nxt = receiver[node]
        if nxt >= 0:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue[qn] = nxt
                qn += 1
    if qn != n:
        return queue, qn, 2
    return queue, qn, 0


@optional_njit(cache=True)
def _donor_csr_kernel(receiver: np.ndarray):
    n = receiver.size
    counts = np.zeros(n, dtype=np.int64)
    edge_count = 0
    for node in range(n):
        nxt = receiver[node]
        if nxt >= 0:
            counts[nxt] += 1
            edge_count += 1
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    for i in range(n):
        offsets[i + 1] = offsets[i] + counts[i]
    donors = np.empty(edge_count, dtype=np.int64)
    cursor = offsets[:-1].copy()
    for node in range(n):
        target = receiver[node]
        if target >= 0:
            pos = cursor[target]
            donors[pos] = node
            cursor[target] += 1
    return offsets, donors


def topological_order(receiver: np.ndarray) -> np.ndarray:
    """Return reusable upstream-to-downstream order in O(N).

    The traversal is JIT compiled when Numba is installed. It is deliberately
    independent of terrain elevation, so one order can route water, drainage area,
    sediment and other scalar fluxes over the same receiver graph.
    """
    recv = np.asarray(receiver, dtype=np.int64).ravel()
    queue, qn, error = _topological_order_kernel(recv)
    if error == 1:
        raise ValueError("receiver contains out-of-range index")
    if error == 2:
        raise ValueError("receiver graph contains a directed cycle")
    return np.asarray(queue[:qn], dtype=np.int64)


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
            if keep.size != recv.size:
                raise ValueError("subset size does not match drainage graph")
            src = src[keep[src] & keep[recv[src]]]
        if src.size:
            np.add.at(out, recv[src], 1)
        return out.reshape(self.shape)

    def donor_csr(self) -> tuple[np.ndarray, np.ndarray]:
        """Return CSR offsets and donors; JIT compiled when available."""
        return _donor_csr_kernel(self.receiver)
