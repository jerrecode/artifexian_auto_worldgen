from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
import sys
import time
from typing import Generic, Hashable, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass(slots=True, frozen=True)
class CacheStats:
    items: int
    bytes_used: int
    max_bytes: int
    hits: int
    misses: int
    evictions: int
    hit_rate: float


def estimate_bytes(value: object) -> int:
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return max(0, nbytes)
    try:
        return max(0, int(sys.getsizeof(value)))
    except TypeError:
        return 0


class ByteBoundLRUCache(Generic[K, V]):
    """Thread-safe LRU cache bounded by estimated resident bytes and optional TTL."""

    def __init__(self, max_bytes: int, *, ttl_seconds: float | None = None):
        self.max_bytes = max(0, int(max_bytes))
        self.ttl_seconds = None if ttl_seconds is None else max(0.0, float(ttl_seconds))
        self._data: OrderedDict[K, tuple[V, int, float]] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def _expired(self, born: float, now: float) -> bool:
        return self.ttl_seconds is not None and now - born > self.ttl_seconds

    def get(self, key: K, default=None):
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self._misses += 1
                return default
            value, size, born = item
            if self._expired(born, now):
                self._bytes -= size
                del self._data[key]
                self._misses += 1
                self._evictions += 1
                return default
            self._data.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: K, value: V, *, size_bytes: int | None = None) -> bool:
        """Insert or replace *key* without destroying a valid value on rejection.

        A replacement larger than the cache budget cannot be stored.  The previous
        implementation popped the existing value before making that determination,
        so a failed oversized update silently deleted otherwise valid cached state.
        Capacity rejection is now checked first and is therefore transactional from
        the caller's perspective.
        """
        size = estimate_bytes(value) if size_bytes is None else max(0, int(size_bytes))
        with self._lock:
            if self.max_bytes == 0 or size > self.max_bytes:
                return False
            old = self._data.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            self._data[key] = (value, size, time.monotonic())
            self._bytes += size
            self._evict_to_limit()
            return key in self._data

    def pop(self, key: K, default=None):
        with self._lock:
            item = self._data.pop(key, None)
            if item is None:
                return default
            self._bytes -= item[1]
            return item[0]

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._bytes = 0

    def _evict_to_limit(self) -> None:
        while self._bytes > self.max_bytes and self._data:
            _, (_, size, _) = self._data.popitem(last=False)
            self._bytes -= size
            self._evictions += 1

    def prune_expired(self) -> int:
        if self.ttl_seconds is None:
            return 0
        now = time.monotonic()
        removed = 0
        with self._lock:
            for key in list(self._data):
                _, size, born = self._data[key]
                if self._expired(born, now):
                    del self._data[key]
                    self._bytes -= size
                    self._evictions += 1
                    removed += 1
        return removed

    def keys(self) -> tuple[K, ...]:
        with self._lock:
            return tuple(self._data.keys())

    def stats(self) -> CacheStats:
        with self._lock:
            total = self._hits + self._misses
            return CacheStats(
                items=len(self._data),
                bytes_used=self._bytes,
                max_bytes=self.max_bytes,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                hit_rate=(self._hits / total) if total else 0.0,
            )
