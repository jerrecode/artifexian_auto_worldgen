from __future__ import annotations

"""Explicit byte-bounded RAM residency for static planetary tile products.

Persistent tile files are the authority. This module only owns temporary process
references to decoded arrays/bytes. Removing an entry from this cache never removes
or modifies its backing file; a later request simply reloads the static product.
"""

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Hashable, Iterable

import numpy as np


@dataclass(slots=True, frozen=True)
class ResidentTileCacheStats:
    items: int
    bytes_used: int
    max_bytes: int
    hits: int
    misses: int
    evictions: int
    rejected_oversize: int


@dataclass(slots=True)
class _ResidentEntry:
    value: object
    size_bytes: int


class ResidentTileMemoryCache:
    """Thread-safe process-local LRU for decoded tile products.

    The cache is deliberately independent from persistent disk retention. Values are
    strongly referenced only while resident. Eviction drops the cache's reference;
    callers that still hold the returned object can continue using it safely, and the
    Python object becomes reclaimable when their references are also released.
    """

    def __init__(self, *, max_bytes: int = 256 * 1024**2) -> None:
        limit = int(max_bytes)
        if limit < 0:
            raise ValueError("resident memory max_bytes must be >= 0")
        self.max_bytes = limit
        self._entries: OrderedDict[Hashable, _ResidentEntry] = OrderedDict()
        self._bytes_used = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._rejected_oversize = 0
        self._lock = RLock()

    @staticmethod
    def size_bytes(value: object) -> int:
        if isinstance(value, np.ndarray):
            return max(0, int(value.nbytes))
        if isinstance(value, (bytes, bytearray, memoryview)):
            return len(value)
        nbytes = getattr(value, "nbytes", None)
        if nbytes is not None:
            try:
                return max(0, int(nbytes))
            except (TypeError, ValueError, OverflowError):
                pass
        try:
            return max(0, int(value.__sizeof__()))
        except (AttributeError, TypeError, ValueError, OverflowError):
            return 0

    def get(self, key: Hashable) -> object | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return entry.value

    def peek(self, key: Hashable) -> object | None:
        """Return a resident value without changing hit/miss or LRU statistics."""
        with self._lock:
            entry = self._entries.get(key)
            return None if entry is None else entry.value

    def put(self, key: Hashable, value: object, *, size_bytes: int | None = None) -> bool:
        size = self.size_bytes(value) if size_bytes is None else max(0, int(size_bytes))
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes_used -= previous.size_bytes

            if self.max_bytes == 0 or size > self.max_bytes:
                self._rejected_oversize += 1
                return False

            self._entries[key] = _ResidentEntry(value=value, size_bytes=size)
            self._bytes_used += size
            self._prune_locked()
            return key in self._entries

    def _prune_locked(self) -> None:
        while self._bytes_used > self.max_bytes and self._entries:
            _key, entry = self._entries.popitem(last=False)
            self._bytes_used -= entry.size_bytes
            self._evictions += 1

    def prune(self) -> int:
        with self._lock:
            before = self._evictions
            self._prune_locked()
            return self._evictions - before

    def discard(self, key: Hashable) -> bool:
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                return False
            self._bytes_used -= entry.size_bytes
            return True

    def discard_where(self, predicate) -> int:
        """Drop every cache reference whose key satisfies ``predicate(key)``."""
        with self._lock:
            keys = [key for key in self._entries if predicate(key)]
            for key in keys:
                entry = self._entries.pop(key)
                self._bytes_used -= entry.size_bytes
            return len(keys)

    def clear(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._bytes_used = 0
            return count

    def contains(self, key: Hashable) -> bool:
        with self._lock:
            return key in self._entries

    def keys(self) -> tuple[Hashable, ...]:
        with self._lock:
            return tuple(self._entries.keys())

    def stats(self) -> ResidentTileCacheStats:
        with self._lock:
            return ResidentTileCacheStats(
                items=len(self._entries),
                bytes_used=self._bytes_used,
                max_bytes=self.max_bytes,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                rejected_oversize=self._rejected_oversize,
            )


__all__ = ["ResidentTileCacheStats", "ResidentTileMemoryCache"]
