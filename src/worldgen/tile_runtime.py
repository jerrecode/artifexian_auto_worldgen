from __future__ import annotations

"""Runtime coordination for sparse planetary tile generation.

The scientific :class:`PlanetTilePyramid` stays deterministic and synchronous. This
module adds execution policy needed by interactive clients: per-tile request
coalescing, persistent byte quotas, LRU eviction, and cache statistics. Keeping these
concerns separate prevents viewer/service concurrency policy from leaking into the
physical tile-generation contract.

A deliberately precomputed complete prefix can be pinned. The persistent LRU then
protects every generated product at or below that z-level while deeper opportunistic
content remains evictable. An explicit precompute request therefore cannot be
silently undone by later interactive navigation.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from threading import Lock, RLock
from typing import Hashable, Iterator, Sequence

from .planet_tiles import PlanetTilePyramid, TileKey, TileResult
from .tile_pins import PinnedPrefix, path_is_pinned, read_pinned_prefix


@dataclass(slots=True, frozen=True)
class PersistentTileCacheStats:
    files: int
    bytes_used: int
    max_bytes: int | None
    evictions: int
    pinned_files: int = 0
    pinned_bytes: int = 0
    pinned_maximum_level: int | None = None


@dataclass(slots=True)
class _KeyLock:
    lock: Lock
    references: int = 0


class KeyedRequestCoalescer:
    """Serialize work for one logical key while allowing unrelated keys in parallel.

    This is intentionally a keyed mutex rather than a memoized Future. A second
    caller waits for the first caller for the same tile, then re-enters the normal
    cache-aware generator. Therefore overlapping field requests cannot perform
    duplicate simultaneous writes, while a later caller may still add a field that
    the first caller did not request.
    """

    def __init__(self) -> None:
        self._guard = RLock()
        self._locks: dict[Hashable, _KeyLock] = {}

    @contextmanager
    def hold(self, key: Hashable) -> Iterator[None]:
        with self._guard:
            entry = self._locks.get(key)
            if entry is None:
                entry = _KeyLock(Lock())
                self._locks[key] = entry
            entry.references += 1
        entry.lock.acquire()
        try:
            yield
        finally:
            entry.lock.release()
            with self._guard:
                entry.references -= 1
                if entry.references == 0 and not entry.lock.locked():
                    self._locks.pop(key, None)

    def active_keys(self) -> int:
        with self._guard:
            return len(self._locks)


class PersistentTileFileLRU:
    """Filesystem LRU quota for generated tile products.

    File modification time is the access clock. Runtime reads/generation call
    :meth:`touch` for returned products. `tileset.json` and `precompute/` control
    metadata are never managed as payload. Temporary dot-files are ignored so an
    interrupted atomic write cannot become a cache candidate.

    If a complete prefix is pinned, any managed product whose path contains a z-level
    at or below the pinned maximum is non-evictable. When the pinned prefix alone is
    larger than the configured byte quota, the cache intentionally remains over
    budget instead of violating the archival pin.
    """

    def __init__(self, root: str | Path, *, max_bytes: int | None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.max_bytes = None if max_bytes is None else max(0, int(max_bytes))
        self._evictions = 0
        self._lock = RLock()

    def _managed_files(self) -> list[Path]:
        if not self.root.exists():
            return []
        control_root = self.root / "precompute"
        files: list[Path] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if path == self.root / "tileset.json":
                continue
            if control_root == path or control_root in path.parents:
                continue
            if path.name.startswith("."):
                continue
            files.append(path)
        return files

    @staticmethod
    def _size(path: Path) -> int:
        try:
            return max(0, int(path.stat().st_size))
        except FileNotFoundError:
            return 0

    def _pin(self) -> PinnedPrefix | None:
        return read_pinned_prefix(self.root)

    def touch(self, paths: Sequence[str | Path]) -> None:
        with self._lock:
            for value in paths:
                path = Path(value)
                try:
                    os.utime(path, None)
                except FileNotFoundError:
                    pass

    def _remove_empty_parents(self, start: Path) -> None:
        parent = start.parent
        while parent != self.root and self.root in parent.parents:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    def prune(self, *, protected: Sequence[str | Path] = ()) -> tuple[Path, ...]:
        """Evict least-recently-used unpinned files until the byte budget is met.

        `protected` files are never deleted during the call. This lets a request
        generate/touch its response and then enforce the quota without deleting the
        exact files being returned. Pinned prefix products are also permanently
        protected until the pin is removed or changed.

        If protected/pinned data alone exceed the whole quota, the cache may remain
        over budget; preservation takes precedence over the soft runtime quota.
        """
        if self.max_bytes is None:
            return ()
        protected_set = {Path(p).resolve() for p in protected}
        with self._lock:
            pin = self._pin()
            entries: list[tuple[int, str, int, Path, bool]] = []
            total = 0
            for path in self._managed_files():
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                size = max(0, int(stat.st_size))
                total += size
                pinned = path_is_pinned(self.root, path, pin) if pin is not None else False
                entries.append((int(stat.st_mtime_ns), str(path), size, path, pinned))
            if total <= self.max_bytes:
                return ()
            entries.sort(key=lambda item: (item[0], item[1]))
            removed: list[Path] = []
            for _mtime, _name, size, path, pinned in entries:
                if total <= self.max_bytes:
                    break
                if pinned:
                    continue
                try:
                    resolved = path.resolve()
                except FileNotFoundError:
                    continue
                if resolved in protected_set:
                    continue
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                total -= size
                self._evictions += 1
                removed.append(path)
                self._remove_empty_parents(path)
            return tuple(removed)

    def stats(self) -> PersistentTileCacheStats:
        with self._lock:
            pin = self._pin()
            files = self._managed_files()
            pinned_files = 0
            pinned_bytes = 0
            total_bytes = 0
            for path in files:
                size = self._size(path)
                total_bytes += size
                if pin is not None and path_is_pinned(self.root, path, pin):
                    pinned_files += 1
                    pinned_bytes += size
            return PersistentTileCacheStats(
                files=len(files),
                bytes_used=total_bytes,
                max_bytes=self.max_bytes,
                evictions=self._evictions,
                pinned_files=pinned_files,
                pinned_bytes=pinned_bytes,
                pinned_maximum_level=(pin.maximum_level if pin is not None else None),
            )


class PlanetTileRuntime:
    """Bounded, concurrency-safe interactive wrapper around `PlanetTilePyramid`."""

    def __init__(
        self,
        pyramid: PlanetTilePyramid,
        *,
        disk_cache_max_bytes: int | None = None,
    ) -> None:
        self.pyramid = pyramid
        self.coalescer = KeyedRequestCoalescer()
        self.disk_cache = PersistentTileFileLRU(
            pyramid.root, max_bytes=disk_cache_max_bytes
        )

    @staticmethod
    def _request_key(key: TileKey) -> tuple[str, int, int, int]:
        key.validate()
        return key.face, int(key.level), int(key.x), int(key.y)

    def generate_tile(
        self, key: TileKey, fields: Sequence[str] = ("elevation_m",)
    ) -> TileResult:
        request_key = self._request_key(key)
        with self.coalescer.hold(request_key):
            result = self.pyramid.generate_tile(key, fields)
            returned = [*result.fields.values(), result.metadata_path]
            self.disk_cache.touch(returned)
            self.disk_cache.prune(protected=returned)
            return result

    def load_field(self, key: TileKey, field: str, *, generate: bool = True):
        request_key = self._request_key(key)
        with self.coalescer.hold(request_key):
            path = self.pyramid._field_path(key, field)
            if not path.exists() and generate:
                self.pyramid.generate_tile(key, (field,))
            values = self.pyramid.load_field(key, field, generate=False)
            metadata = self.pyramid._metadata_path(key)
            self.disk_cache.touch((path, metadata))
            self.disk_cache.prune(protected=(path, metadata))
            return values

    def prune(self) -> tuple[Path, ...]:
        return self.disk_cache.prune()

    def cache_stats(self) -> PersistentTileCacheStats:
        return self.disk_cache.stats()


__all__ = [
    "KeyedRequestCoalescer",
    "PersistentTileCacheStats",
    "PersistentTileFileLRU",
    "PlanetTileRuntime",
]
