from __future__ import annotations

"""Runtime coordination for sparse planetary tile generation.

Persistent tile files are authoritative and independent of process memory. This
module adds execution policy needed by interactive clients: per-tile request
coalescing, a byte-bounded decoded RAM LRU, persistent disk quotas, and cache
statistics. A tile can therefore stay permanently precomputed on disk while its
decoded data is repeatedly loaded into and evicted from RAM as camera locality
changes.

A deliberately precomputed complete prefix can be pinned. The persistent LRU then
protects every generated product at or below that z-level while deeper opportunistic
content remains evictable. RAM residency remains independent: pinned files are not
implicitly pinned in memory.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from threading import Lock, RLock
from typing import Hashable, Iterator, Sequence

import numpy as np

from .planet_tiles import PlanetTilePyramid, TileKey, TileResult
from .tile_memory import ResidentTileCacheStats, ResidentTileMemoryCache
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
    """Serialize work for one logical key while allowing unrelated keys in parallel."""

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

    Runtime access changes modification time only for evictable products. Pinned
    precomputed files are intentionally left untouched on reads, so a completed
    static prefix behaves as archival content rather than a mutable disk cache.
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
        """Advance LRU time for evictable files without mutating pinned static files."""
        with self._lock:
            pin = self._pin()
            for value in paths:
                path = Path(value)
                if pin is not None and path_is_pinned(self.root, path, pin):
                    continue
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

        Files in ``protected`` are kept for the current response. Pinned prefix files
        are always kept. If those protected classes alone exceed the configured disk
        quota, persistence wins and the cache remains over budget.
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
    """Concurrency-safe static-file loader with bounded process RAM residency.

    Disk and RAM are independent tiers:

    * ``PlanetTilePyramid`` owns deterministic static files;
    * ``PersistentTileFileLRU`` may evict only unpinned opportunistic disk products;
    * ``ResidentTileMemoryCache`` owns temporary decoded arrays/bytes in process RAM.

    Evicting RAM never deletes disk data. Evicting disk never invalidates an object
    already returned to a caller. Callers should release their own references when a
    tile leaves the active viewport if prompt physical-memory reclamation is desired.
    """

    DEFAULT_MEMORY_CACHE_BYTES = 256 * 1024**2

    def __init__(
        self,
        pyramid: PlanetTilePyramid,
        *,
        disk_cache_max_bytes: int | None = None,
        memory_cache_max_bytes: int = DEFAULT_MEMORY_CACHE_BYTES,
    ) -> None:
        self.pyramid = pyramid
        self.coalescer = KeyedRequestCoalescer()
        self.disk_cache = PersistentTileFileLRU(
            pyramid.root, max_bytes=disk_cache_max_bytes
        )
        self.memory_cache = ResidentTileMemoryCache(max_bytes=memory_cache_max_bytes)

    @staticmethod
    def _request_key(key: TileKey) -> tuple[str, int, int, int]:
        key.validate()
        return key.face, int(key.level), int(key.x), int(key.y)

    @classmethod
    def _field_memory_key(cls, key: TileKey, field: str) -> tuple[object, ...]:
        return ("field", *cls._request_key(key), str(field))

    def _product_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser().resolve()
        root = self.pyramid.root.resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"tile product must be below {root}: {path}") from exc
        return path

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

    def load_field(self, key: TileKey, field: str, *, generate: bool = True) -> np.ndarray:
        """Load a scientific field into the byte-bounded decoded RAM cache."""
        memory_key = self._field_memory_key(key, field)
        cached = self.memory_cache.get(memory_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        request_key = self._request_key(key)
        with self.coalescer.hold(("load-field", *request_key, str(field))):
            # Another waiter may have populated the resident cache while this caller
            # was blocked on the keyed lock.
            cached = self.memory_cache.peek(memory_key)
            if cached is not None:
                return cached  # type: ignore[return-value]

            path = self.pyramid._field_path(key, field)
            if not path.exists():
                if not generate:
                    raise FileNotFoundError(path)
                self.pyramid.generate_tile(key, (field,))

            values = np.load(path, allow_pickle=False)
            if not isinstance(values, np.ndarray):
                values = np.asarray(values)
            # Static tile fields are immutable runtime inputs. Read-only arrays catch
            # accidental in-place modification that would otherwise diverge RAM from
            # the authoritative file.
            try:
                values.flags.writeable = False
            except ValueError:
                pass
            self.memory_cache.put(memory_key, values, size_bytes=int(values.nbytes))
            metadata = self.pyramid._metadata_path(key)
            self.disk_cache.touch((path, metadata))
            self.disk_cache.prune(protected=(path, metadata))
            return values

    def load_product_bytes(self, path: str | Path) -> bytes:
        """Load any static tile product (PNG/mesh/vector/etc.) through the RAM LRU."""
        product = self._product_path(path)
        memory_key = ("product-bytes", str(product))
        cached = self.memory_cache.get(memory_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        with self.coalescer.hold(memory_key):
            cached = self.memory_cache.peek(memory_key)
            if cached is not None:
                return cached  # type: ignore[return-value]
            payload = product.read_bytes()
            self.memory_cache.put(memory_key, payload, size_bytes=len(payload))
            self.disk_cache.touch((product,))
            self.disk_cache.prune(protected=(product,))
            return payload

    def release_field(self, key: TileKey, field: str) -> bool:
        """Drop the runtime cache reference for one decoded field; keep its file."""
        return self.memory_cache.discard(self._field_memory_key(key, field))

    def release_tile(self, key: TileKey) -> int:
        """Drop all resident objects belonging to one tile address."""
        address = self._request_key(key)

        def belongs(cache_key: Hashable) -> bool:
            if not isinstance(cache_key, tuple):
                return False
            if len(cache_key) >= 5 and cache_key[0] == "field":
                return tuple(cache_key[1:5]) == address
            if len(cache_key) >= 2 and cache_key[0] == "product-bytes":
                level_token = f"/z{key.level:02d}/"
                face_token = f"/{key.face}/"
                x_token = f"/x{key.x:08d}/"
                y_token = f"/y{key.y:08d}."
                text = str(cache_key[1]).replace("\\", "/")
                return all(token in text for token in (level_token, face_token, x_token, y_token))
            return False

        return self.memory_cache.discard_where(belongs)

    def release_product(self, path: str | Path) -> bool:
        product = self._product_path(path)
        return self.memory_cache.discard(("product-bytes", str(product)))

    def clear_memory_cache(self) -> int:
        """Unload every runtime-cached tile object without deleting static files."""
        return self.memory_cache.clear()

    def prune(self) -> tuple[Path, ...]:
        return self.disk_cache.prune()

    def cache_stats(self) -> PersistentTileCacheStats:
        return self.disk_cache.stats()

    def memory_cache_stats(self) -> ResidentTileCacheStats:
        return self.memory_cache.stats()


__all__ = [
    "KeyedRequestCoalescer",
    "PersistentTileCacheStats",
    "PersistentTileFileLRU",
    "PlanetTileRuntime",
    "ResidentTileCacheStats",
    "ResidentTileMemoryCache",
]
