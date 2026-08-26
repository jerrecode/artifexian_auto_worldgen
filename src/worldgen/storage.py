from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
import hashlib
import json
import os
import sqlite3
import tempfile
import time
from typing import Mapping

import numpy as np


@dataclass(slots=True, frozen=True)
class StoredArray:
    key: str
    path: Path
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    file_bytes: int
    created_at: float
    last_access: float


class MappedArrayStore(AbstractContextManager["MappedArrayStore"]):
    """Random-access NumPy array store backed by .npy files + SQLite metadata.

    Arrays are opened with ``numpy.load(..., mmap_mode=...)`` so callers can slice
    large datasets without loading the complete file. A byte cap and LRU pruning
    keep temporary/persistent stores bounded.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        max_bytes: int = 4 * 1024**3,
        persistent: bool = False,
    ) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self.persistent = bool(persistent or root is not None)
        self._temp: tempfile.TemporaryDirectory[str] | None = None
        if root is None:
            self._temp = tempfile.TemporaryDirectory(prefix="worldgen-arrays-")
            self.root = Path(self._temp.name)
        else:
            self.root = Path(root).expanduser().resolve()
            self.root.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.root / "arrays"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "index.sqlite3"
        self._lock = RLock()
        self._db = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS arrays (
                key TEXT PRIMARY KEY,
                relpath TEXT NOT NULL UNIQUE,
                shape_json TEXT NOT NULL,
                dtype TEXT NOT NULL,
                nbytes INTEGER NOT NULL,
                file_bytes INTEGER NOT NULL,
                created_at REAL NOT NULL,
                last_access REAL NOT NULL
            )
            """
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_arrays_lru ON arrays(last_access)")
        self._db.commit()

    def __enter__(self) -> "MappedArrayStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        return None

    @staticmethod
    def _filename_for_key(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest() + ".npy"

    def put(self, key: str, array: np.ndarray, *, overwrite: bool = True) -> StoredArray:
        """Atomically store an array while preserving existing data on rejection.

        The new payload is fully serialized and size-checked before replacing the
        currently indexed file.  This matters for bounded stores: an oversized
        replacement must not destroy a valid smaller value merely because the LRU
        pruning pass would evict the replacement immediately afterwards.
        """
        if not key:
            raise ValueError("Array key must be non-empty")
        arr = np.asarray(array)
        filename = self._filename_for_key(key)
        path = self.data_dir / filename
        tmp = self.data_dir / (filename + f".tmp-{os.getpid()}-{time.time_ns()}")
        with self._lock:
            exists = self._db.execute("SELECT 1 FROM arrays WHERE key=?", (key,)).fetchone() is not None
            if exists and not overwrite:
                raise KeyError(f"Array already exists: {key}")
            try:
                with tmp.open("wb") as f:
                    np.save(f, arr, allow_pickle=False)
                    f.flush()
                    os.fsync(f.fileno())
                file_bytes = tmp.stat().st_size
                if self.max_bytes <= 0 or file_bytes > self.max_bytes:
                    raise RuntimeError(
                        f"Array {key!r} requires {file_bytes} bytes, exceeding store cap {self.max_bytes}"
                    )
                os.replace(tmp, path)
                now = time.time()
                self._db.execute(
                    """
                    INSERT INTO arrays(key, relpath, shape_json, dtype, nbytes, file_bytes, created_at, last_access)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        relpath=excluded.relpath,
                        shape_json=excluded.shape_json,
                        dtype=excluded.dtype,
                        nbytes=excluded.nbytes,
                        file_bytes=excluded.file_bytes,
                        last_access=excluded.last_access
                    """,
                    (key, str(path.relative_to(self.root)), json.dumps(arr.shape), arr.dtype.str,
                     int(arr.nbytes), int(file_bytes), now, now),
                )
                self._db.commit()
                self.prune()
                info = self.info(key)
                if info is None:
                    raise RuntimeError(f"Array {key!r} was evicted immediately because the store cap is too small")
                return info
            finally:
                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass

    def open(self, key: str, *, mode: str = "r") -> np.memmap:
        if mode not in ("r", "r+", "c"):
            raise ValueError("mode must be one of 'r', 'r+', or 'c'")
        with self._lock:
            row = self._db.execute("SELECT relpath FROM arrays WHERE key=?", (key,)).fetchone()
            if row is None:
                raise KeyError(key)
            path = self.root / row[0]
            if not path.exists():
                self._db.execute("DELETE FROM arrays WHERE key=?", (key,))
                self._db.commit()
                raise KeyError(f"Array metadata exists but file is missing: {key}")
            self._db.execute("UPDATE arrays SET last_access=? WHERE key=?", (time.time(), key))
            self._db.commit()
        return np.load(path, mmap_mode=mode, allow_pickle=False)

    def info(self, key: str) -> StoredArray | None:
        with self._lock:
            row = self._db.execute(
                "SELECT relpath, shape_json, dtype, nbytes, file_bytes, created_at, last_access FROM arrays WHERE key=?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return StoredArray(
            key=key,
            path=self.root / row[0],
            shape=tuple(json.loads(row[1])),
            dtype=row[2],
            nbytes=int(row[3]),
            file_bytes=int(row[4]),
            created_at=float(row[5]),
            last_access=float(row[6]),
        )

    def keys(self) -> tuple[str, ...]:
        with self._lock:
            rows = self._db.execute("SELECT key FROM arrays ORDER BY key").fetchall()
        return tuple(row[0] for row in rows)

    def disk_usage_bytes(self) -> int:
        with self._lock:
            row = self._db.execute("SELECT COALESCE(SUM(file_bytes), 0) FROM arrays").fetchone()
        return int(row[0] or 0)

    def delete(self, key: str) -> bool:
        with self._lock:
            row = self._db.execute("SELECT relpath FROM arrays WHERE key=?", (key,)).fetchone()
            if row is None:
                return False
            path = self.root / row[0]
            self._db.execute("DELETE FROM arrays WHERE key=?", (key,))
            self._db.commit()
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return True

    def prune(self) -> int:
        removed = 0
        if self.max_bytes <= 0:
            for key in self.keys():
                removed += int(self.delete(key))
            return removed
        with self._lock:
            usage = self.disk_usage_bytes()
            while usage > self.max_bytes:
                row = self._db.execute(
                    "SELECT key, file_bytes FROM arrays ORDER BY last_access ASC, created_at ASC LIMIT 1"
                ).fetchone()
                if row is None:
                    break
                key, file_bytes = row
                if self.delete(key):
                    usage -= int(file_bytes)
                    removed += 1
                else:
                    break
        return removed

    def close(self) -> None:
        with self._lock:
            if getattr(self, "_db", None) is not None:
                self._db.close()
                self._db = None  # type: ignore[assignment]
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None


def store_array_mapping(
    arrays: Mapping[str, np.ndarray],
    root: str | Path,
    *,
    max_bytes: int = 16 * 1024**3,
) -> MappedArrayStore:
    store = MappedArrayStore(root, max_bytes=max_bytes, persistent=True)
    try:
        for key, value in arrays.items():
            store.put(key, np.asarray(value))
        return store
    except Exception:
        store.close()
        raise
