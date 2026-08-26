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
from typing import Callable, Mapping

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
    """Crash-transactional random-access array store.

    Payloads are immutable content-addressed ``.npy`` objects. Replacing a logical
    key follows this ordering:

    1. serialize/fsync a new immutable object;
    2. atomically publish that object;
    3. commit the SQLite metadata pointer;
    4. only then delete the old, now-unreferenced object.

    A crash before step 3 preserves the old logical value and may leave only an
    orphan object, which :meth:`reconcile` removes on the next open. A crash after
    step 3 leaves the new logical value valid even if old-object cleanup did not run.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        max_bytes: int = 4 * 1024**3,
        persistent: bool = False,
        failure_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self.persistent = bool(persistent or root is not None)
        self._failure_injector = failure_injector
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
        self._db.execute("PRAGMA synchronous=FULL")
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
        self.reconcile()

    def __enter__(self) -> "MappedArrayStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        return None

    def _fail(self, stage: str) -> None:
        hook = self._failure_injector
        if hook is not None:
            hook(stage)

    @staticmethod
    def _key_hash(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def _object_path(self, key: str, content_sha256: str) -> Path:
        return self.data_dir / f"{self._key_hash(key)}-{content_sha256}.npy"

    def put(self, key: str, array: np.ndarray, *, overwrite: bool = True) -> StoredArray:
        """Transactionally replace a logical array key without destroying old data."""
        if not key:
            raise ValueError("Array key must be non-empty")
        arr = np.asarray(array)
        tmp = self.data_dir / f".object.tmp-{os.getpid()}-{time.time_ns()}"

        with self._lock:
            row = self._db.execute(
                "SELECT relpath, created_at FROM arrays WHERE key=?", (key,)
            ).fetchone()
            if row is not None and not overwrite:
                raise KeyError(f"Array already exists: {key}")
            old_relpath = None if row is None else str(row[0])
            old_created = None if row is None else float(row[1])
            self._fail("before_object_write")
            object_path: Path | None = None
            metadata_committed = False
            try:
                with tmp.open("wb") as f:
                    np.save(f, arr, allow_pickle=False)
                    f.flush()
                    os.fsync(f.fileno())
                self._fail("after_object_write")
                file_bytes = int(tmp.stat().st_size)
                if self.max_bytes <= 0 or file_bytes > self.max_bytes:
                    raise RuntimeError(
                        f"Array {key!r} requires {file_bytes} bytes, exceeding store cap {self.max_bytes}"
                    )

                content_hash = self._file_sha256(tmp)
                object_path = self._object_path(key, content_hash)
                if object_path.exists():
                    tmp.unlink()
                else:
                    os.replace(tmp, object_path)
                    self._fsync_directory(self.data_dir)
                self._fail("after_object_publish")

                now = time.time()
                created = old_created if old_created is not None else now
                self._db.execute("BEGIN IMMEDIATE")
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
                    (
                        key,
                        str(object_path.relative_to(self.root)),
                        json.dumps(arr.shape),
                        arr.dtype.str,
                        int(arr.nbytes),
                        file_bytes,
                        created,
                        now,
                    ),
                )
                self._fail("before_db_commit")
                self._db.commit()
                metadata_committed = True
                self._fail("after_db_commit")

                if old_relpath is not None and old_relpath != str(object_path.relative_to(self.root)):
                    old_path = self.root / old_relpath
                    self._fail("during_old_object_delete")
                    try:
                        old_path.unlink()
                        self._fsync_directory(old_path.parent)
                    except FileNotFoundError:
                        pass

                self.prune()
                info = self.info(key)
                if info is None:
                    raise RuntimeError(
                        f"Array {key!r} was evicted immediately because the store cap is too small"
                    )
                return info
            except BaseException:
                # SQLite may have an active transaction if the injected/real failure
                # occurred after BEGIN but before commit.
                if not metadata_committed:
                    try:
                        self._db.rollback()
                    except sqlite3.Error:
                        pass
                raise
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

    def reconcile(self) -> dict[str, int]:
        """Remove stale metadata and orphan immutable objects after interrupted writes."""
        with self._lock:
            rows = self._db.execute("SELECT key, relpath FROM arrays").fetchall()
            referenced = {str(relpath) for _, relpath in rows}
            missing_keys = [key for key, relpath in rows if not (self.root / relpath).is_file()]
            if missing_keys:
                self._db.executemany("DELETE FROM arrays WHERE key=?", [(k,) for k in missing_keys])
                self._db.commit()

            orphaned = 0
            for path in self.data_dir.glob("*.npy"):
                rel = str(path.relative_to(self.root))
                if rel not in referenced:
                    try:
                        path.unlink()
                        orphaned += 1
                    except FileNotFoundError:
                        pass
            for path in self.data_dir.glob(".object.tmp-*"):
                try:
                    path.unlink()
                    orphaned += 1
                except FileNotFoundError:
                    pass
            if orphaned:
                self._fsync_directory(self.data_dir)
            return {"missing_rows_removed": len(missing_keys), "orphan_files_removed": orphaned}

    def delete(self, key: str) -> bool:
        """Delete metadata first; a crash during payload cleanup leaves only an orphan."""
        with self._lock:
            row = self._db.execute("SELECT relpath FROM arrays WHERE key=?", (key,)).fetchone()
            if row is None:
                return False
            path = self.root / row[0]
            self._db.execute("DELETE FROM arrays WHERE key=?", (key,))
            self._db.commit()
            try:
                path.unlink()
                self._fsync_directory(path.parent)
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
