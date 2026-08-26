from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
import hashlib
import json
import os
import pickle
import sqlite3
import tempfile
import time
from typing import Any


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(slots=True, frozen=True)
class CheckpointInfo:
    stage: str
    cache_key: str
    path: Path
    file_bytes: int
    created_at: float
    last_access: float
    payload_sha256: str


@dataclass(slots=True, frozen=True)
class CheckpointStats:
    entries: int
    bytes_used: int
    max_bytes: int
    hits: int
    misses: int
    writes: int
    evictions: int


def stable_json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def package_source_fingerprint(package_dir: str | Path | None = None) -> str:
    """Hash installed worldgen Python sources so code changes invalidate checkpoints."""
    root = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix().encode("utf-8")
        h.update(len(rel).to_bytes(4, "little"))
        h.update(rel)
        data = path.read_bytes()
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()


def stage_cache_key(stage: str, config_dict: dict[str, Any], source_fingerprint: str) -> str:
    payload = {
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "stage": stage,
        "config": config_dict,
        "source_fingerprint": source_fingerprint,
    }
    return stable_json_hash(payload)


class CheckpointStore:
    """Transactional, byte-capped stage checkpoint store.

    Payloads use pickle protocol 5 because pipeline stages are Python dataclasses
    containing NumPy arrays and structured metadata. Files are written to a sibling
    temporary path, fsynced, hashed, atomically renamed, then indexed in SQLite.
    A failed process therefore never exposes a half-written checkpoint as valid.
    """

    def __init__(self, root: str | Path, *, max_bytes: int = 64 * 1024**3) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.root / "objects"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max(0, int(max_bytes))
        self.db_path = self.root / "checkpoints.sqlite3"
        self._lock = RLock()
        self._hits = self._misses = self._writes = self._evictions = 0
        self._db = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                stage TEXT NOT NULL,
                cache_key TEXT PRIMARY KEY,
                relpath TEXT NOT NULL UNIQUE,
                file_bytes INTEGER NOT NULL,
                payload_sha256 TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_access REAL NOT NULL
            )
            """
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_checkpoint_stage ON checkpoints(stage)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_checkpoint_lru ON checkpoints(last_access)")
        self._db.commit()
        self.reconcile()

    def _filename(self, cache_key: str) -> str:
        return f"{cache_key}.pkl"

    def _row_to_info(self, row) -> CheckpointInfo:
        return CheckpointInfo(
            stage=str(row[0]), cache_key=str(row[1]), path=self.root / row[2],
            file_bytes=int(row[3]), payload_sha256=str(row[4]),
            created_at=float(row[5]), last_access=float(row[6]),
        )

    def info(self, cache_key: str) -> CheckpointInfo | None:
        with self._lock:
            row = self._db.execute(
                "SELECT stage,cache_key,relpath,file_bytes,payload_sha256,created_at,last_access FROM checkpoints WHERE cache_key=?",
                (cache_key,),
            ).fetchone()
        return None if row is None else self._row_to_info(row)

    def get(self, cache_key: str) -> Any | None:
        with self._lock:
            row = self._db.execute(
                "SELECT stage,cache_key,relpath,file_bytes,payload_sha256,created_at,last_access FROM checkpoints WHERE cache_key=?",
                (cache_key,),
            ).fetchone()
            if row is None:
                self._misses += 1
                return None
            info = self._row_to_info(row)
            if not info.path.exists():
                self._db.execute("DELETE FROM checkpoints WHERE cache_key=?", (cache_key,))
                self._db.commit()
                self._misses += 1
                return None
            try:
                data = info.path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if digest != info.payload_sha256:
                    raise ValueError("checkpoint digest mismatch")
                value = pickle.loads(data)
            except Exception:
                # Corrupt/incompatible entries are invalidated rather than poisoning
                # a resumed run.
                self._db.execute("DELETE FROM checkpoints WHERE cache_key=?", (cache_key,))
                self._db.commit()
                try:
                    info.path.unlink()
                except FileNotFoundError:
                    pass
                self._misses += 1
                return None
            self._db.execute("UPDATE checkpoints SET last_access=? WHERE cache_key=?", (time.time(), cache_key))
            self._db.commit()
            self._hits += 1
            return value

    def put(self, stage: str, cache_key: str, value: Any) -> CheckpointInfo | None:
        if self.max_bytes == 0:
            return None
        filename = self._filename(cache_key)
        path = self.data_dir / filename
        tmp = self.data_dir / f".{filename}.tmp-{os.getpid()}-{time.time_ns()}"
        data = pickle.dumps(value, protocol=5)
        digest = hashlib.sha256(data).hexdigest()
        with self._lock:
            with tmp.open("wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            now = time.time()
            size = path.stat().st_size
            self._db.execute(
                """
                INSERT INTO checkpoints(stage,cache_key,relpath,file_bytes,payload_sha256,created_at,last_access)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(cache_key) DO UPDATE SET
                  stage=excluded.stage, relpath=excluded.relpath,
                  file_bytes=excluded.file_bytes, payload_sha256=excluded.payload_sha256,
                  last_access=excluded.last_access
                """,
                (stage, cache_key, str(path.relative_to(self.root)), int(size), digest, now, now),
            )
            self._db.commit()
            self._writes += 1
            self.prune()
            return self.info(cache_key)

    def invalidate_stage(self, stage: str) -> int:
        with self._lock:
            rows = self._db.execute("SELECT cache_key,relpath FROM checkpoints WHERE stage=?", (stage,)).fetchall()
            for _, rel in rows:
                try:
                    (self.root / rel).unlink()
                except FileNotFoundError:
                    pass
            self._db.execute("DELETE FROM checkpoints WHERE stage=?", (stage,))
            self._db.commit()
            return len(rows)

    def clear(self) -> int:
        with self._lock:
            rows = self._db.execute("SELECT relpath FROM checkpoints").fetchall()
            for (rel,) in rows:
                try:
                    (self.root / rel).unlink()
                except FileNotFoundError:
                    pass
            self._db.execute("DELETE FROM checkpoints")
            self._db.commit()
            return len(rows)

    def disk_usage_bytes(self) -> int:
        with self._lock:
            row = self._db.execute("SELECT COALESCE(SUM(file_bytes),0) FROM checkpoints").fetchone()
        return int(row[0] or 0)

    def prune(self) -> int:
        removed = 0
        with self._lock:
            usage = self.disk_usage_bytes()
            while usage > self.max_bytes:
                row = self._db.execute(
                    "SELECT cache_key,relpath,file_bytes FROM checkpoints ORDER BY last_access ASC, created_at ASC LIMIT 1"
                ).fetchone()
                if row is None:
                    break
                key, rel, size = row
                try:
                    (self.root / rel).unlink()
                except FileNotFoundError:
                    pass
                self._db.execute("DELETE FROM checkpoints WHERE cache_key=?", (key,))
                self._db.commit()
                usage -= int(size)
                removed += 1
                self._evictions += 1
        return removed

    def reconcile(self) -> None:
        """Drop metadata for missing files and orphan payload files after crashes."""
        with self._lock:
            rows = self._db.execute("SELECT cache_key,relpath FROM checkpoints").fetchall()
            referenced: set[Path] = set()
            for key, rel in rows:
                p = self.root / rel
                referenced.add(p.resolve())
                if not p.exists():
                    self._db.execute("DELETE FROM checkpoints WHERE cache_key=?", (key,))
            self._db.commit()
            for p in self.data_dir.glob("*.pkl"):
                if p.resolve() not in referenced:
                    try:
                        p.unlink()
                    except FileNotFoundError:
                        pass
            for p in self.data_dir.glob(".*.tmp-*"):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass

    def stats(self) -> CheckpointStats:
        with self._lock:
            entries = int(self._db.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0])
            bytes_used = self.disk_usage_bytes()
            return CheckpointStats(entries, bytes_used, self.max_bytes, self._hits, self._misses, self._writes, self._evictions)

    def close(self) -> None:
        with self._lock:
            if getattr(self, "_db", None) is not None:
                self._db.close()
                self._db = None  # type: ignore[assignment]
