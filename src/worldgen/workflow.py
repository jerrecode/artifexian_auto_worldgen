from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
import hashlib
import json
import os
import pickle
import tempfile
import time
from typing import Any, Iterable


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(slots=True, frozen=True)
class StageSpec:
    name: str
    dependencies: tuple[str, ...] = ()
    config_sections: tuple[str, ...] = ()
    version: str = "1"
    estimated_memory_mb: int = 0
    preferred_backend: str = "serial"


@dataclass(slots=True)
class StageRecord:
    name: str
    key: str
    seconds: float
    loaded_from_checkpoint: bool
    checkpoint_path: str | None = None
    size_bytes: int | None = None


class StageRegistry:
    """Small DAG registry used by tooling, validation and future stage scheduling."""

    def __init__(self) -> None:
        self._specs: dict[str, StageSpec] = {}

    def register(self, spec: StageSpec) -> None:
        if spec.name in self._specs:
            raise KeyError(f"stage already registered: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> StageSpec:
        return self._specs[name]

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def topological_order(self) -> tuple[str, ...]:
        temporary: set[str] = set()
        permanent: set[str] = set()
        out: list[str] = []

        def visit(name: str) -> None:
            if name in permanent:
                return
            if name in temporary:
                raise ValueError(f"stage dependency cycle involving {name!r}")
            if name not in self._specs:
                raise KeyError(f"unknown stage dependency: {name}")
            temporary.add(name)
            for dep in self._specs[name].dependencies:
                visit(dep)
            temporary.remove(name)
            permanent.add(name)
            out.append(name)

        for name in self._specs:
            visit(name)
        return tuple(out)


class StageCheckpointStore:
    """Atomic content-addressed checkpoint storage for arbitrary stage values.

    Stage values are pickle protocol 5 snapshots. They are deliberately internal
    execution artifacts rather than long-term interchange files; public world
    outputs continue to use NPZ/JSON/GeoJSON and optional scientific stores.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.root / "objects"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self._index = self._load_index()

    def _load_index(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") == CHECKPOINT_SCHEMA_VERSION:
                return payload
        except (OSError, ValueError, TypeError):
            pass
        return {"schema_version": CHECKPOINT_SCHEMA_VERSION, "stages": {}}

    def _flush_index(self) -> None:
        payload = json.dumps(self._index, indent=2, sort_keys=True)
        fd, tmp_name = tempfile.mkstemp(prefix="index-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, self.index_path)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    def path_for_key(self, key: str) -> Path:
        return self.data_dir / f"{key}.pkl"

    def has(self, key: str) -> bool:
        entry = self._index.get("stages", {}).get(key)
        return bool(entry and self.path_for_key(key).is_file())

    def load(self, key: str) -> Any:
        path = self.path_for_key(key)
        with path.open("rb") as f:
            return pickle.load(f)

    def save(self, stage_name: str, key: str, value: Any) -> tuple[Path, int]:
        path = self.path_for_key(key)
        fd, tmp_name = tempfile.mkstemp(prefix=f"{stage_name}-", suffix=".tmp", dir=self.data_dir)
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(value, f, protocol=5)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
        size = path.stat().st_size
        self._index.setdefault("stages", {})[key] = {
            "name": stage_name,
            "path": str(path.relative_to(self.root)),
            "size_bytes": size,
            "created_at": time.time(),
        }
        self._flush_index()
        return path, size

    def clear(self) -> None:
        for path in self.data_dir.glob("*.pkl"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        self._index = {"schema_version": CHECKPOINT_SCHEMA_VERSION, "stages": {}}
        self._flush_index()


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _plain(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def canonical_hash(value: Any, *, digest_size: int = 20) -> str:
    raw = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=digest_size).hexdigest()


def _sections_for_stage(name: str) -> tuple[str, ...]:
    """Return configuration sections that directly control a pipeline stage.

    Dependency keys are chained separately, so a changed upstream stage also
    invalidates all of its descendants without making every stage depend on the
    complete configuration.
    """
    if name == "astronomy":
        return ("astronomy",)
    if name == "tectonics":
        return ("resolution", "tectonics", "noise", "astronomy")
    if name.startswith("terrain"):
        return ("terrain", "tectonics", "simulation")
    if name == "noise_cache":
        return ("noise", "resolution")
    if name.startswith("ocean"):
        return ("ocean", "terrain", "simulation")
    if name.startswith("climate"):
        return ("climate", "terrain", "astronomy", "simulation")
    if name.startswith("geology"):
        return ("noise",)
    if name.startswith("surface") and name != "surface_appearance":
        return ("hydrology", "noise", "simulation")
    if name == "hydrology_final":
        return ("hydrology",)
    if name == "weather":
        return ("weather",)
    if name == "surface_appearance":
        return ("appearance",)
    if name == "resources":
        return ("resources",)
    if name == "society":
        return ("society",)
    if name == "output":
        return ("output",)
    return ()


def stage_key(
    *,
    stage_name: str,
    seed: int,
    config: Any,
    dependency_keys: Iterable[str] = (),
    implementation_version: str = "1",
) -> str:
    config_dict = config.to_dict() if hasattr(config, "to_dict") else _plain(config)
    sections = _sections_for_stage(stage_name)
    relevant = {name: config_dict.get(name) for name in sections}
    payload = {
        "schema": CHECKPOINT_SCHEMA_VERSION,
        "stage": stage_name,
        "implementation_version": implementation_version,
        "seed": int(seed),
        "config": relevant,
        "dependencies": list(dependency_keys),
    }
    return canonical_hash(payload, digest_size=24)
