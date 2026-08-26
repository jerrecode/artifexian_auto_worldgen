from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from typing import Any, Iterable

import numpy as np
import scipy

from .workflow import StageRecord, canonical_hash


MANIFEST_SCHEMA_VERSION = 1


@dataclass(slots=True)
class RunManifest:
    schema_version: int
    generator_version: str
    seed: int
    config_hash: str
    resolution: tuple[int, int]
    runtime: dict[str, Any]
    environment: dict[str, Any]
    stages: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resolution"] = list(self.resolution)
        return payload


def generator_version() -> str:
    try:
        return importlib.metadata.version("artifexian-auto-worldgen")
    except importlib.metadata.PackageNotFoundError:
        return "0.3.0+source"


def _git_commit_hint() -> str | None:
    for key in ("GITHUB_SHA", "WORLDGEN_GIT_COMMIT", "SOURCE_COMMIT"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def collect_output_inventory(root: str | Path, *, hash_limit_mb: float = 256.0) -> list[dict[str, Any]]:
    base = Path(root)
    limit = max(0, int(hash_limit_mb * 1024**2))
    out: list[dict[str, Any]] = []
    for path in sorted(p for p in base.rglob("*") if p.is_file() and p.name != "run_manifest.json"):
        size = path.stat().st_size
        record: dict[str, Any] = {
            "path": str(path.relative_to(base)),
            "size_bytes": int(size),
        }
        if size <= limit:
            record["sha256"] = _sha256_file(path)
        else:
            record["sha256"] = None
            record["hash_skipped_reason"] = f"file exceeds {hash_limit_mb:g} MiB manifest hashing limit"
        out.append(record)
    return out


def build_run_manifest(
    config: Any,
    *,
    stage_records: Iterable[StageRecord] = (),
    timings: dict[str, float] | None = None,
    runtime_plan: dict[str, Any] | None = None,
    output_root: str | Path | None = None,
) -> RunManifest:
    config_dict = config.to_dict() if hasattr(config, "to_dict") else config
    resolution = (
        int(getattr(config.resolution, "width")),
        int(getattr(config.resolution, "height")),
    )
    env = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "git_commit": _git_commit_hint(),
    }
    runtime = dict(runtime_plan or {})
    if timings is not None:
        runtime["stage_seconds"] = {k: float(v) for k, v in timings.items()}
        runtime["total_stage_seconds"] = float(sum(timings.values()))
    records = [asdict(r) for r in stage_records]
    outputs = collect_output_inventory(output_root) if output_root is not None else []
    return RunManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        generator_version=generator_version(),
        seed=int(config.seed),
        config_hash=canonical_hash(config_dict, digest_size=32),
        resolution=resolution,
        runtime=runtime,
        environment=env,
        stages=records,
        outputs=outputs,
    )


def write_run_manifest(
    path: str | Path,
    config: Any,
    *,
    stage_records: Iterable[StageRecord] = (),
    timings: dict[str, float] | None = None,
    runtime_plan: dict[str, Any] | None = None,
    output_root: str | Path | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(
        config,
        stage_records=stage_records,
        timings=timings,
        runtime_plan=runtime_plan,
        output_root=output_root,
    )
    target.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return target
