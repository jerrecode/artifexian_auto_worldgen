from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import scipy

from .algorithm_contracts import ENGINE_VERSION, default_algorithm_versions
from .checkpoint import package_source_fingerprint, stable_json_hash
from .identity import build_world_identity
from .atmogen_adapter import atmogen_runtime_metadata


MANIFEST_SCHEMA_VERSION = 2


def _git_commit() -> str | None:
    for key in ("GITHUB_SHA", "WORLDGEN_GIT_COMMIT"):
        value = os.environ.get(key)
        if value:
            return value
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, timeout=2
        ).strip() or None
    except Exception:
        return None


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def output_inventory(root: str | Path, *, with_hashes: bool = False) -> list[dict[str, Any]]:
    base = Path(root)
    result: list[dict[str, Any]] = []
    if not base.exists():
        return result
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        item: dict[str, Any] = {
            "path": path.relative_to(base).as_posix(),
            "bytes": int(path.stat().st_size),
        }
        if with_hashes:
            item["sha256"] = sha256_file(path)
        result.append(item)
    return result


def build_run_manifest(
    *,
    config,
    runtime_plan=None,
    timings: Mapping[str, float] | None = None,
    output_root: str | Path | None = None,
    checkpoint_stats: Mapping[str, Any] | None = None,
    with_output_hashes: bool = False,
    overrides: Mapping[str, Any] | None = None,
    algorithm_versions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    runtime = None
    if runtime_plan is not None:
        runtime = asdict(runtime_plan) if is_dataclass(runtime_plan) else dict(runtime_plan)

    seed_value = getattr(config, "seed", cfg.get("seed", 0))
    seed = int(seed_value) if isinstance(seed_value, int) and not isinstance(seed_value, bool) else seed_value
    resolution_cfg = cfg.get("resolution", {}) if isinstance(cfg, dict) else {}
    width = int(getattr(getattr(config, "resolution", None), "width", resolution_cfg.get("width", 0)))
    height = int(getattr(getattr(config, "resolution", None), "height", resolution_cfg.get("height", 0)))
    config_hash = stable_json_hash(cfg)

    versions = dict(algorithm_versions or default_algorithm_versions())
    override_map = dict(overrides or {})
    identity = build_world_identity(
        seed,
        configuration=cfg,
        algorithm_versions=versions,
        overrides=override_map,
    )

    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "seed": seed,
        "canonical_master_seed": identity.master_seed.canonical_hex,
        "world_fingerprint": identity.fingerprint,
        "resolution": [width, height],
        "config_sha256": config_hash,
        "generator": {
            "version": ENGINE_VERSION,
            "git_commit": _git_commit(),
            "source_fingerprint_sha256": package_source_fingerprint(),
        },
        "reproducibility": {
            "seed": seed,
            "master_seed": identity.master_seed.to_manifest(),
            "world_fingerprint": identity.fingerprint,
            "world_fingerprint_schema_version": identity.schema_version,
            "resolution": [width, height],
            "config_sha256": config_hash,
            "config": cfg,
            "algorithm_versions": versions,
            "overrides": override_map,
        },
        "environment": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "atmogen": {
            **atmogen_runtime_metadata(),
            "enabled": bool(cfg.get("atmogen", {}).get("enabled", False)),
        },
        "runtime_plan": runtime,
        "stage_seconds": dict(timings or {}),
        "checkpoint_stats": dict(checkpoint_stats or {}),
    }
    if output_root is not None:
        manifest["outputs"] = output_inventory(output_root, with_hashes=with_output_hashes)
    return manifest


def write_run_manifest(path: str | Path, **kwargs) -> dict[str, Any]:
    manifest = build_run_manifest(**kwargs)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
