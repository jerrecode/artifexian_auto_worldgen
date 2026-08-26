from __future__ import annotations

"""Source-code fingerprints for dependency-sensitive stage checkpoints."""

from pathlib import Path
import hashlib
from typing import Iterable


_COMMON_PHYSICAL = {
    "config.py",
    "rng.py",
    "grid.py",
    "topology.py",
    "topology_base.py",
    "pipeline.py",
    "pipeline_base.py",
}

_STAGE_FILES: dict[str, set[str]] = {
    "astronomy": {"astronomy.py"},
    "tectonics": {"tectonics.py", "noise.py", "mathops.py"},
    "noise_cache": {"noise.py", "mathops.py", "tiling.py"},
    "terrain": {"terrain.py", "noise.py"},
    "ocean": {"ocean.py", "noise.py"},
    "climate": {"climate.py", "noise.py"},
    "geology": {"geology.py", "noise.py"},
    "surface": {
        "hydrology.py",
        "hydrology_base.py",
        "surface_evolution.py",
        "flow_refresh.py",
        "drainage.py",
        "priority_flood.py",
        "noise.py",
        "mathops.py",
    },
    "hydrology_final": {
        "hydrology.py",
        "hydrology_base.py",
        "drainage.py",
        "priority_flood.py",
        "mathops.py",
    },
    "weather": {"weather.py"},
    "surface_appearance": {"appearance.py"},
    "resources": {"resources.py"},
    "society": {"society.py"},
    "output": {"render.py", "imageops.py", "manifest.py"},
}


def fingerprint_source_files(
    relative_paths: Iterable[str],
    *,
    package_dir: str | Path | None = None,
) -> str:
    """Hash a deterministic set of source files including names and contents."""
    root = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    h = hashlib.sha256()
    paths = sorted(set(str(p).replace("\\", "/") for p in relative_paths))
    for rel in paths:
        path = root / rel
        if not path.is_file():
            # Missing declared dependencies must change the fingerprint rather than
            # silently disappearing from it.
            marker = f"MISSING:{rel}".encode("utf-8")
            h.update(len(marker).to_bytes(4, "little"))
            h.update(marker)
            continue
        rel_b = rel.encode("utf-8")
        data = path.read_bytes()
        h.update(len(rel_b).to_bytes(4, "little"))
        h.update(rel_b)
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()


def stage_source_files(stage_name: str) -> tuple[str, ...]:
    """Return conservative direct code dependencies for a concrete pipeline stage."""
    name = str(stage_name)
    if name == "astronomy":
        kind = "astronomy"
    elif name == "tectonics":
        kind = "tectonics"
    elif name == "noise_cache":
        kind = "noise_cache"
    elif name.startswith("terrain"):
        kind = "terrain"
    elif name.startswith("ocean"):
        kind = "ocean"
    elif name.startswith("climate"):
        kind = "climate"
    elif name.startswith("geology"):
        kind = "geology"
    elif name.startswith("surface") and name != "surface_appearance":
        kind = "surface"
    elif name == "hydrology_final":
        kind = "hydrology_final"
    elif name == "weather":
        kind = "weather"
    elif name == "surface_appearance":
        kind = "surface_appearance"
    elif name == "resources":
        kind = "resources"
    elif name == "society":
        kind = "society"
    elif name == "output":
        kind = "output"
    else:
        # Unknown future stages conservatively hash every Python module.
        root = Path(__file__).resolve().parent
        return tuple(
            sorted(
                p.relative_to(root).as_posix()
                for p in root.rglob("*.py")
                if "__pycache__" not in p.parts
            )
        )
    return tuple(sorted(_COMMON_PHYSICAL | _STAGE_FILES[kind]))


def stage_source_fingerprint(
    stage_name: str,
    *,
    package_dir: str | Path | None = None,
) -> str:
    return fingerprint_source_files(
        stage_source_files(stage_name), package_dir=package_dir
    )


__all__ = [
    "fingerprint_source_files",
    "stage_source_files",
    "stage_source_fingerprint",
]
