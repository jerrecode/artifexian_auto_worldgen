from __future__ import annotations

"""Persistent pin metadata for deliberately precomputed planetary tile prefixes.

A complete-prefix precompute is an archival promise, not merely a cache warm-up.
The pin manifest lets the interactive disk LRU preserve products belonging to the
precomputed z0..Z prefix while continuing to evict deeper opportunistic tiles.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Mapping, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .planet_tiles import PlanetTilePyramid


PIN_SCHEMA_VERSION = 1
PIN_FILENAME = "pinned_prefix.json"
_LEVEL_SEGMENT = re.compile(r"(?:^|/)z(\d+)(?:/|$)")


@dataclass(slots=True, frozen=True)
class PinnedPrefix:
    maximum_level: int
    source_sha256: str
    pinned_at: str
    schema_version: int = PIN_SCHEMA_VERSION


def pin_manifest_path(tile_root: str | Path) -> Path:
    return Path(tile_root).expanduser().resolve() / "precompute" / PIN_FILENAME


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def read_pinned_prefix(tile_root: str | Path) -> PinnedPrefix | None:
    path = pin_manifest_path(tile_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema = int(payload.get("schema_version", -1))
        maximum_level = int(payload["maximum_level"])
        source_sha256 = str(payload["source_sha256"])
        pinned_at = str(payload["pinned_at"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid tile prefix pin manifest {path}: {exc}") from exc
    if schema != PIN_SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported tile prefix pin schema {schema}; expected {PIN_SCHEMA_VERSION}"
        )
    if maximum_level < 0:
        raise RuntimeError("pinned prefix maximum_level cannot be negative")
    if len(source_sha256) != 64:
        raise RuntimeError("pinned prefix source_sha256 is invalid")
    return PinnedPrefix(
        maximum_level=maximum_level,
        source_sha256=source_sha256,
        pinned_at=pinned_at,
        schema_version=schema,
    )


def pin_complete_prefix(pyramid: "PlanetTilePyramid", maximum_level: int) -> PinnedPrefix:
    """Persistently protect the completed z0..Z prefix from runtime LRU eviction.

    Pinning a shallower prefix never reduces an existing deeper pin. The source
    fingerprint must match, so a pin cannot silently migrate between worlds.
    """
    level = int(maximum_level)
    if level < 0 or level > int(pyramid.spec.maximum_level):
        raise ValueError(
            f"pin depth must be in [0, {pyramid.spec.maximum_level}]"
        )
    source_sha256 = pyramid._source_hash()
    current = read_pinned_prefix(pyramid.root)
    if current is not None:
        if current.source_sha256 != source_sha256:
            raise RuntimeError(
                "existing pinned-prefix manifest belongs to a different source world"
            )
        level = max(level, current.maximum_level)
    value = PinnedPrefix(
        maximum_level=level,
        source_sha256=source_sha256,
        pinned_at=datetime.now(timezone.utc).isoformat(),
    )
    _atomic_json(
        pin_manifest_path(pyramid.root),
        {
            "schema_version": value.schema_version,
            "maximum_level": value.maximum_level,
            "source_sha256": value.source_sha256,
            "pinned_at": value.pinned_at,
            "semantics": (
                "all generated tile products with a z-level at or below maximum_level "
                "are protected from the interactive persistent LRU"
            ),
        },
    )
    return value


def clear_pinned_prefix(tile_root: str | Path) -> bool:
    path = pin_manifest_path(tile_root)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def path_tile_level(tile_root: str | Path, path: str | Path) -> int | None:
    """Extract a `zNN` level segment from a generated product path below tile_root."""
    root = Path(tile_root).expanduser().resolve()
    candidate = Path(path).expanduser().resolve()
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError:
        return None
    match = _LEVEL_SEGMENT.search(relative)
    return int(match.group(1)) if match is not None else None


def path_is_pinned(
    tile_root: str | Path,
    path: str | Path,
    pinned_prefix: PinnedPrefix | None = None,
) -> bool:
    pin = pinned_prefix if pinned_prefix is not None else read_pinned_prefix(tile_root)
    if pin is None:
        return False
    level = path_tile_level(tile_root, path)
    return level is not None and level <= pin.maximum_level


__all__ = [
    "PIN_FILENAME",
    "PIN_SCHEMA_VERSION",
    "PinnedPrefix",
    "clear_pinned_prefix",
    "path_is_pinned",
    "path_tile_level",
    "pin_complete_prefix",
    "pin_manifest_path",
    "read_pinned_prefix",
]
