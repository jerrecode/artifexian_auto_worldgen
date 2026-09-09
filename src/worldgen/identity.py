from __future__ import annotations

"""Stable world identity, semantic seed derivation, and cache fingerprints.

The functions in this module deliberately avoid Python's process-randomized ``hash``
and mutable global RNG streams.  They are small, dependency-light primitives meant
to sit below every scientific subsystem, scheduler, cache, CLI, and viewer.
"""

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any, Mapping

import numpy as np

SEED_DERIVATION_VERSION = "worldgen-seed-v1"
WORLD_FINGERPRINT_SCHEMA_VERSION = 1
_COMPONENT_FINGERPRINT_SCHEMA_VERSION = 1
NON_CANONICAL_CONFIG_SECTIONS = frozenset({"output", "runtime", "storage", "logging", "progress", "viewer", "ui"})
_HEX_SEED = re.compile(r"^[+]?(?:0[xX])([0-9a-fA-F]+)$")

SeedInput = int | str | bytes | bytearray | memoryview
SemanticId = str | int | float | bytes | bytearray | memoryview | bool | None


@dataclass(frozen=True, slots=True)
class CanonicalSeed:
    """Canonical 256-bit master seed plus provenance of the user input."""

    original_input: str
    input_kind: str
    canonical_hex: str
    derivation_version: str = SEED_DERIVATION_VERSION

    @property
    def bytes(self) -> bytes:
        return bytes.fromhex(self.canonical_hex)

    @property
    def uint256(self) -> int:
        return int.from_bytes(self.bytes, "big", signed=False)

    def to_manifest(self) -> dict[str, str]:
        return {
            "original_input": self.original_input,
            "input_kind": self.input_kind,
            "canonical_hex": self.canonical_hex,
            "derivation_version": self.derivation_version,
        }


def _int_payload(value: int) -> bytes:
    if isinstance(value, bool):
        raise TypeError("boolean is not a valid integer master seed")
    sign = b"-" if value < 0 else b"+"
    magnitude = abs(int(value))
    width = max(1, (magnitude.bit_length() + 7) // 8)
    body = magnitude.to_bytes(width, "big", signed=False)
    return b"int\x00" + sign + len(body).to_bytes(8, "big") + body


def canonicalize_master_seed(value: SeedInput) -> CanonicalSeed:
    """Normalize arbitrary supported seed input to a stable 256-bit identity.

    Integer values and strings explicitly written as ``0x...`` share the same
    canonical identity.  Other strings are treated as exact UTF-8 text, including
    whitespace, so a textual world name is not silently coerced to a number.
    """

    if isinstance(value, bool):
        raise TypeError("master seed must be int, str, or bytes-like, not bool")

    if isinstance(value, int):
        original = str(value)
        kind = "integer"
        payload = _int_payload(value)
    elif isinstance(value, str):
        match = _HEX_SEED.fullmatch(value)
        if match is not None:
            integer = int(match.group(1), 16)
            original = value
            kind = "hexadecimal"
            payload = _int_payload(integer)
        else:
            raw = value.encode("utf-8")
            original = value
            kind = "utf8"
            payload = b"utf8\x00" + len(raw).to_bytes(8, "big") + raw
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        original = raw.hex()
        kind = "bytes"
        payload = b"bytes\x00" + len(raw).to_bytes(8, "big") + raw
    else:
        raise TypeError("master seed must be int, str, or bytes-like")

    digest = hashlib.blake2b(
        b"worldgen-master-seed\x00" + SEED_DERIVATION_VERSION.encode("ascii") + b"\x00" + payload,
        digest_size=32,
    ).hexdigest()
    return CanonicalSeed(
        original_input=original,
        input_kind=kind,
        canonical_hex=digest,
    )


def _semantic_bytes(value: Any) -> bytes:
    """Type-tagged deterministic encoding for semantic stream identities."""

    if value is None:
        return b"n"
    if isinstance(value, bool):
        return b"t" if value else b"f"
    if isinstance(value, int):
        payload = _int_payload(value)
        return b"i" + len(payload).to_bytes(8, "big") + payload
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("semantic floating identifiers must be finite")
        normalized = 0.0 if value == 0.0 else value
        return b"d" + struct.pack(">d", normalized)
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"s" + len(raw).to_bytes(8, "big") + raw
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return b"b" + len(raw).to_bytes(8, "big") + raw
    if isinstance(value, tuple):
        parts = [_semantic_bytes(v) for v in value]
        return b"(" + len(parts).to_bytes(8, "big") + b"".join(
            len(p).to_bytes(8, "big") + p for p in parts
        )
    if isinstance(value, list):
        parts = [_semantic_bytes(v) for v in value]
        return b"[" + len(parts).to_bytes(8, "big") + b"".join(
            len(p).to_bytes(8, "big") + p for p in parts
        )
    raise TypeError(f"unsupported semantic identifier type: {type(value).__name__}")


class SeedDeriver:
    """Derive independent named/counter-based RNG streams from one master seed.

    ``generator`` uses NumPy Philox, a counter-based bit generator suitable for
    independent parallel streams.  ``counter_uint64`` is stricter still: a result is
    a pure function of semantic identity plus an explicit counter, so iteration or
    task scheduling order cannot change it.
    """

    def __init__(self, master_seed: SeedInput | CanonicalSeed):
        self.master_seed = (
            master_seed if isinstance(master_seed, CanonicalSeed)
            else canonicalize_master_seed(master_seed)
        )

    def derive_bytes(self, *semantic_ids: Any, length: int = 32) -> bytes:
        length = int(length)
        if not 1 <= length <= 64:
            raise ValueError("derived digest length must be in [1, 64]")
        h = hashlib.blake2b(
            digest_size=length,
            key=self.master_seed.bytes,
            person=b"wgen-sem-v1",
        )
        h.update(len(semantic_ids).to_bytes(8, "big"))
        for value in semantic_ids:
            encoded = _semantic_bytes(value)
            h.update(len(encoded).to_bytes(8, "big"))
            h.update(encoded)
        return h.digest()

    def stream_hex(self, *semantic_ids: Any) -> str:
        return self.derive_bytes(*semantic_ids, length=32).hex()

    def generator(self, *semantic_ids: Any) -> np.random.Generator:
        seed = int.from_bytes(self.derive_bytes(*semantic_ids, length=32), "little")
        return np.random.Generator(np.random.Philox(seed))

    def counter_uint64(self, counter: int, *semantic_ids: Any) -> int:
        if isinstance(counter, bool) or int(counter) < 0:
            raise ValueError("counter must be a non-negative integer")
        digest = self.derive_bytes("counter", *semantic_ids, int(counter), length=8)
        return int.from_bytes(digest, "big", signed=False)

    def counter_uniform01(self, counter: int, *semantic_ids: Any) -> float:
        # IEEE-754 binary64 has 53 bits of integer precision.  Use the high 53 bits
        # so all returned values are exactly representable in [0, 1).
        value = self.counter_uint64(counter, *semantic_ids) >> 11
        return value * (1.0 / (1 << 53))


def _canonical_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _canonical_jsonable(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical mappings require string keys")
            out[key] = _canonical_jsonable(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_canonical_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_jsonable(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        )
    if isinstance(value, np.generic):
        return _canonical_jsonable(value.item())
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__bytes_hex__": bytes(value).hex()}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical world identity cannot contain NaN or infinity")
        return 0.0 if value == 0.0 else value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    normalized = _canonical_jsonable(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class WorldIdentity:
    master_seed: CanonicalSeed
    fingerprint: str
    schema_version: int = WORLD_FINGERPRINT_SCHEMA_VERSION

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "master_seed": self.master_seed.to_manifest(),
            "world_fingerprint": self.fingerprint,
        }


def build_world_identity(
    master_seed: SeedInput | CanonicalSeed,
    *,
    configuration: Mapping[str, Any] | None = None,
    algorithm_versions: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> WorldIdentity:
    seed = master_seed if isinstance(master_seed, CanonicalSeed) else canonicalize_master_seed(master_seed)
    cfg = dict(configuration or {})
    # The seed has its own canonical representation.  Operational/output settings
    # are deliberately excluded as well: changing log verbosity, cache quota, UI
    # state, or export format must not create a new physical world.
    cfg.pop("seed", None)
    for section in NON_CANONICAL_CONFIG_SECTIONS:
        cfg.pop(section, None)
    payload = {
        "schema_version": WORLD_FINGERPRINT_SCHEMA_VERSION,
        "master_seed_canonical_hex": seed.canonical_hex,
        "seed_derivation_version": seed.derivation_version,
        "configuration": cfg,
        "algorithm_versions": dict(algorithm_versions or {}),
        "overrides": dict(overrides or {}),
    }
    digest = hashlib.sha256(
        b"worldgen-world-fingerprint-v1\x00" + canonical_json_bytes(payload)
    ).hexdigest()
    return WorldIdentity(master_seed=seed, fingerprint=digest)


def component_fingerprint(
    world_fingerprint: str,
    subsystem: str,
    *,
    algorithm_version: str | int,
    address: Any = None,
    lod: int | None = None,
    dependencies: Mapping[str, str] | None = None,
) -> str:
    """Content-address a deterministic subsystem product or tile."""

    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(world_fingerprint)):
        raise ValueError("world_fingerprint must be a 64-hex-character SHA-256 value")
    if not subsystem:
        raise ValueError("subsystem must be non-empty")
    if lod is not None and int(lod) < 0:
        raise ValueError("lod must be non-negative")
    payload = {
        "schema_version": _COMPONENT_FINGERPRINT_SCHEMA_VERSION,
        "world_fingerprint": str(world_fingerprint).lower(),
        "subsystem": str(subsystem),
        "algorithm_version": str(algorithm_version),
        "address": address,
        "lod": None if lod is None else int(lod),
        "dependencies": dict(dependencies or {}),
    }
    return hashlib.sha256(
        b"worldgen-component-fingerprint-v1\x00" + canonical_json_bytes(payload)
    ).hexdigest()


__all__ = [
    "CanonicalSeed",
    "NON_CANONICAL_CONFIG_SECTIONS",
    "SEED_DERIVATION_VERSION",
    "SeedDeriver",
    "SeedInput",
    "WORLD_FINGERPRINT_SCHEMA_VERSION",
    "WorldIdentity",
    "build_world_identity",
    "canonical_json_bytes",
    "canonicalize_master_seed",
    "component_fingerprint",
]
