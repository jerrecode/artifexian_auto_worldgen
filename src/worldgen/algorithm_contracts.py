from __future__ import annotations

"""Versioned canonical algorithm identities for reproducible world fingerprints.

Human-facing release versions are useful provenance, but they are insufficient as
cache/world identities: two working trees can share a package version while their
physical kernels differ.  This module therefore combines explicit contract versions
with stage-specific source fingerprints already maintained by ``fingerprints.py``.
"""

from typing import Any

from .fingerprints import stage_source_fingerprint
from .identity import SEED_DERIVATION_VERSION

ENGINE_VERSION = "0.5.0"
ALGORITHM_CONTRACT_SCHEMA_VERSION = 1

CANONICAL_PHYSICAL_STAGES = (
    "astronomy",
    "tectonics",
    "terrain",
    "ocean",
    "climate",
    "geology",
    "surface",
    "hydrology_final",
    "weather",
    "surface_appearance",
    "resources",
    "society",
)


def default_algorithm_versions() -> dict[str, Any]:
    """Return canonical algorithm/version material for world identity.

    The mapping is intentionally independent of GUI/export/logging source files.  A
    physics-kernel edit changes at least one stage hash and therefore invalidates the
    relevant world identity conservatively; an output-only edit does not.
    """

    return {
        "schema_version": ALGORITHM_CONTRACT_SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "seed_derivation": SEED_DERIVATION_VERSION,
        "stage_source_sha256": {
            stage: stage_source_fingerprint(stage) for stage in CANONICAL_PHYSICAL_STAGES
        },
    }


__all__ = [
    "ALGORITHM_CONTRACT_SCHEMA_VERSION",
    "CANONICAL_PHYSICAL_STAGES",
    "ENGINE_VERSION",
    "default_algorithm_versions",
]
