from __future__ import annotations

"""Public hydrology facade with optimized and physically richer execution layers.

The long-standing physical kernels remain in ``hydrology_base``.  This facade patches
performance/conservation-sensitive primitives at import time, then exposes advanced
water balance, channel hierarchy and watershed results without duplicating the stable
Priority-Flood and spherical receiver implementation.
"""

from . import hydrology_base as _base
from . import hydrology_advanced as _advanced
from .priority_flood import (
    priority_flood,
    priority_flood_reference,
    numba_priority_flood_available,
)
from .hydrology_advanced import (
    build_hydrology_advanced,
    evolve_surface_advanced,
    runoff_mm_advanced,
    transport_sediment_topological,
)
from .multicondensate_water_balance import build_multicondensate_water_balance


# Keep the legacy/ordinary conservative bucket as the default, but let a hydrologic
# climate view activate the mass-conservative multicomponent bucket.  The cached
# water-balance function in hydrology_advanced resolves ``build_water_balance`` from
# its module globals at call time, so this dispatch automatically reaches runoff,
# surface evolution and final hydrology without duplicating those algorithms.
_original_build_water_balance = _advanced.build_water_balance


def _build_water_balance_dispatch(climate, land, geology, cfg):
    if getattr(climate, "hydrologic_forcing", None) is not None:
        return build_multicondensate_water_balance(climate, land, geology, cfg)
    return _original_build_water_balance(climate, land, geology, cfg)


_advanced.build_water_balance = _build_water_balance_dispatch

# Install deterministic optimized/conservative backends into the shared equations.
# Both the interval and adaptive surface-evolution policies resolve these names from
# hydrology_base at call time.
_base._priority_flood = priority_flood
_base._runoff_mm = runoff_mm_advanced
_base._transport_sediment = transport_sediment_topological

from .hydrology_base import *  # noqa: F401,F403,E402

# Public advanced wrappers.  Existing callers continue using the same function names.
build_hydrology = build_hydrology_advanced
evolve_surface = evolve_surface_advanced

# Preserve historically importable private name for tests/tools.
_priority_flood = priority_flood

__all__ = [name for name in dir(_base) if not name.startswith("_")]
__all__ += [
    "build_hydrology",
    "evolve_surface",
    "priority_flood",
    "priority_flood_reference",
    "numba_priority_flood_available",
]
